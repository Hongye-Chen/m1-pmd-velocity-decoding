"""Focused synthetic tests for the clean-room joint decoder protocol."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from neural_decoding.joint_decoder import (
    AssessmentTrial,
    CandidateSpec,
    DecoderModel,
    JointDecoderError,
    ObservationModel,
    ProcessModel,
    StateNormalization,
    TrialBounds,
    build_assessment_trial,
    build_scheduled_observations,
    filter_trial,
    fit_observation_model,
    fit_process_model,
    select_candidate,
    target_identity_sha256,
    validate_oof_fill,
)


def _bounds(row: int, first: int, last: int, timestamps: np.ndarray) -> TrialBounds:
    return TrialBounds(
        trial_row=row,
        first_index=first,
        last_index=last,
        first_time=float(timestamps[first]),
        last_time=float(timestamps[last]),
    )


def _model(state_name: str = "V", *, singular_q: bool = False) -> DecoderModel:
    components = {
        "V": ("vx", "vy"),
        "PV": ("px", "py", "vx", "vy"),
    }[state_name]
    dimension = len(components)
    normalization = StateNormalization(
        mean=np.zeros(dimension),
        scale=np.ones(dimension),
        training_rows=(0,),
    )
    transition = np.eye(dimension) * 0.95
    if state_name == "PV":
        transition[0, 2] = 0.01
        transition[1, 3] = 0.01
    process_covariance = np.zeros((dimension, dimension))
    if not singular_q:
        process_covariance += np.eye(dimension) * 0.01
    else:
        process_covariance[-1, -1] = 0.01
    process = ProcessModel(
        state_name=state_name,
        components=components,
        normalization=normalization,
        transition=transition,
        offset=np.zeros(dimension),
        process_covariance=process_covariance,
        initial_mean=np.zeros(dimension),
        initial_covariance=np.eye(dimension),
        training_rows=(0,),
        diagnostics={},
    )
    observation_matrix = np.zeros((3, dimension))
    observation_matrix[0, -2] = 1.0
    observation_matrix[1, -1] = 1.0
    observation_matrix[2, -2:] = [0.5, -0.25]
    residual_covariance = np.array(
        [[0.8, 0.1, 0.0], [0.1, 0.7, 0.05], [0.0, 0.05, 0.9]]
    )
    cholesky = np.linalg.cholesky(residual_covariance)
    inverse_observation = np.linalg.solve(
        residual_covariance, observation_matrix
    )
    observation = ObservationModel(
        count_mean=np.zeros(3),
        count_scale=np.ones(3),
        observation_matrix=observation_matrix,
        offset=np.zeros(3),
        residual_covariance=residual_covariance,
        residual_cholesky=cholesky,
        residual_inverse_observation=inverse_observation,
        prewhitened_observation=np.linalg.solve(cholesky, observation_matrix),
        shrinkage=0.5,
        training_rows=(0,),
        diagnostics={},
    )
    return DecoderModel(
        spec=CandidateSpec(state_name, 20, 0),
        process=process,
        observation=observation,
    )


def test_schedule_alignment_is_half_open_and_never_reuses_spikes() -> None:
    timestamps = np.arange(31, dtype=np.float64) / 100.0
    bounds = {
        0: _bounds(0, 0, 10, timestamps),
        1: _bounds(1, 20, 30, timestamps),
    }
    spikes = [
        np.array(
            [
                0.0,
                0.01,
                0.019999999,
                0.02,
                0.099999999,
                0.10,
                0.15,
                0.20,
                0.21,
                0.299999999,
                0.30,
            ]
        )
    ]
    schedule = build_scheduled_observations(
        timestamps, spikes, bounds, width_ms=10, lag_ms=0
    )

    first_trial = schedule.counts[schedule.trial_row == 0, 0]
    assert first_trial[:3].tolist() == [1, 2, 1]
    assert int(first_trial.sum()) == 5
    assert np.all(
        schedule.interval_start_time[schedule.trial_row == 1] >= timestamps[20]
    )
    assert np.all(
        schedule.interval_end_time[schedule.trial_row == 0] <= timestamps[10]
    )
    keys = list(zip(schedule.trial_row, schedule.interval_ordinal))
    assert len(keys) == len(set(keys))


def test_fifty_ms_width_and_lag_have_exact_interval_phase() -> None:
    timestamps = np.arange(16, dtype=np.float64) / 100.0
    trial_bounds = _bounds(0, 0, 15, timestamps)
    # The spike on 50 ms belongs only to the second fresh interval.
    schedule = build_scheduled_observations(
        timestamps,
        [np.array([0.0, 0.049999999, 0.05, 0.099999999, 0.10])],
        {0: trial_bounds},
        width_ms=50,
        lag_ms=50,
    )

    assert schedule.interval_start_time.tolist() == [0.0, 0.05]
    assert schedule.interval_end_time.tolist() == [0.05, 0.10]
    assert schedule.state_index.tolist() == [10, 15]
    assert schedule.counts[:, 0].tolist() == [2, 2]


@pytest.mark.parametrize(
    ("width_ms", "expected_offsets"),
    [(20, [2, 4, 6, 8, 10]), (50, [5, 10])],
)
def test_multirate_schedule_has_prediction_only_steps(
    width_ms: int, expected_offsets: list[int]
) -> None:
    timestamps = np.arange(11, dtype=np.float64) / 100.0
    trial_bounds = _bounds(0, 0, 10, timestamps)
    schedule = build_scheduled_observations(
        timestamps,
        [np.array([0.005, 0.025, 0.055]) for _ in range(3)],
        {0: trial_bounds},
        width_ms=width_ms,
        lag_ms=0,
    )
    assessment = build_assessment_trial(
        schedule, trial_bounds, initial_position=None
    )
    model = _model()
    model = DecoderModel(
        spec=CandidateSpec("V", width_ms, 0),
        process=model.process,
        observation=model.observation,
    )
    trace = filter_trial(model, assessment)

    assert assessment.update_offsets.tolist() == expected_offsets
    assert trace.velocity.shape == (11, 2)
    assert set(range(1, 11)) - set(expected_offsets)


def test_state_and_count_normalization_are_fold_specific() -> None:
    timestamps = np.arange(40, dtype=np.float64) / 100.0
    bounds = {
        0: _bounds(0, 0, 9, timestamps),
        1: _bounds(1, 10, 19, timestamps),
        2: _bounds(2, 20, 29, timestamps),
        3: _bounds(3, 30, 39, timestamps),
    }
    state = np.column_stack(
        [
            np.r_[
                np.arange(10),
                np.arange(10) + 10.0,
                np.arange(10) + 100.0,
                np.arange(10) + 110.0,
            ],
            np.r_[
                np.arange(10) * 2.0,
                np.arange(10) * 2.0 + 20.0,
                np.arange(10) * 2.0 + 200.0,
                np.arange(10) * 2.0 + 220.0,
            ],
        ]
    )
    first = fit_process_model(state, bounds, (0, 1), "V")
    second = fit_process_model(state, bounds, (2, 3), "V")
    assert not np.array_equal(
        first.normalization.mean, second.normalization.mean
    )
    assert first.normalization.training_rows == (0, 1)
    assert second.normalization.training_rows == (2, 3)

    schedule = build_scheduled_observations(
        timestamps,
        [
            np.array([0.001, 0.012, 0.034, 0.101, 0.111, 0.122, 0.145, 0.167]),
            np.array(
                [
                    0.004,
                    0.024,
                    0.051,
                    0.104,
                    0.125,
                    0.136,
                    0.158,
                    0.179,
                    0.201,
                    0.211,
                    0.222,
                    0.245,
                    0.267,
                    0.304,
                    0.325,
                    0.336,
                    0.358,
                    0.379,
                ]
            ),
        ],
        bounds,
        width_ms=10,
        lag_ms=0,
    )
    observation_first = fit_observation_model(
        schedule, first.normalization.apply(state), (0, 1)
    )
    observation_second = fit_observation_model(
        schedule, second.normalization.apply(state), (2, 3)
    )
    assert not np.array_equal(
        observation_first.count_mean, observation_second.count_mean
    )
    assert observation_first.training_rows == (0, 1)
    assert observation_second.training_rows == (2, 3)


def test_filter_interface_cannot_receive_assessment_truth() -> None:
    names = {field.name for field in fields(AssessmentTrial)}
    assert names == {
        "trial_row",
        "n_steps",
        "update_offsets",
        "counts",
        "initial_position",
    }
    assert names.isdisjoint({"velocity", "acceleration", "position", "targets"})


def test_position_initialization_is_once_and_each_trial_resets() -> None:
    model = _model("PV")
    trial = AssessmentTrial(
        trial_row=4,
        n_steps=8,
        update_offsets=np.array([2, 4, 6]),
        counts=np.array([[1, 0, 1], [0, 1, 0], [2, 1, 0]], dtype=np.uint16),
        initial_position=np.array([3.0, -2.0]),
    )
    first = filter_trial(model, trial)
    _ = filter_trial(
        model,
        AssessmentTrial(
            trial_row=5,
            n_steps=8,
            update_offsets=trial.update_offsets,
            counts=np.full_like(trial.counts, 9),
            initial_position=np.array([-7.0, 8.0]),
        ),
    )
    repeated = filter_trial(model, trial)

    assert first.normalized_state[0, :2].tolist() == [3.0, -2.0]
    assert np.array_equal(first.velocity, repeated.velocity)
    assert np.array_equal(first.posterior_covariance, repeated.posterior_covariance)
    assert np.all(first.posterior_covariance[0, :2] == 0.0)
    assert np.all(first.posterior_covariance[0, :, :2] == 0.0)


def test_rank_deficient_process_covariance_and_recursion_remain_stable() -> None:
    model = _model(singular_q=True)
    assert np.linalg.matrix_rank(model.process.process_covariance) == 1
    trace = filter_trial(
        model,
        AssessmentTrial(
            trial_row=0,
            n_steps=30,
            update_offsets=np.arange(2, 30, 2),
            counts=np.tile(np.array([[1, 0, 1]], dtype=np.uint16), (14, 1)),
            initial_position=None,
        ),
    )
    assert np.all(np.isfinite(trace.velocity))
    eigenvalues = np.linalg.eigvalsh(trace.posterior_covariance)
    assert float(eigenvalues.min()) >= -1e-12


def test_optimized_sequential_update_matches_direct_stable_recursion() -> None:
    model = _model()
    trial = AssessmentTrial(
        trial_row=0,
        n_steps=12,
        update_offsets=np.array([2, 4, 6, 8, 10]),
        counts=np.array(
            [[1, 0, 1], [0, 1, 0], [2, 1, 0], [1, 2, 1], [0, 0, 1]],
            dtype=np.uint16,
        ),
        initial_position=None,
    )
    optimized = filter_trial(model, trial)

    mean = model.process.initial_mean.copy()
    covariance = model.process.initial_covariance.copy()
    means = np.empty_like(optimized.normalized_state)
    covariances = np.empty_like(optimized.posterior_covariance)
    updates = dict(zip(trial.update_offsets.tolist(), trial.counts))
    identity = np.eye(mean.size)
    for step in range(trial.n_steps):
        if step:
            mean = model.process.transition @ mean + model.process.offset
            covariance = (
                model.process.transition
                @ covariance
                @ model.process.transition.T
                + model.process.process_covariance
            )
        if step in updates:
            observation = model.observation.normalize_counts(updates[step])
            innovation = observation - (
                model.observation.observation_matrix @ mean
                + model.observation.offset
            )
            matrix = (
                model.observation.observation_matrix
                @ covariance
                @ model.observation.observation_matrix.T
                + model.observation.residual_covariance
            )
            gain = np.linalg.solve(
                matrix,
                model.observation.observation_matrix @ covariance,
            ).T
            mean = mean + gain @ innovation
            left = identity - gain @ model.observation.observation_matrix
            covariance = (
                left @ covariance @ left.T
                + gain @ model.observation.residual_covariance @ gain.T
            )
            covariance = (covariance + covariance.T) * 0.5
        means[step] = mean
        covariances[step] = covariance

    assert np.allclose(optimized.normalized_state, means, atol=1e-11, rtol=1e-11)
    assert np.allclose(
        optimized.posterior_covariance, covariances, atol=1e-11, rtol=1e-11
    )


def test_common_oof_rows_are_filled_once_and_hash_deterministically() -> None:
    predictions = np.arange(20, dtype=np.float64).reshape(10, 2)
    validate_oof_fill(predictions, np.ones(10, dtype=np.int64))
    with pytest.raises(JointDecoderError, match="filled once"):
        validate_oof_fill(predictions, np.array([1] * 9 + [2]))
    with pytest.raises(JointDecoderError, match="incomplete"):
        broken = predictions.copy()
        broken[3, 0] = np.nan
        validate_oof_fill(broken, np.ones(10, dtype=np.int64))

    rows = np.repeat([0, 1], 5)
    indices = np.arange(10)
    times = indices / 100.0
    assert target_identity_sha256(rows, indices, times) == target_identity_sha256(
        rows.copy(), indices.copy(), times.copy()
    )
    assert target_identity_sha256(rows, indices, times) != target_identity_sha256(
        rows, indices + 1, times
    )


def test_selection_ties_resolve_by_dimension_width_lag_then_name() -> None:
    records = [
        {
            "state_name": "VA",
            "state_dimension": 4,
            "neural_width_ms": 10,
            "interval_end_lag_ms": 0,
            "out_of_fold_r2_mean": 0.5,
            "passed": True,
        },
        {
            "state_name": "V",
            "state_dimension": 2,
            "neural_width_ms": 20,
            "interval_end_lag_ms": 0,
            "out_of_fold_r2_mean": 0.5 - 5e-13,
            "passed": True,
        },
        {
            "state_name": "V",
            "state_dimension": 2,
            "neural_width_ms": 10,
            "interval_end_lag_ms": 50,
            "out_of_fold_r2_mean": 0.5 - 5e-13,
            "passed": True,
        },
        {
            "state_name": "V",
            "state_dimension": 2,
            "neural_width_ms": 10,
            "interval_end_lag_ms": 0,
            "out_of_fold_r2_mean": 0.5 - 5e-13,
            "passed": True,
        },
    ]
    selected = select_candidate(records)
    assert selected["state_name"] == "V"
    assert selected["neural_width_ms"] == 10
    assert selected["interval_end_lag_ms"] == 0
