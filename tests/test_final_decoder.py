"""Synthetic final-fit, isolation, and serialization tests for CA-06C."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from neural_decoding.final_decoder import (
    FROZEN_SPEC,
    ValidationIdentity,
    ValidationScoringLedger,
    decoder_from_dict,
    decoder_to_dict,
    fit_frozen_decoder,
)
from neural_decoding.joint_decoder import (
    JointDecoderError,
    TrialBounds,
    build_assessment_trial,
    filter_trial,
)


def _synthetic_training_session():
    n_trials = 123
    steps_per_trial = 12
    n_samples = n_trials * steps_per_trial
    timestamps = np.arange(n_samples, dtype=np.float64) / 100.0
    sample = np.arange(n_samples, dtype=np.float64)
    trial = sample // steps_per_trial
    position = np.column_stack(
        [
            np.sin(sample * 0.017) + trial * 0.002,
            np.cos(sample * 0.013) - trial * 0.003,
        ]
    )
    velocity = np.column_stack(
        [
            np.cos(sample * 0.021) * 4.0 + trial * 0.004,
            np.sin(sample * 0.019) * 3.0 - trial * 0.005,
        ]
    )
    acceleration = np.column_stack(
        [
            np.sin(sample * 0.031) * 2.0 + trial * 0.006,
            np.cos(sample * 0.029) * 2.5 - trial * 0.007,
        ]
    )
    bounds = {
        row: TrialBounds(
            trial_row=row,
            first_index=row * steps_per_trial,
            last_index=(row + 1) * steps_per_trial - 1,
            first_time=float(timestamps[row * steps_per_trial]),
            last_time=float(timestamps[(row + 1) * steps_per_trial - 1]),
        )
        for row in range(n_trials)
    }
    spike_times = []
    for unit in range(53):
        spikes: list[float] = []
        for row, trial_bounds in bounds.items():
            count = 1 + ((row * 3 + unit * 5) % 4)
            spikes.extend(
                trial_bounds.first_time + 0.001 * np.arange(1, count + 1)
            )
        spike_times.append(np.asarray(spikes, dtype=np.float64))
    unit_ids = tuple(10_000 + index for index in range(53))
    session = SimpleNamespace(
        cursor_position=position,
        cursor_velocity=velocity,
        cursor_acceleration=acceleration,
        behavior_timestamps=timestamps,
        spike_times=tuple(spike_times),
        units=tuple(SimpleNamespace(unit_id=value) for value in unit_ids),
    )
    return session, bounds, unit_ids


def test_final_fit_uses_every_training_trial_and_serializes_exactly() -> None:
    session, bounds, unit_ids = _synthetic_training_session()

    model, schedule = fit_frozen_decoder(session, bounds)
    payload = decoder_to_dict(
        model,
        unit_ids=unit_ids,
        provenance={"recording_sha256": "synthetic"},
    )
    reloaded, reloaded_unit_ids = decoder_from_dict(payload)

    assert model.spec == FROZEN_SPEC
    assert model.process.training_rows == tuple(range(123))
    assert model.observation.training_rows == tuple(range(123))
    assert reloaded_unit_ids == unit_ids

    first = bounds[0]
    assessment = build_assessment_trial(
        schedule,
        first,
        initial_position=session.cursor_position[first.first_index],
    )
    original_trace = filter_trial(model, assessment)
    reloaded_trace = filter_trial(reloaded, assessment)
    assert np.allclose(
        original_trace.velocity, reloaded_trace.velocity, atol=1e-12, rtol=1e-12
    )
    assert np.allclose(
        original_trace.velocity_covariance,
        reloaded_trace.velocity_covariance,
        atol=1e-12,
        rtol=1e-12,
    )


def test_validation_identity_and_truth_are_separate_types() -> None:
    identity_fields = {field.name for field in fields(ValidationIdentity)}
    scoring_fields = {field.name for field in fields(ValidationScoringLedger)}

    assert "velocity" not in identity_fields
    assert scoring_fields == {"velocity", "truth_sha256"}


def test_final_fit_does_not_materialize_behavior_outside_training_bounds() -> None:
    session, bounds, _ = _synthetic_training_session()
    session.behavior_timestamps = np.append(
        session.behavior_timestamps,
        session.behavior_timestamps[-1] + 0.01,
    )
    for name in ("cursor_position", "cursor_velocity", "cursor_acceleration"):
        values = getattr(session, name)
        setattr(session, name, np.vstack([values, [np.nan, np.nan]]))

    model, _ = fit_frozen_decoder(session, bounds)

    assert model.process.training_rows == tuple(range(123))


def test_reload_rejects_provenance_that_differs_from_frozen_inputs() -> None:
    session, bounds, unit_ids = _synthetic_training_session()
    model, _ = fit_frozen_decoder(session, bounds)
    expected = {"recording_sha256": "expected"}
    payload = decoder_to_dict(
        model,
        unit_ids=unit_ids,
        provenance={"recording_sha256": "changed"},
    )

    with pytest.raises(JointDecoderError, match="provenance"):
        decoder_from_dict(payload, expected_provenance=expected)
