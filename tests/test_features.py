"""Synthetic tests for the causal feature protocol and lag selection support.

Every defect guarded here would raise the reported score rather than fail: a
reversed lag, a window crossing a trial boundary, candidate lags scored on different
target sets, or statistics fitted on an assessment fold. Bin-edge cases use the
edges the specification computes rather than decimal literals, since ``1.15`` is not
the same double as ``1.21 - 0.05``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_decoding.features import (
    CANDIDATE_LAGS_MS,
    DEFAULT_SPEC,
    FeatureError,
    FeatureSpec,
    build_features,
    build_lag_features,
    fit_normalization,
    subset_by_trials,
)
from neural_decoding.splits import build_trial_table

SAMPLE_RATE_HZ = 100.0


def _timeline(n_samples: int) -> np.ndarray:
    """A uniform 100 Hz timebase on exact hundredths of a second."""
    return np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE_HZ


def _velocity(timestamps: np.ndarray) -> np.ndarray:
    """Velocity that encodes its own timestamp, so lag errors become visible."""
    return np.column_stack([timestamps, -2.0 * timestamps])


def _trials(specs: list[tuple[int, float, float]], *, result: str = "R") -> pd.DataFrame:
    """Build an eligible trial table through the production split logic."""
    records = []
    for row, start, stop in specs:
        span = stop - start
        records.append(
            {
                "trial_row": row,
                "start_time": start,
                "stop_time": stop,
                "result": result,
                "num_targets": 4,
                "num_attempted": 4.0,
                "go_cue_time_array": np.asarray(
                    [start + span * fraction for fraction in (0.1, 0.3, 0.5, 0.7)],
                    dtype=np.float64,
                ),
            }
        )
    return build_trial_table(pd.DataFrame.from_records(records))


def _build(
    trials: pd.DataFrame,
    spike_times: list[np.ndarray],
    *,
    split: str = "train",
    n_samples: int = 2000,
    lag_ms: int = 100,
    **kwargs,
):
    timestamps = _timeline(n_samples)
    return build_features(
        timestamps=timestamps,
        velocity=_velocity(timestamps),
        spike_times=spike_times,
        trials=trials,
        split=split,
        lag_ms=lag_ms,
        **kwargs,
    )


def _build_lags(
    trials: pd.DataFrame,
    spike_times: list[np.ndarray],
    *,
    split: str = "train",
    n_samples: int = 2000,
    lags=CANDIDATE_LAGS_MS,
    **kwargs,
):
    timestamps = _timeline(n_samples)
    return build_lag_features(
        timestamps=timestamps,
        velocity=_velocity(timestamps),
        spike_times=spike_times,
        trials=trials,
        split=split,
        lags=lags,
        **kwargs,
    )


def test_bins_are_half_open_and_ordered_oldest_first() -> None:
    # These bounds yield exactly one sample, whose reference time is 1.21 s, and sit
    # off the sample grid so the trial edges cannot be confused with a sample.
    trials = _trials([(0, 1.005, 1.315)])
    reference_time = 1.21
    edges = reference_time + DEFAULT_SPEC.bin_offsets_s()
    spikes = np.sort(
        np.array(
            [
                edges[0],  # first bin, left edge is inclusive
                edges[1],  # second bin
                edges[2] - 1e-9,  # still the second bin, right edge is exclusive
                edges[2],  # third bin
                edges[3],  # fourth bin
                edges[4],  # equals the reference time, so no bin
                edges[0] - 0.001,  # older than the history window
            ]
        )
    )

    features = _build(trials, [spikes])

    assert features.n_samples == 1
    assert features.reference_time[0] == pytest.approx(reference_time, abs=1e-12)
    assert features.counts[0, :, 0].tolist() == [1, 2, 1, 1]


def test_history_covers_exactly_the_specified_window() -> None:
    trials = _trials([(0, 1.005, 3.005)])
    inside = np.array([1.4, 1.6, 1.8])

    features = _build(trials, [inside])
    spec = features.spec

    assert spec.n_bins == 4
    assert spec.bin_ms == 50
    assert spec.history_ms == spec.n_bins * spec.bin_ms
    offsets = spec.bin_offsets_s()
    assert offsets[0] == pytest.approx(-0.2, abs=1e-15)
    assert offsets[-1] == 0.0
    for sample in range(features.n_samples):
        window = features.reference_time[sample] + np.array([-spec.history_s, 0.0])
        expected = int(np.count_nonzero((inside >= window[0]) & (inside < window[1])))
        assert int(features.counts[sample, :, 0].sum()) == expected


@pytest.mark.parametrize("lag_ms", CANDIDATE_LAGS_MS)
def test_target_is_the_velocity_one_lag_after_the_reference(lag_ms: int) -> None:
    trials = _trials([(0, 1.005, 3.005)])

    features = _build(trials, [np.array([1.5])], lag_ms=lag_ms)

    lag = features.target_time - features.reference_time
    assert np.allclose(lag, lag_ms / 1000.0, atol=1e-12)
    # The target is never before the neural data that predicts it.
    assert np.all(features.target_time >= features.reference_time)
    # Velocity encodes its own timestamp, so the target must be the velocity at the
    # later time.
    assert np.allclose(features.targets[:, 0], features.target_time, atol=1e-12)
    assert np.allclose(features.targets[:, 1], -2.0 * features.target_time, atol=1e-12)
    assert features.lag_ms == lag_ms


def test_a_negative_lag_is_refused() -> None:
    trials = _trials([(0, 1.005, 3.005)])

    with pytest.raises(FeatureError, match="predict the past from the future"):
        _build(trials, [np.array([1.5])], lag_ms=-100)


def test_every_candidate_lag_predicts_identical_targets() -> None:
    trials = _trials([(0, 1.005, 3.005), (1, 4.005, 6.005)])
    spikes = [np.arange(1.0, 6.5, 0.017), np.arange(1.1, 6.5, 0.023)]

    built = _build_lags(trials, spikes)

    assert tuple(built) == CANDIDATE_LAGS_MS
    reference = built[CANDIDATE_LAGS_MS[0]]
    for lag, features in built.items():
        assert np.array_equal(features.target_index, reference.target_index)
        assert np.array_equal(features.target_time, reference.target_time)
        assert np.array_equal(features.targets, reference.targets)
        assert np.array_equal(features.trial_row, reference.trial_row)
        assert features.eligibility_lag_ms == max(CANDIDATE_LAGS_MS)
        # Only the neural window moves with the lag.
        expected_shift = int(round(lag / 10))
        assert np.array_equal(
            features.reference_index, features.target_index - expected_shift
        )

    # Different windows must actually produce different features, otherwise the
    # comparison between lags would be vacuous.
    assert not np.array_equal(built[0].counts, built[200].counts)


def test_common_target_set_uses_the_largest_candidate_lag() -> None:
    trials = _trials([(0, 1.005, 3.005)])
    spikes = [np.array([1.5])]

    alone = _build_lags(trials, spikes, lags=(0,))
    shared = _build_lags(trials, spikes, lags=(0, 200))

    # The shared set gives up exactly the samples whose 200 ms history would not fit.
    assert alone[0].n_samples - shared[0].n_samples == 20
    assert shared[0].target_index[0] - alone[0].target_index[0] == 20


def test_no_window_crosses_a_trial_boundary_at_any_lag() -> None:
    trials = _trials([(0, 1.005, 3.005), (1, 4.005, 6.005)])
    # Spikes exist only outside the trials, so a window that leaked past a trial
    # boundary would pick them up.
    outside = np.array([0.5, 3.2, 3.5, 3.9, 6.5, 7.0])

    built = _build_lags(trials, [outside])

    starts = trials.set_index("trial_row")["start_time"]
    stops = trials.set_index("trial_row")["stop_time"]
    for features in built.values():
        assert features.n_samples > 0
        assert int(features.counts.sum()) == 0
        for row, reference, target in zip(
            features.trial_row, features.reference_time, features.target_time
        ):
            assert reference - features.spec.history_s >= starts[row] - 1e-12
            assert target <= stops[row] + 1e-12


def test_each_split_uses_only_its_own_eligible_trials() -> None:
    trials = pd.concat(
        [
            _trials([(0, 1.005, 3.005), (151, 5.005, 7.005), (183, 9.005, 11.005)]),
            _trials([(1, 12.005, 14.005)], result="A"),
        ],
        ignore_index=True,
    )
    spikes = [np.array([1.5, 5.5, 9.5, 12.5])]

    train = _build(trials, spikes, split="train")
    validation = _build(trials, spikes, split="validation")

    assert set(train.trial_row.tolist()) == {0}
    assert set(validation.trial_row.tolist()) == {151}
    assert set(train.trial_row.tolist()).isdisjoint(validation.trial_row.tolist())
    assert 183 not in train.trial_row.tolist()
    assert 183 not in validation.trial_row.tolist()


def test_held_out_features_are_refused_without_an_explicit_override() -> None:
    trials = _trials([(183, 9.005, 11.005)])
    spikes = [np.array([9.5])]

    with pytest.raises(FeatureError, match="held-out"):
        _build(trials, spikes, split="test")
    with pytest.raises(FeatureError, match="held-out"):
        _build_lags(trials, spikes, split="test")

    allowed = _build(trials, spikes, split="test", allow_held_out=True)
    assert allowed.n_samples > 0


def test_a_trial_subset_keeps_whole_trials_in_order() -> None:
    trials = _trials([(0, 1.005, 3.005), (1, 4.005, 6.005), (2, 7.005, 9.005)])
    features = _build(trials, [np.array([1.5, 4.5, 7.5])])

    subset = subset_by_trials(features, [0, 2], label="fold 1 training")

    assert subset.trial_rows.tolist() == [0, 2]
    assert subset.subset_label == "fold 1 training"
    assert subset.source == "train[fold 1 training]"
    # Every sample of each named trial is present, and none of the others.
    for row in (0, 2):
        expected = int(np.count_nonzero(features.trial_row == row))
        assert int(np.count_nonzero(subset.trial_row == row)) == expected
    assert 1 not in subset.trial_row.tolist()
    assert np.all(np.diff(subset.target_index) > 0)


def test_a_trial_subset_refuses_trials_it_does_not_contain() -> None:
    trials = _trials([(0, 1.005, 3.005)])
    features = _build(trials, [np.array([1.5])])

    with pytest.raises(FeatureError, match="not present"):
        subset_by_trials(features, [0, 7])
    with pytest.raises(FeatureError, match="at least one trial"):
        subset_by_trials(features, [])


def test_fold_normalization_never_sees_its_assessment_fold() -> None:
    trials = _trials([(0, 1.005, 3.005), (1, 4.005, 6.005), (2, 7.005, 9.005)])
    spikes = [np.arange(1.0, 9.5, 0.05), np.arange(4.0, 9.5, 0.011)]
    features = _build(trials, spikes)

    fold_training = subset_by_trials(features, [0, 1], label="fold 3 training")
    assessment = subset_by_trials(features, [2], label="fold 3 assessment")

    normalization = fit_normalization(fold_training)

    assert normalization.fitted_on == "train[fold 3 training]"
    assert normalization.trial_rows == (0, 1)
    assert normalization.n_samples == fold_training.n_samples
    # The statistics come from the fold-training samples alone.
    assert np.allclose(
        normalization.feature_mean, fold_training.counts.astype(float).mean(axis=0)
    )
    assert not np.allclose(
        normalization.feature_mean, features.counts.astype(float).mean(axis=0)
    )

    standardized_fold = normalization.apply_features(fold_training.counts)
    varying = normalization.feature_std > normalization.std_floor
    assert np.allclose(standardized_fold.mean(axis=0)[varying], 0.0, atol=1e-10)

    # The assessment fold is transformed with those same statistics, so its own mean
    # is not removed.
    standardized_assessment = normalization.apply_features(assessment.counts)
    assert not np.allclose(standardized_assessment.mean(axis=0)[varying], 0.0, atol=1e-6)


def test_normalization_is_refused_outside_training_data() -> None:
    trials = _trials([(0, 1.005, 3.005), (151, 5.005, 7.005)])
    spikes = [np.array([1.4, 1.9, 2.4, 5.4, 5.9, 6.4])]
    validation = _build(trials, spikes, split="validation")

    with pytest.raises(FeatureError, match="must be fitted on"):
        fit_normalization(validation)


def test_full_training_normalization_round_trips_targets() -> None:
    trials = _trials([(0, 1.005, 3.005)])
    features = _build(trials, [np.arange(1.0, 3.5, 0.03)])

    normalization = fit_normalization(features)

    assert normalization.fitted_on == "train"
    assert normalization.lag_ms == features.lag_ms
    assert np.allclose(
        normalization.invert_targets(normalization.apply_targets(features.targets)),
        features.targets,
    )


def test_repeated_builds_produce_identical_samples_in_identical_order() -> None:
    trials = _trials([(0, 1.005, 3.005), (1, 4.005, 6.005)])
    spikes = [np.array([1.4, 1.9, 4.4]), np.array([2.2, 5.5])]

    first = _build(trials, spikes)
    second = _build(trials, spikes)

    assert np.array_equal(first.counts, second.counts)
    assert np.array_equal(first.targets, second.targets)
    assert np.array_equal(first.reference_index, second.reference_index)
    assert np.array_equal(first.trial_row, second.trial_row)
    assert first.sample_manifest().equals(second.sample_manifest())
    assert np.all(np.diff(first.target_index) > 0)


def test_unsorted_spike_times_are_refused() -> None:
    trials = _trials([(0, 1.005, 3.005)])

    with pytest.raises(FeatureError, match="not sorted"):
        _build(trials, [np.array([1.9, 1.4, 2.4])])


def test_a_non_uniform_timebase_is_refused() -> None:
    trials = _trials([(0, 1.005, 3.005)])
    timestamps = _timeline(2000)
    timestamps[500:] += 0.5

    with pytest.raises(FeatureError, match="not uniformly sampled"):
        build_features(
            timestamps=timestamps,
            velocity=_velocity(timestamps),
            spike_times=[np.array([1.5])],
            trials=trials,
            split="train",
            lag_ms=100,
        )


def test_history_must_divide_into_whole_bins() -> None:
    with pytest.raises(FeatureError, match="not divisible"):
        FeatureSpec(history_ms=200, n_bins=3)


def test_a_protocol_off_the_sample_grid_is_refused() -> None:
    trials = _trials([(0, 1.005, 3.005)])

    with pytest.raises(FeatureError, match="not a whole number"):
        _build(trials, [np.array([1.5])], spec=FeatureSpec(cadence_ms=7))
    with pytest.raises(FeatureError, match="not a whole number"):
        _build(trials, [np.array([1.5])], lag_ms=7)


def test_a_grid_aligned_trial_start_keeps_its_first_reference_sample() -> None:
    # 124.37 + 0.2 rounds above the stored timestamp for 124.57, so positioning the
    # window by comparing times would silently drop this trial's first sample. The
    # window is counted in samples instead, which this guards against regressing.
    start = 124.37
    n_samples = 13_000
    timestamps = _timeline(n_samples)
    first_reference = int(round(start * SAMPLE_RATE_HZ)) + DEFAULT_SPEC.history_ms // 10
    assert start + DEFAULT_SPEC.history_s > timestamps[first_reference]

    trials = _trials([(0, start, start + 2.0)])
    features = _build(
        trials, [np.array([start + 0.1])], n_samples=n_samples, lag_ms=0
    )

    assert features.reference_index[0] == first_reference
    assert features.reference_time[0] == timestamps[first_reference]
