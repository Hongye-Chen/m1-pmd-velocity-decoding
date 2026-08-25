"""Synthetic tests for the causal feature protocol.

Every defect guarded here would raise the reported score rather than fail: a
reversed target lag predicts the past from the future, a window crossing a trial
boundary mixes unrelated reaches, and statistics fitted outside the training split
leak information into the model input. None of them raises an exception on its own.

The inputs are small hand-built timelines whose correct answer is known
independently. Bin-edge cases are placed at the edges the specification actually
computes, not at decimal literals, because a literal such as ``1.15`` is not the
same double as ``1.21 - 0.05`` and would make the assertion test rounding instead
of the half-open rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_decoding.features import (
    DEFAULT_SPEC,
    FeatureError,
    FeatureSpec,
    build_features,
    fit_normalization,
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
    **kwargs,
):
    timestamps = _timeline(n_samples)
    return build_features(
        timestamps=timestamps,
        velocity=_velocity(timestamps),
        spike_times=spike_times,
        trials=trials,
        split=split,
        **kwargs,
    )


def test_bins_are_half_open_and_ordered_oldest_first() -> None:
    # These bounds yield exactly one reference time, at 1.21 s, and sit off the
    # sample grid so the trial edges cannot be confused with a sample.
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
    # A spike is counted once across the whole history, never double counted.
    for sample in range(features.n_samples):
        window = features.reference_time[sample] + np.array([-spec.history_s, 0.0])
        expected = int(np.count_nonzero((inside >= window[0]) & (inside < window[1])))
        assert int(features.counts[sample, :, 0].sum()) == expected


def test_target_is_the_velocity_one_lag_after_the_reference_time() -> None:
    trials = _trials([(0, 1.005, 3.005)])

    features = _build(trials, [np.array([1.5])])

    lag = features.target_time - features.reference_time
    assert np.allclose(lag, DEFAULT_SPEC.target_lag_s, atol=1e-12)
    # Direction: the target is always in the future of the neural data.
    assert np.all(features.target_time > features.reference_time)
    # Identity: velocity encodes its own timestamp, so the target must be the
    # velocity at the later time and never at the reference time.
    assert np.allclose(features.targets[:, 0], features.target_time, atol=1e-12)
    assert np.allclose(features.targets[:, 1], -2.0 * features.target_time, atol=1e-12)
    assert not np.any(np.isclose(features.targets[:, 0], features.reference_time))


def test_a_negative_lag_is_refused() -> None:
    with pytest.raises(FeatureError, match="predict the past from the future"):
        FeatureSpec(target_lag_ms=-100)


def test_no_window_crosses_a_trial_boundary() -> None:
    trials = _trials([(0, 1.005, 3.005), (1, 4.005, 6.005)])
    # Spikes exist only outside the trials, so a window that leaked past a trial
    # boundary would pick them up.
    outside = np.array([0.5, 3.2, 3.5, 3.9, 6.5, 7.0])

    features = _build(trials, [outside])

    assert features.n_samples > 0
    assert int(features.counts.sum()) == 0

    starts = trials.set_index("trial_row")["start_time"]
    stops = trials.set_index("trial_row")["stop_time"]
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

    allowed = _build(trials, spikes, split="test", allow_held_out=True)
    assert allowed.n_samples > 0


def test_normalization_is_fitted_on_training_data_only() -> None:
    trials = _trials([(0, 1.005, 3.005), (151, 5.005, 7.005)])
    spikes = [np.array([1.4, 1.9, 2.4]), np.array([5.4, 5.9, 6.4])]
    train = _build(trials, spikes, split="train")
    validation = _build(trials, spikes, split="validation")

    with pytest.raises(FeatureError, match="must be fitted on"):
        fit_normalization(validation)

    normalization = fit_normalization(train)

    assert normalization.fitted_on == "train"
    assert normalization.n_samples == train.n_samples
    assert np.allclose(normalization.feature_mean, train.counts.astype(float).mean(axis=0))
    assert np.allclose(normalization.target_mean, train.targets.mean(axis=0))

    standardized_train = normalization.apply_features(train.counts)
    varying = normalization.feature_std > normalization.std_floor
    assert np.allclose(standardized_train.mean(axis=0)[varying], 0.0, atol=1e-10)

    # Validation is transformed with the training statistics, so its own mean must
    # not have been removed.
    standardized_validation = normalization.apply_targets(validation.targets)
    assert not np.allclose(standardized_validation.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(
        normalization.invert_targets(normalization.apply_targets(validation.targets)),
        validation.targets,
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
    # Samples stay ordered by trial and then by time.
    assert np.all(np.diff(first.reference_index) > 0)


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
        )


def test_history_must_divide_into_whole_bins() -> None:
    with pytest.raises(FeatureError, match="not divisible"):
        FeatureSpec(history_ms=200, n_bins=3)


def test_a_protocol_off_the_sample_grid_is_refused() -> None:
    trials = _trials([(0, 1.005, 3.005)])

    with pytest.raises(FeatureError, match="not a whole number"):
        _build(trials, [np.array([1.5])], spec=FeatureSpec(cadence_ms=7))


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
    features = _build(trials, [np.array([start + 0.1])], n_samples=n_samples)

    assert features.reference_index[0] == first_reference
    assert features.reference_time[0] == timestamps[first_reference]
