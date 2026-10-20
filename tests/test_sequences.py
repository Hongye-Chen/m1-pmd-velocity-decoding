"""Tests for sequence building and normalization on synthetic spikes.

Spikes sit exactly on bin edges, so off-by-one errors, reversed bin order, or a
window crossing a trial boundary show up as wrong counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_decoding.features import FeatureSpec, build_features
from neural_decoding.sequences import (
    InputRepresentation,
    SequenceError,
    build_sequences,
    fit_sequence_normalization,
)
from neural_decoding.splits import TEST, TRAIN, VALIDATION

_TIMESTAMPS = np.arange(200, dtype=np.float64) * 0.01
_SPEC = FeatureSpec(history_ms=200, n_bins=4, cadence_ms=10)


def _trials(split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_row": [0, 1],
            "start_time": [0.0, 1.0],
            "stop_time": [0.99, 1.99],
            "split": [split, split],
            "eligible": [True, True],
        }
    )


def _identity(spike_times, split: str = TRAIN):
    velocity = np.column_stack([np.sin(_TIMESTAMPS * 7.0), np.cos(_TIMESTAMPS * 5.0)])
    return build_features(
        timestamps=_TIMESTAMPS,
        velocity=velocity,
        spike_times=spike_times,
        trials=_trials(split),
        split=split,
        lag_ms=50,
        eligibility_lag_ms=50,
        spec=_SPEC,
        allow_held_out=split == TEST,
    )


def _sequences(identity, spike_times, representation, split: str = TRAIN):
    return build_sequences(
        identity,
        timestamps=_TIMESTAMPS,
        spike_times=spike_times,
        trials=_trials(split),
        representation=representation,
    )


def test_bins_are_chronological_half_open_and_end_at_the_lag() -> None:
    representation = InputRepresentation(history_ms=100, bin_ms=20)
    placeholder = _identity([np.empty(0)])
    reference = float(placeholder.reference_time[0])
    edges = reference + representation.edges_s()
    spikes = np.sort(
        np.array(
            [
                edges[0] - 1e-6,  # before the window
                edges[0],  # left edge of the oldest bin is included
                edges[2] - 1e-6,  # last instant of bin 1
                edges[2],  # first instant of bin 2
                edges[5] - 1e-6,  # last instant of the newest bin
                reference,  # exactly t - lag belongs to no bin
            ]
        )
    )

    identity = _identity([spikes])
    sequences = _sequences(identity, [spikes], representation)

    assert identity.target_index[0] - identity.reference_index[0] == 5
    assert sequences.counts.shape == (identity.n_samples, 5, 1)
    assert sequences.counts[0, :, 0].tolist() == [1, 1, 1, 0, 1]
    assert np.array_equal(sequences.target_index, identity.target_index)


def test_the_full_window_reproduces_the_baseline_counts() -> None:
    rng = np.random.default_rng(3)
    spikes = [np.sort(rng.uniform(0.0, 2.0, size=150)) for _ in range(3)]
    identity = _identity(spikes)

    sequences = _sequences(identity, spikes, InputRepresentation(history_ms=200, bin_ms=50))

    assert np.array_equal(sequences.counts, identity.counts)


def test_a_window_reaching_before_its_trial_is_refused() -> None:
    spikes = [np.array([0.5, 1.5])]
    identity = _identity(spikes)

    # The first target of the second trial sits 250 ms after its start, so a
    # 300 ms history would read spikes from the first trial.
    with pytest.raises(SequenceError, match="trial boundary"):
        _sequences(identity, spikes, InputRepresentation(history_ms=300, bin_ms=50))


def test_normalization_uses_training_elements_with_one_scale_per_unit() -> None:
    rng = np.random.default_rng(5)
    spikes = [np.sort(rng.uniform(0.0, 2.0, size=200)), np.empty(0)]
    representation = InputRepresentation(history_ms=100, bin_ms=10)
    sequences = _sequences(_identity(spikes), spikes, representation)

    normalization = fit_sequence_normalization(sequences)

    elements = sequences.counts.reshape(-1, 2).astype(np.float64)
    assert normalization.count_mean.shape == (2,)
    assert np.allclose(normalization.count_mean, elements.mean(axis=0))
    assert normalization.count_std[0] == pytest.approx(elements[:, 0].std())
    assert normalization.count_std[1] == 1.0 and normalization.n_units_at_floor == 1
    assert normalization.n_elements == sequences.n_samples * representation.n_bins
    assert np.allclose(normalization.target_mean, sequences.targets.mean(axis=0))

    validation = _sequences(_identity(spikes, VALIDATION), spikes, representation, VALIDATION)
    with pytest.raises(SequenceError, match="train"):
        fit_sequence_normalization(validation)
    applied = normalization.apply_counts(validation.counts)
    expected = (validation.counts - normalization.count_mean) / normalization.count_std
    assert np.allclose(applied, expected, atol=1e-6)


def test_held_out_sequences_are_refused() -> None:
    spikes = [np.array([0.5, 1.5])]
    identity = _identity(spikes, TEST)

    with pytest.raises(SequenceError, match="held-out"):
        _sequences(identity, spikes, InputRepresentation(history_ms=100, bin_ms=50), TEST)
