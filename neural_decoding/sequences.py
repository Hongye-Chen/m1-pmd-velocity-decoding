"""Chronological spike-count sequences for the sequence-to-one GRU decoder.

Every input representation is built on the official baseline target identities,
never on its own eligibility rule, so all representations predict identical
targets. For a target at behavior time ``t`` with neural reference ``t - lag``:

- the sequence covers ``[t - lag - history, t - lag)`` split into chronological
  half-open bins of one width, so element 0 is the oldest and a spike falling
  exactly on ``t - lag`` belongs to no element;
- the earliest bin edge and the target both lie inside the target's own trial.

Count normalization uses one mean and one scale per unit, pooled over every
training sequence element, so the same unit is scaled identically at every
position. Statistics may only be fitted on the training split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neural_decoding.features import FeatureSet
from neural_decoding.splits import TEST, TRAIN

__all__ = [
    "InputRepresentation",
    "SequenceError",
    "SequenceNormalization",
    "SequenceSet",
    "build_sequences",
    "fit_sequence_normalization",
]


class SequenceError(RuntimeError):
    """Raised when sequence construction or normalization would break the protocol."""


@dataclass(frozen=True, order=True)
class InputRepresentation:
    """One history length and bin width, in integer milliseconds."""

    history_ms: int
    bin_ms: int

    def __post_init__(self) -> None:
        if self.history_ms <= 0 or self.bin_ms <= 0:
            raise SequenceError(
                f"History {self.history_ms} ms and width {self.bin_ms} ms must be positive."
            )
        if self.history_ms % self.bin_ms:
            raise SequenceError(
                f"History {self.history_ms} ms is not divisible into {self.bin_ms} ms bins."
            )

    @property
    def n_bins(self) -> int:
        return self.history_ms // self.bin_ms

    @property
    def label(self) -> str:
        return f"history {self.history_ms} ms / width {self.bin_ms} ms"

    def edges_s(self) -> np.ndarray:
        """Return the ``n_bins + 1`` bin edges relative to the neural reference time."""
        edges_ms = np.arange(self.n_bins + 1, dtype=np.int64) * self.bin_ms - self.history_ms
        return edges_ms / 1000.0


@dataclass(frozen=True)
class SequenceSet:
    """Sequences for one split and representation, aligned with their identities.

    ``counts`` is ``(n_samples, n_bins, n_units)`` with element 0 the oldest.
    """

    split: str
    representation: InputRepresentation
    lag_ms: int
    counts: np.ndarray
    targets: np.ndarray
    trial_row: np.ndarray
    reference_index: np.ndarray
    target_index: np.ndarray
    target_time: np.ndarray
    unit_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.counts.shape[2])

    @property
    def trial_rows(self) -> np.ndarray:
        return np.unique(self.trial_row)


@dataclass(frozen=True)
class SequenceNormalization:
    """Per-unit count and per-axis target standardization from training data."""

    count_mean: np.ndarray
    count_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    representation: InputRepresentation
    n_samples: int
    n_elements: int
    n_units_at_floor: int
    std_floor: float
    trial_rows: tuple[int, ...]

    def apply_counts(self, counts: np.ndarray) -> np.ndarray:
        """Standardize ``(n, n_bins, n_units)`` counts into a float32 network input."""
        array = np.asarray(counts)
        if array.ndim != 3 or array.shape[2] != self.count_mean.size:
            raise SequenceError(
                f"Counts with shape {array.shape} do not match {self.count_mean.size} units."
            )
        normalized = (array.astype(np.float64) - self.count_mean) / self.count_std
        return normalized.astype(np.float32)

    def apply_targets(self, targets: np.ndarray) -> np.ndarray:
        return (np.asarray(targets, dtype=np.float64) - self.target_mean) / self.target_std

    def invert_targets(self, normalized: np.ndarray) -> np.ndarray:
        """Return normalized velocity to the recorded cm/s scale."""
        return np.asarray(normalized, dtype=np.float64) * self.target_std + self.target_mean

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitted_on": TRAIN,
            "history_ms": self.representation.history_ms,
            "bin_ms": self.representation.bin_ms,
            "n_samples": self.n_samples,
            "n_count_elements_per_unit": self.n_elements,
            "n_trials": len(self.trial_rows),
            "n_units_at_floor": self.n_units_at_floor,
            "std_floor": self.std_floor,
            "count_mean": self.count_mean.tolist(),
            "count_std": self.count_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }


def build_sequences(
    identity: FeatureSet,
    *,
    timestamps: np.ndarray,
    spike_times: Sequence[np.ndarray],
    trials: pd.DataFrame,
    representation: InputRepresentation,
) -> SequenceSet:
    """Build one representation's sequences on an existing target identity.

    ``identity`` supplies the targets, trial rows, and neural reference times; its
    own counts are not used. The window is checked against each trial's
    behavior-grid bounds rather than assumed from the eligibility rule.
    """
    if identity.split == TEST:
        raise SequenceError(
            "Refusing to build held-out sequences. The held-out split stays unopened "
            "until the experiment is frozen."
        )
    cadence_ms = identity.spec.cadence_ms
    if representation.history_ms % cadence_ms or representation.bin_ms % cadence_ms:
        raise SequenceError(
            f"{representation.label} is not a whole number of {cadence_ms} ms samples."
        )

    times = np.asarray(timestamps, dtype=np.float64)
    first_index, last_index = _trial_grid_bounds(times, trials, identity)
    history_samples = representation.history_ms // cadence_ms
    window_start = identity.reference_index - history_samples
    if np.any(window_start < first_index) or np.any(identity.target_index > last_index):
        raise SequenceError(
            f"{representation.label} would cross a trial boundary for at least one target."
        )
    if np.any(identity.reference_index >= identity.target_index):
        raise SequenceError("Every neural window must end strictly before its target.")

    prepared = [np.asarray(spikes, dtype=np.float64).ravel() for spikes in spike_times]
    if len(prepared) != identity.n_units:
        raise SequenceError(
            f"Got {len(prepared)} spike trains for {identity.n_units} identity units."
        )
    edges = identity.reference_time[:, None] + representation.edges_s()[None, :]
    counts = np.empty(
        (identity.n_samples, representation.n_bins, len(prepared)), dtype=np.uint16
    )
    for position, spikes in enumerate(prepared):
        if spikes.size and not np.all(np.diff(spikes) >= 0):
            raise SequenceError(f"Spike times for unit position {position} are not sorted.")
        unit_counts = np.diff(np.searchsorted(spikes, edges, side="left"), axis=1)
        if unit_counts.size and int(unit_counts.max()) > np.iinfo(np.uint16).max:
            raise SequenceError(f"Unit position {position} overflows the count type.")
        counts[:, :, position] = unit_counts

    return SequenceSet(
        split=identity.split,
        representation=representation,
        lag_ms=identity.lag_ms,
        counts=counts,
        targets=identity.targets.copy(),
        trial_row=identity.trial_row.copy(),
        reference_index=identity.reference_index.copy(),
        target_index=identity.target_index.copy(),
        target_time=identity.target_time.copy(),
        unit_ids=identity.unit_ids.copy(),
    )


def fit_sequence_normalization(
    sequences: SequenceSet, *, std_floor: float = 1e-8
) -> SequenceNormalization:
    """Fit per-unit count and per-axis target statistics on training sequences only."""
    if sequences.split != TRAIN:
        raise SequenceError(
            f"Normalization must be fitted on {TRAIN!r} sequences, not {sequences.split!r}."
        )
    if sequences.n_samples == 0:
        raise SequenceError("Cannot fit normalization on an empty sequence set.")

    elements = sequences.counts.reshape(-1, sequences.n_units).astype(np.float64)
    count_mean = elements.mean(axis=0)
    count_std = elements.std(axis=0)
    at_floor = count_std < std_floor
    count_std = np.where(at_floor, 1.0, count_std)

    targets = np.asarray(sequences.targets, dtype=np.float64)
    target_std = targets.std(axis=0)
    if np.any(target_std < std_floor):
        raise SequenceError("A target axis has no training variance.")

    return SequenceNormalization(
        count_mean=count_mean,
        count_std=count_std,
        target_mean=targets.mean(axis=0),
        target_std=target_std,
        representation=sequences.representation,
        n_samples=sequences.n_samples,
        n_elements=int(elements.shape[0]),
        n_units_at_floor=int(np.count_nonzero(at_floor)),
        std_floor=float(std_floor),
        trial_rows=tuple(int(row) for row in sequences.trial_rows),
    )


def _trial_grid_bounds(
    timestamps: np.ndarray, trials: pd.DataFrame, identity: FeatureSet
) -> tuple[np.ndarray, np.ndarray]:
    """Return each sample's trial first and last behavior-grid index."""
    table = trials.set_index("trial_row")
    rows = identity.trial_rows
    missing = sorted(set(rows.tolist()) - set(table.index.tolist()))
    if missing:
        raise SequenceError(f"Trial rows {missing} are absent from the trial table.")
    records = table.loc[rows]
    if not np.all(records["split"].to_numpy() == identity.split) or not np.all(
        records["eligible"].to_numpy(dtype=bool)
    ):
        raise SequenceError(f"Every identity trial must be an eligible {identity.split!r} trial.")
    first = np.searchsorted(timestamps, records["start_time"].to_numpy(dtype=np.float64), side="left")
    last = np.searchsorted(timestamps, records["stop_time"].to_numpy(dtype=np.float64), side="right") - 1
    position = np.searchsorted(rows, identity.trial_row)
    return first[position], last[position]
