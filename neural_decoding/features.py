"""Causal feature construction from spike history and cursor velocity.

One sample is built from a neural reference time ``t``:

- the history covers ``[t - 200 ms, t)`` as four chronological half-open 50 ms bins,
  so bin 0 is the oldest and a spike falling exactly on a bin's left edge belongs to
  that bin while a spike falling exactly on ``t`` belongs to no bin;
- the target is the stored cursor velocity at ``t + 100 ms``, that is strictly after
  the neural data used to predict it;
- both the whole history and the target must lie inside one trial, so no window ever
  spans a trial boundary.

The protocol is expressed in integer milliseconds because the bin grid has to be
exact: deriving each edge from an integer division keeps ``t - 150 ms`` at the
nearest double to ``0.150`` instead of an accumulated sum whose rounding would make
edge membership ambiguous.

For the same reason the window is positioned by counting samples on the behavior
grid rather than by comparing derived times against stored timestamps. Asking for
the first sample at or after ``trial_start + 0.200`` silently drops a valid sample
whenever the trial starts exactly on the grid, because the float sum lands a
fraction of a nanosecond above the stored timestamp. Counting a whole number of
samples from the first in-trial sample is exact, and it keeps the oldest bin edge
inside the trial by construction.

Normalization is fitted on the training split alone and applied unchanged
elsewhere. Both restrictions are enforced rather than documented, because a
reversed lag, a window crossing a trial boundary, or statistics fitted on
validation data would all raise the reported score instead of failing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from neural_decoding.splits import SPLIT_NAMES, TEST, TRAIN

if TYPE_CHECKING:  # pragma: no cover - annotation only, keeps PyNWB out of imports
    from neural_decoding.dataset import SessionData

__all__ = [
    "DEFAULT_SPEC",
    "EXPECTED_SAMPLE_COUNTS",
    "FeatureError",
    "FeatureSet",
    "FeatureSpec",
    "Normalization",
    "build_features",
    "build_split_features",
    "fit_normalization",
]

# Training and validation example counts from the recorded feasibility run.
EXPECTED_SAMPLE_COUNTS = {"train": 42_407, "validation": 8_063}

_UNIFORMITY_TOLERANCE = 1e-3
_GRID_TOLERANCE = 1e-6


class FeatureError(RuntimeError):
    """Raised when feature construction would violate the causal protocol."""


@dataclass(frozen=True)
class FeatureSpec:
    """The frozen causal feature protocol, in integer milliseconds."""

    history_ms: int = 200
    n_bins: int = 4
    cadence_ms: int = 10
    target_lag_ms: int = 100

    def __post_init__(self) -> None:
        if self.n_bins <= 0:
            raise FeatureError(f"n_bins must be positive, got {self.n_bins}.")
        if self.history_ms <= 0:
            raise FeatureError(f"history_ms must be positive, got {self.history_ms}.")
        if self.history_ms % self.n_bins:
            raise FeatureError(
                f"history_ms {self.history_ms} is not divisible into {self.n_bins} bins."
            )
        if self.cadence_ms <= 0:
            raise FeatureError(f"cadence_ms must be positive, got {self.cadence_ms}.")
        if self.target_lag_ms < 0:
            raise FeatureError(
                f"target_lag_ms must not be negative, got {self.target_lag_ms}; a "
                "negative lag would predict the past from the future."
            )

    @property
    def bin_ms(self) -> int:
        return self.history_ms // self.n_bins

    @property
    def history_s(self) -> float:
        return self.history_ms / 1000.0

    @property
    def cadence_s(self) -> float:
        return self.cadence_ms / 1000.0

    @property
    def target_lag_s(self) -> float:
        return self.target_lag_ms / 1000.0

    def bin_offsets_s(self) -> np.ndarray:
        """Return the ``n_bins + 1`` bin edges relative to the reference time."""
        edges_ms = np.arange(self.n_bins + 1, dtype=np.int64) * self.bin_ms - self.history_ms
        return edges_ms / 1000.0

    def as_dict(self) -> dict[str, int]:
        return {
            "history_ms": self.history_ms,
            "n_bins": self.n_bins,
            "bin_ms": self.bin_ms,
            "cadence_ms": self.cadence_ms,
            "target_lag_ms": self.target_lag_ms,
        }


DEFAULT_SPEC = FeatureSpec()


@dataclass(frozen=True)
class FeatureSet:
    """Causal samples for one split, in deterministic trial-then-time order.

    ``counts`` is ``(n_samples, n_bins, n_units)`` with bin 0 the oldest. Flattening
    it with :attr:`flat_counts` yields bin-major columns, which is the layout the
    regression models consume.
    """

    split: str
    spec: FeatureSpec
    counts: np.ndarray
    targets: np.ndarray
    reference_index: np.ndarray
    target_index: np.ndarray
    reference_time: np.ndarray
    target_time: np.ndarray
    trial_row: np.ndarray
    unit_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.counts.shape[1])

    @property
    def n_units(self) -> int:
        return int(self.counts.shape[2])

    @property
    def flat_counts(self) -> np.ndarray:
        """Return counts as ``(n_samples, n_bins * n_units)`` in bin-major order."""
        return self.counts.reshape(self.n_samples, self.n_bins * self.n_units)

    def sample_manifest(self) -> pd.DataFrame:
        """Return the per-sample identity record, aligned row-for-row with the arrays."""
        return pd.DataFrame(
            {
                "sample_index": np.arange(self.n_samples, dtype=np.int64),
                "split": self.split,
                "trial_row": self.trial_row,
                "reference_index": self.reference_index,
                "reference_time": self.reference_time,
                "target_index": self.target_index,
                "target_time": self.target_time,
                "target_velocity_x": self.targets[:, 0],
                "target_velocity_y": self.targets[:, 1],
                "total_spikes": self.counts.sum(axis=(1, 2), dtype=np.int64),
            }
        )


@dataclass(frozen=True)
class Normalization:
    """Feature and target standardization fitted on one split only."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    fitted_on: str
    n_samples: int
    std_floor: float
    n_features_at_floor: int

    def apply_features(self, counts: np.ndarray) -> np.ndarray:
        """Standardize spike counts with the stored statistics."""
        if counts.shape[1:] != self.feature_mean.shape:
            raise FeatureError(
                f"Counts with shape {counts.shape[1:]} do not match the fitted "
                f"statistics with shape {self.feature_mean.shape}."
            )
        return (counts.astype(np.float64) - self.feature_mean) / self.feature_std

    def apply_targets(self, targets: np.ndarray) -> np.ndarray:
        """Standardize targets with the stored statistics."""
        return (np.asarray(targets, dtype=np.float64) - self.target_mean) / self.target_std

    def invert_targets(self, normalized: np.ndarray) -> np.ndarray:
        """Return standardized targets to the recorded cm/s scale."""
        return np.asarray(normalized, dtype=np.float64) * self.target_std + self.target_mean

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "n_samples": self.n_samples,
            "std_floor": self.std_floor,
            "n_features_at_floor": self.n_features_at_floor,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Normalization:
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
            target_mean=np.asarray(payload["target_mean"], dtype=np.float64),
            target_std=np.asarray(payload["target_std"], dtype=np.float64),
            fitted_on=str(payload["fitted_on"]),
            n_samples=int(payload["n_samples"]),
            std_floor=float(payload["std_floor"]),
            n_features_at_floor=int(payload["n_features_at_floor"]),
        )


def build_features(
    *,
    timestamps: np.ndarray,
    velocity: np.ndarray,
    spike_times: Sequence[np.ndarray],
    trials: pd.DataFrame,
    split: str,
    spec: FeatureSpec = DEFAULT_SPEC,
    unit_ids: Sequence[int] | None = None,
    allow_held_out: bool = False,
) -> FeatureSet:
    """Build causal samples for one split from eligible trials only.

    Args:
        timestamps: Uniformly sampled behavior timebase.
        velocity: Cursor velocity, one row per timestamp.
        spike_times: Sorted spike times per unit, in stable unit order.
        trials: Trial table carrying ``trial_row``, ``start_time``, ``stop_time``,
            ``split``, and ``eligible``.
        split: Split to build.
        spec: Causal protocol to apply.
        unit_ids: Identifiers recorded alongside the features.
        allow_held_out: Required to build the held-out split, which must stay
            untouched until the frozen evaluation.
    """
    if split not in SPLIT_NAMES:
        raise FeatureError(f"Unknown split {split!r}; expected one of {list(SPLIT_NAMES)}.")
    if split == TEST and not allow_held_out:
        raise FeatureError(
            "Refusing to build held-out features. The held-out split stays unopened "
            "until the experiment is frozen; pass allow_held_out to override."
        )

    timestamps = np.asarray(timestamps, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] != timestamps.size:
        raise FeatureError(
            f"Velocity shape {velocity.shape} does not match {timestamps.size} timestamps."
        )

    missing = [
        name
        for name in ("trial_row", "start_time", "stop_time", "split", "eligible")
        if name not in trials.columns
    ]
    if missing:
        raise FeatureError(f"Trial table is missing required columns: {missing}")

    interval = _sample_interval(timestamps)
    history_samples = _samples_for(spec.history_s, interval, "History")
    lag_samples = _samples_for(spec.target_lag_s, interval, "Target lag")
    stride = max(1, _samples_for(spec.cadence_s, interval, "Cadence"))

    prepared = _prepared_spike_times(spike_times)
    offsets = spec.bin_offsets_s()

    selected = trials.loc[
        (trials["split"] == split) & trials["eligible"].astype(bool)
    ].sort_values("trial_row")

    index_chunks: list[np.ndarray] = []
    row_chunks: list[np.ndarray] = []
    for row, start, stop in zip(
        selected["trial_row"].to_numpy(dtype=np.int64),
        selected["start_time"].to_numpy(dtype=np.float64),
        selected["stop_time"].to_numpy(dtype=np.float64),
    ):
        indices = _reference_indices(
            timestamps, float(start), float(stop), history_samples, lag_samples, stride
        )
        if indices.size:
            index_chunks.append(indices)
            row_chunks.append(np.full(indices.size, int(row), dtype=np.int64))

    if not index_chunks:
        raise FeatureError(f"No eligible trial in split {split!r} is long enough for one sample.")

    reference_index = np.concatenate(index_chunks)
    trial_row = np.concatenate(row_chunks)
    target_index = reference_index + lag_samples
    reference_time = timestamps[reference_index]
    target_time = timestamps[target_index]

    counts = np.empty((reference_index.size, spec.n_bins, len(prepared)), dtype=np.uint16)
    for position, spikes in enumerate(prepared):
        unit_counts = _count_in_bins(spikes, reference_time, offsets)
        peak = int(unit_counts.max()) if unit_counts.size else 0
        if peak > np.iinfo(np.uint16).max:
            raise FeatureError(
                f"Unit at position {position} reaches {peak} spikes in one bin, which "
                "overflows the count representation."
            )
        counts[:, :, position] = unit_counts

    identifiers = (
        np.arange(len(prepared), dtype=np.int64)
        if unit_ids is None
        else np.asarray(unit_ids, dtype=np.int64)
    )
    if identifiers.size != len(prepared):
        raise FeatureError(
            f"Got {identifiers.size} unit identifiers for {len(prepared)} units."
        )

    return FeatureSet(
        split=split,
        spec=spec,
        counts=counts,
        targets=velocity[target_index],
        reference_index=reference_index,
        target_index=target_index,
        reference_time=reference_time,
        target_time=target_time,
        trial_row=trial_row,
        unit_ids=identifiers,
    )


def build_split_features(
    session: SessionData,
    split: str,
    spec: FeatureSpec = DEFAULT_SPEC,
    *,
    allow_held_out: bool = False,
) -> FeatureSet:
    """Build causal samples for one split of a loaded session."""
    return build_features(
        timestamps=session.behavior_timestamps,
        velocity=session.cursor_velocity,
        spike_times=session.spike_times,
        trials=session.trials,
        split=split,
        spec=spec,
        unit_ids=[unit.unit_id for unit in session.units],
        allow_held_out=allow_held_out,
    )


def fit_normalization(
    feature_set: FeatureSet, *, std_floor: float = 1e-8
) -> Normalization:
    """Fit standardization statistics on the training split.

    Fitting on any other split would let validation or held-out data influence the
    model input, so it is refused rather than warned about.
    """
    if feature_set.split != TRAIN:
        raise FeatureError(
            f"Normalization must be fitted on the {TRAIN!r} split, not "
            f"{feature_set.split!r}."
        )

    counts = feature_set.counts.astype(np.float64)
    feature_mean = counts.mean(axis=0)
    feature_std = counts.std(axis=0)
    at_floor = feature_std < std_floor
    feature_std = np.where(at_floor, 1.0, feature_std)

    targets = feature_set.targets.astype(np.float64)
    target_std = targets.std(axis=0)
    if np.any(target_std < std_floor):
        raise FeatureError("A target axis has no variance; standardization would be undefined.")

    return Normalization(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=targets.mean(axis=0),
        target_std=target_std,
        fitted_on=feature_set.split,
        n_samples=feature_set.n_samples,
        std_floor=std_floor,
        n_features_at_floor=int(np.count_nonzero(at_floor)),
    )


def _sample_interval(timestamps: np.ndarray) -> float:
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise FeatureError("At least two behavior timestamps are required.")
    steps = np.diff(timestamps)
    if np.any(steps <= 0):
        raise FeatureError("Behavior timestamps must be strictly increasing.")
    interval = float(np.median(steps))
    if not np.all(np.abs(steps - interval) <= interval * _UNIFORMITY_TOLERANCE):
        raise FeatureError(
            "Behavior timestamps are not uniformly sampled, so a fixed sample offset "
            "cannot represent the target lag."
        )
    return interval


def _prepared_spike_times(spike_times: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(spike_times) == 0:
        raise FeatureError("At least one unit is required.")
    prepared: list[np.ndarray] = []
    for position, spikes in enumerate(spike_times):
        array = np.asarray(spikes, dtype=np.float64).ravel()
        if array.size and not np.all(np.diff(array) >= 0):
            raise FeatureError(
                f"Spike times for the unit at position {position} are not sorted, so "
                "bin counts would be wrong."
            )
        prepared.append(array)
    return prepared


def _samples_for(duration_s: float, interval: float, label: str) -> int:
    """Return a duration as a whole number of behavior samples."""
    exact = duration_s / interval
    count = int(round(exact))
    if abs(exact - count) > _GRID_TOLERANCE:
        raise FeatureError(
            f"{label} of {duration_s * 1000:.4f} ms is not a whole number of "
            f"{interval * 1000:.4f} ms behavior samples."
        )
    return count


def _reference_indices(
    timestamps: np.ndarray,
    trial_start: float,
    trial_stop: float,
    history_samples: int,
    lag_samples: int,
    stride: int,
) -> np.ndarray:
    """Return reference sample indices whose history and target fit inside one trial.

    The bounds are integer sample positions, so the first usable reference sits a
    whole history length after the first in-trial sample and the last one leaves a
    whole lag before the final in-trial sample.
    """
    first_in_trial = int(np.searchsorted(timestamps, trial_start, side="left"))
    last_in_trial = int(np.searchsorted(timestamps, trial_stop, side="right")) - 1
    first = first_in_trial + history_samples
    last = min(last_in_trial - lag_samples, timestamps.size - 1 - lag_samples)
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, stride, dtype=np.int64)


def _count_in_bins(
    spike_times: np.ndarray, reference_time: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Return half-open spike counts per bin for one unit at every reference time."""
    edges = reference_time[:, None] + offsets[None, :]
    positions = np.searchsorted(spike_times, edges, side="left")
    return np.diff(positions, axis=1)
