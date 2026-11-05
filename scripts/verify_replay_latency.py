"""Replay the validation targets incrementally and time each frozen decoder.

Run from the repository root with the project environment active::

    python -m scripts.verify_replay_latency

Exit codes: 0 when every check passes, 1 when a check or the replay fails, and 2
when the recording or a frozen artifact is unavailable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import scipy
import torch

from neural_decoding import joint_decoder
from neural_decoding.config import NWB_PATH_ENV_VAR, ConfigError, load_config, project_root
from neural_decoding.dataset import DatasetError, compute_sha256
from neural_decoding.features import FeatureError, FeatureSet, build_split_features
from neural_decoding.final_decoder import (
    FROZEN_SPEC,
    ValidationIdentity,
    build_validation_ledgers,
    filter_official_targets,
)
from neural_decoding.gru import GruDecoder, GruError, predict_normalized
from neural_decoding.joint_decoder import (
    DecoderModel,
    JointDecoderError,
    ObservationModel,
    ScheduledObservations,
    TrialBounds,
    build_assessment_trial,
    build_scheduled_observations,
    build_trial_bounds,
    filter_trial,
)
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.metrics import summarize_predictions
from neural_decoding.models import ModelError, RidgeDecoder
from neural_decoding.sequences import SequenceError, SequenceNormalization, build_sequences
from neural_decoding.splits import VALIDATION
from scripts.evaluate_held_out import (
    AXIS_NAMES,
    CADENCE_MS,
    GRU_SEEDS,
    HISTORY_MS,
    LAG_MS,
    EvaluationError,
    PreparedEvaluation,
    prepare_evaluation,
)

logger = get_logger("verify_replay_latency")

VALIDATION_SOURCE = Path("results/baselines/validation_predictions.csv")
KALMAN_SUMMARY = Path("results/kalman_validation/summary.json")
KALMAN_CONFIGURATION = Path("results/kalman_validation/configuration.json")
GRU_SUMMARY = Path("results/gru_architecture_selection/summary.json")
OUTPUT_PATH = Path("results/replay_latency/summary.json")

EXPECTED_TARGETS = 8_183
EXPECTED_TRIALS = 24
RIDGE_ATOL = 1e-10
GRU_ATOL = 1e-5
GRU_RTOL = 1e-5
RECORD_TOLERANCE = 1e-12
STATE_TOLERANCE = 1e-12
COVARIANCE_TOLERANCE = 1e-10
SCHEDULE_TOLERANCE_S = 1e-9

_IDENTITY_COLUMNS = (
    "sample_index",
    "trial_row",
    "reference_index",
    "target_index",
    "target_time",
    "target_velocity_x",
    "target_velocity_y",
)
_SCHEDULE_FIELDS = (
    "trial_row",
    "interval_ordinal",
    "interval_start_time",
    "interval_end_time",
    "state_index",
    "counts",
)


class ReplayError(RuntimeError):
    """Raised when the replay would read data it could not yet have."""


@dataclass(frozen=True)
class SpikeStream:
    """Every unit's spikes merged into one stream ordered by time, then unit."""

    times: np.ndarray
    units: np.ndarray
    n_units: int

    @classmethod
    def merge(cls, spike_times: Sequence[np.ndarray]) -> SpikeStream:
        trains = [np.asarray(spikes, dtype=np.float64).ravel() for spikes in spike_times]
        times = np.concatenate(trains)
        units = np.concatenate(
            [np.full(train.size, position, dtype=np.int64) for position, train in enumerate(trains)]
        )
        order = np.lexsort((units, times))
        return cls(times=times[order], units=units[order], n_units=len(trains))


class SpikeBuffer:
    """A rolling window over the spike stream for the current trial.

    Spikes are revealed only once the replay clock has passed them and are
    discarded once no later window can include them.
    """

    def __init__(self, stream: SpikeStream) -> None:
        self._stream = stream
        self._first = self._start = self._end = 0
        self.clock = -np.inf

    def reset(self, trial_start: float) -> None:
        """Empty the buffer; spikes before the trial start are never revealed to it."""
        position = int(np.searchsorted(self._stream.times, trial_start, side="left"))
        self._first = self._start = self._end = position
        self.clock = trial_start

    def advance(self, clock: float) -> None:
        """Reveal the spikes that occurred strictly before ``clock``."""
        if clock < self.clock:
            raise ReplayError("The replay clock moved backwards.")
        self.clock = clock
        self._end = int(np.searchsorted(self._stream.times, clock, side="left"))

    def discard_before(self, time_s: float) -> None:
        position = int(np.searchsorted(self._stream.times, time_s, side="left"))
        self._start = max(self._start, min(position, self._end))

    def count(self, edges: np.ndarray) -> np.ndarray:
        """Return per-unit counts in the half-open bins between consecutive edges."""
        if edges[-1] > self.clock:
            raise ReplayError("A neural window reaches past the replay clock.")
        times = self._stream.times[self._start : self._end]
        units = self._stream.units[self._start : self._end]
        n_bins, n_units = edges.size - 1, self._stream.n_units
        bins = np.searchsorted(edges, times, side="right") - 1
        inside = (bins >= 0) & (bins < n_bins)
        flat = np.bincount(bins[inside] * n_units + units[inside], minlength=n_bins * n_units)
        return flat.reshape(n_bins, n_units).astype(np.uint16)

    @property
    def latest_spike_time(self) -> float:
        """Return the time of the latest spike revealed in this trial."""
        if self._end > self._first:
            return float(self._stream.times[self._end - 1])
        return -np.inf


class ReplayStep(NamedTuple):
    trial_row: int
    target_index: int
    counts: np.ndarray
    latest_spike_time: float
    started_ns: int


@dataclass(frozen=True)
class BatchReferences:
    """The official validation targets and every model's batch predictions."""

    bounds: dict[int, TrialBounds]
    trial_starts: dict[int, float]
    identity: ValidationIdentity
    truth: np.ndarray
    manifest: pd.DataFrame
    features: FeatureSet
    ridge: np.ndarray
    gru: dict[int, np.ndarray]
    schedule: ScheduledObservations
    initial_positions: dict[int, np.ndarray]
    kalman: np.ndarray


def build_references(root: Path, prepared: PreparedEvaluation) -> BatchReferences:
    """Rebuild the official validation targets and batch predictions in memory."""
    session, models = prepared.session, prepared.models
    timestamps = session.behavior_timestamps
    trials = session.trials
    rows = tuple(
        sorted(
            int(row)
            for row in trials.loc[
                (trials["split"] == VALIDATION) & trials["eligible"].astype(bool), "trial_row"
            ]
        )
    )
    bounds = build_trial_bounds(timestamps, trials, rows, required_split=VALIDATION)
    starts = trials.set_index("trial_row")["start_time"]

    source = root / VALIDATION_SOURCE
    manifest = pd.read_csv(source, float_precision="round_trip")
    identity, ledger = build_validation_ledgers(
        manifest.loc[:, list(_IDENTITY_COLUMNS)],
        timestamps=timestamps,
        velocity=session.cursor_velocity,
        bounds=bounds,
        source_sha256=compute_sha256(source),
    )
    features = build_split_features(
        session, VALIDATION, lag_ms=LAG_MS, eligibility_lag_ms=LAG_MS, spec=prepared.spec
    )

    normalization = models.gru_normalization
    sequences = build_sequences(
        features,
        timestamps=timestamps,
        spike_times=session.spike_times,
        trials=trials,
        representation=normalization.representation,
    )
    if not np.array_equal(sequences.counts, features.counts):
        raise ReplayError("The GRU input differs from the official spike-count window.")
    device = torch.device(prepared.device.device)
    inputs = torch.from_numpy(normalization.apply_counts(sequences.counts)).to(device)
    batch_size = int(models.gru_configuration["optimization"]["batch_size"])
    gru = {
        seed: normalization.invert_targets(
            predict_normalized(models.gru[seed].to(device), inputs, batch_size=batch_size)
        )
        for seed in GRU_SEEDS
    }

    schedule = build_scheduled_observations(
        timestamps,
        session.spike_times,
        bounds,
        width_ms=FROZEN_SPEC.neural_width_ms,
        lag_ms=FROZEN_SPEC.interval_end_lag_ms,
    )
    initial_positions = {
        row: session.cursor_position[trial.first_index].copy() for row, trial in bounds.items()
    }
    kalman = filter_official_targets(
        models.kalman, schedule, bounds, identity, initial_positions=initial_positions
    ).prediction

    return BatchReferences(
        bounds=bounds,
        trial_starts={row: float(starts.loc[row]) for row in rows},
        identity=identity,
        truth=ledger.velocity,
        manifest=manifest,
        features=features,
        ridge=models.ridge.predict(features),
        gru=gru,
        schedule=schedule,
        initial_positions=initial_positions,
        kalman=kalman,
    )


def replay_targets(
    timestamps: np.ndarray,
    bounds: Mapping[int, TrialBounds],
    trial_starts: Mapping[int, float],
    stream: SpikeStream,
    offsets: np.ndarray,
) -> Iterator[ReplayStep]:
    """Yield each target's spike-count window in timestamp order, resetting every trial.

    The clock advances one behavior sample at a time; a target at ``t`` is counted
    over ``offsets`` around the sample 50 ms earlier once the clock reaches ``t``.
    """
    buffer = SpikeBuffer(stream)
    first_offset = (HISTORY_MS + LAG_MS) // CADENCE_MS
    lag_steps = LAG_MS // CADENCE_MS
    for row in sorted(bounds):
        trial = bounds[row]
        buffer.reset(trial_starts[row])
        for index in range(trial.first_index, trial.last_index + 1):
            started = time.perf_counter_ns()
            buffer.advance(float(timestamps[index]))
            if index - trial.first_index < first_offset:
                continue
            edges = timestamps[index - lag_steps] + offsets
            buffer.discard_before(float(edges[0]))
            yield ReplayStep(row, index, buffer.count(edges), buffer.latest_spike_time, started)


def replay_kalman_schedule(
    timestamps: np.ndarray,
    bounds: Mapping[int, TrialBounds],
    trial_starts: Mapping[int, float],
    stream: SpikeStream,
) -> tuple[ScheduledObservations, dict[tuple[int, int], int]]:
    """Count each fresh 50 ms interval at the step that uses it, 50 ms after it ends.

    Also returns the time spent on each 10 ms step, keyed by trial row and step.
    """
    width = FROZEN_SPEC.neural_width_ms // CADENCE_MS
    lag = FROZEN_SPEC.interval_end_lag_ms // CADENCE_MS
    buffer = SpikeBuffer(stream)
    records: dict[str, list[Any]] = {name: [] for name in _SCHEDULE_FIELDS}
    elapsed: dict[tuple[int, int], int] = {}
    for row in sorted(bounds):
        trial = bounds[row]
        buffer.reset(trial_starts[row])
        interval_start = 0
        for step in range(trial.n_steps):
            started = time.perf_counter_ns()
            index = trial.first_index + step
            buffer.advance(float(timestamps[index]))
            if step == interval_start + width + lag:
                first = trial.first_index + interval_start
                edges = timestamps[[first, first + width]]
                buffer.discard_before(float(edges[0]))
                records["counts"].append(buffer.count(edges)[0])
                records["trial_row"].append(row)
                records["interval_ordinal"].append(interval_start // width)
                records["interval_start_time"].append(edges[0])
                records["interval_end_time"].append(edges[1])
                records["state_index"].append(index)
                interval_start += width
            elapsed[(row, step)] = time.perf_counter_ns() - started
    schedule = ScheduledObservations(
        width_ms=FROZEN_SPEC.neural_width_ms,
        lag_ms=FROZEN_SPEC.interval_end_lag_ms,
        trial_row=np.asarray(records["trial_row"], dtype=np.int64),
        interval_ordinal=np.asarray(records["interval_ordinal"], dtype=np.int64),
        interval_start_time=np.asarray(records["interval_start_time"], dtype=np.float64),
        interval_end_time=np.asarray(records["interval_end_time"], dtype=np.float64),
        state_index=np.asarray(records["state_index"], dtype=np.int64),
        counts=np.vstack(records["counts"]).astype(np.uint16),
    )
    return schedule, elapsed


@contextmanager
def instrumented_filter(log: list[tuple[str, int, np.ndarray | None]], *, capture_updates: bool):
    """Record the end of every filter step and, optionally, every count vector it updates on.

    The official filter runs unchanged; two of its internal calls are wrapped.
    """
    original_velocity = joint_decoder._physical_velocity
    original_normalize = ObservationModel.normalize_counts

    def physical_velocity(mean, covariance, process):
        result = original_velocity(mean, covariance, process)
        log.append(("step", time.perf_counter_ns(), None))
        return result

    def normalize_counts(self, counts):
        log.append(("update", time.perf_counter_ns(), np.array(counts, copy=True)))
        return original_normalize(self, counts)

    joint_decoder._physical_velocity = physical_velocity
    if capture_updates:
        ObservationModel.normalize_counts = normalize_counts
    try:
        yield
    finally:
        joint_decoder._physical_velocity = original_velocity
        ObservationModel.normalize_counts = original_normalize


def read_filter_log(
    log: Sequence[tuple[str, int, np.ndarray | None]],
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    """Return each step's end time, the steps that ran an update, and the counts used."""
    step_ends: list[int] = []
    update_steps: list[int] = []
    update_counts: list[np.ndarray] = []
    pending: np.ndarray | None = None
    for kind, stamp, counts in log:
        if kind == "update":
            if pending is not None:
                raise ReplayError("Two measurement updates ran within one filter step.")
            pending = counts
            continue
        if pending is not None:
            update_steps.append(len(step_ends))
            update_counts.append(pending)
            pending = None
        step_ends.append(stamp)
    return np.asarray(step_ends, dtype=np.int64), update_steps, update_counts


def verify_kalman(
    model: DecoderModel,
    replayed: ScheduledObservations,
    batch: ScheduledObservations,
    bounds: Mapping[int, TrialBounds],
    initial_positions: Mapping[int, np.ndarray],
    timestamps: np.ndarray,
) -> dict[str, Any]:
    """Instrument the official filter on the replayed intervals and check every step."""
    process = model.process
    transition, offset, noise = (
        process.transition,
        process.offset,
        process.process_covariance,
    )
    dimension = transition.shape[0]
    position = list(process.position_indices)
    other = [index for index in range(dimension) if index not in position]
    width = FROZEN_SPEC.neural_width_ms // CADENCE_MS
    lag = FROZEN_SPEC.interval_end_lag_ms // CADENCE_MS
    width_s = FROZEN_SPEC.neural_width_ms / 1000.0
    lag_s = FROZEN_SPEC.interval_end_lag_ms / 1000.0

    same_schedule = (replayed.width_ms, replayed.lag_ms) == (batch.width_ms, batch.lag_ms) and all(
        np.array_equal(getattr(replayed, name), getattr(batch, name)) for name in _SCHEDULE_FIELDS
    )
    initial_deviation = mean_deviation = covariance_deviation = 0.0
    intervals_fresh = updates_as_scheduled = counts_as_scheduled = steps_complete = True
    n_steps = n_updates = 0
    forward: dict[int, np.ndarray] = {}
    for row in sorted(bounds):
        trial = bounds[row]
        assessment = build_assessment_trial(
            replayed, trial, initial_position=initial_positions[row]
        )
        log: list[tuple[str, int, np.ndarray | None]] = []
        with instrumented_filter(log, capture_updates=True):
            trace = filter_trial(model, assessment)
        step_ends, executed, used_counts = read_filter_log(log)
        forward[row] = trace.velocity
        n_steps += trial.n_steps
        n_updates += len(executed)
        steps_complete &= step_ends.size == trial.n_steps == trace.velocity.shape[0]

        expected = np.arange(width + lag, trial.n_steps, width, dtype=np.int64)
        updates_as_scheduled &= (
            np.array_equal(np.asarray(executed, dtype=np.int64), expected)
            and np.array_equal(trace.update_offsets, expected)
        )
        counts_as_scheduled &= len(used_counts) == assessment.counts.shape[0] and all(
            np.array_equal(used, scheduled)
            for used, scheduled in zip(used_counts, assessment.counts)
        )

        mask = replayed.trial_row == row
        starts = replayed.interval_start_time[mask]
        ends = replayed.interval_end_time[mask]
        state_index = replayed.state_index[mask]
        intervals_fresh &= bool(
            starts.size == expected.size
            and starts[0] == timestamps[trial.first_index]
            and np.array_equal(starts[1:], ends[:-1])
            and np.allclose(ends - starts, width_s, atol=SCHEDULE_TOLERANCE_S, rtol=0.0)
            and np.allclose(
                timestamps[state_index] - ends, lag_s, atol=SCHEDULE_TOLERANCE_S, rtol=0.0
            )
            and np.array_equal(state_index - trial.first_index, expected)
        )

        state = trace.normalized_state
        covariance = trace.posterior_covariance
        start = process.initial_mean.copy()
        start[position] = (
            initial_positions[row] - process.normalization.mean[position]
        ) / process.normalization.scale[position]
        initial_deviation = max(
            initial_deviation,
            float(np.max(np.abs(state[0] - start))),
            float(np.max(np.abs(covariance[0][position, :]))),
            float(
                np.max(
                    np.abs(
                        covariance[0][np.ix_(other, other)]
                        - process.initial_covariance[np.ix_(other, other)]
                    )
                )
            ),
        )
        predicted_mean = state[:-1] @ transition.T + offset
        predicted_covariance = transition @ covariance[:-1] @ transition.T + noise
        update_rows = expected - 1
        free = np.setdiff1d(np.arange(trial.n_steps - 1), update_rows)
        at_updates = trace.predicted_covariance_at_updates - predicted_covariance[update_rows]
        mean_deviation = max(
            mean_deviation, float(np.max(np.abs(state[free + 1] - predicted_mean[free])))
        )
        covariance_deviation = max(
            covariance_deviation,
            float(np.max(np.abs(covariance[free + 1] - predicted_covariance[free]))),
            float(np.max(np.abs(at_updates))),
        )

    order_independent = all(
        np.array_equal(
            filter_trial(
                model,
                build_assessment_trial(
                    replayed, bounds[row], initial_position=initial_positions[row]
                ),
            ).velocity,
            forward[row],
        )
        for row in sorted(bounds, reverse=True)
    )
    return {
        "same_schedule": same_schedule,
        "intervals_fresh": intervals_fresh,
        "updates_as_scheduled": updates_as_scheduled,
        "counts_as_scheduled": counts_as_scheduled,
        "steps_complete": steps_complete,
        "order_independent": order_independent,
        "initial_deviation": initial_deviation,
        "mean_deviation": mean_deviation,
        "covariance_deviation": covariance_deviation,
        "n_steps": n_steps,
        "n_updates": n_updates,
    }


def ridge_input(ridge: RidgeDecoder, counts: np.ndarray) -> np.ndarray:
    return ridge.normalization.apply_features(counts[np.newaxis]).reshape(1, -1)


def ridge_output(ridge: RidgeDecoder, design: np.ndarray) -> np.ndarray:
    return ridge.normalization.invert_targets(design @ ridge.coefficients.T + ridge.intercept)[0]


def gru_input(
    normalization: SequenceNormalization, counts: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Return one chronological four-bin sequence on the device."""
    tensor = torch.from_numpy(normalization.apply_counts(counts[np.newaxis])).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return tensor


def gru_output(
    model: GruDecoder, normalization: SequenceNormalization, sequence: torch.Tensor
) -> np.ndarray:
    """Decode one sequence from a zero hidden state and return cm/s."""
    return normalization.invert_targets(model(sequence).cpu().numpy())[0]


def run_replay(
    prepared: PreparedEvaluation, references: BatchReferences
) -> tuple[list[dict[str, object]], dict[str, Any], dict[str, Any]]:
    """Replay every model against its batch reference and return checks and details."""
    session, models = prepared.session, prepared.models
    timestamps = session.behavior_timestamps
    features = references.features
    stream = SpikeStream.merge(session.spike_times)
    offsets = prepared.spec.bin_offsets_s()
    normalization = models.gru_normalization
    if not np.array_equal(offsets, normalization.representation.edges_s()):
        raise ReplayError("Ridge and the GRU do not share one four-bin window.")
    device = torch.device(prepared.device.device)
    networks = {seed: models.gru[seed].to(device).eval() for seed in GRU_SEEDS}

    rows: list[int] = []
    indices: list[int] = []
    latest: list[float] = []
    counts: list[np.ndarray] = []
    ridge: list[np.ndarray] = []
    gru: dict[int, list[np.ndarray]] = {seed: [] for seed in GRU_SEEDS}
    with torch.no_grad():
        for step in replay_targets(
            timestamps, references.bounds, references.trial_starts, stream, offsets
        ):
            rows.append(step.trial_row)
            indices.append(step.target_index)
            latest.append(step.latest_spike_time)
            counts.append(step.counts)
            ridge.append(ridge_output(models.ridge, ridge_input(models.ridge, step.counts)))
            sequence = gru_input(normalization, step.counts, device)
            for seed in GRU_SEEDS:
                gru[seed].append(gru_output(networks[seed], normalization, sequence))
    replay_rows = np.asarray(rows, dtype=np.int64)
    replay_index = np.asarray(indices, dtype=np.int64)
    replay_time = timestamps[replay_index]
    replay_ridge = np.vstack(ridge)
    replay_gru = {seed: np.vstack(values) for seed, values in gru.items()}

    replayed_schedule, _ = replay_kalman_schedule(
        timestamps, references.bounds, references.trial_starts, stream
    )
    kalman = verify_kalman(
        models.kalman,
        replayed_schedule,
        references.schedule,
        references.bounds,
        references.initial_positions,
        timestamps,
    )
    replay_kalman = filter_official_targets(
        models.kalman,
        replayed_schedule,
        references.bounds,
        references.identity,
        initial_positions=references.initial_positions,
    ).prediction

    ordered_trials = [references.bounds[row] for row in sorted(references.bounds)]
    same_order = (
        replay_rows.size == features.n_samples
        and np.array_equal(replay_rows, features.trial_row)
        and np.array_equal(replay_index, features.target_index)
        and bool(np.all(np.diff(replay_time) > 0.0))
        and all(
            earlier.last_index < later.first_index
            for earlier, later in zip(ordered_trials, ordered_trials[1:])
        )
    )
    causal_margin = float(np.max(np.asarray(latest) - replay_time))
    ridge_difference = _max_abs(replay_ridge, references.ridge)
    gru_difference = {seed: _max_abs(replay_gru[seed], references.gru[seed]) for seed in GRU_SEEDS}
    gru_close = all(
        np.allclose(replay_gru[seed], references.gru[seed], atol=GRU_ATOL, rtol=GRU_RTOL)
        for seed in GRU_SEEDS
    )
    kalman_difference = _max_abs(replay_kalman, references.kalman)
    finite = all(
        np.all(np.isfinite(values))
        for values in (replay_ridge, replay_kalman, *replay_gru.values())
    )

    checks = [
        _check(
            "replay visits every target once, in trial and time order",
            same_order,
            f"{replay_rows.size} targets from {np.unique(replay_rows).size} trials; "
            "the buffer is emptied at every trial start",
        ),
        _check(
            "replay never reveals a spike at or after the current time",
            causal_margin < 0.0,
            f"latest revealed spike precedes its target by at least {-causal_margin * 1000:.3f} ms",
        ),
        _check(
            "replayed four-bin counts equal the batch counts",
            np.array_equal(np.stack(counts), features.counts),
            "200 ms in four 50 ms bins over [t - 250 ms, t - 50 ms) for every target",
        ),
        _check(
            "every replayed prediction is finite",
            finite,
            "Ridge, Kalman, and three GRU seeds",
        ),
        _check(
            "Ridge replay matches its batch predictions",
            ridge_difference <= RIDGE_ATOL,
            f"largest difference {ridge_difference:.1e} cm/s (tolerance {RIDGE_ATOL:g})",
        ),
        _check(
            "every GRU seed's replay matches its batch predictions",
            gru_close,
            "largest differences "
            + ", ".join(f"seed {seed} {gru_difference[seed]:.1e}" for seed in GRU_SEEDS)
            + f" cm/s (atol = rtol = {GRU_ATOL:g}); hidden state starts at zero for every target",
        ),
        _check(
            "each Kalman trial restarts from its recorded position and the trained prior",
            kalman["initial_deviation"] <= STATE_TOLERANCE and kalman["order_independent"],
            f"largest initial-state deviation {kalman['initial_deviation']:.1e}; filtering "
            "the trials in reverse order gives identical outputs",
        ),
        _check(
            "the Kalman state propagates once per 10 ms step",
            kalman["steps_complete"]
            and kalman["mean_deviation"] <= STATE_TOLERANCE
            and kalman["covariance_deviation"] <= COVARIANCE_TOLERANCE,
            f"{kalman['n_steps']} steps; largest deviation from one transition "
            f"{kalman['mean_deviation']:.1e} (state), {kalman['covariance_deviation']:.1e} "
            "(covariance)",
        ),
        _check(
            "Kalman updates use fresh, contiguous 50 ms intervals ending 50 ms earlier",
            kalman["same_schedule"] and kalman["intervals_fresh"],
            f"{replayed_schedule.n_observations} replayed intervals equal the official schedule",
        ),
        _check(
            "Kalman steps between updates are prediction-only",
            kalman["updates_as_scheduled"] and kalman["counts_as_scheduled"],
            f"{kalman['n_updates']} updates, each on its own interval, with four "
            "prediction-only steps between consecutive updates",
        ),
        _check(
            "the official Kalman path reproduces its batch output from the replayed intervals",
            kalman_difference <= RECORD_TOLERANCE,
            f"largest difference {kalman_difference:.1e} cm/s",
        ),
    ]
    details = {
        "targets": int(replay_rows.size),
        "trials": int(np.unique(replay_rows).size),
        "counts_identical_to_batch": bool(np.array_equal(np.stack(counts), features.counts)),
        "ridge": {"max_abs_difference_cm_s": ridge_difference, "atol": RIDGE_ATOL},
        "gru": {
            "atol": GRU_ATOL,
            "rtol": GRU_RTOL,
            "max_abs_difference_cm_s": {str(seed): gru_difference[seed] for seed in GRU_SEEDS},
        },
        "kalman": {
            "max_abs_difference_cm_s": kalman_difference,
            "filter_steps": kalman["n_steps"],
            "updates": kalman["n_updates"],
            "prediction_only_steps": kalman["n_steps"] - kalman["n_updates"],
            "max_initial_state_deviation": kalman["initial_deviation"],
            "max_propagation_deviation": {
                "state": kalman["mean_deviation"],
                "covariance": kalman["covariance_deviation"],
            },
        },
    }
    context = {"stream": stream, "offsets": offsets, "networks": networks, "device": device}
    return checks, details, context


def reference_checks(
    root: Path, prepared: PreparedEvaluation, references: BatchReferences
) -> list[dict[str, object]]:
    """Confirm the targets and batch predictions are the recorded official ones."""
    features, identity = references.features, references.identity
    recorded_identity = _read_json(root / KALMAN_CONFIGURATION)["final_fit"][
        "validation_identity_sha256"
    ]
    gru_targets = prepared.models.gru_configuration["targets"]
    same_targets = (
        identity.n_samples == EXPECTED_TARGETS
        and np.unique(identity.trial_row).size == EXPECTED_TRIALS
        and identity.identity_sha256 == recorded_identity
        and identity.source_sha256 == gru_targets["validation_source_sha256"]
        and np.array_equal(features.trial_row, identity.trial_row)
        and np.array_equal(features.target_index, identity.target_index)
        and np.array_equal(features.reference_index, identity.reference_index)
        and np.allclose(features.target_time, identity.target_time, atol=RECORD_TOLERANCE, rtol=0.0)
    )

    saved_ridge = references.manifest[["ridge_velocity_x", "ridge_velocity_y"]].to_numpy(
        dtype=np.float64
    )
    ridge_difference = _max_abs(references.ridge, saved_ridge)

    selected = _read_json(root / GRU_SUMMARY)["selected"]
    scores = [
        summarize_predictions(features.targets, references.gru[seed], axis_names=AXIS_NAMES)
        for seed in GRU_SEEDS
    ]
    r2_mean = np.array([score["r2_mean"] for score in scores])
    recomputed = {
        "r2_mean": float(r2_mean.mean()),
        "r2_seed_sd": float(r2_mean.std(ddof=1)),
        "r2_x_mean": float(np.mean([score["r2"]["x"] for score in scores])),
        "r2_y_mean": float(np.mean([score["r2"]["y"] for score in scores])),
    }
    gru_difference = max(abs(recomputed[key] - float(selected[key])) for key in recomputed)

    kalman_metrics = summarize_predictions(
        references.truth, references.kalman, axis_names=AXIS_NAMES
    )
    kalman_difference = _largest_difference(
        _read_json(root / KALMAN_SUMMARY)["validation_metrics"], kalman_metrics
    )

    return [
        _check(
            "the targets are the official 8,183 validation targets from 24 trials",
            same_targets,
            f"identity SHA-256 {identity.identity_sha256[:12]}... matches the Kalman "
            "validation record; the source matches the GRU configuration",
        ),
        _check(
            "the Ridge batch reference matches the saved validation predictions",
            ridge_difference <= RECORD_TOLERANCE,
            f"largest difference {ridge_difference:.1e} cm/s",
        ),
        _check(
            "the GRU batch references reproduce the recorded validation R2",
            gru_difference <= RECORD_TOLERANCE,
            f"seed-mean R2 {recomputed['r2_mean']:.6f}; largest difference "
            f"{gru_difference:.1e}",
        ),
        _check(
            "the Kalman batch reference reproduces the recorded validation metrics",
            kalman_difference <= RECORD_TOLERANCE,
            f"mean R2 {kalman_metrics['r2_mean']:.6f}; largest difference "
            f"{kalman_difference:.1e}",
        ),
    ]


def measure_latency(
    prepared: PreparedEvaluation, references: BatchReferences, context: Mapping[str, Any]
) -> dict[str, Any]:
    """Time preprocessing and inference per target for every model, after the checks."""
    session, models = prepared.session, prepared.models
    timestamps = session.behavior_timestamps
    stream, offsets = context["stream"], context["offsets"]
    device, networks = context["device"], context["networks"]
    normalization = models.gru_normalization

    def timed(prepare: Callable[[np.ndarray], Any], infer: Callable[[Any], Any]):
        def single_pass() -> tuple[np.ndarray, np.ndarray]:
            preprocessing: list[int] = []
            inference: list[int] = []
            with _paused_garbage_collection(), torch.no_grad():
                for step in replay_targets(
                    timestamps, references.bounds, references.trial_starts, stream, offsets
                ):
                    model_input = prepare(step.counts)
                    prepared_at = time.perf_counter_ns()
                    infer(model_input)
                    finished = time.perf_counter_ns()
                    preprocessing.append(prepared_at - step.started_ns)
                    inference.append(finished - prepared_at)
            return np.asarray(preprocessing), np.asarray(inference)

        single_pass()
        return single_pass()

    def kalman_pass() -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        with _paused_garbage_collection():
            _, preprocessing = replay_kalman_schedule(
                timestamps, references.bounds, references.trial_starts, stream
            )
            inference: dict[tuple[int, int], int] = {}
            for row in sorted(references.bounds):
                assessment = build_assessment_trial(
                    references.schedule,
                    references.bounds[row],
                    initial_position=references.initial_positions[row],
                )
                log: list[tuple[str, int, np.ndarray | None]] = []
                with instrumented_filter(log, capture_updates=False):
                    filter_trial(models.kalman, assessment)
                ends = np.asarray([stamp for _, stamp, _ in log], dtype=np.int64)
                for step in range(1, ends.size):
                    inference[(row, step)] = int(ends[step] - ends[step - 1])
        return preprocessing, inference

    results: dict[str, Any] = {}
    ridge_times = timed(
        lambda counts: ridge_input(models.ridge, counts),
        lambda design: ridge_output(models.ridge, design),
    )
    results["ridge"] = _latency_entry("cpu", *ridge_times)

    kalman_pass()
    step_preprocessing, step_inference = kalman_pass()
    first = {row: trial.first_index for row, trial in references.bounds.items()}
    keys = [
        (int(row), int(index) - first[int(row)])
        for row, index in zip(references.features.trial_row, references.features.target_index)
    ]
    update_steps = set(
        zip(
            references.schedule.trial_row.tolist(),
            (
                references.schedule.state_index
                - np.asarray([first[int(row)] for row in references.schedule.trial_row])
            ).tolist(),
        )
    )
    kalman_preprocessing = np.asarray([step_preprocessing[key] for key in keys])
    kalman_inference = np.asarray([step_inference[key] for key in keys])
    at_update = np.asarray([key in update_steps for key in keys])
    results["kalman"] = {
        **_latency_entry("cpu", kalman_preprocessing, kalman_inference),
        "update_steps": int(at_update.sum()),
        "total_at_update_steps_us": _distribution(
            kalman_preprocessing[at_update] + kalman_inference[at_update]
        ),
    }

    for seed in GRU_SEEDS:
        network = networks[seed]
        results[f"gru_seed{seed}"] = _latency_entry(
            device.type,
            *timed(
                lambda counts: gru_input(normalization, counts, device),
                lambda sequence, network=network: gru_output(network, normalization, sequence),
            ),
        )

    return {
        "procedure": "per target in timestamp order with trial resets, after the untimed "
        "verification replay; models loaded and every input in memory, no file I/O; "
        "time.perf_counter_ns with garbage collection paused; each model is timed on the "
        "second of two consecutive passes, the first discarded as warm-up. Preprocessing "
        "reveals the spikes that arrived since the previous 10 ms step, discards expired ones, and "
        "counts the model's window (Ridge and GRU also normalize it; the GRU copies it to "
        "the device). Inference returns the velocity in cm/s (GRU: forward pass and copy "
        "back to the host; Kalman: one 10 ms filter step, including any scheduled update).",
        "unit": "microseconds",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "gru_device": prepared.device.as_dict(),
        },
        "models": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    root = project_root()
    try:
        config = load_config()
        configure_logging(config.run.log_level)
        prepared = prepare_evaluation(root, config)
        logger.info("Rebuilding the official validation batch references")
        references = build_references(root, prepared)
        checks = reference_checks(root, prepared, references)
    except (ConfigError, DatasetError, FileNotFoundError) as error:
        print(f"Cannot prepare the replay: {error}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, to the selected recording described in README.md."
        )
        return 2
    except (
        EvaluationError,
        FeatureError,
        GruError,
        JointDecoderError,
        ModelError,
        ReplayError,
        SequenceError,
        KeyError,
        ValueError,
    ) as error:
        print(f"The batch references could not be rebuilt: {error}")
        return 1

    try:
        logger.info("Replaying %d validation targets", references.features.n_samples)
        replay_checks, replay, context = run_replay(prepared, references)
        checks.extend(replay_checks)
        passed = all(bool(check["passed"]) for check in checks)
        latency: dict[str, Any] | None = None
        if passed:
            logger.info("Measuring latency")
            latency = measure_latency(prepared, references, context)
    except (JointDecoderError, ReplayError, ValueError) as error:
        print(f"The replay failed: {error}")
        return 1

    output = root / OUTPUT_PATH
    summary: dict[str, Any] = {
        "procedure": {
            "targets": "the official validation targets; held-out trials are not built",
            "references": "batch predictions regenerated in memory from the frozen Ridge "
            "decoder, the three GRU checkpoints, and the official recursive Kalman "
            "validation path",
            "replay": "every unit's spikes merged in time order and revealed only once the "
            "10 ms behavior clock has passed them; the buffer is emptied at each trial start "
            "and holds only spikes a later window can use",
            "kalman": "the official filter runs unchanged on 50 ms intervals counted from "
            "the same replay and is instrumented to record every step and update",
        },
        "recording": {
            "name": prepared.session.identity.name,
            "sha256": prepared.session.identity.sha256,
        },
        "targets": {
            "validation_targets": references.identity.n_samples,
            "validation_trials": int(np.unique(references.identity.trial_row).size),
            "identity_sha256": references.identity.identity_sha256,
            "source": VALIDATION_SOURCE.as_posix(),
            "source_sha256": references.identity.source_sha256,
        },
        "replay": replay,
        "checks": checks,
        "all_checks_passed": passed,
        "latency": latency
        if latency is not None
        else "not measured because a check failed",
        "held_out": "not built or read",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _print_report(summary, output)
    return 0 if passed else 1


@contextmanager
def _paused_garbage_collection():
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def _latency_entry(
    device: str, preprocessing: np.ndarray, inference: np.ndarray
) -> dict[str, Any]:
    return {
        "device": device,
        "targets": int(preprocessing.size),
        "preprocessing_us": _distribution(preprocessing),
        "inference_us": _distribution(inference),
        "total_us": _distribution(preprocessing + inference),
    }


def _distribution(nanoseconds: np.ndarray) -> dict[str, float]:
    microseconds = np.asarray(nanoseconds, dtype=np.float64) / 1000.0
    return {
        "median": round(float(np.median(microseconds)), 1),
        "p95": round(float(np.percentile(microseconds, 95)), 1),
    }


def _max_abs(first: np.ndarray, second: np.ndarray) -> float:
    if np.shape(first) != np.shape(second):
        return float("inf")
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def _largest_difference(expected: Any, actual: Any) -> float:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            return float("inf")
        return max(
            (_largest_difference(expected[key], actual[key]) for key in expected),
            default=0.0,
        )
    if expected is None or actual is None:
        return 0.0 if expected is None and actual is None else float("inf")
    return abs(float(expected) - float(actual))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _print_report(summary: Mapping[str, Any], output: Path) -> None:
    checks = summary["checks"]
    width = max(len(str(check["name"])) for check in checks)
    print("Checks")
    for check in checks:
        status = "pass" if check["passed"] else "FAIL"
        print(f"  {str(check['name']):<{width}}  {status:<4}  {check['detail']}")

    latency = summary["latency"]
    if isinstance(latency, Mapping):
        print("\nLatency per target (microseconds, median / 95th percentile)")
        for name, entry in latency["models"].items():
            print(
                f"  {name:<10} {entry['device']:<5}"
                f"  preprocessing {entry['preprocessing_us']['median']:>7} / "
                f"{entry['preprocessing_us']['p95']:<7}"
                f"  inference {entry['inference_us']['median']:>7} / "
                f"{entry['inference_us']['p95']:<7}"
                f"  total {entry['total_us']['median']:>7} / {entry['total_us']['p95']}"
            )
    else:
        print(f"\nLatency {latency}")
    try:
        print(f"\nSummary: {output.resolve().relative_to(project_root()).as_posix()}")
    except ValueError:
        print(f"\nSummary: {output}")


if __name__ == "__main__":
    raise SystemExit(main())
