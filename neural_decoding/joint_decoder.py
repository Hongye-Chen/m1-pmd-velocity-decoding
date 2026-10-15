"""Clean-room primitives for joint causal decoder selection.

This module implements the approved Step 6R state, observation, scheduling, and
recursive-filtering rules.  It intentionally does not depend on any earlier
Kalman implementation.  Selection orchestration and artifact writing live in the
separate ``joint_decoder_selection`` module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular
from sklearn.covariance import oas

from neural_decoding.metrics import summarize_predictions

__all__ = [
    "CADENCE_MS",
    "INTERVAL_END_LAGS_MS",
    "NEURAL_WIDTHS_MS",
    "STATE_COMPONENTS",
    "AssessmentTrial",
    "CandidateSpec",
    "DecoderModel",
    "FilterTrace",
    "FoldAssignments",
    "JointDecoderError",
    "ObservationModel",
    "ProcessModel",
    "ScheduledObservations",
    "StateNormalization",
    "TargetLedger",
    "TrialBounds",
    "assemble_state",
    "build_assessment_trial",
    "build_official_target_ledger",
    "build_scheduled_observations",
    "build_trial_bounds",
    "candidate_grid",
    "evaluate_calibration",
    "filter_trial",
    "fit_observation_model",
    "fit_process_model",
    "fold_assignments_from_frame",
    "innovation_diagnostics",
    "select_candidate",
    "target_identity_sha256",
    "validate_oof_fill",
]

CADENCE_MS = 10
NEURAL_WIDTHS_MS: tuple[int, ...] = (10, 20, 50)
INTERVAL_END_LAGS_MS: tuple[int, ...] = (0, 50, 100)
STATE_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "V": ("vx", "vy"),
    "PV": ("px", "py", "vx", "vy"),
    "VA": ("vx", "vy", "ax", "ay"),
    "PVA": ("px", "py", "vx", "vy", "ax", "ay"),
}

_POSITION_COMPONENTS = ("px", "py")
_VELOCITY_COMPONENTS = ("vx", "vy")
_STD_FLOOR = 1e-8
_INITIAL_COVARIANCE_RIDGE = 1e-6
_PSD_TOLERANCE_FACTOR = 256.0
_TIE_TOLERANCE = 1e-12


class JointDecoderError(RuntimeError):
    """Raised when the joint decoder protocol or a numerical gate is violated."""


@dataclass(frozen=True, order=True)
class CandidateSpec:
    """One member of the frozen state/width/lag grid."""

    state_name: str
    neural_width_ms: int
    interval_end_lag_ms: int

    def __post_init__(self) -> None:
        if self.state_name not in STATE_COMPONENTS:
            raise JointDecoderError(f"Unknown state family {self.state_name!r}.")
        if self.neural_width_ms not in NEURAL_WIDTHS_MS:
            raise JointDecoderError(
                f"Width {self.neural_width_ms} is outside {NEURAL_WIDTHS_MS}."
            )
        if self.interval_end_lag_ms not in INTERVAL_END_LAGS_MS:
            raise JointDecoderError(
                f"Lag {self.interval_end_lag_ms} is outside {INTERVAL_END_LAGS_MS}."
            )

    @property
    def components(self) -> tuple[str, ...]:
        return STATE_COMPONENTS[self.state_name]

    @property
    def dimension(self) -> int:
        return len(self.components)


@dataclass(frozen=True)
class TrialBounds:
    """Inclusive behavior-grid bounds for one eligible training trial."""

    trial_row: int
    first_index: int
    last_index: int
    first_time: float
    last_time: float

    @property
    def n_steps(self) -> int:
        return self.last_index - self.first_index + 1


@dataclass(frozen=True)
class StateNormalization:
    """Fold-training normalization for one candidate state."""

    mean: np.ndarray
    scale: np.ndarray
    training_rows: tuple[int, ...]

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array - self.mean) / self.scale

    def invert(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array * self.scale + self.mean


@dataclass(frozen=True)
class ProcessModel:
    """Normalized affine process model and fold-training prior."""

    state_name: str
    components: tuple[str, ...]
    normalization: StateNormalization
    transition: np.ndarray
    offset: np.ndarray
    process_covariance: np.ndarray
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    training_rows: tuple[int, ...]
    diagnostics: dict[str, float | int | list[float]]

    @property
    def velocity_indices(self) -> tuple[int, int]:
        return tuple(self.components.index(name) for name in _VELOCITY_COMPONENTS)  # type: ignore[return-value]

    @property
    def position_indices(self) -> tuple[int, int] | None:
        if "px" not in self.components:
            return None
        return tuple(self.components.index(name) for name in _POSITION_COMPONENTS)  # type: ignore[return-value]


@dataclass(frozen=True)
class ScheduledObservations:
    """Fresh non-overlapping count vectors paired with update state rows."""

    width_ms: int
    lag_ms: int
    trial_row: np.ndarray
    interval_ordinal: np.ndarray
    interval_start_time: np.ndarray
    interval_end_time: np.ndarray
    state_index: np.ndarray
    counts: np.ndarray

    @property
    def n_observations(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.counts.shape[1])


@dataclass(frozen=True)
class ObservationModel:
    """Normalized affine neural observation model."""

    count_mean: np.ndarray
    count_scale: np.ndarray
    observation_matrix: np.ndarray
    offset: np.ndarray
    residual_covariance: np.ndarray
    residual_cholesky: np.ndarray
    residual_inverse_observation: np.ndarray
    prewhitened_observation: np.ndarray
    shrinkage: float
    training_rows: tuple[int, ...]
    diagnostics: dict[str, float | int | list[float]]

    def normalize_counts(self, counts: np.ndarray) -> np.ndarray:
        array = np.asarray(counts, dtype=np.float64)
        return (array - self.count_mean) / self.count_scale


@dataclass(frozen=True)
class DecoderModel:
    """One fold-fitted recursive decoder."""

    spec: CandidateSpec
    process: ProcessModel
    observation: ObservationModel


@dataclass(frozen=True)
class AssessmentTrial:
    """Truth-isolated inputs available to recursive filtering."""

    trial_row: int
    n_steps: int
    update_offsets: np.ndarray
    counts: np.ndarray
    initial_position: np.ndarray | None


@dataclass(frozen=True)
class FilterTrace:
    """Sequential outputs and in-memory diagnostics for one assessment trial."""

    velocity: np.ndarray
    velocity_covariance: np.ndarray
    normalized_state: np.ndarray
    posterior_covariance: np.ndarray
    update_offsets: np.ndarray
    innovations: np.ndarray
    predicted_covariance_at_updates: np.ndarray
    max_abs_gain: float


@dataclass(frozen=True)
class TargetLedger:
    """Official scoring identities and truth, kept outside the filter interface."""

    sample_index: np.ndarray
    trial_row: np.ndarray
    target_index: np.ndarray
    target_time: np.ndarray
    velocity: np.ndarray
    identity_sha256: str
    source_sha256: str

    @property
    def n_samples(self) -> int:
        return int(self.target_index.size)


@dataclass(frozen=True)
class FoldAssignments:
    """Validated tracked Step 4 fold assignments."""

    groups: tuple[tuple[int, ...], ...]
    source_sha256: str

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(row for group in self.groups for row in group)

    def assessment_rows(self, fold_index: int) -> tuple[int, ...]:
        return self.groups[fold_index]

    def training_rows(self, fold_index: int) -> tuple[int, ...]:
        return tuple(
            row for index, group in enumerate(self.groups) if index != fold_index for row in group
        )


def candidate_grid() -> tuple[CandidateSpec, ...]:
    """Return all 36 frozen candidates in deterministic order."""
    return tuple(
        CandidateSpec(state, width, lag)
        for state in STATE_COMPONENTS
        for width in NEURAL_WIDTHS_MS
        for lag in INTERVAL_END_LAGS_MS
    )


def build_trial_bounds(
    timestamps: np.ndarray,
    trials: pd.DataFrame,
    trial_rows: Iterable[int],
    *,
    required_split: str = "train",
) -> dict[int, TrialBounds]:
    """Map requested eligible rows from one declared split to behavior-grid bounds."""
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise JointDecoderError("Behavior timestamps must be a finite one-dimensional grid.")
    steps = np.diff(times)
    if np.any(steps <= 0.0) or not np.allclose(
        steps, CADENCE_MS / 1000.0, atol=1e-9, rtol=0.0
    ):
        raise JointDecoderError("Step 6R requires a uniform 10 ms behavior grid.")

    requested = tuple(sorted({int(row) for row in trial_rows}))
    table = trials.set_index("trial_row", drop=False)
    bounds: dict[int, TrialBounds] = {}
    for row in requested:
        if row not in table.index:
            raise JointDecoderError(f"Trial row {row} is absent from the trial table.")
        record = table.loc[row]
        if str(record["split"]) != required_split or not bool(record["eligible"]):
            raise JointDecoderError(
                f"Trial row {row} is not eligible {required_split} data."
            )
        first = int(np.searchsorted(times, float(record["start_time"]), side="left"))
        last = int(np.searchsorted(times, float(record["stop_time"]), side="right")) - 1
        if first < 0 or last >= times.size or last < first:
            raise JointDecoderError(f"Trial row {row} has invalid behavior-grid bounds.")
        bounds[row] = TrialBounds(
            trial_row=row,
            first_index=first,
            last_index=last,
            first_time=float(times[first]),
            last_time=float(times[last]),
        )
    if len(bounds) != len(requested):
        raise JointDecoderError("Trial bounds are incomplete.")
    return bounds


def assemble_state(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    state_name: str,
) -> np.ndarray:
    """Assemble one frozen state family from aligned recorded behavior."""
    if state_name not in STATE_COMPONENTS:
        raise JointDecoderError(f"Unknown state family {state_name!r}.")
    arrays = {
        "p": np.asarray(position, dtype=np.float64),
        "v": np.asarray(velocity, dtype=np.float64),
        "a": np.asarray(acceleration, dtype=np.float64),
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes))[1:] != (2,):
        raise JointDecoderError("Position, velocity, and acceleration must align as (n, 2).")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise JointDecoderError("State behavior contains non-finite values.")
    pieces = {
        "V": (arrays["v"],),
        "PV": (arrays["p"], arrays["v"]),
        "VA": (arrays["v"], arrays["a"]),
        "PVA": (arrays["p"], arrays["v"], arrays["a"]),
    }
    return np.column_stack(pieces[state_name])


def fit_process_model(
    state: np.ndarray,
    bounds: Mapping[int, TrialBounds],
    training_rows: Iterable[int],
    state_name: str,
) -> ProcessModel:
    """Fit normalization, affine transition, PSD Q, and initial prior."""
    rows = tuple(sorted({int(row) for row in training_rows}))
    if not rows:
        raise JointDecoderError("A process fit needs at least one training trial.")
    missing = set(rows) - set(bounds)
    if missing:
        raise JointDecoderError(f"Missing trial bounds for rows {sorted(missing)}.")

    raw_state = np.asarray(state, dtype=np.float64)
    state_rows = np.concatenate(
        [np.arange(bounds[row].first_index, bounds[row].last_index + 1) for row in rows]
    )
    fitted = raw_state[state_rows]
    mean = fitted.mean(axis=0)
    scale = fitted.std(axis=0)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)):
        raise JointDecoderError("State normalization is non-finite.")
    if np.any(scale < _STD_FLOOR):
        raise JointDecoderError("A state component has no usable fold-training variance.")
    normalization = StateNormalization(mean=mean, scale=scale, training_rows=rows)
    normalized = normalization.apply(raw_state)

    previous_chunks: list[np.ndarray] = []
    next_chunks: list[np.ndarray] = []
    residual_rows: list[np.ndarray] = []
    first_rows: list[np.ndarray] = []
    for row in rows:
        trial = bounds[row]
        if trial.n_steps < 2:
            raise JointDecoderError(f"Trial row {row} has no within-trial transition.")
        indices = np.arange(trial.first_index, trial.last_index + 1)
        previous_chunks.append(normalized[indices[:-1]])
        next_chunks.append(normalized[indices[1:]])
        residual_rows.append(np.full(indices.size - 1, row, dtype=np.int64))
        first_rows.append(normalized[trial.first_index])

    previous = np.vstack(previous_chunks)
    following = np.vstack(next_chunks)
    design = np.column_stack([previous, np.ones(previous.shape[0])])
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, following, rcond=None)
    transition = coefficients[:-1].T
    offset = coefficients[-1]
    residuals = following - (previous @ transition.T + offset)
    process_covariance = _project_roundoff_psd(
        residuals.T @ residuals / max(1, residuals.shape[0] - 1),
        "process covariance",
    )

    starts = np.vstack(first_rows)
    initial_mean = starts.mean(axis=0)
    initial_covariance = _sample_covariance(starts)
    ridge_scale = max(float(np.trace(initial_covariance)) / starts.shape[1], 1.0)
    initial_covariance = _project_roundoff_psd(
        initial_covariance
        + np.eye(starts.shape[1]) * (_INITIAL_COVARIANCE_RIDGE * ridge_scale),
        "initial covariance",
    )

    autocorrelation = _within_trial_autocorrelation(
        residuals, np.concatenate(residual_rows), max_lag=10
    )
    eigenvalues = np.linalg.eigvalsh(process_covariance)
    diagnostics: dict[str, float | int | list[float]] = {
        "transition_design_rank": int(rank),
        "transition_design_condition": _condition_from_singular_values(singular_values),
        "transition_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(transition)))
        ),
        "process_covariance_rank": int(np.linalg.matrix_rank(process_covariance)),
        "process_covariance_min_eigenvalue": float(eigenvalues.min()),
        "process_covariance_max_eigenvalue": float(eigenvalues.max()),
        "process_residual_autocorrelation_abs_max": _finite_abs_max(autocorrelation),
        "process_residual_autocorrelation_by_component": [
            _finite_abs_max(autocorrelation[:, component])
            for component in range(autocorrelation.shape[1])
        ],
    }
    return ProcessModel(
        state_name=state_name,
        components=STATE_COMPONENTS[state_name],
        normalization=normalization,
        transition=transition,
        offset=offset,
        process_covariance=process_covariance,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        training_rows=rows,
        diagnostics=diagnostics,
    )


def build_scheduled_observations(
    timestamps: np.ndarray,
    spike_times: Sequence[np.ndarray],
    bounds: Mapping[int, TrialBounds],
    *,
    width_ms: int,
    lag_ms: int,
) -> ScheduledObservations:
    """Count one fresh half-open neural interval for every scheduled update."""
    if width_ms not in NEURAL_WIDTHS_MS or lag_ms not in INTERVAL_END_LAGS_MS:
        raise JointDecoderError("Observation schedule is outside the frozen grid.")
    times = np.asarray(timestamps, dtype=np.float64)
    prepared_spikes = [np.asarray(spikes, dtype=np.float64).ravel() for spikes in spike_times]
    if not prepared_spikes:
        raise JointDecoderError("At least one neural unit is required.")
    for position, spikes in enumerate(prepared_spikes):
        if np.any(~np.isfinite(spikes)) or np.any(np.diff(spikes) < 0.0):
            raise JointDecoderError(f"Spike train {position} is non-finite or unsorted.")

    width_steps = width_ms // CADENCE_MS
    lag_steps = lag_ms // CADENCE_MS
    row_chunks: list[np.ndarray] = []
    ordinal_chunks: list[np.ndarray] = []
    start_chunks: list[np.ndarray] = []
    end_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []

    for row in sorted(bounds):
        trial = bounds[row]
        starts = np.arange(0, trial.n_steps, width_steps, dtype=np.int64)
        right = starts + width_steps
        associated = right + lag_steps
        keep = associated < trial.n_steps
        starts = starts[keep]
        right = right[keep]
        associated = associated[keep]
        if starts.size == 0:
            continue
        left_times = times[trial.first_index + starts]
        right_times = times[trial.first_index + right]
        counts = np.empty((starts.size, len(prepared_spikes)), dtype=np.uint16)
        for unit, spikes in enumerate(prepared_spikes):
            left_positions = np.searchsorted(spikes, left_times, side="left")
            right_positions = np.searchsorted(spikes, right_times, side="left")
            values = right_positions - left_positions
            if values.size and int(values.max()) > np.iinfo(np.uint16).max:
                raise JointDecoderError("A scheduled count exceeds uint16 capacity.")
            counts[:, unit] = values

        row_chunks.append(np.full(starts.size, row, dtype=np.int64))
        ordinal_chunks.append(np.arange(starts.size, dtype=np.int64))
        start_chunks.append(left_times)
        end_chunks.append(right_times)
        state_chunks.append(trial.first_index + associated)
        count_chunks.append(counts)

    if not count_chunks:
        raise JointDecoderError("The schedule produced no neural observations.")
    scheduled = ScheduledObservations(
        width_ms=width_ms,
        lag_ms=lag_ms,
        trial_row=np.concatenate(row_chunks),
        interval_ordinal=np.concatenate(ordinal_chunks),
        interval_start_time=np.concatenate(start_chunks),
        interval_end_time=np.concatenate(end_chunks),
        state_index=np.concatenate(state_chunks),
        counts=np.vstack(count_chunks),
    )
    _validate_schedule(scheduled, bounds)
    return scheduled


def fit_observation_model(
    schedule: ScheduledObservations,
    normalized_state: np.ndarray,
    training_rows: Iterable[int],
) -> ObservationModel:
    """Fit fold-specific count normalization, affine H/d, and OAS full R."""
    rows = tuple(sorted({int(row) for row in training_rows}))
    mask = np.isin(schedule.trial_row, np.asarray(rows, dtype=np.int64))
    if not np.any(mask):
        raise JointDecoderError("An observation fit needs scheduled training updates.")
    counts = schedule.counts[mask].astype(np.float64)
    count_mean = counts.mean(axis=0)
    count_scale = counts.std(axis=0)
    count_scale = np.where(count_scale < _STD_FLOOR, 1.0, count_scale)
    observations = (counts - count_mean) / count_scale

    states = np.asarray(normalized_state, dtype=np.float64)[schedule.state_index[mask]]
    design = np.column_stack([states, np.ones(states.shape[0])])
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, observations, rcond=None
    )
    observation_matrix = coefficients[:-1].T
    offset = coefficients[-1]
    residuals = observations - (states @ observation_matrix.T + offset)

    residual_covariance, shrinkage = oas(residuals, assume_centered=True)
    residual_covariance = (residual_covariance + residual_covariance.T) * 0.5
    if not np.all(np.isfinite(residual_covariance)):
        raise JointDecoderError("Observation covariance is non-finite.")
    try:
        residual_cholesky = np.linalg.cholesky(residual_covariance)
    except np.linalg.LinAlgError as error:
        raise JointDecoderError("Analytically shrunk observation covariance is not PD.") from error
    residual_inverse_observation = cho_solve(
        (residual_cholesky, True), observation_matrix, check_finite=False
    )
    prewhitened_observation = solve_triangular(
        residual_cholesky,
        observation_matrix,
        lower=True,
        check_finite=False,
    )

    residual_trial_rows = schedule.trial_row[mask]
    autocorrelation = _within_trial_autocorrelation(
        residuals, residual_trial_rows, max_lag=10
    )
    eigenvalues = np.linalg.eigvalsh(residual_covariance)
    diagnostics: dict[str, float | int | list[float]] = {
        "observation_design_rank": int(rank),
        "observation_design_condition": _condition_from_singular_values(singular_values),
        "observation_covariance_rank": int(np.linalg.matrix_rank(residual_covariance)),
        "observation_covariance_min_eigenvalue": float(eigenvalues.min()),
        "observation_covariance_max_eigenvalue": float(eigenvalues.max()),
        "observation_covariance_condition": float(eigenvalues.max() / eigenvalues.min()),
        "observation_covariance_shrinkage": float(shrinkage),
        "observation_residual_autocorrelation_abs_max": _finite_abs_max(autocorrelation),
        "count_features_at_scale_floor": int(np.count_nonzero(counts.std(axis=0) < _STD_FLOOR)),
        "n_training_observations": int(counts.shape[0]),
    }
    return ObservationModel(
        count_mean=count_mean,
        count_scale=count_scale,
        observation_matrix=observation_matrix,
        offset=offset,
        residual_covariance=residual_covariance,
        residual_cholesky=residual_cholesky,
        residual_inverse_observation=residual_inverse_observation,
        prewhitened_observation=prewhitened_observation,
        shrinkage=float(shrinkage),
        training_rows=rows,
        diagnostics=diagnostics,
    )


def build_assessment_trial(
    schedule: ScheduledObservations,
    bounds: TrialBounds,
    *,
    initial_position: np.ndarray | None,
) -> AssessmentTrial:
    """Build the only data structure accepted by recursive filtering."""
    mask = schedule.trial_row == bounds.trial_row
    offsets = schedule.state_index[mask] - bounds.first_index
    if np.any(offsets <= 0) or np.any(offsets >= bounds.n_steps):
        raise JointDecoderError("An assessment update lies outside its trial.")
    if np.unique(offsets).size != offsets.size:
        raise JointDecoderError("More than one neural interval is assigned to a state step.")
    position = (
        None
        if initial_position is None
        else np.asarray(initial_position, dtype=np.float64).reshape(2).copy()
    )
    if position is not None and not np.all(np.isfinite(position)):
        raise JointDecoderError("Assessment starting position is non-finite.")
    return AssessmentTrial(
        trial_row=bounds.trial_row,
        n_steps=bounds.n_steps,
        update_offsets=offsets.astype(np.int64),
        counts=schedule.counts[mask].copy(),
        initial_position=position,
    )


def filter_trial(model: DecoderModel, trial: AssessmentTrial) -> FilterTrace:
    """Run one assessment trial recursively from spikes and the previous posterior."""
    process = model.process
    observation = model.observation
    dimension = len(process.components)
    if trial.counts.shape != (trial.update_offsets.size, observation.count_mean.size):
        raise JointDecoderError("Assessment counts do not match their update schedule.")

    mean = process.initial_mean.copy()
    covariance = process.initial_covariance.copy()
    position_indices = process.position_indices
    if position_indices is None:
        if trial.initial_position is not None:
            raise JointDecoderError("A position-free state must not receive assessment position.")
    else:
        if trial.initial_position is None:
            raise JointDecoderError("A position state requires exactly one starting position.")
        normalized_position = (
            trial.initial_position - process.normalization.mean[list(position_indices)]
        ) / process.normalization.scale[list(position_indices)]
        mean[list(position_indices)] = normalized_position
        unobserved = [index for index in range(dimension) if index not in position_indices]
        conditioned_covariance = np.zeros_like(covariance)
        conditioned_covariance[np.ix_(unobserved, unobserved)] = covariance[
            np.ix_(unobserved, unobserved)
        ]
        covariance = conditioned_covariance

    velocity = np.empty((trial.n_steps, 2), dtype=np.float64)
    velocity_covariance = np.empty((trial.n_steps, 2, 2), dtype=np.float64)
    state_trace = np.empty((trial.n_steps, dimension), dtype=np.float64)
    covariance_trace = np.empty((trial.n_steps, dimension, dimension), dtype=np.float64)
    innovations: list[np.ndarray] = []
    predicted_at_updates: list[np.ndarray] = []
    max_abs_gain = 0.0

    update_by_offset = {
        int(offset): index for index, offset in enumerate(trial.update_offsets.tolist())
    }
    for step in range(trial.n_steps):
        if step:
            mean = process.transition @ mean + process.offset
            covariance = (
                process.transition @ covariance @ process.transition.T
                + process.process_covariance
            )
            covariance = _assert_psd(covariance, "predicted covariance")

        update_index = update_by_offset.get(step)
        if update_index is not None:
            normalized_count = observation.normalize_counts(trial.counts[update_index])
            innovation = normalized_count - (
                observation.observation_matrix @ mean + observation.offset
            )
            predicted_at_updates.append(covariance.copy())
            mean, covariance, gain_max = _measurement_update(
                mean, covariance, innovation, observation
            )
            innovations.append(innovation)
            max_abs_gain = max(max_abs_gain, gain_max)

        covariance = _assert_psd(covariance, "posterior covariance")
        state_trace[step] = mean
        covariance_trace[step] = covariance
        velocity[step], velocity_covariance[step] = _physical_velocity(
            mean, covariance, process
        )

    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(covariance_trace)):
        raise JointDecoderError("Recursive filtering produced non-finite outputs.")
    return FilterTrace(
        velocity=velocity,
        velocity_covariance=velocity_covariance,
        normalized_state=state_trace,
        posterior_covariance=covariance_trace,
        update_offsets=trial.update_offsets.copy(),
        innovations=(
            np.vstack(innovations)
            if innovations
            else np.empty((0, observation.count_mean.size), dtype=np.float64)
        ),
        predicted_covariance_at_updates=(
            np.stack(predicted_at_updates)
            if predicted_at_updates
            else np.empty((0, dimension, dimension), dtype=np.float64)
        ),
        max_abs_gain=max_abs_gain,
    )


def innovation_diagnostics(
    model: DecoderModel,
    innovations: np.ndarray,
    predicted_covariances: np.ndarray,
    trial_rows: np.ndarray,
) -> dict[str, float]:
    """Return exact low-rank-whitened innovation summaries within trials."""
    residual = np.asarray(innovations, dtype=np.float64)
    predicted = np.asarray(predicted_covariances, dtype=np.float64)
    rows = np.asarray(trial_rows, dtype=np.int64)
    if residual.shape[0] != predicted.shape[0] or rows.size != residual.shape[0]:
        raise JointDecoderError("Innovation diagnostics are not row-aligned.")
    if residual.shape[0] == 0:
        raise JointDecoderError("Innovation diagnostics require at least one update.")

    observation = model.observation
    base = solve_triangular(
        observation.residual_cholesky,
        residual.T,
        lower=True,
        check_finite=False,
    ).T
    whitened = np.empty_like(base)
    for index, covariance in enumerate(predicted):
        factor = _psd_factor(covariance)
        if factor.shape[1] == 0:
            whitened[index] = base[index]
            continue
        low_rank = observation.prewhitened_observation @ factor
        left, singular, _ = np.linalg.svd(low_rank, full_matrices=False)
        projection = left.T @ base[index]
        adjustment = (1.0 / np.sqrt(1.0 + np.square(singular))) - 1.0
        whitened[index] = base[index] + left @ (adjustment * projection)

    autocorrelation = _within_trial_autocorrelation(whitened, rows, max_lag=10)
    squared_norm = np.square(whitened).sum(axis=1)
    return {
        "innovation_whitened_autocorrelation_abs_max": _finite_abs_max(autocorrelation),
        "innovation_whitened_squared_norm_mean": float(squared_norm.mean()),
        "innovation_whitened_absolute_max": float(np.max(np.abs(whitened))),
        "innovation_absolute_max": float(np.max(np.abs(residual))),
    }


def evaluate_calibration(
    truth: np.ndarray,
    prediction: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    """Summarize velocity uncertainty, regularized NEES, and 95% coverage."""
    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    covariances = np.asarray(covariance, dtype=np.float64)
    if actual.shape != predicted.shape or covariances.shape != (actual.shape[0], 2, 2):
        raise JointDecoderError("Calibration arrays are not aligned.")
    errors = actual - predicted
    nees = np.empty(actual.shape[0], dtype=np.float64)
    posterior_variance = np.empty((actual.shape[0], 2), dtype=np.float64)
    for index, matrix in enumerate(covariances):
        symmetric = _assert_psd(matrix, "velocity posterior covariance")
        posterior_variance[index] = np.diag(symmetric)
        scale = max(float(np.trace(symmetric)) / 2.0, 1.0)
        diagnostic = symmetric + np.eye(2) * (np.finfo(float).eps * scale * 64.0)
        nees[index] = float(errors[index] @ np.linalg.solve(diagnostic, errors[index]))
    empirical_mse = np.square(errors).mean(axis=0)
    mean_variance = posterior_variance.mean(axis=0)
    ratio = mean_variance / np.maximum(empirical_mse, np.finfo(float).tiny)
    return {
        "velocity_nees_mean": float(nees.mean()),
        "velocity_nees_median": float(np.median(nees)),
        "velocity_95_ellipse_coverage": float(np.mean(nees <= 5.991464547107979)),
        "posterior_to_empirical_variance_ratio_x": float(ratio[0]),
        "posterior_to_empirical_variance_ratio_y": float(ratio[1]),
        "velocity_posterior_std_x": float(np.sqrt(mean_variance[0])),
        "velocity_posterior_std_y": float(np.sqrt(mean_variance[1])),
    }


def build_official_target_ledger(
    manifest: pd.DataFrame | None,
    *,
    timestamps: np.ndarray,
    velocity: np.ndarray,
    bounds: Mapping[int, TrialBounds],
    source_sha256: str,
) -> TargetLedger:
    """Reconstruct Step 5 identities and optionally verify an identity manifest."""
    reconstructed_index: list[np.ndarray] = []
    reconstructed_rows: list[np.ndarray] = []
    warmup_steps = (200 + 50) // CADENCE_MS
    for row in sorted(bounds):
        trial = bounds[row]
        first = trial.first_index + warmup_steps
        if first <= trial.last_index:
            indices = np.arange(first, trial.last_index + 1, dtype=np.int64)
            reconstructed_index.append(indices)
            reconstructed_rows.append(np.full(indices.size, row, dtype=np.int64))
    expected_index = np.concatenate(reconstructed_index)
    expected_rows = np.concatenate(reconstructed_rows)
    times = np.asarray(timestamps, dtype=np.float64)
    expected_time = times[expected_index]
    if manifest is None:
        sample_index = np.arange(expected_index.size, dtype=np.int64)
        trial_row = expected_rows
        target_index = expected_index
        target_time = expected_time
    else:
        required = ("sample_index", "trial_row", "target_index", "target_time")
        missing = [name for name in required if name not in manifest.columns]
        if missing:
            raise JointDecoderError(f"Official target manifest is missing {missing}.")
        frame = manifest.loc[:, required].copy()
        sample_index = frame["sample_index"].to_numpy(dtype=np.int64)
        trial_row = frame["trial_row"].to_numpy(dtype=np.int64)
        target_index = frame["target_index"].to_numpy(dtype=np.int64)
        target_time = frame["target_time"].to_numpy(dtype=np.float64)
        if not np.array_equal(
            sample_index, np.arange(sample_index.size, dtype=np.int64)
        ):
            raise JointDecoderError("Official target sample indices are not canonical.")
        if (
            not np.array_equal(target_index, expected_index)
            or not np.array_equal(trial_row, expected_rows)
            or not np.allclose(target_time, expected_time, atol=1e-12, rtol=0.0)
        ):
            raise JointDecoderError(
                "Step 5 target identities differ from independent 250 ms "
                "warm-up reconstruction."
            )
    if target_index.size != 43_022 or np.unique(trial_row).size != 123:
        raise JointDecoderError(
            f"Expected 43,022 targets from 123 trials, got {target_index.size} "
            f"from {np.unique(trial_row).size}."
        )
    truth = np.asarray(velocity, dtype=np.float64)[target_index].copy()
    if truth.shape != (43_022, 2) or not np.all(np.isfinite(truth)):
        raise JointDecoderError("Official training truth has an invalid shape or values.")
    identity_hash = target_identity_sha256(trial_row, target_index, target_time)
    return TargetLedger(
        sample_index=sample_index,
        trial_row=trial_row,
        target_index=target_index,
        target_time=target_time,
        velocity=truth,
        identity_sha256=identity_hash,
        source_sha256=source_sha256,
    )


def target_identity_sha256(
    trial_row: np.ndarray, target_index: np.ndarray, target_time: np.ndarray
) -> str:
    """Hash canonical target identity arrays with explicit little-endian encodings."""
    digest = hashlib.sha256(b"step5-training-target-identity-v1\0")
    for array, dtype in (
        (trial_row, "<i8"),
        (target_index, "<i8"),
        (target_time, "<f8"),
    ):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        digest.update(np.asarray([canonical.size], dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def fold_assignments_from_frame(
    frame: pd.DataFrame,
    *,
    eligible_training_rows: Iterable[int],
    source_sha256: str,
) -> FoldAssignments:
    """Validate, but never regenerate, the tracked five fold groups."""
    required = ("trial_row", "group_index", "assessed_in_fold")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise JointDecoderError(f"Fold assignments are missing columns {missing}.")
    table = frame.loc[:, required].copy()
    if table["trial_row"].duplicated().any():
        raise JointDecoderError("A training trial appears more than once in the folds.")
    if not np.array_equal(
        table["group_index"].to_numpy(dtype=np.int64),
        table["assessed_in_fold"].to_numpy(dtype=np.int64),
    ):
        raise JointDecoderError("Fold group and assessment indices disagree.")
    expected = {int(row) for row in eligible_training_rows}
    present = {int(row) for row in table["trial_row"]}
    if present != expected or len(present) != 123:
        raise JointDecoderError("Tracked folds do not contain exactly the eligible training rows.")
    groups = tuple(
        tuple(
            sorted(
                table.loc[table["group_index"] == index, "trial_row"]
                .to_numpy(dtype=np.int64)
                .tolist()
            )
        )
        for index in range(5)
    )
    if any(not group for group in groups) or max(map(len, groups)) - min(map(len, groups)) > 1:
        raise JointDecoderError("Tracked folds are not five complete near-equal groups.")
    return FoldAssignments(groups=groups, source_sha256=source_sha256)


def validate_oof_fill(predictions: np.ndarray, fill_count: np.ndarray) -> None:
    """Require one finite out-of-fold prediction for every official target."""
    values = np.asarray(predictions, dtype=np.float64)
    counts = np.asarray(fill_count, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or counts.shape != (values.shape[0],):
        raise JointDecoderError("Out-of-fold arrays have invalid shapes.")
    if not np.all(counts == 1):
        raise JointDecoderError(
            f"Every OOF row must be filled once; counts range {counts.min()}-{counts.max()}."
        )
    if not np.all(np.isfinite(values)):
        raise JointDecoderError("Out-of-fold predictions are incomplete or non-finite.")


def select_candidate(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Apply the frozen score and deterministic tie-breaking rule."""
    passing = [
        record
        for record in records
        if bool(record.get("passed", False))
        and np.isfinite(float(record.get("out_of_fold_r2_mean", np.nan)))
    ]
    if not passing:
        raise JointDecoderError("No candidate passed the numerical and integrity gates.")
    best_score = max(float(record["out_of_fold_r2_mean"]) for record in passing)
    tied = [
        record
        for record in passing
        if best_score - float(record["out_of_fold_r2_mean"]) <= _TIE_TOLERANCE
    ]
    return min(
        tied,
        key=lambda record: (
            int(record["state_dimension"]),
            int(record["neural_width_ms"]),
            int(record["interval_end_lag_ms"]),
            str(record["state_name"]),
        ),
    )


def _validate_schedule(
    schedule: ScheduledObservations, bounds: Mapping[int, TrialBounds]
) -> None:
    keys = list(zip(schedule.trial_row.tolist(), schedule.interval_ordinal.tolist()))
    if len(keys) != len(set(keys)):
        raise JointDecoderError("A neural interval appears more than once.")
    for row in np.unique(schedule.trial_row):
        trial = bounds[int(row)]
        mask = schedule.trial_row == row
        starts = schedule.interval_start_time[mask]
        ends = schedule.interval_end_time[mask]
        if (
            np.any(starts < trial.first_time - 1e-12)
            or np.any(ends > trial.last_time + 1e-12)
            or np.any(ends <= starts)
        ):
            raise JointDecoderError(f"A neural interval crosses trial row {row}.")
        if starts.size > 1 and np.any(starts[1:] < ends[:-1] - 1e-12):
            raise JointDecoderError(f"Neural intervals overlap in trial row {row}.")
    if np.any(schedule.counts < 0):
        raise JointDecoderError("Neural counts must be non-negative.")


def _measurement_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    innovation: np.ndarray,
    observation: ObservationModel,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stable low-rank information update, including singular PSD priors."""
    factor = _psd_factor(covariance)
    if factor.shape[1] == 0:
        return mean, covariance, 0.0
    gram = (
        observation.observation_matrix.T @ observation.residual_inverse_observation
    )
    small = np.eye(factor.shape[1]) + factor.T @ gram @ factor
    try:
        small_cholesky = np.linalg.cholesky((small + small.T) * 0.5)
    except np.linalg.LinAlgError as error:
        raise JointDecoderError("The reduced innovation covariance is not PD.") from error
    posterior = factor @ cho_solve(
        (small_cholesky, True), factor.T, check_finite=False
    )
    posterior = _assert_psd(posterior, "updated covariance")
    gain = posterior @ observation.residual_inverse_observation.T
    updated_mean = mean + gain @ innovation
    if not np.all(np.isfinite(updated_mean)):
        raise JointDecoderError("The measurement update produced a non-finite state.")
    return updated_mean, posterior, float(np.max(np.abs(gain)))


def _physical_velocity(
    mean: np.ndarray, covariance: np.ndarray, process: ProcessModel
) -> tuple[np.ndarray, np.ndarray]:
    velocity_indices = list(process.velocity_indices)
    physical_state = process.normalization.invert(mean)
    scales = process.normalization.scale[velocity_indices]
    physical_covariance = covariance[np.ix_(velocity_indices, velocity_indices)] * np.outer(
        scales, scales
    )
    return physical_state[velocity_indices], (physical_covariance + physical_covariance.T) * 0.5


def _sample_covariance(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise JointDecoderError("Covariance estimation needs at least two rows.")
    centred = array - array.mean(axis=0)
    return (centred.T @ centred) / (array.shape[0] - 1)


def _project_roundoff_psd(matrix: np.ndarray, label: str) -> np.ndarray:
    symmetric = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = np.finfo(float).eps * max(symmetric.shape) * scale * _PSD_TOLERANCE_FACTOR
    if float(eigenvalues.min()) < -tolerance:
        raise JointDecoderError(
            f"{label} has a substantive negative eigenvalue {eigenvalues.min():.3e}."
        )
    clipped = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    projected = (eigenvectors * clipped) @ eigenvectors.T
    return (projected + projected.T) * 0.5


def _assert_psd(matrix: np.ndarray, label: str) -> np.ndarray:
    return _project_roundoff_psd(matrix, label)


def _psd_factor(matrix: np.ndarray) -> np.ndarray:
    symmetric = _assert_psd(matrix, "covariance factor")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(eigenvalues.max()), 1.0)
    tolerance = np.finfo(float).eps * max(symmetric.shape) * scale * _PSD_TOLERANCE_FACTOR
    keep = eigenvalues > tolerance
    return eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])


def _within_trial_autocorrelation(
    values: np.ndarray, trial_rows: np.ndarray, *, max_lag: int
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    rows = np.asarray(trial_rows, dtype=np.int64)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] != rows.size:
        raise JointDecoderError("Autocorrelation values and trial rows are not aligned.")
    result = np.full((max_lag, array.shape[1]), np.nan, dtype=np.float64)
    centred = array - array.mean(axis=0)
    variance = np.square(centred).sum(axis=0)
    for lag in range(1, max_lag + 1):
        products: list[np.ndarray] = []
        for row in np.unique(rows):
            trial_values = centred[rows == row]
            if trial_values.shape[0] > lag:
                products.append(trial_values[:-lag] * trial_values[lag:])
        if products:
            numerator = np.vstack(products).sum(axis=0)
            result[lag - 1] = np.divide(
                numerator,
                variance,
                out=np.full(array.shape[1], np.nan),
                where=variance > 0.0,
            )
    return result


def _finite_abs_max(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.abs(array[np.isfinite(array)])
    return float(finite.max()) if finite.size else 0.0


def _condition_from_singular_values(singular_values: np.ndarray) -> float:
    singular = np.asarray(singular_values, dtype=np.float64)
    if singular.size == 0 or singular[-1] <= 0.0:
        return float("inf")
    return float(singular[0] / singular[-1])
