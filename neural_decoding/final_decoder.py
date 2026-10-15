"""Final-fit and serialization support for the frozen clean-room decoder.

The functions here do not select or tune a configuration.  They accept only the
Step 6R winner (PVA, 50 ms width, 50 ms interval-end lag), fit it on declared
training trial bounds, and expose a truth-isolated filtering interface for one
official validation run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular

from neural_decoding.joint_decoder import (
    CandidateSpec,
    DecoderModel,
    JointDecoderError,
    ObservationModel,
    ProcessModel,
    ScheduledObservations,
    StateNormalization,
    TrialBounds,
    build_assessment_trial,
    build_scheduled_observations,
    filter_trial,
    fit_observation_model,
    fit_process_model,
    validate_oof_fill,
)

if TYPE_CHECKING:
    from neural_decoding.dataset import SessionData

__all__ = [
    "FROZEN_SPEC",
    "FinalFilterResult",
    "ValidationIdentity",
    "ValidationScoringLedger",
    "build_validation_ledgers",
    "decoder_from_dict",
    "decoder_to_dict",
    "filter_official_targets",
    "fit_frozen_decoder",
]

FROZEN_SPEC = CandidateSpec("PVA", 50, 50)
_MODEL_SCHEMA_VERSION = 1
_EXPECTED_UNITS = 53
_EXPECTED_TRAINING_TRIALS = 123
_EXPECTED_VALIDATION_TRIALS = 24
_EXPECTED_VALIDATION_TARGETS = 8_183
_CADENCE_MS = 10
_HISTORY_MS = 200
_ELIGIBILITY_LAG_MS = 50
_CSV_ROUND_TRIP_ATOL = 1e-12


@dataclass(frozen=True)
class ValidationIdentity:
    """Official CA-05 validation identities without behavior truth."""

    sample_index: np.ndarray
    trial_row: np.ndarray
    reference_index: np.ndarray
    target_index: np.ndarray
    target_time: np.ndarray
    identity_sha256: str
    source_sha256: str

    @property
    def n_samples(self) -> int:
        return int(self.target_index.size)


@dataclass(frozen=True)
class ValidationScoringLedger:
    """Validation truth kept separate from recursive filtering."""

    velocity: np.ndarray
    truth_sha256: str


@dataclass(frozen=True)
class FinalFilterResult:
    """Finalized predictions and in-memory numerical diagnostics."""

    prediction: np.ndarray
    velocity_covariance: np.ndarray
    fill_count: np.ndarray
    innovations: np.ndarray
    predicted_covariance_at_updates: np.ndarray
    innovation_trial_row: np.ndarray
    max_abs_gain: float
    max_abs_normalized_state: float
    max_abs_posterior_covariance: float
    posterior_min_eigenvalue: float
    posterior_max_eigenvalue: float
    posterior_min_rank: int


def fit_frozen_decoder(
    session: SessionData,
    training_bounds: Mapping[int, TrialBounds],
) -> tuple[DecoderModel, ScheduledObservations]:
    """Fit the frozen PVA/50/50 decoder on all eligible training trials."""
    rows = tuple(sorted(int(row) for row in training_bounds))
    if len(rows) != _EXPECTED_TRAINING_TRIALS:
        raise JointDecoderError(
            f"Final fitting requires 123 training trials, got {len(rows)}."
        )
    if len(session.units) != _EXPECTED_UNITS or len(session.spike_times) != _EXPECTED_UNITS:
        raise JointDecoderError("Final fitting requires the frozen 53-unit order.")
    state = _assemble_training_state(session, training_bounds)
    process = fit_process_model(state, training_bounds, rows, FROZEN_SPEC.state_name)
    schedule = build_scheduled_observations(
        session.behavior_timestamps,
        session.spike_times,
        training_bounds,
        width_ms=FROZEN_SPEC.neural_width_ms,
        lag_ms=FROZEN_SPEC.interval_end_lag_ms,
    )
    observation = fit_observation_model(
        schedule,
        process.normalization.apply(state),
        rows,
    )
    if process.training_rows != rows or observation.training_rows != rows:
        raise JointDecoderError("A final fitted component omitted a training trial.")
    return (
        DecoderModel(spec=FROZEN_SPEC, process=process, observation=observation),
        schedule,
    )


def build_validation_ledgers(
    manifest: pd.DataFrame,
    *,
    timestamps: np.ndarray,
    velocity: np.ndarray,
    bounds: Mapping[int, TrialBounds],
    source_sha256: str,
) -> tuple[ValidationIdentity, ValidationScoringLedger]:
    """Verify exact CA-05 validation identities and truth against reconstruction."""
    required = (
        "sample_index",
        "trial_row",
        "reference_index",
        "target_index",
        "target_time",
        "target_velocity_x",
        "target_velocity_y",
    )
    missing = [name for name in required if name not in manifest.columns]
    if missing:
        raise JointDecoderError(f"CA-05 validation manifest is missing {missing}.")
    frame = manifest.loc[:, required]
    sample_index = frame["sample_index"].to_numpy(dtype=np.int64)
    trial_row = frame["trial_row"].to_numpy(dtype=np.int64)
    reference_index = frame["reference_index"].to_numpy(dtype=np.int64)
    target_index = frame["target_index"].to_numpy(dtype=np.int64)
    target_time = frame["target_time"].to_numpy(dtype=np.float64)
    recorded_truth = frame[
        ["target_velocity_x", "target_velocity_y"]
    ].to_numpy(dtype=np.float64)

    expected_indices: list[np.ndarray] = []
    expected_rows: list[np.ndarray] = []
    warmup_steps = (_HISTORY_MS + _ELIGIBILITY_LAG_MS) // _CADENCE_MS
    for row in sorted(bounds):
        trial = bounds[row]
        first = trial.first_index + warmup_steps
        if first <= trial.last_index:
            indices = np.arange(first, trial.last_index + 1, dtype=np.int64)
            expected_indices.append(indices)
            expected_rows.append(np.full(indices.size, row, dtype=np.int64))
    reconstructed_index = np.concatenate(expected_indices)
    reconstructed_rows = np.concatenate(expected_rows)
    reconstructed_reference = reconstructed_index - (
        _ELIGIBILITY_LAG_MS // _CADENCE_MS
    )
    times = np.asarray(timestamps, dtype=np.float64)
    true_velocity = np.asarray(velocity, dtype=np.float64)[reconstructed_index]

    identity_checks = {
        "sample_index": np.array_equal(
            sample_index, np.arange(sample_index.size, dtype=np.int64)
        ),
        "trial_row": np.array_equal(trial_row, reconstructed_rows),
        "reference_index": np.array_equal(
            reference_index, reconstructed_reference
        ),
        "target_index": np.array_equal(target_index, reconstructed_index),
        "target_time": np.allclose(
            target_time,
            times[reconstructed_index],
            atol=_CSV_ROUND_TRIP_ATOL,
            rtol=0.0,
        ),
        "true_velocity": np.allclose(
            recorded_truth,
            true_velocity,
            atol=_CSV_ROUND_TRIP_ATOL,
            rtol=0.0,
        ),
    }
    if not all(identity_checks.values()):
        failed = [name for name, passed in identity_checks.items() if not passed]
        detail = ""
        if target_time.shape == times[reconstructed_index].shape:
            detail += (
                f"; max target-time difference "
                f"{float(np.max(np.abs(target_time - times[reconstructed_index]))):.3e}"
            )
        if recorded_truth.shape == true_velocity.shape:
            detail += (
                f"; max truth difference "
                f"{float(np.max(np.abs(recorded_truth - true_velocity))):.3e}"
            )
        raise JointDecoderError(
            "CA-05 validation evidence differs from the independent "
            f"frozen-protocol reconstruction for {failed}{detail}."
        )
    if (
        sample_index.size != _EXPECTED_VALIDATION_TARGETS
        or np.unique(trial_row).size != _EXPECTED_VALIDATION_TRIALS
    ):
        raise JointDecoderError(
            f"Expected 8,183 validation targets from 24 trials, got "
            f"{sample_index.size} from {np.unique(trial_row).size}."
        )

    identity_hash = _validation_identity_sha256(
        sample_index,
        trial_row,
        reference_index,
        target_index,
        target_time,
    )
    truth_hash = _array_sha256("ca05-validation-truth-v1", recorded_truth)
    return (
        ValidationIdentity(
            sample_index=sample_index,
            trial_row=trial_row,
            reference_index=reference_index,
            target_index=target_index,
            target_time=target_time,
            identity_sha256=identity_hash,
            source_sha256=source_sha256,
        ),
        ValidationScoringLedger(
            velocity=recorded_truth.copy(),
            truth_sha256=truth_hash,
        ),
    )


def filter_official_targets(
    model: DecoderModel,
    schedule: ScheduledObservations,
    bounds: Mapping[int, TrialBounds],
    identity: ValidationIdentity,
    *,
    initial_positions: Mapping[int, np.ndarray],
) -> FinalFilterResult:
    """Filter validation trials without making scoring truth accessible."""
    if model.spec != FROZEN_SPEC:
        raise JointDecoderError("Official filtering accepts only PVA/50/50.")
    if set(bounds) != set(initial_positions):
        raise JointDecoderError("Every validation trial needs one starting position.")

    predictions = np.full((identity.n_samples, 2), np.nan, dtype=np.float64)
    covariances = np.full((identity.n_samples, 2, 2), np.nan, dtype=np.float64)
    fill_count = np.zeros(identity.n_samples, dtype=np.int64)
    innovation_chunks: list[np.ndarray] = []
    predicted_covariance_chunks: list[np.ndarray] = []
    innovation_row_chunks: list[np.ndarray] = []
    max_gain = 0.0
    max_state = 0.0
    max_covariance = 0.0
    min_eigenvalue = float("inf")
    max_eigenvalue = 0.0
    min_rank = FROZEN_SPEC.dimension

    for row in sorted(bounds):
        trial_bounds = bounds[row]
        assessment = build_assessment_trial(
            schedule,
            trial_bounds,
            initial_position=initial_positions[row],
        )
        trace = filter_trial(model, assessment)
        positions = np.flatnonzero(identity.trial_row == row)
        offsets = identity.target_index[positions] - trial_bounds.first_index
        if (
            positions.size == 0
            or np.any(offsets < 0)
            or np.any(offsets >= trial_bounds.n_steps)
        ):
            raise JointDecoderError(
                f"Official targets are outside validation trial row {row}."
            )
        predictions[positions] = trace.velocity[offsets]
        covariances[positions] = trace.velocity_covariance[offsets]
        fill_count[positions] += 1
        innovation_chunks.append(trace.innovations)
        predicted_covariance_chunks.append(
            trace.predicted_covariance_at_updates
        )
        innovation_row_chunks.append(
            np.full(trace.innovations.shape[0], row, dtype=np.int64)
        )
        max_gain = max(max_gain, trace.max_abs_gain)
        max_state = max(max_state, float(np.max(np.abs(trace.normalized_state))))
        max_covariance = max(
            max_covariance, float(np.max(np.abs(trace.posterior_covariance)))
        )
        eigenvalues = np.linalg.eigvalsh(trace.posterior_covariance)
        min_eigenvalue = min(min_eigenvalue, float(eigenvalues.min()))
        max_eigenvalue = max(max_eigenvalue, float(eigenvalues.max()))
        min_rank = min(
            min_rank,
            min(
                int(np.linalg.matrix_rank(matrix))
                for matrix in trace.posterior_covariance
            ),
        )

    validate_oof_fill(predictions, fill_count)
    if not np.all(np.isfinite(covariances)):
        raise JointDecoderError("Validation velocity covariances are incomplete.")
    return FinalFilterResult(
        prediction=predictions,
        velocity_covariance=covariances,
        fill_count=fill_count,
        innovations=np.vstack(innovation_chunks),
        predicted_covariance_at_updates=np.concatenate(
            predicted_covariance_chunks
        ),
        innovation_trial_row=np.concatenate(innovation_row_chunks),
        max_abs_gain=max_gain,
        max_abs_normalized_state=max_state,
        max_abs_posterior_covariance=max_covariance,
        posterior_min_eigenvalue=min_eigenvalue,
        posterior_max_eigenvalue=max_eigenvalue,
        posterior_min_rank=min_rank,
    )


def decoder_to_dict(
    model: DecoderModel,
    *,
    unit_ids: Sequence[int],
    provenance: Mapping[str, str],
) -> dict[str, object]:
    """Serialize the fitted decoder without validation predictions or truth."""
    if model.spec != FROZEN_SPEC:
        raise JointDecoderError("Only the frozen PVA/50/50 model may be serialized.")
    identifiers = tuple(int(value) for value in unit_ids)
    if len(identifiers) != _EXPECTED_UNITS or len(set(identifiers)) != len(identifiers):
        raise JointDecoderError("Model serialization requires 53 unique unit IDs.")
    process = model.process
    observation = model.observation
    return {
        "schema_version": _MODEL_SCHEMA_VERSION,
        "configuration": {
            "state_name": model.spec.state_name,
            "state_components": list(model.spec.components),
            "state_dimension": model.spec.dimension,
            "neural_width_ms": model.spec.neural_width_ms,
            "interval_end_lag_ms": model.spec.interval_end_lag_ms,
            "state_and_output_cadence_ms": _CADENCE_MS,
        },
        "unit_ids": list(identifiers),
        "provenance": dict(provenance),
        "process": {
            "normalization_mean": process.normalization.mean.tolist(),
            "normalization_scale": process.normalization.scale.tolist(),
            "transition": process.transition.tolist(),
            "offset": process.offset.tolist(),
            "process_covariance": process.process_covariance.tolist(),
            "initial_mean": process.initial_mean.tolist(),
            "initial_covariance": process.initial_covariance.tolist(),
            "training_rows": list(process.training_rows),
            "diagnostics": process.diagnostics,
        },
        "observation": {
            "count_mean": observation.count_mean.tolist(),
            "count_scale": observation.count_scale.tolist(),
            "observation_matrix": observation.observation_matrix.tolist(),
            "offset": observation.offset.tolist(),
            "residual_covariance": observation.residual_covariance.tolist(),
            "shrinkage": observation.shrinkage,
            "training_rows": list(observation.training_rows),
            "diagnostics": observation.diagnostics,
        },
    }


def decoder_from_dict(
    payload: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, str] | None = None,
) -> tuple[DecoderModel, tuple[int, ...]]:
    """Load and validate a serialized frozen decoder."""
    if int(payload.get("schema_version", -1)) != _MODEL_SCHEMA_VERSION:
        raise JointDecoderError("Unsupported final decoder schema version.")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise JointDecoderError("Serialized decoder has no configuration mapping.")
    spec = CandidateSpec(
        str(configuration.get("state_name")),
        int(configuration.get("neural_width_ms", -1)),
        int(configuration.get("interval_end_lag_ms", -1)),
    )
    if (
        spec != FROZEN_SPEC
        or tuple(configuration.get("state_components", ())) != spec.components
        or int(configuration.get("state_dimension", -1)) != spec.dimension
        or int(configuration.get("state_and_output_cadence_ms", -1))
        != _CADENCE_MS
    ):
        raise JointDecoderError("Serialized decoder configuration is not frozen PVA/50/50.")
    unit_ids = tuple(int(value) for value in payload.get("unit_ids", ()))
    if len(unit_ids) != _EXPECTED_UNITS or len(set(unit_ids)) != len(unit_ids):
        raise JointDecoderError("Serialized decoder unit identity is invalid.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in provenance.items()
    ):
        raise JointDecoderError("Serialized decoder provenance is invalid.")
    if expected_provenance is not None and dict(provenance) != dict(
        expected_provenance
    ):
        raise JointDecoderError("Serialized decoder provenance differs from frozen inputs.")

    process_payload = payload.get("process")
    observation_payload = payload.get("observation")
    if not isinstance(process_payload, Mapping) or not isinstance(
        observation_payload, Mapping
    ):
        raise JointDecoderError("Serialized decoder parameter sections are missing.")
    dimension = FROZEN_SPEC.dimension
    normalization = StateNormalization(
        mean=_finite_array(
            process_payload, "normalization_mean", (dimension,)
        ),
        scale=_finite_array(
            process_payload, "normalization_scale", (dimension,)
        ),
        training_rows=tuple(
            int(row) for row in process_payload.get("training_rows", ())
        ),
    )
    if np.any(normalization.scale <= 0.0):
        raise JointDecoderError("Serialized state scales must be positive.")
    process_covariance = _finite_array(
        process_payload, "process_covariance", (dimension, dimension)
    )
    initial_covariance = _finite_array(
        process_payload, "initial_covariance", (dimension, dimension)
    )
    _require_psd(process_covariance, "serialized Q")
    _require_psd(initial_covariance, "serialized initial covariance")
    process = ProcessModel(
        state_name=FROZEN_SPEC.state_name,
        components=FROZEN_SPEC.components,
        normalization=normalization,
        transition=_finite_array(
            process_payload, "transition", (dimension, dimension)
        ),
        offset=_finite_array(process_payload, "offset", (dimension,)),
        process_covariance=process_covariance,
        initial_mean=_finite_array(
            process_payload, "initial_mean", (dimension,)
        ),
        initial_covariance=initial_covariance,
        training_rows=normalization.training_rows,
        diagnostics=dict(process_payload.get("diagnostics", {})),
    )

    count_mean = _finite_array(
        observation_payload, "count_mean", (_EXPECTED_UNITS,)
    )
    count_scale = _finite_array(
        observation_payload, "count_scale", (_EXPECTED_UNITS,)
    )
    if np.any(count_scale <= 0.0):
        raise JointDecoderError("Serialized neural scales must be positive.")
    observation_matrix = _finite_array(
        observation_payload,
        "observation_matrix",
        (_EXPECTED_UNITS, dimension),
    )
    residual_covariance = _finite_array(
        observation_payload,
        "residual_covariance",
        (_EXPECTED_UNITS, _EXPECTED_UNITS),
    )
    try:
        residual_cholesky = np.linalg.cholesky(
            (residual_covariance + residual_covariance.T) * 0.5
        )
    except np.linalg.LinAlgError as error:
        raise JointDecoderError("Serialized R is not positive definite.") from error
    residual_inverse_observation = cho_solve(
        (residual_cholesky, True),
        observation_matrix,
        check_finite=False,
    )
    prewhitened_observation = solve_triangular(
        residual_cholesky,
        observation_matrix,
        lower=True,
        check_finite=False,
    )
    observation_rows = tuple(
        int(row) for row in observation_payload.get("training_rows", ())
    )
    if observation_rows != process.training_rows:
        raise JointDecoderError("Serialized process and observation training rows differ.")
    observation = ObservationModel(
        count_mean=count_mean,
        count_scale=count_scale,
        observation_matrix=observation_matrix,
        offset=_finite_array(
            observation_payload, "offset", (_EXPECTED_UNITS,)
        ),
        residual_covariance=residual_covariance,
        residual_cholesky=residual_cholesky,
        residual_inverse_observation=residual_inverse_observation,
        prewhitened_observation=prewhitened_observation,
        shrinkage=float(observation_payload.get("shrinkage", np.nan)),
        training_rows=observation_rows,
        diagnostics=dict(observation_payload.get("diagnostics", {})),
    )
    if not np.isfinite(observation.shrinkage):
        raise JointDecoderError("Serialized OAS shrinkage is non-finite.")
    return DecoderModel(spec=FROZEN_SPEC, process=process, observation=observation), unit_ids


def _finite_array(
    payload: Mapping[str, Any], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.asarray(payload.get(key), dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise JointDecoderError(
            f"Serialized {key} has shape {array.shape}; expected finite {shape}."
        )
    return array


def _assemble_training_state(
    session: SessionData,
    bounds: Mapping[int, TrialBounds],
) -> np.ndarray:
    """Materialize behavior only inside declared training-trial bounds."""
    n_samples = int(np.asarray(session.behavior_timestamps).size)
    sources = (
        np.asarray(session.cursor_position),
        np.asarray(session.cursor_velocity),
        np.asarray(session.cursor_acceleration),
    )
    if any(source.shape != (n_samples, 2) for source in sources):
        raise JointDecoderError("Training behavior arrays must align as (n, 2).")

    state = np.full((n_samples, FROZEN_SPEC.dimension), np.nan, dtype=np.float64)
    for row in sorted(bounds):
        trial = bounds[row]
        indices = slice(trial.first_index, trial.last_index + 1)
        chunk = np.column_stack([source[indices] for source in sources])
        if not np.all(np.isfinite(chunk)):
            raise JointDecoderError(
                f"Training state contains non-finite values in trial row {row}."
            )
        state[indices] = chunk
    return state


def _require_psd(matrix: np.ndarray, label: str) -> None:
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = (
        np.finfo(float).eps * max(matrix.shape) * scale * 256.0
    )
    if float(eigenvalues.min()) < -tolerance:
        raise JointDecoderError(f"{label} is not positive semidefinite.")


def _validation_identity_sha256(
    sample_index: np.ndarray,
    trial_row: np.ndarray,
    reference_index: np.ndarray,
    target_index: np.ndarray,
    target_time: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"ca05-validation-identity-v1\0")
    for array, dtype in (
        (sample_index, "<i8"),
        (trial_row, "<i8"),
        (reference_index, "<i8"),
        (target_index, "<i8"),
        (target_time, "<f8"),
    ):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        digest.update(np.asarray([canonical.size], dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _array_sha256(namespace: str, array: np.ndarray) -> str:
    digest = hashlib.sha256(namespace.encode("ascii") + b"\0")
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()
