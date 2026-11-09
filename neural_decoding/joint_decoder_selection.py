"""Training-only orchestration for the joint decoder grid."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from neural_decoding.joint_decoder import (
    CandidateSpec,
    DecoderModel,
    FoldAssignments,
    JointDecoderError,
    ScheduledObservations,
    TargetLedger,
    TrialBounds,
    assemble_state,
    build_assessment_trial,
    build_scheduled_observations,
    candidate_grid,
    evaluate_calibration,
    filter_trial,
    fit_observation_model,
    fit_process_model,
    innovation_diagnostics,
    select_candidate,
    validate_oof_fill,
)
from neural_decoding.metrics import summarize_predictions
from neural_decoding.logging_utils import get_logger

if TYPE_CHECKING:
    from neural_decoding.dataset import SessionData

__all__ = ["SelectionRun", "run_selection"]

logger = get_logger("joint_decoder_selection")


@dataclass(frozen=True)
class SelectionRun:
    """In-memory result of all 36 candidates and 180 fold fits."""

    candidates: tuple[dict[str, object], ...]
    selected: dict[str, object] | None
    checks: tuple[dict[str, object], ...]

    @property
    def all_checks_passed(self) -> bool:
        return all(bool(check["passed"]) for check in self.checks)


def run_selection(
    session: SessionData,
    *,
    bounds: dict[int, TrialBounds],
    folds: FoldAssignments,
    ledger: TargetLedger,
    run_label: str = "selection",
) -> SelectionRun:
    """Evaluate the complete frozen grid without persisting temporary arrays."""
    run_started = perf_counter()
    eligible_rows = tuple(sorted(bounds))
    if set(eligible_rows) != set(folds.rows):
        raise JointDecoderError("Trial bounds and tracked folds contain different rows.")

    logger.info("%s: preparing four state arrays and nine neural schedules", run_label)
    states = {
        state_name: assemble_state(
            session.cursor_position,
            session.cursor_velocity,
            session.cursor_acceleration,
            state_name,
        )
        for state_name in ("V", "PV", "VA", "PVA")
    }
    schedules = {
        (width, lag): build_scheduled_observations(
            session.behavior_timestamps,
            session.spike_times,
            bounds,
            width_ms=width,
            lag_ms=lag,
        )
        for width in (10, 20, 50)
        for lag in (0, 50, 100)
    }

    process_cache: dict[tuple[str, int], object] = {}
    records: list[dict[str, object]] = []
    grid = candidate_grid()
    for candidate_index, spec in enumerate(grid, start=1):
        candidate_started = perf_counter()
        logger.info(
            "%s: candidate %d/%d %s, width %d ms, lag %d ms",
            run_label,
            candidate_index,
            len(grid),
            spec.state_name,
            spec.neural_width_ms,
            spec.interval_end_lag_ms,
        )
        try:
            record = _evaluate_candidate(
                spec,
                session=session,
                state=states[spec.state_name],
                schedule=schedules[(spec.neural_width_ms, spec.interval_end_lag_ms)],
                bounds=bounds,
                folds=folds,
                ledger=ledger,
                process_cache=process_cache,
            )
        except (JointDecoderError, np.linalg.LinAlgError, ValueError) as error:
            record = _failed_record(spec, str(error))
        records.append(record)
        logger.info(
            "%s: candidate %d/%d %s in %.1f s (mean R2 %s)",
            run_label,
            candidate_index,
            len(grid),
            "passed" if record["passed"] else "FAILED",
            perf_counter() - candidate_started,
            (
                f"{float(record['out_of_fold_r2_mean']):.6f}"
                if record["passed"]
                else str(record["failure_reason"])
            ),
        )

    checks: list[dict[str, object]] = [
        _check(
            "tracked folds cover five complete disjoint groups",
            len(folds.groups) == 5
            and len(set(folds.rows)) == 123
            and len(folds.rows) == 123,
            f"groups sized {[len(group) for group in folds.groups]}; SHA-256 "
            f"{folds.source_sha256}",
        ),
        _check(
            "only eligible training trials were prepared",
            set(bounds) == set(folds.rows) and len(bounds) == 123,
            "123 eligible training rows; validation and held-out streams were not constructed",
        ),
        _check(
            "official Step 5 target identities are fixed",
            ledger.n_samples == 43_022 and np.unique(ledger.trial_row).size == 123,
            f"{ledger.n_samples} targets; identity SHA-256 {ledger.identity_sha256}",
        ),
        _check(
            "the complete frozen grid was evaluated",
            len(records) == 36
            and {
                (
                    str(record["state_name"]),
                    int(record["neural_width_ms"]),
                    int(record["interval_end_lag_ms"]),
                )
                for record in records
            }
            == {
                (spec.state_name, spec.neural_width_ms, spec.interval_end_lag_ms)
                for spec in candidate_grid()
            },
            f"{len(records)} candidates and {len(records) * 5} fold fits attempted",
        ),
        _check(
            "candidate outputs satisfy recursive numerical and isolation gates",
            all(bool(record["passed"]) for record in records),
            f"{sum(bool(record['passed']) for record in records)} of 36 candidates passed",
        ),
    ]

    selected: dict[str, object] | None = None
    try:
        winner = dict(select_candidate(records))
    except JointDecoderError as error:
        checks.append(_check("a finite candidate is selectable", False, str(error)))
    else:
        checks.append(
            _check(
                "the frozen deterministic ranking rule was applied",
                True,
                f"{winner['state_name']}, {winner['neural_width_ms']} ms, "
                f"{winner['interval_end_lag_ms']} ms lag",
            )
        )
        useful = (
            float(winner["out_of_fold_r2_x"]) > 0.0
            and float(winner["out_of_fold_r2_y"]) > 0.0
        )
        checks.append(
            _check(
                "the winner has strictly positive finite R2 on both axes",
                useful,
                f"x {float(winner['out_of_fold_r2_x']):.6f}, "
                f"y {float(winner['out_of_fold_r2_y']):.6f}",
            )
        )
        if useful and all(bool(check["passed"]) for check in checks):
            selected = winner

    result = SelectionRun(
        candidates=tuple(records),
        selected=selected,
        checks=tuple(checks),
    )
    logger.info("%s completed in %.1f s", run_label, perf_counter() - run_started)
    return result


def _evaluate_candidate(
    spec: CandidateSpec,
    *,
    session: SessionData,
    state: np.ndarray,
    schedule: ScheduledObservations,
    bounds: dict[int, TrialBounds],
    folds: FoldAssignments,
    ledger: TargetLedger,
    process_cache: dict[tuple[str, int], object],
) -> dict[str, object]:
    predictions = np.full((ledger.n_samples, 2), np.nan, dtype=np.float64)
    velocity_covariances = np.full(
        (ledger.n_samples, 2, 2), np.nan, dtype=np.float64
    )
    fill_count = np.zeros(ledger.n_samples, dtype=np.int64)
    fold_scores: list[float] = []
    fold_process_diagnostics: list[dict[str, object]] = []
    fold_observation_diagnostics: list[dict[str, object]] = []
    fold_innovation_diagnostics: list[dict[str, float]] = []
    maximum_gain = 0.0
    maximum_state = 0.0
    maximum_covariance = 0.0
    posterior_min_eigenvalue = float("inf")
    posterior_max_eigenvalue = 0.0
    posterior_min_rank = spec.dimension

    for fold_index in range(5):
        training_rows = folds.training_rows(fold_index)
        assessment_rows = folds.assessment_rows(fold_index)
        process_key = (spec.state_name, fold_index)
        process = process_cache.get(process_key)
        if process is None:
            process = fit_process_model(
                state, bounds, training_rows, spec.state_name
            )
            process_cache[process_key] = process
        if tuple(process.training_rows) != tuple(sorted(training_rows)):
            raise JointDecoderError("A cached process fit came from the wrong fold.")

        normalized_state = process.normalization.apply(state)
        observation = fit_observation_model(
            schedule, normalized_state, training_rows
        )
        if set(observation.training_rows) & set(assessment_rows):
            raise JointDecoderError("Observation fitting included an assessment trial.")
        model = DecoderModel(spec=spec, process=process, observation=observation)

        innovation_chunks: list[np.ndarray] = []
        predicted_covariance_chunks: list[np.ndarray] = []
        innovation_row_chunks: list[np.ndarray] = []
        fold_positions: list[np.ndarray] = []
        for row in assessment_rows:
            trial_bounds = bounds[row]
            position = (
                session.cursor_position[trial_bounds.first_index]
                if process.position_indices is not None
                else None
            )
            assessment = build_assessment_trial(
                schedule,
                trial_bounds,
                initial_position=position,
            )
            trace = filter_trial(model, assessment)
            target_positions = np.flatnonzero(ledger.trial_row == row)
            target_offsets = (
                ledger.target_index[target_positions] - trial_bounds.first_index
            )
            if (
                target_offsets.size == 0
                or np.any(target_offsets < 0)
                or np.any(target_offsets >= trial_bounds.n_steps)
            ):
                raise JointDecoderError(
                    f"Official targets are outside assessment trial row {row}."
                )
            predictions[target_positions] = trace.velocity[target_offsets]
            velocity_covariances[target_positions] = trace.velocity_covariance[
                target_offsets
            ]
            fill_count[target_positions] += 1
            fold_positions.append(target_positions)

            innovation_chunks.append(trace.innovations)
            predicted_covariance_chunks.append(
                trace.predicted_covariance_at_updates
            )
            innovation_row_chunks.append(
                np.full(trace.innovations.shape[0], row, dtype=np.int64)
            )
            maximum_gain = max(maximum_gain, trace.max_abs_gain)
            maximum_state = max(
                maximum_state, float(np.max(np.abs(trace.normalized_state)))
            )
            maximum_covariance = max(
                maximum_covariance,
                float(np.max(np.abs(trace.posterior_covariance))),
            )
            eigenvalues = np.linalg.eigvalsh(trace.posterior_covariance)
            posterior_min_eigenvalue = min(
                posterior_min_eigenvalue, float(eigenvalues.min())
            )
            posterior_max_eigenvalue = max(
                posterior_max_eigenvalue, float(eigenvalues.max())
            )
            ranks = [
                int(np.linalg.matrix_rank(matrix))
                for matrix in trace.posterior_covariance
            ]
            posterior_min_rank = min(posterior_min_rank, min(ranks))

        positions = np.concatenate(fold_positions)
        fold_metrics = summarize_predictions(
            ledger.velocity[positions],
            predictions[positions],
            axis_names=("x", "y"),
        )
        fold_scores.append(float(fold_metrics["r2_mean"]))
        fold_process_diagnostics.append(process.diagnostics)
        fold_observation_diagnostics.append(observation.diagnostics)
        fold_innovation_diagnostics.append(
            innovation_diagnostics(
                model,
                np.vstack(innovation_chunks),
                np.concatenate(predicted_covariance_chunks),
                np.concatenate(innovation_row_chunks),
            )
        )

    validate_oof_fill(predictions, fill_count)
    metrics = summarize_predictions(
        ledger.velocity, predictions, axis_names=("x", "y")
    )
    calibration = evaluate_calibration(
        ledger.velocity, predictions, velocity_covariances
    )
    target_std = ledger.velocity.std(axis=0)
    prediction_std = predictions.std(axis=0)
    ratios = prediction_std / target_std

    record: dict[str, object] = {
        "state_name": spec.state_name,
        "state_dimension": spec.dimension,
        "neural_width_ms": spec.neural_width_ms,
        "interval_end_lag_ms": spec.interval_end_lag_ms,
        "cadence_ms": 10,
        "n_samples": ledger.n_samples,
        "out_of_fold_r2_x": float(metrics["r2"]["x"]),
        "out_of_fold_r2_y": float(metrics["r2"]["y"]),
        "out_of_fold_r2_mean": float(metrics["r2_mean"]),
        "pearson_r_x": _metric_value(metrics["pearson_r"]["x"]),
        "pearson_r_y": _metric_value(metrics["pearson_r"]["y"]),
        "pearson_r_mean": _metric_value(metrics["pearson_r_mean"]),
        "rmse_cm_s_x": float(metrics["rmse_cm_s"]["x"]),
        "rmse_cm_s_y": float(metrics["rmse_cm_s"]["y"]),
        "rmse_cm_s_mean": float(metrics["rmse_cm_s_mean"]),
        "prediction_to_target_std_x": float(ratios[0]),
        "prediction_to_target_std_y": float(ratios[1]),
        **{
            f"fold{index}_r2_mean": value
            for index, value in enumerate(fold_scores)
        },
        "fold_r2_min": float(min(fold_scores)),
        "fold_r2_max": float(max(fold_scores)),
        "fold_r2_spread": float(max(fold_scores) - min(fold_scores)),
        "transition_spectral_radius_max": _maximum(
            fold_process_diagnostics, "transition_spectral_radius"
        ),
        "transition_design_condition_max": _maximum(
            fold_process_diagnostics, "transition_design_condition"
        ),
        "process_residual_autocorrelation_abs_max": _maximum(
            fold_process_diagnostics,
            "process_residual_autocorrelation_abs_max",
        ),
        "process_covariance_rank_min": _minimum(
            fold_process_diagnostics, "process_covariance_rank"
        ),
        "process_covariance_min_eigenvalue_min": _minimum(
            fold_process_diagnostics, "process_covariance_min_eigenvalue"
        ),
        "process_covariance_max_eigenvalue_max": _maximum(
            fold_process_diagnostics, "process_covariance_max_eigenvalue"
        ),
        "observation_design_condition_max": _maximum(
            fold_observation_diagnostics, "observation_design_condition"
        ),
        "observation_residual_autocorrelation_abs_max": _maximum(
            fold_observation_diagnostics,
            "observation_residual_autocorrelation_abs_max",
        ),
        "observation_covariance_rank_min": _minimum(
            fold_observation_diagnostics, "observation_covariance_rank"
        ),
        "observation_covariance_min_eigenvalue_min": _minimum(
            fold_observation_diagnostics,
            "observation_covariance_min_eigenvalue",
        ),
        "observation_covariance_condition_max": _maximum(
            fold_observation_diagnostics, "observation_covariance_condition"
        ),
        "observation_covariance_shrinkage_min": _minimum(
            fold_observation_diagnostics, "observation_covariance_shrinkage"
        ),
        "observation_covariance_shrinkage_max": _maximum(
            fold_observation_diagnostics, "observation_covariance_shrinkage"
        ),
        "innovation_whitened_autocorrelation_abs_max": _maximum(
            fold_innovation_diagnostics,
            "innovation_whitened_autocorrelation_abs_max",
        ),
        "innovation_whitened_squared_norm_mean_max": _maximum(
            fold_innovation_diagnostics,
            "innovation_whitened_squared_norm_mean",
        ),
        "innovation_absolute_max": _maximum(
            fold_innovation_diagnostics, "innovation_absolute_max"
        ),
        "posterior_covariance_rank_min": posterior_min_rank,
        "posterior_covariance_min_eigenvalue_min": posterior_min_eigenvalue,
        "posterior_covariance_max_eigenvalue_max": posterior_max_eigenvalue,
        "kalman_gain_absolute_max": maximum_gain,
        "normalized_state_absolute_max": maximum_state,
        "posterior_covariance_absolute_max": maximum_covariance,
        **calibration,
        "passed": True,
        "failure_reason": "",
    }
    if not _finite_record(record):
        raise JointDecoderError("A required candidate metric or diagnostic is non-finite.")
    return record


def _failed_record(spec: CandidateSpec, reason: str) -> dict[str, object]:
    return {
        "state_name": spec.state_name,
        "state_dimension": spec.dimension,
        "neural_width_ms": spec.neural_width_ms,
        "interval_end_lag_ms": spec.interval_end_lag_ms,
        "cadence_ms": 10,
        "n_samples": 0,
        "passed": False,
        "failure_reason": reason[:300],
    }


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _maximum(records: list[dict[str, object]], key: str) -> float:
    return float(max(float(record[key]) for record in records))


def _minimum(records: list[dict[str, object]], key: str) -> float:
    return float(min(float(record[key]) for record in records))


def _metric_value(value: object) -> float:
    if value is None:
        raise JointDecoderError("A required prediction correlation is undefined.")
    return float(value)


def _finite_record(record: dict[str, object]) -> bool:
    ignored = {
        "state_name",
        "passed",
        "failure_reason",
    }
    for key, value in record.items():
        if key in ignored or isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(value)):
                return False
    return True
