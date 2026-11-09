"""Fit the selected PVA Kalman decoder and evaluate it on validation trials.

Run input checks with::

    python -m scripts.validate_frozen_decoder --preflight
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_decoding.config import NWB_PATH_ENV_VAR, ConfigError, load_config, project_root
from neural_decoding.dataset import DatasetError, SessionData, compute_sha256, load_session
from neural_decoding.final_decoder import (
    FROZEN_SPEC,
    FinalFilterResult,
    ValidationIdentity,
    ValidationScoringLedger,
    build_validation_ledgers,
    decoder_from_dict,
    decoder_to_dict,
    filter_official_targets,
    fit_frozen_decoder,
)
from neural_decoding.joint_decoder import (
    DecoderModel,
    FoldAssignments,
    JointDecoderError,
    TargetLedger,
    TrialBounds,
    build_assessment_trial,
    build_official_target_ledger,
    build_scheduled_observations,
    build_trial_bounds,
    evaluate_calibration,
    filter_trial,
    fold_assignments_from_frame,
    innovation_diagnostics,
)
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.metrics import r2_per_axis, summarize_predictions
from neural_decoding.splits import TRAIN, VALIDATION

logger = get_logger("validate_frozen_decoder")

_SELECTION_DIRECTORY = Path("results/internal/kalman_decoder_selection")
_SELECTION_CANDIDATES = _SELECTION_DIRECTORY / "candidates.csv"
_SELECTION_SUMMARY = _SELECTION_DIRECTORY / "summary.json"
_SELECTED_CONFIG = _SELECTION_DIRECTORY / "selected_config.json"
_FOLDS = Path("results/selection/fold_assignments.csv")
_BASELINE_SUMMARY = Path("results/baseline_summary.json")
_BASELINE_PREDICTIONS = Path("results/baselines/validation_predictions.csv")
_OUTPUT_DIRECTORY = Path("results/internal/kalman_validation")


@dataclass(frozen=True)
class VerifiedInputs:
    """Validated training, validation, and selection inputs."""

    session: SessionData
    training_bounds: dict[int, TrialBounds]
    folds: FoldAssignments
    training_ledger: TargetLedger
    selected_config: dict[str, Any]
    selection_summary: dict[str, Any]
    baseline_summary: dict[str, Any]
    hashes: dict[str, str]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run provenance, training-only fit, serialization, and validation-"
        "identity checks without filtering validation or writing official artifacts.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    configure_logging(config.run.log_level)
    root = project_root()
    output_directory = root / _OUTPUT_DIRECTORY
    if output_directory.exists():
        print(
            "Refusing to run: the one-time CA-06C output directory already exists at "
            f"{output_directory}."
        )
        return 1

    try:
        inputs = _verify_frozen_inputs(root, config)
        logger.info("Fitting frozen PVA/50/50 on 123 eligible training trials")
        model, training_schedule = fit_frozen_decoder(
            inputs.session, inputs.training_bounds
        )
        model_provenance = {
            "recording_sha256": inputs.session.identity.sha256,
            "fold_assignments_sha256": inputs.folds.source_sha256,
            "training_target_identity_sha256": (
                inputs.training_ledger.identity_sha256
            ),
            "selection_candidates_sha256": inputs.hashes[
                "selection_candidates_sha256"
            ],
            "selection_summary_sha256": inputs.hashes[
                "selection_summary_sha256"
            ],
            "selected_config_sha256": inputs.hashes[
                "selected_config_sha256"
            ],
        }
        model_payload = decoder_to_dict(
            model,
            unit_ids=[unit.unit_id for unit in inputs.session.units],
            provenance=model_provenance,
        )
        reloaded_model, reloaded_unit_ids = decoder_from_dict(
            model_payload,
            expected_provenance=model_provenance,
        )
        _training_only_reload_check(
            inputs.session,
            inputs.training_bounds,
            training_schedule,
            model,
            reloaded_model,
            reloaded_unit_ids,
        )
    except (ConfigError, DatasetError, FileNotFoundError) as error:
        print(f"Cannot prepare frozen decoder validation: {error}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, and retain all reviewed selection inputs."
        )
        return 2
    except (JointDecoderError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"Frozen decoder preflight failed: {error}")
        return 1

    if args.preflight:
        try:
            validation_identity, _, _ = _load_validation_evidence(
                root, inputs.session
            )
        except (JointDecoderError, FileNotFoundError, ValueError) as error:
            print(f"Frozen decoder validation-identity preflight failed: {error}")
            return 1
        print("PRE-FLIGHT PASSED")
        print("  frozen configuration  PVA / 50 ms width / 50 ms lag")
        print("  final fit              123 eligible training trials")
        print("  serialization          training replay matches within 1e-12")
        print(
            "  validation identity   "
            f"{validation_identity.n_samples} CA-05 rows verified"
        )
        print("  validation filtering   not run")
        print("  held-out               not constructed or filtered")
        return 0

    # Directory creation is the irreversible marker that the one authorized
    # validation run has begun.  There is deliberately no overwrite/force flag.
    output_directory.mkdir(parents=True, exist_ok=False)
    model_path = output_directory / "model.json"
    predictions_path = output_directory / "validation_predictions.csv"
    configuration_path = output_directory / "configuration.json"
    summary_path = output_directory / "summary.json"

    try:
        model_path.write_text(
            json.dumps(model_payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        persisted_payload = json.loads(model_path.read_text(encoding="utf-8"))
        reloaded_model, reloaded_unit_ids = decoder_from_dict(
            persisted_payload,
            expected_provenance=model_provenance,
        )
        _training_only_reload_check(
            inputs.session,
            inputs.training_bounds,
            training_schedule,
            model,
            reloaded_model,
            reloaded_unit_ids,
        )
        validation_identity, scoring_ledger, validation_bounds = (
            _load_validation_evidence(root, inputs.session)
        )
        validation_trial_ids = _validation_trial_ids(
            inputs.session,
            validation_bounds,
        )
        validation_schedule = build_scheduled_observations(
            inputs.session.behavior_timestamps,
            inputs.session.spike_times,
            validation_bounds,
            width_ms=FROZEN_SPEC.neural_width_ms,
            lag_ms=FROZEN_SPEC.interval_end_lag_ms,
        )
        initial_positions = {
            row: inputs.session.cursor_position[bounds.first_index].copy()
            for row, bounds in validation_bounds.items()
        }
        logger.info(
            "Starting the one official validation filter run over %d trials",
            len(validation_bounds),
        )
        result = filter_official_targets(
            model,
            validation_schedule,
            validation_bounds,
            validation_identity,
            initial_positions=initial_positions,
        )
        _write_predictions(
            predictions_path,
            validation_identity,
            scoring_ledger,
            result,
            validation_trial_ids=validation_trial_ids,
        )

        # Required reload verification replays the same frozen model and schedule;
        # it is not a second model choice or reported evaluation.
        reloaded_result = filter_official_targets(
            reloaded_model,
            validation_schedule,
            validation_bounds,
            validation_identity,
            initial_positions=initial_positions,
        )
        reload_exact = (
            np.allclose(
                result.prediction,
                reloaded_result.prediction,
                atol=1e-12,
                rtol=1e-12,
            )
            and np.allclose(
                result.velocity_covariance,
                reloaded_result.velocity_covariance,
                atol=1e-12,
                rtol=1e-12,
            )
        )
        metrics = summarize_predictions(
            scoring_ledger.velocity,
            result.prediction,
            axis_names=("x", "y"),
        )
        diagnostics = _validation_diagnostics(
            model,
            result,
            scoring_ledger,
        )
        checks = _official_checks(
            inputs=inputs,
            model=model,
            reloaded_unit_ids=reloaded_unit_ids,
            validation_bounds=validation_bounds,
            validation_trial_ids=validation_trial_ids,
            validation_identity=validation_identity,
            result=result,
            reload_exact=reload_exact,
            predictions_path=predictions_path,
            metrics=metrics,
        )

        model_hash = compute_sha256(model_path)
        predictions_hash = compute_sha256(predictions_path)
        configuration = _official_configuration(
            inputs=inputs,
            validation_identity=validation_identity,
            scoring_ledger=scoring_ledger,
            model_hash=model_hash,
            predictions_hash=predictions_hash,
        )
        configuration_path.write_text(
            json.dumps(configuration, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        configuration_hash = compute_sha256(configuration_path)
        accepted = all(bool(check["passed"]) for check in checks)
        summary = _official_summary(
            inputs=inputs,
            validation_identity=validation_identity,
            scoring_ledger=scoring_ledger,
            metrics=metrics,
            diagnostics=diagnostics,
            checks=checks,
            accepted=accepted,
            artifacts={
                "model": _repo_relative(model_path),
                "validation_predictions": _repo_relative(predictions_path),
                "configuration": _repo_relative(configuration_path),
                "summary": _repo_relative(summary_path),
            },
            artifact_hashes={
                "model_sha256": model_hash,
                "validation_predictions_sha256": predictions_hash,
                "configuration_sha256": configuration_hash,
            },
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        # The directory remains as the permanent one-run marker.  Persist a
        # structural-failure summary and deliberately refuse any automatic retry.
        failure = {
            "procedure": "one official frozen Kalman validation",
            "accepted": False,
            "structural_failure": f"{type(error).__name__}: {error}",
            "rerun_permitted": False,
            "held_out": "not constructed or filtered",
        }
        summary_path.write_text(
            json.dumps(failure, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"OFFICIAL VALIDATION FAILED STRUCTURALLY: {error}")
        print("No automatic repair or rerun is permitted.")
        return 1

    _print_official_result(metrics, diagnostics, checks, accepted)
    return 0 if accepted else 1


def _verify_frozen_inputs(root: Path, config) -> VerifiedInputs:
    paths = {
        "selection_candidates": root / _SELECTION_CANDIDATES,
        "selection_summary": root / _SELECTION_SUMMARY,
        "selected_config": root / _SELECTED_CONFIG,
        "fold_assignments": root / _FOLDS,
        "baseline_summary": root / _BASELINE_SUMMARY,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reviewed input artifacts: {missing}")
    hashes = {
        f"{name}_sha256": compute_sha256(path) for name, path in paths.items()
    }
    selected = json.loads(paths["selected_config"].read_text(encoding="utf-8"))
    selection_summary = json.loads(
        paths["selection_summary"].read_text(encoding="utf-8")
    )
    baseline_summary = json.loads(
        paths["baseline_summary"].read_text(encoding="utf-8")
    )
    candidates = pd.read_csv(paths["selection_candidates"])
    if len(candidates) != 36 or not bool(selection_summary.get("all_checks_passed")):
        raise JointDecoderError("The reviewed Step 6R selection did not pass completely.")
    selected_identity = (
        str(selected.get("state_name")),
        int(selected.get("neural_width_ms", -1)),
        int(selected.get("interval_end_lag_ms", -1)),
    )
    if selected_identity != ("PVA", 50, 50):
        raise JointDecoderError(f"Unexpected selected configuration {selected_identity}.")
    summary_selected = selection_summary.get("selected", {})
    if (
        str(summary_selected.get("state_name")),
        int(summary_selected.get("neural_width_ms", -1)),
        int(summary_selected.get("interval_end_lag_ms", -1)),
    ) != selected_identity:
        raise JointDecoderError("Selection summary and selected configuration disagree.")
    passing = candidates.loc[candidates["passed"].astype(bool)]
    winner = passing.sort_values(
        [
            "out_of_fold_r2_mean",
            "state_dimension",
            "neural_width_ms",
            "interval_end_lag_ms",
            "state_name",
        ],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    if (
        str(winner["state_name"]),
        int(winner["neural_width_ms"]),
        int(winner["interval_end_lag_ms"]),
    ) != selected_identity:
        raise JointDecoderError("The candidate table does not reproduce the frozen winner.")

    session = load_session(config)
    evidence = selected.get("selection_evidence", {})
    selection_provenance = selection_summary.get("provenance", {})
    if (
        session.identity.sha256 != str(evidence.get("recording_sha256"))
        or session.identity.sha256
        != str(selection_provenance.get("recording", {}).get("sha256"))
        or session.identity.sha256
        != str(baseline_summary.get("recording", {}).get("sha256"))
    ):
        raise JointDecoderError("Recording hashes disagree across frozen artifacts.")

    eligible_training = tuple(
        session.trials.loc[
            (session.trials["split"] == TRAIN) & session.trials["eligible"],
            "trial_row",
        ]
        .sort_values()
        .to_numpy(dtype=np.int64)
        .tolist()
    )
    training_bounds = build_trial_bounds(
        session.behavior_timestamps,
        session.trials,
        eligible_training,
        required_split=TRAIN,
    )
    fold_frame = pd.read_csv(paths["fold_assignments"])
    folds = fold_assignments_from_frame(
        fold_frame,
        eligible_training_rows=eligible_training,
        source_sha256=hashes["fold_assignments_sha256"],
    )
    if (
        folds.source_sha256 != str(evidence.get("fold_assignments_sha256"))
        or folds.source_sha256
        != str(
            selection_provenance.get("fold_assignments", {}).get("sha256")
        )
    ):
        raise JointDecoderError("Tracked fold-assignment hashes disagree.")
    training_ledger = build_official_target_ledger(
        None,
        timestamps=session.behavior_timestamps,
        velocity=session.cursor_velocity,
        bounds=training_bounds,
        source_sha256=hashes["baseline_summary_sha256"],
    )
    if (
        training_ledger.identity_sha256
        != str(evidence.get("training_target_identity_sha256"))
        or training_ledger.identity_sha256
        != str(
            selection_provenance.get("official_training_targets", {}).get(
                "identity_sha256"
            )
        )
    ):
        raise JointDecoderError("Official training-target identity hashes disagree.")
    if hashes["baseline_summary_sha256"] != str(
        selection_provenance.get("official_training_targets", {}).get(
            "source_sha256"
        )
    ):
        raise JointDecoderError("Step 5 summary hash differs from Step 6R provenance.")
    if (
        int(baseline_summary.get("samples", {}).get("validation_samples", -1))
        != 8_183
        or int(baseline_summary.get("samples", {}).get("validation_trials", -1))
        != 24
    ):
        raise JointDecoderError("CA-05 does not declare 8,183 targets from 24 trials.")
    return VerifiedInputs(
        session=session,
        training_bounds=training_bounds,
        folds=folds,
        training_ledger=training_ledger,
        selected_config=selected,
        selection_summary=selection_summary,
        baseline_summary=baseline_summary,
        hashes=hashes,
    )


def _training_only_reload_check(
    session: SessionData,
    bounds: Mapping[int, TrialBounds],
    schedule,
    model: DecoderModel,
    reloaded_model: DecoderModel,
    reloaded_unit_ids: tuple[int, ...],
) -> None:
    expected_ids = tuple(unit.unit_id for unit in session.units)
    if reloaded_unit_ids != expected_ids:
        raise JointDecoderError("Reloaded model unit order differs from the session.")
    row = min(bounds)
    trial = bounds[row]
    assessment = build_assessment_trial(
        schedule,
        trial,
        initial_position=session.cursor_position[trial.first_index],
    )
    original = filter_trial(model, assessment)
    reloaded = filter_trial(reloaded_model, assessment)
    if (
        not np.allclose(
            original.velocity, reloaded.velocity, atol=1e-12, rtol=1e-12
        )
        or not np.allclose(
            original.velocity_covariance,
            reloaded.velocity_covariance,
            atol=1e-12,
            rtol=1e-12,
        )
    ):
        raise JointDecoderError("Serialized model does not exactly replay training data.")


def _load_validation_evidence(
    root: Path,
    session: SessionData,
) -> tuple[
    ValidationIdentity,
    ValidationScoringLedger,
    dict[int, TrialBounds],
]:
    path = root / _BASELINE_PREDICTIONS
    if not path.is_file():
        raise FileNotFoundError(f"Missing CA-05 validation predictions: {path}")
    eligible_validation = tuple(
        session.trials.loc[
            (session.trials["split"] == VALIDATION) & session.trials["eligible"],
            "trial_row",
        ]
        .sort_values()
        .to_numpy(dtype=np.int64)
        .tolist()
    )
    bounds = build_trial_bounds(
        session.behavior_timestamps,
        session.trials,
        eligible_validation,
        required_split=VALIDATION,
    )
    manifest = pd.read_csv(
        path,
        usecols=[
            "sample_index",
            "trial_row",
            "reference_index",
            "target_index",
            "target_time",
            "target_velocity_x",
            "target_velocity_y",
        ],
    )
    identity, scoring = build_validation_ledgers(
        manifest,
        timestamps=session.behavior_timestamps,
        velocity=session.cursor_velocity,
        bounds=bounds,
        source_sha256=compute_sha256(path),
    )
    return identity, scoring, bounds


def _validation_trial_ids(
    session: SessionData,
    bounds: Mapping[int, TrialBounds],
) -> dict[int, int]:
    """Map validation rows to their NWB trial IDs."""
    table = session.trials.set_index("trial_row")
    mapping = {
        int(row): int(table.at[int(row), "trial_id"])
        for row in sorted(bounds)
    }
    if len(mapping) != 24 or len(set(mapping.values())) != 24:
        raise JointDecoderError("Validation trial IDs are incomplete or duplicated.")
    return mapping


def _write_predictions(
    path: Path,
    identity: ValidationIdentity,
    scoring: ValidationScoringLedger,
    result: FinalFilterResult,
    *,
    validation_trial_ids: Mapping[int, int],
) -> None:
    covariance = result.velocity_covariance
    frame = pd.DataFrame(
        {
            "sample_index": identity.sample_index,
            "trial_row": identity.trial_row,
            "trial_id": np.asarray(
                [validation_trial_ids[int(row)] for row in identity.trial_row],
                dtype=np.int64,
            ),
            "reference_index": identity.reference_index,
            "target_index": identity.target_index,
            "target_time": identity.target_time,
            "target_velocity_x": scoring.velocity[:, 0],
            "target_velocity_y": scoring.velocity[:, 1],
            "predicted_velocity_x": result.prediction[:, 0],
            "predicted_velocity_y": result.prediction[:, 1],
            "posterior_variance_x": covariance[:, 0, 0],
            "posterior_covariance_xy": covariance[:, 0, 1],
            "posterior_variance_y": covariance[:, 1, 1],
        }
    )
    frame.to_csv(path, index=False, lineterminator="\n")


def _validation_diagnostics(
    model: DecoderModel,
    result: FinalFilterResult,
    scoring: ValidationScoringLedger,
) -> dict[str, object]:
    innovation = innovation_diagnostics(
        model,
        result.innovations,
        result.predicted_covariance_at_updates,
        result.innovation_trial_row,
    )
    innovation_covariance_lower_bound = float(
        np.linalg.eigvalsh(
            (
                model.observation.residual_covariance
                + model.observation.residual_covariance.T
            )
            * 0.5
        ).min()
    )
    innovation["innovation_covariance_min_eigenvalue_lower_bound"] = (
        innovation_covariance_lower_bound
    )
    calibration = evaluate_calibration(
        scoring.velocity,
        result.prediction,
        result.velocity_covariance,
    )
    prediction_scale = result.prediction.std(axis=0) / scoring.velocity.std(axis=0)
    return {
        "process": model.process.diagnostics,
        "observation_fit": model.observation.diagnostics,
        "innovation": innovation,
        "calibration": calibration,
        "prediction_to_target_std": {
            "x": float(prediction_scale[0]),
            "y": float(prediction_scale[1]),
        },
        "recursion": {
            "posterior_covariance_rank_min": result.posterior_min_rank,
            "posterior_covariance_min_eigenvalue": result.posterior_min_eigenvalue,
            "posterior_covariance_max_eigenvalue": result.posterior_max_eigenvalue,
            "kalman_gain_absolute_max": result.max_abs_gain,
            "normalized_state_absolute_max": result.max_abs_normalized_state,
            "posterior_covariance_absolute_max": (
                result.max_abs_posterior_covariance
            ),
        },
    }


def _official_checks(
    *,
    inputs: VerifiedInputs,
    model: DecoderModel,
    reloaded_unit_ids: tuple[int, ...],
    validation_bounds: Mapping[int, TrialBounds],
    validation_trial_ids: Mapping[int, int],
    validation_identity: ValidationIdentity,
    result: FinalFilterResult,
    reload_exact: bool,
    predictions_path: Path,
    metrics: dict[str, object],
) -> list[dict[str, object]]:
    validation_rows = set(validation_bounds)
    stored = pd.read_csv(predictions_path)
    stored_trial_ids = stored["trial_id"].to_numpy(dtype=np.int64)
    expected_trial_ids = np.asarray(
        [validation_trial_ids[int(row)] for row in validation_identity.trial_row],
        dtype=np.int64,
    )
    stored_prediction = stored[
        ["predicted_velocity_x", "predicted_velocity_y"]
    ].to_numpy(dtype=np.float64)
    stored_truth = stored[
        ["target_velocity_x", "target_velocity_y"]
    ].to_numpy(dtype=np.float64)
    reproduced_r2 = r2_per_axis(stored_truth, stored_prediction)
    innovation_covariance_lower_bound = float(
        np.linalg.eigvalsh(
            (
                model.observation.residual_covariance
                + model.observation.residual_covariance.T
            )
            * 0.5
        ).min()
    )
    checks = [
        _check(
            "the reviewed frozen configuration was used unchanged",
            model.spec == FROZEN_SPEC,
            "PVA / 50 ms width / 50 ms lag / 10 ms cadence",
        ),
        _check(
            "all final model parameters were fitted on eligible training trials",
            set(model.process.training_rows) == set(inputs.training_bounds)
            and set(model.observation.training_rows) == set(inputs.training_bounds)
            and len(model.process.training_rows) == 123,
            "state, process, prior, count, observation, and covariance fits use 123 training trials",
        ),
        _check(
            "validation identities exactly match CA-05",
            validation_identity.n_samples == 8_183
            and np.unique(validation_identity.trial_row).size == 24,
            f"8,183 rows; identity SHA-256 {validation_identity.identity_sha256}",
        ),
        _check(
            "only eligible validation trials were filtered",
            len(validation_rows) == 24
            and validation_rows == set(validation_trial_ids),
            "24 validation trials; no held-out trial bounds were constructed",
        ),
        _check(
            "validation trial IDs match the frozen recording",
            np.array_equal(stored_trial_ids, expected_trial_ids)
            and np.unique(stored_trial_ids).size == 24,
            "trial IDs are mapped from the same 24 CA-05 trial rows",
        ),
        _check(
            "every validation target received one finite prediction",
            np.all(result.fill_count == 1)
            and np.all(np.isfinite(result.prediction)),
            "8,183 predictions in canonical order",
        ),
        _check(
            "posterior and innovation covariances passed numerical gates",
            result.posterior_min_eigenvalue >= -1e-12
            and np.all(np.isfinite(result.velocity_covariance))
            and np.all(
                np.isfinite(result.predicted_covariance_at_updates)
            )
            and innovation_covariance_lower_bound > 0.0,
            f"minimum posterior eigenvalue {result.posterior_min_eigenvalue:.3e}; "
            "innovation minimum-eigenvalue lower bound "
            f"{innovation_covariance_lower_bound:.3e}",
        ),
        _check(
            "saved model reload reproduces validation predictions",
            reload_exact,
            "prediction and velocity-covariance arrays match within 1e-12",
        ),
        _check(
            "saved model preserves the frozen 53-unit order",
            reloaded_unit_ids
            == tuple(unit.unit_id for unit in inputs.session.units),
            "53 unit IDs in production loader order",
        ),
        _check(
            "saved prediction artifact reproduces reported R2",
            len(stored) == 8_183
            and np.allclose(
                stored_prediction,
                result.prediction,
                atol=1e-12,
                rtol=1e-12,
            )
            and np.allclose(
                reproduced_r2,
                [metrics["r2"]["x"], metrics["r2"]["y"]],
                atol=1e-12,
                rtol=0.0,
            ),
            "stored predictions and true velocities reproduce both axes",
        ),
    ]
    return checks


def _official_configuration(
    *,
    inputs: VerifiedInputs,
    validation_identity: ValidationIdentity,
    scoring_ledger: ValidationScoringLedger,
    model_hash: str,
    predictions_hash: str,
) -> dict[str, object]:
    return {
        "configuration": inputs.selected_config,
        "source": {
            "selected_config_path": _SELECTED_CONFIG.as_posix(),
            "selected_config_sha256": inputs.hashes["selected_config_sha256"],
            "selection_summary_sha256": inputs.hashes["selection_summary_sha256"],
            "selection_candidates_sha256": inputs.hashes[
                "selection_candidates_sha256"
            ],
        },
        "final_fit": {
            "training_trials": 123,
            "validation_trials": 24,
            "validation_targets": 8_183,
            "validation_identity_sha256": validation_identity.identity_sha256,
            "validation_truth_sha256": scoring_ledger.truth_sha256,
            "validation_source_sha256": validation_identity.source_sha256,
            "held_out": "not constructed or filtered",
        },
        "artifacts": {
            "model_sha256": model_hash,
            "validation_predictions_sha256": predictions_hash,
        },
    }


def _official_summary(
    *,
    inputs: VerifiedInputs,
    validation_identity: ValidationIdentity,
    scoring_ledger: ValidationScoringLedger,
    metrics: dict[str, object],
    diagnostics: dict[str, object],
    checks: list[dict[str, object]],
    accepted: bool,
    artifacts: dict[str, str],
    artifact_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "procedure": {
            "name": "one official validation of the frozen clean-room Kalman decoder",
            "configuration": "PVA / 50 ms width / 50 ms interval-end lag",
            "training": "all 123 eligible training trials",
            "validation": "one frozen run on the 8,183 CA-05 validation targets",
            "selection_or_tuning": "none",
            "acceptance": "structural gates only; performance and calibration are interpretive",
        },
        "provenance": {
            "recording_sha256": inputs.session.identity.sha256,
            "fold_assignments_sha256": inputs.folds.source_sha256,
            "training_target_identity_sha256": inputs.training_ledger.identity_sha256,
            "validation_identity_sha256": validation_identity.identity_sha256,
            "validation_truth_sha256": scoring_ledger.truth_sha256,
            "validation_source_sha256": validation_identity.source_sha256,
            **inputs.hashes,
        },
        "validation_metrics": metrics,
        "diagnostics": diagnostics,
        "checks": checks,
        "all_checks_passed": accepted,
        "accepted_as_official_kalman_result": accepted,
        "held_out": "not constructed, filtered, or scored",
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
    }


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _print_official_result(
    metrics: dict[str, object],
    diagnostics: dict[str, object],
    checks: list[dict[str, object]],
    accepted: bool,
) -> None:
    print("Official frozen Kalman validation")
    print(
        f"  R2    x {float(metrics['r2']['x']):.6f}  "
        f"y {float(metrics['r2']['y']):.6f}  "
        f"mean {float(metrics['r2_mean']):.6f}"
    )
    print(
        f"  r     mean {float(metrics['pearson_r_mean']):.6f}  "
        f"RMSE mean {float(metrics['rmse_cm_s_mean']):.6f} cm/s"
    )
    calibration = diagnostics["calibration"]
    print(
        f"  NEES  mean {float(calibration['velocity_nees_mean']):.6f}  "
        f"95% coverage {float(calibration['velocity_95_ellipse_coverage']):.2%}"
    )
    print("\nChecks")
    width = max(len(str(check["name"])) for check in checks)
    for check in checks:
        status = "pass" if check["passed"] else "FAIL"
        print(f"  {str(check['name']):<{width}}  {status:<4}  {check['detail']}")
    print(f"\n{'ACCEPTED' if accepted else 'NOT ACCEPTED'}")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
