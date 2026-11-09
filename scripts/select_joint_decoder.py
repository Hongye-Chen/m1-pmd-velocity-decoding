"""Select the causal Kalman decoder configuration from training trials.

Run from the repository root with the project environment active::

    python -m scripts.select_joint_decoder

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neural_decoding.config import NWB_PATH_ENV_VAR, ConfigError, load_config, project_root
from neural_decoding.dataset import DatasetError, compute_sha256, load_session
from neural_decoding.joint_decoder import (
    JointDecoderError,
    build_official_target_ledger,
    build_trial_bounds,
    fold_assignments_from_frame,
)
from neural_decoding.joint_decoder_selection import SelectionRun, run_selection
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.splits import TRAIN

logger = get_logger("select_joint_decoder")

_FOLD_PATH = Path("results/selection/fold_assignments.csv")
_TARGET_SOURCE_PATH = Path("results/baseline_summary.json")
_OUTPUT_DIRECTORY = Path("results/internal/kalman_decoder_selection")


def main(argv: list[str] | None = None) -> int:
    if argv:
        raise SystemExit("This frozen workflow accepts no command-line overrides.")
    config = load_config()
    configure_logging(config.run.log_level)
    root = project_root()
    fold_path = root / _FOLD_PATH
    target_path = root / _TARGET_SOURCE_PATH

    try:
        session = load_session(config)
        eligible_rows = tuple(
            session.trials.loc[
                (session.trials["split"] == TRAIN) & session.trials["eligible"],
                "trial_row",
            ]
            .sort_values()
            .to_numpy(dtype=np.int64)
            .tolist()
        )
        if len(eligible_rows) != 123:
            raise JointDecoderError(
                f"Expected 123 eligible training trials, got {len(eligible_rows)}."
            )
        bounds = build_trial_bounds(
            session.behavior_timestamps, session.trials, eligible_rows
        )
        folds = _load_folds(fold_path, eligible_rows)
        ledger = _load_targets(target_path, session, bounds)
    except (ConfigError, DatasetError, FileNotFoundError) as error:
        print(f"Cannot run joint decoder selection: {error}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, and retain the tracked Step 4/Step 5 inputs."
        )
        return 2
    except (JointDecoderError, ValueError, pd.errors.ParserError) as error:
        print(f"Cannot run joint decoder selection: {error}")
        return 1

    try:
        logger.info("Running 36 candidates and 180 fold fits")
        first = run_selection(
            session,
            bounds=bounds,
            folds=folds,
            ledger=ledger,
            run_label="determinism pass 1/2",
        )
        logger.info("Repeating the complete workflow for deterministic reproduction")
        second = run_selection(
            session,
            bounds=bounds,
            folds=folds,
            ledger=ledger,
            run_label="determinism pass 2/2",
        )
    except (JointDecoderError, np.linalg.LinAlgError, ValueError) as error:
        print(f"Cannot run joint decoder selection: {error}")
        return 1

    deterministic = _canonical_run(first) == _canonical_run(second)
    checks = [
        *first.checks,
        {
            "name": "a complete repeat reproduced the in-memory selection exactly",
            "passed": deterministic,
            "detail": (
                "all 36 candidate rows, compact diagnostics, checks, and selected "
                "identity matched exactly"
                if deterministic
                else "the repeated workflow differed; no configuration was selected"
            ),
        },
    ]
    all_checks_passed = all(bool(check["passed"]) for check in checks)
    selected = first.selected if deterministic and all_checks_passed else None

    output_directory = root / _OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    candidate_path = output_directory / "candidates.csv"
    summary_path = output_directory / "summary.json"
    selected_path = output_directory / "selected_config.json"

    candidate_frame = pd.DataFrame(first.candidates)
    if len(candidate_frame) != 36:
        print("Cannot persist selection: candidate table does not have exactly 36 rows.")
        return 1
    candidate_frame.to_csv(candidate_path, index=False, lineterminator="\n")

    artifacts = {
        "candidates": _repo_relative(candidate_path),
        "summary": _repo_relative(summary_path),
        "selected_config": (
            _repo_relative(selected_path) if selected is not None else None
        ),
    }
    summary = _build_summary(
        session=session,
        folds=folds,
        ledger=ledger,
        run=first,
        selected=selected,
        checks=checks,
        artifacts=artifacts,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if selected is None:
        selected_path.unlink(missing_ok=True)
    else:
        selected_path.write_text(
            json.dumps(
                _selected_configuration(selected, session.identity.sha256, folds, ledger),
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    _print_report(first, selected, checks, artifacts)
    return 0 if selected is not None else 1


def _load_folds(path: Path, eligible_rows: tuple[int, ...]):
    if not path.is_file():
        raise FileNotFoundError(f"Tracked fold assignments are missing: {path}")
    source_sha256 = compute_sha256(path)
    frame = pd.read_csv(path)
    return fold_assignments_from_frame(
        frame,
        eligible_training_rows=eligible_rows,
        source_sha256=source_sha256,
    )


def _load_targets(path: Path, session, bounds):
    if not path.is_file():
        raise FileNotFoundError(f"Official Step 5 summary is missing: {path}")
    source_sha256 = compute_sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("samples", {}).get("training_samples", -1)) != 43_022
        or int(payload.get("configuration", {}).get("history_ms", -1)) != 200
        or int(payload.get("configuration", {}).get("lag_ms", -1)) != 50
        or int(payload.get("configuration", {}).get("eligibility_lag_ms", -1)) != 50
    ):
        raise JointDecoderError(
            "The Step 5 summary does not declare the frozen 43,022-row, "
            "200 ms history, 50 ms lag training target protocol."
        )
    return build_official_target_ledger(
        None,
        timestamps=session.behavior_timestamps,
        velocity=session.cursor_velocity,
        bounds=bounds,
        source_sha256=source_sha256,
    )


def _build_summary(
    *,
    session,
    folds,
    ledger,
    run: SelectionRun,
    selected: dict[str, object] | None,
    checks: list[dict[str, object]],
    artifacts: dict[str, str | None],
) -> dict[str, object]:
    passing = [
        record for record in run.candidates if bool(record.get("passed", False))
    ]
    ranking = sorted(
        passing,
        key=lambda record: (
            -float(record["out_of_fold_r2_mean"]),
            int(record["state_dimension"]),
            int(record["neural_width_ms"]),
            int(record["interval_end_lag_ms"]),
            str(record["state_name"]),
        ),
    )[:5]
    return {
        "procedure": {
            "name": "clean-room joint causal decoder selection",
            "split_used": "eligible training trials only",
            "state_grid": ["V", "PV", "VA", "PVA"],
            "neural_width_grid_ms": [10, 20, 50],
            "interval_end_lag_grid_ms": [0, 50, 100],
            "state_and_output_cadence_ms": 10,
            "folds": 5,
            "temporary_fits_per_run": 180,
            "transition": "full affine least squares in fold-normalized state",
            "process_covariance": (
                "full sample residual covariance (n-1 divisor); only eigenvalues "
                "within 256 * machine-epsilon * dimension * max spectral scale "
                "below zero are projected to zero; legitimate rank deficiency retained"
            ),
            "observation": "affine count model in fold-normalized count coordinates",
            "observation_covariance": (
                "full Oracle Approximating Shrinkage covariance toward the "
                "scaled-identity target; analytic strength, not tuned"
            ),
            "recursion": (
                "stable low-rank information update algebraically equivalent to "
                "the Kalman update; no explicit matrix inverse"
            ),
            "selection_metric": (
                "arithmetic mean of horizontal and vertical R2 on concatenated "
                "out-of-fold predictions"
            ),
            "tie_breaking": [
                "highest finite concatenated mean R2",
                "within 1e-12, lower state dimension",
                "then narrower neural width",
                "then smaller interval-end lag",
                "then lexical state name",
            ],
            "whiteness_and_calibration": "reported diagnostics, not hard gates",
        },
        "provenance": {
            "recording": {
                "name": session.identity.name,
                "sha256": session.identity.sha256,
            },
            "fold_assignments": {
                "path": _FOLD_PATH.as_posix(),
                "sha256": folds.source_sha256,
                "group_sizes": [len(group) for group in folds.groups],
            },
            "official_training_targets": {
                "reconstructed_from": (
                    "the frozen Step 5 200 ms history plus selected 50 ms lag "
                    "eligibility rule"
                ),
                "step5_summary_path": _TARGET_SOURCE_PATH.as_posix(),
                "source_sha256": ledger.source_sha256,
                "identity_sha256": ledger.identity_sha256,
                "samples": ledger.n_samples,
                "trials": int(np.unique(ledger.trial_row).size),
            },
            "validation_stream": "not constructed, filtered, or scored",
            "held_out_stream": "not constructed, filtered, or scored",
        },
        "grid": {
            "candidates_expected": 36,
            "candidates_recorded": len(run.candidates),
            "candidates_passing_integrity": len(passing),
        },
        "selected": (
            None
            if selected is None
            else {
                "state_name": selected["state_name"],
                "state_dimension": selected["state_dimension"],
                "neural_width_ms": selected["neural_width_ms"],
                "interval_end_lag_ms": selected["interval_end_lag_ms"],
                "out_of_fold_r2_x": selected["out_of_fold_r2_x"],
                "out_of_fold_r2_y": selected["out_of_fold_r2_y"],
                "out_of_fold_r2_mean": selected["out_of_fold_r2_mean"],
                "pearson_r_mean": selected["pearson_r_mean"],
                "rmse_cm_s_mean": selected["rmse_cm_s_mean"],
                "compact_diagnostics": {
                    "process_residual_autocorrelation_abs_max": selected[
                        "process_residual_autocorrelation_abs_max"
                    ],
                    "innovation_whitened_autocorrelation_abs_max": selected[
                        "innovation_whitened_autocorrelation_abs_max"
                    ],
                    "process_covariance_rank_min": selected[
                        "process_covariance_rank_min"
                    ],
                    "observation_covariance_min_eigenvalue_min": selected[
                        "observation_covariance_min_eigenvalue_min"
                    ],
                    "posterior_covariance_min_eigenvalue_min": selected[
                        "posterior_covariance_min_eigenvalue_min"
                    ],
                    "velocity_nees_mean": selected["velocity_nees_mean"],
                    "velocity_95_ellipse_coverage": selected[
                        "velocity_95_ellipse_coverage"
                    ],
                    "kalman_gain_absolute_max": selected[
                        "kalman_gain_absolute_max"
                    ],
                },
            }
        ),
        "ranking_preview": [
            {
                "rank": index,
                "state_name": record["state_name"],
                "neural_width_ms": record["neural_width_ms"],
                "interval_end_lag_ms": record["interval_end_lag_ms"],
                "out_of_fold_r2_mean": record["out_of_fold_r2_mean"],
            }
            for index, record in enumerate(ranking, start=1)
        ],
        "checks": checks,
        "all_checks_passed": all(bool(check["passed"]) for check in checks),
        "artifacts": artifacts,
    }


def _selected_configuration(
    selected: dict[str, object],
    recording_sha256: str,
    folds,
    ledger,
) -> dict[str, object]:
    return {
        "state_name": selected["state_name"],
        "state_components": {
            "V": ["vx", "vy"],
            "PV": ["px", "py", "vx", "vy"],
            "VA": ["vx", "vy", "ax", "ay"],
            "PVA": ["px", "py", "vx", "vy", "ax", "ay"],
        }[str(selected["state_name"])],
        "neural_width_ms": selected["neural_width_ms"],
        "interval_end_lag_ms": selected["interval_end_lag_ms"],
        "state_and_output_cadence_ms": 10,
        "timing_convention": (
            "at state timestamp t consume one fresh 53-unit count from "
            "[t-L-W, t-L); tile intervals from the first behavior-grid point, "
            "update every W/10 steps, and predict only between updates"
        ),
        "transition_estimator": "full affine stable least squares",
        "normalization": {
            "state": (
                "fold-training population mean and standard deviation (ddof=0); "
                "reject a component below 1e-8"
            ),
            "neural_counts": (
                "fold-training population mean and standard deviation (ddof=0); "
                "replace a scale below 1e-8 with 1"
            ),
        },
        "process_covariance_estimator": {
            "method": "full fold-training transition-residual sample covariance",
            "divisor": "n_residuals - 1",
            "symmetrization": "(Q + Q.T) / 2",
            "negative_eigenvalue_policy": (
                "raise below -tol; project values in [-tol, 0) to zero"
            ),
            "tolerance": (
                "256 * machine_epsilon * matrix_dimension * "
                "max(max_abs_eigenvalue, 1)"
            ),
            "rank_deficiency": "preserved",
        },
        "observation_estimator": "full affine stable least squares",
        "observation_covariance_estimator": {
            "method": "Oracle Approximating Shrinkage",
            "implementation": "sklearn.covariance.oas",
            "assume_centered_residuals": True,
            "target": "scaled identity",
            "shrinkage_strength": "analytic fold-training estimate, not tuned",
            "requirement": "finite, symmetric, and positive definite",
        },
        "covariance_recursion": {
            "measurement_update": (
                "stable low-rank information form algebraically equivalent to "
                "the Kalman update"
            ),
            "explicit_matrix_inverse": False,
            "posterior_policy": (
                "symmetrize; reject eigenvalues below the process-covariance "
                "tolerance formula; project only roundoff negatives"
            ),
        },
        "initialization": {
            "position": (
                "recorded once at trial start for PV/PVA; unavailable thereafter"
            ),
            "velocity_and_acceleration": (
                "fold-training initial prior; recorded assessment values never supplied"
            ),
            "known_position_cross_covariance": 0.0,
            "prior_covariance_regularization": (
                "fixed 1e-6 times max(mean diagonal covariance, 1)"
            ),
            "reset_each_trial": True,
        },
        "fit_instructions": (
            "Refit every normalization, prior, transition, covariance, and "
            "observation map on all eligible training trials before one official "
            "validation run; do not reuse cross-validation parameters."
        ),
        "selection_evidence": {
            "out_of_fold_r2_x": selected["out_of_fold_r2_x"],
            "out_of_fold_r2_y": selected["out_of_fold_r2_y"],
            "out_of_fold_r2_mean": selected["out_of_fold_r2_mean"],
            "recording_sha256": recording_sha256,
            "fold_assignments_sha256": folds.source_sha256,
            "training_target_identity_sha256": ledger.identity_sha256,
        },
    }


def _canonical_run(run: SelectionRun) -> str:
    return json.dumps(
        {
            "candidates": run.candidates,
            "selected": run.selected,
            "checks": run.checks,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _print_report(
    run: SelectionRun,
    selected: dict[str, object] | None,
    checks: list[dict[str, object]],
    artifacts: dict[str, str | None],
) -> None:
    passing = sum(bool(record.get("passed", False)) for record in run.candidates)
    print(f"Candidates  {len(run.candidates)} evaluated, {passing} passed integrity gates")
    if selected is None:
        print("Selected    none")
    else:
        print(
            f"Selected    {selected['state_name']}, "
            f"{selected['neural_width_ms']} ms width, "
            f"{selected['interval_end_lag_ms']} ms lag"
        )
        print(
            f"OOF mean R2 {float(selected['out_of_fold_r2_mean']):.6f} "
            f"(x {float(selected['out_of_fold_r2_x']):.6f}, "
            f"y {float(selected['out_of_fold_r2_y']):.6f})"
        )
    print("\nChecks")
    width = max(len(str(check["name"])) for check in checks)
    for check in checks:
        status = "pass" if check["passed"] else "FAIL"
        print(f"  {str(check['name']):<{width}}  {status:<4}  {check['detail']}")
    print("\nArtifacts")
    for path in artifacts.values():
        if path is not None:
            print(f"  {path}")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
