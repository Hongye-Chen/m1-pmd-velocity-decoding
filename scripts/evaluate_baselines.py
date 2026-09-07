"""Fit the official constant and Ridge baselines and evaluate them once on validation.

Run from the repository root with the project environment active::

    python -m scripts.evaluate_baselines

The command reads the lag and alpha frozen by the selection step, builds training and
validation features at that lag, fits normalization once on the complete training
set, fits a fresh constant baseline and a fresh Ridge model, and scores each once on
validation. Nothing here is tuned: the configuration is read from disk rather than
chosen, and the held-out split is not touched.

Target eligibility uses the selected lag, the frozen rule for this and every later
step, so all reported models are evaluated on identical validation samples.

Exit codes: 0 when every check passes, 1 when a check fails, and 2 when the
recording or the selected configuration is not available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neural_decoding.config import (
    NWB_PATH_ENV_VAR,
    ConfigError,
    load_config,
    project_root,
)
from neural_decoding.dataset import DatasetError, SessionData, load_session
from neural_decoding.features import (
    FeatureError,
    FeatureSet,
    FeatureSpec,
    build_split_features,
)
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.metrics import MetricError, r2_per_axis, summarize_predictions
from neural_decoding.models import ConstantBaseline, ModelError, RidgeDecoder
from neural_decoding.selection import RIDGE_ALPHAS
from neural_decoding.splits import TEST, TRAIN, VALIDATION

logger = get_logger("evaluate_baselines")

AXIS_NAMES = ("x", "y")

_CONSISTENCY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Check:
    """One runtime-validation check against the baseline run."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SelectedConfiguration:
    """The lag and alpha frozen by the selection step, read rather than chosen."""

    lag_ms: int
    ridge_alpha: float
    recording_sha256: str
    lag_grid_ms: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    source: str

    @classmethod
    def read(cls, path: Path) -> SelectedConfiguration:
        if not path.is_file():
            raise ConfigError(
                f"The selected configuration is missing at {path}. Run "
                "`python -m scripts.select_lag_alpha` first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in ("lag_ms", "ridge_alpha") if key not in payload]
        if missing:
            raise ConfigError(f"{path} is missing required keys: {missing}")
        return cls(
            lag_ms=int(payload["lag_ms"]),
            ridge_alpha=float(payload["ridge_alpha"]),
            recording_sha256=str(payload.get("recording_sha256", "")),
            lag_grid_ms=tuple(int(lag) for lag in payload.get("lag_grid_ms", ())),
            alpha_grid=tuple(float(a) for a in payload.get("alpha_grid", ())),
            source=path.name,
        )


def run_checks(
    session: SessionData,
    selection: SelectedConfiguration,
    training: FeatureSet,
    validation: FeatureSet,
    baseline: ConstantBaseline,
    decoder: RidgeDecoder,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, dict[str, object]],
    written_predictions: Path,
    written_models: dict[str, Path],
) -> list[Check]:
    """Validate that the official numbers were produced by the official workflow."""
    checks: list[Check] = []
    trials = session.trials
    eligible_training = set(
        trials.loc[(trials["split"] == TRAIN) & trials["eligible"], "trial_row"].tolist()
    )
    eligible_validation = set(
        trials.loc[
            (trials["split"] == VALIDATION) & trials["eligible"], "trial_row"
        ].tolist()
    )
    held_out_rows = set(trials.loc[trials["split"] == TEST, "trial_row"].tolist())
    training_rows = set(training.trial_rows.tolist())
    validation_rows = set(validation.trial_rows.tolist())

    checks.append(
        Check(
            "the configuration was read from the selection step, not chosen here",
            selection.lag_ms in selection.lag_grid_ms
            and selection.ridge_alpha in selection.alpha_grid
            and selection.ridge_alpha in RIDGE_ALPHAS,
            f"lag {selection.lag_ms} ms and alpha {selection.ridge_alpha:g} read from "
            f"{selection.source}, both inside the frozen grid",
        )
    )
    checks.append(
        Check(
            "the recording is the one the configuration was selected on",
            selection.recording_sha256 == session.identity.sha256,
            f"checksum {session.identity.sha256[:12]}... matches"
            if selection.recording_sha256 == session.identity.sha256
            else "checksum differs from the selected configuration",
        )
    )
    checks.append(
        Check(
            "both splits were built at the selected lag",
            training.lag_ms == selection.lag_ms and validation.lag_ms == selection.lag_ms,
            f"lag {selection.lag_ms} ms for training and validation",
        )
    )
    checks.append(
        Check(
            "target eligibility uses the selected lag",
            training.eligibility_lag_ms == selection.lag_ms
            and validation.eligibility_lag_ms == selection.lag_ms,
            f"eligibility anchored on {selection.lag_ms} ms, the frozen rule for "
            "every reported model",
        )
    )

    checks.append(
        Check(
            "every eligible training trial was fitted and no other",
            training_rows == eligible_training,
            f"{len(training_rows)} of {len(eligible_training)} eligible training trials",
        )
    )
    checks.append(
        Check(
            "every eligible validation trial was evaluated and no other",
            validation_rows == eligible_validation,
            f"{len(validation_rows)} of {len(eligible_validation)} eligible "
            "validation trials",
        )
    )
    checks.append(
        Check(
            "the training and validation samples are disjoint",
            training_rows.isdisjoint(validation_rows),
            f"{len(training_rows)} training and {len(validation_rows)} validation "
            "trials share no trial",
        )
    )
    checks.append(
        Check(
            "the held-out split was not read",
            held_out_rows.isdisjoint(training_rows | validation_rows),
            f"all {len(held_out_rows)} assigned held-out trial rows remain untouched",
        )
    )

    normalization = decoder.normalization
    checks.append(
        Check(
            "normalization was fitted once on the complete training set",
            normalization.fitted_on == TRAIN
            and normalization.n_samples == training.n_samples
            and set(normalization.trial_rows) == eligible_training,
            f"fitted on {normalization.fitted_on!r} over {normalization.n_samples} "
            f"samples from {len(normalization.trial_rows)} trials",
        )
    )

    standardized_validation = normalization.apply_features(validation.counts)
    residual_mean = float(np.max(np.abs(standardized_validation.mean(axis=0))))
    checks.append(
        Check(
            "validation was standardized with the training statistics",
            residual_mean > _CONSISTENCY_TOLERANCE,
            f"standardized validation features retain a mean offset of up to "
            f"{residual_mean:.4f}, so they were not re-centred on themselves",
        )
    )

    training_mean = training.targets.mean(axis=0)
    checks.append(
        Check(
            "the constant baseline is the mean of the complete training targets",
            bool(np.allclose(baseline.velocity, training_mean, atol=0.0, rtol=1e-12))
            and baseline.fitted_on == TRAIN,
            "baseline velocity ("
            + ", ".join(f"{value:.4f}" for value in baseline.velocity)
            + ") cm/s equals the training mean",
        )
    )
    self_score = r2_per_axis(training.targets, baseline.predict(training))
    checks.append(
        Check(
            "the constant baseline scores exactly zero on its own training targets",
            bool(np.all(np.abs(self_score) < _CONSISTENCY_TOLERANCE)),
            f"training R2 {self_score[0]:+.2e} and {self_score[1]:+.2e}, the "
            "arithmetic identity of predicting a set's own mean",
        )
    )

    expected_features = training.n_bins * training.n_units
    checks.append(
        Check(
            "the Ridge model has one coefficient per feature column and axis",
            decoder.coefficients.shape == (len(AXIS_NAMES), expected_features)
            and decoder.intercept.shape == (len(AXIS_NAMES),),
            f"{decoder.coefficients.shape[0]} axes x {expected_features} columns "
            f"({training.n_bins} bins x {training.n_units} units)",
        )
    )
    checks.append(
        Check(
            "the Ridge model used the selected alpha",
            decoder.alpha == selection.ridge_alpha,
            f"alpha {decoder.alpha:g}",
        )
    )

    checks.append(
        Check(
            "each model produced one finite prediction per validation sample",
            all(
                array.shape == validation.targets.shape
                and bool(np.all(np.isfinite(array)))
                for array in predictions.values()
            ),
            f"{validation.n_samples} predictions for each of "
            f"{len(predictions)} models",
        )
    )

    reloaded_baseline = ConstantBaseline.from_dict(
        json.loads(written_models["constant_baseline"].read_text(encoding="utf-8"))
    )
    reloaded_decoder = RidgeDecoder.from_dict(
        json.loads(written_models["ridge_decoder"].read_text(encoding="utf-8"))
    )
    reproduced = {
        "constant": reloaded_baseline.predict(validation),
        "ridge": reloaded_decoder.predict(validation),
    }
    checks.append(
        Check(
            "the saved model artifacts reproduce the predictions exactly",
            all(
                np.array_equal(reproduced[name], predictions[name])
                for name in predictions
            ),
            "both models reloaded from JSON and predicted identically, bit for bit",
        )
    )

    stored = pd.read_csv(written_predictions)
    recomputed_matches: list[str] = []
    for name in predictions:
        columns = [f"{name}_velocity_{axis}" for axis in AXIS_NAMES]
        scores = r2_per_axis(
            stored[[f"target_velocity_{axis}" for axis in AXIS_NAMES]].to_numpy(),
            stored[columns].to_numpy(),
        )
        reported = [metrics[name]["r2"][axis] for axis in AXIS_NAMES]
        if not np.allclose(scores, reported, atol=_CONSISTENCY_TOLERANCE, rtol=0.0):
            recomputed_matches.append(name)
    checks.append(
        Check(
            "the reported metrics recompute from the saved per-sample predictions",
            not recomputed_matches and len(stored) == validation.n_samples,
            f"{len(stored)} saved rows reproduce every reported R2"
            if not recomputed_matches
            else f"{recomputed_matches} do not reproduce",
        )
    )

    manifest_aligned = np.array_equal(
        stored["target_index"].to_numpy(dtype=np.int64), validation.target_index
    ) and np.array_equal(
        stored["trial_row"].to_numpy(dtype=np.int64), validation.trial_row
    )
    checks.append(
        Check(
            "every saved prediction is identified by its validation sample",
            manifest_aligned,
            "trial row and behaviour sample index recorded for all "
            f"{validation.n_samples} predictions",
        )
    )

    return checks


def build_summary(
    session: SessionData,
    spec: FeatureSpec,
    selection: SelectedConfiguration,
    training: FeatureSet,
    validation: FeatureSet,
    baseline: ConstantBaseline,
    decoder: RidgeDecoder,
    metrics: dict[str, dict[str, object]],
    training_metrics: dict[str, dict[str, object]],
    checks: list[Check],
    artifacts: dict[str, str],
) -> dict[str, object]:
    """Assemble the machine-readable baseline summary."""
    return {
        "recording": {"name": session.identity.name, "sha256": session.identity.sha256},
        "configuration": {
            "lag_ms": selection.lag_ms,
            "ridge_alpha": selection.ridge_alpha,
            "read_from": selection.source,
            "selected_by": "training-only trial-grouped cross-validation, in an "
            "earlier step",
            "eligibility_lag_ms": training.eligibility_lag_ms,
            "eligibility_rule": "target eligibility uses the selected lag, frozen "
            "for every reported model so all models share one validation sample set",
            **spec.as_dict(),
        },
        "procedure": {
            "normalization": "fitted once on the complete training set and applied "
            "unchanged to validation",
            "constant_baseline": "predicts the mean training velocity for every "
            "sample and uses no neural data",
            "ridge": "fresh model fitted on the complete training set at the "
            "selected alpha",
            "evaluation": "each model scored once on validation; no tuning followed",
            "metric": "per-axis coefficient of determination, Pearson correlation, "
            "and root-mean-square error, each reported per axis and as the "
            "arithmetic mean of the axes",
        },
        "samples": {
            "training_trials": int(training.trial_rows.size),
            "training_samples": training.n_samples,
            "validation_trials": int(validation.trial_rows.size),
            "validation_samples": validation.n_samples,
            "feature_columns": training.n_bins * training.n_units,
            "n_units": training.n_units,
        },
        "models": {
            "constant_baseline": {
                "velocity_cm_s": dict(zip(AXIS_NAMES, baseline.velocity.tolist())),
                "fitted_on": baseline.fitted_on,
                "n_training_samples": baseline.n_samples,
            },
            "ridge": {
                "alpha": decoder.alpha,
                "fitted_on": decoder.fitted_on,
                "n_training_samples": decoder.n_samples,
                "n_features": decoder.n_features,
                "solver": "cholesky",
                "coefficient_absolute_max": float(np.max(np.abs(decoder.coefficients))),
            },
        },
        "validation_metrics": metrics,
        "training_diagnostics": {
            "purpose": "training-set fit quality, reported for the train-to-validation "
            "gap only and not a project result",
            **training_metrics,
        },
        "not_accessed": {"held_out": "not read while establishing the baselines"},
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in checks
        ],
        "all_checks_passed": all(check.passed for check in checks),
        "artifacts": artifacts,
    }


def render_markdown(summary: dict[str, object]) -> str:
    """Render the concise human-readable baseline summary."""
    configuration = summary["configuration"]
    samples = summary["samples"]
    validation = summary["validation_metrics"]
    training = summary["training_diagnostics"]

    lines = [
        "# Official constant and Ridge baselines",
        "",
        "## Validation results",
        "",
        "| Model | R2 x | R2 y | R2 mean | Pearson x | Pearson y | RMSE x (cm/s) | "
        "RMSE y (cm/s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, name in (("Constant baseline", "constant"), ("Ridge", "ridge")):
        entry = validation[name]
        lines.append(
            f"| {label} | {_number(entry['r2']['x'])} | {_number(entry['r2']['y'])} | "
            f"**{_number(entry['r2_mean'])}** | {_number(entry['pearson_r']['x'])} | "
            f"{_number(entry['pearson_r']['y'])} | {_number(entry['rmse_cm_s']['x'])} | "
            f"{_number(entry['rmse_cm_s']['y'])} |"
        )

    lines += [
        "",
        f"Both models were evaluated once on the same {samples['validation_samples']} "
        f"validation samples from {samples['validation_trials']} trials.",
        "",
        "The constant baseline predicts the mean training velocity "
        f"({_number(summary['models']['constant_baseline']['velocity_cm_s']['x'])}, "
        f"{_number(summary['models']['constant_baseline']['velocity_cm_s']['y'])}) cm/s "
        "for every sample. It uses no neural data, so its correlation is undefined "
        "rather than zero and its score is what predicting a fixed velocity earns "
        "on this validation set.",
        "",
        "## Configuration",
        "",
        f"- Lag: **{configuration['lag_ms']} ms**, read from "
        f"`{configuration['read_from']}`.",
        f"- Ridge alpha: **{configuration['ridge_alpha']:g}**, read from the same file.",
        f"- Neural history: {configuration['history_ms']} ms in "
        f"{configuration['n_bins']} bins of {configuration['bin_ms']} ms, sampled "
        f"every {configuration['cadence_ms']} ms.",
        f"- Eligibility: {configuration['eligibility_rule']}.",
        f"- Training: {samples['training_samples']} samples from "
        f"{samples['training_trials']} trials, {samples['feature_columns']} feature "
        f"columns over {samples['n_units']} units.",
        "",
        "## Procedure",
        "",
    ]
    lines += [
        f"- {key.replace('_', ' ').capitalize()}: {value}."
        for key, value in summary["procedure"].items()
    ]

    lines += [
        "",
        "## Training-set fit",
        "",
        f"Reported for the train-to-validation gap only, not as a project result. "
        f"Ridge mean R2 on the training set is "
        f"{_number(training['ridge']['r2_mean'])}, against "
        f"{_number(validation['ridge']['r2_mean'])} on validation.",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in summary["checks"]:
        lines.append(
            f"| {check['name']} | {'pass' if check['passed'] else 'FAIL'} | "
            f"{check['detail']} |"
        )

    lines += [
        "",
        f"Overall: {'all checks passed' if summary['all_checks_passed'] else 'FAILURES PRESENT'}.",
        "",
        "## Artifacts",
        "",
    ]
    lines += [f"- `{path}`" for path in summary["artifacts"].values()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--configuration",
        type=Path,
        metavar="PATH",
        help="Selected configuration JSON; defaults to "
        "results/selected_configuration.json.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    configure_logging(config.run.log_level)

    spec = FeatureSpec(
        history_ms=config.features.history_ms,
        n_bins=config.features.n_bins,
        cadence_ms=config.features.cadence_ms,
    )
    configuration_path = (
        args.configuration
        if args.configuration is not None
        else config.paths.results_dir / "selected_configuration.json"
    )

    try:
        selection = SelectedConfiguration.read(configuration_path)
        session = load_session(config)
    except (ConfigError, DatasetError) as exc:
        print(f"Cannot establish the baselines: {exc}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, to the selected recording described in README.md."
        )
        return 2

    logger.info(
        "Establishing baselines at lag %s ms with alpha %g",
        selection.lag_ms,
        selection.ridge_alpha,
    )

    try:
        training = build_split_features(
            session,
            TRAIN,
            lag_ms=selection.lag_ms,
            eligibility_lag_ms=selection.lag_ms,
            spec=spec,
        )
        validation = build_split_features(
            session,
            VALIDATION,
            lag_ms=selection.lag_ms,
            eligibility_lag_ms=selection.lag_ms,
            spec=spec,
        )
        baseline = ConstantBaseline.fit(training)
        decoder = RidgeDecoder.fit(training, alpha=selection.ridge_alpha)

        predictions = {
            "constant": baseline.predict(validation),
            "ridge": decoder.predict(validation),
        }
        metrics = {
            name: summarize_predictions(
                validation.targets, array, axis_names=AXIS_NAMES
            )
            for name, array in predictions.items()
        }
        training_metrics = {
            "constant": summarize_predictions(
                training.targets, baseline.predict(training), axis_names=AXIS_NAMES
            ),
            "ridge": summarize_predictions(
                training.targets, decoder.predict(training), axis_names=AXIS_NAMES
            ),
        }
    except (FeatureError, MetricError, ModelError) as exc:
        print(f"Cannot establish the baselines: {exc}")
        return 1

    results_dir = config.paths.results_dir
    baselines_dir = results_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    written_models = {
        "constant_baseline": baselines_dir / "constant_baseline.json",
        "ridge_decoder": baselines_dir / "ridge_decoder.json",
    }
    written_models["constant_baseline"].write_text(
        json.dumps(baseline.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    written_models["ridge_decoder"].write_text(
        json.dumps(decoder.as_dict(), indent=2) + "\n", encoding="utf-8"
    )

    predictions_path = baselines_dir / "validation_predictions.csv"
    manifest = validation.sample_manifest()
    predictions_frame = pd.DataFrame(
        {
            "sample_index": manifest["sample_index"],
            "trial_row": manifest["trial_row"],
            "reference_index": manifest["reference_index"],
            "target_index": manifest["target_index"],
            "target_time": manifest["target_time"],
            "target_velocity_x": validation.targets[:, 0],
            "target_velocity_y": validation.targets[:, 1],
            "constant_velocity_x": predictions["constant"][:, 0],
            "constant_velocity_y": predictions["constant"][:, 1],
            "ridge_velocity_x": predictions["ridge"][:, 0],
            "ridge_velocity_y": predictions["ridge"][:, 1],
        }
    )
    predictions_frame.to_csv(predictions_path, index=False)

    artifacts = {
        "validation_predictions": _repo_relative(predictions_path),
        "constant_baseline_model": _repo_relative(written_models["constant_baseline"]),
        "ridge_decoder_model": _repo_relative(written_models["ridge_decoder"]),
    }

    checks = run_checks(
        session,
        selection,
        training,
        validation,
        baseline,
        decoder,
        predictions,
        metrics,
        predictions_path,
        written_models,
    )

    metrics_path = results_dir / "baseline_metrics.json"
    summary_path = results_dir / "baseline_summary.json"
    report_path = results_dir / "baseline_summary.md"
    artifacts["baseline_metrics"] = _repo_relative(metrics_path)
    artifacts["summary_json"] = _repo_relative(summary_path)
    artifacts["summary_markdown"] = _repo_relative(report_path)

    summary = build_summary(
        session,
        spec,
        selection,
        training,
        validation,
        baseline,
        decoder,
        metrics,
        training_metrics,
        checks,
        artifacts,
    )

    metrics_path.write_text(
        json.dumps(
            {
                "split": VALIDATION,
                "lag_ms": selection.lag_ms,
                "ridge_alpha": selection.ridge_alpha,
                "eligibility_lag_ms": training.eligibility_lag_ms,
                "n_samples": validation.n_samples,
                "n_trials": int(validation.trial_rows.size),
                "recording_sha256": session.identity.sha256,
                "models": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    _print_report(metrics, training_metrics, checks, artifacts)
    failures = [check for check in checks if not check.passed]
    if failures:
        print(f"\nFAILED: {len(failures)} of {len(checks)} checks did not pass")
        return 1
    print(f"\nPASSED: all {len(checks)} checks")
    return 0


def _print_report(
    metrics: dict[str, dict[str, object]],
    training_metrics: dict[str, dict[str, object]],
    checks: list[Check],
    artifacts: dict[str, str],
) -> None:
    print("Validation results")
    for label, name in (("constant", "constant"), ("ridge", "ridge")):
        entry = metrics[name]
        print(
            f"  {label:<9} mean R2 {_number(entry['r2_mean']):>9}"
            f"  (x {_number(entry['r2']['x'])}, y {_number(entry['r2']['y'])})"
            f"  mean RMSE {_number(entry['rmse_cm_s_mean'])} cm/s"
            f"  mean r {_number(entry['pearson_r_mean'])}"
        )

    print("\nTraining-set fit (diagnostic only)")
    for name in ("constant", "ridge"):
        print(f"  {name:<9} mean R2 {_number(training_metrics[name]['r2_mean']):>9}")

    width = max(len(check.name) for check in checks)
    print("\nChecks")
    for check in checks:
        status = "pass" if check.passed else "FAIL"
        print(f"  {check.name:<{width}}  {status:<4}  {check.detail}")

    print("\nArtifacts")
    for path in artifacts.values():
        print(f"  {path}")


def _number(value: float | None) -> str:
    """Render a metric, or ``undefined`` where it has no value."""
    return "undefined" if value is None else f"{value:.6f}"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
