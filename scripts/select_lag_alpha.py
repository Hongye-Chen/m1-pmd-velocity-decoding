"""Select the shared neural-to-target lag and Ridge alpha from training trials only.

Run from the repository root with the ``BCI`` environment active::

    conda deactivate
    conda activate BCI
    python -m scripts.select_lag_alpha

The command builds training features for every candidate lag over one shared target
set, scores each lag and alpha with trial-grouped cross-validation, applies the
frozen tie-breaking rules, and writes the selected configuration together with the
full results. Validation and held-out trials are never read, and the temporary Ridge
models fitted here are discarded: the official models are fitted in a later step.

Exit codes: 0 when every check passes, 1 when a check fails, and 2 when the
recording is not configured or cannot be identified.
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
from neural_decoding.features import FeatureError, FeatureSet, FeatureSpec, build_session_lag_features
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.selection import (
    N_GROUPS,
    RIDGE_ALPHAS,
    TIE_TOLERANCE,
    CandidateResult,
    CrossValidationReport,
    SelectionError,
    contiguous_trial_groups,
    cross_validate,
    select_candidate,
)
from neural_decoding.splits import TEST, TRAIN, VALIDATION

logger = get_logger("select_lag_alpha")

SELECTION_SPLIT = TRAIN


@dataclass(frozen=True)
class Check:
    """One runtime-validation check against the selection run."""

    name: str
    passed: bool
    detail: str


def run_checks(
    session: SessionData,
    feature_sets: dict[int, FeatureSet],
    report: CrossValidationReport,
    selected: CandidateResult,
    lags: tuple[int, ...],
    alphas: tuple[float, ...],
) -> list[Check]:
    """Validate that the selection used training data only and followed the rules."""
    checks: list[Check] = []
    trials = session.trials
    eligible_training = set(
        trials.loc[(trials["split"] == TRAIN) & trials["eligible"], "trial_row"].tolist()
    )
    validation_rows = set(trials.loc[trials["split"] == VALIDATION, "trial_row"].tolist())
    held_out_rows = set(trials.loc[trials["split"] == TEST, "trial_row"].tolist())

    used_rows: set[int] = set()
    for features in feature_sets.values():
        used_rows |= set(features.trial_rows.tolist())

    checks.append(
        Check(
            "only eligible training trials were used",
            used_rows == eligible_training,
            f"{len(used_rows)} of {len(eligible_training)} eligible training trials",
        )
    )
    checks.append(
        Check(
            "no validation or held-out trial was read",
            used_rows.isdisjoint(validation_rows | held_out_rows),
            f"all {len(validation_rows)} assigned validation and {len(held_out_rows)} "
            "assigned held-out trial rows remain untouched",
        )
    )
    checks.append(
        Check(
            "every feature set is the training split",
            all(features.split == SELECTION_SPLIT for features in feature_sets.values()),
            f"split {SELECTION_SPLIT!r} only",
        )
    )

    group_sizes = [len(fold.assessment_rows) for fold in report.folds]
    assessed = [row for fold in report.folds for row in fold.assessment_rows]
    checks.append(
        Check(
            "trials form contiguous groups of near-equal size",
            len(report.folds) == N_GROUPS and max(group_sizes) - min(group_sizes) <= 1,
            f"{N_GROUPS} groups sized {group_sizes}",
        )
    )
    checks.append(
        Check(
            "every trial is assessed exactly once",
            sorted(assessed) == sorted(eligible_training),
            f"{len(assessed)} assessments over {len(eligible_training)} trials",
        )
    )
    checks.append(
        Check(
            "no trial appears in both sides of a fold",
            all(
                set(fold.training_rows).isdisjoint(fold.assessment_rows)
                and set(fold.training_rows) | set(fold.assessment_rows) == eligible_training
                for fold in report.folds
            ),
            "fold training and assessment trials are disjoint and complete",
        )
    )

    expected_training = {fold.index: set(fold.training_rows) for fold in report.folds}
    leaking = [
        record
        for record in report.normalizations
        if set(record.training_rows) & set(record.assessment_rows)
        or set(record.training_rows) != expected_training[record.fold_index]
    ]
    checks.append(
        Check(
            "each fold was standardized without its assessment trials",
            not leaking,
            f"{len(report.normalizations)} fold normalizations, one per lag and fold"
            if not leaking
            else f"{len(leaking)} normalizations saw assessment trials",
        )
    )

    reference = feature_sets[lags[0]]
    shared = all(
        np.array_equal(feature_sets[lag].target_index, reference.target_index)
        and np.array_equal(feature_sets[lag].targets, reference.targets)
        for lag in lags
    )
    checks.append(
        Check(
            "every lag was scored on identical target samples",
            shared,
            f"{reference.n_samples} shared target samples across {len(lags)} lags",
        )
    )

    checks.append(
        Check(
            "the frozen grid was searched completely",
            len(report.candidates) == len(lags) * len(alphas)
            and {candidate.lag_ms for candidate in report.candidates} == set(lags)
            and {candidate.alpha for candidate in report.candidates} == set(alphas),
            f"{len(lags)} lags x {len(alphas)} alphas = {len(report.candidates)} candidates",
        )
    )
    checks.append(
        Check(
            "per-fold diagnostics cover every candidate and fold",
            len(report.fold_results) == len(lags) * len(alphas) * N_GROUPS,
            f"{len(report.fold_results)} fold results",
        )
    )
    checks.append(
        Check(
            "out-of-fold predictions are complete and finite",
            all(
                candidate.predictions.shape == reference.targets.shape
                and bool(np.all(np.isfinite(candidate.predictions)))
                for candidate in report.candidates
            ),
            f"{reference.n_samples} predictions per candidate, none unfilled",
        )
    )

    recomputed = select_candidate(report.candidates)
    checks.append(
        Check(
            "the selection follows the frozen tie-breaking rules",
            recomputed.lag_ms == selected.lag_ms and recomputed.alpha == selected.alpha,
            f"lag {selected.lag_ms} ms with alpha {selected.alpha}",
        )
    )
    checks.append(
        Check(
            "the selected configuration is inside the frozen grid",
            selected.lag_ms in lags and selected.alpha in alphas,
            f"lag {selected.lag_ms} ms in {list(lags)}, alpha {selected.alpha} in {list(alphas)}",
        )
    )

    best = max(candidate.r2_mean for candidate in report.candidates)
    tied = [c for c in report.candidates if best - c.r2_mean <= TIE_TOLERANCE]
    checks.append(
        Check(
            "the winning score is not a near-tie decided by noise",
            len(tied) == 1,
            f"{len(tied)} candidate within {TIE_TOLERANCE:g} of the best score"
            if len(tied) == 1
            else f"{len(tied)} candidates tied; smaller lag then larger alpha applied",
        )
    )

    return checks


def build_summary(
    session: SessionData,
    spec: FeatureSpec,
    feature_sets: dict[int, FeatureSet],
    report: CrossValidationReport,
    selected: CandidateResult,
    lags: tuple[int, ...],
    alphas: tuple[float, ...],
    checks: list[Check],
    artifacts: dict[str, str],
) -> dict[str, object]:
    """Assemble the machine-readable selection summary."""
    reference = feature_sets[lags[0]]
    ranked = sorted(report.candidates, key=lambda c: (-c.r2_mean, c.lag_ms, -c.alpha))
    return {
        "recording": {"name": session.identity.name, "sha256": session.identity.sha256},
        "procedure": {
            "split_used": SELECTION_SPLIT,
            "n_groups": N_GROUPS,
            "grouping": (
                "eligible training trials sorted by original row, then divided into "
                "contiguous groups of near-equal size with the earliest groups taking "
                "any remainder"
            ),
            "lag_grid_ms": list(lags),
            "alpha_grid": list(alphas),
            "eligibility_lag_ms": reference.eligibility_lag_ms,
            "normalization": "fitted on each fold's training trials, applied unchanged to its assessment trials",
            "metric": "arithmetic mean of the per-axis coefficient of determination on concatenated out-of-fold predictions",
            "tie_breaking": [
                "highest out-of-fold mean score",
                f"within {TIE_TOLERANCE:g}, smaller lag",
                "if still tied, larger alpha",
            ],
            "models": "temporary Ridge models, discarded after selection",
            **spec.as_dict(),
        },
        "samples": {
            "training_trials": int(reference.trial_rows.size),
            "shared_target_samples": reference.n_samples,
            "feature_columns": reference.n_bins * reference.n_units,
        },
        "selected": {
            "lag_ms": selected.lag_ms,
            "ridge_alpha": selected.alpha,
            "out_of_fold_r2_x": selected.r2_x,
            "out_of_fold_r2_y": selected.r2_y,
            "out_of_fold_r2_mean": selected.r2_mean,
            "fold_r2_means": list(selected.fold_r2_means),
            "fold_r2_spread": selected.fold_r2_spread,
        },
        "ranking": [
            {
                "rank": position + 1,
                "lag_ms": candidate.lag_ms,
                "ridge_alpha": candidate.alpha,
                "out_of_fold_r2_mean": candidate.r2_mean,
                "fold_r2_spread": candidate.fold_r2_spread,
            }
            for position, candidate in enumerate(ranked)
        ],
        "best_per_lag": {
            str(lag): max(c.r2_mean for c in report.candidates if c.lag_ms == lag)
            for lag in lags
        },
        "not_accessed": {
            "validation": "not read during selection",
            "held_out": "not read during selection",
        },
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in checks
        ],
        "all_checks_passed": all(check.passed for check in checks),
        "artifacts": artifacts,
    }


def render_markdown(summary: dict[str, object]) -> str:
    """Render the concise human-readable selection summary."""
    procedure = summary["procedure"]
    samples = summary["samples"]
    selected = summary["selected"]

    lines = [
        "# Lag and Ridge alpha selection",
        "",
        "## Selected configuration",
        "",
        f"- Lag: **{selected['lag_ms']} ms**",
        f"- Ridge alpha: **{selected['ridge_alpha']:g}**",
        f"- Out-of-fold mean R2: {selected['out_of_fold_r2_mean']:.6f} "
        f"(horizontal {selected['out_of_fold_r2_x']:.6f}, "
        f"vertical {selected['out_of_fold_r2_y']:.6f})",
        f"- Per-fold mean R2: "
        + ", ".join(f"{value:.4f}" for value in selected["fold_r2_means"])
        + f" (spread {selected['fold_r2_spread']:.4f})",
        "",
        "## Procedure",
        "",
        f"- Split used: `{procedure['split_used']}` only. Validation and held-out "
        "trials were not read.",
        f"- {samples['training_trials']} eligible training trials in "
        f"{procedure['n_groups']} contiguous groups; {procedure['grouping']}.",
        f"- {samples['shared_target_samples']} shared target samples, "
        f"{samples['feature_columns']} feature columns.",
        f"- Lag grid: {procedure['lag_grid_ms']} ms. Alpha grid: {procedure['alpha_grid']}.",
        f"- Normalization {procedure['normalization']}.",
        f"- Selection metric: {procedure['metric']}.",
        f"- Models: {procedure['models']}.",
        "",
        "## Best alpha per lag",
        "",
        "| Lag (ms) | Best out-of-fold mean R2 |",
        "| --- | --- |",
    ]
    for lag, score in summary["best_per_lag"].items():
        lines.append(f"| {lag} | {score:.6f} |")

    lines += [
        "",
        "## Ranking",
        "",
        "| Rank | Lag (ms) | Alpha | Out-of-fold mean R2 | Fold spread |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in summary["ranking"]:
        lines.append(
            f"| {entry['rank']} | {entry['lag_ms']} | {entry['ridge_alpha']:g} | "
            f"{entry['out_of_fold_r2_mean']:.6f} | {entry['fold_r2_spread']:.4f} |"
        )

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in summary["checks"]:
        lines.append(
            f"| {check['name']} | {'pass' if check['passed'] else 'FAIL'} | {check['detail']} |"
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
        "--lags",
        type=int,
        nargs="+",
        metavar="MS",
        help="Candidate lags in milliseconds; defaults to features.candidate_lags_ms.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    configure_logging(config.run.log_level)

    spec = FeatureSpec(
        history_ms=config.features.history_ms,
        n_bins=config.features.n_bins,
        cadence_ms=config.features.cadence_ms,
    )
    lags = tuple(sorted(args.lags)) if args.lags else config.features.candidate_lags_ms
    alphas = RIDGE_ALPHAS

    try:
        session = load_session(config)
    except (ConfigError, DatasetError) as exc:
        print(f"Cannot run selection: {exc}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, to the selected recording described in README.md."
        )
        return 2

    try:
        feature_sets = build_session_lag_features(
            session, SELECTION_SPLIT, lags=lags, spec=spec
        )
        logger.info("Scoring %s lags x %s alphas", len(lags), len(alphas))
        report = cross_validate(feature_sets, alphas=alphas, n_groups=N_GROUPS)
        selected = select_candidate(report.candidates)
    except (FeatureError, SelectionError) as exc:
        print(f"Cannot run selection: {exc}")
        return 1

    results_dir = config.paths.results_dir
    selection_dir = results_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    groups = contiguous_trial_groups(feature_sets[lags[0]].trial_rows, N_GROUPS)
    fold_assignments = pd.DataFrame(
        [
            {"trial_row": row, "group_index": index, "assessed_in_fold": index}
            for index, group in enumerate(groups)
            for row in group
        ]
    )
    paths = {
        "fold_assignments": selection_dir / "fold_assignments.csv",
        "cv_results": selection_dir / "cv_results.csv",
        "fold_metrics": selection_dir / "fold_metrics.csv",
        "fold_normalization": selection_dir / "fold_normalization.csv",
    }
    fold_assignments.to_csv(paths["fold_assignments"], index=False)

    cv_results = pd.DataFrame(
        [
            {
                "lag_ms": candidate.lag_ms,
                "ridge_alpha": candidate.alpha,
                "n_samples": candidate.n_samples,
                "out_of_fold_r2_x": candidate.r2_x,
                "out_of_fold_r2_y": candidate.r2_y,
                "out_of_fold_r2_mean": candidate.r2_mean,
                "fold_r2_spread": candidate.fold_r2_spread,
                **{
                    f"fold{index}_r2_mean": value
                    for index, value in enumerate(candidate.fold_r2_means)
                },
            }
            for candidate in report.candidates
        ]
    ).sort_values(["lag_ms", "ridge_alpha"])
    cv_results.to_csv(paths["cv_results"], index=False)

    fold_metrics = pd.DataFrame(
        [
            {
                "lag_ms": result.lag_ms,
                "ridge_alpha": result.alpha,
                "fold_index": result.fold_index,
                "n_samples": result.n_samples,
                "r2_x": result.r2_x,
                "r2_y": result.r2_y,
                "r2_mean": result.r2_mean,
            }
            for result in report.fold_results
        ]
    ).sort_values(["lag_ms", "ridge_alpha", "fold_index"])
    fold_metrics.to_csv(paths["fold_metrics"], index=False)

    fold_normalization = pd.DataFrame(
        [
            {
                "lag_ms": record.lag_ms,
                "fold_index": record.fold_index,
                "fitted_on": record.fitted_on,
                "n_training_trials": len(record.training_rows),
                "n_training_samples": record.n_samples,
                "n_assessment_trials": len(record.assessment_rows),
                "overlap_with_assessment": len(
                    set(record.training_rows) & set(record.assessment_rows)
                ),
            }
            for record in report.normalizations
        ]
    ).sort_values(["lag_ms", "fold_index"])
    fold_normalization.to_csv(paths["fold_normalization"], index=False)

    for name, path in paths.items():
        artifacts[name] = _repo_relative(path)

    predictions_path = selection_dir / "out_of_fold_predictions.npz"
    np.savez_compressed(
        predictions_path,
        target_index=feature_sets[lags[0]].target_index,
        trial_row=feature_sets[lags[0]].trial_row,
        targets=feature_sets[lags[0]].targets,
        **{
            f"lag{candidate.lag_ms:03d}ms_alpha{candidate.alpha:g}": candidate.predictions
            for candidate in report.candidates
        },
    )
    artifacts["out_of_fold_predictions"] = _repo_relative(predictions_path)

    checks = run_checks(session, feature_sets, report, selected, lags, alphas)

    configuration_path = results_dir / "selected_configuration.json"
    summary_path = results_dir / "selection_summary.json"
    report_path = results_dir / "selection_summary.md"
    artifacts["selected_configuration"] = _repo_relative(configuration_path)
    artifacts["summary_json"] = _repo_relative(summary_path)
    artifacts["summary_markdown"] = _repo_relative(report_path)

    summary = build_summary(
        session, spec, feature_sets, report, selected, lags, alphas, checks, artifacts
    )

    configuration_path.write_text(
        json.dumps(
            {
                "lag_ms": selected.lag_ms,
                "ridge_alpha": selected.alpha,
                "selected_by": "training-only trial-grouped cross-validation",
                "selection_metric": "mean out-of-fold per-axis R2",
                "out_of_fold_r2_mean": selected.r2_mean,
                "lag_grid_ms": list(lags),
                "alpha_grid": list(alphas),
                "n_groups": N_GROUPS,
                "history_ms": spec.history_ms,
                "n_bins": spec.n_bins,
                "cadence_ms": spec.cadence_ms,
                "recording_sha256": session.identity.sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    _print_report(report, selected, checks, artifacts)
    failures = [check for check in checks if not check.passed]
    if failures:
        print(f"\nFAILED: {len(failures)} of {len(checks)} checks did not pass")
        return 1
    print(f"\nPASSED: all {len(checks)} checks")
    return 0


def _print_report(
    report: CrossValidationReport,
    selected: CandidateResult,
    checks: list[Check],
    artifacts: dict[str, str],
) -> None:
    print(f"Selected    lag {selected.lag_ms} ms, alpha {selected.alpha:g}")
    print(
        f"Out-of-fold mean R2 {selected.r2_mean:.6f} "
        f"(x {selected.r2_x:.6f}, y {selected.r2_y:.6f})"
    )

    print("\nBest alpha per lag")
    for lag in sorted({candidate.lag_ms for candidate in report.candidates}):
        best = max(
            (c for c in report.candidates if c.lag_ms == lag), key=lambda c: c.r2_mean
        )
        marker = "  <- selected" if best.alpha == selected.alpha and lag == selected.lag_ms else ""
        print(
            f"  lag {lag:>3} ms  alpha {best.alpha:>8g}  mean R2 {best.r2_mean:.6f}"
            f"  fold spread {best.fold_r2_spread:.4f}{marker}"
        )

    width = max(len(check.name) for check in checks)
    print("\nChecks")
    for check in checks:
        status = "pass" if check.passed else "FAIL"
        print(f"  {check.name:<{width}}  {status:<4}  {check.detail}")

    print("\nArtifacts")
    for path in artifacts.values():
        print(f"  {path}")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
