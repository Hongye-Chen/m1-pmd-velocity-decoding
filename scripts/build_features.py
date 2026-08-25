"""Build and validate the causal training and validation features.

Run from the repository root with the ``BCI`` environment active::

    conda deactivate
    conda activate BCI
    python -m scripts.build_features

The command builds samples for the training and validation splits only, fits
normalization on training data alone, writes the feature arrays, the fitted
normalization, and the per-sample manifests, then validates the produced artifacts
against the causal protocol. The held-out split is never built here.

Exit codes: 0 when every check passes, 1 when a check fails, and 2 when the
recording is not configured or cannot be identified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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
    DEFAULT_SPEC,
    EXPECTED_SAMPLE_COUNTS,
    FeatureError,
    FeatureSet,
    Normalization,
    build_split_features,
    fit_normalization,
)
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.splits import TEST, TRAIN, VALIDATION

logger = get_logger("build_features")

BUILT_SPLITS = (TRAIN, VALIDATION)
_LAG_TOLERANCE_S = 1e-9
# The behavior grid is 10 ms, so containment is asserted at grid resolution rather
# than at double precision.
_GRID_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class Check:
    """One runtime-validation check against the produced features."""

    name: str
    passed: bool
    detail: str


def run_checks(
    session: SessionData,
    feature_sets: dict[str, FeatureSet],
    normalization: Normalization,
    manifests: dict[str, pd.DataFrame],
    artifact_dir: Path,
) -> list[Check]:
    """Validate the produced features against the causal protocol."""
    checks: list[Check] = []
    spec = DEFAULT_SPEC
    trials = session.trials.set_index("trial_row")
    n_units = len(session.units)

    checks.append(
        Check(
            "feature protocol matches the frozen specification",
            spec.history_ms == 200
            and spec.n_bins == 4
            and spec.bin_ms == 50
            and spec.cadence_ms == 10
            and spec.target_lag_ms == 100,
            f"{spec.history_ms} ms history as {spec.n_bins} x {spec.bin_ms} ms bins, "
            f"{spec.cadence_ms} ms cadence, {spec.target_lag_ms} ms target lag",
        )
    )

    for split, features in feature_sets.items():
        expected = EXPECTED_SAMPLE_COUNTS[split]
        checks.append(
            Check(
                f"{split} sample count matches the feasibility reference",
                features.n_samples == expected,
                f"{features.n_samples} samples (reference {expected}, "
                f"difference {features.n_samples - expected:+d})",
            )
        )
        checks.append(
            Check(
                f"{split} feature shape",
                features.counts.shape == (features.n_samples, spec.n_bins, n_units),
                f"{features.counts.shape}",
            )
        )
        checks.append(
            Check(
                f"{split} target shape",
                features.targets.shape == (features.n_samples, 2),
                f"{features.targets.shape}",
            )
        )
        checks.append(
            Check(
                f"{split} counts are non-negative integers without overflow",
                np.issubdtype(features.counts.dtype, np.unsignedinteger)
                and int(features.counts.max()) < np.iinfo(features.counts.dtype).max,
                f"dtype {features.counts.dtype}, maximum {int(features.counts.max())} "
                f"spikes in one bin",
            )
        )
        checks.append(
            Check(
                f"{split} targets and timestamps are finite",
                bool(np.all(np.isfinite(features.targets)))
                and bool(np.all(np.isfinite(features.reference_time)))
                and bool(np.all(np.isfinite(features.target_time))),
                f"{features.n_samples} samples checked",
            )
        )

        lag = features.target_time - features.reference_time
        deviation = float(np.max(np.abs(lag - spec.target_lag_s)))
        checks.append(
            Check(
                f"{split} target sits exactly one lag after the reference",
                deviation <= _LAG_TOLERANCE_S
                and bool(np.all(features.target_time > features.reference_time)),
                f"maximum deviation {deviation:.3e} s from {spec.target_lag_s} s",
            )
        )

        starts = trials.loc[features.trial_row, "start_time"].to_numpy(dtype=np.float64)
        stops = trials.loc[features.trial_row, "stop_time"].to_numpy(dtype=np.float64)
        history_slack = float(np.min(features.reference_time - spec.history_s - starts))
        checks.append(
            Check(
                f"{split} history stays inside its own trial",
                history_slack >= -_GRID_TOLERANCE_S,
                f"tightest margin {history_slack:.3e} s before the trial start",
            )
        )
        target_slack = float(np.max(features.target_time - stops))
        checks.append(
            Check(
                f"{split} target stays inside its own trial",
                target_slack <= _GRID_TOLERANCE_S,
                f"tightest margin {-target_slack:.3e} s before the trial stop",
            )
        )

        checks.append(
            Check(
                f"{split} samples are ordered and unique",
                bool(np.all(np.diff(features.reference_index) > 0)),
                "reference indices strictly increase across the split",
            )
        )

        used = set(features.trial_row.tolist())
        allowed = set(
            session.trials.loc[
                (session.trials["split"] == split) & session.trials["eligible"],
                "trial_row",
            ].tolist()
        )
        checks.append(
            Check(
                f"{split} uses only eligible trials of its own split",
                used <= allowed,
                f"{len(used)} of {len(allowed)} eligible trials contributed",
            )
        )
        checks.append(
            Check(
                f"{split} every eligible trial contributed a sample",
                used == allowed,
                f"{len(allowed - used)} eligible trials produced no sample",
            )
        )
        checks.append(
            Check(
                f"{split} unit order matches the session",
                features.unit_ids.tolist() == [unit.unit_id for unit in session.units],
                f"{features.n_units} units in loader order",
            )
        )

    train_rows = set(feature_sets[TRAIN].trial_row.tolist())
    validation_rows = set(feature_sets[VALIDATION].trial_row.tolist())
    held_out_rows = set(
        session.trials.loc[session.trials["split"] == TEST, "trial_row"].tolist()
    )
    checks.append(
        Check(
            "training and validation samples share no trial",
            train_rows.isdisjoint(validation_rows),
            f"{len(train_rows)} training and {len(validation_rows)} validation trials",
        )
    )
    checks.append(
        Check(
            "no held-out trial appears in any built split",
            held_out_rows.isdisjoint(train_rows | validation_rows),
            f"{len(held_out_rows)} held-out trials remain unused",
        )
    )
    held_out_files = [
        path.name for path in artifact_dir.glob("*") if TEST in path.name.lower()
    ]
    checks.append(
        Check(
            "no held-out artifact was written",
            not held_out_files,
            "artifact directory contains no held-out file"
            if not held_out_files
            else f"found {held_out_files}",
        )
    )

    train = feature_sets[TRAIN]
    recomputed_mean = train.counts.astype(np.float64).mean(axis=0)
    recomputed_std = train.counts.astype(np.float64).std(axis=0)
    recomputed_std = np.where(recomputed_std < normalization.std_floor, 1.0, recomputed_std)
    checks.append(
        Check(
            "normalization was fitted on training data only",
            normalization.fitted_on == TRAIN
            and normalization.n_samples == train.n_samples
            and np.array_equal(normalization.feature_mean, recomputed_mean)
            and np.array_equal(normalization.feature_std, recomputed_std),
            f"fitted on {normalization.fitted_on} with {normalization.n_samples} samples",
        )
    )

    standardized_train = normalization.apply_features(train.counts)
    varying = normalization.feature_std > normalization.std_floor
    train_mean = float(np.max(np.abs(standardized_train.mean(axis=0)[varying])))
    train_std = float(np.max(np.abs(standardized_train.std(axis=0)[varying] - 1.0)))
    checks.append(
        Check(
            "standardized training features are centred and scaled",
            train_mean < 1e-9 and train_std < 1e-9,
            f"largest mean {train_mean:.3e}, largest scale error {train_std:.3e}",
        )
    )

    standardized_validation = normalization.apply_features(feature_sets[VALIDATION].counts)
    validation_mean = float(np.max(np.abs(standardized_validation.mean(axis=0)[varying])))
    checks.append(
        Check(
            "validation is transformed with the unchanged training statistics",
            validation_mean > 1e-9,
            f"validation retains a residual mean of {validation_mean:.3e} rather than 0",
        )
    )

    for split, manifest in manifests.items():
        features = feature_sets[split]
        aligned = (
            len(manifest) == features.n_samples
            and np.array_equal(
                manifest["reference_time"].to_numpy(dtype=np.float64),
                features.reference_time,
            )
            and np.array_equal(
                manifest["target_time"].to_numpy(dtype=np.float64), features.target_time
            )
            and np.array_equal(manifest["trial_row"].to_numpy(dtype=np.int64), features.trial_row)
            and np.array_equal(
                manifest["target_velocity_x"].to_numpy(dtype=np.float64), features.targets[:, 0]
            )
        )
        checks.append(
            Check(
                f"{split} manifest is aligned with the arrays",
                aligned,
                f"{len(manifest)} manifest rows for {features.n_samples} samples",
            )
        )

    return checks


def build_summary(
    session: SessionData,
    feature_sets: dict[str, FeatureSet],
    normalization: Normalization,
    checks: list[Check],
    artifacts: dict[str, str],
    hashes: dict[str, str],
) -> dict[str, object]:
    """Assemble the machine-readable feature-validation summary."""
    spec = DEFAULT_SPEC
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recording": {
            "name": session.identity.name,
            "sha256": session.identity.sha256,
        },
        "protocol": {
            **spec.as_dict(),
            "bin_offsets_s": spec.bin_offsets_s().tolist(),
            "history_window": "[reference - 200 ms, reference), half-open, bin 0 oldest",
            "target": "stored cursor velocity at reference + 100 ms",
            "flattened_layout": "bin-major: bin 0 units 0..n, then bin 1, and so on",
            "counts_shape": "(n_samples, n_bins, n_units)",
        },
        "splits": {
            split: {
                "samples": features.n_samples,
                "reference_samples": EXPECTED_SAMPLE_COUNTS[split],
                "difference": features.n_samples - EXPECTED_SAMPLE_COUNTS[split],
                "trials": int(np.unique(features.trial_row).size),
                "units": features.n_units,
                "feature_columns": features.n_bins * features.n_units,
                "reference_time_s": [
                    float(features.reference_time.min()),
                    float(features.reference_time.max()),
                ],
                "spikes_per_sample_mean": float(
                    features.counts.sum(axis=(1, 2), dtype=np.int64).mean()
                ),
                "max_spikes_in_one_bin": int(features.counts.max()),
                "target_velocity_mean": features.targets.mean(axis=0).tolist(),
                "target_velocity_std": features.targets.std(axis=0).tolist(),
            }
            for split, features in feature_sets.items()
        },
        "normalization": {
            "fitted_on": normalization.fitted_on,
            "n_samples": normalization.n_samples,
            "n_features_at_floor": normalization.n_features_at_floor,
            "feature_mean_range": [
                float(normalization.feature_mean.min()),
                float(normalization.feature_mean.max()),
            ],
            "feature_std_range": [
                float(normalization.feature_std.min()),
                float(normalization.feature_std.max()),
            ],
            "target_mean": normalization.target_mean.tolist(),
            "target_std": normalization.target_std.tolist(),
        },
        "held_out": {
            "built": False,
            "note": "The held-out split is not constructed before the frozen evaluation.",
        },
        "array_sha256": hashes,
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in checks
        ],
        "all_checks_passed": all(check.passed for check in checks),
        "artifacts": artifacts,
    }


def render_markdown(summary: dict[str, object]) -> str:
    """Render the concise human-readable feature-validation summary."""
    protocol = summary["protocol"]
    splits = summary["splits"]
    normalization = summary["normalization"]

    lines = [
        "# Feature validation summary",
        "",
        f"Generated {summary['generated_at']}.",
        "",
        "## Causal protocol",
        "",
        f"- History: {protocol['history_window']}",
        f"- Bins: {protocol['n_bins']} x {protocol['bin_ms']} ms, offsets "
        f"{protocol['bin_offsets_s']} s",
        f"- Prediction cadence: {protocol['cadence_ms']} ms",
        f"- Target: {protocol['target']}",
        f"- Array layout: {protocol['counts_shape']}, flattened {protocol['flattened_layout']}",
        "",
        "## Samples",
        "",
        "| Split | Samples | Feasibility reference | Difference | Trials | Feature columns |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for split, info in splits.items():
        lines.append(
            f"| {split} | {info['samples']} | {info['reference_samples']} | "
            f"{info['difference']:+d} | {info['trials']} | {info['feature_columns']} |"
        )

    lines += [
        "",
        "The held-out split is not built at this stage.",
        "",
        "## Normalization",
        "",
        f"- Fitted on `{normalization['fitted_on']}` using {normalization['n_samples']} samples",
        f"- Feature means span {normalization['feature_mean_range'][0]:.4f} to "
        f"{normalization['feature_mean_range'][1]:.4f} spikes per bin",
        f"- Feature deviations span {normalization['feature_std_range'][0]:.4f} to "
        f"{normalization['feature_std_range'][1]:.4f}",
        f"- Features held at the deviation floor: {normalization['n_features_at_floor']}",
        f"- Target mean {normalization['target_mean']} cm/s, deviation "
        f"{normalization['target_std']} cm/s",
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
    parser.parse_args(argv)

    config = load_config()
    configure_logging(config.run.log_level)

    try:
        session = load_session(config)
    except (ConfigError, DatasetError) as exc:
        print(f"Cannot build features: {exc}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, to the selected recording described in README.md."
        )
        return 2

    results_dir = config.paths.results_dir
    artifact_dir = results_dir / "features"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        feature_sets = {
            split: build_split_features(session, split) for split in BUILT_SPLITS
        }
        normalization = fit_normalization(feature_sets[TRAIN])
    except FeatureError as exc:
        print(f"Cannot build features: {exc}")
        return 1

    artifacts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    manifests: dict[str, pd.DataFrame] = {}

    for split, features in feature_sets.items():
        array_path = artifact_dir / f"{split}.npz"
        np.savez_compressed(
            array_path,
            counts=features.counts,
            targets=features.targets,
            reference_index=features.reference_index,
            target_index=features.target_index,
            reference_time=features.reference_time,
            target_time=features.target_time,
            trial_row=features.trial_row,
            unit_ids=features.unit_ids,
        )
        manifest = features.sample_manifest()
        manifest_path = artifact_dir / f"samples_{split}.csv"
        manifest.to_csv(manifest_path, index=False)

        manifests[split] = manifest
        artifacts[f"{split}_arrays"] = _repo_relative(array_path)
        artifacts[f"{split}_manifest"] = _repo_relative(manifest_path)
        hashes[f"{split}_counts"] = _array_hash(features.counts)
        hashes[f"{split}_targets"] = _array_hash(features.targets)
        logger.info("Wrote %s samples for %s", features.n_samples, split)

    normalization_path = results_dir / "feature_normalization.json"
    normalization_path.write_text(
        json.dumps(normalization.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    artifacts["normalization"] = _repo_relative(normalization_path)

    checks = run_checks(session, feature_sets, normalization, manifests, artifact_dir)

    summary_path = results_dir / "feature_validation.json"
    report_path = results_dir / "feature_validation.md"
    artifacts["summary_json"] = _repo_relative(summary_path)
    artifacts["summary_markdown"] = _repo_relative(report_path)

    summary = build_summary(session, feature_sets, normalization, checks, artifacts, hashes)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    _print_report(feature_sets, checks, artifacts)
    failures = [check for check in checks if not check.passed]
    if failures:
        print(f"\nFAILED: {len(failures)} of {len(checks)} checks did not pass")
        return 1
    print(f"\nPASSED: all {len(checks)} checks")
    return 0


def _print_report(
    feature_sets: dict[str, FeatureSet], checks: list[Check], artifacts: dict[str, str]
) -> None:
    for split, features in feature_sets.items():
        print(
            f"{split:<11} {features.n_samples} samples  "
            f"counts {features.counts.shape}  targets {features.targets.shape}  "
            f"reference {EXPECTED_SAMPLE_COUNTS[split]}"
        )
    print("held-out    not built")

    width = max(len(check.name) for check in checks)
    print("\nChecks")
    for check in checks:
        status = "pass" if check.passed else "FAIL"
        print(f"  {check.name:<{width}}  {status:<4}  {check.detail}")

    print("\nArtifacts")
    for path in artifacts.values():
        print(f"  {path}")


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
