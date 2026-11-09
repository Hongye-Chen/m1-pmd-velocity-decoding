"""Build and validate causal neural features for each candidate lag.

Run from the repository root with the project environment active::

    python -m scripts.build_features
    python -m scripts.build_features --lags 0 100 200

Exit codes: 0 when every check passes, 1 when a check fails, and 2 when the
recording is not configured or cannot be identified.
"""

from __future__ import annotations

import argparse
import hashlib
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
    Normalization,
    build_session_lag_features,
    fit_normalization,
)
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.splits import TEST, TRAIN, VALIDATION

logger = get_logger("build_features")

BUILT_SPLIT = TRAIN
_LAG_TOLERANCE_S = 1e-9
# The behavior grid is 10 ms, so containment is asserted at grid resolution rather
# than at double precision.
_GRID_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class Check:
    """Result of one feature validation check."""

    name: str
    passed: bool
    detail: str


def run_checks(
    session: SessionData,
    spec: FeatureSpec,
    feature_sets: dict[int, FeatureSet],
    normalizations: dict[int, Normalization],
    manifests: dict[int, pd.DataFrame],
    artifact_dir: Path,
) -> list[Check]:
    """Validate the training features and target alignment."""
    checks: list[Check] = []
    trials = session.trials.set_index("trial_row")
    n_units = len(session.units)
    lags = sorted(feature_sets)
    reference_set = feature_sets[lags[0]]

    checks.append(
        Check(
            "history window matches the configured protocol",
            spec.history_ms == spec.n_bins * spec.bin_ms and spec.cadence_ms > 0,
            f"{spec.history_ms} ms history as {spec.n_bins} x {spec.bin_ms} ms bins, "
            f"{spec.cadence_ms} ms cadence",
        )
    )
    checks.append(
        Check(
            "candidate lags built",
            len(lags) >= 1,
            f"{lags} ms, eligibility anchored on {reference_set.eligibility_lag_ms} ms",
        )
    )

    eligible_training = set(
        session.trials.loc[
            (session.trials["split"] == TRAIN) & session.trials["eligible"], "trial_row"
        ].tolist()
    )
    other_split_rows = set(
        session.trials.loc[session.trials["split"] != TRAIN, "trial_row"].tolist()
    )

    for lag in lags:
        features = feature_sets[lag]
        label = f"lag {lag} ms"

        checks.append(
            Check(
                f"{label} feature and target shapes",
                features.counts.shape == (features.n_samples, spec.n_bins, n_units)
                and features.targets.shape == (features.n_samples, 2),
                f"counts {features.counts.shape}, targets {features.targets.shape}",
            )
        )
        checks.append(
            Check(
                f"{label} counts are non-negative integers without overflow",
                np.issubdtype(features.counts.dtype, np.unsignedinteger)
                and int(features.counts.max()) < np.iinfo(features.counts.dtype).max,
                f"dtype {features.counts.dtype}, maximum {int(features.counts.max())} "
                "spikes in one bin",
            )
        )
        checks.append(
            Check(
                f"{label} targets and timestamps are finite",
                bool(np.all(np.isfinite(features.targets)))
                and bool(np.all(np.isfinite(features.reference_time)))
                and bool(np.all(np.isfinite(features.target_time))),
                f"{features.n_samples} samples checked",
            )
        )

        realized = features.target_time - features.reference_time
        deviation = float(np.max(np.abs(realized - lag / 1000.0)))
        checks.append(
            Check(
                f"{label} target sits exactly one lag after the reference",
                deviation <= _LAG_TOLERANCE_S
                and bool(np.all(features.target_time >= features.reference_time)),
                f"maximum deviation {deviation:.3e} s from {lag / 1000.0} s",
            )
        )

        starts = trials.loc[features.trial_row, "start_time"].to_numpy(dtype=np.float64)
        stops = trials.loc[features.trial_row, "stop_time"].to_numpy(dtype=np.float64)
        history_slack = float(np.min(features.reference_time - spec.history_s - starts))
        target_slack = float(np.max(features.target_time - stops))
        checks.append(
            Check(
                f"{label} history and target stay inside their own trial",
                history_slack >= -_GRID_TOLERANCE_S and target_slack <= _GRID_TOLERANCE_S,
                f"history margin {history_slack:.3e} s, target margin {-target_slack:.3e} s",
            )
        )
        checks.append(
            Check(
                f"{label} samples are ordered and unique",
                bool(np.all(np.diff(features.target_index) > 0)),
                "target indices strictly increase across the split",
            )
        )

        used = set(features.trial_row.tolist())
        checks.append(
            Check(
                f"{label} uses every eligible training trial and nothing else",
                used == eligible_training,
                f"{len(used)} of {len(eligible_training)} eligible training trials",
            )
        )
        checks.append(
            Check(
                f"{label} contains no validation or held-out trial",
                used.isdisjoint(other_split_rows),
                f"{len(other_split_rows)} trials outside training remain unused",
            )
        )
        checks.append(
            Check(
                f"{label} unit order matches the session",
                features.unit_ids.tolist() == [unit.unit_id for unit in session.units],
                f"{features.n_units} units in loader order",
            )
        )

        manifest = manifests[lag]
        aligned = (
            len(manifest) == features.n_samples
            and np.array_equal(
                manifest["reference_time"].to_numpy(dtype=np.float64), features.reference_time
            )
            and np.array_equal(
                manifest["target_time"].to_numpy(dtype=np.float64), features.target_time
            )
            and np.array_equal(manifest["trial_row"].to_numpy(dtype=np.int64), features.trial_row)
            and np.array_equal(
                manifest["target_velocity_x"].to_numpy(dtype=np.float64), features.targets[:, 0]
            )
            and set(manifest["lag_ms"].unique().tolist()) == {lag}
        )
        checks.append(
            Check(
                f"{label} manifest is aligned with the arrays",
                aligned,
                f"{len(manifest)} manifest rows for {features.n_samples} samples",
            )
        )

        normalization = normalizations[lag]
        recomputed_mean = features.counts.astype(np.float64).mean(axis=0)
        recomputed_std = features.counts.astype(np.float64).std(axis=0)
        recomputed_std = np.where(recomputed_std < normalization.std_floor, 1.0, recomputed_std)
        checks.append(
            Check(
                f"{label} normalization is fitted on the full training set only",
                normalization.fitted_on == TRAIN
                and normalization.n_samples == features.n_samples
                and set(normalization.trial_rows) == eligible_training
                and np.array_equal(normalization.feature_mean, recomputed_mean)
                and np.array_equal(normalization.feature_std, recomputed_std),
                f"fitted on {normalization.fitted_on} over {len(normalization.trial_rows)} "
                f"trials and {normalization.n_samples} samples",
            )
        )

    identical = all(
        np.array_equal(feature_sets[lag].target_index, reference_set.target_index)
        and np.array_equal(feature_sets[lag].target_time, reference_set.target_time)
        and np.array_equal(feature_sets[lag].targets, reference_set.targets)
        and np.array_equal(feature_sets[lag].trial_row, reference_set.trial_row)
        for lag in lags
    )
    checks.append(
        Check(
            "every lag predicts identical targets in identical order",
            identical,
            f"{reference_set.n_samples} shared target samples across {len(lags)} lags",
        )
    )
    windows_differ = len(lags) == 1 or any(
        not np.array_equal(feature_sets[lag].reference_index, reference_set.reference_index)
        for lag in lags[1:]
    )
    checks.append(
        Check(
            "each lag shifts the neural window",
            windows_differ,
            "reference samples differ between lags while targets are shared",
        )
    )

    unexpected = sorted(
        path.name
        for path in artifact_dir.glob("*")
        if VALIDATION in path.name.lower() or TEST in path.name.lower()
    )
    checks.append(
        Check(
            "no validation or held-out artifact was written",
            not unexpected,
            "training artifacts only" if not unexpected else f"found {unexpected}",
        )
    )

    return checks


def build_summary(
    session: SessionData,
    spec: FeatureSpec,
    feature_sets: dict[int, FeatureSet],
    normalizations: dict[int, Normalization],
    checks: list[Check],
    artifacts: dict[str, str],
    hashes: dict[str, str],
) -> dict[str, object]:
    """Build the feature-validation summary."""
    lags = sorted(feature_sets)
    reference_set = feature_sets[lags[0]]
    return {
        "recording": {
            "name": session.identity.name,
            "sha256": session.identity.sha256,
        },
        "protocol": {
            **spec.as_dict(),
            "bin_offsets_s": spec.bin_offsets_s().tolist(),
            "history_window": (
                f"[reference - {spec.history_ms} ms, reference), half-open, bin 0 oldest"
            ),
            "target": "stored cursor velocity at reference + lag",
            "candidate_lags_ms": lags,
            "eligibility_lag_ms": reference_set.eligibility_lag_ms,
            "eligibility_rule": (
                "Target samples are those leaving room for the largest candidate lag "
                "and a whole history before it, so every lag is scored on identical "
                "targets."
            ),
            "flattened_layout": "bin-major: bin 0 units 0..n, then bin 1, and so on",
            "counts_shape": "(n_samples, n_bins, n_units)",
        },
        "split_built": BUILT_SPLIT,
        "shared_targets": {
            "samples": reference_set.n_samples,
            "trials": int(reference_set.trial_rows.size),
            "target_time_s": [
                float(reference_set.target_time.min()),
                float(reference_set.target_time.max()),
            ],
            "velocity_mean": reference_set.targets.mean(axis=0).tolist(),
            "velocity_std": reference_set.targets.std(axis=0).tolist(),
        },
        "lags": {
            str(lag): {
                "samples": feature_sets[lag].n_samples,
                "feature_columns": feature_sets[lag].n_bins * feature_sets[lag].n_units,
                "units": feature_sets[lag].n_units,
                "reference_time_s": [
                    float(feature_sets[lag].reference_time.min()),
                    float(feature_sets[lag].reference_time.max()),
                ],
                "spikes_per_sample_mean": float(
                    feature_sets[lag].counts.sum(axis=(1, 2), dtype=np.int64).mean()
                ),
                "max_spikes_in_one_bin": int(feature_sets[lag].counts.max()),
                "normalization_fitted_on": normalizations[lag].fitted_on,
                "normalization_samples": normalizations[lag].n_samples,
                "normalization_features_at_floor": normalizations[lag].n_features_at_floor,
            }
            for lag in lags
        },
        "not_built": {
            "validation": "Built later, once the lag is selected from training data.",
            "held_out": "Not constructed before the frozen evaluation.",
        },
        "array_sha256": hashes,
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in checks
        ],
        "all_checks_passed": all(check.passed for check in checks),
        "artifacts": artifacts,
    }


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
    lags = tuple(args.lags) if args.lags else config.features.candidate_lags_ms

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
        feature_sets = build_session_lag_features(
            session, BUILT_SPLIT, lags=lags, spec=spec
        )
        normalizations = {lag: fit_normalization(features) for lag, features in feature_sets.items()}
    except FeatureError as exc:
        print(f"Cannot build features: {exc}")
        return 1

    artifacts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    manifests: dict[int, pd.DataFrame] = {}

    for lag in sorted(feature_sets):
        features = feature_sets[lag]
        stem = f"{BUILT_SPLIT}_lag{lag:03d}ms"
        array_path = artifact_dir / f"{stem}.npz"
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
            lag_ms=np.int64(lag),
            eligibility_lag_ms=np.int64(features.eligibility_lag_ms),
        )
        manifest = features.sample_manifest()
        manifest_path = artifact_dir / f"samples_{stem}.csv"
        manifest.to_csv(manifest_path, index=False)

        manifests[lag] = manifest
        artifacts[f"{stem}_arrays"] = _repo_relative(array_path)
        artifacts[f"{stem}_manifest"] = _repo_relative(manifest_path)
        hashes[f"{stem}_counts"] = _array_hash(features.counts)
        hashes[f"{stem}_targets"] = _array_hash(features.targets)
        logger.info("Wrote %s training samples for lag %s ms", features.n_samples, lag)

    checks = run_checks(session, spec, feature_sets, normalizations, manifests, artifact_dir)

    summary_path = results_dir / "feature_validation.json"
    artifacts["summary_json"] = _repo_relative(summary_path)

    summary = build_summary(
        session, spec, feature_sets, normalizations, checks, artifacts, hashes
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _print_report(feature_sets, checks, artifacts)
    failures = [check for check in checks if not check.passed]
    if failures:
        print(f"\nFAILED: {len(failures)} of {len(checks)} checks did not pass")
        return 1
    print(f"\nPASSED: all {len(checks)} checks")
    return 0


def _print_report(
    feature_sets: dict[int, FeatureSet], checks: list[Check], artifacts: dict[str, str]
) -> None:
    lags = sorted(feature_sets)
    shared = feature_sets[lags[0]]
    print(
        f"Shared targets  {shared.n_samples} samples across "
        f"{shared.trial_rows.size} training trials"
    )
    for lag in lags:
        features = feature_sets[lag]
        print(
            f"  lag {lag:>3} ms   counts {features.counts.shape}  "
            f"references {features.reference_time.min():.3f}"
            f"-{features.reference_time.max():.3f} s"
        )
    print("validation      not built")
    print("held-out        not built")

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
