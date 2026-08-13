"""Validate the selected recording and write the dataset manifests.

Run from the repository root with the ``BCI`` environment active::

    conda deactivate
    conda activate BCI
    python -m scripts.validate_dataset

The command verifies the configured NWB file against the frozen dataset decision,
writes the unit and trial/split manifests, and writes a validation summary. These
checks run against the actual recording rather than in the test suite, so the test
suite stays independent of local data.

Exit codes: 0 when every check passes, 1 when a check fails, and 2 when the
recording is not configured or cannot be identified.
"""

from __future__ import annotations

import argparse
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
from neural_decoding.dataset import (
    DANDISET_ID,
    DANDISET_VERSION,
    EXPECTED_AREA_UNIT_COUNTS,
    EXPECTED_BEHAVIOR_SAMPLES,
    EXPECTED_FILE_SIZE_BYTES,
    EXPECTED_RETAINED_TRIALS,
    EXPECTED_SHA256,
    EXPECTED_SPIKES_REMOVED_AFTER_END,
    EXPECTED_SPIKES_REMOVED_BEFORE_START,
    EXPECTED_TRIAL_COUNT,
    EXPECTED_UNIT_COUNT,
    BEHAVIOR_SERIES,
    DatasetError,
    SessionData,
    load_session,
)
from neural_decoding.logging_utils import configure_logging
from neural_decoding.splits import (
    ELIGIBILITY_FLAGS,
    EXCLUDED_TRIAL_ROWS,
    SPLIT_NAMES,
    SplitError,
    assign_split,
    check_split_integrity,
)


@dataclass(frozen=True)
class Check:
    """One runtime-validation check against the actual recording."""

    name: str
    passed: bool
    detail: str


def run_checks(
    session: SessionData,
    *,
    unit_manifest: pd.DataFrame,
    trial_manifest: pd.DataFrame,
    manifest_paths: tuple[Path, ...],
) -> list[Check]:
    """Check the loaded session against the frozen dataset decision."""
    checks: list[Check] = []
    identity = session.identity

    checks.append(
        Check(
            "file size matches the selected asset",
            identity.size_bytes == EXPECTED_FILE_SIZE_BYTES,
            f"{identity.size_bytes} bytes",
        )
    )
    checks.append(
        Check(
            "file SHA-256 matches the selected asset",
            identity.sha256 == EXPECTED_SHA256,
            identity.sha256,
        )
    )

    units = session.units
    checks.append(
        Check(
            "sorted unit count",
            len(units) == EXPECTED_UNIT_COUNT,
            f"{len(units)} units (expected {EXPECTED_UNIT_COUNT})",
        )
    )
    area_counts = session.area_unit_counts()
    checks.append(
        Check(
            "units per cortical area",
            area_counts == EXPECTED_AREA_UNIT_COUNTS,
            f"{area_counts} (expected {EXPECTED_AREA_UNIT_COUNTS})",
        )
    )
    unit_rows = [unit.unit_row for unit in units]
    checks.append(
        Check(
            "unit order is stable and contiguous",
            unit_rows == list(range(len(units))),
            "rows 0.." + str(len(units) - 1),
        )
    )

    timestamps = session.behavior_timestamps
    checks.append(
        Check(
            "behavior sample count",
            timestamps.size == EXPECTED_BEHAVIOR_SAMPLES,
            f"{timestamps.size} samples (expected {EXPECTED_BEHAVIOR_SAMPLES})",
        )
    )
    checks.append(
        Check(
            "behavior timestamps are finite",
            bool(np.all(np.isfinite(timestamps))),
            f"[{timestamps[0]:.3f}, {timestamps[-1]:.3f}] s",
        )
    )
    checks.append(
        Check(
            "behavior timestamps strictly increase",
            bool(np.all(np.diff(timestamps) > 0)),
            f"median step {float(np.median(np.diff(timestamps))):.6f} s",
        )
    )
    checks.append(
        Check(
            "cursor series share one timebase",
            set(session.timestamps_shared_by) == set(BEHAVIOR_SERIES),
            ", ".join(session.timestamps_shared_by),
        )
    )

    for name, array in (
        ("position", session.cursor_position),
        ("velocity", session.cursor_velocity),
    ):
        checks.append(
            Check(
                f"cursor {name} shape",
                array.shape == (timestamps.size, 2),
                f"{array.shape}",
            )
        )
        checks.append(
            Check(
                f"cursor {name} values are finite",
                bool(np.all(np.isfinite(array))),
                f"{int(np.count_nonzero(~np.isfinite(array)))} non-finite values",
            )
        )

    spike_times = session.spike_times
    checks.append(
        Check(
            "spike times are finite",
            all(bool(np.all(np.isfinite(spikes))) for spikes in spike_times),
            f"{sum(spikes.size for spikes in spike_times)} spikes retained",
        )
    )
    checks.append(
        Check(
            "spike times are ordered within each unit",
            all(bool(np.all(np.diff(spikes) >= 0)) for spikes in spike_times),
            f"{len(spike_times)} units checked",
        )
    )

    trimming = session.trimming
    checks.append(
        Check(
            "trimmed spikes lie inside the common interval",
            all(
                bool(
                    np.all(
                        (spikes >= trimming.interval_start)
                        & (spikes <= trimming.interval_end)
                    )
                )
                for spikes in spike_times
            ),
            f"[{trimming.interval_start:.3f}, {trimming.interval_end:.3f}] s",
        )
    )
    checks.append(
        Check(
            "trimming removed the documented neural margins",
            trimming.n_removed_before_start == EXPECTED_SPIKES_REMOVED_BEFORE_START
            and trimming.n_removed_after_end == EXPECTED_SPIKES_REMOVED_AFTER_END,
            f"{trimming.n_removed_before_start} before start and "
            f"{trimming.n_removed_after_end} after end (expected "
            f"{EXPECTED_SPIKES_REMOVED_BEFORE_START} and "
            f"{EXPECTED_SPIKES_REMOVED_AFTER_END})",
        )
    )

    trials = session.trials
    checks.append(
        Check(
            "trial count",
            len(trials) == EXPECTED_TRIAL_COUNT,
            f"{len(trials)} trials (expected {EXPECTED_TRIAL_COUNT})",
        )
    )
    checks.append(
        Check(
            "trial identifiers match their original rows",
            bool(
                np.array_equal(
                    trials["trial_id"].to_numpy(dtype=int),
                    trials["trial_row"].to_numpy(dtype=int),
                )
            ),
            "NWB trial ids equal zero-based row numbers",
        )
    )

    try:
        check_split_integrity(trials, expected_rows=range(EXPECTED_TRIAL_COUNT))
    except SplitError as exc:
        checks.append(Check("splits are disjoint and cover every trial row", False, str(exc)))
    else:
        checks.append(
            Check(
                "splits are disjoint and cover every trial row",
                True,
                f"{len(trials)} rows across {len(SPLIT_NAMES)} splits",
            )
        )

    relabelled = [
        int(row)
        for row, split in zip(trials["trial_row"], trials["split"])
        if assign_split(int(row)) != split
    ]
    checks.append(
        Check(
            "split labels follow the original trial rows",
            not relabelled,
            "no trial was relabelled by filtering"
            if not relabelled
            else f"relabelled rows {relabelled}",
        )
    )

    retained = session.retained_trial_counts()
    checks.append(
        Check(
            "retained trials per split",
            retained == EXPECTED_RETAINED_TRIALS,
            f"{retained} (expected {EXPECTED_RETAINED_TRIALS})",
        )
    )

    retained_rows = {int(row) for row in trials.loc[trials["eligible"], "trial_row"]}
    checks.append(
        Check(
            "frozen artifact rows are excluded",
            EXCLUDED_TRIAL_ROWS.isdisjoint(retained_rows),
            f"rows {sorted(EXCLUDED_TRIAL_ROWS)} are not retained",
        )
    )

    checks.append(
        Check(
            "manifests written",
            all(path.is_file() for path in manifest_paths)
            and len(unit_manifest) == EXPECTED_UNIT_COUNT
            and len(trial_manifest) == EXPECTED_TRIAL_COUNT,
            f"{len(unit_manifest)} unit rows and {len(trial_manifest)} trial rows",
        )
    )

    return checks


def build_summary(
    session: SessionData, checks: list[Check], artifacts: dict[str, str]
) -> dict[str, object]:
    """Assemble the machine-readable validation summary."""
    trials = session.trials
    trimming = session.trimming
    timestamps = session.behavior_timestamps
    steps = np.diff(timestamps)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "dandiset_id": DANDISET_ID,
            "version": DANDISET_VERSION,
            "subject_id": session.subject_id,
            "session_identifier": session.identifier,
            "session_description": session.session_description,
            "session_start_time": session.session_start_time.isoformat(),
        },
        "file": {
            "name": session.identity.name,
            "size_bytes": session.identity.size_bytes,
            "sha256": session.identity.sha256,
        },
        "units": {
            "total": len(session.units),
            "by_area": session.area_unit_counts(),
        },
        "behavior": {
            "samples": int(timestamps.size),
            "start_s": float(timestamps[0]),
            "end_s": float(timestamps[-1]),
            "duration_s": float(timestamps[-1] - timestamps[0]),
            "median_step_s": float(np.median(steps)),
            "sampling_rate_hz": float(1.0 / np.median(steps)),
            "series_sharing_timebase": list(session.timestamps_shared_by),
        },
        "spikes": {
            "raw": trimming.n_spikes_raw,
            "trimmed": trimming.n_spikes_trimmed,
            "removed_before_start": trimming.n_removed_before_start,
            "removed_after_end": trimming.n_removed_after_end,
            "behavior_interval_s": [trimming.behavior_start, trimming.behavior_end],
            "neural_interval_s": [trimming.neural_start, trimming.neural_end],
            "common_interval_s": [trimming.interval_start, trimming.interval_end],
        },
        "trials": {
            "total": int(len(trials)),
            "eligible": int(trials["eligible"].sum()),
            "excluded_rows": sorted(EXCLUDED_TRIAL_ROWS),
            "split_row_ranges": {
                name: [
                    int(trials.loc[trials["split"] == name, "trial_row"].min()),
                    int(trials.loc[trials["split"] == name, "trial_row"].max()),
                ]
                for name in SPLIT_NAMES
            },
            "assigned_by_split": {
                name: int((trials["split"] == name).sum()) for name in SPLIT_NAMES
            },
            "retained_by_split": session.retained_trial_counts(),
            "rule_pass_counts": {
                flag: int(trials[flag].sum()) for flag in ELIGIBILITY_FLAGS
            },
        },
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in checks
        ],
        "all_checks_passed": all(check.passed for check in checks),
        "artifacts": artifacts,
    }


def render_markdown(summary: dict[str, object]) -> str:
    """Render the concise human-readable validation summary."""
    dataset = summary["dataset"]
    file_info = summary["file"]
    units = summary["units"]
    behavior = summary["behavior"]
    spikes = summary["spikes"]
    trials = summary["trials"]

    lines = [
        "# Dataset validation summary",
        "",
        f"Generated {summary['generated_at']}.",
        "",
        "## Recording",
        "",
        f"- Dandiset {dataset['dandiset_id']}, version `{dataset['version']}`",
        f"- Subject {dataset['subject_id']}, session start {dataset['session_start_time']}",
        f"- File `{file_info['name']}`, {file_info['size_bytes']} bytes",
        f"- SHA-256 `{file_info['sha256']}`",
        "",
        "## Neural and behavioral data",
        "",
        f"- {units['total']} sorted units: "
        + ", ".join(f"{count} {area}" for area, count in sorted(units["by_area"].items())),
        f"- {behavior['samples']} cursor samples over {behavior['duration_s']:.2f} s "
        f"at {behavior['sampling_rate_hz']:.2f} Hz",
        f"- Cursor series sharing one timebase: "
        + ", ".join(behavior["series_sharing_timebase"]),
        f"- Spikes {spikes['raw']} raw, {spikes['trimmed']} retained after trimming to "
        f"[{spikes['common_interval_s'][0]:.3f}, {spikes['common_interval_s'][1]:.3f}] s",
        f"- Removed {spikes['removed_before_start']} spikes before the behavioral start "
        f"and {spikes['removed_after_end']} after the behavioral end",
        "",
        "## Trials and splits",
        "",
        f"- {trials['total']} trials, {trials['eligible']} eligible after the frozen rules",
        f"- Excluded artifact rows: {trials['excluded_rows']}",
        "",
        "| Split | Original rows | Assigned | Retained |",
        "| --- | --- | --- | --- |",
    ]
    for name, bounds in trials["split_row_ranges"].items():
        lines.append(
            f"| {name} | {bounds[0]}-{bounds[1]} | {trials['assigned_by_split'][name]} "
            f"| {trials['retained_by_split'][name]} |"
        )

    lines += [
        "",
        "Eligibility rule pass counts:",
        "",
    ]
    lines += [f"- `{flag}`: {count}" for flag, count in trials["rule_pass_counts"].items()]

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
    parser.parse_args(argv)

    config = load_config()
    configure_logging(config.run.log_level)

    try:
        session = load_session(config)
    except (ConfigError, DatasetError) as exc:
        print(f"Cannot validate the dataset: {exc}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, to the selected recording described in README.md."
        )
        return 2

    results_dir = config.paths.results_dir
    manifest_dir = results_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    unit_manifest = session.unit_manifest()
    trial_manifest = session.trial_manifest()
    units_path = manifest_dir / "units.csv"
    trials_path = manifest_dir / "trials.csv"
    unit_manifest.to_csv(units_path, index=False)
    trial_manifest.to_csv(trials_path, index=False)

    checks = run_checks(
        session,
        unit_manifest=unit_manifest,
        trial_manifest=trial_manifest,
        manifest_paths=(units_path, trials_path),
    )

    summary_path = results_dir / "dataset_validation.json"
    report_path = results_dir / "dataset_validation.md"
    artifacts = {
        "unit_manifest": _repo_relative(units_path),
        "trial_manifest": _repo_relative(trials_path),
        "summary_json": _repo_relative(summary_path),
        "summary_markdown": _repo_relative(report_path),
    }

    summary = build_summary(session, checks, artifacts)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    _print_report(session, checks, artifacts)
    failures = [check for check in checks if not check.passed]
    if failures:
        print(f"\nFAILED: {len(failures)} of {len(checks)} checks did not pass")
        return 1
    print(f"\nPASSED: all {len(checks)} checks")
    return 0


def _print_report(
    session: SessionData, checks: list[Check], artifacts: dict[str, str]
) -> None:
    print(f"Recording   {session.identity.name}")
    print(f"Subject     {session.subject_id}")
    print(f"Units       {len(session.units)} {session.area_unit_counts()}")
    print(f"Trials      {len(session.trials)} total, {session.retained_trial_counts()} retained")

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
