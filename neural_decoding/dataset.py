"""Loading of the selected DANDI 000688 recording session.

Only the recording named in the frozen dataset decision is accepted, and its size
and SHA-256 are verified before any data is read. Spike times are trimmed to the
interval covered by both the neural and behavioral recordings, since spikes outside
it have no cursor measurement to align against.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from neural_decoding.config import ProjectConfig, load_config
from neural_decoding.logging_utils import get_logger
from neural_decoding.splits import (
    REQUIRED_CUE_COUNT,
    build_trial_table,
    retained_counts,
)

__all__ = [
    "AREA_BY_ELECTRODE_LOCATION",
    "DANDISET_ID",
    "DANDISET_VERSION",
    "EXPECTED_AREA_UNIT_COUNTS",
    "EXPECTED_BEHAVIOR_SAMPLES",
    "EXPECTED_FILE_SIZE_BYTES",
    "EXPECTED_RETAINED_TRIALS",
    "EXPECTED_SHA256",
    "EXPECTED_SPIKES_REMOVED_AFTER_END",
    "EXPECTED_SPIKES_REMOVED_BEFORE_START",
    "EXPECTED_TRIAL_COUNT",
    "EXPECTED_UNIT_COUNT",
    "BEHAVIOR_SERIES",
    "DatasetError",
    "FileIdentity",
    "SessionData",
    "TrimmingReport",
    "UnitInfo",
    "compute_sha256",
    "load_session",
    "verify_file_identity",
]

logger = get_logger("dataset")

DANDISET_ID = "000688"
DANDISET_VERSION = "0.250122.1735"
SUBJECT = "T"
SESSION = "RT-20130906"
NWB_FILE_NAME = "sub-T_ses-RT-20130906_behavior+ecephys.nwb"

EXPECTED_FILE_SIZE_BYTES = 41_528_428
EXPECTED_SHA256 = "83befb05b5d265c92e03c544c420e2bc0b6c7d3248fa14492c17f39a8aa1f124"

EXPECTED_UNIT_COUNT = 53
EXPECTED_AREA_UNIT_COUNTS = {"M1": 12, "PMd": 41}
EXPECTED_BEHAVIOR_SAMPLES = 107_316
EXPECTED_TRIAL_COUNT = 216
EXPECTED_RETAINED_TRIALS = {"train": 123, "validation": 24, "test": 29}
EXPECTED_SPIKES_REMOVED_BEFORE_START = 5
EXPECTED_SPIKES_REMOVED_AFTER_END = 277

AREA_BY_ELECTRODE_LOCATION = {
    "Primary Motor Cortex": "M1",
    "Dorsal Premotor Cortex": "PMd",
}

BEHAVIOR_SERIES: tuple[str, ...] = ("cursor_pos", "cursor_vel", "cursor_acc")

_HASH_CHUNK_BYTES = 1 << 20


class DatasetError(RuntimeError):
    """Raised when the recording is absent, misidentified, or internally inconsistent."""


@dataclass(frozen=True)
class FileIdentity:
    """Identity of the recording on disk."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class UnitInfo:
    """Metadata for one sorted unit, in stable NWB row order."""

    unit_row: int
    unit_id: int
    electrode_id: int
    electrode_location: str
    area: str
    group_name: str
    n_spikes_raw: int
    n_spikes_trimmed: int
    first_spike_time: float
    last_spike_time: float


@dataclass(frozen=True)
class TrimmingReport:
    """Result of restricting spikes to the common neural-behavior interval."""

    behavior_start: float
    behavior_end: float
    neural_start: float
    neural_end: float
    interval_start: float
    interval_end: float
    n_spikes_raw: int
    n_spikes_trimmed: int
    n_removed_before_start: int
    n_removed_after_end: int


@dataclass(frozen=True)
class SessionData:
    """One fully loaded recording session."""

    identity: FileIdentity
    identifier: str
    session_description: str
    session_start_time: datetime
    subject_id: str
    units: tuple[UnitInfo, ...]
    spike_times: tuple[np.ndarray, ...]
    behavior_timestamps: np.ndarray
    cursor_position: np.ndarray
    cursor_velocity: np.ndarray
    cursor_acceleration: np.ndarray
    timestamps_shared_by: tuple[str, ...]
    trials: pd.DataFrame
    trimming: TrimmingReport

    def area_unit_counts(self) -> dict[str, int]:
        """Return the number of units recorded in each cortical area."""
        counts: dict[str, int] = {}
        for unit in self.units:
            counts[unit.area] = counts.get(unit.area, 0) + 1
        return counts

    def retained_trial_counts(self) -> dict[str, int]:
        """Return the number of eligible trials in each split."""
        return retained_counts(self.trials)

    def unit_manifest(self) -> pd.DataFrame:
        """Return the machine-readable unit manifest."""
        return pd.DataFrame([asdict(unit) for unit in self.units])

    def trial_manifest(self) -> pd.DataFrame:
        """Return the machine-readable trial and split manifest."""
        table = self.trials.copy()
        cues = [np.asarray(value, dtype=float).ravel() for value in table["go_cue_time_array"]]
        for position in range(REQUIRED_CUE_COUNT):
            table[f"go_cue_{position + 1}"] = [
                float(cue[position]) if position < cue.size else np.nan for cue in cues
            ]
        table = table.drop(columns=["go_cue_time_array"])

        leading = ["trial_id", "trial_row", "split", "start_time", "stop_time"]
        ordered = leading + [name for name in table.columns if name not in leading]
        return table[ordered]


def compute_sha256(path: Path | str, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_identity(
    path: Path | str,
    *,
    expected_size: int = EXPECTED_FILE_SIZE_BYTES,
    expected_sha256: str = EXPECTED_SHA256,
) -> FileIdentity:
    """Confirm that the file on disk is the selected recording."""
    resolved = Path(path)
    if not resolved.is_file():
        raise DatasetError(f"The configured NWB file does not exist: {resolved}")

    size = resolved.stat().st_size
    if size != expected_size:
        raise DatasetError(
            f"{resolved.name} is {size} bytes but the selected recording is "
            f"{expected_size} bytes. The configured path points at a different file."
        )

    logger.info("Verifying SHA-256 of %s", resolved.name)
    digest = compute_sha256(resolved)
    if digest != expected_sha256:
        raise DatasetError(
            f"{resolved.name} has SHA-256 {digest} but the selected recording is "
            f"{expected_sha256}. The configured path points at a different file."
        )
    return FileIdentity(name=resolved.name, size_bytes=size, sha256=digest)


def load_session(
    config: ProjectConfig | None = None,
    *,
    path: Path | str | None = None,
    verify_identity: bool = True,
) -> SessionData:
    """Load the selected session: units, cursor behavior, trials, and splits.

    Args:
        config: Configuration used to resolve the recording; loaded when omitted.
        path: Explicit recording path, overriding the configuration.
        verify_identity: Whether to check size and SHA-256 before reading.
    """
    resolved = Path(path) if path is not None else (config or load_config()).require_nwb_file()

    if verify_identity:
        identity = verify_file_identity(resolved)
    else:
        if not resolved.is_file():
            raise DatasetError(f"The configured NWB file does not exist: {resolved}")
        identity = FileIdentity(
            name=resolved.name, size_bytes=resolved.stat().st_size, sha256=""
        )

    logger.info("Reading %s", resolved.name)
    with NWBHDF5IO(str(resolved), mode="r") as io:
        nwb = io.read()
        metadata = _read_metadata(nwb)
        unit_records, raw_spikes = _read_units(nwb)
        timestamps, position, velocity, acceleration, shared_by = _read_behavior(nwb)
        raw_trials = _read_trials(nwb)

    trimming, spike_times = _trim_spikes(raw_spikes, timestamps)
    units = tuple(
        UnitInfo(
            **record,
            n_spikes_raw=int(raw.size),
            n_spikes_trimmed=int(trimmed.size),
            first_spike_time=float(trimmed[0]) if trimmed.size else float("nan"),
            last_spike_time=float(trimmed[-1]) if trimmed.size else float("nan"),
        )
        for record, raw, trimmed in zip(unit_records, raw_spikes, spike_times)
    )

    trials = build_trial_table(raw_trials)
    logger.info(
        "Loaded %d units, %d behavior samples, %d trials",
        len(units),
        timestamps.size,
        len(trials),
    )

    return SessionData(
        identity=identity,
        units=units,
        spike_times=spike_times,
        behavior_timestamps=timestamps,
        cursor_position=position,
        cursor_velocity=velocity,
        cursor_acceleration=acceleration,
        timestamps_shared_by=shared_by,
        trials=trials,
        trimming=trimming,
        **metadata,
    )


def _read_metadata(nwb) -> dict[str, object]:
    subject = nwb.subject
    return {
        "identifier": str(nwb.identifier),
        "session_description": str(nwb.session_description),
        "session_start_time": nwb.session_start_time,
        "subject_id": str(subject.subject_id) if subject is not None else "",
    }


def _read_units(nwb) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    units = nwb.units
    if units is None or len(units) == 0:
        raise DatasetError("The recording contains no sorted units.")

    unit_ids = np.asarray(units.id[:], dtype=int)
    records: list[dict[str, object]] = []
    spikes: list[np.ndarray] = []

    for row in range(len(units)):
        region = units["electrodes"][row]
        if len(region) != 1:
            raise DatasetError(
                f"Unit row {row} references {len(region)} electrodes; expected exactly one."
            )
        location = str(region["location"].iloc[0])
        area = AREA_BY_ELECTRODE_LOCATION.get(location)
        if area is None:
            raise DatasetError(
                f"Unit row {row} sits on an electrode with unexpected location {location!r}."
            )
        records.append(
            {
                "unit_row": row,
                "unit_id": int(unit_ids[row]),
                "electrode_id": int(region.index[0]),
                "electrode_location": location,
                "area": area,
                "group_name": str(region["group_name"].iloc[0]),
            }
        )
        spikes.append(np.asarray(units["spike_times"][row], dtype=float))

    return records, spikes


def _read_behavior(
    nwb,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    behavior = nwb.processing.get("behavior")
    if behavior is None:
        raise DatasetError("The recording has no 'behavior' processing module.")

    position_series = behavior["Position"].spatial_series["cursor_pos"]
    velocity_series = behavior["Velocity"].time_series["cursor_vel"]
    acceleration_series = behavior["Acceleration"].time_series["cursor_acc"]

    timestamps = np.asarray(position_series.timestamps[:], dtype=float)
    shared = ["cursor_pos"]
    for name, series in (
        ("cursor_vel", velocity_series),
        ("cursor_acc", acceleration_series),
    ):
        other = np.asarray(series.timestamps[:], dtype=float)
        if not np.array_equal(timestamps, other):
            raise DatasetError(
                f"{name} timestamps differ from cursor_pos timestamps, so the cursor "
                "series cannot be treated as aligned."
            )
        shared.append(name)

    position = np.asarray(position_series.data[:], dtype=float)
    velocity = np.asarray(velocity_series.data[:], dtype=float)
    acceleration = np.asarray(acceleration_series.data[:], dtype=float)
    for name, array in (
        ("cursor_pos", position),
        ("cursor_vel", velocity),
        ("cursor_acc", acceleration),
    ):
        if array.shape != (timestamps.size, 2):
            raise DatasetError(
                f"{name} has shape {array.shape}; expected ({timestamps.size}, 2) "
                "on the shared cursor timebase."
            )
        if not np.all(np.isfinite(array)):
            raise DatasetError(f"{name} contains non-finite values.")
    return timestamps, position, velocity, acceleration, tuple(shared)


def _read_trials(nwb) -> pd.DataFrame:
    if nwb.trials is None or len(nwb.trials) == 0:
        raise DatasetError("The recording contains no trials table.")

    table = nwb.trials.to_dataframe()
    keep = ["start_time", "stop_time", "result", "num_targets", "num_attempted"]
    missing = [name for name in [*keep, "go_cue_time_array"] if name not in table.columns]
    if missing:
        raise DatasetError(f"The trials table is missing required columns: {missing}")

    trials = table[keep].copy()
    trials.insert(0, "trial_id", np.asarray(table.index, dtype=int))
    # The original NWB row is the only basis for split assignment, so it is captured
    # here, before any filtering can reorder the table.
    trials.insert(1, "trial_row", np.arange(len(table), dtype=int))
    trials["result"] = [str(value) for value in table["result"]]
    trials["go_cue_time_array"] = [
        np.asarray(value, dtype=float).ravel() for value in table["go_cue_time_array"]
    ]
    return trials.reset_index(drop=True)


def _trim_spikes(
    raw_spikes: list[np.ndarray], timestamps: np.ndarray
) -> tuple[TrimmingReport, tuple[np.ndarray, ...]]:
    if timestamps.size == 0:
        raise DatasetError("The recording contains no behavioral timestamps.")

    populated = [spikes for spikes in raw_spikes if spikes.size]
    if not populated:
        raise DatasetError("Every sorted unit is silent.")

    behavior_start = float(timestamps[0])
    behavior_end = float(timestamps[-1])
    neural_start = float(min(spikes.min() for spikes in populated))
    neural_end = float(max(spikes.max() for spikes in populated))

    interval_start = max(behavior_start, neural_start)
    interval_end = min(behavior_end, neural_end)
    if not interval_start < interval_end:
        raise DatasetError(
            "The neural and behavioral recordings do not overlap: "
            f"[{neural_start}, {neural_end}] versus [{behavior_start}, {behavior_end}]."
        )

    trimmed: list[np.ndarray] = []
    removed_before = 0
    removed_after = 0
    for spikes in raw_spikes:
        removed_before += int(np.count_nonzero(spikes < interval_start))
        removed_after += int(np.count_nonzero(spikes > interval_end))
        trimmed.append(spikes[(spikes >= interval_start) & (spikes <= interval_end)])

    report = TrimmingReport(
        behavior_start=behavior_start,
        behavior_end=behavior_end,
        neural_start=neural_start,
        neural_end=neural_end,
        interval_start=interval_start,
        interval_end=interval_end,
        n_spikes_raw=int(sum(spikes.size for spikes in raw_spikes)),
        n_spikes_trimmed=int(sum(spikes.size for spikes in trimmed)),
        n_removed_before_start=removed_before,
        n_removed_after_end=removed_after,
    )
    return report, tuple(trimmed)
