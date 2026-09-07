"""Trial split assignment and the frozen eligibility rules for the session.

Splits are assigned from the original NWB trial row, never from a filtered or
renumbered table. Eligibility is evaluated afterwards and only removes trials, so
dropping one can never move another into a different split. Reversing that order
would leak held-out trials into training without any visible failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations

import numpy as np
import pandas as pd

__all__ = [
    "ELIGIBILITY_FLAGS",
    "EXCLUDED_TRIAL_ROWS",
    "FIRST_TRIAL_ROW",
    "LAST_TRIAL_ROW",
    "REQUIRED_ATTEMPTED_TARGETS",
    "REQUIRED_CUE_COUNT",
    "REQUIRED_TRIAL_COLUMNS",
    "REWARD_RESULT",
    "SPLIT_NAMES",
    "SPLIT_ROW_RANGES",
    "SplitError",
    "TEST",
    "TRAIN",
    "VALIDATION",
    "assign_split",
    "assign_splits",
    "build_trial_table",
    "check_split_integrity",
    "retained_counts",
    "retained_trials",
    "split_row_sets",
]

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLIT_NAMES: tuple[str, ...] = (TRAIN, VALIDATION, TEST)

# Frozen chronological ranges over the original zero-based trial rows.
SPLIT_ROW_RANGES: Mapping[str, tuple[int, int]] = {
    TRAIN: (0, 150),
    VALIDATION: (151, 182),
    TEST: (183, 215),
}

FIRST_TRIAL_ROW = 0
LAST_TRIAL_ROW = 215

# Rows carrying gross position and velocity transients, excluded before modelling.
EXCLUDED_TRIAL_ROWS = frozenset({109, 189})

REWARD_RESULT = "R"
REQUIRED_ATTEMPTED_TARGETS = 4
REQUIRED_CUE_COUNT = 4

ELIGIBILITY_FLAGS: tuple[str, ...] = (
    "is_reward",
    "has_required_attempts",
    "has_required_finite_cues",
    "cues_within_trial_bounds",
    "is_not_excluded_row",
)

REQUIRED_TRIAL_COLUMNS: tuple[str, ...] = (
    "trial_row",
    "start_time",
    "stop_time",
    "result",
    "num_attempted",
    "go_cue_time_array",
)


class SplitError(RuntimeError):
    """Raised when split assignment or split integrity is violated."""


def assign_split(trial_row: int) -> str:
    """Return the frozen split for one original trial row."""
    row = int(trial_row)
    for name, (low, high) in SPLIT_ROW_RANGES.items():
        if low <= row <= high:
            return name
    raise SplitError(
        f"Trial row {row} lies outside the frozen session range "
        f"{FIRST_TRIAL_ROW}-{LAST_TRIAL_ROW}."
    )


def assign_splits(trial_rows: Iterable[int]) -> list[str]:
    """Return the frozen split for each original trial row."""
    return [assign_split(row) for row in trial_rows]


def build_trial_table(raw_trials: pd.DataFrame) -> pd.DataFrame:
    """Attach split labels and frozen eligibility flags to a trial table.

    The input must already carry a ``trial_row`` column holding the original
    zero-based NWB row, because every split decision is made from that number
    alone.
    """
    missing = [name for name in REQUIRED_TRIAL_COLUMNS if name not in raw_trials.columns]
    if missing:
        raise SplitError(f"Trial table is missing required columns: {missing}")

    table = raw_trials.reset_index(drop=True).copy()
    rows = table["trial_row"].to_numpy(dtype=int)

    table["split"] = assign_splits(rows)

    cues = [np.asarray(value, dtype=float).ravel() for value in table["go_cue_time_array"]]
    cue_counts = np.array([cue.size for cue in cues], dtype=int)
    finite_counts = np.array([int(np.isfinite(cue).sum()) for cue in cues], dtype=int)
    starts = table["start_time"].to_numpy(dtype=float)
    stops = table["stop_time"].to_numpy(dtype=float)

    table["n_cues"] = cue_counts
    table["n_finite_cues"] = finite_counts
    table["is_reward"] = table["result"].astype(str).to_numpy() == REWARD_RESULT
    table["has_required_attempts"] = (
        table["num_attempted"].to_numpy(dtype=float) == REQUIRED_ATTEMPTED_TARGETS
    )
    table["has_required_finite_cues"] = (cue_counts == REQUIRED_CUE_COUNT) & (
        finite_counts == REQUIRED_CUE_COUNT
    )
    table["cues_within_trial_bounds"] = [
        bool(
            cue.size > 0
            and np.all(np.isfinite(cue))
            and np.all(cue >= start)
            and np.all(cue <= stop)
        )
        for cue, start, stop in zip(cues, starts, stops)
    ]
    table["is_not_excluded_row"] = ~np.isin(rows, sorted(EXCLUDED_TRIAL_ROWS))

    table["eligible"] = np.logical_and.reduce(
        [table[flag].to_numpy(dtype=bool) for flag in ELIGIBILITY_FLAGS]
    )
    return table


def retained_trials(table: pd.DataFrame) -> pd.DataFrame:
    """Return only the eligible trials, keeping their original split labels."""
    return table.loc[table["eligible"]].copy()


def retained_counts(table: pd.DataFrame) -> dict[str, int]:
    """Return the number of eligible trials in each split."""
    kept = retained_trials(table)
    return {name: int((kept["split"] == name).sum()) for name in SPLIT_NAMES}


def split_row_sets(table: pd.DataFrame) -> dict[str, set[int]]:
    """Return the original trial rows belonging to each split."""
    return {
        name: {int(row) for row in table.loc[table["split"] == name, "trial_row"]}
        for name in SPLIT_NAMES
    }


def check_split_integrity(
    table: pd.DataFrame, *, expected_rows: Iterable[int] | None = None
) -> None:
    """Raise ``SplitError`` unless every row sits in exactly one split."""
    duplicated = table["trial_row"][table["trial_row"].duplicated()].tolist()
    if duplicated:
        raise SplitError(f"Trial rows appear more than once: {sorted(set(duplicated))}")

    row_sets = split_row_sets(table)
    for first, second in combinations(SPLIT_NAMES, 2):
        overlap = row_sets[first] & row_sets[second]
        if overlap:
            raise SplitError(
                f"Splits {first!r} and {second!r} share trial rows: {sorted(overlap)}"
            )

    assigned = set().union(*row_sets.values())
    present = {int(row) for row in table["trial_row"]}
    unassigned = present - assigned
    if unassigned:
        raise SplitError(f"Trial rows were not assigned to a split: {sorted(unassigned)}")

    if expected_rows is not None:
        expected = {int(row) for row in expected_rows}
        if present != expected:
            raise SplitError(
                "Trial rows do not match the expected session rows: "
                f"missing {sorted(expected - present)}, unexpected {sorted(present - expected)}"
            )
