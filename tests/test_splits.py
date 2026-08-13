"""Synthetic tests for trial split assignment and the frozen eligibility rules.

A defect here would move held-out trials into training or silently change which
trials are modelled, and neither failure would raise or look wrong in a metric.
The inputs are small hand-built tables so the expected outcome of every rule is
known independently of the recording.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_decoding.splits import (
    EXCLUDED_TRIAL_ROWS,
    LAST_TRIAL_ROW,
    SPLIT_ROW_RANGES,
    SplitError,
    assign_split,
    build_trial_table,
    check_split_integrity,
    retained_counts,
    retained_trials,
)


def _trial(
    row: int,
    *,
    result: str = "R",
    num_attempted: float = 4.0,
    cues: list[float] | None = None,
    start: float = 100.0,
    stop: float = 110.0,
) -> dict[str, object]:
    """One synthetic trial that is eligible unless an argument breaks a rule."""
    return {
        "trial_row": row,
        "start_time": start,
        "stop_time": stop,
        "result": result,
        "num_targets": 4,
        "num_attempted": num_attempted,
        "go_cue_time_array": np.asarray(
            [101.0, 103.0, 105.0, 107.0] if cues is None else cues, dtype=float
        ),
    }


def _table(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (0, "train"),
        (150, "train"),
        (151, "validation"),
        (182, "validation"),
        (183, "test"),
        (215, "test"),
    ],
)
def test_split_boundaries_match_the_frozen_ranges(row: int, expected: str) -> None:
    assert assign_split(row) == expected


@pytest.mark.parametrize("row", [-1, LAST_TRIAL_ROW + 1])
def test_rows_outside_the_session_are_rejected(row: int) -> None:
    with pytest.raises(SplitError, match="outside the frozen session range"):
        assign_split(row)


def test_each_frozen_rule_removes_only_its_own_trial() -> None:
    table = build_trial_table(
        _table(
            [
                _trial(0),
                _trial(1, result="A"),
                _trial(2, num_attempted=3.0),
                _trial(3, num_attempted=float("nan")),
                _trial(4, cues=[101.0, 103.0, 105.0, float("nan")]),
                _trial(5, cues=[101.0, 103.0, 105.0]),
                _trial(6, cues=[99.0, 103.0, 105.0, 107.0]),
                _trial(7, cues=[101.0, 103.0, 105.0, 111.0]),
            ]
        )
    )

    assert table["eligible"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert table["is_reward"].tolist() == [True, False] + [True] * 6
    assert table["has_required_attempts"].tolist() == [True, True, False, False] + [True] * 4
    assert table["has_required_finite_cues"].tolist() == [True] * 4 + [False, False, True, True]
    assert table["cues_within_trial_bounds"].tolist() == [True] * 4 + [False, True, False, False]


def test_artifact_rows_are_excluded_although_every_other_rule_passes() -> None:
    rows = sorted(EXCLUDED_TRIAL_ROWS)
    table = build_trial_table(_table([_trial(row) for row in [*rows, rows[0] + 1]]))

    assert table["is_not_excluded_row"].tolist() == [False, False, True]
    assert table["eligible"].tolist() == [False, False, True]
    # The rows still carry their split, so exclusion cannot silently move a trial.
    assert table["split"].tolist() == [assign_split(row) for row in [*rows, rows[0] + 1]]


def test_splits_are_disjoint_and_cover_every_session_row() -> None:
    table = build_trial_table(_table([_trial(row) for row in range(LAST_TRIAL_ROW + 1)]))

    check_split_integrity(table, expected_rows=range(LAST_TRIAL_ROW + 1))

    counts = {name: int((table["split"] == name).sum()) for name in SPLIT_ROW_RANGES}
    assert counts == {
        name: high - low + 1 for name, (low, high) in SPLIT_ROW_RANGES.items()
    }
    assert sum(counts.values()) == LAST_TRIAL_ROW + 1


def test_eligibility_filtering_never_changes_a_split_label() -> None:
    rows = [0, 150, 151, 182, 183, 215, *sorted(EXCLUDED_TRIAL_ROWS)]
    table = build_trial_table(
        _table(
            [
                _trial(row, result="A" if index % 3 == 0 else "R")
                for index, row in enumerate(rows)
            ]
        )
    )
    before = dict(zip(table["trial_row"], table["split"]))

    kept = retained_trials(table)

    assert not kept.empty
    assert all(before[row] == split for row, split in zip(kept["trial_row"], kept["split"]))
    assert all(assign_split(int(row)) == split for row, split in zip(kept["trial_row"], kept["split"]))
    assert EXCLUDED_TRIAL_ROWS.isdisjoint(set(kept["trial_row"]))


def test_retained_counts_report_eligible_trials_per_split() -> None:
    table = build_trial_table(
        _table(
            [
                _trial(0),
                _trial(1),
                _trial(2, result="F"),
                _trial(151),
                _trial(183),
                _trial(184, num_attempted=2.0),
            ]
        )
    )

    assert retained_counts(table) == {"train": 2, "validation": 1, "test": 1}


def test_duplicate_trial_rows_are_rejected() -> None:
    table = build_trial_table(_table([_trial(5), _trial(5)]))

    with pytest.raises(SplitError, match="appear more than once"):
        check_split_integrity(table)


def test_missing_required_columns_are_rejected() -> None:
    incomplete = _table([_trial(0)]).drop(columns=["num_attempted"])

    with pytest.raises(SplitError, match="missing required columns"):
        build_trial_table(incomplete)
