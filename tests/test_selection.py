"""Tests for lag and Ridge alpha selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_decoding.features import build_lag_features
from neural_decoding.selection import (
    N_GROUPS,
    TIE_TOLERANCE,
    CandidateResult,
    SelectionError,
    contiguous_trial_groups,
    cross_validate,
    make_folds,
    select_candidate,
)
from neural_decoding.splits import build_trial_table

SAMPLE_RATE_HZ = 100.0


def _candidate(lag_ms: int, alpha: float, r2_mean: float) -> CandidateResult:
    return CandidateResult(
        lag_ms=lag_ms,
        alpha=alpha,
        n_samples=10,
        r2_x=r2_mean,
        r2_y=r2_mean,
        r2_mean=r2_mean,
        fold_r2_means=(r2_mean,),
        predictions=np.zeros((10, 2)),
    )


def _synthetic_feature_sets(n_trials: int = 5, lags=(0, 100)):
    """Build small training feature sets whose targets track their own timestamps."""
    timestamps = np.arange(4000, dtype=np.float64) / SAMPLE_RATE_HZ
    rng = np.random.default_rng(0)
    velocity = np.column_stack(
        [np.sin(timestamps), np.cos(timestamps)]
    ) + rng.normal(scale=0.01, size=(timestamps.size, 2))

    records = []
    for row in range(n_trials):
        start = 1.005 + row * 5.0
        stop = start + 3.0
        records.append(
            {
                "trial_row": row,
                "start_time": start,
                "stop_time": stop,
                "result": "R",
                "num_targets": 4,
                "num_attempted": 4.0,
                "go_cue_time_array": np.asarray(
                    [start + 3.0 * fraction for fraction in (0.1, 0.3, 0.5, 0.7)],
                    dtype=np.float64,
                ),
            }
        )
    trials = build_trial_table(pd.DataFrame.from_records(records))

    spikes = [
        np.sort(rng.uniform(0.0, 40.0, size=900)),
        np.sort(rng.uniform(0.0, 40.0, size=1200)),
    ]
    return build_lag_features(
        timestamps=timestamps,
        velocity=velocity,
        spike_times=spikes,
        trials=trials,
        split="train",
        lags=lags,
    )


def test_groups_are_contiguous_and_partition_every_trial() -> None:
    rows = list(range(123))

    groups = contiguous_trial_groups(rows, N_GROUPS)

    assert [len(group) for group in groups] == [25, 25, 25, 24, 24]
    assert sum(len(group) for group in groups) == 123
    # Contiguous chronological blocks, in order, with no trial repeated or dropped.
    assert [row for group in groups for row in group] == rows
    for group in groups:
        assert list(group) == list(range(group[0], group[-1] + 1))


def test_groups_use_the_original_rows_not_positions() -> None:
    rows = [3, 7, 11, 20, 41, 55]

    groups = contiguous_trial_groups(rows, 3)

    assert groups == [(3, 7), (11, 20), (41, 55)]


def test_too_few_trials_for_the_group_count_is_refused() -> None:
    with pytest.raises(SelectionError, match="at least one trial"):
        contiguous_trial_groups([1, 2], 5)


def test_each_fold_assesses_one_group_and_trains_on_the_rest() -> None:
    rows = list(range(123))

    folds = make_folds(rows, N_GROUPS)

    assert len(folds) == N_GROUPS
    assessed: list[int] = []
    for fold in folds:
        assert set(fold.training_rows).isdisjoint(fold.assessment_rows)
        assert set(fold.training_rows) | set(fold.assessment_rows) == set(rows)
        assessed.extend(fold.assessment_rows)
    # Every trial is assessed exactly once across the folds.
    assert sorted(assessed) == rows


def test_the_clear_winner_is_selected() -> None:
    candidates = [
        _candidate(0, 1.0, 0.10),
        _candidate(100, 10.0, 0.50),
        _candidate(200, 100.0, 0.30),
    ]

    assert select_candidate(candidates).lag_ms == 100


def test_ties_prefer_the_smaller_lag() -> None:
    candidates = [
        _candidate(150, 10.0, 0.50),
        _candidate(50, 10.0, 0.50),
        _candidate(100, 10.0, 0.50),
    ]

    assert select_candidate(candidates).lag_ms == 50


def test_scores_within_the_tolerance_count_as_tied() -> None:
    candidates = [
        # Marginally the highest score, but not by a meaningful amount.
        _candidate(200, 10.0, 0.50 + TIE_TOLERANCE / 2),
        _candidate(50, 10.0, 0.50),
    ]

    assert select_candidate(candidates).lag_ms == 50


def test_a_difference_beyond_the_tolerance_is_respected() -> None:
    candidates = [
        _candidate(200, 10.0, 0.50 + 1e-9),
        _candidate(50, 10.0, 0.50),
    ]

    assert select_candidate(candidates).lag_ms == 200


def test_a_remaining_tie_prefers_the_larger_alpha() -> None:
    candidates = [
        _candidate(100, 0.1, 0.50),
        _candidate(100, 1000.0, 0.50),
        _candidate(100, 10.0, 0.50),
    ]

    selected = select_candidate(candidates)

    assert selected.lag_ms == 100
    assert selected.alpha == 1000.0


def test_cross_validation_assesses_every_sample_exactly_once() -> None:
    feature_sets = _synthetic_feature_sets()

    report = cross_validate(feature_sets, alphas=(1.0, 100.0), n_groups=N_GROUPS)
    candidates, fold_results = report.candidates, report.fold_results

    reference = feature_sets[min(feature_sets)]
    assert len(report.folds) == N_GROUPS
    assert len(candidates) == len(feature_sets) * 2
    assert len(fold_results) == len(feature_sets) * 2 * N_GROUPS
    # One normalization per lag and fold, never per alpha.
    assert len(report.normalizations) == len(feature_sets) * N_GROUPS

    for candidate in candidates:
        # No NaN survives, so every sample received exactly one out-of-fold value.
        assert candidate.predictions.shape == reference.targets.shape
        assert np.all(np.isfinite(candidate.predictions))
        assert candidate.n_samples == reference.n_samples

    # Fold sample counts add up to the whole training set.
    for lag in feature_sets:
        counts = [
            result.n_samples
            for result in fold_results
            if result.lag_ms == lag and result.alpha == 1.0
        ]
        assert sum(counts) == reference.n_samples


def test_cross_validation_scores_every_lag_on_the_same_targets() -> None:
    feature_sets = _synthetic_feature_sets(lags=(0, 100, 200))

    candidates = cross_validate(feature_sets, alphas=(1.0,), n_groups=N_GROUPS).candidates

    # Each candidate is scored against one shared target array, so the sample count
    # and the fold structure cannot differ between lags.
    assert len({candidate.n_samples for candidate in candidates}) == 1
    assert {candidate.lag_ms for candidate in candidates} == {0, 100, 200}


def test_mismatched_target_sets_are_refused() -> None:
    feature_sets = _synthetic_feature_sets(lags=(0, 100))
    lags = sorted(feature_sets)
    # Break the shared-target guarantee the feature pipeline provides.
    tampered = dict(feature_sets)
    victim = tampered[lags[1]]
    tampered[lags[1]] = type(victim)(
        **{
            **{field: getattr(victim, field) for field in victim.__dataclass_fields__},
            "targets": victim.targets + 1.0,
        }
    )

    with pytest.raises(SelectionError, match="different target samples"):
        cross_validate(tampered, alphas=(1.0,), n_groups=N_GROUPS)


def test_each_fold_is_normalized_without_its_assessment_trials() -> None:
    feature_sets = _synthetic_feature_sets()
    reference = feature_sets[min(feature_sets)]
    folds = make_folds(reference.trial_rows, N_GROUPS)

    from neural_decoding.features import fit_normalization, subset_by_trials

    for fold in folds:
        training = subset_by_trials(
            reference, fold.training_rows, label=f"fold {fold.index} training"
        )
        normalization = fit_normalization(training)

        # The statistics come from the fold's training trials and no others.
        assert set(normalization.trial_rows) == set(fold.training_rows)
        assert set(normalization.trial_rows).isdisjoint(fold.assessment_rows)
        assert normalization.fitted_on == f"train[fold {fold.index} training]"


def test_duplicate_alphas_are_refused() -> None:
    feature_sets = _synthetic_feature_sets(lags=(0,))

    with pytest.raises(SelectionError, match="duplicates"):
        cross_validate(feature_sets, alphas=(1.0, 1.0), n_groups=N_GROUPS)
