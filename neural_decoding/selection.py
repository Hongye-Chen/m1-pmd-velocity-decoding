"""Training-only grouped cross-validation for lag and Ridge-alpha selection.

The project needs one shared neural-to-target lag for every later decoder. It is
chosen here with cross-validation over eligible training trials only, so validation
and held-out data stay untouched by the decision.

Three properties make the comparison trustworthy, and each is enforced rather than
assumed:

- trials are grouped into contiguous folds and never split, because samples 10 ms
  apart inside one reach are almost identical and splitting a trial would leave
  near-duplicates on both sides of the fold boundary;
- normalization is fitted on each fold's training trials and applied unchanged to
  its assessment trials, so a fold is never standardized using its own statistics;
- every candidate lag is scored on the same target samples, which the feature
  pipeline guarantees and this module re-checks before scoring anything.

Out-of-fold predictions are written into a preallocated array at each sample's
original position. That makes the concatenated result ordered by construction, and
it turns "every sample was assessed exactly once" into a fact that can be verified
by looking for unfilled entries rather than a property that has to be trusted.

The Ridge models fitted here are temporary selection instruments and are discarded.
The official models are fitted later on the complete training set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from neural_decoding.features import (
    FeatureSet,
    Normalization,
    fit_normalization,
    subset_by_trials,
)
from neural_decoding.logging_utils import get_logger

__all__ = [
    "N_GROUPS",
    "RIDGE_ALPHAS",
    "TIE_TOLERANCE",
    "CandidateResult",
    "CrossValidationReport",
    "Fold",
    "FoldNormalization",
    "FoldResult",
    "SelectionError",
    "contiguous_trial_groups",
    "cross_validate",
    "make_folds",
    "mean_r2",
    "r2_per_axis",
    "select_candidate",
]

logger = get_logger("selection")

RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
N_GROUPS = 5
TIE_TOLERANCE = 1e-12

# Cholesky is requested explicitly rather than left to the solver heuristic, so the
# selection does not silently change method with a different data shape or version.
_RIDGE_SOLVER = "cholesky"


class SelectionError(RuntimeError):
    """Raised when the selection procedure would produce an untrustworthy comparison."""


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold, described by whole trials."""

    index: int
    training_rows: tuple[int, ...]
    assessment_rows: tuple[int, ...]


@dataclass(frozen=True)
class FoldResult:
    """Assessment metrics for one candidate on one fold."""

    lag_ms: int
    alpha: float
    fold_index: int
    n_samples: int
    r2_x: float
    r2_y: float
    r2_mean: float


@dataclass(frozen=True)
class FoldNormalization:
    """Provenance of the statistics used to standardize one fold and lag."""

    lag_ms: int
    fold_index: int
    fitted_on: str
    training_rows: tuple[int, ...]
    assessment_rows: tuple[int, ...]
    n_samples: int


@dataclass(frozen=True)
class CandidateResult:
    """Out-of-fold result for one lag and alpha over the whole training set."""

    lag_ms: int
    alpha: float
    n_samples: int
    r2_x: float
    r2_y: float
    r2_mean: float
    fold_r2_means: tuple[float, ...]
    predictions: np.ndarray

    @property
    def fold_r2_spread(self) -> float:
        """Return the range of per-fold scores, as a stability diagnostic."""
        return float(max(self.fold_r2_means) - min(self.fold_r2_means))


@dataclass(frozen=True)
class CrossValidationReport:
    """Everything the selection produced, including its own provenance."""

    candidates: tuple[CandidateResult, ...]
    fold_results: tuple[FoldResult, ...]
    folds: tuple[Fold, ...]
    normalizations: tuple[FoldNormalization, ...]


def contiguous_trial_groups(
    trial_rows: Iterable[int], n_groups: int = N_GROUPS
) -> list[tuple[int, ...]]:
    """Split trial rows into contiguous groups of near-equal size.

    Rows are sorted first, so groups are chronological blocks of whole trials. When
    the count does not divide evenly the earliest groups take the extra trial, which
    keeps the division deterministic.
    """
    if n_groups <= 0:
        raise SelectionError(f"n_groups must be positive, got {n_groups}.")

    rows = np.asarray(sorted({int(row) for row in trial_rows}), dtype=np.int64)
    if rows.size < n_groups:
        raise SelectionError(
            f"Cannot split {rows.size} trials into {n_groups} groups; each group "
            "needs at least one trial."
        )
    return [tuple(int(row) for row in chunk) for chunk in np.array_split(rows, n_groups)]


def make_folds(trial_rows: Iterable[int], n_groups: int = N_GROUPS) -> list[Fold]:
    """Return one fold per group, assessing that group and training on the rest."""
    groups = contiguous_trial_groups(trial_rows, n_groups)
    folds: list[Fold] = []
    for index, assessment in enumerate(groups):
        training = tuple(
            row for position, group in enumerate(groups) if position != index for row in group
        )
        folds.append(
            Fold(index=index, training_rows=tuple(sorted(training)), assessment_rows=assessment)
        )
    return folds


def r2_per_axis(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the coefficient of determination for each target axis.

    The total sum of squares uses the mean of the supplied true values, so the score
    is relative to predicting that set's own mean.
    """
    true = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if true.shape != predicted.shape:
        raise SelectionError(
            f"Cannot score predictions of shape {predicted.shape} against targets of "
            f"shape {true.shape}."
        )
    if true.ndim != 2 or true.shape[0] < 2:
        raise SelectionError("Scoring needs at least two samples with one row per sample.")
    if not np.all(np.isfinite(predicted)):
        raise SelectionError("Predictions contain non-finite values.")

    residual = np.square(true - predicted).sum(axis=0)
    total = np.square(true - true.mean(axis=0)).sum(axis=0)
    if np.any(total <= 0.0):
        raise SelectionError("A target axis has no variance, so its score is undefined.")
    return 1.0 - residual / total


def mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the arithmetic mean of the per-axis scores, the selection metric."""
    return float(r2_per_axis(y_true, y_pred).mean())


def cross_validate(
    feature_sets: dict[int, FeatureSet],
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    n_groups: int = N_GROUPS,
) -> CrossValidationReport:
    """Score every lag and alpha with trial-grouped out-of-fold predictions.

    Args:
        feature_sets: One training feature set per candidate lag, all sharing the
            same target samples.
        alphas: Ridge penalties to evaluate.
        n_groups: Number of contiguous trial groups, one fold each.

    Returns:
        The candidate scores, per-fold scores, fold definitions, and the provenance
        of every fold's normalization.
    """
    if not feature_sets:
        raise SelectionError("At least one candidate lag is required.")
    if not alphas:
        raise SelectionError("At least one alpha is required.")
    if len(set(alphas)) != len(alphas):
        raise SelectionError(f"Alphas contain duplicates: {list(alphas)}")

    lags = sorted(feature_sets)
    reference = feature_sets[lags[0]]
    _require_shared_targets(feature_sets, lags, reference)

    folds = make_folds(reference.trial_rows, n_groups)
    targets = reference.targets

    # Preallocated with NaN so an unassessed sample is detectable afterwards rather
    # than silently scored as zero.
    predictions = {
        (lag, float(alpha)): np.full(targets.shape, np.nan, dtype=np.float64)
        for lag in lags
        for alpha in alphas
    }
    fold_results: list[FoldResult] = []
    normalizations: list[FoldNormalization] = []

    for fold in folds:
        assessment_mask = np.isin(
            reference.trial_row, np.asarray(fold.assessment_rows, dtype=np.int64)
        )
        positions = np.nonzero(assessment_mask)[0]
        if positions.size == 0:
            raise SelectionError(f"Fold {fold.index} has no assessment samples.")

        for lag in lags:
            features = feature_sets[lag]
            training = subset_by_trials(
                features, fold.training_rows, label=f"fold {fold.index} training"
            )
            if training.n_samples + positions.size != features.n_samples:
                raise SelectionError(
                    f"Fold {fold.index} does not partition the training samples: "
                    f"{training.n_samples} + {positions.size} != {features.n_samples}."
                )

            normalization = fit_normalization(training)
            normalizations.append(
                FoldNormalization(
                    lag_ms=lag,
                    fold_index=fold.index,
                    fitted_on=normalization.fitted_on,
                    training_rows=normalization.trial_rows,
                    assessment_rows=fold.assessment_rows,
                    n_samples=normalization.n_samples,
                )
            )
            design_train = _design_matrix(normalization, training.counts)
            design_assess = _design_matrix(normalization, features.counts[assessment_mask])
            scaled_targets = normalization.apply_targets(training.targets)
            assessment_targets = targets[assessment_mask]

            for alpha in alphas:
                model = Ridge(alpha=float(alpha), fit_intercept=True, solver=_RIDGE_SOLVER)
                model.fit(design_train, scaled_targets)
                predicted = normalization.invert_targets(model.predict(design_assess))
                predictions[(lag, float(alpha))][positions] = predicted

                scores = r2_per_axis(assessment_targets, predicted)
                fold_results.append(
                    FoldResult(
                        lag_ms=lag,
                        alpha=float(alpha),
                        fold_index=fold.index,
                        n_samples=int(positions.size),
                        r2_x=float(scores[0]),
                        r2_y=float(scores[1]),
                        r2_mean=float(scores.mean()),
                    )
                )
        logger.info(
            "Fold %s scored %s lags x %s alphas on %s samples",
            fold.index,
            len(lags),
            len(alphas),
            positions.size,
        )

    candidates: list[CandidateResult] = []
    for lag in lags:
        for alpha in alphas:
            out_of_fold = predictions[(lag, float(alpha))]
            unfilled = int(np.count_nonzero(~np.isfinite(out_of_fold)))
            if unfilled:
                raise SelectionError(
                    f"Lag {lag} ms with alpha {alpha} left {unfilled} out-of-fold "
                    "entries unfilled, so some samples were never assessed."
                )
            scores = r2_per_axis(targets, out_of_fold)
            fold_means = tuple(
                result.r2_mean
                for result in fold_results
                if result.lag_ms == lag and result.alpha == float(alpha)
            )
            candidates.append(
                CandidateResult(
                    lag_ms=lag,
                    alpha=float(alpha),
                    n_samples=int(out_of_fold.shape[0]),
                    r2_x=float(scores[0]),
                    r2_y=float(scores[1]),
                    r2_mean=float(scores.mean()),
                    fold_r2_means=fold_means,
                    predictions=out_of_fold,
                )
            )

    return CrossValidationReport(
        candidates=tuple(candidates),
        fold_results=tuple(fold_results),
        folds=tuple(folds),
        normalizations=tuple(normalizations),
    )


def select_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    """Return the winning candidate under the frozen tie-breaking rules.

    Highest out-of-fold mean score wins. Scores within :data:`TIE_TOLERANCE` of the
    best are treated as tied, and among those the smaller lag wins, then the larger
    alpha. Preferring the smaller lag keeps the shortest neural window that performs
    equally well, and preferring the larger alpha keeps the more regularized model.
    """
    if not candidates:
        raise SelectionError("There are no candidates to select from.")

    best_score = max(candidate.r2_mean for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if best_score - candidate.r2_mean <= TIE_TOLERANCE
    ]
    smallest_lag = min(candidate.lag_ms for candidate in tied)
    tied = [candidate for candidate in tied if candidate.lag_ms == smallest_lag]
    largest_alpha = max(candidate.alpha for candidate in tied)
    tied = [candidate for candidate in tied if candidate.alpha == largest_alpha]

    if len(tied) != 1:
        raise SelectionError(
            f"Tie-breaking left {len(tied)} candidates, which should be impossible "
            "for a grid of unique lag and alpha pairs."
        )
    return tied[0]


def _require_shared_targets(
    feature_sets: dict[int, FeatureSet], lags: Sequence[int], reference: FeatureSet
) -> None:
    for lag in lags[1:]:
        other = feature_sets[lag]
        if not (
            np.array_equal(other.target_index, reference.target_index)
            and np.array_equal(other.targets, reference.targets)
            and np.array_equal(other.trial_row, reference.trial_row)
        ):
            raise SelectionError(
                f"Lag {lag} ms is scored on different target samples than lag "
                f"{lags[0]} ms, so the candidates are not comparable."
            )
        if other.split != reference.split:
            raise SelectionError(
                f"Lag {lag} ms comes from split {other.split!r} rather than "
                f"{reference.split!r}."
            )


def _design_matrix(normalization: Normalization, counts: np.ndarray) -> np.ndarray:
    """Standardize counts and flatten them to bin-major columns for regression."""
    standardized = normalization.apply_features(counts)
    return standardized.reshape(standardized.shape[0], -1)
