"""Shared metrics for two-dimensional cursor velocity decoding.

Every metric is computed per axis and only then averaged, so a decoder that tracks
one axis well and the other poorly cannot hide behind a pooled score. Correlation is
undefined, not zero, for a prediction without variance.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MetricError",
    "mean_r2",
    "pearson_r_per_axis",
    "r2_per_axis",
    "rmse_per_axis",
    "summarize_predictions",
]


class MetricError(RuntimeError):
    """Raised when a score would be meaningless rather than merely poor."""


def r2_per_axis(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the coefficient of determination for each target axis.

    The total sum of squares uses the mean of the supplied true values, so the score
    is relative to predicting that set's own mean.
    """
    true, predicted = _validated_pair(y_true, y_pred)
    residual = np.square(true - predicted).sum(axis=0)
    total = np.square(true - true.mean(axis=0)).sum(axis=0)
    if np.any(total <= 0.0):
        raise MetricError("A target axis has no variance, so its score is undefined.")
    return 1.0 - residual / total


def mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the arithmetic mean of the per-axis scores, the selection metric."""
    return float(r2_per_axis(y_true, y_pred).mean())


def pearson_r_per_axis(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the Pearson correlation for each target axis, NaN where undefined."""
    true, predicted = _validated_pair(y_true, y_pred)
    centred_true = true - true.mean(axis=0)
    centred_pred = predicted - predicted.mean(axis=0)
    scale = np.sqrt(
        np.square(centred_true).sum(axis=0) * np.square(centred_pred).sum(axis=0)
    )
    # Constant axes are found from the raw spread, not the centred one: subtracting
    # the mean of many identical values leaves a last-bit residue, and dividing that
    # by its own magnitude would report a correlation computed from rounding noise.
    usable = (
        (np.ptp(true, axis=0) > 0.0) & (np.ptp(predicted, axis=0) > 0.0) & (scale > 0.0)
    )

    correlation = np.full(true.shape[1], np.nan, dtype=np.float64)
    np.divide(
        (centred_true * centred_pred).sum(axis=0),
        scale,
        out=correlation,
        where=usable,
    )
    return correlation


def rmse_per_axis(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the root-mean-square error for each target axis, in the target units."""
    true, predicted = _validated_pair(y_true, y_pred)
    return np.sqrt(np.square(true - predicted).mean(axis=0))


def summarize_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, *, axis_names: tuple[str, str] = ("x", "y")
) -> dict[str, object]:
    """Return every project metric, per axis and as the across-axis mean."""
    r2 = r2_per_axis(y_true, y_pred)
    correlation = pearson_r_per_axis(y_true, y_pred)
    rmse = rmse_per_axis(y_true, y_pred)
    if r2.size != len(axis_names):
        raise MetricError(
            f"Got {r2.size} target axes but {len(axis_names)} axis names "
            f"{list(axis_names)}."
        )

    return {
        "n_samples": int(np.asarray(y_true).shape[0]),
        "r2": _named(r2, axis_names),
        "r2_mean": float(r2.mean()),
        "pearson_r": _named(correlation, axis_names),
        "pearson_r_mean": _mean_or_none(correlation),
        "rmse_cm_s": _named(rmse, axis_names),
        "rmse_cm_s_mean": float(rmse.mean()),
    }


def _named(values: np.ndarray, axis_names: tuple[str, str]) -> dict[str, float | None]:
    return {
        name: (None if not np.isfinite(value) else float(value))
        for name, value in zip(axis_names, values)
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    """Return the mean, withheld as ``None`` when any axis value is undefined."""
    return float(values.mean()) if np.all(np.isfinite(values)) else None


def _validated_pair(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if true.shape != predicted.shape:
        raise MetricError(
            f"Cannot score predictions of shape {predicted.shape} against targets of "
            f"shape {true.shape}."
        )
    if true.ndim != 2 or true.shape[0] < 2:
        raise MetricError("Scoring needs at least two samples with one row per sample.")
    if not np.all(np.isfinite(predicted)):
        raise MetricError("Predictions contain non-finite values.")
    if not np.all(np.isfinite(true)):
        raise MetricError("Targets contain non-finite values.")
    return true, predicted
