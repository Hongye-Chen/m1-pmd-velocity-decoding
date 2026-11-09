"""Tests for decoding metrics and axis aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from neural_decoding.metrics import (
    MetricError,
    mean_r2,
    pearson_r_per_axis,
    r2_per_axis,
    rmse_per_axis,
    summarize_predictions,
)


def test_r2_matches_values_computed_by_hand() -> None:
    truth = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0], [6.0, 40.0]])

    # A perfect prediction scores 1 on both axes.
    assert r2_per_axis(truth, truth).tolist() == [1.0, 1.0]

    # Predicting the mean of each axis scores 0.
    mean_prediction = np.tile(truth.mean(axis=0), (truth.shape[0], 1))
    assert np.allclose(r2_per_axis(truth, mean_prediction), [0.0, 0.0])

    # A known residual: total sum of squares is 20 on the first axis, and a
    # residual of 1 per sample gives 1 - 4/20 = 0.8.
    offset = truth + np.array([1.0, 0.0])
    scores = r2_per_axis(truth, offset)
    assert scores[0] == pytest.approx(0.8)
    assert scores[1] == pytest.approx(1.0)
    assert mean_r2(truth, offset) == pytest.approx(0.9)


def test_axes_are_scored_independently() -> None:
    truth = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    # Perfect on the first axis, mean-only on the second.
    predicted = np.column_stack([truth[:, 0], np.full(4, truth[:, 1].mean())])

    scores = r2_per_axis(truth, predicted)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert mean_r2(truth, predicted) == pytest.approx(0.5)


def test_the_reported_mean_averages_the_axes_instead_of_pooling_them() -> None:
    # The second axis has a hundred times the spread of the first, so a pooled score
    # would be dominated by it.
    truth = np.array([[0.0, 0.0], [1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    # Mean-only on the first axis, perfect on the second.
    predicted = np.column_stack([np.full(4, truth[:, 0].mean()), truth[:, 1]])

    residual = np.square(truth - predicted).sum(axis=0)
    total = np.square(truth - truth.mean(axis=0)).sum(axis=0)
    pooled = 1.0 - residual.sum() / total.sum()

    assert mean_r2(truth, predicted) == pytest.approx(0.5)
    # Pooling would report near-perfect performance for a decoder that learned
    # nothing about the first axis.
    assert pooled == pytest.approx(0.9999, abs=1e-4)
    assert mean_r2(truth, predicted) != pytest.approx(pooled)


def test_rmse_is_reported_per_axis_in_target_units() -> None:
    truth = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    # Errors of 3 on the first axis and 4 on the second, in every sample.
    predicted = np.tile(np.array([3.0, 4.0]), (4, 1))

    errors = rmse_per_axis(truth, predicted)

    assert errors.tolist() == [3.0, 4.0]
    # The mean is the mean of the axes, not the Euclidean norm of the error.
    assert errors.mean() == pytest.approx(3.5)


def test_correlation_ignores_amplitude_where_r2_does_not() -> None:
    truth = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # The right shape at half the amplitude: perfectly correlated, badly scaled.
    predicted = truth * 0.5

    correlation = pearson_r_per_axis(truth, predicted)
    scores = r2_per_axis(truth, predicted)

    assert np.allclose(correlation, [1.0, 1.0])
    # Squared residuals sum to 7.5 against a total sum of squares of 5, so the
    # halved amplitude scores worse than predicting the mean would have.
    assert np.allclose(scores, [-0.5, -0.5])


def test_a_constant_prediction_has_no_correlation_rather_than_zero() -> None:
    truth = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    baseline = np.tile(truth.mean(axis=0), (4, 1))

    correlation = pearson_r_per_axis(truth, baseline)

    assert np.all(np.isnan(correlation))

    summary = summarize_predictions(truth, baseline)
    # Reported as absent rather than as an uncorrelated result, with the
    # across-axis mean withheld instead of invented.
    assert summary["pearson_r"] == {"x": None, "y": None}
    assert summary["pearson_r_mean"] is None
    assert summary["r2_mean"] == pytest.approx(0.0)


def test_a_constant_prediction_stays_undefined_despite_centring_residue() -> None:
    # The constant baseline at realistic size. Subtracting the mean of thousands of
    # identical velocities does not give exactly zero, so a check on the centred
    # spread would find variance where there is none.
    rng = np.random.default_rng(0)
    truth = rng.normal(scale=7.0, size=(8183, 2))
    baseline = np.tile(truth.mean(axis=0), (truth.shape[0], 1))

    assert np.ptp(baseline, axis=0).tolist() == [0.0, 0.0]
    residue = np.abs(baseline - baseline.mean(axis=0)).max()
    assert residue > 0.0

    assert np.all(np.isnan(pearson_r_per_axis(truth, baseline)))


def test_the_summary_reports_each_axis_with_its_own_mean() -> None:
    truth = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    predicted = np.column_stack([truth[:, 0], truth[:, 1] + 1.0])

    summary = summarize_predictions(truth, predicted)

    assert summary["n_samples"] == 4
    assert summary["r2"]["x"] == pytest.approx(1.0)
    assert summary["rmse_cm_s"]["x"] == pytest.approx(0.0)
    assert summary["rmse_cm_s"]["y"] == pytest.approx(1.0)
    assert summary["rmse_cm_s_mean"] == pytest.approx(0.5)
    assert summary["r2_mean"] == pytest.approx(
        (summary["r2"]["x"] + summary["r2"]["y"]) / 2.0
    )
    assert summary["pearson_r_mean"] == pytest.approx(1.0)


def test_non_finite_predictions_are_refused() -> None:
    truth = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    predicted = truth.copy()
    predicted[1, 0] = np.nan

    with pytest.raises(MetricError, match="non-finite"):
        r2_per_axis(truth, predicted)


def test_mismatched_shapes_are_refused() -> None:
    truth = np.zeros((4, 2))

    with pytest.raises(MetricError, match="shape"):
        r2_per_axis(truth, np.zeros((3, 2)))


def test_a_target_axis_without_variance_is_refused() -> None:
    truth = np.column_stack([np.arange(4.0), np.ones(4)])

    with pytest.raises(MetricError, match="no variance"):
        r2_per_axis(truth, truth)
