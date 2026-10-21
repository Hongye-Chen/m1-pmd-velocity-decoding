"""Tests for the held-out evaluator's sample alignment, scoring, and run guards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_decoding.config import load_config
from neural_decoding.dataset import DatasetError
from neural_decoding.metrics import summarize_predictions
from scripts import evaluate_held_out as evaluator
from scripts.evaluate_held_out import EvaluationError, ModelPrediction

_TRUTH = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
_PATTERN = np.array([1.0, -1.0, 1.0, -1.0])


def _identity() -> pd.DataFrame:
    target_index = np.array([30, 31, 32, 75], dtype=np.int64)
    return pd.DataFrame(
        {
            "sample_index": np.arange(4, dtype=np.int64),
            "trial_row": np.array([183, 183, 183, 184], dtype=np.int64),
            "trial_id": np.array([183, 183, 183, 184], dtype=np.int64),
            "reference_index": target_index - 5,
            "target_index": target_index,
            "target_time": target_index * 0.01,
            "target_velocity_x": _TRUTH[:, 0],
            "target_velocity_y": _TRUTH[:, 1],
        }
    )


def _prediction(name: str, velocity: np.ndarray, identity: pd.DataFrame) -> ModelPrediction:
    return ModelPrediction(
        name=name,
        trial_row=identity["trial_row"].to_numpy(),
        target_index=identity["target_index"].to_numpy(),
        target_time=identity["target_time"].to_numpy(),
        velocity=np.asarray(velocity, dtype=np.float64),
    )


def _table(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    identity = _identity()
    return evaluator.build_prediction_table(
        identity,
        [_prediction(name, velocity, identity) for name, velocity in predictions.items()],
    )


def _swap_first_rows(prediction: ModelPrediction) -> ModelPrediction:
    order = np.array([1, 0, 2, 3])
    return replace(
        prediction,
        trial_row=prediction.trial_row[order],
        target_index=prediction.target_index[order],
        target_time=prediction.target_time[order],
        velocity=prediction.velocity[order],
    )


def _drop_last_row(prediction: ModelPrediction) -> ModelPrediction:
    return replace(
        prediction,
        trial_row=prediction.trial_row[:-1],
        target_index=prediction.target_index[:-1],
        target_time=prediction.target_time[:-1],
        velocity=prediction.velocity[:-1],
    )


def _shift_time(prediction: ModelPrediction) -> ModelPrediction:
    return replace(prediction, target_time=prediction.target_time + 0.01)


def _other_trial(prediction: ModelPrediction) -> ModelPrediction:
    rows = prediction.trial_row.copy()
    rows[-1] = 185
    return replace(prediction, trial_row=rows)


def _non_finite(prediction: ModelPrediction) -> ModelPrediction:
    velocity = prediction.velocity.copy()
    velocity[2, 1] = np.nan
    return replace(prediction, velocity=velocity)


def _non_finite_extra(prediction: ModelPrediction) -> ModelPrediction:
    return replace(prediction, extra={"posterior_variance_x": np.array([1.0, 1.0, np.inf, 1.0])})


def test_every_model_is_tabulated_on_the_common_rows() -> None:
    identity = _identity()
    kalman = replace(
        _prediction("kalman", _TRUTH + 0.5, identity),
        extra={"posterior_variance_x": np.array([1.0, 2.0, 3.0, 4.0])},
    )
    table = evaluator.build_prediction_table(
        identity, [_prediction("ridge", _TRUTH, identity), kalman]
    )

    assert list(table.columns[:8]) == list(identity.columns)
    assert np.array_equal(table["ridge_velocity_y"], _TRUTH[:, 1])
    assert np.array_equal(table["kalman_velocity_x"], _TRUTH[:, 0] + 0.5)
    assert np.array_equal(table["kalman_posterior_variance_x"], [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "corrupt",
    [_swap_first_rows, _drop_last_row, _shift_time, _other_trial, _non_finite, _non_finite_extra],
)
def test_a_misaligned_model_is_refused(
    corrupt: Callable[[ModelPrediction], ModelPrediction],
) -> None:
    identity = _identity()
    aligned = _prediction("ridge", _TRUTH, identity)
    corrupted = corrupt(_prediction("gru_seed1", _TRUTH, identity))

    with pytest.raises(EvaluationError, match="gru_seed1"):
        evaluator.build_prediction_table(identity, [aligned, corrupted])


def test_duplicate_model_names_are_refused() -> None:
    identity = _identity()
    prediction = _prediction("ridge", _TRUTH, identity)

    with pytest.raises(EvaluationError, match="unique"):
        evaluator.build_prediction_table(identity, [prediction, prediction])


def test_each_model_is_scored_on_its_own_columns() -> None:
    off_by_one = _TRUTH.copy()
    off_by_one[3, 0] += 1.0
    table = _table({"exact": _TRUTH, "biased": off_by_one})

    metrics = evaluator.score_table(table, ["exact", "biased"])

    assert metrics["exact"]["r2_mean"] == pytest.approx(1.0)
    assert metrics["exact"]["rmse_cm_s_mean"] == pytest.approx(0.0)
    # Horizontal truth has total sum of squares 5; one unit error leaves R2 = 0.8.
    assert metrics["biased"]["r2"]["x"] == pytest.approx(0.8)
    assert metrics["biased"]["r2"]["y"] == pytest.approx(1.0)
    assert metrics["biased"]["rmse_cm_s"]["x"] == pytest.approx(0.5)
    assert metrics["biased"]["n_samples"] == 4


def test_gru_seeds_are_aggregated_by_metric_not_by_prediction() -> None:
    errors = np.column_stack([_PATTERN, 0.25 * _PATTERN])
    seeds = {
        "gru_seed0": _TRUTH + errors,
        "gru_seed1": _TRUTH - errors,
        "gru_seed2": _TRUTH + 0.5 * errors,
    }
    table = _table(seeds)
    metrics = evaluator.score_table(table, list(seeds))

    aggregate = evaluator.aggregate_seed_metrics(
        {seed: metrics[f"gru_seed{seed}"] for seed in (0, 1, 2)}
    )

    seed_r2 = [metrics[name]["r2_mean"] for name in seeds]
    seed_rmse_x = [metrics[name]["rmse_cm_s"]["x"] for name in seeds]
    assert aggregate["seed_mean"]["r2_mean"] == pytest.approx(np.mean(seed_r2))
    assert aggregate["seed_std"]["r2_mean"] == pytest.approx(np.std(seed_r2, ddof=1))
    assert aggregate["seed_mean"]["rmse_cm_s"]["x"] == pytest.approx(np.mean(seed_rmse_x))
    assert aggregate["seed_std"]["rmse_cm_s"]["x"] == pytest.approx(
        np.std(seed_rmse_x, ddof=1)
    )

    ensemble = summarize_predictions(_TRUTH, np.mean(list(seeds.values()), axis=0))
    assert aggregate["seed_mean"]["r2_mean"] < ensemble["r2_mean"] - 0.1


def test_seed_aggregation_needs_more_than_one_seed() -> None:
    metrics = evaluator.score_table(_table({"gru_seed0": _TRUTH}), ["gru_seed0"])

    with pytest.raises(EvaluationError, match="two seeds"):
        evaluator.aggregate_seed_metrics({0: metrics["gru_seed0"]})


def _forbid_data_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("The evaluator reached data or models before its guards.")

    for name in (
        "load_config",
        "load_session",
        "load_official_models",
        "build_split_features",
        "build_trial_bounds",
        "build_sequences",
    ):
        monkeypatch.setattr(evaluator, name, forbidden)


def test_evaluation_refuses_to_run_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_data_access(monkeypatch)
    output = tmp_path / "held_out"

    assert evaluator.main(["--output-dir", str(output)]) == 2
    assert not output.exists()


def test_an_unverified_recording_stops_before_held_out_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_data_access(monkeypatch)
    monkeypatch.setattr(
        evaluator,
        "load_config",
        lambda: load_config(local_config_path=tmp_path / "absent.yaml", use_env=False),
    )

    def wrong_recording(*args: object, **kwargs: object) -> None:
        raise DatasetError("The configured path points at a different file.")

    monkeypatch.setattr(evaluator, "load_session", wrong_recording)
    output = tmp_path / "held_out"

    assert evaluator.main(["--allow-held-out", "--output-dir", str(output)]) == 2
    assert not output.exists()


def test_an_existing_output_directory_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_data_access(monkeypatch)
    output = tmp_path / "held_out"
    output.mkdir()

    assert evaluator.main(["--allow-held-out", "--output-dir", str(output)]) == 2
    assert list(output.iterdir()) == []
