"""Tests for constant and Ridge model fitting and serialization."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from neural_decoding.features import DEFAULT_SPEC, FeatureSet, subset_by_trials
from neural_decoding.metrics import r2_per_axis
from neural_decoding.models import ConstantBaseline, ModelError, RidgeDecoder

N_UNITS = 3
LAG_MS = 50


def _feature_set(
    split: str,
    counts: np.ndarray,
    targets: np.ndarray,
    *,
    n_trials: int = 4,
    subset_label: str | None = None,
) -> FeatureSet:
    """Build a feature set from synthetic counts and targets."""
    n_samples = counts.shape[0]
    index = np.arange(n_samples, dtype=np.int64)
    return FeatureSet(
        split=split,
        spec=DEFAULT_SPEC,
        lag_ms=LAG_MS,
        eligibility_lag_ms=LAG_MS,
        counts=counts.astype(np.uint16),
        targets=targets.astype(np.float64),
        reference_index=index,
        target_index=index + 5,
        reference_time=index / 100.0,
        target_time=(index + 5) / 100.0,
        trial_row=index % n_trials,
        unit_ids=np.arange(counts.shape[2], dtype=np.int64),
        subset_label=subset_label,
    )


def _synthetic_split(split: str, *, seed: int, n_samples: int = 400) -> FeatureSet:
    """Return a feature set whose velocity is a linear readout of its spike counts."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(4.0, size=(n_samples, DEFAULT_SPEC.n_bins, N_UNITS))
    weights = np.array([1.0, -0.5, 0.25])
    recent = counts[:, -1, :].astype(np.float64)
    targets = np.column_stack(
        [recent @ weights, recent @ weights[::-1]]
    ) + rng.normal(scale=0.5, size=(n_samples, 2))
    return _feature_set(split, counts, targets)


def test_the_constant_baseline_is_the_mean_of_the_training_targets() -> None:
    training = _synthetic_split("train", seed=0)

    baseline = ConstantBaseline.fit(training)

    assert np.allclose(baseline.velocity, training.targets.mean(axis=0))
    assert baseline.fitted_on == "train"
    assert baseline.n_samples == training.n_samples


def test_the_constant_baseline_predicts_one_velocity_for_every_sample() -> None:
    training = _synthetic_split("train", seed=0)
    validation = _synthetic_split("validation", seed=1)

    predictions = ConstantBaseline.fit(training).predict(validation)

    assert predictions.shape == validation.targets.shape
    assert np.all(predictions == predictions[0])


def test_the_constant_baseline_scores_below_zero_on_a_shifted_evaluation_set() -> None:
    training = _synthetic_split("train", seed=0)
    validation = _synthetic_split("validation", seed=1)
    shifted = _feature_set(
        "validation", validation.counts, validation.targets + 5.0
    )

    baseline = ConstantBaseline.fit(training)

    # Fitted on the evaluation set it would score zero by construction, so a
    # negative score is the evidence that it was not.
    assert np.all(r2_per_axis(shifted.targets, baseline.predict(shifted)) < 0.0)


def test_fitting_the_constant_baseline_on_validation_is_refused() -> None:
    validation = _synthetic_split("validation", seed=1)

    with pytest.raises(ModelError, match="must be fitted on 'train'"):
        ConstantBaseline.fit(validation)


def test_fitting_the_ridge_decoder_on_validation_is_refused() -> None:
    validation = _synthetic_split("validation", seed=1)

    with pytest.raises(ModelError, match="must be fitted on 'train'"):
        RidgeDecoder.fit(validation, alpha=10.0)


def test_a_whole_trial_training_subset_may_still_be_fitted() -> None:
    training = _synthetic_split("train", seed=0)
    subset = subset_by_trials(training, [0, 1], label="fold 0 training")

    baseline = ConstantBaseline.fit(subset)
    decoder = RidgeDecoder.fit(subset, alpha=10.0)

    assert baseline.fitted_on == "train[fold 0 training]"
    assert decoder.fitted_on == "train[fold 0 training]"
    assert decoder.normalization.n_samples == subset.n_samples


def test_the_ridge_decoder_standardizes_evaluation_data_with_training_statistics() -> None:
    training = _synthetic_split("train", seed=0)
    # The same samples with every count raised by ten: a shift that the set's own
    # statistics would remove and the training statistics must preserve.
    shifted = _feature_set("validation", training.counts + 10, training.targets)

    decoder = RidgeDecoder.fit(training, alpha=10.0)
    on_training = decoder.predict(training)
    on_shifted = decoder.predict(shifted)

    # The stored statistics turn the raw offset into a fixed offset of ten divided
    # by each feature's training deviation, carried through the coefficients.
    standardized_shift = (10.0 / decoder.normalization.feature_std).reshape(-1)
    expected = on_training + (
        standardized_shift @ decoder.coefficients.T
    ) * decoder.normalization.target_std

    assert np.allclose(on_shifted, expected)
    # Refitting the statistics on the shifted set would cancel the offset and
    # return the training predictions, so the two must differ.
    assert not np.allclose(on_shifted, on_training)


def test_the_ridge_decoder_carries_the_statistics_it_was_fitted_with() -> None:
    training = _synthetic_split("train", seed=0)

    decoder = RidgeDecoder.fit(training, alpha=10.0)

    assert decoder.normalization.fitted_on == "train"
    assert decoder.normalization.n_samples == training.n_samples
    assert np.allclose(
        decoder.normalization.feature_mean, training.counts.astype(np.float64).mean(axis=0)
    )
    assert decoder.n_features == DEFAULT_SPEC.n_bins * N_UNITS
    assert decoder.coefficients.shape == (2, DEFAULT_SPEC.n_bins * N_UNITS)


def test_a_decoder_reloaded_from_its_artifact_predicts_identically() -> None:
    training = _synthetic_split("train", seed=0)
    validation = _synthetic_split("validation", seed=1)
    decoder = RidgeDecoder.fit(training, alpha=10.0)

    reloaded = RidgeDecoder.from_dict(json.loads(json.dumps(decoder.as_dict())))
    baseline = ConstantBaseline.fit(training)
    reloaded_baseline = ConstantBaseline.from_dict(
        json.loads(json.dumps(baseline.as_dict()))
    )

    # Bit-for-bit, not merely close: the artifact has to be the model, not a copy
    # that drifts in the last decimal place.
    assert np.array_equal(reloaded.predict(validation), decoder.predict(validation))
    assert np.array_equal(
        reloaded_baseline.predict(validation), baseline.predict(validation)
    )


def test_predicting_features_built_at_another_lag_is_refused() -> None:
    training = _synthetic_split("train", seed=0)
    decoder = RidgeDecoder.fit(training, alpha=10.0)
    other_lag = replace(
        _feature_set("validation", training.counts, training.targets),
        lag_ms=LAG_MS + 50,
    )

    with pytest.raises(ModelError, match="cannot predict"):
        decoder.predict(other_lag)
