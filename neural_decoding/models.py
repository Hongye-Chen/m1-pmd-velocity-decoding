"""Linear decoders: the constant baseline and the Ridge model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from neural_decoding.features import FeatureSet, Normalization, fit_normalization
from neural_decoding.splits import TRAIN

__all__ = [
    "ConstantBaseline",
    "ModelError",
    "RidgeDecoder",
]

# Named explicitly so the fit does not change method with the solver heuristic.
_RIDGE_SOLVER = "cholesky"

_PARAMETER_FIDELITY_TOLERANCE = 1e-9


class ModelError(RuntimeError):
    """Raised when fitting or applying a model would violate the protocol."""


def _require_training(feature_set: FeatureSet, what: str) -> None:
    """Refuse to fit on anything but training data or a training subset."""
    if feature_set.split != TRAIN:
        raise ModelError(
            f"{what} must be fitted on {TRAIN!r} data, not {feature_set.source!r}."
        )
    if feature_set.n_samples == 0:
        raise ModelError(f"{what} cannot be fitted on an empty feature set.")


@dataclass(frozen=True)
class ConstantBaseline:
    """Predicts one velocity, the training mean, for every sample.

    The project's floor: it uses no neural data, so it scores about zero on an
    evaluation set drawn from the same distribution.
    """

    velocity: np.ndarray
    fitted_on: str
    n_samples: int
    trial_rows: tuple[int, ...]
    lag_ms: int

    @classmethod
    def fit(cls, feature_set: FeatureSet) -> ConstantBaseline:
        """Return the baseline defined by the mean of the training targets."""
        _require_training(feature_set, "The constant baseline")
        return cls(
            velocity=feature_set.targets.astype(np.float64).mean(axis=0),
            fitted_on=feature_set.source,
            n_samples=feature_set.n_samples,
            trial_rows=tuple(int(row) for row in feature_set.trial_rows),
            lag_ms=feature_set.lag_ms,
        )

    @property
    def n_targets(self) -> int:
        return int(self.velocity.size)

    def predict(self, feature_set: FeatureSet) -> np.ndarray:
        """Return the stored velocity repeated once per sample."""
        return np.tile(self.velocity, (feature_set.n_samples, 1))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "constant_baseline",
            "description": "predicts the mean training velocity for every sample",
            "fitted_on": self.fitted_on,
            "lag_ms": self.lag_ms,
            "n_samples": self.n_samples,
            "n_trials": len(self.trial_rows),
            "trial_rows": list(self.trial_rows),
            "velocity_cm_s": self.velocity.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstantBaseline:
        return cls(
            velocity=np.asarray(payload["velocity_cm_s"], dtype=np.float64),
            fitted_on=str(payload["fitted_on"]),
            n_samples=int(payload["n_samples"]),
            trial_rows=tuple(int(row) for row in payload.get("trial_rows", ())),
            lag_ms=int(payload["lag_ms"]),
        )


@dataclass(frozen=True)
class RidgeDecoder:
    """Ridge regression from standardized spike counts to cursor velocity in cm/s.

    The decoder owns its normalization, so the training statistics travel with the
    coefficients and cannot be replaced by ones fitted elsewhere.
    """

    normalization: Normalization
    coefficients: np.ndarray
    intercept: np.ndarray
    alpha: float
    lag_ms: int
    n_bins: int
    n_units: int
    fitted_on: str
    n_samples: int
    trial_rows: tuple[int, ...]

    @classmethod
    def fit(cls, feature_set: FeatureSet, *, alpha: float) -> RidgeDecoder:
        """Fit normalization and Ridge coefficients on training data.

        Normalization is fitted here, from this feature set only, so the model
        cannot be handed statistics that already saw the evaluation set.
        """
        _require_training(feature_set, "The Ridge decoder")
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ModelError(f"alpha must be a non-negative finite number, got {alpha}.")

        normalization = fit_normalization(feature_set)
        design = normalization.apply_features(feature_set.counts).reshape(
            feature_set.n_samples, -1
        )
        estimator = Ridge(alpha=alpha, fit_intercept=True, solver=_RIDGE_SOLVER)
        estimator.fit(design, normalization.apply_targets(feature_set.targets))

        decoder = cls(
            normalization=normalization,
            coefficients=np.asarray(estimator.coef_, dtype=np.float64),
            intercept=np.atleast_1d(np.asarray(estimator.intercept_, dtype=np.float64)),
            alpha=alpha,
            lag_ms=feature_set.lag_ms,
            n_bins=feature_set.n_bins,
            n_units=feature_set.n_units,
            fitted_on=feature_set.source,
            n_samples=feature_set.n_samples,
            trial_rows=tuple(int(row) for row in feature_set.trial_rows),
        )

        # Confirm the stored arrays are the ones that were fitted, since predict
        # uses them instead of the estimator.
        reproduced = decoder.predict(feature_set)
        expected = normalization.invert_targets(estimator.predict(design))
        deviation = float(np.max(np.abs(reproduced - expected)))
        if deviation > _PARAMETER_FIDELITY_TOLERANCE:
            raise ModelError(
                "The stored coefficients do not reproduce the fitted estimator: "
                f"predictions differ by up to {deviation:g} cm/s."
            )
        return decoder

    @property
    def n_features(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.coefficients.shape[0])

    def predict(self, feature_set: FeatureSet) -> np.ndarray:
        """Return cm/s predictions, standardizing with the stored training statistics."""
        if feature_set.lag_ms != self.lag_ms:
            raise ModelError(
                f"This decoder was fitted at lag {self.lag_ms} ms and cannot predict "
                f"features built at lag {feature_set.lag_ms} ms."
            )
        if (feature_set.n_bins, feature_set.n_units) != (self.n_bins, self.n_units):
            raise ModelError(
                f"Expected {self.n_bins} bins and {self.n_units} units, got "
                f"{feature_set.n_bins} and {feature_set.n_units}."
            )

        design = self.normalization.apply_features(feature_set.counts).reshape(
            feature_set.n_samples, -1
        )
        scaled = design @ self.coefficients.T + self.intercept
        return self.normalization.invert_targets(scaled)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "ridge",
            "description": (
                "linear map from standardized spike counts in bin-major order to "
                "standardized velocity, inverted to cm/s"
            ),
            "fitted_on": self.fitted_on,
            "lag_ms": self.lag_ms,
            "alpha": self.alpha,
            "solver": _RIDGE_SOLVER,
            "n_samples": self.n_samples,
            "n_trials": len(self.trial_rows),
            "trial_rows": list(self.trial_rows),
            "n_bins": self.n_bins,
            "n_units": self.n_units,
            "n_features": self.n_features,
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
            "normalization": self.normalization.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RidgeDecoder:
        return cls(
            normalization=Normalization.from_dict(payload["normalization"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            intercept=np.asarray(payload["intercept"], dtype=np.float64),
            alpha=float(payload["alpha"]),
            lag_ms=int(payload["lag_ms"]),
            n_bins=int(payload["n_bins"]),
            n_units=int(payload["n_units"]),
            fitted_on=str(payload["fitted_on"]),
            n_samples=int(payload["n_samples"]),
            trial_rows=tuple(int(row) for row in payload.get("trial_rows", ())),
        )
