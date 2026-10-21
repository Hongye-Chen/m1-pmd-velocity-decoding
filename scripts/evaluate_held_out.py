"""Score the official decoders once on the held-out trials.

Run from the repository root with the project environment active::

    python -m scripts.evaluate_held_out --allow-held-out --output-dir PATH

Exit codes: 0 when every check passes, 1 when the evaluation or a check fails,
and 2 when the run is refused or its inputs are unavailable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neural_decoding.config import (
    NWB_PATH_ENV_VAR,
    ConfigError,
    ProjectConfig,
    load_config,
    project_root,
)
from neural_decoding.dataset import (
    EXPECTED_SHA256,
    DatasetError,
    SessionData,
    compute_sha256,
    load_session,
)
from neural_decoding.features import (
    FeatureError,
    FeatureSet,
    FeatureSpec,
    build_split_features,
)
from neural_decoding.final_decoder import (
    FROZEN_SPEC,
    ValidationIdentity,
    decoder_from_dict,
    filter_official_targets,
)
from neural_decoding.gru import (
    DeviceInfo,
    GruDecoder,
    GruError,
    configure_determinism,
    load_checkpoint,
    load_protocol,
    predict_normalized,
)
from neural_decoding.joint_decoder import (
    DecoderModel,
    JointDecoderError,
    TrialBounds,
    build_scheduled_observations,
    build_trial_bounds,
    target_identity_sha256,
)
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.metrics import summarize_predictions
from neural_decoding.models import ConstantBaseline, ModelError, RidgeDecoder
from neural_decoding.sequences import (
    InputRepresentation,
    SequenceError,
    SequenceNormalization,
    build_sequences,
)
from neural_decoding.splits import TEST, TRAIN

logger = get_logger("evaluate_held_out")

OFFICIAL_ARTIFACTS: dict[str, Path] = {
    "constant": Path("results/baselines/constant_baseline.json"),
    "ridge": Path("results/baselines/ridge_decoder.json"),
    "kalman_model": Path("results/kalman_validation/model.json"),
    "kalman_configuration": Path("results/kalman_validation/configuration.json"),
    "gru_configuration": Path(
        "results/gru_architecture_selection/final_configuration.json"
    ),
    "gru_seed0": Path("results/gru_architecture_selection/checkpoint_seed0.pt"),
    "gru_seed1": Path("results/gru_architecture_selection/checkpoint_seed1.pt"),
    "gru_seed2": Path("results/gru_architecture_selection/checkpoint_seed2.pt"),
}

GRU_SEEDS: tuple[int, ...] = (0, 1, 2)
LAG_MS = 50
HISTORY_MS = 200
N_BINS = 4
CADENCE_MS = 10
EXPECTED_UNITS = 53
EXPECTED_TRAINING_TRIALS = 123
EXPECTED_HELD_OUT_TRIALS = 29
AXIS_NAMES = ("x", "y")

PREDICTIONS_FILE = "held_out_predictions.csv"
SUMMARY_FILE = "summary.json"

IDENTITY_COLUMNS: tuple[str, ...] = (
    "sample_index",
    "trial_row",
    "trial_id",
    "reference_index",
    "target_index",
    "target_time",
)
TRUTH_COLUMNS: tuple[str, ...] = ("target_velocity_x", "target_velocity_y")

_METRIC_FIELDS: tuple[tuple[str, str | None], ...] = (
    ("r2", "x"),
    ("r2", "y"),
    ("r2_mean", None),
    ("pearson_r", "x"),
    ("pearson_r", "y"),
    ("pearson_r_mean", None),
    ("rmse_cm_s", "x"),
    ("rmse_cm_s", "y"),
    ("rmse_cm_s_mean", None),
)
_TABLE_TOLERANCE = 1e-12


class EvaluationError(RuntimeError):
    """Raised when the held-out evaluation would depart from the frozen protocol."""


@dataclass(frozen=True)
class OfficialModels:
    """Every frozen official decoder, loaded from its tracked artifact."""

    constant: ConstantBaseline
    ridge: RidgeDecoder
    kalman: DecoderModel
    gru: Mapping[int, GruDecoder]
    gru_normalization: SequenceNormalization
    gru_configuration: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedEvaluation:
    """Everything verified before any held-out trial may be built."""

    session: SessionData
    models: OfficialModels
    spec: FeatureSpec
    device: DeviceInfo


@dataclass(frozen=True)
class ModelPrediction:
    """One model's predictions with the target rows it was asked to predict."""

    name: str
    trial_row: np.ndarray
    target_index: np.ndarray
    target_time: np.ndarray
    velocity: np.ndarray
    extra: Mapping[str, np.ndarray] = field(default_factory=dict)


def prepare_evaluation(root: Path, config: ProjectConfig) -> PreparedEvaluation:
    """Verify the recording and load every official model without building held-out data."""
    spec = FeatureSpec(
        history_ms=config.features.history_ms,
        n_bins=config.features.n_bins,
        cadence_ms=config.features.cadence_ms,
    )
    if spec != FeatureSpec(history_ms=HISTORY_MS, n_bins=N_BINS, cadence_ms=CADENCE_MS):
        raise EvaluationError(
            f"The configured neural window {spec.as_dict()} is not the official "
            f"{HISTORY_MS} ms in {N_BINS} bins at {CADENCE_MS} ms cadence."
        )
    session = load_session(config)
    models = load_official_models(root, session)
    protocol = load_protocol(
        root / models.gru_configuration["protocols"]["input_selection"]["path"]
    )
    device = configure_determinism(protocol)
    return PreparedEvaluation(session=session, models=models, spec=spec, device=device)


def load_official_models(root: Path, session: SessionData) -> OfficialModels:
    """Load the tracked official models and confirm they share one frozen protocol."""
    paths = {name: root / path for name, path in OFFICIAL_ARTIFACTS.items()}
    missing = [
        OFFICIAL_ARTIFACTS[name].as_posix() for name, path in paths.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Official model artifacts are missing: {missing}")

    unit_ids = [int(unit.unit_id) for unit in session.units]
    _require(
        len(unit_ids) == EXPECTED_UNITS,
        f"The recording has {len(unit_ids)} units, not {EXPECTED_UNITS}.",
    )
    training_rows = _eligible_rows(session, TRAIN)
    _require(
        len(training_rows) == EXPECTED_TRAINING_TRIALS,
        f"The recording has {len(training_rows)} eligible training trials, not "
        f"{EXPECTED_TRAINING_TRIALS}.",
    )

    constant = ConstantBaseline.from_dict(_read_json(paths["constant"]))
    _require(
        constant.fitted_on == TRAIN
        and constant.lag_ms == LAG_MS
        and constant.trial_rows == training_rows
        and constant.velocity.shape == (2,)
        and bool(np.all(np.isfinite(constant.velocity))),
        "The constant baseline is not the official fit on the eligible training trials.",
    )

    ridge = RidgeDecoder.from_dict(_read_json(paths["ridge"]))
    normalization = ridge.normalization
    _require(
        ridge.fitted_on == TRAIN
        and ridge.lag_ms == LAG_MS
        and normalization.lag_ms == LAG_MS
        and (ridge.n_bins, ridge.n_units) == (N_BINS, len(unit_ids))
        and ridge.coefficients.shape == (2, N_BINS * len(unit_ids))
        and ridge.intercept.shape == (2,)
        and normalization.feature_mean.shape == (N_BINS, len(unit_ids))
        and normalization.feature_std.shape == (N_BINS, len(unit_ids))
        and ridge.trial_rows == training_rows
        and normalization.trial_rows == training_rows
        and all(
            np.all(np.isfinite(array))
            for array in (
                ridge.coefficients,
                ridge.intercept,
                normalization.feature_mean,
                normalization.feature_std,
                normalization.target_mean,
                normalization.target_std,
            )
        ),
        "The Ridge decoder is not the official fit on the eligible training trials.",
    )

    kalman_payload = _read_json(paths["kalman_model"])
    kalman, kalman_units = decoder_from_dict(kalman_payload)
    kalman_configuration = _read_json(paths["kalman_configuration"])["configuration"]
    _require(
        (
            str(kalman_configuration["state_name"]),
            int(kalman_configuration["neural_width_ms"]),
            int(kalman_configuration["interval_end_lag_ms"]),
            int(kalman_configuration["state_and_output_cadence_ms"]),
        )
        == (
            FROZEN_SPEC.state_name,
            FROZEN_SPEC.neural_width_ms,
            FROZEN_SPEC.interval_end_lag_ms,
            CADENCE_MS,
        )
        and FROZEN_SPEC.interval_end_lag_ms == LAG_MS,
        "The Kalman configuration is not the frozen PVA / 50 ms / 50 ms decoder.",
    )
    _require(
        list(kalman_units) == unit_ids,
        "The Kalman decoder's unit order differs from the recording.",
    )
    _require(
        kalman.process.training_rows == training_rows
        and kalman.observation.training_rows == training_rows,
        "The Kalman decoder was not fitted on exactly the eligible training trials.",
    )
    _require(
        kalman_payload["provenance"]["recording_sha256"] == session.identity.sha256,
        "The Kalman decoder was fitted on a different recording.",
    )

    gru_configuration = _read_json(paths["gru_configuration"])
    timing = gru_configuration["input_representation"]
    architecture = gru_configuration["architecture"]
    representation = InputRepresentation(int(timing["history_ms"]), int(timing["bin_ms"]))
    _require(
        representation == InputRepresentation(HISTORY_MS, HISTORY_MS // N_BINS)
        and int(timing["n_bins"]) == N_BINS
        and int(timing["interval_end_lag_ms"]) == LAG_MS
        and int(timing["cadence_ms"]) == CADENCE_MS,
        "The GRU input is not 200 ms of history in 50 ms bins at the 50 ms lag.",
    )
    _require(
        tuple(int(seed) for seed in gru_configuration["seeds"]) == GRU_SEEDS
        and all(
            gru_configuration["checkpoints"][str(seed)]["file"]
            == OFFICIAL_ARTIFACTS[f"gru_seed{seed}"].name
            for seed in GRU_SEEDS
        ),
        "The GRU configuration does not name the three official seed checkpoints.",
    )
    _require(
        [int(unit) for unit in gru_configuration["unit_ids"]] == unit_ids,
        "The GRU unit order differs from the recording.",
    )
    _require(
        gru_configuration["recording"]["sha256"] == session.identity.sha256,
        "The GRU was trained on a different recording.",
    )
    gru_normalization = SequenceNormalization.from_dict(
        gru_configuration["normalization"], trial_rows=training_rows
    )
    _require(
        gru_normalization.representation == representation
        and gru_normalization.count_mean.shape == (len(unit_ids),),
        "The stored GRU normalization does not match its input representation.",
    )

    expected_metadata = {
        "history_ms": representation.history_ms,
        "bin_ms": representation.bin_ms,
        "interval_end_lag_ms": LAG_MS,
        "num_layers": int(architecture["num_layers"]),
        "hidden_size": int(architecture["hidden_size"]),
        "dropout": float(architecture["dropout"]),
        "learning_rate": float(gru_configuration["optimization"]["learning_rate"]),
        "n_units": len(unit_ids),
    }
    gru: dict[int, GruDecoder] = {}
    for seed in GRU_SEEDS:
        model, metadata = load_checkpoint(paths[f"gru_seed{seed}"])
        expected = {**expected_metadata, "seed": seed}
        differing = sorted(key for key, value in expected.items() if metadata.get(key) != value)
        _require(
            not differing and model.num_layers == expected["num_layers"],
            f"The seed {seed} GRU checkpoint has unexpected metadata: {differing}",
        )
        gru[seed] = model

    return OfficialModels(
        constant=constant,
        ridge=ridge,
        kalman=kalman,
        gru=gru,
        gru_normalization=gru_normalization,
        gru_configuration=gru_configuration,
    )


def build_held_out_targets(
    session: SessionData, spec: FeatureSpec
) -> tuple[FeatureSet, dict[int, TrialBounds]]:
    """Build the common held-out targets and confirm they follow the official rule."""
    rows = _eligible_rows(session, TEST)
    _require(
        len(rows) == EXPECTED_HELD_OUT_TRIALS,
        f"The recording has {len(rows)} eligible held-out trials, not "
        f"{EXPECTED_HELD_OUT_TRIALS}.",
    )
    features = build_split_features(
        session,
        TEST,
        lag_ms=LAG_MS,
        eligibility_lag_ms=LAG_MS,
        spec=spec,
        allow_held_out=True,
    )
    _require(
        tuple(int(row) for row in features.trial_rows) == rows,
        "Not every eligible held-out trial contributed targets.",
    )
    bounds = build_trial_bounds(
        session.behavior_timestamps, session.trials, rows, required_split=TEST
    )
    expected_index, expected_rows = expected_target_rows(bounds)
    _require(
        np.array_equal(features.target_index, expected_index)
        and np.array_equal(features.trial_row, expected_rows)
        and np.array_equal(
            features.reference_index, expected_index - LAG_MS // CADENCE_MS
        ),
        "The held-out targets do not follow the official target rule.",
    )
    return features, bounds


def expected_target_rows(
    bounds: Mapping[int, TrialBounds],
) -> tuple[np.ndarray, np.ndarray]:
    """List every target whose history before the lag fits inside its own trial."""
    first_offset = (HISTORY_MS + LAG_MS) // CADENCE_MS
    indices: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for row in sorted(bounds):
        trial = bounds[row]
        span = np.arange(
            trial.first_index + first_offset, trial.last_index + 1, dtype=np.int64
        )
        indices.append(span)
        rows.append(np.full(span.size, row, dtype=np.int64))
    return np.concatenate(indices), np.concatenate(rows)


def identity_frame(session: SessionData, features: FeatureSet) -> pd.DataFrame:
    """Return target identifiers and true velocity in canonical order."""
    trial_ids = session.trials.set_index("trial_row")["trial_id"]
    return pd.DataFrame(
        {
            "sample_index": np.arange(features.n_samples, dtype=np.int64),
            "trial_row": features.trial_row,
            "trial_id": trial_ids.loc[features.trial_row].to_numpy(dtype=np.int64),
            "reference_index": features.reference_index,
            "target_index": features.target_index,
            "target_time": features.target_time,
            "target_velocity_x": features.targets[:, 0],
            "target_velocity_y": features.targets[:, 1],
        }
    )


def predict_kalman(
    session: SessionData,
    model: DecoderModel,
    features: FeatureSet,
    bounds: Mapping[int, TrialBounds],
) -> tuple[ModelPrediction, np.ndarray]:
    """Filter each held-out trial from its start, seeded only by its starting position."""
    schedule = build_scheduled_observations(
        session.behavior_timestamps,
        session.spike_times,
        bounds,
        width_ms=FROZEN_SPEC.neural_width_ms,
        lag_ms=FROZEN_SPEC.interval_end_lag_ms,
    )
    # The official filter reads only target identities from this container, never truth.
    targets = ValidationIdentity(
        sample_index=np.arange(features.n_samples, dtype=np.int64),
        trial_row=features.trial_row.copy(),
        reference_index=features.reference_index.copy(),
        target_index=features.target_index.copy(),
        target_time=features.target_time.copy(),
        identity_sha256=target_identity_sha256(
            features.trial_row, features.target_index, features.target_time
        ),
        source_sha256=session.identity.sha256,
    )
    initial_positions = {
        row: session.cursor_position[trial.first_index].copy()
        for row, trial in bounds.items()
    }
    result = filter_official_targets(
        model, schedule, bounds, targets, initial_positions=initial_positions
    )
    covariance = result.velocity_covariance
    prediction = ModelPrediction(
        name="kalman",
        trial_row=targets.trial_row,
        target_index=targets.target_index,
        target_time=targets.target_time,
        velocity=result.prediction,
        extra={
            "posterior_variance_x": covariance[:, 0, 0],
            "posterior_covariance_xy": covariance[:, 0, 1],
            "posterior_variance_y": covariance[:, 1, 1],
        },
    )
    return prediction, result.fill_count


def predict_gru(
    session: SessionData, models: OfficialModels, features: FeatureSet, device: str
) -> list[ModelPrediction]:
    """Predict with each GRU seed separately on the common targets."""
    normalization = models.gru_normalization
    sequences = build_sequences(
        features,
        timestamps=session.behavior_timestamps,
        spike_times=session.spike_times,
        trials=session.trials,
        representation=normalization.representation,
        allow_held_out=True,
    )
    _require(
        np.array_equal(sequences.counts, features.counts),
        "The GRU input differs from the official spike-count window.",
    )
    target = torch.device(device)
    inputs = torch.from_numpy(normalization.apply_counts(sequences.counts)).to(target)
    batch_size = int(models.gru_configuration["optimization"]["batch_size"])
    predictions: list[ModelPrediction] = []
    for seed in GRU_SEEDS:
        normalized = predict_normalized(
            models.gru[seed].to(target), inputs, batch_size=batch_size
        )
        predictions.append(
            ModelPrediction(
                name=f"gru_seed{seed}",
                trial_row=sequences.trial_row,
                target_index=sequences.target_index,
                target_time=sequences.target_time,
                velocity=normalization.invert_targets(normalized),
            )
        )
    return predictions


def build_prediction_table(
    identity: pd.DataFrame, predictions: Sequence[ModelPrediction]
) -> pd.DataFrame:
    """Return identifiers, true velocity, and every prediction, refusing misaligned models."""
    missing = [
        name for name in (*IDENTITY_COLUMNS, *TRUTH_COLUMNS) if name not in identity.columns
    ]
    if missing:
        raise EvaluationError(f"The target identity is missing columns {missing}.")
    names = [prediction.name for prediction in predictions]
    if not names or len(set(names)) != len(names):
        raise EvaluationError(f"Every model needs one unique name, got {names}.")

    trial_row = identity["trial_row"].to_numpy(dtype=np.int64)
    target_index = identity["target_index"].to_numpy(dtype=np.int64)
    target_time = identity["target_time"].to_numpy(dtype=np.float64)
    columns: dict[str, np.ndarray] = {
        name: identity[name].to_numpy() for name in (*IDENTITY_COLUMNS, *TRUTH_COLUMNS)
    }
    for prediction in predictions:
        _require_aligned(prediction, trial_row, target_index, target_time)
        for axis, axis_name in enumerate(AXIS_NAMES):
            columns[f"{prediction.name}_velocity_{axis_name}"] = np.asarray(
                prediction.velocity[:, axis], dtype=np.float64
            )
        for key, values in prediction.extra.items():
            columns[f"{prediction.name}_{key}"] = np.asarray(values, dtype=np.float64)
    return pd.DataFrame(columns)


def score_table(table: pd.DataFrame, names: Sequence[str]) -> dict[str, dict[str, object]]:
    """Score each named model's columns against the true velocity in the table."""
    truth = table[list(TRUTH_COLUMNS)].to_numpy(dtype=np.float64)
    return {
        name: summarize_predictions(
            truth,
            table[[f"{name}_velocity_{axis}" for axis in AXIS_NAMES]].to_numpy(
                dtype=np.float64
            ),
            axis_names=AXIS_NAMES,
        )
        for name in names
    }


def aggregate_seed_metrics(
    seed_metrics: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the mean and sample standard deviation (ddof=1) of each seed-level metric.

    Every seed is scored on its own predictions first; predictions are never
    averaged across seeds.
    """
    seeds = sorted(seed_metrics)
    if len(seeds) < 2:
        raise EvaluationError("Seed variability needs at least two seeds.")
    mean: dict[str, Any] = {}
    spread: dict[str, Any] = {}
    for key, axis in _METRIC_FIELDS:
        values = [
            seed_metrics[seed][key] if axis is None else seed_metrics[seed][key][axis]
            for seed in seeds
        ]
        if any(value is None for value in values):
            center = deviation = None
        else:
            array = np.asarray(values, dtype=np.float64)
            center, deviation = float(array.mean()), float(array.std(ddof=1))
        if axis is None:
            mean[key], spread[key] = center, deviation
        else:
            mean.setdefault(key, {})[axis] = center
            spread.setdefault(key, {})[axis] = deviation
    return {"seed_mean": mean, "seed_std": spread}


def run_evaluation(
    prepared: PreparedEvaluation, predictions_path: Path, summary_path: Path
) -> dict[str, Any]:
    """Build the held-out targets, predict once with every model, save, then score."""
    session, models = prepared.session, prepared.models
    features, bounds = build_held_out_targets(session, prepared.spec)
    logger.info(
        "Predicting %d held-out targets from %d trials", features.n_samples, len(bounds)
    )

    kalman, kalman_fill = predict_kalman(session, models.kalman, features, bounds)
    predictions = [
        _from_features("constant", features, models.constant.predict(features)),
        _from_features("ridge", features, models.ridge.predict(features)),
        kalman,
        *predict_gru(session, models, features, prepared.device.device),
    ]
    identity = identity_frame(session, features)
    table = build_prediction_table(identity, predictions)
    table.to_csv(predictions_path, index=False, lineterminator="\n")

    stored = pd.read_csv(predictions_path, float_precision="round_trip")
    names = [prediction.name for prediction in predictions]
    metrics = score_table(stored, names)
    gru_metrics = {
        "seeds": {str(seed): metrics[f"gru_seed{seed}"] for seed in GRU_SEEDS},
        **aggregate_seed_metrics({seed: metrics[f"gru_seed{seed}"] for seed in GRU_SEEDS}),
    }

    held_out_rows = tuple(sorted(bounds))
    trial_ids = session.trials.set_index("trial_row")["trial_id"]
    prediction_columns = [
        column
        for column in stored.columns
        if column not in IDENTITY_COLUMNS and column not in TRUTH_COLUMNS
    ]
    stored_matches = (
        list(stored.columns) == list(table.columns)
        and len(stored) == len(table)
        and np.allclose(
            stored.to_numpy(dtype=np.float64),
            table.to_numpy(dtype=np.float64),
            atol=_TABLE_TOLERANCE,
            rtol=0.0,
        )
    )
    checks = [
        _check(
            "the recording matches the frozen identity",
            session.identity.sha256 == EXPECTED_SHA256,
            f"SHA-256 {session.identity.sha256[:12]}... verified before any held-out "
            "trial was built",
        ),
        _check(
            "exactly the eligible held-out trials were evaluated",
            held_out_rows == _eligible_rows(session, TEST)
            and len(held_out_rows) == EXPECTED_HELD_OUT_TRIALS,
            f"{len(held_out_rows)} held-out trials",
        ),
        _check(
            "the targets follow the official target rule",
            np.array_equal(
                stored["target_index"].to_numpy(dtype=np.int64),
                expected_target_rows(bounds)[0],
            ),
            f"{features.n_samples} targets with {HISTORY_MS} ms of history before the "
            f"{LAG_MS} ms lag inside each trial",
        ),
        _check(
            "every model has one finite prediction for every target",
            len(names) == 3 + len(GRU_SEEDS)
            and bool(np.all(np.isfinite(stored[prediction_columns].to_numpy(np.float64)))),
            f"{len(names)} prediction sets over identical ordered target rows",
        ),
        _check(
            "the Kalman filter filled every target exactly once",
            bool(np.all(kalman_fill == 1)),
            "each target read from its own trial's recursive pass",
        ),
        _check(
            "the saved table reproduces the predictions that were scored",
            stored_matches,
            f"{len(stored)} rows round-trip within {_TABLE_TOLERANCE:g}",
        ),
    ]

    summary: dict[str, Any] = {
        "procedure": {
            "name": "one-time held-out evaluation of the frozen official decoders",
            "models": "loaded from tracked artifacts; nothing was refitted, tuned, or "
            "selected",
            "targets": f"every eligible held-out trial; each target needs {HISTORY_MS} ms "
            f"of neural history before the {LAG_MS} ms lag inside its trial on the "
            f"{CADENCE_MS} ms behavior grid; every model predicts the same rows in the "
            "same order",
            "metrics": "per-axis and mean R2, Pearson correlation, and RMSE in cm/s, "
            "computed from the saved prediction table",
            "gru_seeds": "each seed scored separately; arithmetic mean and sample "
            "standard deviation (ddof=1) of the seed-level metrics; no ensemble and no "
            "seed selection",
        },
        "recording": {"name": session.identity.name, "sha256": session.identity.sha256},
        "models": _model_summary(models, prepared.device),
        "samples": {
            "held_out_trials": len(held_out_rows),
            "held_out_trial_rows": list(held_out_rows),
            "held_out_trial_ids": [int(trial_ids.loc[row]) for row in held_out_rows],
            "held_out_targets": features.n_samples,
            "target_identity_sha256": target_identity_sha256(
                features.trial_row, features.target_index, features.target_time
            ),
        },
        "metrics": {
            "constant": metrics["constant"],
            "ridge": metrics["ridge"],
            "kalman": metrics["kalman"],
            "gru": gru_metrics,
        },
        "checks": checks,
        "all_checks_passed": all(bool(check["passed"]) for check in checks),
        "artifacts": {
            "predictions": _repo_relative(predictions_path),
            "summary": _repo_relative(summary_path),
        },
        "artifact_hashes": {"predictions_sha256": compute_sha256(predictions_path)},
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-held-out",
        action="store_true",
        help="Authorize the one-time evaluation on the held-out trials.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="New directory for the evaluation outputs; an existing one is refused.",
    )
    args = parser.parse_args(argv)

    if not args.allow_held_out:
        print(
            "Refusing to run: the held-out trials stay unopened unless "
            "--allow-held-out is given."
        )
        return 2
    output_directory = args.output_dir.resolve()
    if output_directory.exists():
        print(
            f"Refusing to run: {output_directory} already exists. Held-out outputs are "
            "never overwritten or reused."
        )
        return 2

    try:
        config = load_config()
        configure_logging(config.run.log_level)
        prepared = prepare_evaluation(project_root(), config)
    except (ConfigError, DatasetError, FileNotFoundError) as error:
        print(f"Cannot prepare the held-out evaluation: {error}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, to the selected recording described in README.md."
        )
        return 2
    except (
        EvaluationError,
        FeatureError,
        GruError,
        JointDecoderError,
        ModelError,
        SequenceError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"The official models did not load as frozen: {error}")
        return 1

    output_directory.mkdir(parents=True, exist_ok=False)
    predictions_path = output_directory / PREDICTIONS_FILE
    summary_path = output_directory / SUMMARY_FILE
    try:
        summary = run_evaluation(prepared, predictions_path, summary_path)
    except Exception as error:
        failure = {
            "procedure": "one-time held-out evaluation of the frozen official decoders",
            "completed": False,
            "failure": f"{type(error).__name__}: {error}",
            "retry": "not permitted without review and renewed authorization",
        }
        summary_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(f"HELD-OUT EVALUATION FAILED: {error}")
        print(f"{output_directory} is preserved for review and must not be reused.")
        return 1

    _print_report(summary)
    return 0 if summary["all_checks_passed"] else 1


def _from_features(name: str, features: FeatureSet, velocity: np.ndarray) -> ModelPrediction:
    return ModelPrediction(
        name=name,
        trial_row=features.trial_row,
        target_index=features.target_index,
        target_time=features.target_time,
        velocity=np.asarray(velocity, dtype=np.float64),
    )


def _require_aligned(
    prediction: ModelPrediction,
    trial_row: np.ndarray,
    target_index: np.ndarray,
    target_time: np.ndarray,
) -> None:
    """Refuse a model whose rows, order, or prediction shape differ from the targets."""
    if not (
        np.array_equal(np.asarray(prediction.trial_row), trial_row)
        and np.array_equal(np.asarray(prediction.target_index), target_index)
        and np.array_equal(np.asarray(prediction.target_time), target_time)
    ):
        raise EvaluationError(
            f"{prediction.name} does not predict the common target rows in their "
            "canonical order."
        )
    velocity = np.asarray(prediction.velocity, dtype=np.float64)
    if velocity.shape != (trial_row.size, 2) or not np.all(np.isfinite(velocity)):
        raise EvaluationError(
            f"{prediction.name} did not produce one finite two-axis prediction per target."
        )
    for key, values in prediction.extra.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (trial_row.size,) or not np.all(np.isfinite(array)):
            raise EvaluationError(
                f"{prediction.name} {key} is not one finite value per target."
            )


def _model_summary(models: OfficialModels, device: DeviceInfo) -> dict[str, Any]:
    architecture = models.gru_configuration["architecture"]
    representation = models.gru_normalization.representation
    return {
        "constant": {"artifact": OFFICIAL_ARTIFACTS["constant"].as_posix()},
        "ridge": {
            "artifact": OFFICIAL_ARTIFACTS["ridge"].as_posix(),
            "alpha": models.ridge.alpha,
            "lag_ms": models.ridge.lag_ms,
        },
        "kalman": {
            "model": OFFICIAL_ARTIFACTS["kalman_model"].as_posix(),
            "configuration": OFFICIAL_ARTIFACTS["kalman_configuration"].as_posix(),
            "state": FROZEN_SPEC.state_name,
            "neural_width_ms": FROZEN_SPEC.neural_width_ms,
            "interval_end_lag_ms": FROZEN_SPEC.interval_end_lag_ms,
            "initialization": "recorded position at each trial start only",
        },
        "gru": {
            "configuration": OFFICIAL_ARTIFACTS["gru_configuration"].as_posix(),
            "checkpoints": {
                str(seed): OFFICIAL_ARTIFACTS[f"gru_seed{seed}"].as_posix()
                for seed in GRU_SEEDS
            },
            "num_layers": int(architecture["num_layers"]),
            "hidden_size": int(architecture["hidden_size"]),
            "dropout": float(architecture["dropout"]),
            "history_ms": representation.history_ms,
            "bin_ms": representation.bin_ms,
            "interval_end_lag_ms": LAG_MS,
            "device": device.as_dict(),
        },
    }


def _eligible_rows(session: SessionData, split: str) -> tuple[int, ...]:
    trials = session.trials
    selected = trials.loc[
        (trials["split"] == split) & trials["eligible"].astype(bool), "trial_row"
    ]
    return tuple(sorted(int(row) for row in selected))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _print_report(summary: Mapping[str, Any]) -> None:
    metrics = summary["metrics"]
    gru = metrics["gru"]
    rows = [
        ("constant", metrics["constant"]),
        ("ridge", metrics["ridge"]),
        ("kalman", metrics["kalman"]),
        *((f"gru seed {seed}", entry) for seed, entry in gru["seeds"].items()),
        ("gru seed mean", gru["seed_mean"]),
    ]
    print("Held-out results")
    for label, entry in rows:
        print(
            f"  {label:<14} mean R2 {_number(entry['r2_mean']):>9}"
            f"  (x {_number(entry['r2']['x'])}, y {_number(entry['r2']['y'])})"
            f"  mean r {_number(entry['pearson_r_mean'])}"
            f"  mean RMSE {_number(entry['rmse_cm_s_mean'])} cm/s"
        )
    print(f"  gru seed std   mean R2 {_number(gru['seed_std']['r2_mean']):>9}")

    checks = summary["checks"]
    width = max(len(str(check["name"])) for check in checks)
    print("\nChecks")
    for check in checks:
        status = "pass" if check["passed"] else "FAIL"
        print(f"  {str(check['name']):<{width}}  {status:<4}  {check['detail']}")

    print("\nArtifacts")
    for path in summary["artifacts"].values():
        print(f"  {path}")


def _number(value: float | None) -> str:
    """Render a metric, or ``undefined`` where it has no value."""
    return "undefined" if value is None else f"{value:.6f}"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
