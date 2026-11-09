"""Choose the GRU architecture and learning rate on the validation set.

Usage, from the repository root::

    python -m scripts.select_gru_architecture

Exit codes: 0 if all checks pass, 1 if a check or run fails, 2 if the recording
or an input file is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neural_decoding.config import NWB_PATH_ENV_VAR, ConfigError, load_config, project_root
from neural_decoding.dataset import DatasetError, SessionData, compute_sha256, load_session
from neural_decoding.features import FeatureError, FeatureSet, FeatureSpec, build_split_features
from neural_decoding.gru import (
    DeviceInfo,
    GruError,
    GruProtocol,
    configure_determinism,
    load_checkpoint,
    load_protocol,
    predict_normalized,
    save_checkpoint,
    train_run,
)
from neural_decoding.gru_architecture import (
    ArchitectureScore,
    ArchitectureSearch,
    GruArchitecture,
    RunPlan,
    baseline_architecture,
    load_architecture_search,
    plan_runs,
    rank_architectures,
    text_matches_sha256,
    verify_artifact_hashes,
)
from neural_decoding.joint_decoder import target_identity_sha256
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.metrics import MetricError, r2_per_axis, summarize_predictions
from neural_decoding.sequences import (
    InputRepresentation,
    SequenceError,
    SequenceNormalization,
    build_sequences,
    fit_sequence_normalization,
)
from neural_decoding.splits import TRAIN, VALIDATION

logger = get_logger("select_gru_architecture")

AXIS_NAMES = ("x", "y")

_SEARCH = Path("configs/gru_architecture_selection.yaml")
_BASELINE_SUMMARY = Path("results/baseline_summary.json")
_OUTPUT_DIRECTORY = Path("results/internal/gru_architecture_selection")
_CSV_TOLERANCE = 1e-12
_PROBE_SAMPLES = 2048
_PROGRESS_EVERY = 10
_METRIC_COLUMNS = {
    "r2_x": ("r2", "x"),
    "r2_y": ("r2", "y"),
    "r2_mean": ("r2_mean", None),
    "pearson_r_x": ("pearson_r", "x"),
    "pearson_r_y": ("pearson_r", "y"),
    "pearson_r_mean": ("pearson_r_mean", None),
    "rmse_cm_s_x": ("rmse_cm_s", "x"),
    "rmse_cm_s_y": ("rmse_cm_s", "y"),
    "rmse_cm_s_mean": ("rmse_cm_s_mean", None),
}


@dataclass(frozen=True)
class FrozenInput:
    """Input-selection results used by the architecture search."""

    protocol: GruProtocol
    protocol_path: Path
    protocol_sha256: str
    protocol_bytes_identical: bool
    directory: Path
    relative_directory: Path
    summary: dict[str, Any]
    selected: dict[str, Any]
    runs: pd.DataFrame
    histories: pd.DataFrame
    predictions: pd.DataFrame
    representation: InputRepresentation
    baseline: GruArchitecture


@dataclass(frozen=True)
class PreparedData:
    """Normalized training and validation data shared by every run."""

    training: FeatureSet
    validation: FeatureSet
    normalization: SequenceNormalization
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: np.ndarray


@dataclass(frozen=True)
class CandidateRun:
    """One configuration/seed entry, trained here or reused from input selection."""

    architecture: GruArchitecture
    seed: int
    source: str
    epochs_run: int
    best_epoch: int
    stopped_early: bool
    best_validation_mse: float
    history: list[dict[str, Any]]
    normalized_prediction: np.ndarray
    metrics: dict[str, Any]
    checks: dict[str, bool]
    state: dict[str, torch.Tensor] | None = None
    reference: dict[str, str] | None = None

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    config = load_config()
    configure_logging(config.run.log_level)
    root = project_root()
    output = root / _OUTPUT_DIRECTORY
    if output.exists():
        print(f"Refusing to run: {output} already exists and is never overwritten.")
        return 1

    try:
        search_path = root / _SEARCH
        search = load_architecture_search(search_path)
        search_sha256 = compute_sha256(search_path)
        frozen = _load_frozen_input(root, search)
        plan = plan_runs(search, frozen.protocol.seeds, frozen.baseline)
        device_info = configure_determinism(frozen.protocol)
        if device_info.as_dict() != frozen.selected["device"]:
            raise GruError(
                f"This device {device_info.as_dict()} differs from the input-selection "
                f"device {frozen.selected['device']}."
            )
        session = load_session(config)
        data = _prepare_data(root, session, frozen, device_info)
        reused = _restore_reused(frozen, data, plan, device_info)
    except (ConfigError, DatasetError, FileNotFoundError) as error:
        print(f"Cannot select the GRU architecture: {error}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, and keep the frozen input-selection artifacts."
        )
        return 2
    except (GruError, FeatureError, SequenceError, MetricError, KeyError, ValueError) as error:
        print(f"GRU architecture selection stopped before training: {error}")
        print("Nothing was trained or written.")
        return 1

    logger.info(
        "Device %s (%s), torch %s, CUDA %s, cuDNN %s; strict determinism on",
        device_info.device,
        device_info.device_name,
        device_info.torch_version,
        device_info.cuda_version,
        device_info.cudnn_version,
    )
    logger.info(
        "Frozen input verified: %s; %d reused runs reproduce their saved predictions",
        frozen.representation.label,
        len(reused),
    )
    try:
        probes = _determinism_probes(data, frozen.protocol, search)
        fingerprint = _input_fingerprint(data)
        trained = _train_new(plan, data, frozen.protocol)
        inputs_unchanged = _input_fingerprint(data) == fingerprint
    except (GruError, MetricError, RuntimeError) as error:
        print(f"GRU architecture selection stopped: {error}")
        print("No configuration was selected; nothing was written.")
        return 1

    order = {architecture: index for index, architecture in enumerate(search.architectures)}
    candidates = sorted(reused + trained, key=lambda run: (order[run.architecture], run.seed))
    checks = _pre_training_checks(frozen, data) + _search_checks(
        candidates, frozen, search, probes, inputs_unchanged
    )

    output.mkdir(parents=True, exist_ok=False)
    candidates_path = output / "candidates.csv"
    _candidates_frame(candidates).to_csv(candidates_path, index=False, lineterminator="\n")
    artifact_hashes: dict[str, str] = {"candidates.csv": compute_sha256(candidates_path)}

    ranked: list[ArchitectureScore] = []
    selected: ArchitectureScore | None = None
    if all(check["passed"] for check in checks):
        ranked = rank_architectures(
            _scores(candidates, search, frozen.protocol), tolerance=search.tie_tolerance
        )
        axis_means = _axis_means(ranked[0].architecture, candidates)
        positive = all(np.isfinite(value) and value > 0 for value in axis_means.values())
        checks.append(
            _check(
                "the selected configuration has positive seed-mean R2 on each axis",
                positive,
                ", ".join(f"{axis} {value:.6f}" for axis, value in axis_means.items()),
            )
        )
        if positive:
            selected = ranked[0]
            retained = _write_final(
                output,
                selected.architecture,
                candidates,
                frozen,
                data,
                session,
                search_sha256,
                device_info,
            )
            checks.extend(retained.pop("checks"))
            artifact_hashes.update(retained)

    summary = _summary(
        search=search,
        search_sha256=search_sha256,
        frozen=frozen,
        device_info=device_info,
        session=session,
        data=data,
        candidates=candidates,
        ranked=ranked,
        selected=selected,
        checks=checks,
        artifact_hashes=artifact_hashes,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _print_report(ranked, selected, checks, output)
    return 0 if summary["all_checks_passed"] else 1


def _load_frozen_input(root: Path, search: ArchitectureSearch) -> FrozenInput:
    """Load and validate the input-selection results."""
    directory = root / search.input_artifacts
    protocol_path = root / search.input_protocol
    paths = [protocol_path, directory / "summary.json", directory / "selected_input.json"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen input-selection files: {missing}")

    protocol_sha256 = search.input_protocol_sha256
    if not text_matches_sha256(protocol_path, protocol_sha256):
        raise GruError(f"{search.input_protocol} differs from its frozen SHA-256.")
    if compute_sha256(directory / "summary.json") != search.input_summary_sha256:
        raise GruError("The input-selection summary differs from its frozen SHA-256.")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary["all_checks_passed"] or summary["selected"] is None:
        raise GruError("The frozen input selection did not pass every check.")
    if summary["protocol"]["sha256"] != protocol_sha256:
        raise GruError("The input-selection summary was produced by a different protocol.")
    mismatched = verify_artifact_hashes(directory, summary["artifact_hashes"])
    if mismatched:
        raise GruError(f"Input-selection artifacts differ from their recorded hashes: {mismatched}")

    selected = json.loads((directory / "selected_input.json").read_text(encoding="utf-8"))
    recorded = {
        name: digest
        for name, digest in selected["artifact_hashes"].items()
        if summary["artifact_hashes"].get(name) != digest
    }
    if recorded or selected["protocol"]["sha256"] != protocol_sha256:
        raise GruError(f"The selected-input record disagrees with the summary: {sorted(recorded)}")

    protocol = load_protocol(protocol_path)
    timing = selected["input_representation"]
    representation = InputRepresentation(int(timing["history_ms"]), int(timing["bin_ms"]))
    consistent = {
        "selected representation": (
            (representation.history_ms, representation.bin_ms),
            (summary["selected"]["history_ms"], summary["selected"]["bin_ms"]),
        ),
        "lag": (int(timing["interval_end_lag_ms"]), protocol.lag_ms),
        "seeds": (selected["seeds"], list(protocol.seeds)),
        "baseline model": (
            selected["baseline_model"],
            {"hidden_size": protocol.hidden_size, "learning_rate": protocol.learning_rate},
        ),
    }
    differing = [name for name, (found, required) in consistent.items() if found != required]
    if differing:
        raise GruError(f"The selected-input record disagrees with its protocol: {differing}")

    runs = _read_exact_csv(directory / "runs.csv")
    if len(runs) != len(protocol.representations) * len(protocol.seeds) or not bool(
        runs["checks_passed"].all()
    ):
        raise GruError("The input-selection run table is incomplete or has failing runs.")
    return FrozenInput(
        protocol=protocol,
        protocol_path=search.input_protocol,
        protocol_sha256=protocol_sha256,
        protocol_bytes_identical=compute_sha256(protocol_path) == protocol_sha256,
        directory=directory,
        relative_directory=search.input_artifacts,
        summary=summary,
        selected=selected,
        runs=runs,
        histories=_read_exact_csv(directory / "training_histories.csv"),
        predictions=_read_exact_csv(directory / "validation_predictions.csv"),
        representation=representation,
        baseline=baseline_architecture(protocol),
    )


def _prepare_data(
    root: Path, session: SessionData, frozen: FrozenInput, device_info: DeviceInfo
) -> PreparedData:
    """Rebuild and verify the selected targets, sequences, and normalization."""
    selected, protocol = frozen.selected, frozen.protocol
    if session.identity.sha256 != selected["recording"]["sha256"]:
        raise GruError("The recording differs from the one used for input selection.")
    baseline_path = root / _BASELINE_SUMMARY
    if compute_sha256(baseline_path) != frozen.summary["provenance"]["baseline_summary_sha256"]:
        raise GruError("The official baseline summary changed after input selection.")
    configuration = json.loads(baseline_path.read_text(encoding="utf-8"))["configuration"]
    spec = FeatureSpec(
        history_ms=int(configuration["history_ms"]),
        n_bins=int(configuration["n_bins"]),
        cadence_ms=int(configuration["cadence_ms"]),
    )
    training, validation = (
        build_split_features(
            session, split, lag_ms=protocol.lag_ms, eligibility_lag_ms=protocol.lag_ms, spec=spec
        )
        for split in (TRAIN, VALIDATION)
    )

    targets = selected["targets"]
    identities = {
        "training samples": (training.n_samples, targets["training_samples"]),
        "training trials": (int(training.trial_rows.size), targets["training_trials"]),
        "training identity": (
            target_identity_sha256(training.trial_row, training.target_index, training.target_time),
            targets["training_identity_sha256"],
        ),
        "validation samples": (validation.n_samples, targets["validation_samples"]),
        "validation trials": (int(validation.trial_rows.size), targets["validation_trials"]),
        "validation identity": (
            _validation_identity_sha256(validation),
            targets["validation_identity_sha256"],
        ),
        "protocol sample counts": (
            (training.n_samples, validation.n_samples),
            (protocol.training_samples, protocol.validation_samples),
        ),
        "unit order": ([int(unit) for unit in training.unit_ids], selected["unit_ids"]),
    }
    differing = [name for name, (found, required) in identities.items() if found != required]
    stored = frozen.predictions
    exact = {
        "sample_index": np.arange(validation.n_samples, dtype=np.int64),
        "trial_row": validation.trial_row,
        "reference_index": validation.reference_index,
        "target_index": validation.target_index,
        "target_time": validation.target_time,
        "target_velocity_x": validation.targets[:, 0],
        "target_velocity_y": validation.targets[:, 1],
    }
    if len(stored) != validation.n_samples:
        differing.append("saved validation rows")
    else:
        differing += [
            name
            for name, values in exact.items()
            if not np.array_equal(stored[name].to_numpy(dtype=values.dtype), values)
        ]
    if differing:
        raise GruError(f"The rebuilt targets differ from the frozen record: {differing}")

    sequences = {
        split: build_sequences(
            identity,
            timestamps=session.behavior_timestamps,
            spike_times=session.spike_times,
            trials=session.trials,
            representation=frozen.representation,
        )
        for split, identity in ((TRAIN, training), (VALIDATION, validation))
    }
    if frozen.representation == InputRepresentation(spec.history_ms, spec.bin_ms) and not all(
        np.array_equal(sequences[split].counts, identity.counts)
        for split, identity in ((TRAIN, training), (VALIDATION, validation))
    ):
        raise GruError("The rebuilt sequences differ from the official baseline spike counts.")
    normalization = fit_sequence_normalization(sequences[TRAIN], std_floor=protocol.std_floor)
    if normalization.as_dict() != selected["normalization"]:
        raise GruError("The refitted training normalization differs from the frozen record.")

    device = torch.device(device_info.device)
    return PreparedData(
        training=training,
        validation=validation,
        normalization=normalization,
        train_x=torch.from_numpy(normalization.apply_counts(sequences[TRAIN].counts)).to(device),
        train_y=torch.from_numpy(
            normalization.apply_targets(sequences[TRAIN].targets).astype(np.float32)
        ).to(device),
        validation_x=torch.from_numpy(
            normalization.apply_counts(sequences[VALIDATION].counts)
        ).to(device),
        validation_y=normalization.apply_targets(sequences[VALIDATION].targets),
    )


def _restore_reused(
    frozen: FrozenInput, data: PreparedData, plan: RunPlan, device_info: DeviceInfo
) -> list[CandidateRun]:
    """Load the reused input-selection checkpoints and verify their metrics."""
    device = torch.device(device_info.device)
    protocol, representation, runs = frozen.protocol, frozen.representation, frozen.runs
    restored = []
    for architecture, seed in plan.reused:
        name = f"checkpoint_seed{seed}.pt"
        row = runs[
            (runs["history_ms"] == representation.history_ms)
            & (runs["bin_ms"] == representation.bin_ms)
            & (runs["seed"] == seed)
        ]
        if len(row) != 1:
            raise GruError(f"The input-selection table has no single run for seed {seed}.")
        row = row.iloc[0]

        model, metadata = load_checkpoint(frozen.directory / name)
        expected = {
            "history_ms": representation.history_ms,
            "bin_ms": representation.bin_ms,
            "interval_end_lag_ms": protocol.lag_ms,
            "seed": seed,
            "best_epoch": int(row["best_epoch"]),
            "hidden_size": architecture.hidden_size,
            "learning_rate": architecture.learning_rate,
            "n_units": data.training.n_units,
            "protocol_sha256": frozen.protocol_sha256,
        }
        differing = [key for key, value in expected.items() if metadata.get(key) != value]
        if differing or model.num_layers != architecture.num_layers:
            raise GruError(f"Reused checkpoint {name} has unexpected metadata: {differing}")

        normalized = predict_normalized(
            model.to(device), data.validation_x, batch_size=protocol.batch_size
        )
        prediction = data.normalization.invert_targets(normalized)
        metrics = summarize_predictions(
            data.validation.targets, prediction, axis_names=AXIS_NAMES
        )
        validation_mse = _mse(normalized, data.validation_y)
        saved = frozen.predictions[[f"seed{seed}_velocity_{axis}" for axis in AXIS_NAMES]]
        history = [
            {key: value for key, value in record.items() if key != "seed"}
            for record in frozen.histories[frozen.histories["seed"] == seed].to_dict("records")
        ]
        best_record = next(
            (record for record in history if record["epoch"] == int(row["best_epoch"])), None
        )
        reproduced = (
            np.array_equal(prediction, saved.to_numpy(dtype=np.float64))
            and all(_metric(metrics, column) == float(row[column]) for column in _METRIC_COLUMNS)
            and validation_mse == float(row["best_validation_mse"])
            and len(history) == int(row["epochs_run"])
            and best_record is not None
            and bool(best_record["improved"])
        )
        if not reproduced:
            raise GruError(f"Reused checkpoint {name} does not reproduce its frozen record.")

        restored.append(
            CandidateRun(
                architecture=architecture,
                seed=seed,
                source="reused",
                epochs_run=int(row["epochs_run"]),
                best_epoch=int(row["best_epoch"]),
                stopped_early=bool(row["stopped_early"]),
                best_validation_mse=validation_mse,
                history=history,
                normalized_prediction=normalized,
                metrics=metrics,
                checks={
                    "completed": _completed(
                        int(row["best_epoch"]),
                        int(row["epochs_run"]),
                        bool(row["stopped_early"]),
                        protocol,
                    ),
                    "checkpoint restored": validation_mse
                    == float(best_record["validation_mse"]),
                    "finite": _finite(prediction, metrics),
                    "all targets": prediction.shape == (protocol.validation_samples, 2),
                },
                reference={
                    "checkpoint": (frozen.relative_directory / name).as_posix(),
                    "sha256": frozen.summary["artifact_hashes"][name],
                },
            )
        )
    return restored


def _determinism_probes(
    data: PreparedData, protocol: GruProtocol, search: ArchitectureSearch
) -> dict[str, bool]:
    """Check training determinism for each layer and dropout setting."""
    probe = replace(protocol, max_epochs=2)
    subset_x, subset_y = data.train_x[:_PROBE_SAMPLES], data.train_y[:_PROBE_SAMPLES]
    reference = subset_y.double().cpu().numpy()
    families = sorted({(item.num_layers, item.dropout) for item in search.architectures})
    results = {}
    for num_layers, dropout in families:
        replays = [
            train_run(
                train_x=subset_x,
                train_y=subset_y,
                validation_x=subset_x,
                validation_y=reference,
                seed=protocol.seeds[0],
                protocol=probe,
                hidden_size=max(search.hidden_sizes),
                learning_rate=search.learning_rates[0],
                num_layers=num_layers,
                dropout=dropout,
            )
            for _ in range(2)
        ]
        first, second = (replay.state for replay in replays)
        label = f"{num_layers} layer(s) / dropout {dropout:g}"
        results[label] = first.keys() == second.keys() and all(
            torch.equal(first[name], second[name]) for name in first
        )
        if not results[label]:
            raise GruError(f"Two identical training replays diverged for {label}.")
    logger.info("Deterministic replay probes passed for %s", ", ".join(results))
    return results


def _train_new(plan: RunPlan, data: PreparedData, protocol: GruProtocol) -> list[CandidateRun]:
    trained: list[CandidateRun] = []
    started = time.perf_counter()
    total = len(plan.new)
    for index, (architecture, seed) in enumerate(plan.new, start=1):
        logger.info("Run %d/%d: %s, seed %d", index, total, architecture.label, seed)
        run_started = time.perf_counter()
        result = train_run(
            train_x=data.train_x,
            train_y=data.train_y,
            validation_x=data.validation_x,
            validation_y=data.validation_y,
            seed=seed,
            protocol=protocol,
            hidden_size=architecture.hidden_size,
            learning_rate=architecture.learning_rate,
            num_layers=architecture.num_layers,
            dropout=architecture.dropout,
            progress=_progress_logger(run_started),
        )
        prediction = data.normalization.invert_targets(result.normalized_prediction)
        metrics = summarize_predictions(
            data.validation.targets, prediction, axis_names=AXIS_NAMES
        )
        trained.append(
            CandidateRun(
                architecture=architecture,
                seed=seed,
                source="trained",
                epochs_run=result.epochs_run,
                best_epoch=result.best_epoch,
                stopped_early=result.stopped_early,
                best_validation_mse=result.best_validation_mse,
                history=result.history,
                normalized_prediction=result.normalized_prediction,
                metrics=metrics,
                checks={
                    "completed": _completed(
                        result.best_epoch, result.epochs_run, result.stopped_early, protocol
                    ),
                    "checkpoint restored": (
                        result.restored_validation_mse == result.best_validation_mse
                    ),
                    "finite": _finite(prediction, metrics),
                    "all targets": prediction.shape == (protocol.validation_samples, 2),
                },
                state=result.state,
            )
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "  finished: %d epochs, best epoch %d, validation mean R2 %.6f "
            "(%.1f s this run, %.1f min total, about %.0f min left)",
            result.epochs_run,
            result.best_epoch,
            metrics["r2_mean"],
            time.perf_counter() - run_started,
            elapsed / 60.0,
            elapsed / index * (total - index) / 60.0,
        )
    return trained


def _progress_logger(started: float):
    def report(record: dict[str, float | int | bool]) -> None:
        epoch = int(record["epoch"])
        if epoch == 1 or epoch % _PROGRESS_EVERY == 0:
            logger.info(
                "  epoch %3d  train MSE %.5f  validation MSE %.5f  best epoch %3d  %.1f s",
                epoch,
                float(record["train_mse"]),
                float(record["validation_mse"]),
                int(record["best_epoch"]),
                time.perf_counter() - started,
            )

    return report


def _input_fingerprint(data: PreparedData) -> str:
    digest = hashlib.sha256()
    for tensor in (data.train_x, data.train_y, data.validation_x):
        digest.update(tensor.detach().cpu().numpy().tobytes())
    digest.update(np.ascontiguousarray(data.validation_y).tobytes())
    return digest.hexdigest()


def _completed(best_epoch: int, epochs_run: int, stopped_early: bool, protocol: GruProtocol) -> bool:
    return 1 <= best_epoch <= epochs_run <= protocol.max_epochs and (
        stopped_early or epochs_run == protocol.max_epochs
    )


def _finite(prediction: np.ndarray, metrics: dict[str, Any]) -> bool:
    return bool(np.all(np.isfinite(prediction))) and all(
        metrics[key] is not None and np.isfinite(metrics[key])
        for key in ("r2_mean", "pearson_r_mean", "rmse_cm_s_mean")
    )


def _pre_training_checks(frozen: FrozenInput, data: PreparedData) -> list[dict[str, Any]]:
    """Return the checks completed before training."""
    return [
        _check(
            "the frozen input selection matches its recorded hashes",
            True,
            "protocol"
            + ("" if frozen.protocol_bytes_identical else " (identical apart from line endings)")
            + ", summary, selected input, run table, checkpoints, histories, and predictions",
        ),
        _check(
            "the device and library versions equal the input-selection record",
            True,
            f"{frozen.selected['device']['device_name']}, torch "
            f"{frozen.selected['device']['torch_version']}, strict determinism",
        ),
        _check(
            "the exact frozen training and validation targets were rebuilt",
            True,
            f"{data.training.n_samples} training ({data.training.trial_rows.size} trials) and "
            f"{data.validation.n_samples} validation ({data.validation.trial_rows.size} trials) "
            "targets; identity hashes equal the frozen record",
        ),
        _check(
            "the frozen training-only normalization was reproduced exactly",
            True,
            f"{frozen.representation.label} sequences; per-unit count and per-axis target "
            "statistics equal the frozen record",
        ),
        _check(
            "the reused checkpoints reproduce their frozen predictions and metrics",
            True,
            f"{len(frozen.protocol.seeds)} reloaded {frozen.baseline.label} runs; "
            "predictions, R2, correlation, RMSE, and best validation MSE match exactly",
        ),
    ]


def _search_checks(
    candidates: list[CandidateRun],
    frozen: FrozenInput,
    search: ArchitectureSearch,
    probes: dict[str, bool],
    inputs_unchanged: bool,
) -> list[dict[str, Any]]:
    seeds = sorted(frozen.protocol.seeds)
    trained = [run for run in candidates if run.source == "trained"]
    reused = [run for run in candidates if run.source == "reused"]
    seeds_present = all(
        sorted(run.seed for run in candidates if run.architecture == architecture) == seeds
        for architecture in search.architectures
    )
    reused_frozen = (
        len(reused) == search.expected_reused_runs
        and all(run.architecture == frozen.baseline for run in reused)
        and not any(run.architecture == frozen.baseline for run in trained)
        and all(
            run.reference is not None
            and run.reference["sha256"]
            == frozen.summary["artifact_hashes"][f"checkpoint_seed{run.seed}.pt"]
            for run in reused
        )
    )
    unique_runs = len(frozen.runs) + len(trained)

    def failing(name: str) -> list[str]:
        return [f"{run.architecture.label}/seed {run.seed}" for run in candidates if not run.checks[name]]

    return [
        _check(
            "every planned new run completed",
            len(trained) == search.expected_new_runs and not failing("completed"),
            f"{len(trained)} of {search.expected_new_runs} new runs stopped by patience or "
            "the epoch limit",
        ),
        _check(
            "every configuration has every seed",
            seeds_present and len(candidates) == len(search.architectures) * len(seeds),
            f"seeds {seeds} for each of {len(search.architectures)} configurations: "
            f"{len(candidates)} entries",
        ),
        _check(
            "the reused entries are the frozen input-selection runs",
            reused_frozen
            and unique_runs == search.expected_new_runs + len(frozen.runs),
            f"{len(reused)} reused {frozen.baseline.label} entries; {unique_runs} unique "
            f"training runs ({len(frozen.runs)} input-selection and {len(trained)} new)",
        ),
        _check(
            "every run used the frozen input and targets",
            inputs_unchanged and not failing("all targets"),
            f"one {frozen.representation.label} input; {frozen.protocol.training_samples} "
            f"training and {frozen.protocol.validation_samples} validation targets, "
            "unchanged across runs",
        ),
        _check(
            "every run restored its best checkpoint",
            not failing("checkpoint restored"),
            "restored validation MSE equals the best recorded epoch",
        ),
        _check(
            "every prediction and metric is finite",
            not failing("finite"),
            f"{frozen.protocol.validation_samples} finite predictions per run",
        ),
        _check(
            "training is deterministic for every searched layer and dropout setting",
            bool(probes) and all(probes.values()),
            "two identical training replays produced bit-identical weights for "
            + ", ".join(probes),
        ),
    ]


def _scores(
    candidates: list[CandidateRun], search: ArchitectureSearch, protocol: GruProtocol
) -> list[ArchitectureScore]:
    return [
        ArchitectureScore(
            architecture=architecture,
            seed_scores=tuple(
                float(run.metrics["r2_mean"])
                for seed in protocol.seeds
                for run in candidates
                if run.architecture == architecture and run.seed == seed
            ),
        )
        for architecture in search.architectures
    ]


def _axis_means(architecture: GruArchitecture, candidates: list[CandidateRun]) -> dict[str, float]:
    members = [run for run in candidates if run.architecture == architecture]
    return {
        axis: float(np.mean([run.metrics["r2"][axis] for run in members])) for axis in AXIS_NAMES
    }


def _write_final(
    output: Path,
    architecture: GruArchitecture,
    candidates: list[CandidateRun],
    frozen: FrozenInput,
    data: PreparedData,
    session: SessionData,
    search_sha256: str,
    device_info: DeviceInfo,
) -> dict[str, Any]:
    """Save the selected configuration's three runs and verify the files."""
    protocol, representation = frozen.protocol, frozen.representation
    selected = sorted(
        (run for run in candidates if run.architecture == architecture), key=lambda run: run.seed
    )
    checkpoints: dict[int, Path] = {}
    copies_intact = True
    for run in selected:
        path = output / f"checkpoint_seed{run.seed}.pt"
        if run.source == "reused":
            shutil.copyfile(frozen.directory / path.name, path)
            copies_intact &= compute_sha256(path) == frozen.summary["artifact_hashes"][path.name]
        else:
            save_checkpoint(
                path,
                run.state,
                {
                    "history_ms": representation.history_ms,
                    "bin_ms": representation.bin_ms,
                    "interval_end_lag_ms": protocol.lag_ms,
                    "seed": run.seed,
                    "best_epoch": run.best_epoch,
                    **architecture.as_dict(),
                    "n_units": data.training.n_units,
                    "protocol_sha256": frozen.protocol_sha256,
                    "search_sha256": search_sha256,
                },
            )
        checkpoints[run.seed] = path

    histories_path = output / "training_histories.csv"
    if all(run.source == "reused" for run in selected):
        shutil.copyfile(frozen.directory / histories_path.name, histories_path)
        copies_intact &= (
            compute_sha256(histories_path) == frozen.summary["artifact_hashes"][histories_path.name]
        )
    else:
        pd.DataFrame(
            [{"seed": run.seed, **record} for run in selected for record in run.history]
        ).to_csv(histories_path, index=False, lineterminator="\n")

    validation = data.validation
    trial_ids = session.trials.set_index("trial_row")["trial_id"]
    columns: dict[str, np.ndarray] = {
        "sample_index": np.arange(validation.n_samples, dtype=np.int64),
        "trial_row": validation.trial_row,
        "trial_id": trial_ids.loc[validation.trial_row].to_numpy(dtype=np.int64),
        "reference_index": validation.reference_index,
        "target_index": validation.target_index,
        "target_time": validation.target_time,
        "target_velocity_x": validation.targets[:, 0],
        "target_velocity_y": validation.targets[:, 1],
    }
    for run in selected:
        prediction = data.normalization.invert_targets(run.normalized_prediction)
        columns[f"seed{run.seed}_velocity_x"] = prediction[:, 0]
        columns[f"seed{run.seed}_velocity_y"] = prediction[:, 1]
    predictions_path = output / "validation_predictions.csv"
    pd.DataFrame(columns).to_csv(predictions_path, index=False, lineterminator="\n")

    device = torch.device(device_info.device)
    reproduced = []
    for run in selected:
        model, metadata = load_checkpoint(checkpoints[run.seed])
        restored = predict_normalized(
            model.to(device), data.validation_x, batch_size=protocol.batch_size
        )
        reproduced.append(
            np.array_equal(restored, run.normalized_prediction)
            and metadata["seed"] == run.seed
            and model.num_layers == architecture.num_layers
            and int(metadata["hidden_size"]) == architecture.hidden_size
            and float(metadata.get("dropout", 0.0)) == architecture.dropout
            and float(metadata["learning_rate"]) == architecture.learning_rate
        )

    stored = _read_exact_csv(predictions_path)
    truth = stored[["target_velocity_x", "target_velocity_y"]].to_numpy(dtype=np.float64)
    recomputed = [
        np.allclose(
            r2_per_axis(
                truth,
                stored[[f"seed{run.seed}_velocity_{axis}" for axis in AXIS_NAMES]].to_numpy(),
            ),
            [run.metrics["r2"][axis] for axis in AXIS_NAMES],
            atol=_CSV_TOLERANCE,
            rtol=0.0,
        )
        for run in selected
    ]

    retained = {
        **{f"checkpoint_seed{seed}.pt": compute_sha256(path) for seed, path in checkpoints.items()},
        "training_histories.csv": compute_sha256(histories_path),
        "validation_predictions.csv": compute_sha256(predictions_path),
    }
    checks = [
        _check(
            "the three final checkpoints reproduce their predictions after reload",
            len(reproduced) == len(protocol.seeds) and all(reproduced) and copies_intact,
            "reloaded from disk; validation predictions identical bit for bit",
        ),
        _check(
            "the saved predictions reproduce every final run's R2",
            all(recomputed) and len(stored) == protocol.validation_samples,
            f"{len(stored)} rows per seed with trial and sample identifiers",
        ),
    ]
    if all(check["passed"] for check in checks):
        configuration_path = output / "final_configuration.json"
        configuration = _final_configuration(
            architecture, selected, frozen, search_sha256, device_info, retained
        )
        configuration_path.write_text(
            json.dumps(configuration, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        retained["final_configuration.json"] = compute_sha256(configuration_path)
    return {**retained, "checks": checks}


def _final_configuration(
    architecture: GruArchitecture,
    selected: list[CandidateRun],
    frozen: FrozenInput,
    search_sha256: str,
    device_info: DeviceInfo,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    protocol, record = frozen.protocol, frozen.selected
    return {
        "input_representation": record["input_representation"],
        "architecture": {
            "num_layers": architecture.num_layers,
            "hidden_size": architecture.hidden_size,
            "dropout": architecture.dropout,
            "dropout_placement": "on every layer's output sequence except the last",
            "head": "linear, two outputs (velocity x and y)",
            "hidden_state": "zero for every independent sample",
            "initialization": "PyTorch default, seeded per run",
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": architecture.learning_rate,
            "betas": list(protocol.betas),
            "epsilon": protocol.epsilon,
            "weight_decay": protocol.weight_decay,
            "scheduler": "none",
            "loss": "mse in normalized velocity coordinates",
            "batch_size": protocol.batch_size,
            "max_epochs": protocol.max_epochs,
            "early_stopping_patience": protocol.patience,
            "early_stopping_min_delta": protocol.min_delta,
            "gradient_clip_norm": protocol.gradient_clip_norm,
            "data_loader_workers": 0,
        },
        "seeds": list(protocol.seeds),
        "checkpoints": {
            str(run.seed): {
                "file": f"checkpoint_seed{run.seed}.pt",
                "source": run.source,
                "epochs_run": run.epochs_run,
                "best_epoch": run.best_epoch,
            }
            for run in selected
        },
        "normalization": record["normalization"],
        "unit_ids": record["unit_ids"],
        "targets": record["targets"],
        "recording": record["recording"],
        "device": device_info.as_dict(),
        "protocols": {
            "input_selection": {
                "path": frozen.protocol_path.as_posix(),
                "sha256": frozen.protocol_sha256,
            },
            "architecture_selection": {"path": _SEARCH.as_posix(), "sha256": search_sha256},
        },
        "input_selection_summary": {
            "path": (frozen.relative_directory / "summary.json").as_posix(),
            "sha256": compute_sha256(frozen.directory / "summary.json"),
        },
        "artifact_hashes": artifact_hashes,
    }


def _summary(
    *,
    search: ArchitectureSearch,
    search_sha256: str,
    frozen: FrozenInput,
    device_info: DeviceInfo,
    session: SessionData,
    data: PreparedData,
    candidates: list[CandidateRun],
    ranked: list[ArchitectureScore],
    selected: ArchitectureScore | None,
    checks: list[dict[str, Any]],
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    protocol = frozen.protocol
    order = {score.architecture: position + 1 for position, score in enumerate(ranked)}
    best = ranked[0].mean if ranked else None
    configurations = []
    for score in _scores(candidates, search, protocol):
        members = [run for run in candidates if run.architecture == score.architecture]
        configurations.append(
            {
                **score.architecture.as_dict(),
                "source": members[0].source,
                "rank": order.get(score.architecture),
                "seed_r2_mean": dict(zip((str(seed) for seed in protocol.seeds), score.seed_scores)),
                "r2_mean": score.mean,
                "r2_seed_sd": score.spread,
                "r2_seed_min": float(min(score.seed_scores)),
                "r2_seed_max": float(max(score.seed_scores)),
                **{
                    f"r2_{axis}_mean": value
                    for axis, value in _axis_means(score.architecture, candidates).items()
                },
                "pearson_r_mean": float(np.mean([run.metrics["pearson_r_mean"] for run in members])),
                "rmse_cm_s_mean": float(np.mean([run.metrics["rmse_cm_s_mean"] for run in members])),
                "best_epoch_mean": float(np.mean([run.best_epoch for run in members])),
                "tied_with_best": best is not None and best - score.mean <= search.tie_tolerance,
                "selected": selected is not None and score.architecture == selected.architecture,
            }
        )
    input_selected = frozen.summary["selected"]
    return {
        "procedure": {
            "name": "GRU architecture and learning-rate selection",
            "development_set": "validation, used for early stopping and selection",
            "fixed": "every input-selection rule except layer count, hidden size, learning "
            "rate, and dropout",
            "grid": {
                "hidden_sizes": list(search.hidden_sizes),
                "learning_rates": list(search.learning_rates),
                "dropout_by_num_layers": {
                    str(layers): list(values) for layers, values in search.dropout_by_num_layers
                },
                "seeds": list(protocol.seeds),
            },
            "runs": {
                "configurations": len(search.architectures),
                "entries": len(candidates),
                "new_runs": sum(run.source == "trained" for run in candidates),
                "reused_runs": sum(run.source == "reused" for run in candidates),
                "input_selection_runs": len(frozen.runs),
                "unique_training_runs": len(frozen.runs)
                + sum(run.source == "trained" for run in candidates),
            },
            "ranking": "arithmetic mean of seed-level mean R2; within the tie tolerance lower "
            "seed sample SD, then fewer layers, smaller hidden size, smaller dropout, and "
            "smaller learning rate",
            "tie_tolerance": search.tie_tolerance,
        },
        "input_selection": {
            "summary": (frozen.relative_directory / "summary.json").as_posix(),
            "summary_sha256": search.input_summary_sha256,
            "protocol": {"path": frozen.protocol_path.as_posix(), "sha256": frozen.protocol_sha256},
            "selected": {
                "history_ms": input_selected["history_ms"],
                "bin_ms": input_selected["bin_ms"],
                "n_bins": input_selected["n_bins"],
                "r2_mean": input_selected["r2_mean"],
                "r2_seed_sd": input_selected["r2_seed_sd"],
            },
            "reused_architecture": frozen.baseline.as_dict(),
        },
        "protocol": {"path": _SEARCH.as_posix(), "sha256": search_sha256},
        "device": device_info.as_dict(),
        "provenance": {
            "recording_sha256": session.identity.sha256,
            "training_identity_sha256": frozen.selected["targets"]["training_identity_sha256"],
            "validation_identity_sha256": frozen.selected["targets"][
                "validation_identity_sha256"
            ],
            "baseline_summary_sha256": frozen.summary["provenance"]["baseline_summary_sha256"],
            "input_selection_validation_predictions_sha256": frozen.summary["artifact_hashes"][
                "validation_predictions.csv"
            ],
        },
        "samples": {
            "training_samples": data.training.n_samples,
            "training_trials": int(data.training.trial_rows.size),
            "validation_samples": data.validation.n_samples,
            "validation_trials": int(data.validation.trial_rows.size),
            "n_units": int(data.training.n_units),
        },
        "configurations": configurations,
        "selected": None
        if selected is None
        else {
            **selected.architecture.as_dict(),
            "source": next(
                run.source for run in candidates if run.architecture == selected.architecture
            ),
            "r2_mean": selected.mean,
            "r2_seed_sd": selected.spread,
            **{
                f"r2_{axis}_mean": value
                for axis, value in _axis_means(selected.architecture, candidates).items()
            },
        },
        "checks": checks,
        "all_checks_passed": selected is not None and all(check["passed"] for check in checks),
        "held_out": "not constructed",
        "artifact_hashes": artifact_hashes,
    }


def _candidates_frame(candidates: list[CandidateRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **run.architecture.as_dict(),
                "seed": run.seed,
                "source": run.source,
                "reused_checkpoint": run.reference["checkpoint"] if run.reference else "",
                "reused_checkpoint_sha256": run.reference["sha256"] if run.reference else "",
                "epochs_run": run.epochs_run,
                "best_epoch": run.best_epoch,
                "stopped_early": run.stopped_early,
                "best_validation_mse": run.best_validation_mse,
                **{column: _metric(run.metrics, column) for column in _METRIC_COLUMNS},
                "checks_passed": run.passed,
            }
            for run in candidates
        ]
    )


def _read_exact_csv(path: Path) -> pd.DataFrame:
    """Read a CSV so every float parses back to the exact value that was written."""
    return pd.read_csv(path, float_precision="round_trip")


def _metric(metrics: dict[str, Any], column: str) -> float:
    key, axis = _METRIC_COLUMNS[column]
    value = metrics[key] if axis is None else metrics[key][axis]
    return float("nan") if value is None else float(value)


def _mse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction, dtype=np.float64) - reference)))


def _validation_identity_sha256(identity: FeatureSet) -> str:
    digest = hashlib.sha256(b"step5-validation-target-identity-v1\0")
    for array, dtype in (
        (identity.trial_row, "<i8"),
        (identity.target_index, "<i8"),
        (identity.target_time, "<f8"),
    ):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        digest.update(np.asarray([canonical.size], dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _print_report(
    ranked: list[ArchitectureScore],
    selected: ArchitectureScore | None,
    checks: list[dict[str, Any]],
    output: Path,
) -> None:
    print("GRU architecture and learning-rate selection (validation development set)")
    for position, score in enumerate(ranked, start=1):
        seeds = ", ".join(f"{value:.6f}" for value in score.seed_scores)
        print(
            f"  {position:>2}. {score.architecture.label:<44} mean R2 {score.mean:.6f}"
            f"  sd {score.spread:.6f}  seeds [{seeds}]"
        )
    print(f"  Selected: {'none' if selected is None else selected.architecture.label}")
    width = max(len(str(check["name"])) for check in checks)
    print("\nChecks")
    for check in checks:
        status = "pass" if check["passed"] else "FAIL"
        print(f"  {str(check['name']):<{width}}  {status:<4}  {check['detail']}")
    print(f"\nArtifacts in {output}")


if __name__ == "__main__":
    raise SystemExit(main())
