"""Choose the GRU input history and bin width on the validation set.

Usage, from the repository root::

    python -m scripts.select_gru_input

Exit codes: 0 if all checks pass, 1 if a check or run fails, 2 if the recording
or an input file is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    RepresentationScore,
    RunResult,
    configure_determinism,
    load_checkpoint,
    load_protocol,
    predict_normalized,
    rank_representations,
    save_checkpoint,
    train_run,
)
from neural_decoding.joint_decoder import target_identity_sha256
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.metrics import MetricError, r2_per_axis, summarize_predictions
from neural_decoding.sequences import (
    InputRepresentation,
    SequenceError,
    SequenceNormalization,
    SequenceSet,
    build_sequences,
    fit_sequence_normalization,
)
from neural_decoding.splits import TRAIN, VALIDATION

logger = get_logger("select_gru_input")

AXIS_NAMES = ("x", "y")

_PROTOCOL = Path("configs/gru_input_selection.yaml")
_BASELINE_SUMMARY = Path("results/baseline_summary.json")
_BASELINE_PREDICTIONS = Path("results/baselines/validation_predictions.csv")
_KALMAN_SELECTION = Path("results/kalman_decoder_selection/selected_config.json")
_OUTPUT_DIRECTORY = Path("results/internal/gru_input_selection")
_CSV_TOLERANCE = 1e-12
_PROBE_SAMPLES = 2048
_PROGRESS_EVERY = 10


@dataclass(frozen=True)
class TargetEvidence:
    """Baseline target identities and rebuilt feature sets."""

    training: FeatureSet
    validation: FeatureSet
    training_identity_sha256: str
    validation_identity_sha256: str
    hashes: dict[str, str]


@dataclass(frozen=True)
class CompletedRun:
    """One finished run with its scores and runtime checks."""

    representation: InputRepresentation
    result: RunResult
    metrics: dict[str, Any]
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


@dataclass
class PreparedRepresentation:
    """Normalized inputs and integrity checks for one representation."""

    normalization: SequenceNormalization
    validation_x: np.ndarray
    checks: dict[str, bool]


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
        protocol_path = root / _PROTOCOL
        protocol = load_protocol(protocol_path)
        protocol_sha256 = compute_sha256(protocol_path)
        device_info = configure_determinism(protocol)
        session = load_session(config)
        evidence = _verify_targets(root, session, protocol)
    except (ConfigError, DatasetError, FileNotFoundError) as error:
        print(f"Cannot select the GRU input representation: {error}")
        print(
            "Set paths.nwb_file in configs/local.yaml, or set "
            f"{NWB_PATH_ENV_VAR}, and keep the official baseline artifacts."
        )
        return 2
    except (GruError, FeatureError, SequenceError, KeyError, ValueError) as error:
        print(f"GRU input selection failed before training: {error}")
        return 1

    logger.info(
        "Device %s (%s), torch %s, CUDA %s, cuDNN %s; strict determinism on",
        device_info.device,
        device_info.device_name,
        device_info.torch_version,
        device_info.cuda_version,
        device_info.cudnn_version,
    )
    try:
        runs, prepared, probe_passed = _run_search(session, protocol, device_info, evidence)
    except (GruError, SequenceError, MetricError, RuntimeError) as error:
        print(f"GRU input selection stopped: {error}")
        print("No representation was selected; nothing was written.")
        return 1

    checks = _search_checks(runs, prepared, protocol, probe_passed)
    output.mkdir(parents=True, exist_ok=False)
    runs_path = output / "runs.csv"
    _runs_frame(runs).to_csv(runs_path, index=False, lineterminator="\n")

    selected: RepresentationScore | None = None
    ranked: list[RepresentationScore] = []
    artifact_hashes: dict[str, str] = {"runs.csv": compute_sha256(runs_path)}
    if all(check["passed"] for check in checks):
        ranked = rank_representations(_scores(runs, protocol), tolerance=protocol.tie_tolerance)
        selected = ranked[0]
        retained = _write_selected(
            output,
            selected.representation,
            runs,
            prepared[selected.representation],
            session,
            evidence,
            protocol,
            protocol_sha256,
            device_info,
        )
        checks.extend(retained.pop("checks"))
        artifact_hashes.update(retained)

    summary = _summary(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        device_info=device_info,
        session=session,
        evidence=evidence,
        runs=runs,
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


def _verify_targets(root: Path, session: SessionData, protocol: GruProtocol) -> TargetEvidence:
    """Rebuild the baseline targets and compare them with the saved results."""
    paths = {
        "baseline_summary": root / _BASELINE_SUMMARY,
        "baseline_validation_predictions": root / _BASELINE_PREDICTIONS,
        "kalman_selected_config": root / _KALMAN_SELECTION,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")
    hashes = {f"{name}_sha256": compute_sha256(path) for name, path in paths.items()}

    baseline = json.loads(paths["baseline_summary"].read_text(encoding="utf-8"))
    configuration = baseline["configuration"]
    expected = {
        "recording": (baseline["recording"]["sha256"], session.identity.sha256),
        "lag": (int(configuration["lag_ms"]), protocol.lag_ms),
        "eligibility lag": (int(configuration["eligibility_lag_ms"]), protocol.lag_ms),
        "history": (int(configuration["history_ms"]), protocol.eligibility_history_ms),
        "cadence": (int(configuration["cadence_ms"]), protocol.cadence_ms),
        "training samples": (
            int(baseline["samples"]["training_samples"]),
            protocol.training_samples,
        ),
        "validation samples": (
            int(baseline["samples"]["validation_samples"]),
            protocol.validation_samples,
        ),
    }
    differing = [name for name, (found, required) in expected.items() if found != required]
    if differing:
        raise GruError(f"The official baseline record disagrees with the protocol: {differing}")

    spec = FeatureSpec(
        history_ms=int(configuration["history_ms"]),
        n_bins=int(configuration["n_bins"]),
        cadence_ms=int(configuration["cadence_ms"]),
    )
    identities = {
        split: build_split_features(
            session, split, lag_ms=protocol.lag_ms, eligibility_lag_ms=protocol.lag_ms, spec=spec
        )
        for split in (TRAIN, VALIDATION)
    }
    training, validation = identities[TRAIN], identities[VALIDATION]

    kalman = json.loads(paths["kalman_selected_config"].read_text(encoding="utf-8"))
    frozen_training_sha = str(kalman["selection_evidence"]["training_target_identity_sha256"])
    training_sha = target_identity_sha256(
        training.trial_row, training.target_index, training.target_time
    )
    if training.n_samples != protocol.training_samples or training_sha != frozen_training_sha:
        raise GruError(
            f"Training targets ({training.n_samples}, {training_sha[:12]}...) differ from the "
            f"frozen official identity ({protocol.training_samples}, {frozen_training_sha[:12]}...)."
        )

    stored = pd.read_csv(paths["baseline_validation_predictions"])
    exact = {
        "sample_index": np.arange(validation.n_samples, dtype=np.int64),
        "trial_row": validation.trial_row,
        "reference_index": validation.reference_index,
        "target_index": validation.target_index,
    }
    approximate = {
        "target_time": validation.target_time,
        "target_velocity_x": validation.targets[:, 0],
        "target_velocity_y": validation.targets[:, 1],
    }
    mismatched = [
        name
        for name, values in exact.items()
        if len(stored) != values.size
        or not np.array_equal(stored[name].to_numpy(dtype=np.int64), values)
    ] + [
        name
        for name, values in approximate.items()
        if len(stored) != values.size
        or not np.allclose(
            stored[name].to_numpy(dtype=np.float64), values, atol=_CSV_TOLERANCE, rtol=0.0
        )
    ]
    if mismatched or validation.n_samples != protocol.validation_samples:
        raise GruError(
            f"Validation targets differ from the official baseline record in {mismatched} "
            f"({validation.n_samples} samples)."
        )

    return TargetEvidence(
        training=training,
        validation=validation,
        training_identity_sha256=training_sha,
        validation_identity_sha256=_validation_identity_sha256(validation),
        hashes=hashes,
    )


def _run_search(
    session: SessionData,
    protocol: GruProtocol,
    device_info: DeviceInfo,
    evidence: TargetEvidence,
) -> tuple[list[CompletedRun], dict[InputRepresentation, PreparedRepresentation], bool]:
    device = torch.device(device_info.device)
    representations = protocol.representations
    total = len(representations) * len(protocol.seeds)
    runs: list[CompletedRun] = []
    prepared: dict[InputRepresentation, PreparedRepresentation] = {}
    probe_passed = False
    started = time.perf_counter()

    for representation in representations:
        logger.info("Building %s sequences", representation.label)
        sequences = {
            split: build_sequences(
                identity,
                timestamps=session.behavior_timestamps,
                spike_times=session.spike_times,
                trials=session.trials,
                representation=representation,
            )
            for split, identity in (
                (TRAIN, evidence.training),
                (VALIDATION, evidence.validation),
            )
        }
        normalization = fit_sequence_normalization(sequences[TRAIN], std_floor=protocol.std_floor)
        checks = _representation_checks(representation, sequences, normalization, evidence)
        train_x = torch.from_numpy(normalization.apply_counts(sequences[TRAIN].counts)).to(device)
        train_y = torch.from_numpy(
            normalization.apply_targets(sequences[TRAIN].targets).astype(np.float32)
        ).to(device)
        validation_np = normalization.apply_counts(sequences[VALIDATION].counts)
        validation_x = torch.from_numpy(validation_np).to(device)
        validation_y = normalization.apply_targets(sequences[VALIDATION].targets)
        prepared[representation] = PreparedRepresentation(normalization, validation_np, checks)

        if not probe_passed:
            probe_passed = _determinism_probe(train_x, train_y, protocol)
            if not probe_passed:
                raise GruError("Two identical training replays diverged on the search device.")
            logger.info("Deterministic replay probe passed on %s", device_info.device)

        for seed in protocol.seeds:
            index = len(runs) + 1
            logger.info("Run %d/%d: %s, seed %d", index, total, representation.label, seed)
            run_started = time.perf_counter()
            result = train_run(
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                validation_y=validation_y,
                seed=seed,
                protocol=protocol,
                hidden_size=protocol.hidden_size,
                learning_rate=protocol.learning_rate,
                progress=_progress_logger(run_started),
            )
            prediction = normalization.invert_targets(result.normalized_prediction)
            metrics = summarize_predictions(
                sequences[VALIDATION].targets, prediction, axis_names=AXIS_NAMES
            )
            runs.append(
                CompletedRun(
                    representation=representation,
                    result=result,
                    metrics=metrics,
                    checks=_run_checks(result, metrics, prediction, protocol),
                )
            )
            logger.info(
                "  finished: %d epochs, best epoch %d, validation mean R2 %.6f "
                "(%.1f s this run, %.1f min total)",
                result.epochs_run,
                result.best_epoch,
                metrics["r2_mean"],
                time.perf_counter() - run_started,
                (time.perf_counter() - started) / 60.0,
            )
        del train_x, train_y, validation_x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return runs, prepared, probe_passed


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


def _determinism_probe(
    train_x: torch.Tensor, train_y: torch.Tensor, protocol: GruProtocol
) -> bool:
    """Repeat a short training run and compare the resulting weights."""
    probe = replace(protocol, max_epochs=2)
    subset_x, subset_y = train_x[:_PROBE_SAMPLES], train_y[:_PROBE_SAMPLES]
    reference = subset_y.double().cpu().numpy()
    replays = [
        train_run(
            train_x=subset_x,
            train_y=subset_y,
            validation_x=subset_x,
            validation_y=reference,
            seed=protocol.seeds[0],
            protocol=probe,
            hidden_size=protocol.hidden_size,
            learning_rate=protocol.learning_rate,
        )
        for _ in range(2)
    ]
    first, second = (replay.state for replay in replays)
    return first.keys() == second.keys() and all(
        torch.equal(first[name], second[name]) for name in first
    )


def _representation_checks(
    representation: InputRepresentation,
    sequences: dict[str, SequenceSet],
    normalization: SequenceNormalization,
    evidence: TargetEvidence,
) -> dict[str, bool]:
    identities = {TRAIN: evidence.training, VALIDATION: evidence.validation}
    identical = all(
        np.array_equal(sequences[split].trial_row, identity.trial_row)
        and np.array_equal(sequences[split].reference_index, identity.reference_index)
        and np.array_equal(sequences[split].target_index, identity.target_index)
        and np.array_equal(sequences[split].target_time, identity.target_time)
        and np.array_equal(sequences[split].targets, identity.targets)
        for split, identity in identities.items()
    )
    checks = {
        "official targets": identical,
        "training-only normalization": (
            normalization.n_samples == evidence.training.n_samples
            and normalization.n_elements == evidence.training.n_samples * representation.n_bins
            and set(normalization.trial_rows) == set(evidence.training.trial_rows.tolist())
        ),
    }
    baseline_window = InputRepresentation(
        evidence.training.spec.history_ms, evidence.training.spec.bin_ms
    )
    if representation == baseline_window:
        checks["matches baseline counts"] = all(
            np.array_equal(sequences[split].counts, identity.counts)
            for split, identity in identities.items()
        )
    return checks


def _run_checks(
    result: RunResult, metrics: dict[str, Any], prediction: np.ndarray, protocol: GruProtocol
) -> dict[str, bool]:
    finite_metrics = all(
        metrics[key] is not None and np.isfinite(metrics[key])
        for key in ("r2_mean", "pearson_r_mean", "rmse_cm_s_mean")
    )
    return {
        "completed": (
            1 <= result.best_epoch <= result.epochs_run <= protocol.max_epochs
            and (result.stopped_early or result.epochs_run == protocol.max_epochs)
        ),
        "checkpoint restored": result.restored_validation_mse == result.best_validation_mse,
        "finite": bool(np.all(np.isfinite(prediction))) and finite_metrics,
        "all targets": prediction.shape == (protocol.validation_samples, 2),
    }


def _search_checks(
    runs: list[CompletedRun],
    prepared: dict[InputRepresentation, PreparedRepresentation],
    protocol: GruProtocol,
    probe_passed: bool,
) -> list[dict[str, Any]]:
    expected_runs = len(protocol.representations) * len(protocol.seeds)
    seeds_present = all(
        sorted(run.result.seed for run in runs if run.representation == representation)
        == sorted(protocol.seeds)
        for representation in protocol.representations
    )
    representation_checks = [item.checks for item in prepared.values()]
    baseline_matches = [
        checks["matches baseline counts"]
        for checks in representation_checks
        if "matches baseline counts" in checks
    ]

    def failing(name: str) -> list[str]:
        return [
            f"{run.representation.history_ms}/{run.representation.bin_ms}/seed {run.result.seed}"
            for run in runs
            if not run.checks[name]
        ]

    return [
        _check(
            "every planned run completed",
            len(runs) == expected_runs and not failing("completed"),
            f"{len(runs)} of {expected_runs} runs stopped by patience or the epoch limit",
        ),
        _check(
            "every representation has every seed",
            seeds_present,
            f"seeds {list(protocol.seeds)} for each of {len(protocol.representations)} "
            "representations",
        ),
        _check(
            "every run used the exact official targets",
            all(checks["official targets"] for checks in representation_checks)
            and not failing("all targets"),
            f"{protocol.training_samples} training and {protocol.validation_samples} "
            "validation targets, identical for every representation",
        ),
        _check(
            "the 200 ms / 50 ms sequences equal the baseline spike counts",
            baseline_matches == [True],
            "bit-identical to the official Ridge feature counts on both splits",
        ),
        _check(
            "normalization was fitted on training sequences only",
            all(checks["training-only normalization"] for checks in representation_checks),
            "one count scale per unit over every training element; training targets only",
        ),
        _check(
            "every run restored its best checkpoint",
            not failing("checkpoint restored"),
            "restored validation MSE equals the best recorded epoch exactly",
        ),
        _check(
            "every prediction and metric is finite",
            not failing("finite"),
            f"{protocol.validation_samples} finite predictions per run",
        ),
        _check(
            "training is deterministic on the search device",
            probe_passed,
            "two identical training replays produced bit-identical weights",
        ),
    ]


def _scores(runs: list[CompletedRun], protocol: GruProtocol) -> list[RepresentationScore]:
    return [
        RepresentationScore(
            representation=representation,
            seed_scores=tuple(
                float(run.metrics["r2_mean"])
                for seed in protocol.seeds
                for run in runs
                if run.representation == representation and run.result.seed == seed
            ),
        )
        for representation in protocol.representations
    ]


def _write_selected(
    output: Path,
    representation: InputRepresentation,
    runs: list[CompletedRun],
    prepared: PreparedRepresentation,
    session: SessionData,
    evidence: TargetEvidence,
    protocol: GruProtocol,
    protocol_sha256: str,
    device_info: DeviceInfo,
) -> dict[str, Any]:
    """Save the selected representation's runs and verify the files."""
    selected = sorted(
        (run for run in runs if run.representation == representation),
        key=lambda run: run.result.seed,
    )
    n_units = int(evidence.validation.n_units)
    checkpoints: dict[int, Path] = {}
    for run in selected:
        path = output / f"checkpoint_seed{run.result.seed}.pt"
        save_checkpoint(
            path,
            run.result.state,
            {
                "history_ms": representation.history_ms,
                "bin_ms": representation.bin_ms,
                "interval_end_lag_ms": protocol.lag_ms,
                "seed": run.result.seed,
                "best_epoch": run.result.best_epoch,
                "hidden_size": protocol.hidden_size,
                "learning_rate": protocol.learning_rate,
                "n_units": n_units,
                "protocol_sha256": protocol_sha256,
            },
        )
        checkpoints[run.result.seed] = path

    histories_path = output / "training_histories.csv"
    pd.DataFrame(
        [{"seed": run.result.seed, **record} for run in selected for record in run.result.history]
    ).to_csv(histories_path, index=False, lineterminator="\n")

    validation = evidence.validation
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
        prediction = prepared.normalization.invert_targets(run.result.normalized_prediction)
        columns[f"seed{run.result.seed}_velocity_x"] = prediction[:, 0]
        columns[f"seed{run.result.seed}_velocity_y"] = prediction[:, 1]
    predictions_path = output / "validation_predictions.csv"
    pd.DataFrame(columns).to_csv(predictions_path, index=False, lineterminator="\n")

    device = torch.device(device_info.device)
    validation_x = torch.from_numpy(prepared.validation_x).to(device)
    reproduced = []
    for run in selected:
        model, metadata = load_checkpoint(checkpoints[run.result.seed])
        restored = predict_normalized(
            model.to(device), validation_x, batch_size=protocol.batch_size
        )
        reproduced.append(
            np.array_equal(restored, run.result.normalized_prediction)
            and metadata["seed"] == run.result.seed
            and metadata["protocol_sha256"] == protocol_sha256
        )

    stored = pd.read_csv(predictions_path)
    truth = stored[["target_velocity_x", "target_velocity_y"]].to_numpy(dtype=np.float64)
    recomputed = [
        np.allclose(
            r2_per_axis(
                truth,
                stored[[f"seed{run.result.seed}_velocity_{axis}" for axis in AXIS_NAMES]].to_numpy(),
            ),
            [run.metrics["r2"][axis] for axis in AXIS_NAMES],
            atol=_CSV_TOLERANCE,
            rtol=0.0,
        )
        for run in selected
    ]

    retained = {
        **{
            f"checkpoint_seed{seed}.pt": compute_sha256(path)
            for seed, path in checkpoints.items()
        },
        "training_histories.csv": compute_sha256(histories_path),
        "validation_predictions.csv": compute_sha256(predictions_path),
    }
    checks = [
        _check(
            "the three retained checkpoints reproduce their predictions after reload",
            len(reproduced) == len(protocol.seeds) and all(reproduced),
            "reloaded from disk; validation predictions identical bit for bit",
        ),
        _check(
            "the saved predictions reproduce every retained run's R2",
            all(recomputed) and len(stored) == protocol.validation_samples,
            f"{len(stored)} rows per seed with trial and sample identifiers",
        ),
    ]
    if all(check["passed"] for check in checks):
        configuration_path = output / "selected_input.json"
        configuration = _selected_configuration(
            representation,
            prepared.normalization,
            session,
            evidence,
            protocol,
            protocol_sha256,
            device_info,
            retained,
        )
        configuration_path.write_text(
            json.dumps(configuration, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        retained["selected_input.json"] = compute_sha256(configuration_path)
    return {**retained, "checks": checks}


def _selected_configuration(
    representation: InputRepresentation,
    normalization: SequenceNormalization,
    session: SessionData,
    evidence: TargetEvidence,
    protocol: GruProtocol,
    protocol_sha256: str,
    device_info: DeviceInfo,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "input_representation": {
            "history_ms": representation.history_ms,
            "bin_ms": representation.bin_ms,
            "n_bins": representation.n_bins,
            "interval_end_lag_ms": protocol.lag_ms,
            "cadence_ms": protocol.cadence_ms,
            "sequence_interval": "[t - lag - history, t - lag), chronological half-open bins",
        },
        "baseline_model": {
            "hidden_size": protocol.hidden_size,
            "learning_rate": protocol.learning_rate,
        },
        "protocol": {"path": _PROTOCOL.as_posix(), "sha256": protocol_sha256},
        "seeds": list(protocol.seeds),
        "normalization": normalization.as_dict(),
        "unit_ids": [int(unit) for unit in evidence.training.unit_ids],
        "targets": {
            "training_samples": evidence.training.n_samples,
            "training_trials": int(evidence.training.trial_rows.size),
            "training_identity_sha256": evidence.training_identity_sha256,
            "validation_samples": evidence.validation.n_samples,
            "validation_trials": int(evidence.validation.trial_rows.size),
            "validation_identity_sha256": evidence.validation_identity_sha256,
            "validation_source": _BASELINE_PREDICTIONS.as_posix(),
            "validation_source_sha256": evidence.hashes[
                "baseline_validation_predictions_sha256"
            ],
        },
        "recording": {"name": session.identity.name, "sha256": session.identity.sha256},
        "device": device_info.as_dict(),
        "artifact_hashes": artifact_hashes,
    }


def _summary(
    *,
    protocol: GruProtocol,
    protocol_sha256: str,
    device_info: DeviceInfo,
    session: SessionData,
    evidence: TargetEvidence,
    runs: list[CompletedRun],
    ranked: list[RepresentationScore],
    selected: RepresentationScore | None,
    checks: list[dict[str, Any]],
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    order = {score.representation: position + 1 for position, score in enumerate(ranked)}
    best = ranked[0].mean if ranked else None
    representations = []
    for score in _scores(runs, protocol):
        members = [run for run in runs if run.representation == score.representation]
        representations.append(
            {
                "history_ms": score.representation.history_ms,
                "bin_ms": score.representation.bin_ms,
                "n_bins": score.representation.n_bins,
                "rank": order.get(score.representation),
                "seed_r2_mean": dict(zip((str(seed) for seed in protocol.seeds), score.seed_scores)),
                "r2_mean": score.mean,
                "r2_seed_sd": score.spread,
                "r2_seed_min": float(min(score.seed_scores)),
                "r2_seed_max": float(max(score.seed_scores)),
                "r2_x_mean": float(np.mean([run.metrics["r2"]["x"] for run in members])),
                "r2_y_mean": float(np.mean([run.metrics["r2"]["y"] for run in members])),
                "pearson_r_mean": float(np.mean([run.metrics["pearson_r_mean"] for run in members])),
                "rmse_cm_s_mean": float(np.mean([run.metrics["rmse_cm_s_mean"] for run in members])),
                "tied_with_best": best is not None and best - score.mean <= protocol.tie_tolerance,
                "selected": selected is not None and score.representation == selected.representation,
            }
        )
    return {
        "procedure": {
            "name": "GRU input-representation selection",
            "development_set": "validation, used for early stopping and selection",
            "baseline_model": {
                "hidden_size": protocol.hidden_size,
                "learning_rate": protocol.learning_rate,
            },
            "grid": {
                "histories_ms": list(protocol.histories_ms),
                "bin_widths_ms": list(protocol.bin_widths_ms),
                "seeds": list(protocol.seeds),
                "interval_end_lag_ms": protocol.lag_ms,
            },
            "ranking": "arithmetic mean of seed-level mean R2; within the tie tolerance "
            "lower seed sample SD, then shorter history, then larger bin width",
            "tie_tolerance": protocol.tie_tolerance,
        },
        "protocol": {"path": _PROTOCOL.as_posix(), "sha256": protocol_sha256},
        "device": device_info.as_dict(),
        "provenance": {
            "recording_sha256": session.identity.sha256,
            "training_identity_sha256": evidence.training_identity_sha256,
            "validation_identity_sha256": evidence.validation_identity_sha256,
            **evidence.hashes,
        },
        "samples": {
            "training_samples": evidence.training.n_samples,
            "training_trials": int(evidence.training.trial_rows.size),
            "validation_samples": evidence.validation.n_samples,
            "validation_trials": int(evidence.validation.trial_rows.size),
            "n_units": int(evidence.training.n_units),
        },
        "representations": representations,
        "selected": None
        if selected is None
        else {
            "history_ms": selected.representation.history_ms,
            "bin_ms": selected.representation.bin_ms,
            "n_bins": selected.representation.n_bins,
            "r2_mean": selected.mean,
            "r2_seed_sd": selected.spread,
        },
        "checks": checks,
        "all_checks_passed": selected is not None and all(check["passed"] for check in checks),
        "held_out": "not constructed",
        "artifact_hashes": artifact_hashes,
    }


def _runs_frame(runs: list[CompletedRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "history_ms": run.representation.history_ms,
                "bin_ms": run.representation.bin_ms,
                "n_bins": run.representation.n_bins,
                "seed": run.result.seed,
                "epochs_run": run.result.epochs_run,
                "best_epoch": run.result.best_epoch,
                "stopped_early": run.result.stopped_early,
                "best_validation_mse": run.result.best_validation_mse,
                "r2_x": run.metrics["r2"]["x"],
                "r2_y": run.metrics["r2"]["y"],
                "r2_mean": run.metrics["r2_mean"],
                "pearson_r_x": run.metrics["pearson_r"]["x"],
                "pearson_r_y": run.metrics["pearson_r"]["y"],
                "pearson_r_mean": run.metrics["pearson_r_mean"],
                "rmse_cm_s_x": run.metrics["rmse_cm_s"]["x"],
                "rmse_cm_s_y": run.metrics["rmse_cm_s"]["y"],
                "rmse_cm_s_mean": run.metrics["rmse_cm_s_mean"],
                "checks_passed": run.passed,
            }
            for run in runs
        ]
    )


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
    ranked: list[RepresentationScore],
    selected: RepresentationScore | None,
    checks: list[dict[str, Any]],
    output: Path,
) -> None:
    print("GRU input-representation selection (validation development set)")
    for position, score in enumerate(ranked, start=1):
        seeds = ", ".join(f"{value:.6f}" for value in score.seed_scores)
        print(
            f"  {position}. {score.representation.label:<26} mean R2 {score.mean:.6f}"
            f"  sd {score.spread:.6f}  seeds [{seeds}]"
        )
    if selected is None:
        print("  Selected: none")
    else:
        print(f"  Selected: {selected.representation.label}")
    width = max(len(str(check["name"])) for check in checks)
    print("\nChecks")
    for check in checks:
        status = "pass" if check["passed"] else "FAIL"
        print(f"  {str(check['name']):<{width}}  {status:<4}  {check['detail']}")
    print(f"\nArtifacts in {output}")


if __name__ == "__main__":
    raise SystemExit(main())
