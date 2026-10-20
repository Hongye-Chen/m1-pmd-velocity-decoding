"""GRU velocity decoder, training loop, and checkpoint helpers.

Each sample is decoded on its own: the hidden state starts at zero and is not
carried over between samples. A linear head maps the last hidden state to
(vx, vy) in normalized units. Training uses PyTorch's deterministic algorithms,
so a run is reproducible on the same device and library versions.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from neural_decoding.seeding import set_global_seed
from neural_decoding.sequences import InputRepresentation

__all__ = [
    "DeviceInfo",
    "EarlyStopping",
    "GruDecoder",
    "GruError",
    "GruProtocol",
    "RepresentationScore",
    "RunResult",
    "configure_determinism",
    "load_checkpoint",
    "load_protocol",
    "predict_normalized",
    "rank_representations",
    "save_checkpoint",
    "train_run",
    "validate_architecture",
]

_CHECKPOINT_FORMAT = 1


class GruError(RuntimeError):
    """Raised when the GRU protocol, training, or a checkpoint is invalid."""


@dataclass(frozen=True)
class GruProtocol:
    """The tracked GRU input-selection protocol, parsed strictly."""

    lag_ms: int
    cadence_ms: int
    histories_ms: tuple[int, ...]
    bin_widths_ms: tuple[int, ...]
    eligibility_history_ms: int
    training_samples: int
    validation_samples: int
    hidden_size: int
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    gradient_clip_norm: float
    seeds: tuple[int, ...]
    std_floor: float
    cublas_workspace_config: str
    tie_tolerance: float

    @property
    def representations(self) -> tuple[InputRepresentation, ...]:
        """Return every history/width pair, shorter history and narrower width first."""
        return tuple(
            InputRepresentation(history, width)
            for history in self.histories_ms
            for width in self.bin_widths_ms
        )


def load_protocol(path: Path | str) -> GruProtocol:
    """Read the tracked protocol, refusing unknown, missing, or unsupported settings."""
    source = Path(path)
    if not source.is_file():
        raise GruError(f"The GRU protocol is missing at {source}.")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    sections = {
        "timing": {"interval_end_lag_ms", "cadence_ms", "histories_ms", "bin_widths_ms"},
        "targets": {"eligibility_history_ms", "training_samples", "validation_samples"},
        "architecture": {"num_layers", "hidden_size", "dropout"},
        "optimization": {
            "optimizer",
            "learning_rate",
            "betas",
            "epsilon",
            "weight_decay",
            "scheduler",
            "loss",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "gradient_clip_norm",
            "data_loader_workers",
        },
        "seeds": None,
        "normalization": {"std_floor"},
        "device": {
            "deterministic_algorithms",
            "cudnn_benchmark",
            "allow_tf32",
            "cublas_workspace_config",
        },
        "selection": {"tie_tolerance"},
    }
    if not isinstance(data, dict) or set(data) != set(sections):
        raise GruError(f"{source} must define exactly the sections {sorted(sections)}.")
    for name, keys in sections.items():
        if keys is not None and (not isinstance(data[name], dict) or set(data[name]) != keys):
            raise GruError(f"Section {name!r} of {source} must define exactly {sorted(keys)}.")

    timing, targets = data["timing"], data["targets"]
    architecture, optimization = data["architecture"], data["optimization"]
    device = data["device"]
    fixed = {
        "architecture.num_layers": (architecture["num_layers"], 1),
        "architecture.dropout": (float(architecture["dropout"]), 0.0),
        "optimization.optimizer": (optimization["optimizer"], "adamw"),
        "optimization.scheduler": (optimization["scheduler"], "none"),
        "optimization.loss": (optimization["loss"], "mse"),
        "optimization.data_loader_workers": (optimization["data_loader_workers"], 0),
        "device.deterministic_algorithms": (device["deterministic_algorithms"], True),
        "device.cudnn_benchmark": (device["cudnn_benchmark"], False),
        "device.allow_tf32": (device["allow_tf32"], False),
    }
    unsupported = [name for name, (value, required) in fixed.items() if value != required]
    if unsupported:
        raise GruError(f"Unsupported GRU protocol settings: {unsupported}.")

    protocol = GruProtocol(
        lag_ms=int(timing["interval_end_lag_ms"]),
        cadence_ms=int(timing["cadence_ms"]),
        histories_ms=tuple(int(value) for value in timing["histories_ms"]),
        bin_widths_ms=tuple(int(value) for value in timing["bin_widths_ms"]),
        eligibility_history_ms=int(targets["eligibility_history_ms"]),
        training_samples=int(targets["training_samples"]),
        validation_samples=int(targets["validation_samples"]),
        hidden_size=int(architecture["hidden_size"]),
        learning_rate=float(optimization["learning_rate"]),
        betas=tuple(float(value) for value in optimization["betas"]),  # type: ignore[arg-type]
        epsilon=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
        batch_size=int(optimization["batch_size"]),
        max_epochs=int(optimization["max_epochs"]),
        patience=int(optimization["early_stopping_patience"]),
        min_delta=float(optimization["early_stopping_min_delta"]),
        gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        seeds=tuple(int(value) for value in data["seeds"]),
        std_floor=float(data["normalization"]["std_floor"]),
        cublas_workspace_config=str(device["cublas_workspace_config"]),
        tie_tolerance=float(data["selection"]["tie_tolerance"]),
    )
    if len(protocol.betas) != 2 or len(set(protocol.seeds)) != len(protocol.seeds):
        raise GruError("The protocol needs two betas and distinct seeds.")
    if any(history > protocol.eligibility_history_ms for history in protocol.histories_ms):
        raise GruError("Every history must fit inside the target eligibility history.")
    for representation in protocol.representations:
        if representation.bin_ms % protocol.cadence_ms:
            raise GruError(f"{representation.label} is off the {protocol.cadence_ms} ms grid.")
    return protocol


@dataclass(frozen=True)
class DeviceInfo:
    """The one device used for a search and the versions that ran on it."""

    device: str
    device_name: str
    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    allow_tf32: bool
    cublas_workspace_config: str | None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def configure_determinism(protocol: GruProtocol) -> DeviceInfo:
    """Enable strict determinism and select CUDA when available, otherwise CPU."""
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and current != protocol.cublas_workspace_config:
        raise GruError(
            "CUDA was initialized before the deterministic cuBLAS workspace was configured."
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = protocol.cublas_workspace_config
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.are_deterministic_algorithms_enabled() or (
        torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise GruError("PyTorch did not enter strict deterministic mode.")

    use_cuda = torch.cuda.is_available()
    return DeviceInfo(
        device="cuda" if use_cuda else "cpu",
        device_name=torch.cuda.get_device_name(0) if use_cuda else "cpu",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version() if use_cuda else None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def validate_architecture(num_layers: int, dropout: float) -> None:
    """Refuse a layer count or dropout the decoder cannot apply as declared."""
    if int(num_layers) != num_layers or num_layers < 1:
        raise GruError(f"The GRU needs a positive whole number of layers, not {num_layers}.")
    if not 0.0 <= dropout < 1.0:
        raise GruError(f"Dropout must lie in [0, 1), not {dropout}.")
    if num_layers == 1 and dropout != 0.0:
        raise GruError(
            "Dropout acts only between stacked GRU layers; a one-layer decoder must use 0."
        )


class GruDecoder(nn.Module):
    """Stacked GRU layers read out by a linear two-output velocity head.

    Layers are separate one-layer GRUs with dropout applied to every layer's output
    sequence except the last, the same rule as a multi-layer ``nn.GRU``. Applying
    it explicitly keeps cuDNN's internal RNN dropout state out of training: on
    Windows that cached state aborts the interpreter at exit. The first layer is
    ``gru`` and later layers are ``upper``, so a one-layer checkpoint holds only
    ``gru`` and ``head`` parameters.
    """

    def __init__(
        self, n_units: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        validate_architecture(num_layers, dropout)
        self.gru = nn.GRU(input_size=n_units, hidden_size=hidden_size, batch_first=True)
        self.upper = nn.ModuleList(
            nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
            for _ in range(int(num_layers) - 1)
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden_size, 2)

    @property
    def num_layers(self) -> int:
        return 1 + len(self.upper)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, n_bins, n_units)`` sequences to ``(batch, 2)`` velocities."""
        initial = sequences.new_zeros(1, sequences.shape[0], self.gru.hidden_size)
        output, final = self.gru(sequences, initial)
        for layer in self.upper:
            output, final = layer(self.dropout(output), initial)
        return self.head(final[-1])


class EarlyStopping:
    """Track the best validation loss and stop after ``patience`` stale epochs."""

    def __init__(self, *, patience: int, min_delta: float) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value = float("inf")
        self.best_epoch = 0
        self.stale_epochs = 0

    def update(self, epoch: int, value: float) -> bool:
        """Record one epoch and return whether it improved by more than ``min_delta``."""
        if value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience


@dataclass(frozen=True)
class RunResult:
    """One completed run restored to its best validation checkpoint."""

    seed: int
    epochs_run: int
    best_epoch: int
    stopped_early: bool
    best_validation_mse: float
    restored_validation_mse: float
    history: list[dict[str, float | int | bool]]
    state: dict[str, torch.Tensor] = field(repr=False)
    normalized_prediction: np.ndarray = field(repr=False)


def predict_normalized(
    model: GruDecoder, sequences: torch.Tensor, *, batch_size: int
) -> np.ndarray:
    """Return normalized velocity predictions in sample order, without gradients."""
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, sequences.shape[0], batch_size):
            chunks.append(model(sequences[start : start + batch_size]).cpu().numpy())
    return np.concatenate(chunks)


def train_run(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: np.ndarray,
    seed: int,
    protocol: GruProtocol,
    hidden_size: int,
    learning_rate: float,
    num_layers: int = 1,
    dropout: float = 0.0,
    progress: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> RunResult:
    """Train one seeded run with validation early stopping and restore its best state.

    ``train_x`` and ``train_y`` are normalized tensors on the search device;
    ``validation_y`` holds normalized validation targets as float64.
    """
    if train_x.shape[0] != train_y.shape[0] or validation_x.shape[0] != validation_y.shape[0]:
        raise GruError("Sequences and targets are not row-aligned.")
    device = train_x.device
    set_global_seed(seed)
    model = GruDecoder(int(train_x.shape[2]), hidden_size, num_layers, dropout).to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=protocol.betas,
        eps=protocol.epsilon,
        weight_decay=protocol.weight_decay,
    )
    stopping = EarlyStopping(patience=protocol.patience, min_delta=protocol.min_delta)
    reference = np.asarray(validation_y, dtype=np.float64)
    n_train = int(train_x.shape[0])
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int | bool]] = []

    for epoch in range(1, protocol.max_epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        total = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, n_train, protocol.batch_size):
            batch = order[start : start + protocol.batch_size]
            loss = nn.functional.mse_loss(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol.gradient_clip_norm)
            optimizer.step()
            total += loss.detach().double() * batch.numel()
        train_loss = float(total.item()) / n_train

        prediction = predict_normalized(model, validation_x, batch_size=protocol.batch_size)
        validation_mse = _mse(prediction, reference)
        if not np.isfinite(train_loss) or not np.isfinite(validation_mse):
            raise GruError(f"Seed {seed} produced a non-finite loss at epoch {epoch}.")
        improved = stopping.update(epoch, validation_mse)
        if improved:
            best_state = {
                name: value.detach().to("cpu", copy=True)
                for name, value in model.state_dict().items()
            }
        record = {
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": validation_mse,
            "improved": improved,
            "best_epoch": stopping.best_epoch,
        }
        history.append(record)
        if progress is not None:
            progress(record)
        if stopping.should_stop:
            break

    if best_state is None:
        raise GruError(f"Seed {seed} never recorded a validation checkpoint.")
    model.load_state_dict(best_state)
    restored = predict_normalized(model, validation_x, batch_size=protocol.batch_size)
    return RunResult(
        seed=seed,
        epochs_run=len(history),
        best_epoch=stopping.best_epoch,
        stopped_early=stopping.should_stop,
        best_validation_mse=stopping.best_value,
        restored_validation_mse=_mse(restored, reference),
        history=history,
        state=best_state,
        normalized_prediction=restored,
    )


@dataclass(frozen=True)
class RepresentationScore:
    """Seed-aggregated validation score for one input representation."""

    representation: InputRepresentation
    seed_scores: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.seed_scores))

    @property
    def spread(self) -> float:
        """Return the sample standard deviation (ddof=1) of the seed scores."""
        return float(np.std(self.seed_scores, ddof=1)) if len(self.seed_scores) > 1 else 0.0


def rank_representations(
    scores: Sequence[RepresentationScore], *, tolerance: float
) -> list[RepresentationScore]:
    """Return representations best first under the frozen aggregate and tie rules.

    The highest seed-mean score wins. Every representation within ``tolerance`` of
    it ties and is ordered by lower spread, then shorter history, then larger bin
    width; the rest follow by descending mean.
    """
    if not scores:
        raise GruError("At least one representation is required for ranking.")
    if any(not np.all(np.isfinite(score.seed_scores)) for score in scores):
        raise GruError("Every seed score must be finite before ranking.")
    best = max(score.mean for score in scores)
    tied = [score for score in scores if best - score.mean <= tolerance]
    rest = [score for score in scores if best - score.mean > tolerance]
    tied.sort(
        key=lambda score: (
            score.spread,
            score.representation.history_ms,
            -score.representation.bin_ms,
        )
    )
    rest.sort(key=lambda score: -score.mean)
    return tied + rest


def save_checkpoint(
    path: Path, state: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> None:
    """Write a restored model state with the metadata needed to rebuild it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": _CHECKPOINT_FORMAT,
            "metadata": dict(metadata),
            "state_dict": {name: tensor.detach().cpu() for name, tensor in state.items()},
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[GruDecoder, dict[str, Any]]:
    """Rebuild a decoder from a checkpoint without executing pickled code.

    Metadata without ``num_layers`` or ``dropout`` describes a one-layer,
    dropout-free decoder.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _CHECKPOINT_FORMAT:
        raise GruError(f"{path} is not a supported GRU checkpoint.")
    metadata = dict(payload["metadata"])
    model = GruDecoder(
        int(metadata["n_units"]),
        int(metadata["hidden_size"]),
        int(metadata.get("num_layers", 1)),
        float(metadata.get("dropout", 0.0)),
    )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise GruError(f"{path} does not match the architecture in its metadata.") from error
    return model, metadata


def _mse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction, dtype=np.float64) - reference)))
