"""Grid search over GRU depth, hidden size, dropout, and learning rate.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neural_decoding.dataset import compute_sha256
from neural_decoding.gru import GruError, GruProtocol, validate_architecture

__all__ = [
    "ArchitectureScore",
    "ArchitectureSearch",
    "GruArchitecture",
    "RunPlan",
    "baseline_architecture",
    "load_architecture_search",
    "plan_runs",
    "rank_architectures",
    "text_matches_sha256",
    "verify_artifact_hashes",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class GruArchitecture:
    """One searched configuration of the decoder and its learning rate."""

    num_layers: int
    hidden_size: int
    dropout: float
    learning_rate: float

    def __post_init__(self) -> None:
        validate_architecture(self.num_layers, self.dropout)
        if self.hidden_size <= 0 or not self.learning_rate > 0:
            raise GruError(f"{self} needs a positive hidden size and learning rate.")

    @property
    def label(self) -> str:
        layers = "1 layer" if self.num_layers == 1 else f"{self.num_layers} layers"
        return (
            f"{layers} / hidden {self.hidden_size} / dropout {self.dropout:g} "
            f"/ lr {self.learning_rate:g}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True)
class ArchitectureSearch:
    """The tracked architecture-search protocol, parsed strictly."""

    input_protocol: Path
    input_protocol_sha256: str
    input_artifacts: Path
    input_summary_sha256: str
    hidden_sizes: tuple[int, ...]
    learning_rates: tuple[float, ...]
    dropout_by_num_layers: tuple[tuple[int, tuple[float, ...]], ...]
    expected_configurations: int
    expected_new_runs: int
    expected_reused_runs: int
    tie_tolerance: float

    @property
    def architectures(self) -> tuple[GruArchitecture, ...]:
        """Return every configuration in declared order: layers, dropout, hidden size, rate."""
        return tuple(
            GruArchitecture(num_layers, hidden_size, dropout, learning_rate)
            for num_layers, dropouts in self.dropout_by_num_layers
            for dropout in dropouts
            for hidden_size in self.hidden_sizes
            for learning_rate in self.learning_rates
        )


def load_architecture_search(path: Path | str) -> ArchitectureSearch:
    """Read the tracked search protocol, refusing unknown, missing, or inapplicable settings."""
    source = Path(path)
    if not source.is_file():
        raise GruError(f"The GRU architecture-search protocol is missing at {source}.")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    sections = {
        "input_selection": {"protocol", "protocol_sha256", "artifacts", "summary_sha256"},
        "search": {"hidden_sizes", "learning_rates", "dropout_by_num_layers"},
        "runs": {"configurations", "new_runs", "reused_runs"},
        "selection": {"tie_tolerance"},
    }
    if not isinstance(data, dict) or set(data) != set(sections):
        raise GruError(f"{source} must define exactly the sections {sorted(sections)}.")
    for name, keys in sections.items():
        if not isinstance(data[name], dict) or set(data[name]) != keys:
            raise GruError(f"Section {name!r} of {source} must define exactly {sorted(keys)}.")

    inputs, grid, runs = data["input_selection"], data["search"], data["runs"]
    hashes = (str(inputs["protocol_sha256"]), str(inputs["summary_sha256"]))
    if not all(_SHA256.fullmatch(value) for value in hashes):
        raise GruError("Frozen input-selection hashes must be lowercase SHA-256 digests.")

    hidden_sizes = tuple(int(value) for value in grid["hidden_sizes"])
    learning_rates = tuple(float(value) for value in grid["learning_rates"])
    dropout_map = grid["dropout_by_num_layers"]
    if not isinstance(dropout_map, dict) or not dropout_map:
        raise GruError("dropout_by_num_layers must map each layer count to its dropout values.")
    dropout_by_num_layers = tuple(
        (int(layers), tuple(float(value) for value in dropout_map[layers]))
        for layers in sorted(dropout_map)
    )
    distinct = [hidden_sizes, learning_rates] + [values for _, values in dropout_by_num_layers]
    if any(not values or len(set(values)) != len(values) for values in distinct):
        raise GruError("Every searched value list must be non-empty and free of duplicates.")

    search = ArchitectureSearch(
        input_protocol=Path(inputs["protocol"]),
        input_protocol_sha256=hashes[0],
        input_artifacts=Path(inputs["artifacts"]),
        input_summary_sha256=hashes[1],
        hidden_sizes=hidden_sizes,
        learning_rates=learning_rates,
        dropout_by_num_layers=dropout_by_num_layers,
        expected_configurations=int(runs["configurations"]),
        expected_new_runs=int(runs["new_runs"]),
        expected_reused_runs=int(runs["reused_runs"]),
        tie_tolerance=float(data["selection"]["tie_tolerance"]),
    )
    if len(search.architectures) != search.expected_configurations:
        raise GruError(
            f"The grid defines {len(search.architectures)} configurations, not the declared "
            f"{search.expected_configurations}."
        )
    return search


def baseline_architecture(protocol: GruProtocol) -> GruArchitecture:
    """Return the one-layer, dropout-free architecture every input-selection run used."""
    return GruArchitecture(1, protocol.hidden_size, 0.0, protocol.learning_rate)


@dataclass(frozen=True)
class RunPlan:
    """The configuration/seed runs to train and the frozen runs reused instead."""

    new: tuple[tuple[GruArchitecture, int], ...]
    reused: tuple[tuple[GruArchitecture, int], ...]


def plan_runs(
    search: ArchitectureSearch, seeds: Sequence[int], reused: GruArchitecture
) -> RunPlan:
    """Plan every configuration with every seed, training all but the reused one."""
    if reused not in search.architectures:
        raise GruError(f"The reused architecture ({reused.label}) is not in the search grid.")
    plan = RunPlan(
        new=tuple(
            (architecture, int(seed))
            for architecture in search.architectures
            if architecture != reused
            for seed in seeds
        ),
        reused=tuple((reused, int(seed)) for seed in seeds),
    )
    counts = (len(plan.new), len(plan.reused))
    if counts != (search.expected_new_runs, search.expected_reused_runs):
        raise GruError(
            f"The plan has {counts[0]} new and {counts[1]} reused runs, not the declared "
            f"{search.expected_new_runs} and {search.expected_reused_runs}."
        )
    return plan


@dataclass(frozen=True)
class ArchitectureScore:
    """Seed-aggregated validation score for one configuration."""

    architecture: GruArchitecture
    seed_scores: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.seed_scores))

    @property
    def spread(self) -> float:
        """Return the sample standard deviation (ddof=1) of the seed scores."""
        return float(np.std(self.seed_scores, ddof=1)) if len(self.seed_scores) > 1 else 0.0


def rank_architectures(
    scores: Sequence[ArchitectureScore], *, tolerance: float
) -> list[ArchitectureScore]:
    """Return configurations best first under the frozen aggregate and tie rules.

    The highest seed-mean score wins. Every configuration within ``tolerance`` of
    it ties and is ordered by lower spread, then fewer layers, smaller hidden size,
    smaller dropout, and smaller learning rate; the rest follow by descending mean.
    """
    if not scores:
        raise GruError("At least one configuration is required for ranking.")
    if any(not np.all(np.isfinite(score.seed_scores)) for score in scores):
        raise GruError("Every seed score must be finite before ranking.")
    best = max(score.mean for score in scores)
    tied = [score for score in scores if best - score.mean <= tolerance]
    rest = [score for score in scores if best - score.mean > tolerance]
    tied.sort(
        key=lambda score: (
            score.spread,
            score.architecture.num_layers,
            score.architecture.hidden_size,
            score.architecture.dropout,
            score.architecture.learning_rate,
        )
    )
    rest.sort(key=lambda score: -score.mean)
    return tied + rest


def text_matches_sha256(path: Path, digest: str) -> bool:
    """Return whether a text file hashes to ``digest`` with LF or with CRLF line endings.

    Git may check a tracked text file out with either line ending, which changes
    its bytes but not the content a protocol hash pins.
    """
    data = Path(path).read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    variants = (data, lf, lf.replace(b"\n", b"\r\n"))
    return digest in {hashlib.sha256(variant).hexdigest() for variant in variants}


def verify_artifact_hashes(directory: Path, expected: Mapping[str, str]) -> list[str]:
    """Return the recorded file names that are missing or no longer match their hash."""
    mismatched = []
    for name, digest in sorted(expected.items()):
        path = Path(directory) / name
        if Path(name).name != name or not path.is_file() or compute_sha256(path) != digest:
            mismatched.append(name)
    return mismatched
