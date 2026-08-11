"""Configuration loading for the neural decoding project.

Settings are merged from three sources, in increasing order of precedence:

1. ``configs/default.yaml`` — tracked defaults that contain no machine-specific value;
2. ``configs/local.yaml`` — untracked overrides holding the local dataset path;
3. the ``NEURAL_DECODING_NWB_FILE`` environment variable, then explicit overrides
   passed to :func:`load_config`.

The NWB recording lives outside the repository, so its path is never stored in a
tracked file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_SEED",
    "LOCAL_CONFIG_PATH",
    "NWB_PATH_ENV_VAR",
    "ConfigError",
    "ProjectConfig",
    "ProjectPaths",
    "RunSettings",
    "load_config",
    "project_root",
]

NWB_PATH_ENV_VAR = "NEURAL_DECODING_NWB_FILE"
DEFAULT_SEED = 42

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG_PATH = _PROJECT_ROOT / "configs" / "local.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed, or points at absent data."""


def project_root() -> Path:
    """Return the repository root directory."""
    return _PROJECT_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the project."""

    nwb_file: Path | None
    results_dir: Path
    reports_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class RunSettings:
    """Run-wide execution settings."""

    seed: int
    log_level: str


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration."""

    paths: ProjectPaths
    run: RunSettings
    root: Path
    sources: tuple[Path, ...]

    def require_nwb_file(self) -> Path:
        """Return the configured NWB path, raising if it is unset or absent."""
        path = self.paths.nwb_file
        if path is None:
            raise ConfigError(
                "No NWB file is configured. Set paths.nwb_file in configs/local.yaml "
                f"or set the {NWB_PATH_ENV_VAR} environment variable. See README.md."
            )
        if not path.is_file():
            raise ConfigError(f"The configured NWB file does not exist: {path}")
        return path

    def ensure_directories(self) -> None:
        """Create the output directories referenced by this configuration."""
        for directory in (self.paths.results_dir, self.paths.reports_dir, self.paths.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


def load_config(
    config_path: Path | str | None = None,
    local_config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    use_env: bool = True,
    root: Path | str | None = None,
) -> ProjectConfig:
    """Load and resolve the project configuration.

    Args:
        config_path: Defaults file to read; ``configs/default.yaml`` when omitted.
        local_config_path: Untracked overrides file; ``configs/local.yaml`` when
            omitted. A missing file is ignored.
        overrides: Nested mapping applied last, mirroring the YAML structure.
        use_env: Whether ``NEURAL_DECODING_NWB_FILE`` may override the file path.
        root: Directory that relative paths resolve against; the repository root
            when omitted.
    """
    root_dir = Path(root).resolve() if root is not None else _PROJECT_ROOT

    defaults_file = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not defaults_file.is_file():
        raise ConfigError(f"Missing configuration file: {defaults_file}")

    local_file = (
        Path(local_config_path) if local_config_path is not None else LOCAL_CONFIG_PATH
    )

    sources = [defaults_file]
    data = _read_yaml(defaults_file)

    local_data = _read_yaml(local_file) if local_file.is_file() else {}
    if local_data:
        data = _merge(data, local_data)
        sources.append(local_file)

    if use_env:
        env_path = os.environ.get(NWB_PATH_ENV_VAR)
        if env_path:
            data = _merge(data, {"paths": {"nwb_file": env_path}})

    if overrides:
        data = _merge(data, overrides)

    paths_section = data.get("paths") or {}
    run_section = data.get("run") or {}

    nwb_value = paths_section.get("nwb_file")
    paths = ProjectPaths(
        nwb_file=_resolve_path(nwb_value, root_dir) if nwb_value else None,
        results_dir=_resolve_path(paths_section.get("results_dir", "results"), root_dir),
        reports_dir=_resolve_path(paths_section.get("reports_dir", "reports"), root_dir),
        log_dir=_resolve_path(paths_section.get("log_dir", "logs"), root_dir),
    )

    run = RunSettings(
        seed=_coerce_seed(run_section.get("seed", DEFAULT_SEED)),
        log_level=str(run_section.get("log_level", "INFO")).upper(),
    )

    return ProjectConfig(paths=paths, run=run, root=root_dir, sources=tuple(sources))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: Path | str, root: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else root / path


def _coerce_seed(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"run.seed must be an integer, got {value!r}") from error
