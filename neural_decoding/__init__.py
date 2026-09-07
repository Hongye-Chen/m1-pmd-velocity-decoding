"""Offline decoding of two-dimensional cursor velocity from M1/PMd spiking activity.

This package holds the reusable building blocks shared by the project scripts and
tests: configuration, logging, seeding, session loading, trial splits, feature
construction, lag selection, metrics, and decoders.
"""

from neural_decoding.config import (
    DEFAULT_SEED,
    ConfigError,
    ProjectConfig,
    ProjectPaths,
    RunSettings,
    load_config,
    project_root,
)
from neural_decoding.logging_utils import configure_logging, get_logger
from neural_decoding.seeding import new_rng, set_global_seed

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "DEFAULT_SEED",
    "ProjectConfig",
    "ProjectPaths",
    "RunSettings",
    "configure_logging",
    "get_logger",
    "load_config",
    "new_rng",
    "project_root",
    "set_global_seed",
    "__version__",
]
