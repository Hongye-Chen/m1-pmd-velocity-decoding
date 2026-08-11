"""Logging setup shared by project scripts and library code."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["LOGGER_NAME", "configure_logging", "get_logger"]

LOGGER_NAME = "neural_decoding"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MANAGED_HANDLER_ATTR = "_neural_decoding_managed"


def configure_logging(
    level: int | str = "INFO",
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Install the project stream handler, and optionally a file handler.

    Handlers installed by earlier calls are removed first, so repeated calls do not
    duplicate output. Handlers owned by other code, such as a test runner, are left
    untouched.

    Args:
        level: Level name or numeric level applied to the root logger.
        log_file: Optional file that receives the same formatted records. Parent
            directories are created when needed.

    Returns:
        The project logger.
    """
    resolved_level = _resolve_level(level)

    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _MANAGED_HANDLER_ATTR, False)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, _MANAGED_HANDLER_ATTR, True)
    root.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _MANAGED_HANDLER_ATTR, True)
        root.addHandler(file_handler)

    root.setLevel(resolved_level)
    return get_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger, or a named child of it."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def _resolve_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    try:
        return logging.getLevelNamesMapping()[str(level).upper()]
    except KeyError as error:
        raise ValueError(f"Unknown logging level: {level!r}") from error
