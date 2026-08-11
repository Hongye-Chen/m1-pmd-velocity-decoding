"""Smoke checks for the package, logging, and seeding."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import neural_decoding
from neural_decoding.logging_utils import _MANAGED_HANDLER_ATTR as MANAGED
from neural_decoding.logging_utils import LOGGER_NAME, configure_logging
from neural_decoding.seeding import new_rng, set_global_seed


def test_project_package_imports_and_reports_a_version() -> None:
    assert isinstance(neural_decoding.__version__, str)
    assert neural_decoding.__version__


@pytest.fixture
def restored_root_logger() -> Iterator[logging.Logger]:
    """Undo any change a test makes to the root logger."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield root
    for handler in list(root.handlers):
        if handler not in original_handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in original_handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(original_level)


def _managed_handlers(root: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in root.handlers if getattr(handler, MANAGED, False)]


def test_repeated_configuration_keeps_one_stream_handler(
    restored_root_logger: logging.Logger,
) -> None:
    configure_logging()
    assert len(_managed_handlers(restored_root_logger)) == 1
    assert restored_root_logger.level == logging.INFO

    configure_logging(level="DEBUG")
    assert len(_managed_handlers(restored_root_logger)) == 1
    assert restored_root_logger.level == logging.DEBUG


def test_configuration_leaves_foreign_handlers_alone(
    restored_root_logger: logging.Logger,
) -> None:
    foreign = logging.NullHandler()
    restored_root_logger.addHandler(foreign)

    configure_logging()

    assert foreign in restored_root_logger.handlers


def test_log_file_receives_records_and_is_created(
    tmp_path: Path, restored_root_logger: logging.Logger
) -> None:
    log_file = tmp_path / "nested" / "run.log"

    logger = configure_logging(level="INFO", log_file=log_file)
    logger.info("structure check")

    assert log_file.is_file()
    contents = log_file.read_text(encoding="utf-8")
    assert "structure check" in contents
    assert LOGGER_NAME in contents


def test_unknown_logging_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown logging level"):
        configure_logging(level="not-a-level")


def test_global_seed_makes_python_and_numpy_reproducible() -> None:
    assert set_global_seed(123) == 123
    first = (random.random(), np.random.rand(3).tolist())

    set_global_seed(123)
    second = (random.random(), np.random.rand(3).tolist())

    assert first == second


def test_new_rng_is_reproducible_and_independent_of_global_state() -> None:
    set_global_seed(7)
    expected = new_rng(99).normal(size=4).tolist()

    set_global_seed(8)
    np.random.rand(10)
    actual = new_rng(99).normal(size=4).tolist()

    assert actual == expected
