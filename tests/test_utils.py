"""tests/test_utils.py - Unit tests for utility modules (paths and logging)."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    ECHONEXT_RAW_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SYNTHETIC_DATA_DIR,
    get_data_dir,
    get_figures_dir,
    get_processed_dir,
    get_project_root,
    get_raw_echonext_dir,
    get_reports_dir,
    get_synthetic_dir,
)


def test_project_paths_exist_and_consistent() -> None:
    """Verify all standard top-level project paths resolve and are consistent."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()
    assert (PROJECT_ROOT / "tests").exists()
    assert CONFIGS_DIR.exists()
    assert DATA_DIR.is_absolute()
    assert REPORTS_DIR.is_absolute()
    assert RAW_DATA_DIR.parent == DATA_DIR
    assert ECHONEXT_RAW_DIR.parent == RAW_DATA_DIR
    assert PROCESSED_DATA_DIR.parent == DATA_DIR
    assert SYNTHETIC_DATA_DIR.parent == DATA_DIR
    assert FIGURES_DIR.parent == REPORTS_DIR


def test_helper_path_resolvers() -> None:
    """Verify helper functions returning standard subdirectories."""
    assert get_project_root() == PROJECT_ROOT
    assert isinstance(get_data_dir(), Path)
    assert get_raw_echonext_dir() == ECHONEXT_RAW_DIR
    assert isinstance(get_processed_dir(), Path)
    assert isinstance(get_reports_dir(), Path)
    assert isinstance(get_figures_dir(), Path)
    assert isinstance(get_synthetic_dir(), Path)


def test_get_logger_configuration() -> None:
    """Verify logger setup produces valid configured Logger instance."""
    logger = get_logger("test_custom_logger", level=logging.DEBUG)
    assert logger.name == "test_custom_logger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1

    # Re-retrieving the same logger does not duplicate handlers
    num_handlers = len(logger.handlers)
    logger_cached = get_logger("test_custom_logger")
    assert len(logger_cached.handlers) == num_handlers
