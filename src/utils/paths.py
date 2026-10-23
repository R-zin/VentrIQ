from __future__ import annotations

import os
from pathlib import Path

# Project root: resolves to the repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Standard data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ECHONEXT_RAW_DIR = RAW_DATA_DIR / "echonext"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def get_data_dir() -> Path:
    """Return the base data directory, creating it if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_raw_echonext_dir() -> Path:
    """Return the raw EchoNext data directory."""
    return ECHONEXT_RAW_DIR


def get_processed_dir() -> Path:
    """Return the processed data directory, creating it if needed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DATA_DIR


def get_reports_dir() -> Path:
    """Return the reports directory, creating it if needed."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def get_figures_dir() -> Path:
    """Return the figures directory inside reports, creating it if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def get_synthetic_dir() -> Path:
    """Return the synthetic data directory, creating it if needed."""
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return SYNTHETIC_DATA_DIR

