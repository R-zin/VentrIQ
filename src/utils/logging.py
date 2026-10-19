from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str = "ventriq", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standard logger for VentriQ modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
