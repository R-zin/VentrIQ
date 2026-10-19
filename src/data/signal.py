"""src/data/signal.py - Legacy signal preprocessing module.

This module is maintained for backward compatibility.
Please use `src.preprocessing.signal.ECGSignalPreprocessor` for the updated,
configurable, zero-phase filtering and quality control pipeline.
"""

from src.preprocessing.signal import signal_preprocessing

__all__ = ["signal_preprocessing"]
