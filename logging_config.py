"""
logging configuration for babbage pipeline.
"""

import logging
import os


def setup_logging(level=None):
    """configure logging for babbage pipeline."""
    if level is None:
        level = os.environ.get('BABBAGE_LOG_LEVEL', 'INFO')

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_logger(name: str) -> logging.Logger:
    """get logger for module."""
    return logging.getLogger(name)
