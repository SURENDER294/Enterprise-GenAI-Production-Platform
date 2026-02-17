"""
logger.py

Centralized logging configuration using loguru.
Provides a consistent, structured logger for the entire project.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded successfully")
"""

import sys
import os
from pathlib import Path
from loguru import logger

# -- Remove default loguru handler so we configure everything ourselves
logger.remove()

# -- Determine log level from env var, defaulting to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# -- Console handler: colored, human-readable
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    backtrace=True,
    diagnose=True,
)

# -- File handler: JSON-style for log aggregation tools
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    level=LOG_LEVEL,
    rotation="1 day",
    retention="7 days",
    compression="zip",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    ),
    backtrace=True,
    diagnose=False,  # Don't expose variable values in production logs
)


def get_logger(name: str):
    """
    Return a logger instance bound to the given module name.

    Args:
        name: Usually __name__ from the calling module.

    Returns:
        A loguru logger bound with the module context.
    """
    return logger.bind(name=name)
