"""Structured logging configuration for FaceTwin."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "facetwin",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger instance.
        level: Logging severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to append log entries.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if logger was already initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
