"""Loguru configuration for backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
LOG_DIR = DATA_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"


def setup_logging() -> None:
    """Configure loguru: console + file."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        LOG_FILE,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
