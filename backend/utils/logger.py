"""
Structured logging with loguru.
"""

import sys
from loguru import logger
from config import settings


def setup_logger() -> None:
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=fmt, level="DEBUG" if settings.debug else "INFO", colorize=True)
    logger.add(
        "logs/isl_{time:YYYY-MM-DD}.log",
        format=fmt,
        level="DEBUG",
        rotation="1 day",
        retention="14 days",
        compression="gz",
    )
