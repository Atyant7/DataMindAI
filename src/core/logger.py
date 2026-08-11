"""Central logging configuration for DataMindAI."""

import logging
from logging.handlers import RotatingFileHandler

from src.core.config import LOGS_DIR


_LOGGER_NAME = "datamindai"
_CONFIGURED = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured DataMindAI logger."""
    global _CONFIGURED

    logger = logging.getLogger(name or _LOGGER_NAME)

    if not _CONFIGURED:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )

            file_handler = RotatingFileHandler(
                LOGS_DIR / "datamindai.log",
                maxBytes=2_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        _CONFIGURED = True

    if name and name != _LOGGER_NAME:
        child_logger = logging.getLogger(name)
        child_logger.setLevel(logging.INFO)
        child_logger.propagate = True
        return child_logger

    return logging.getLogger(_LOGGER_NAME)