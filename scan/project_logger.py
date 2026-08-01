from __future__ import annotations

import logging

from scan.config import settings

#: Third-party loggers that are chatty at INFO/DEBUG and drown out SCAN's own output.
NOISY_LOGGERS = ("httpx", "httpcore", "LiteLLM", "litellm", "openai", "opentelemetry", "backoff")

_PACKAGE_LOGGER = "scan"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module, so log lines identify where they came from."""
    return logging.getLogger(name)


def configure_logging(level: int | None = None) -> None:
    """Install SCAN's logging configuration.

    Called from the CLI entry point rather than at import time: ``basicConfig`` mutates the
    *root* logger, so doing it on import both hijacked logging for any host application that
    imported ``scan`` and applied SCAN's level to every third-party library.
    """
    level = settings.COMPUTED_LOG_LEVEL if level is None else level

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    package_logger = logging.getLogger(_PACKAGE_LOGGER)
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False

    # Raising SCAN's own verbosity should not also turn on every dependency's INFO logs.
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


logger = logging.getLogger(_PACKAGE_LOGGER)
