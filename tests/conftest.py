import logging

import pytest

from scan.config import settings
from scan.project_logger import NOISY_LOGGERS

TOPIC = "Some topic"


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    """Give every test deterministic credentials.

    monkeypatch (rather than plain assignment) so a test that changes a setting cannot leak
    that change into later tests, and so a developer's real .env cannot influence the suite.
    """
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "my_key")
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", "search_key")


@pytest.fixture(autouse=True)
def restore_logging():
    """Undo configure_logging()'s global mutations after every test.

    configure_logging() replaces the "scan" logger's handlers and sets propagate=False. Left in
    place that silently breaks pytest's caplog -- which reads records off the root logger -- for
    every test that happens to run afterwards, giving order-dependent failures.
    """
    touched = [logging.getLogger("scan"), *(logging.getLogger(n) for n in NOISY_LOGGERS)]
    saved = [(log, log.handlers[:], log.level, log.propagate) for log in touched]
    try:
        yield
    finally:
        for log, handlers, level, propagate in saved:
            log.handlers[:] = handlers
            log.setLevel(level)
            log.propagate = propagate
