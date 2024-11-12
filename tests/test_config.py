import logging

import pytest

from scan.config import Settings


@pytest.mark.parametrize(
    "log_level, expected",
    (
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ),
)
def test_log_level(log_level, expected):
    settings = Settings(OPENAI_API_KEY="somekey", LOG_LEVEL=log_level)

    assert settings.COMPUTED_LOG_LEVEL == expected
