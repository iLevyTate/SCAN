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


def test_missing_openai_key_does_not_raise_at_construction():
    # Regression: OPENAI_API_KEY used to be required, so constructing Settings without
    # it raised a ValidationError at import time before main() could handle it nicely.
    settings = Settings(OPENAI_API_KEY=None)

    assert settings.OPENAI_API_KEY is None
