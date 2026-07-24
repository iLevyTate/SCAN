import logging
import os

import pytest

from scan.config import Settings, apply_environment


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


def test_log_level_is_case_insensitive():
    # Deliberately outside the Literal: the `before` validator accepts any string, which is
    # what an env var actually supplies.
    assert Settings(LOG_LEVEL="debug").LOG_LEVEL == "DEBUG"  # type: ignore[arg-type]


def test_invalid_log_level_warns_and_falls_back():
    # Regression: an unrecognised LOG_LEVEL raised a ValidationError from the module-level
    # `settings = Settings()`, i.e. an unhandled traceback at import time.
    with pytest.warns(UserWarning, match="Unrecognised LOG_LEVEL"):
        settings = Settings(LOG_LEVEL="verbose")  # type: ignore[arg-type]

    assert settings.LOG_LEVEL == "WARNING"


def test_missing_openai_key_does_not_raise_at_construction():
    # Regression: OPENAI_API_KEY used to be required, so constructing Settings without
    # it raised a ValidationError at import time before main() could handle it nicely.
    settings = Settings(OPENAI_API_KEY=None)

    assert settings.OPENAI_API_KEY is None


def test_unrelated_env_file_keys_are_ignored(tmp_path, monkeypatch):
    # Regression: pydantic-settings defaults to extra="forbid", so one unrelated key in a
    # developer's .env crashed both the CLI and pytest collection at import time.
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=abc\nSOME_UNRELATED_TOOL_TOKEN=xyz\n")
    monkeypatch.chdir(tmp_path)
    # Real environment variables win over the .env file, and litellm calls load_dotenv() at
    # import, so clear it to be sure we are reading the file under test.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings()

    assert settings.OPENAI_API_KEY == "abc"


def test_serper_api_key_is_accepted_as_a_legacy_alias(monkeypatch):
    # The search tool talks to serpapi.com, so the setting is SERPAPI_API_KEY now; existing
    # .env files using the old SERPER_API_KEY name must keep working.
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "legacy_value")

    assert Settings().SERPAPI_API_KEY == "legacy_value"


def test_serpapi_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "legacy_value")
    monkeypatch.setenv("SERPAPI_API_KEY", "current_value")

    assert Settings().SERPAPI_API_KEY == "current_value"


def test_apply_environment_exports_keys(monkeypatch):
    # Regression: crewai/litellm read credentials from os.environ, and the ChatOpenAI object
    # SCAN builds never carries the key across (crewai reads `api_key`, langchain stores
    # `openai_api_key`). Without this export the run only worked because litellm happens to
    # call load_dotenv() at import.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    apply_environment(Settings(OPENAI_API_KEY="k", SERPAPI_API_KEY="s"))

    assert os.environ["OPENAI_API_KEY"] == "k"
    assert os.environ["SERPAPI_API_KEY"] == "s"
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_apply_environment_respects_telemetry_opt_in(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    apply_environment(Settings(DISABLE_TELEMETRY=False))

    assert "OTEL_SDK_DISABLED" not in os.environ


def test_apply_environment_does_not_clobber_existing_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "already_set")

    apply_environment(Settings(OPENAI_API_KEY="from_settings"))

    assert os.environ["OPENAI_API_KEY"] == "already_set"
