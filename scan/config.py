from __future__ import annotations

import logging
import os
import warnings
from typing import Literal

from pydantic import AliasChoices, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class Settings(BaseSettings):
    # extra="ignore" matters: pydantic-settings defaults to "forbid", which made a single
    # unrelated key in a developer's .env raise at import time and take down both the CLI
    # and pytest collection.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Optional at construction so importing this module never crashes when the key
    # is absent; presence is validated explicitly at runtime (see scan.main.main).
    OPENAI_API_KEY: str | None = None
    # The search tool is langchain's SerpAPIWrapper, which talks to serpapi.com -- not to
    # serper.dev. SERPER_API_KEY is accepted as a legacy alias so existing .env files keep
    # working, but the correct name is SERPAPI_API_KEY.
    SERPAPI_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SERPAPI_API_KEY", "SERPER_API_KEY"),
    )
    DLPFC_MODEL: str = "gpt-4o"
    VMPFC_MODEL: str = "gpt-4o"
    OFC_MODEL: str = "gpt-4o"
    ACC_MODEL: str = "gpt-4o"
    MPFC_MODEL: str = "gpt-4o"
    MAX_TOKENS: int = 4000
    # crewai ships opentelemetry tracing to telemetry.crewai.com on every run. Disabled by
    # default; set DISABLE_TELEMETRY=false to opt back in.
    DISABLE_TELEMETRY: bool = True
    LOG_LEVEL: LogLevelName = "WARNING"

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Accept any capitalisation, and fall back to WARNING rather than crashing at import."""
        if not isinstance(value, str):
            return value
        candidate = value.strip().upper()
        if candidate in _LOG_LEVELS:
            return candidate
        warnings.warn(
            f"Unrecognised LOG_LEVEL {value!r}; falling back to WARNING. "
            f"Valid values are {', '.join(_LOG_LEVELS)}.",
            stacklevel=2,
        )
        return "WARNING"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def COMPUTED_LOG_LEVEL(
        self,
    ) -> int:
        return _LOG_LEVELS[self.LOG_LEVEL]


settings = Settings()


def apply_environment(config: Settings | None = None) -> None:
    """Publish settings that third-party libraries only read from the process environment.

    crewai does not use the ``ChatOpenAI`` object it is handed: it rebuilds its own LLM and
    reads ``api_key`` off the langchain object, which is ``None`` because langchain stores it
    as ``openai_api_key``. litellm therefore falls back to ``os.environ``. Exporting the key
    here makes that path deterministic instead of relying on litellm's import-time
    ``load_dotenv()`` happening to find the project's .env.
    """
    config = config if config is not None else settings
    if config.OPENAI_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", config.OPENAI_API_KEY)
    if config.SERPAPI_API_KEY:
        os.environ.setdefault("SERPAPI_API_KEY", config.SERPAPI_API_KEY)
    if config.DISABLE_TELEMETRY:
        # Honoured by opentelemetry's TracerProvider, which is what crewai builds. Must be set
        # before the first Crew is constructed.
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")
