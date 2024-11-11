from __future__ import annotations

import logging
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: str
    DLPFC_MODEL: str = "gpt-4"
    VMPFC_MODEL: str = "gpt-4"
    OFC_MODEL: str = "gpt-4"
    ACC_MODEL: str = "gpt-4"
    MPFC_MODEL: str = "gpt-4"
    # Default logging to warning because crewai has a logger that is extrememly noisy and I haven't
    # found a way to turn it off.
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def COMPUTED_LOG_LEVEL(
        self,
    ) -> int:
        if self.LOG_LEVEL == "DEBUG":
            return logging.DEBUG
        if self.LOG_LEVEL == "INFO":
            return logging.INFO
        if self.LOG_LEVEL == "WARNING":
            return logging.WARNING
        if self.LOG_LEVEL == "WARNING":
            return logging.WARNING
        if self.LOG_LEVEL == "ERROR":
            return logging.ERROR
        return logging.CRITICAL


settings = Settings()  # type: ignore
