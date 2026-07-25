from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from scan.config import settings as default_settings
from scan.project_logger import get_logger

if TYPE_CHECKING:
    from scan.config import Settings

logger = get_logger(__name__)

__all__ = ["OpenAIWrapper", "build_llm"]


class OpenAIWrapper:
    """Wrapper for OpenAI LLM interactions using LangChain."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings if settings is not None else default_settings
        self.model_name = model_name
        self.max_tokens = max_tokens if max_tokens is not None else self.settings.MAX_TOKENS
        self.llm = self.initialize_llm()

    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the LangChain OpenAI LLM."""
        try:
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=0,
                max_tokens=self.max_tokens,
                # crewai does not use this object: it rebuilds its own LLM from a handful of
                # attributes and reads the key from the environment (langchain stores the key as
                # `openai_api_key`, so crewai's `getattr(llm, "api_key")` is always None).
                # scan.config.apply_environment exports OPENAI_API_KEY so that path works; setting
                # it here keeps the wrapper usable on its own.
                api_key=(
                    SecretStr(self.settings.OPENAI_API_KEY)
                    if self.settings.OPENAI_API_KEY
                    else None
                ),
                verbose=False,
            )
            logger.info(f"LLM initialized successfully with model: {self.model_name}")
            return llm
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to initialize LLM with model {self.model_name}: {e}")
            raise


def build_llm(model_name: str, settings: Settings | None = None) -> ChatOpenAI:
    """Build the language model object handed to a crewai Agent.

    The single seam where SCAN decides what kind of LLM object crewai receives.
    """
    return OpenAIWrapper(model_name=model_name, settings=settings).llm
