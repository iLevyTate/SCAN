from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from scan.config import settings
from scan.project_logger import get_logger

logger = get_logger(__name__)


class OpenAIWrapper:
    """Wrapper for OpenAI LLM interactions using LangChain."""

    def __init__(self, model_name: str, max_tokens: int | None = None) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS
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
                api_key=SecretStr(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None,
                verbose=False,
            )
            logger.info(f"LLM initialized successfully with model: {self.model_name}")
            return llm
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to initialize LLM with model {self.model_name}: {e}")
            raise
