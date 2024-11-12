from __future__ import annotations

from langchain.callbacks.manager import CallbackManager
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_openai import ChatOpenAI  # Updated import

from scan.project_logger import logger


class OpenAIWrapper:
    """Wrapper for OpenAI LLM interactions using LangChain."""

    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 1000) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.callback_handler = OpenAICallbackHandler()
        self.callback_manager = CallbackManager([self.callback_handler])
        self.llm = self.initialize_llm()

    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the LangChain OpenAI LLM."""
        try:
            llm = ChatOpenAI(
                name=self.model_name,
                temperature=0,
                max_tokens=self.max_tokens,
                callback_manager=self.callback_manager,
                verbose=False,
            )
            logger.info(f"LLM initialized successfully with model: {self.model_name}")
            return llm
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to initialize LLM with model {self.model_name}: {e}")
            raise

    def get_token_usage(self) -> dict[str, int]:
        """Retrieve the token usage from the callback handler."""
        return {
            "total_tokens": self.callback_handler.total_tokens,
            "prompt_tokens": self.callback_handler.prompt_tokens,
            "completion_tokens": self.callback_handler.completion_tokens,
        }
