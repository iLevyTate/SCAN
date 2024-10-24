# scan/openai_llm.py

import os
import logging
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.chat_models import ChatOpenAI

# Configure logging
logger = logging.getLogger(__name__)

class OpenAIWrapper:
    """Wrapper for OpenAI LLM interactions using LangChain."""

    def __init__(self, api_key: str, model_name: str = "gpt-4", max_tokens: int = 1000):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        logger.info(f"Initializing LLM with model: {self.model_name}")

        # Set up callback manager for token usage tracking
        self.callback_handler = OpenAICallbackHandler()
        self.callback_manager = CallbackManager([self.callback_handler])

        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0,
                max_tokens=self.max_tokens,
                max_retries=3,
                callback_manager=self.callback_manager,
                verbose=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM with model {self.model_name}: {e}")
            raise

    def get_token_usage(self):
        """Retrieve the token usage from the callback handler."""
        return {
            "total_tokens": self.callback_handler.total_tokens,
            "prompt_tokens": self.callback_handler.prompt_tokens,
            "completion_tokens": self.callback_handler.completion_tokens,
        }
