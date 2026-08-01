"""Construction of the language model handed to each crewai agent.

crewai does not use a langchain ``ChatOpenAI`` object: ``Agent.post_init_setup`` scrapes nine
attributes off it and rebuilds its own ``crewai.llm.LLM``. Five of those nine -- including
``timeout`` and ``api_key`` -- are always ``None`` on a ``ChatOpenAI``, because langchain stores
them under different names (``request_timeout``, ``openai_api_key``). Everything else about the
langchain client was constructed and thrown away on every agent.

Building the ``LLM`` directly means crewai takes its ``isinstance(self.llm, LLM)`` branch and
uses the object verbatim, so the timeout and the key actually reach litellm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.llm import LLM

from scan.config import settings as default_settings
from scan.project_logger import get_logger

if TYPE_CHECKING:
    from scan.config import Settings

logger = get_logger(__name__)

__all__ = ["build_llm"]


def build_llm(model_name: str, settings: Settings | None = None) -> LLM:
    """Build the language model for one agent.

    A fresh instance per agent on purpose: crewai's executor mutates ``llm.stop`` in place, so
    a shared object would accumulate another agent's stop words.
    """
    config = settings if settings is not None else default_settings
    llm = LLM(
        model=model_name,
        temperature=0,
        max_tokens=config.MAX_TOKENS,
        timeout=config.REQUEST_TIMEOUT,
        api_key=config.OPENAI_API_KEY,
    )
    logger.info(f"LLM initialized successfully with model: {model_name}")
    return llm
