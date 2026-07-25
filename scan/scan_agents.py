from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from crewai import Agent

from scan.config import settings as default_settings
from scan.openai_llm import build_llm
from scan.project_logger import get_logger
from scan.roles import ROLES, PFCRole, RoleName, get_role
from scan.tools.search_tools import SearchTools

if TYPE_CHECKING:
    from langchain.tools import Tool

    from scan.config import Settings

logger = get_logger(__name__)

__all__ = ["PFCAgents", "RoleName"]


class PFCAgents:
    """Builds one crewai agent per prefrontal-cortex role defined in :mod:`scan.roles`."""

    def __init__(self, topic: str, settings: Settings | None = None) -> None:
        self.topic = topic
        self.settings = settings if settings is not None else default_settings
        self.tools = self._build_tools()

    @property
    def agent_models(self) -> dict[RoleName, str]:
        """Role -> configured model, resolved from settings on each access.

        Deliberately not cached at import: callers override settings (``--model``, ``.env``,
        tests) after this module is imported.
        """
        return {role.name: getattr(self.settings, role.model_setting) for role in ROLES}

    def _build_tools(self) -> list[Tool]:
        """Build the shared tool list for agents (search enabled when configured)."""
        if not self.settings.SERPAPI_API_KEY:
            logger.info("SERPAPI_API_KEY not set; agents will run without the search tool.")
            return []
        return [SearchTools(settings=self.settings).get_search_tool()]

    @cached_property
    def agents(self) -> dict[RoleName, Agent]:
        """The agents, built once.

        Cached so the objects handed to ``Crew(agents=...)`` are the same objects the tasks
        hold; crewai matches tasks to agents by identity.
        """
        agent_dict: dict[RoleName, Agent] = {}
        for role in ROLES:
            agent_dict[role.name] = self.create_agent(role.name)
            logger.info(f"{role.name} agent initialized with model: {self.agent_models[role.name]}")
        return agent_dict

    def create_agent(self, role_name: RoleName | str) -> Agent:
        """Creates an agent with the specified role."""
        role: PFCRole = get_role(role_name)
        return Agent(
            role=role.name,
            backstory=role.backstory(self.topic),
            goal=role.goal,
            llm=build_llm(
                model_name=getattr(self.settings, role.model_setting),
                settings=self.settings,
            ),
            verbose=False,
            tools=self.tools,
        )

    def get_all_agents(self) -> list[Agent]:
        """Returns a list of all initialized agents."""
        return list(self.agents.values())
