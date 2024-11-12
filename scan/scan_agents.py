# scan/scan_agents.py

from textwrap import dedent
from typing import Literal, TypeAlias, cast

from crewai import Agent

from scan.config import settings
from scan.openai_llm import OpenAIWrapper
from scan.project_logger import logger

RoleName: TypeAlias = Literal["DLPFC", "VMPFC", "OFC", "ACC", "MPFC"]


class PFCAgents:
    """Defines the PFC agents with their respective roles and responsibilities."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.agent_models = {
            "DLPFC": settings.DLPFC_MODEL,
            "VMPFC": settings.VMPFC_MODEL,
            "OFC": settings.OFC_MODEL,
            "ACC": settings.ACC_MODEL,
            "MPFC": settings.MPFC_MODEL,
        }
        self.agents: dict[str, Agent] = {}
        self.initialize_agents()

    def initialize_agents(self) -> None:
        """Initialize all PFC agents."""
        for role_name in self.agent_models.keys():
            agent = self.create_agent(cast(RoleName, role_name))
            self.agents[role_name] = agent
            logger.info(f"{role_name} agent initialized with model: {self.agent_models[role_name]}")

    def create_agent(self, role_name: RoleName) -> Agent:
        """Creates an agent with the specified role."""
        model_name = self.agent_models[role_name]
        llm_wrapper = OpenAIWrapper(model_name=model_name)
        llm = llm_wrapper.llm

        backstory = dedent(f"""
            You are the {role_name}, focusing on {self.get_backstory(role_name)} for the topic '{self.topic}'.
            Please ensure you follow the task instructions precisely and provide concise responses.
        """).strip()

        goal = self.get_goal(role_name)

        return Agent(
            role=role_name,
            backstory=backstory,
            goal=goal,
            llm=llm,
            memory=True,
            verbose=False,
            tools=[],
        )

    def get_backstory(self, role_name: RoleName) -> str:
        """Returns the backstory for the given role."""
        if role_name == "DLPFC":
            return "executive functions like planning and decision-making"
        elif role_name == "VMPFC":
            return "assessing emotional outcomes and risks"
        elif role_name == "OFC":
            return "balancing rewards against emotional risks"
        elif role_name == "ACC":
            return "resolving conflicts between emotional, reward-based, and logical inputs"
        else:
            return "understanding social dynamics and self-reflection"

    def get_goal(self, role_name: RoleName) -> str:
        """Returns the goal for the given role."""
        if role_name == "DLPFC":
            return dedent("""
                Make decisions based on integrated logical, emotional, and social perspectives.
                Ensure you synthesize information effectively and provide strategic recommendations.
            """)
        elif role_name == "VMPFC":
            return dedent("""
                Provide emotional insights to aid in decision-making.
                Assess emotional factors thoroughly and concisely.
            """)
        elif role_name == "OFC":
            return dedent("""
                Assess actions based on rewards and manage impulses effectively.
                Provide a concise evaluation of potential rewards and risks.
            """)
        elif role_name == "ACC":
            return dedent("""
                Resolve conflicts in the decision-making process.
                Analyze conflicts carefully and provide clear resolution strategies.
            """)
        else:
            return dedent("""
                Analyze social interactions and provide insights for personal growth.
                Focus on social cognition aspects relevant to the topic.
            """)

    def get_all_agents(self) -> list[Agent]:
        """Returns a list of all initialized agents."""
        return list(self.agents.values())
