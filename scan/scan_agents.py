# scan/scan_agents.py

import logging
from textwrap import dedent
from crewai import Agent
from scan.openai_llm import OpenAIWrapper  # Updated OpenAIWrapper with LangChain

# Configure logging
logger = logging.getLogger(__name__)

class PFCAgents:
    """Defines the PFC agents with their respective roles and responsibilities."""

    def __init__(self, api_key: str, topic: str):
        """
        Initializes the PFCAgents class.

        Args:
            api_key (str): OpenAI API key.
            topic (str): The topic for analysis.
        """
        self.api_key = api_key
        self.topic = topic
        self.agent_models = {
            'DLPFC': 'ft:gpt-4o-mini-2024-07-18:personal:dlpfcv2:A4gAvc9k',
            'VMPFC': 'ft:gpt-4o-mini-2024-07-18:personal:vmpfcv2:A523M03L',
            'OFC': 'ft:gpt-4o-mini-2024-07-18:personal:ofcv3datasetnew09092924:A5h8dT2o',
            'ACC': 'ft:gpt-4o-mini-2024-07-18:personal:accv4:A5hb0Wp0',
            'MPFC': 'ft:gpt-4o-mini-2024-07-18:personal:mpfcv2:A8xFTDjF'
        }

    def create_agent(self, role_name):
        """Creates an agent with the specified role."""
        model_name = self.agent_models.get(role_name, 'gpt-4')
        logger.info(f"Initializing {role_name} agent with model: {model_name}")
        llm_wrapper = OpenAIWrapper(api_key=self.api_key, model_name=model_name, max_tokens=1000)
        llm = llm_wrapper.llm  # Get the LangChain LLM

        backstory = dedent(f"""
            You are the {role_name}, focusing on {self.get_backstory(role_name)} for the topic '{self.topic}'.
            Please ensure you follow the task instructions precisely and provide concise responses.
        """)
        goal = self.get_goal(role_name)

        return Agent(
            role=role_name,
            backstory=backstory,
            goal=goal,
            llm=llm,
            memory=True,
            verbose=True,
            tools=[],
        )

    def get_backstory(self, role_name):
        """Returns the backstory for the given role."""
        backstories = {
            'DLPFC': "executive functions like planning and decision-making",
            'VMPFC': "assessing emotional outcomes and risks",
            'OFC': "balancing rewards against emotional risks",
            'ACC': "resolving conflicts between emotional, reward-based, and logical inputs",
            'MPFC': "understanding social dynamics and self-reflection",
        }
        return backstories.get(role_name, "")

    def get_goal(self, role_name):
        """Returns the goal for the given role."""
        goals = {
            'DLPFC': dedent("""
                Make decisions based on integrated logical, emotional, and social perspectives.
                Ensure you synthesize information effectively and provide strategic recommendations.
            """),
            'VMPFC': dedent("""
                Provide emotional insights to aid in decision-making.
                Assess emotional factors thoroughly and concisely.
            """),
            'OFC': dedent("""
                Assess actions based on rewards and manage impulses effectively.
                Provide a concise evaluation of potential rewards and risks.
            """),
            'ACC': dedent("""
                Resolve conflicts in the decision-making process.
                Analyze conflicts carefully and provide clear resolution strategies.
            """),
            'MPFC': dedent("""
                Analyze social interactions and provide insights for personal growth.
                Focus on social cognition aspects relevant to the topic.
            """),
        }
        return goals.get(role_name, "")

    def dlpfc_agent(self) -> Agent:
        return self.create_agent('DLPFC')

    def vmpfc_agent(self) -> Agent:
        return self.create_agent('VMPFC')

    def ofc_agent(self) -> Agent:
        return self.create_agent('OFC')

    def acc_agent(self) -> Agent:
        return self.create_agent('ACC')

    def mpfc_agent(self) -> Agent:
        return self.create_agent('MPFC')
