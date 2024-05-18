# scan_agents.py

from crewai import Agent
from textwrap import dedent
from tools.search_tools import SearchTools

class PFCAgents:
    def __init__(self, llm):
        self.llm = llm

    def dlpfc_agent(self):
        return Agent(
            role='Executive Function Manager',
            backstory=dedent("""
                Responsible for managing executive functions such as planning and inhibition, essential in complex cognitive tasks.
            """),
            goal=dedent("""
                Enhance decision-making and cognitive processes
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            cache=True,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
            ]
        )

    def vmpfc_agent(self):
        return Agent(
            role='Emotional and Risk Processor',
            backstory=dedent("""
                Engages in decision-making that involves emotional outcomes and risk assessment.
            """),
            goal=dedent("""
                Process emotional outcomes and assess risks
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            cache=True,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
            ]
        )

    def ofc_agent(self):
        return Agent(
            role='Impulse Control and Reward Evaluation Manager',
            backstory=dedent("""
                Focuses on the assessment of rewards and punishments to guide behavior modifications.
            """),
            goal=dedent("""
                Evaluate actions based on rewards and adjust behaviors
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            cache=True,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
            ]
        )

    def acc_agent(self):
        return Agent(
            role='Performance Monitor and Conflict Resolver',
            backstory=dedent("""
                Acts as a moderator and overseer, ensuring optimal task performance and conflict resolution.
            """),
            goal=dedent("""
                Monitor task performance and resolve conflicts
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            cache=True,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
            ]
        )

    def mpfc_agent(self):
        return Agent(
            role='Social Cognition and Self-Reflection Facilitator',
            backstory=dedent("""
                Engages in understanding social dynamics and internal self-reflection.
            """),
            goal=dedent("""
                Facilitate tasks involving social cognition and introspection
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            cache=True,
            tools=[
                SearchTools.search_internet,
                SearchTools.search_news,
            ]
        )