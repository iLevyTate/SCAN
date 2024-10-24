# scan/scan_tasks.py

from textwrap import dedent

from crewai import Task


class PFCTasks:
    """Defines the tasks assigned to each PFC agent."""

    def __init__(self, agents):
        if not agents:
            raise ValueError("Agents must be initialized.")
        self.agents = agents

    def complex_decision_making_task(self, topic):
        """Task for DLPFC agent to make complex decisions."""
        description = dedent(f"""
            Task: Analyze a complex situation involving '{topic}' by integrating insights from other agents.
            Actions Required:
            - Conduct a thorough analysis of all available data on '{topic}'.
            - Synthesize information to identify key trends and insights.
            - Develop a set of recommendations based on analytical findings.
            Expected Output:
            A comprehensive report that outlines the situation analysis, key findings, and strategic recommendations on '{topic}'.
        """)
        task = Task(
            name="complex_decision_making_task",
            description=description,
            expected_output="A comprehensive analytical report with strategic recommendations.",
            agent=self.agents.dlpfc_agent(),
            dependencies=[
                "emotional_risk_assessment_task",
                "reward_evaluation_task",
                "social_cognition_task",
            ],  # Task dependencies
        )
        return task

    def emotional_risk_assessment_task(self, topic):
        """Task for VMPFC agent to assess emotional risks."""
        description = dedent(f"""
            Task: Evaluate decisions involving high emotional impact related to '{topic}'.
            Actions Required:
            - Identify the emotional and psychological factors at play in decisions about '{topic}'.
            - Assess the potential risks associated with these emotional factors.
            - Recommend strategies to mitigate risks while addressing emotional concerns.
            Expected Output:
            A balanced evaluation report detailing emotional factors, associated risks, and mitigation strategies for '{topic}'.
        """)
        task = Task(
            name="emotional_risk_assessment_task",
            description=description,
            expected_output="An evaluation report with balanced insights into emotional and rational aspects.",
            agent=self.agents.vmpfc_agent(),
        )
        return task

    def reward_evaluation_task(self, topic):
        """Task for OFC agent to evaluate rewards."""
        description = dedent(f"""
            Task: Assess different actions or options based on potential rewards related to '{topic}'.
            Actions Required:
            - Evaluate the potential rewards associated with each option concerning '{topic}'.
            - Consider long-term impacts and sustainability of the rewards.
            - Provide a ranked list of options based on the overall benefit analysis.
            Expected Output:
            A detailed assessment of options with a focus on long-term rewards and strategic benefits related to '{topic}'.
        """)
        task = Task(
            name="reward_evaluation_task",
            description=description,
            expected_output="An assessment document ranking options by potential rewards and strategic value.",
            agent=self.agents.ofc_agent(),
        )
        return task

    def conflict_resolution_task(self, topic):
        """Task for ACC agent to resolve conflicts."""
        description = dedent(f"""
            Task: Resolve conflicts between emotional, reward-based, and logical inputs for '{topic}'.
            Actions Required:
            - Identify sources of conflict within the context of '{topic}'.
            - Analyze the underlying causes of these conflicts.
            - Develop and implement conflict resolution strategies.
            Expected Output:
            A conflict resolution report with actionable steps and outcomes for '{topic}'.
        """)
        task = Task(
            name="conflict_resolution_task",
            description=description,
            expected_output="A report detailing conflict resolution strategies and outcomes.",
            agent=self.agents.acc_agent(),
            dependencies=[
                "emotional_risk_assessment_task",
                "reward_evaluation_task",
            ],  # ACC depends on VMPFC and OFC outputs
        )
        return task

    def social_cognition_task(self, topic):
        """Task for MPFC agent to enhance social cognition."""
        description = dedent(f"""
            Task: Analyze and enhance social dynamics related to '{topic}'.
            Actions Required:
            - Assess current social interactions and their impact on '{topic}'.
            - Identify areas for improvement in social interactions.
            - Propose interventions to enhance social cognition and personal growth.
            Expected Output:
            A strategic plan to improve social interactions related to '{topic}'.
        """)
        task = Task(
            name="social_cognition_task",
            description=description,
            expected_output="A strategic plan with interventions for enhancing social cognition.",
            agent=self.agents.mpfc_agent(),
        )
        return task
