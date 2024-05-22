# scan_tasks.py

from textwrap import dedent
from crewai import Task

class PFCTasks:
    def __init__(self, agents):
        self.agents = agents

    def complex_decision_making_task(self, topic):
        description = dedent(f"""
            Task: Analyze a complex situation involving the topic '{topic}'.
            Actions Required:
            - Conduct a thorough analysis of all available data on '{topic}'.
            - Synthesize information to identify key trends and insights.
            - Develop a set of recommendations based on analytical findings.
            Expected Output:
            A comprehensive report that outlines the situation analysis, key findings, and strategic recommendations on '{topic}'.
        """)
        return Task(
            description=description,
            expected_output="A comprehensive analytical report with strategic recommendations.",
            agent=self.agents.dlpfc_agent(),
        )

    def emotional_risk_assessment_task(self, topic):
        description = dedent(f"""
            Task: Evaluate decisions involving high emotional impact related to '{topic}'.
            Actions Required:
            - Identify the emotional and psychological factors at play in decisions about '{topic}'.
            - Assess the potential risks associated with these emotional factors.
            - Recommend strategies to mitigate risks while addressing emotional concerns.
            Expected Output:
            A balanced evaluation report detailing emotional factors, associated risks, and mitigation strategies for '{topic}'.
        """)
        return Task(
            description=description,
            expected_output="An evaluation report with balanced insights into emotional and rational aspects.",
            agent=self.agents.vmpfc_agent(),
        )

    def reward_evaluation_task(self, topic):
        description = dedent(f"""
            Task: Assess different actions or options based on potential rewards related to '{topic}'.
            Actions Required:
            - Evaluate the potential rewards associated with each option concerning '{topic}'.
            - Consider long-term impacts and sustainability of the rewards.
            - Provide a ranked list of options based on the overall benefit analysis.
            Expected Output:
            A detailed assessment of options with a focus on long-term rewards and strategic benefits related to '{topic}'.
        """)
        return Task(
            description=description,
            expected_output="An assessment document ranking options by potential rewards and strategic value.",
            agent=self.agents.ofc_agent(),
        )

    def conflict_resolution_task(self, topic):
        description = dedent(f"""
            Task: Monitor and resolve conflicts within a team or project related to '{topic}'.
            Actions Required:
            - Identify sources of conflict and involved parties within the context of '{topic}'.
            - Analyze the underlying causes of these conflicts.
            - Develop and implement conflict resolution strategies.
            Expected Output:
            A conflict resolution report with actionable steps and outcomes to enhance team dynamics related to '{topic}'.
        """)
        return Task(
            description=description,
            expected_output="A report detailing conflict resolution strategies and their implementation outcomes.",
            agent=self.agents.acc_agent(),
        )

    def social_cognition_task(self, topic):
        description = dedent(f"""
            Task: Analyze and enhance social dynamics within a team or group to improve communication and effectiveness concerning '{topic}'.
            Actions Required:
            - Assess the current social interactions and their impact on team performance in relation to '{topic}'.
            - Identify areas for improvement in communication and collaboration.
            - Propose interventions to enhance social cognition and team cohesion.
            Expected Output:
            A strategic plan to improve social interactions and team dynamics, backed by social psychological insights related to '{topic}'.
        """)
        return Task(
            description=description,
            expected_output="A strategic plan with interventions for enhancing team dynamics.",
            agent=self.agents.mpfc_agent(),
        )
