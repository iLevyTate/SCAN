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
            Task: Monitor and resolve conflicts related to '{topic}'.
            Actions Required:
            - Identify sources of conflict and involved parties within the context of '{topic}'.
            - Analyze the underlying causes of these conflicts, detecting any errors in performance or communication.
            - Develop and implement conflict resolution strategies to address both cognitive and emotional aspects.
            - Regulate attention and motivation levels to ensure effective conflict resolution and sustained focus on goals.
            Expected Output:
            A conflict resolution report with actionable steps and outcomes to enhance performance, resolve conflicts, and maintain attention and motivation related to '{topic}'.
        """)
        return Task(
            description=description,
            expected_output="A report detailing conflict resolution strategies, performance monitoring, and attention/motivation regulation outcomes.",
            agent=self.agents.acc_agent(),
        )

    def social_cognition_task(self, topic):
        description = dedent(f"""
            Task: Analyze and enhance social dynamics related to '{topic}'.
            Actions Required:
            - Assess the current social interactions and their impact on the topic '{topic}'.
            - Identify areas for improvement in social interactions and self-reflection.
            - Propose interventions to enhance social cognition and personal growth.
            Expected Output:
            A strategic plan to improve social interactions and personal dynamics, backed by social psychological insights related to '{topic}'.
        """)
        return Task(
            description=description,
            expected_output="A strategic plan with interventions for enhancing social cognition and personal dynamics.",
            agent=self.agents.mpfc_agent(),
        )
