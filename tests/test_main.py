from crewai.crews.crew_output import CrewOutput, TaskOutput

from scan.main import CustomCrew


def test_custom_crew_combine_outputs():
    crew = CustomCrew("Some topic")
    data = {
        "complex_decision_making_task": "Something complex",
        "emotional_risk_assessment_task": "Something emotional",
        "reward_evaluation_task": "Something rewarding",
        "conflict_resolution_task": "Something conflicting",
        "social_cognition_task": "Something social",
    }
    result = crew.combine_outputs(data)

    assert (
        result
        == "## SCAN AI Final Report on: Some topic\n\n### Decision-making analysis (DLPFC)\nSomething complex\n\n### Emotional analysis (VMPFC)\nSomething emotional\n\n### Reward evaluation (OFC)\nSomething rewarding\n\n### Conflict resolution (ACC)\nSomething conflicting\n\n### Social insights (MPFC)\nSomething social\n\n"
    )


def test_get_task_outputs():
    crew = CustomCrew("Some topic")
    output = CrewOutput(
        tasks_output=[
            TaskOutput(
                description="first test task", name="task 1", raw="raw task 1", agent="some_agent"
            ),
            TaskOutput(
                description="second test task",
                name="task 2",
                raw="raw task 2",
                agent="some_other_agent",
            ),
        ]
    )
    result = crew.get_task_outputs(output)

    assert result == {"task 1": "raw task 1", "task 2": "raw task 2"}
