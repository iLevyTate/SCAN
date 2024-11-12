import pytest

from scan.scan_agents import PFCAgents
from scan.scan_tasks import PFCTasks


@pytest.fixture
def pfc_tasks():
    return PFCTasks(PFCAgents(topic="Some topic"))


def tests_complex_decision_making_task(pfc_tasks):
    topic = "Test topic"
    task = pfc_tasks.complex_decision_making_task(topic=topic)

    assert task.name == "complex_decision_making_task"
    assert topic in task.description


def test_emotional_risk_assessment_task(pfc_tasks):
    topic = "Test topic"
    task = pfc_tasks.emotional_risk_assessment_task(topic=topic)

    assert task.name == "emotional_risk_assessment_task"
    assert topic in task.description


def test_reward_evaluation_task(pfc_tasks):
    topic = "Test topic"
    task = pfc_tasks.reward_evaluation_task(topic=topic)

    assert task.name == "reward_evaluation_task"
    assert topic in task.description


def test_conflict_resolution_task(pfc_tasks):
    topic = "Test topic"
    task = pfc_tasks.conflict_resolution_task(topic=topic)

    assert task.name == "conflict_resolution_task"
    assert topic in task.description


def test_social_cognition_task(pfc_tasks):
    topic = "Test topic"
    task = pfc_tasks.social_cognition_task(topic=topic)

    assert task.name == "social_cognition_task"
    assert topic in task.description
