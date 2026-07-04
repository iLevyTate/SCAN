import pytest

from scan.config import settings
from scan.scan_agents import PFCAgents


@pytest.fixture
def pfc_agents():
    return PFCAgents(topic="Some topic")


def test_agents(pfc_agents):
    assert list(pfc_agents.agents.keys()) == ["DLPFC", "VMPFC", "OFC", "ACC", "MPFC"]


def test_create_agent(pfc_agents):
    result = pfc_agents.create_agent("DLPFC")

    assert result.role == "DLPFC"
    assert (
        result.backstory
        == "You are the DLPFC, focusing on executive functions like planning and decision-making for the topic 'Some topic'.\nPlease ensure you follow the task instructions precisely and provide concise responses."
    )
    assert (
        result.goal
        == "Make decisions based on integrated logical, emotional, and social perspectives.\nEnsure you synthesize information effectively and provide strategic recommendations."
    )


def test_get_all_agents(pfc_agents):
    assert len(pfc_agents.get_all_agents()) == 5


def test_agents_use_configured_models(pfc_agents):
    # Regression: the model was previously ignored, so every agent silently ran on
    # the default model regardless of the configured *_MODEL settings.
    expected = {
        "DLPFC": settings.DLPFC_MODEL,
        "VMPFC": settings.VMPFC_MODEL,
        "OFC": settings.OFC_MODEL,
        "ACC": settings.ACC_MODEL,
        "MPFC": settings.MPFC_MODEL,
    }
    for role, agent in pfc_agents.agents.items():
        assert agent.llm.model == expected[role]


def test_agents_have_search_tool_when_serper_set(pfc_agents):
    # conftest sets SERPER_API_KEY for the session, so the search tool is wired in.
    assert len(pfc_agents.create_agent("DLPFC").tools) == 1


def test_agents_have_no_tools_without_serper(monkeypatch):
    monkeypatch.setattr(settings, "SERPER_API_KEY", None)
    agents = PFCAgents(topic="Some topic")

    assert agents.tools == []
    assert agents.create_agent("DLPFC").tools == []
