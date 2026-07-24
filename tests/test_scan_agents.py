import pytest

from scan.config import settings
from scan.scan_agents import PFCAgents

ROLES = ("DLPFC", "VMPFC", "OFC", "ACC", "MPFC")


@pytest.fixture
def pfc_agents():
    return PFCAgents(topic="Some topic")


def test_agents(pfc_agents):
    assert list(pfc_agents.agents.keys()) == list(ROLES)


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


def test_agents_use_configured_models(monkeypatch):
    # Regression: the model was previously ignored, so every agent silently ran on
    # the default model regardless of the configured *_MODEL settings.
    #
    # Each role gets a *distinct* model here on purpose. Every *_MODEL default is the same
    # string, so a test using the defaults passes even if create_agent ignores the role
    # entirely -- which it did, undetected, before this was tightened.
    distinct = {role: f"gpt-4o-{index}" for index, role in enumerate(ROLES)}
    for role, model in distinct.items():
        monkeypatch.setattr(settings, f"{role}_MODEL", model)

    agents = PFCAgents(topic="Some topic")

    assert {role: agent.llm.model for role, agent in agents.agents.items()} == distinct


def test_agents_have_search_tool_when_serpapi_set(pfc_agents):
    # conftest sets SERPAPI_API_KEY, so the search tool is wired in.
    assert len(pfc_agents.create_agent("DLPFC").tools) == 1


def test_agents_have_no_tools_without_serpapi(monkeypatch):
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", None)
    agents = PFCAgents(topic="Some topic")

    assert agents.tools == []
    assert agents.create_agent("DLPFC").tools == []


def test_agents_do_not_receive_an_unsupported_memory_kwarg(pfc_agents):
    # Regression: create_agent passed `memory=True` to crewai's Agent, which has no such
    # field. Pydantic silently dropped it, so the setting never did anything.
    agent = pfc_agents.create_agent("DLPFC")

    assert not hasattr(agent, "memory")
