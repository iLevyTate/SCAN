import pytest

from scan.config import settings
from scan.roles import ROLES
from scan.scan_agents import PFCAgents

ROLE_NAMES = [role.name for role in ROLES]


@pytest.fixture
def pfc_agents():
    return PFCAgents(topic="Some topic")


def test_agents(pfc_agents):
    assert list(pfc_agents.agents.keys()) == ROLE_NAMES


@pytest.mark.parametrize("role", ROLES, ids=lambda role: role.name)
def test_create_agent_uses_the_role_table(pfc_agents, role):
    # Regression: only DLPFC's persona was ever asserted, so four of the five branches in each
    # of the old if/elif chains had no coverage at all.
    agent = pfc_agents.create_agent(role.name)

    assert agent.role == role.name
    assert agent.backstory == role.backstory("Some topic")
    assert agent.goal == role.goal


def test_create_agent_rejects_an_unknown_role(pfc_agents):
    with pytest.raises(KeyError, match="Unknown PFC role"):
        pfc_agents.create_agent("NOTAREGION")


def test_get_all_agents(pfc_agents):
    assert len(pfc_agents.get_all_agents()) == len(ROLES)


def test_agents_are_cached_so_crew_and_tasks_share_objects(pfc_agents):
    # crewai matches tasks to agents by identity: if `agents` stopped being cached, the Crew
    # would receive five agents unrelated to the ones the tasks hold, and nothing would fail.
    assert pfc_agents.agents is pfc_agents.agents
    assert pfc_agents.get_all_agents()[0] is pfc_agents.agents[ROLE_NAMES[0]]


def test_agents_use_configured_models(monkeypatch):
    # Regression: the model was previously ignored, so every agent silently ran on the default
    # model. Each role gets a *distinct* model here on purpose -- every *_MODEL default is the
    # same string, so a test using the defaults passes even if the role is ignored entirely.
    distinct = {role.name: f"gpt-4o-{index}" for index, role in enumerate(ROLES)}
    for role in ROLES:
        monkeypatch.setattr(settings, role.model_setting, distinct[role.name])

    agents = PFCAgents(topic="Some topic")

    assert agents.agent_models == distinct
    assert {name: agent.llm.model for name, agent in agents.agents.items()} == distinct


def test_agent_models_resolve_lazily(pfc_agents, monkeypatch):
    # The role table stores the settings *attribute name*, not a resolved value, so overrides
    # applied after import (--model, .env, tests) are still honoured.
    monkeypatch.setattr(settings, "DLPFC_MODEL", "gpt-4o-changed")

    assert pfc_agents.agent_models["DLPFC"] == "gpt-4o-changed"


def test_max_tokens_reaches_the_agent(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOKENS", 321)

    assert PFCAgents(topic="Some topic").agents["DLPFC"].llm.max_tokens == 321


def test_agents_have_search_tool_when_serpapi_set(pfc_agents):
    # conftest sets SERPAPI_API_KEY, so the search tool is wired in.
    assert [tool.name for tool in pfc_agents.create_agent("DLPFC").tools] == ["Search"]


def test_agents_have_no_tools_without_serpapi(monkeypatch):
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", None)
    agents = PFCAgents(topic="Some topic")

    assert agents.tools == []
    assert agents.create_agent("DLPFC").tools == []


def test_agents_do_not_receive_an_unsupported_memory_kwarg(pfc_agents):
    # Regression: create_agent passed `memory=True` to crewai's Agent, which has no such
    # field. Pydantic silently dropped it, so the setting never did anything.
    assert not hasattr(pfc_agents.create_agent("DLPFC"), "memory")


def test_explicit_settings_override_the_singleton():
    overridden = settings.model_copy(update={"DLPFC_MODEL": "gpt-4o-injected"})

    agents = PFCAgents(topic="Some topic", settings=overridden)

    assert agents.agent_models["DLPFC"] == "gpt-4o-injected"
    assert agents.agents["DLPFC"].llm.model == "gpt-4o-injected"
