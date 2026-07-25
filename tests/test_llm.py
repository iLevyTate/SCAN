from crewai import Agent
from crewai.llm import LLM

from scan.config import Settings, settings
from scan.llm import build_llm


def test_build_llm_uses_the_configured_model():
    # Regression: ChatOpenAI was once built with `name=` instead of `model=`, so the configured
    # model was silently ignored and defaulted to gpt-3.5-turbo.
    assert build_llm("gpt-4o").model == "gpt-4o"


def test_build_llm_carries_every_setting_crewai_used_to_drop():
    # Regression: crewai rebuilt its own LLM by scraping attributes off the ChatOpenAI object.
    # `timeout` and `api_key` are stored by langchain as `request_timeout` and
    # `openai_api_key`, so both came back None and were stripped -- leaving a run with no
    # request timeout at all, able to hang indefinitely behind the spinner.
    config = Settings(OPENAI_API_KEY="sk-explicit", MAX_TOKENS=1234, REQUEST_TIMEOUT=42.0)

    llm = build_llm("gpt-4o", settings=config)

    assert llm.max_tokens == 1234
    assert llm.timeout == 42.0
    assert llm.api_key == "sk-explicit"
    assert llm.temperature == 0


def test_crewai_uses_the_llm_object_verbatim():
    # crewai's Agent takes an `isinstance(llm, LLM)` fast path and keeps the object as-is,
    # which is the whole reason for building an LLM rather than a ChatOpenAI.
    llm = build_llm("gpt-4o", settings=Settings(OPENAI_API_KEY="sk-explicit"))

    agent = Agent(role="R", goal="g", backstory="b", llm=llm)

    assert agent.llm is llm
    assert agent.llm.api_key == "sk-explicit"
    assert agent.llm.timeout == settings.REQUEST_TIMEOUT


def test_each_agent_gets_its_own_llm():
    # crewai's executor mutates llm.stop in place, so sharing one object across agents would
    # let one agent's stop words leak into another's.
    assert build_llm("gpt-4o") is not build_llm("gpt-4o")


def test_defaults_come_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOKENS", 777)
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 9.5)

    llm = build_llm("gpt-4o")

    assert (llm.max_tokens, llm.timeout) == (777, 9.5)


def test_build_llm_returns_a_crewai_llm():
    assert isinstance(build_llm("gpt-4o"), LLM)
