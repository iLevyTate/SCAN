from scan.config import settings
from scan.openai_llm import OpenAIWrapper


def test_wrapper_uses_configured_model():
    # Regression: previously ChatOpenAI was built with `name=` instead of `model=`,
    # so the configured model was silently ignored and defaulted to gpt-3.5-turbo.
    wrapper = OpenAIWrapper(model_name="gpt-4o")

    assert wrapper.llm.model_name == "gpt-4o"


def test_wrapper_defaults_max_tokens_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOKENS", 1234)

    assert OpenAIWrapper(model_name="gpt-4o").llm.max_tokens == 1234


def test_wrapper_max_tokens_can_be_overridden():
    assert OpenAIWrapper(model_name="gpt-4o", max_tokens=42).llm.max_tokens == 42
