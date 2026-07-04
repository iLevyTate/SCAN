from scan.openai_llm import OpenAIWrapper


def test_wrapper_uses_configured_model():
    # Regression: previously ChatOpenAI was built with `name=` instead of `model=`,
    # so the configured model was silently ignored and defaulted to gpt-3.5-turbo.
    wrapper = OpenAIWrapper(model_name="gpt-4o")

    assert wrapper.llm.model_name == "gpt-4o"


def test_wrapper_default_model():
    assert OpenAIWrapper().llm.model_name == "gpt-4"
