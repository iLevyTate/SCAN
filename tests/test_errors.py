import openai
import pytest

from scan.config import Settings
from scan.errors import (
    AuthenticationError,
    ConfigurationError,
    MissingEnvironmentVariableError,
    ModelNotAvailableError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ScanError,
)
from scan.provider_errors import translate

CONFIG = Settings(OPENAI_API_KEY="k", DLPFC_MODEL="gpt-4o-typoo", REQUEST_TIMEOUT=12.0)


def _response():
    return openai.NotFoundError, None


@pytest.mark.parametrize(
    "error_class",
    [
        MissingEnvironmentVariableError,
        ModelNotAvailableError,
        AuthenticationError,
        RateLimitError,
        ProviderTimeoutError,
    ],
)
def test_every_error_descends_from_scan_error(error_class):
    # A caller embedding SCAN can catch its failures without catching the provider's.
    assert issubclass(error_class, ScanError)


def test_configuration_and_provider_errors_are_distinguishable():
    assert issubclass(MissingEnvironmentVariableError, ConfigurationError)
    assert issubclass(ModelNotAvailableError, ConfigurationError)
    assert issubclass(AuthenticationError, ProviderError)
    assert not issubclass(ProviderError, ConfigurationError)


def test_missing_environment_variable_names_the_variable():
    error = MissingEnvironmentVariableError("OPENAI_API_KEY")

    assert error.variable_name == "OPENAI_API_KEY"
    assert "OPENAI_API_KEY" in str(error)


def test_model_not_available_points_at_the_setting_to_fix():
    error = ModelNotAvailableError("gpt-4o-typoo", "DLPFC_MODEL", detail="no such model")

    assert "gpt-4o-typoo" in str(error)
    assert "DLPFC_MODEL" in str(error)
    assert "no such model" in str(error)


def test_translate_leaves_scan_errors_alone():
    error = MissingEnvironmentVariableError("OPENAI_API_KEY")

    assert translate(error, CONFIG) is error


def test_translate_leaves_unrelated_errors_alone():
    error = RuntimeError("something else entirely")

    assert translate(error, CONFIG) is error


def test_translate_identifies_a_mistyped_model(monkeypatch):
    # The most likely provider failure, and entirely user-fixable: model names are free-text
    # settings. It used to surface as a raw litellm message with no hint where the fix was.
    raw = openai.NotFoundError.__new__(openai.NotFoundError)
    Exception.__init__(raw, "The model `gpt-4o-typoo` does not exist")

    translated = translate(raw, CONFIG)

    assert isinstance(translated, ModelNotAvailableError)
    assert translated.setting == "DLPFC_MODEL"
    assert "DLPFC_MODEL" in str(translated)


def test_translate_maps_authentication_failures():
    raw = openai.AuthenticationError.__new__(openai.AuthenticationError)
    Exception.__init__(raw, "invalid api key")

    translated = translate(raw, CONFIG)

    assert isinstance(translated, AuthenticationError)
    assert "OPENAI_API_KEY" in str(translated)


def test_translate_maps_rate_limits():
    raw = openai.RateLimitError.__new__(openai.RateLimitError)
    Exception.__init__(raw, "slow down")

    assert isinstance(translate(raw, CONFIG), RateLimitError)


def test_translate_maps_timeouts_and_names_the_setting():
    raw = openai.APITimeoutError.__new__(openai.APITimeoutError)
    Exception.__init__(raw, "timed out")

    translated = translate(raw, CONFIG)

    assert isinstance(translated, ProviderTimeoutError)
    assert "12.0s" in str(translated)
    assert "--timeout" in str(translated)


def test_translate_names_every_region_sharing_an_ambiguous_model():
    # All five *_MODEL defaults are the same string, so a provider complaint about that model
    # genuinely cannot say which region was running. Name them all rather than guess one.
    shared = Settings(OPENAI_API_KEY="k")
    raw = openai.NotFoundError.__new__(openai.NotFoundError)
    Exception.__init__(raw, f"The model `{shared.DLPFC_MODEL}` does not exist")

    translated = translate(raw, shared)

    assert isinstance(translated, ModelNotAvailableError)
    for role_setting in ("ACC_MODEL", "DLPFC_MODEL", "MPFC_MODEL", "OFC_MODEL", "VMPFC_MODEL"):
        assert role_setting in translated.setting


def test_translate_maps_connection_failures():
    raw = openai.APIConnectionError.__new__(openai.APIConnectionError)
    Exception.__init__(raw, "connection refused")

    translated = translate(raw, CONFIG)

    assert isinstance(translated, ProviderError)
    assert "Could not reach the provider" in str(translated)


def test_translate_falls_back_for_an_unrecognised_bad_request():
    raw = openai.BadRequestError.__new__(openai.BadRequestError)
    Exception.__init__(raw, "something the model list cannot explain")

    translated = translate(raw, CONFIG)

    assert isinstance(translated, ProviderError)
    assert not isinstance(translated, ModelNotAvailableError)
