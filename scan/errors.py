"""SCAN's exception hierarchy.

Everything SCAN raises descends from :class:`ScanError`, so a caller embedding SCAN can catch
its failures without also catching the provider's. The split that matters is
:class:`ConfigurationError` (the user can fix this in their .env or on the command line) versus
:class:`ProviderError` (the upstream API said no).
"""

from __future__ import annotations


class ScanError(Exception):
    """Base class for every error SCAN raises itself."""


class ConfigurationError(ScanError):
    """SCAN is misconfigured. The user can fix this without touching the code."""


class MissingEnvironmentVariableError(ConfigurationError):
    """A required environment variable is missing."""

    def __init__(self, variable_name: str) -> None:
        self.variable_name = variable_name
        self.message = f"The environment variable '{variable_name}' is missing."
        super().__init__(self.message)


class ModelNotAvailableError(ConfigurationError):
    """A configured model was rejected by the provider.

    Model names are free-text settings, so a typo is a likely and entirely user-fixable
    failure. It used to surface as a raw litellm message with no hint that the fix was in
    the caller's .env.
    """

    def __init__(self, model: str, setting: str, detail: str = "") -> None:
        self.model = model
        self.setting = setting
        self.message = (
            f"The model {model!r} (configured as {setting}) was rejected by the provider. "
            f"Check the spelling and that your account has access to it."
        )
        if detail:
            self.message = f"{self.message} Provider said: {detail}"
        super().__init__(self.message)


class ProviderError(ScanError):
    """The upstream model provider failed the request."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AuthenticationError(ProviderError):
    """The provider rejected the API key."""


class RateLimitError(ProviderError):
    """The provider rate-limited or ran out of quota."""


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within REQUEST_TIMEOUT."""
