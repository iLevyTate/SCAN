"""Translation of provider exceptions into SCAN's own error hierarchy.

crewai calls litellm, which raises OpenAI SDK exception types. Left untranslated they reach the
CLI's catch-all and print as "An unexpected error occurred: <raw provider text>" -- which for the
most likely failure of all, a mistyped model name, gives the user no hint that the fix is in
their own configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openai

from scan.errors import (
    AuthenticationError,
    ModelNotAvailableError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ScanError,
)
from scan.roles import ROLES

if TYPE_CHECKING:
    from scan.config import Settings

__all__ = ["translate"]


def _model_setting(models: dict[str, str], text: str) -> tuple[str, str] | None:
    """Find which configured model a provider message is complaining about.

    Longest name first: model names nest ("gpt-4o" is a substring of "gpt-4o-typoo"), so a
    shorter default configured on another region would otherwise claim the match.

    Several regions commonly share one model -- every ``*_MODEL`` default is the same string --
    and in that case the provider message cannot tell us which region was running. All the
    candidate settings are named rather than guessing one.
    """
    candidates = sorted(
        ((model, setting) for setting, model in models.items() if model),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for model, _ in candidates:
        if model in text:
            sharing = sorted(setting for setting, value in models.items() if value == model)
            return model, " / ".join(sharing)
    return None


def translate(error: Exception, settings: Settings) -> Exception:
    """Map a provider exception onto a SCAN error, or return it unchanged.

    Returns rather than raises so the caller keeps control of the traceback chain.
    """
    if isinstance(error, ScanError):
        return error

    text = str(error)
    models = {role.model_setting: getattr(settings, role.model_setting) for role in ROLES}

    if isinstance(error, openai.AuthenticationError | openai.PermissionDeniedError):
        return AuthenticationError(
            "The provider rejected your API key. Check OPENAI_API_KEY is set and still valid."
        )
    if isinstance(error, openai.RateLimitError):
        return RateLimitError(
            "The provider rate-limited the request or the account is out of quota. "
            f"Provider said: {text}"
        )
    if isinstance(error, openai.APITimeoutError):
        return ProviderTimeoutError(
            f"The provider did not respond within REQUEST_TIMEOUT "
            f"({settings.REQUEST_TIMEOUT}s). Raise it with --timeout if this keeps happening."
        )
    if isinstance(error, openai.NotFoundError | openai.BadRequestError):
        # A bad model name is by far the most likely cause, and it is fully user-fixable.
        match = _model_setting(models, text)
        if match is not None:
            return ModelNotAvailableError(*match, detail=text)
        return ProviderError(f"The provider rejected the request. Provider said: {text}")
    if isinstance(error, openai.APIConnectionError):
        return ProviderError(f"Could not reach the provider. {text}")
    return error
