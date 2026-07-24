import pytest

from scan.config import settings


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    """Give every test deterministic credentials.

    monkeypatch (rather than plain assignment) so a test that changes a setting cannot leak
    that change into later tests, and so a developer's real .env cannot influence the suite.
    """
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "my_key")
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", "search_key")
