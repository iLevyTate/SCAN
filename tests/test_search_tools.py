import pytest

from scan.config import settings
from scan.tools.search_tools import SearchTools


def test_get_search_tool():
    search_tools = SearchTools()
    result = search_tools.get_search_tool()

    assert result.name == "Search"


def test_search_tool_uses_the_configured_key(monkeypatch):
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", "a_specific_key")

    assert SearchTools().search.serpapi_api_key == "a_specific_key"


def test_search_tool_no_key(monkeypatch):
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", None)

    with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
        SearchTools()
