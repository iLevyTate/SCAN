import pytest

from scan.config import settings
from scan.tools.search_tools import SearchTools


@pytest.fixture
def no_search_key():
    original_key = settings.SERPER_API_KEY
    settings.SERPER_API_KEY = None
    yield
    settings.SERPER_API_KEY = original_key


def test_get_search_tool():
    search_tools = SearchTools()
    result = search_tools.get_search_tool()

    assert result.name == "Search"


@pytest.mark.usefixtures("no_search_key")
def test_search_tool_no_key():
    with pytest.raises(ValueError):
        SearchTools()
