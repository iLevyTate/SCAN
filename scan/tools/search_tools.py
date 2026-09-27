from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool
from langchain_community.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field, PrivateAttr

from scan.config import settings as default_settings
from scan.errors import MissingEnvironmentVariableError

if TYPE_CHECKING:
    from scan.config import Settings

__all__ = ["SearchInput", "SearchTool", "SearchTools"]


class SearchInput(BaseModel):
    """Input schema for the search tool."""

    query: str = Field(description="The search query to run against the internet.")


class SearchTool(BaseTool):
    """crewai tool that runs a query through SerpAPI.

    crewai 1.x validates ``Agent(tools=...)`` against its own ``BaseTool``; the LangChain
    ``Tool`` wrapper this used to return is rejected with "Input should be a valid
    dictionary or instance of BaseTool".
    """

    name: str = "Search"
    description: str = "Useful for answering questions about current events or the internet."
    args_schema: type[BaseModel] = SearchInput
    _search: SerpAPIWrapper = PrivateAttr()

    def __init__(self, search: SerpAPIWrapper, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._search = search

    def _run(self, query: str) -> str:
        return str(self._search.run(query))


class SearchTools:
    """Tools for searching the internet via SerpAPI (serpapi.com)."""

    def __init__(self, settings: Settings | None = None) -> None:
        config = settings if settings is not None else default_settings
        if not config.SERPAPI_API_KEY:
            # Was a bare ValueError with its own message format, for exactly the failure mode
            # MissingEnvironmentVariableError already describes.
            raise MissingEnvironmentVariableError("SERPAPI_API_KEY")
        self.search = SerpAPIWrapper(serpapi_api_key=config.SERPAPI_API_KEY)

    def get_search_tool(self) -> SearchTool:
        """Returns a search tool that agents can use."""
        return SearchTool(self.search)
