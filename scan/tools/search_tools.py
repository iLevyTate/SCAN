from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field

from scan.config import settings as default_settings

if TYPE_CHECKING:
    from scan.config import Settings

__all__ = ["SearchInput", "SearchTools"]


class SearchInput(BaseModel):
    """Input schema for the search tool."""

    query: str = Field(description="The search query to run against the internet.")


class SearchTools:
    """Tools for searching the internet via SerpAPI (serpapi.com)."""

    def __init__(self, settings: Settings | None = None) -> None:
        config = settings if settings is not None else default_settings
        if not config.SERPAPI_API_KEY:
            raise ValueError("The SERPAPI_API_KEY environment variable must be set")
        self.search = SerpAPIWrapper(serpapi_api_key=config.SERPAPI_API_KEY)

    def get_search_tool(self) -> Tool:
        """Returns a search tool that agents can use."""
        return Tool(
            name="Search",
            func=self.search.run,
            description="Useful for answering questions about current events or the internet.",
            args_schema=SearchInput,
        )
