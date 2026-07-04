from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field

from scan.config import settings


class SearchInput(BaseModel):
    """Input schema for the search tool."""

    query: str = Field(description="The search query to run against the internet.")


class SearchTools:
    """Tools for searching the internet."""

    def __init__(self) -> None:
        if not settings.SERPER_API_KEY:
            raise ValueError("The SERPER_API_KEY environment variable must be set")
        self.search = SerpAPIWrapper(serpapi_api_key=settings.SERPER_API_KEY)

    def get_search_tool(self) -> Tool:
        """Returns a search tool that agents can use."""
        return Tool(
            name="Search",
            func=self.search.run,
            description="Useful for answering questions about current events or the internet.",
            args_schema=SearchInput,
        )
