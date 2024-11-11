# scan/tools/search_tools.py
import os

from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper


class SearchTools:
    """Tools for searching the internet."""

    def __init__(self) -> None:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("The SERPER_API_KEY environment variable must be set")
        self.search = SerpAPIWrapper(serpapi_api_key=api_key)

    def get_search_tool(self) -> Tool:
        """Returns a search tool that agents can use."""
        return Tool(
            name="Search",
            func=self.search.run,
            description="Useful for answering questions about current events or the internet.",
        )
