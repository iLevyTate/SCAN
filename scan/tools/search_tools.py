# tools/search_tools.py

import json
import os

import requests
from langchain.tools import tool


class SearchTools:
    @tool("Search the internet")
    def search_internet(query):
        """Useful to search the internet
        about a given topic and return relevant results"""
        return _process_results(query, url="https://google.serper.dev/search")

    @tool("Search news on the internet")
    def search_news(query):
        """Useful to search news about a company, stock or any other
        topic and return relevant results"""
        return _process_results(query, url="https://google.serper.dev/news")


def _process_results(query, url, top_result_to_return=4):
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "content-type": "application/json",
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()["organic"]
    string = []
    for result in results[:top_result_to_return]:
        try:
            string.append(
                "\n".join(
                    [
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}",
                        "\n-----------------",
                    ]
                )
            )
        except KeyError:
            pass
    return "\n".join(string)
