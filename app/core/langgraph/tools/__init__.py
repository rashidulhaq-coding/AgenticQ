"""Tools for the QA agent."""

from langchain_community.tools import DuckDuckGoSearchResults

from app.core.langgraph.tools.weather_tool import get_weather

duckduckgo_search = DuckDuckGoSearchResults(
    name="duckduckgo_search",
    description=(
        "Search the web using DuckDuckGo to find information relevant to the user's question. "
        "Use this tool when you need to look up factual information, current events, definitions, "
        "or any topic that benefits from a web search. Input should be a search query string."
    ),
)

ALL_TOOLS = [duckduckgo_search, get_weather]

__all__ = ["ALL_TOOLS", "duckduckgo_search", "get_weather"]