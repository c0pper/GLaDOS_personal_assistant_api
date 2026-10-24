from src.agents.web_search_agent import run_web_search_agent
from src.config import Config


async def test_uses_tavily_server_with_only_its_tools(fake_mcp):
    fake_mcp.replies = ["Inter won the 2025-2026 Serie A title."]

    result = await run_web_search_agent("Who won the league?")

    assert result == {"response": "Inter won the 2025-2026 Serie A title."}
    assert fake_mcp.session_urls == [Config.TAVILY_MCP_URL]
    offered = {tool["function"]["name"] for tool in fake_mcp.llm_requests[0]["tools"]}
    assert offered == set(Config.TAVILY_MCP_TOOLS)
    assert fake_mcp.llm_requests[0]["model"] == Config.TAVILY_AGENT_MODEL
    assert "searching the web" in fake_mcp.llm_requests[0]["messages"][0]["content"]


async def test_searches_then_reads_a_page(fake_mcp):
    fake_mcp.tool_results = {
        "tavily_search": "Result: https://example.com/race (yesterday)",
        "tavily_extract": "Verstappen won the race.",
    }
    fake_mcp.replies = [
        [("tavily_search", {"query": "who won the race yesterday"})],
        [("tavily_extract", {"urls": ["https://example.com/race"]})],
        "Verstappen won yesterday's race.",
    ]

    result = await run_web_search_agent("Who won the race yesterday?")

    assert result == {"response": "Verstappen won yesterday's race."}
    assert fake_mcp.tool_calls == [
        ("tavily_search", {"query": "who won the race yesterday"}),
        ("tavily_extract", {"urls": ["https://example.com/race"]}),
    ]


async def test_crawl_and_research_are_refused(fake_mcp):
    fake_mcp.replies = [[("tavily_research", {"query": "everything"})], "Not available."]

    await run_web_search_agent("Research everything")

    assert fake_mcp.tool_calls == []
    assert fake_mcp.llm_requests[1]["messages"][-1]["content"] == (
        "Tool error: 'tavily_research' is not available."
    )


async def test_errors_name_the_agent(fake_mcp):
    fake_mcp.replies = [[("tavily_search", {"query": "loop"})]]

    result = await run_web_search_agent("Loop forever")

    assert result["error"].startswith("Web search agent failed:")
    assert len(fake_mcp.llm_requests) == Config.TAVILY_MAX_TOOL_ROUNDS
