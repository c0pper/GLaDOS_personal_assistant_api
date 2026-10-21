from src.agents.grocy_agent import run_grocy_agent
from src.config import Config


async def test_uses_grocy_server_with_only_its_tools(fake_mcp):
    fake_mcp.replies = ["Milk expires tomorrow."]

    result = await run_grocy_agent("What is expiring soon?")

    assert result == {"response": "Milk expires tomorrow."}
    assert fake_mcp.session_urls == [Config.GROCY_MCP_URL]
    offered = {tool["function"]["name"] for tool in fake_mcp.llm_requests[0]["tools"]}
    assert offered == set(Config.GROCY_MCP_TOOLS)
    assert fake_mcp.llm_requests[0]["model"] == Config.GROCY_AGENT_MODEL
    assert "Grocy" in fake_mcp.llm_requests[0]["messages"][0]["content"]


async def test_adds_to_shopping_list_through_tools(fake_mcp):
    fake_mcp.tool_results = {"stock_search_tool": "Milk (id 7)"}
    fake_mcp.replies = [
        [("stock_search_tool", {"query": "milk"})],
        [("shopping_list_add_tool", {"product_id": 7, "amount": 2})],
        "Added 2 milk to the shopping list.",
    ]

    result = await run_grocy_agent("Add two milk to the shopping list")

    assert result == {"response": "Added 2 milk to the shopping list."}
    assert fake_mcp.tool_calls == [
        ("stock_search_tool", {"query": "milk"}),
        ("shopping_list_add_tool", {"product_id": 7, "amount": 2}),
    ]


async def test_home_assistant_tools_are_refused(fake_mcp):
    fake_mcp.replies = [[("ha_call_service", {})], "Not available."]

    await run_grocy_agent("Turn on the lights")

    assert fake_mcp.tool_calls == []
    assert fake_mcp.llm_requests[1]["messages"][-1]["content"] == "Tool error: 'ha_call_service' is not available."


async def test_errors_name_the_agent(fake_mcp):
    fake_mcp.replies = [[("stock_overview_tool", {})]]

    result = await run_grocy_agent("Loop forever")

    assert result["error"].startswith("Grocy agent failed:")
    assert len(fake_mcp.llm_requests) == Config.GROCY_MAX_TOOL_ROUNDS
