from src.agents.home_assistant_agent import MAX_TOOL_RESULT_CHARS, run_home_assistant_agent
from src.config import Config


async def test_direct_answer_without_tools(fake_ha):
    fake_ha.replies = ["Hello."]

    result = await run_home_assistant_agent("Hi")

    assert result == {"response": "Hello."}
    assert fake_ha.tool_calls == []


async def test_tool_calls_are_executed_and_fed_back(fake_ha):
    fake_ha.tool_results = {"ha_search": "light.mood: off"}
    fake_ha.replies = [
        [("ha_search", {"query": "mood"})],
        [("ha_call_service", {"domain": "light", "service": "turn_on", "entity_id": "light.mood"})],
        "Mood lights are on.",
    ]

    result = await run_home_assistant_agent("Turn on the mood lights")

    assert result == {"response": "Mood lights are on."}
    assert fake_ha.tool_calls == [
        ("ha_search", {"query": "mood"}),
        ("ha_call_service", {"domain": "light", "service": "turn_on", "entity_id": "light.mood"}),
    ]
    second_request = fake_ha.llm_requests[1]["messages"]
    assert second_request[-1] == {"role": "tool", "tool_call_id": "call-0", "content": "light.mood: off"}
    assert second_request[-2]["tool_calls"][0]["function"]["name"] == "ha_search"


async def test_only_allowed_tools_are_offered(fake_ha):
    await run_home_assistant_agent("anything")

    offered = {tool["function"]["name"] for tool in fake_ha.llm_requests[0]["tools"]}
    assert offered == set(Config.HOME_ASSISTANT_MCP_TOOLS)
    assert fake_ha.llm_requests[0]["model"] == Config.HOME_ASSISTANT_AGENT_MODEL


async def test_disallowed_tool_is_refused(fake_ha):
    fake_ha.replies = [[("ha_restart", {})], "Cannot restart."]

    result = await run_home_assistant_agent("Restart Home Assistant")

    assert result == {"response": "Cannot restart."}
    assert fake_ha.tool_calls == []
    assert fake_ha.llm_requests[1]["messages"][-1]["content"] == "Tool error: 'ha_restart' is not available."


async def test_long_tool_results_are_truncated(fake_ha):
    fake_ha.tool_results = {"ha_search": "x" * (MAX_TOOL_RESULT_CHARS + 100)}
    fake_ha.replies = [[("ha_search", {})], "Done."]

    await run_home_assistant_agent("Search everything")

    content = fake_ha.llm_requests[1]["messages"][-1]["content"]
    assert content.endswith("[truncated]")
    assert len(content) == MAX_TOOL_RESULT_CHARS + len("\n[truncated]")


async def test_gives_up_after_max_rounds(fake_ha):
    fake_ha.replies = [[("ha_search", {})]]

    result = await run_home_assistant_agent("Loop forever")

    assert result["error"].startswith("Home Assistant agent failed:")
    assert len(fake_ha.llm_requests) == Config.HOME_ASSISTANT_MAX_TOOL_ROUNDS
