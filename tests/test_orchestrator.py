from src.agents.orchestrator_agent import CACHE_NAMESPACE, OrchestratorOutputSchema, route_query
from src.services.llm_cache.llm_cache import cache


async def test_cache_miss_calls_llm_and_caches(fake_llm):
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="Home Assistant Tool"))

    result = await route_query("Turn on the mood lights")

    assert result.tool_name == "Home Assistant Tool"
    assert len(fake_llm.calls) == 1
    assert cache.get(CACHE_NAMESPACE, "Turn on the mood lights") == {
        "tool_name": "Home Assistant Tool",
        "standalone_query": None,
    }


async def test_cache_hit_skips_llm(fake_llm):
    cache.set(CACHE_NAMESPACE, "hello", {"tool_name": "No Tool"})

    result = await route_query("Hello")

    assert result.tool_name == "No Tool"
    assert fake_llm.calls == []


async def test_invalid_cache_entry_is_ignored(fake_llm):
    cache.set(CACHE_NAMESPACE, "hello", {"tool_name": "Removed Tool"})
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="No Tool"))

    result = await route_query("hello")

    assert result.tool_name == "No Tool"
    assert len(fake_llm.calls) == 1


async def test_every_call_starts_with_empty_history(fake_llm):
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="No Tool"))

    await route_query("first query")
    await route_query("second query")

    assert [call.history_length for call in fake_llm.calls] == [0, 0]


async def test_system_prompt_has_date(fake_llm):
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="No Tool"))

    await route_query("anything")

    prompt = fake_llm.calls[0].system_prompt
    assert "Current date in format YYYY-MM-DD" in prompt


async def test_history_is_passed_and_keys_the_cache(fake_llm):
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(
        tool_name="Home Assistant Tool", standalone_query="Turn off the kitchen lights"
    ))

    result = await route_query("Turn it off", "User: Turn on the kitchen lights\nGLaDOS: Done.")
    # Same message without that history must not reuse the follow-up's route.
    await route_query("Turn it off")

    assert result.standalone_query == "Turn off the kitchen lights"
    assert fake_llm.calls[0].user_input.conversation_history == "User: Turn on the kitchen lights\nGLaDOS: Done."
    assert fake_llm.calls[1].user_input.conversation_history is None
    assert len(fake_llm.calls) == 2
