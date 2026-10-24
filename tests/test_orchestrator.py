from typing import Optional

import httpx
import pytest

from src.agents.orchestrator_agent import CACHE_NAMESPACE, OrchestratorOutputSchema, route_query
from src.config import Config
from src.services import jev_client
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


###################
# JEV FAST ROUTING #
###################
def jev_response(
    tool: str, confidence: float = 0.95, needs_rewrite: Optional[float] = None
) -> jev_client.JevResponse:
    answers = {"tool": {"type": "choice", "choice": tool, "confidence": confidence}}
    if needs_rewrite is not None:
        answers["needs_rewrite"] = {"type": "noul", "noul": needs_rewrite}
    return jev_client.JevResponse(answers=answers, model="typesafe/jev-1.14-test")


@pytest.fixture
def jev_enabled(monkeypatch):
    monkeypatch.setattr(Config, "JEV_ENABLED", True)


async def test_jev_high_confidence_route_skips_llm(fake_llm, fake_jev, jev_enabled):
    fake_jev.response = jev_response("Home Assistant Tool")

    result = await route_query("Turn on the mood lights")

    assert result.tool_name == "Home Assistant Tool"
    assert result.standalone_query is None
    assert fake_llm.calls == []
    assert len(fake_jev.calls) == 1
    # Without history there is nothing to rewrite, so Jev is asked only about the tool.
    assert "needs_rewrite" not in fake_jev.calls[0]["questions"]
    assert fake_jev.calls[0]["state"]["message"] == "Turn on the mood lights"
    assert "current_date" in fake_jev.calls[0]["state"]


async def test_route_query_uses_jev_response_from_http(fake_llm, monkeypatch, jev_enabled):
    """End-to-end: route_query -> jev_client -> Decisions API response -> OrchestratorOutputSchema."""
    def handle(request):
        assert request.url.path == "/api/alpha/decisions"
        return httpx.Response(200, json={
            "answers": {"tool": {"type": "choice", "choice": "Grocy Tool", "confidence": 0.93}},
            "model": "typesafe/jev-1.14-test",
            "usage": {"input_tokens": 120, "output_tokens": 1, "cost": 0.000005},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle), timeout=1)
    monkeypatch.setattr(jev_client, "_client", client)

    try:
        result = await route_query("Add milk to the shopping list")
    finally:
        await client.aclose()

    assert result.tool_name == "Grocy Tool"
    assert fake_llm.calls == []


async def test_jev_low_confidence_falls_back_to_llm(fake_llm, fake_jev, jev_enabled):
    fake_jev.response = jev_response("Home Assistant Tool", confidence=0.4)
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="No Tool"))

    result = await route_query("Turn on the mood lights")

    assert result.tool_name == "No Tool"
    assert len(fake_llm.calls) == 1


async def test_jev_error_falls_back_to_llm(fake_llm, fake_jev, jev_enabled):
    fake_jev.error = jev_client.JevError("down")
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="No Tool"))

    result = await route_query("Hello")

    assert result.tool_name == "No Tool"
    assert len(fake_llm.calls) == 1


async def test_jev_no_tool_needs_higher_confidence(fake_llm, fake_jev, jev_enabled):
    fake_jev.response = jev_response("No Tool", confidence=0.8)
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="Home Assistant Tool"))

    result = await route_query("Hello")

    assert result.tool_name == "Home Assistant Tool"
    assert len(fake_llm.calls) == 1


async def test_jev_confident_no_tool_skips_llm(fake_llm, fake_jev, jev_enabled):
    fake_jev.response = jev_response("No Tool", confidence=0.95)

    result = await route_query("Hello")

    assert result.tool_name == "No Tool"
    assert fake_llm.calls == []


async def test_jev_self_contained_message_with_history_skips_llm(fake_llm, fake_jev, jev_enabled):
    fake_jev.response = jev_response("Home Assistant Tool", needs_rewrite=0.05)

    result = await route_query("Turn off the kitchen lights", "User: Turn on the kitchen lights\nGLaDOS: Done.")

    assert result.tool_name == "Home Assistant Tool"
    assert result.standalone_query is None
    assert fake_llm.calls == []
    assert fake_jev.calls[0]["questions"]["needs_rewrite"]["type"] == "noul"
    assert fake_jev.calls[0]["state"]["conversation_history"] == "User: Turn on the kitchen lights\nGLaDOS: Done."


async def test_jev_follow_up_falls_back_to_llm_for_rewrite(fake_llm, fake_jev, jev_enabled):
    fake_jev.response = jev_response("Home Assistant Tool", needs_rewrite=0.9)
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(
        tool_name="Home Assistant Tool", standalone_query="Turn off the kitchen lights"
    ))

    result = await route_query("Turn it off", "User: Turn on the kitchen lights\nGLaDOS: Done.")

    assert result.standalone_query == "Turn off the kitchen lights"
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0].user_input.conversation_history == "User: Turn on the kitchen lights\nGLaDOS: Done."


async def test_jev_unusable_answer_falls_back_to_llm(fake_llm, fake_jev, jev_enabled):
    fake_jev.response = jev_client.JevResponse(answers={"tool": {"type": "noul", "noul": 0.9}})
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="No Tool"))

    result = await route_query("Hello")

    assert result.tool_name == "No Tool"
    assert len(fake_llm.calls) == 1


async def test_cache_hit_skips_jev(fake_llm, fake_jev, jev_enabled):
    cache.set(CACHE_NAMESPACE, "hello", {"tool_name": "No Tool"})

    result = await route_query("Hello")

    assert result.tool_name == "No Tool"
    assert fake_jev.calls == []
    assert fake_llm.calls == []


async def test_jev_disabled_by_default_uses_llm(fake_llm, fake_jev):
    fake_llm.on(OrchestratorOutputSchema, lambda _: OrchestratorOutputSchema(tool_name="No Tool"))

    result = await route_query("Hello")

    assert result.tool_name == "No Tool"
    assert fake_jev.calls == []
    assert len(fake_llm.calls) == 1
