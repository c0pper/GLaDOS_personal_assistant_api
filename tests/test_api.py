import asyncio
import time

import httpx
import pytest

from src.agents.glados_responder_agent import GladosResponderOutputSchema
from src.agents.orchestrator_agent import OrchestratorOutputSchema
from src.agents.vikunja_agent import VikunjaOutputSchema
from src.main import app
from src.services.llm_cache.llm_cache import cache
from tests.test_speech_to_text import STT_PATH, make_wav

@pytest.fixture
async def api():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


def route_by_keyword(user_input):
    message = user_input.chat_message.lower()
    if "light" in message:
        return OrchestratorOutputSchema(tool_name="Home Assistant Tool")
    if "task" in message:
        return OrchestratorOutputSchema(tool_name="Vikunja Tool")
    if "search" in message:
        return OrchestratorOutputSchema(tool_name="SearXNG Tool")
    return OrchestratorOutputSchema(tool_name="No Tool")


def echo_tool_result(user_input):
    return GladosResponderOutputSchema(final_response=f"GLaDOS: {user_input.tool_result}")


@pytest.fixture
def pipeline(fake_llm, fake_http, fake_ha):
    fake_llm.on(OrchestratorOutputSchema, route_by_keyword)
    fake_llm.on(GladosResponderOutputSchema, echo_tool_result)
    fake_llm.on(VikunjaOutputSchema, lambda _: VikunjaOutputSchema(action="get_tasks"))
    fake_ha.replies = ["Turned on the mood lights"]
    fake_http.on("GET", "/api/v1/projects", httpx.Response(200, json=[{"id": 1, "title": "Personal"}]))
    fake_http.on("GET", "/api/v1/tasks", httpx.Response(200, json=[{"title": "Buy milk", "description": "now"}]))
    return fake_llm, fake_http


##########
# /start #
##########
async def test_home_assistant_flow(api, pipeline, fake_ha):
    fake_llm, _ = pipeline

    response = await api.post("/api/start", json={"message": "Turn on the mood lights"})

    assert response.status_code == 200
    assert response.json() == {"message": "GLaDOS: Turned on the mood lights"}
    assert [call.output_schema for call in fake_llm.calls] == [OrchestratorOutputSchema, GladosResponderOutputSchema]
    assert len(fake_ha.llm_requests) == 1


async def test_home_assistant_error_is_returned(api, pipeline, fake_ha):
    async def fail(**_):
        raise RuntimeError("boom")

    fake_ha.create = fail

    response = await api.post("/api/start", json={"message": "Turn on the mood lights"})

    assert response.json() == {"message": "Home Assistant agent failed: RuntimeError('boom')"}


async def test_llm_failure_returns_message_instead_of_500(api, pipeline):
    fake_llm, _ = pipeline

    def rate_limited(_):
        raise RuntimeError("429 rate-limited upstream")

    fake_llm.on(OrchestratorOutputSchema, rate_limited)

    response = await api.post("/api/start", json={"message": "Hello GLaDOS"})

    assert response.status_code == 200
    assert response.json() == {"message": "Sorry, my brain is temporarily unavailable. Try again in a moment."}


async def test_vikunja_flow(api, pipeline):
    response = await api.post("/api/start", json={"message": "What are my pending tasks?"})

    assert response.json() == {"message": "GLaDOS: Pending Tasks: \n\nBuy milk - now"}


async def test_no_tool_goes_straight_to_responder(api, pipeline):
    fake_llm, fake_http = pipeline

    response = await api.post("/api/start", json={"message": "Hello GLaDOS"})

    assert response.json() == {"message": "GLaDOS: None"}
    assert [call.output_schema for call in fake_llm.calls] == [OrchestratorOutputSchema, GladosResponderOutputSchema]
    assert fake_http.requests == []


async def test_unsupported_tool_returns_sorry(api, pipeline):
    fake_llm, _ = pipeline

    response = await api.post("/api/start", json={"message": "Search the web for cake"})

    assert response.json() == {"message": "Sorry, I don't know how to help you with that."}
    assert [call.output_schema for call in fake_llm.calls] == [OrchestratorOutputSchema]


async def test_repeat_query_reuses_route_but_not_stale_reply(api, pipeline, fake_ha):
    fake_llm, _ = pipeline
    fake_ha.replies = ["It is 20 degrees", "It is 25 degrees"]

    first = await api.post("/api/start", json={"message": "What's the light temperature?"})
    second = await api.post("/api/start", json={"message": "What's the light temperature?"})

    assert first.json() == {"message": "GLaDOS: It is 20 degrees"}
    assert second.json() == {"message": "GLaDOS: It is 25 degrees"}
    assert len(fake_llm.calls_for(OrchestratorOutputSchema)) == 1
    assert len(fake_llm.calls_for(GladosResponderOutputSchema)) == 2


async def test_concurrent_requests_do_not_block_each_other(api, pipeline):
    fake_llm, _ = pipeline
    original = fake_llm.run_async
    delay = 0.3

    async def slow_run_async(agent, user_input):
        await asyncio.sleep(delay)
        return await original(agent, user_input)

    fake_llm.run_async = slow_run_async
    messages = ["Turn on the lights", "Show my tasks", "Hello"]

    start = time.perf_counter()
    responses = await asyncio.gather(*(api.post("/api/start", json={"message": m}) for m in messages))
    elapsed = time.perf_counter() - start

    assert all(response.status_code == 200 for response in responses)
    # Each request makes 2-3 LLM calls; run one after another this would take ~7 * delay.
    assert elapsed < 4 * delay


################
# /start_audio #
################
async def test_audio_is_transcribed_then_answered(api, pipeline):
    fake_llm, fake_http = pipeline
    fake_http.on("POST", STT_PATH, httpx.Response(200, json={"text": "Turn on the mood lights", "result": "success"}))

    response = await api.post("/api/start_audio", content=make_wav(), headers={"Content-Type": "audio/wav"})

    assert response.json() == {"message": "GLaDOS: Turned on the mood lights", "transcript": "Turn on the mood lights"}
    assert fake_llm.calls[0].user_input.chat_message == "Turn on the mood lights"


async def test_audio_transcription_error_skips_pipeline(api, pipeline):
    fake_llm, fake_http = pipeline
    fake_http.on("POST", STT_PATH, httpx.Response(200, json={"text": None, "result": "error"}))

    response = await api.post("/api/start_audio", content=make_wav())

    assert response.json() == {"message": "Sorry, I couldn't understand the recording.", "transcript": None}
    assert fake_llm.calls == []


async def test_empty_audio_body(api, pipeline):
    response = await api.post("/api/start_audio", content=b"")

    assert response.json() == {"message": "No audio received."}


###################
# /start_rhasspy #
###################
async def test_rhasspy_uses_raw_text(api, pipeline):
    response = await api.post("/api/start_rhasspy", json={"raw_text": "Hello GLaDOS", "intent": {"name": "x"}})

    assert response.json() == {"message": "GLaDOS: None"}


################
# /cache/wipe #
################
async def test_cache_wipe(api):
    cache.set("orchestrator", "q", {"tool_name": "No Tool"})

    response = await api.post("/api/cache/wipe")

    assert response.json() == {"message": "Cache wiped successfully."}
    assert cache.get("orchestrator", "q") is None
