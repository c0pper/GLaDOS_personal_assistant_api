import json

import httpx

from src.agents.home_assistant_agent import (
    CACHE_NAMESPACE,
    HomeAssistantOutputSchema,
    IntentName,
    invoke_intent,
    run_home_assistant_agent,
)
from src.services.llm_cache.llm_cache import cache

INTENT_PATH = "/api/intent/handle"
SPEECH = {"speech": {"plain": {"speech": "Turned on the lights"}}}


async def test_invoke_intent_success(fake_http):
    fake_http.on("POST", INTENT_PATH, httpx.Response(200, json=SPEECH))

    result = await invoke_intent("TurnOnMoodLights")

    assert result == SPEECH
    request = fake_http.requests_to(INTENT_PATH)[0]
    assert json.loads(request.content) == {"name": "TurnOnMoodLights"}
    assert request.headers["Authorization"] == "Bearer ha-token"


async def test_invoke_intent_http_error_status(fake_http):
    fake_http.on("POST", INTENT_PATH, httpx.Response(500, text="boom"))

    result = await invoke_intent("TurnOnMoodLights")

    assert result == {"error": "Failed to invoke intent: 500 - boom"}


async def test_invoke_intent_transport_error(fake_http):
    def raise_timeout(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    fake_http.on("POST", INTENT_PATH, raise_timeout)

    result = await invoke_intent("TurnOnMoodLights")

    assert result["error"].startswith("Failed to invoke intent:")


async def test_known_intent_skips_llm(fake_llm, fake_http):
    fake_http.on("POST", INTENT_PATH, httpx.Response(200, json=SPEECH))

    result = await run_home_assistant_agent("Turn on the mood lights", "TurnOnMoodLights")

    assert result == SPEECH
    assert fake_llm.calls == []


async def test_missing_intent_falls_back_to_llm_and_caches(fake_llm, fake_http):
    fake_llm.on(HomeAssistantOutputSchema, lambda _: HomeAssistantOutputSchema(intent_name=IntentName.GetTemperature))
    fake_http.on("POST", INTENT_PATH, httpx.Response(200, json=SPEECH))

    await run_home_assistant_agent("What is the temperature?")
    await run_home_assistant_agent("What is the temperature?")

    assert len(fake_llm.calls) == 1
    assert cache.get(CACHE_NAMESPACE, "What is the temperature?") == {"intent_name": "GetTemperature"}
    sent = [json.loads(request.content) for request in fake_http.requests_to(INTENT_PATH)]
    assert sent == [{"name": "GetTemperature"}, {"name": "GetTemperature"}]
