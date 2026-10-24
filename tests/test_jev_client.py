import json

import httpx
import pytest

from src.config import Config
from src.services import jev_client


@pytest.fixture
async def jev_http(monkeypatch):
    """Replaces the Jev HTTP client with a mock transport; tests queue responses."""
    requests = []
    responses = []

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle), timeout=1)
    monkeypatch.setattr(jev_client, "_client", client)
    yield requests, responses
    await client.aclose()


async def test_decide_posts_state_and_questions(jev_http):
    requests, responses = jev_http
    responses.append(httpx.Response(200, json={
        "answers": {"tool": {"type": "choice", "choice": "No Tool", "confidence": 0.9}},
        "model": "typesafe/jev-1.14-20261014",
        "usage": {"input_tokens": 100, "output_tokens": 1, "cost": 0.0000042},
    }))

    response = await jev_client.decide(state={"message": "hello"}, questions={"tool": {"type": "choice"}})

    assert response.answers["tool"]["choice"] == "No Tool"
    assert response.usage.cost == 0.0000042

    body = json.loads(requests[0].content)
    assert body["model"] == Config.JEV_MODEL
    assert body["state"] == {"message": "hello"}
    assert body["questions"] == {"tool": {"type": "choice"}}
    assert requests[0].headers["authorization"] == f"Bearer {Config.OPENROUTER_API_KEY}"


async def test_decide_raises_jev_error_on_http_error(jev_http):
    _, responses = jev_http
    responses.append(httpx.Response(500, text="boom"))

    with pytest.raises(jev_client.JevError):
        await jev_client.decide(state={}, questions={})


async def test_decide_raises_jev_error_on_non_json(jev_http):
    _, responses = jev_http
    responses.append(httpx.Response(200, text="not json"))

    with pytest.raises(jev_client.JevError):
        await jev_client.decide(state={}, questions={})


async def test_circuit_breaker_skips_requests_after_repeated_failures(jev_http):
    requests, responses = jev_http
    for _ in range(jev_client.FAILURE_THRESHOLD):
        responses.append(httpx.Response(500))
        with pytest.raises(jev_client.JevError):
            await jev_client.decide(state={}, questions={})

    with pytest.raises(jev_client.JevError):
        await jev_client.decide(state={}, questions={})

    assert len(requests) == jev_client.FAILURE_THRESHOLD
