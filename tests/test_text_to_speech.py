import json

import httpx

from src.config import Config
from src.services.text_to_speech import speak

TTS_PATH = "/api/services/tts/speak"


async def test_speak_calls_tts_speak_on_configured_speaker(fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(200, json=[]))

    assert await speak("Hello, test subject.") is True

    request = fake_http.requests_to(TTS_PATH)[0]
    assert request.headers["Authorization"] == "Bearer ha-token"
    assert json.loads(request.content) == {
        "entity_id": Config.HOME_ASSISTANT_TTS_ENTITY,
        "media_player_entity_id": Config.HOME_ASSISTANT_TTS_MEDIA_PLAYER,
        "message": "Hello, test subject.",
        "options": {"voice": Config.HOME_ASSISTANT_TTS_VOICE},
    }


async def test_speak_failure_returns_false(fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(500, text="boom"))

    assert await speak("Hello") is False


async def test_speak_transport_error_returns_false(fake_http):
    def raise_timeout(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    fake_http.on("POST", TTS_PATH, raise_timeout)

    assert await speak("Hello") is False
