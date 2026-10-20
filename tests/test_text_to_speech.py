import asyncio
import json

import httpx

from src.config import Config
from src.services import text_to_speech
from src.services.text_to_speech import speak

TTS_PATH = "/api/tts_get_url"
ANNOUNCE_PATH = "/api/services/music_assistant/play_announcement"
AUDIO_URL = "http://ha.test/api/tts_proxy/reply.mp3"


def fake_speaker(fake_http) -> None:
    """Home Assistant renders the speech and Music Assistant accepts the announcement."""
    fake_http.on("POST", TTS_PATH, httpx.Response(200, json={"url": AUDIO_URL}))
    fake_http.on("POST", ANNOUNCE_PATH, httpx.Response(200, json=[]))


async def test_speak_renders_with_voice_then_announces_on_music_assistant(fake_http):
    fake_speaker(fake_http)

    assert await speak("Hello, test subject.") is True
    await asyncio.gather(*text_to_speech._announcements)

    render = fake_http.requests_to(TTS_PATH)[0]
    assert render.headers["Authorization"] == "Bearer ha-token"
    assert json.loads(render.content) == {
        "engine_id": Config.HOME_ASSISTANT_TTS_ENTITY,
        "message": "Hello, test subject.",
        "options": {"voice": Config.HOME_ASSISTANT_TTS_VOICE},
    }
    announce = fake_http.requests_to(ANNOUNCE_PATH)[0]
    assert json.loads(announce.content) == {"entity_id": Config.HOME_ASSISTANT_TTS_MEDIA_PLAYER, "url": AUDIO_URL}


async def test_render_failure_skips_announcement(fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(500, text="piper down"))

    assert await speak("Hello") is False
    assert fake_http.requests_to(ANNOUNCE_PATH) == []


async def test_speak_returns_before_the_announcement_finishes(fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(200, json={"url": AUDIO_URL}))
    release = asyncio.Event()

    async def slow_announcement(request):
        await release.wait()
        return httpx.Response(200, json=[])

    fake_http.on("POST", ANNOUNCE_PATH, slow_announcement)

    assert await speak("Hello") is True
    assert len(text_to_speech._announcements) == 1

    release.set()
    await asyncio.gather(*text_to_speech._announcements)
    assert text_to_speech._announcements == set()


async def test_announcement_failure_does_not_raise(fake_http):
    fake_http.on("POST", TTS_PATH, httpx.Response(200, json={"url": AUDIO_URL}))
    fake_http.on("POST", ANNOUNCE_PATH, httpx.Response(400, text="player unavailable"))

    assert await speak("Hello") is True
    await asyncio.gather(*text_to_speech._announcements)


async def test_transport_error_returns_false(fake_http):
    def raise_timeout(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    fake_http.on("POST", TTS_PATH, raise_timeout)

    assert await speak("Hello") is False
