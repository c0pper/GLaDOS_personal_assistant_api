from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
import os

# Must run before importing the bot: its Config reads the environment at import time.
# Values are forced (not defaulted) so tests never reach Telegram or a real GLaDOS API.
os.environ.update({
    "TELEGRAM_TOKEN": "123:test-token",
    "TELEGRAM_ALLOWED_CHAT_IDS": "42, 7",
    "GLADOS_API_URL": "http://glados.test",
})

import httpx
import pytest

import bot


############
# FAKE API #
############
class FakeApi:
    """Routes (method, path) to a handler returning an httpx.Response; records every request."""

    def __init__(self):
        self.routes: Dict[tuple, Callable[[httpx.Request], httpx.Response]] = {}
        self.requests: List[httpx.Request] = []

    def on(self, method: str, path: str, response: Any) -> None:
        self.routes[(method, path)] = response if callable(response) else (lambda request: response)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.routes.get((request.method, request.url.path))
        if handler is None:
            return httpx.Response(404, text=f"No fake route for {request.method} {request.url.path}")
        return handler(request)


@pytest.fixture
async def fake_api(monkeypatch) -> FakeApi:
    api = FakeApi()
    client = httpx.AsyncClient(base_url=bot.Config.GLADOS_API_URL, transport=httpx.MockTransport(api.handle))
    monkeypatch.setattr(bot, "api_client", client)
    yield api
    await client.aclose()


#################
# FAKE TELEGRAM #
#################
class FakeMessage:
    """The parts of telegram.Message the handlers use; replies are recorded instead of sent."""

    def __init__(self, text: Optional[str] = None, voice: Optional[SimpleNamespace] = None):
        self.text = text
        self.voice = voice
        self.audio = None
        self.replies: List[dict] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append({"text": text})

    async def reply_voice(self, voice: bytes, caption: Optional[str] = None) -> None:
        self.replies.append({"voice": voice, "caption": caption})


class FakeChat:
    def __init__(self, chat_id: int):
        self.id = chat_id
        self.actions: List[str] = []

    async def send_action(self, action: str) -> None:
        self.actions.append(action)


def make_update(text: Optional[str] = None, voice_bytes: Optional[bytes] = None, chat_id: int = 42):
    voice = None
    if voice_bytes is not None:
        async def download_as_bytearray():
            return bytearray(voice_bytes)

        async def get_file():
            return SimpleNamespace(download_as_bytearray=download_as_bytearray)

        voice = SimpleNamespace(mime_type="audio/ogg", get_file=get_file)

    return SimpleNamespace(
        message=FakeMessage(text=text, voice=voice),
        effective_chat=FakeChat(chat_id),
        effective_user=SimpleNamespace(username="tester"),
    )
