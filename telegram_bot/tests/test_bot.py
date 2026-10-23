from datetime import datetime, timezone
import json

from telegram import Chat, Message, Update, Voice
import httpx

import bot
from tests.conftest import make_update

VOICE_REPLY = b"OggS fake voice reply"


########
# TEXT #
########
async def test_text_goes_to_start_with_the_chat_conversation(fake_api):
    fake_api.on("POST", "/api/start", httpx.Response(200, json={"message": "Lights on. Thrilling."}))
    update = make_update(text="Turn on the lights")

    await bot.on_text(update, None)

    assert json.loads(fake_api.requests[0].content) == {
        "message": "Turn on the lights",
        "conversation_id": "telegram:42",
    }
    assert update.message.replies == [{"text": "Lights on. Thrilling."}]
    assert update.effective_chat.actions == ["typing"]


async def test_text_when_api_is_down(fake_api):
    fake_api.on("POST", "/api/start", httpx.Response(500))
    update = make_update(text="Hello")

    await bot.on_text(update, None)

    assert update.message.replies == [{"text": bot.UNREACHABLE_MESSAGE}]


#########
# VOICE #
#########
def answer_voice(fake_api, transcript="Turn on the lights", message="Lights on. Thrilling."):
    fake_api.on("POST", "/api/start_audio", httpx.Response(200, json={
        "message": message, "transcript": transcript, "spoken": True,
    }))
    fake_api.on("POST", "/api/speech", httpx.Response(200, content=VOICE_REPLY, headers={"Content-Type": "audio/ogg"}))


async def test_voice_is_answered_with_a_voice_note(fake_api):
    answer_voice(fake_api)
    update = make_update(voice_bytes=b"recording")

    await bot.on_voice(update, None)

    start_audio, speech = fake_api.requests
    assert start_audio.content == b"recording"
    assert start_audio.url.params["conversation_id"] == "telegram:42"
    assert start_audio.headers["Content-Type"] == "audio/ogg"
    assert json.loads(speech.content) == {"text": "Lights on. Thrilling."}
    assert update.message.replies == [{
        "voice": VOICE_REPLY,
        "caption": "🎙 Turn on the lights\n\nLights on. Thrilling.",
    }]


async def test_voice_reply_falls_back_to_text_without_speech(fake_api):
    answer_voice(fake_api)
    fake_api.on("POST", "/api/speech", httpx.Response(502, json={"message": "Failed to render speech."}))
    update = make_update(voice_bytes=b"recording")

    await bot.on_voice(update, None)

    assert update.message.replies == [{"text": "🎙 Turn on the lights\n\nLights on. Thrilling."}]


async def test_long_voice_reply_is_sent_as_voice_then_text(fake_api):
    answer_voice(fake_api, message="blah " * 300)
    update = make_update(voice_bytes=b"recording")

    await bot.on_voice(update, None)

    voice, text = update.message.replies
    assert voice == {"voice": VOICE_REPLY, "caption": None}
    assert text["text"].endswith("blah ")


async def test_unrecognized_voice_gets_the_error_as_text(fake_api):
    answer_voice(fake_api, transcript=None, message="Sorry, I couldn't understand the recording.")
    update = make_update(voice_bytes=b"recording")

    await bot.on_voice(update, None)

    assert update.message.replies == [{"text": "Sorry, I couldn't understand the recording."}]
    assert [request.url.path for request in fake_api.requests] == ["/api/start_audio"]


############
# COMMANDS #
############
async def test_reset_clears_the_chat_conversation(fake_api):
    fake_api.on("DELETE", "/api/conversations/telegram:42", httpx.Response(200, json={"message": "cleared"}))
    update = make_update(text="/reset")

    await bot.on_reset(update, None)

    assert update.message.replies == [{"text": "Memory wiped. I already feel better."}]


##########
# ACCESS #
##########
def handled_by(chat_id: int, **message_fields) -> list:
    """The callbacks of the handlers that accept a real Telegram update from `chat_id`."""
    message = Message(
        message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=chat_id, type=Chat.PRIVATE), **message_fields
    )
    update = Update(update_id=1, message=message)
    handlers = bot.build_application().handlers[0]
    return [handler.callback for handler in handlers if handler.check_update(update)]


def test_allowed_chat_ids_are_parsed():
    assert bot.Config.TELEGRAM_ALLOWED_CHAT_IDS == (42, 7)


def test_allowed_chat_messages_reach_the_handlers():
    assert handled_by(42, text="Hello") == [bot.on_text]
    assert handled_by(7, voice=Voice("id", "unique-id", duration=2)) == [bot.on_voice]


def test_other_chats_are_only_logged():
    assert handled_by(999, text="Hello") == [bot.on_unauthorized]
    assert handled_by(999, voice=Voice("id", "unique-id", duration=2)) == [bot.on_unauthorized]
