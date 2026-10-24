from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from telegram import Chat, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import httpx


@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ["TELEGRAM_TOKEN"]
    # Anyone can message a bot: only these chats are answered, the others are logged and ignored.
    TELEGRAM_ALLOWED_CHAT_IDS: tuple = tuple(
        int(chat_id) for chat_id in os.environ["TELEGRAM_ALLOWED_CHAT_IDS"].split(",") if chat_id.strip()
    )
    GLADOS_API_URL: str = os.environ.get("GLADOS_API_URL", "http://localhost:8001")
    # Shared secret the API requires in the X-API-Key header.
    GLADOS_API_KEY: str = os.environ["GLADOS_API_KEY"]
    # A request can run several tool rounds (Home Assistant/Grocy agents time out at 90 seconds).
    API_TIMEOUT_SECONDS: float = 180.0
    # Telegram shows a chat action ("typing...") for about 5 seconds, so it is re-sent while waiting.
    CHAT_ACTION_REFRESH_SECONDS: float = 4.0
    MAX_CAPTION_LENGTH: int = 1024


UNREACHABLE_MESSAGE = "My brain is unreachable at the moment. Try again later."

api_client = httpx.AsyncClient(
    base_url=Config.GLADOS_API_URL,
    headers={"X-API-Key": Config.GLADOS_API_KEY},
    timeout=Config.API_TIMEOUT_SECONDS,
)


###########
# HELPERS #
###########
def conversation_id(update: Update) -> str:
    """One GLaDOS conversation (memory) per Telegram chat."""
    return f"telegram:{update.effective_chat.id}"


async def call_api(method: str, path: str, **kwargs) -> Optional[httpx.Response]:
    try:
        response = await api_client.request(method, path, **kwargs)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"GLaDOS API call failed ({method} {path}): {e!r}")
        return None

    return response


@asynccontextmanager
async def showing(chat: Chat, action: str) -> AsyncIterator[None]:
    """Keep a chat action ("typing...", "recording voice...") visible while the body runs."""

    async def send() -> None:
        try:
            await chat.send_action(action)
        except Exception as e:
            # Cosmetic only: never let it break the reply.
            logger.warning(f"Failed to send chat action: {e!r}")

    async def refresh() -> None:
        while True:
            await asyncio.sleep(Config.CHAT_ACTION_REFRESH_SECONDS)
            await send()

    await send()
    task = asyncio.create_task(refresh())
    try:
        yield
    finally:
        task.cancel()


############
# HANDLERS #
############
async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Oh. It's you. Send a message or a voice note, if you must.")


async def on_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response = await call_api("DELETE", f"/api/conversations/{conversation_id(update)}")
    if response is None:
        await update.message.reply_text(UNREACHABLE_MESSAGE)
        return

    await update.message.reply_text("Memory wiped. I already feel better.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    async with showing(update.effective_chat, ChatAction.TYPING):
        response = await call_api(
            "POST", "/api/start", json={"message": update.message.text, "conversation_id": conversation_id(update)}
        )

    await update.message.reply_text(response.json()["message"] if response else UNREACHABLE_MESSAGE)


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Voice notes go through the same path as the other voice clients: the API transcribes them, answers,
    and speaks the reply on the house speaker. The reply also comes back here as a voice note.
    """
    recording = update.message.voice or update.message.audio
    telegram_file = await recording.get_file()
    audio = bytes(await telegram_file.download_as_bytearray())

    async with showing(update.effective_chat, ChatAction.RECORD_VOICE):
        response = await call_api(
            "POST",
            "/api/start_audio",
            params={"conversation_id": conversation_id(update)},
            content=audio,
            headers={"Content-Type": recording.mime_type or "audio/ogg"},
        )
        if response is None:
            await update.message.reply_text(UNREACHABLE_MESSAGE)
            return

        result = response.json()
        if result["transcript"] is None:
            # Nothing was understood: the message is the error, no point in a voice reply.
            await update.message.reply_text(result["message"])
            return

        speech = await call_api("POST", "/api/speech", json={"text": result["message"]})

    # The transcript is shown so misheard requests are easy to spot.
    text = f"🎙 {result['transcript']}\n\n{result['message']}"
    if result.get("warning"):
        # The API had to use its fallback speech-to-text.
        text = f"⚠️ {result['warning']}\n\n{text}"
    if speech is None:
        await update.message.reply_text(text)
    elif len(text) <= Config.MAX_CAPTION_LENGTH:
        await update.message.reply_voice(speech.content, caption=text)
    else:
        await update.message.reply_voice(speech.content)
        await update.message.reply_text(text)


async def on_unauthorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.warning(f"Ignoring message from unauthorized chat {update.effective_chat.id} "
                   f"({user.username if user else 'unknown user'})")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.opt(exception=context.error).error(f"Failed to handle update: {context.error!r}")


#######
# APP #
#######
async def close_api_client(application: Application) -> None:
    await api_client.aclose()


def build_application() -> Application:
    application = Application.builder().token(Config.TELEGRAM_TOKEN).post_shutdown(close_api_client).build()
    allowed = filters.Chat(chat_id=Config.TELEGRAM_ALLOWED_CHAT_IDS)

    application.add_handler(CommandHandler("start", on_start, filters=allowed))
    application.add_handler(CommandHandler("reset", on_reset, filters=allowed))
    application.add_handler(MessageHandler(allowed & filters.TEXT & ~filters.COMMAND, on_text))
    application.add_handler(MessageHandler(allowed & (filters.VOICE | filters.AUDIO), on_voice))
    application.add_handler(MessageHandler(~allowed, on_unauthorized))
    application.add_error_handler(on_error)

    return application


if __name__ == "__main__":
    logger.info(f"Starting GLaDOS Telegram bot for chats {Config.TELEGRAM_ALLOWED_CHAT_IDS} -> {Config.GLADOS_API_URL}")
    # Long polling: only outbound connections, no public URL needed.
    build_application().run_polling(allowed_updates=Update.ALL_TYPES)
