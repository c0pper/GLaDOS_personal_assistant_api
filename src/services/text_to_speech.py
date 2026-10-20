import httpx

from src.config import Config
from src.logger import logger
from src.services.http_clients import home_assistant_client


async def speak(text: str) -> bool:
    """
    Have Home Assistant say `text` on the configured speaker (tts.speak with the configured TTS entity and voice).
    Returns False on failure: the text reply is still usable without audio.
    """
    try:
        response = await home_assistant_client.post(
            "/api/services/tts/speak",
            json={
                "entity_id": Config.HOME_ASSISTANT_TTS_ENTITY,
                "media_player_entity_id": Config.HOME_ASSISTANT_TTS_MEDIA_PLAYER,
                "message": text,
                "options": {"voice": Config.HOME_ASSISTANT_TTS_VOICE},
            },
        )
    except httpx.HTTPError as e:
        logger.error(f"Failed to speak reply: {e!r}")
        return False

    if response.status_code != 200:
        logger.error(f"Failed to speak reply: {response.status_code} - {response.text}")
        return False

    logger.info(f"Reply spoken on {Config.HOME_ASSISTANT_TTS_MEDIA_PLAYER}")
    return True
