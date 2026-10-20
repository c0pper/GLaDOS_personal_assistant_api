from typing import Optional, Set
import asyncio

import httpx

from src.config import Config
from src.logger import logger
from src.services.http_clients import home_assistant_client


async def call_home_assistant(path: str, payload: dict, timeout: Optional[float] = None) -> Optional[httpx.Response]:
    try:
        response = await home_assistant_client.post(
            path, json=payload, timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else timeout
        )
    except httpx.HTTPError as e:
        logger.error(f"Failed to speak reply ({path}): {e!r}")
        return None

    if response.status_code != 200:
        logger.error(f"Failed to speak reply ({path}): {response.status_code} - {response.text}")
        return None

    return response


async def speak(text: str) -> bool:
    """
    Say `text` on the configured Music Assistant speaker as an announcement: music playing there is
    lowered/paused and resumed afterwards, instead of being replaced like a plain tts.speak would do.
    The audio is rendered first with the configured TTS entity and voice, so the voice doesn't depend on
    Music Assistant's own announcement settings.
    Returns False if the speech could not be rendered (the text reply is still usable without audio);
    a failure of the announcement itself only shows up in the logs.
    """
    rendered = await call_home_assistant(
        "/api/tts_get_url",
        {
            "engine_id": Config.HOME_ASSISTANT_TTS_ENTITY,
            "message": text,
            "options": {"voice": Config.HOME_ASSISTANT_TTS_VOICE},
        },
    )
    if rendered is None:
        return False

    # Music Assistant only answers once the announcement has finished playing (chime included, 10+ seconds),
    # so it runs in the background and the caller gets its reply right away.
    task = asyncio.create_task(announce(rendered.json()["url"]))
    _announcements.add(task)
    task.add_done_callback(_announcements.discard)
    return True


# Strong references to running announcements: the event loop only keeps weak ones.
_announcements: Set[asyncio.Task] = set()


async def announce(url: str) -> None:
    # Longer timeout than other Home Assistant calls: the request lasts as long as the announcement.
    response = await call_home_assistant(
        "/api/services/music_assistant/play_announcement",
        {"entity_id": Config.HOME_ASSISTANT_TTS_MEDIA_PLAYER, "url": url},
        timeout=Config.ANNOUNCEMENT_TIMEOUT_SECONDS,
    )
    if response is not None:
        logger.info(f"Reply announced on {Config.HOME_ASSISTANT_TTS_MEDIA_PLAYER}")
