from typing import Optional, Set
import asyncio
import io

import av
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


async def render(text: str) -> Optional[str]:
    """
    Render `text` with the configured TTS entity and voice; returns the URL of the audio on Home Assistant.
    Home Assistant caches renders, so rendering the same reply again (speaker + Telegram) is cheap.
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
        return None

    return rendered.json()["url"]


#############
# SPEAKER #
#############
async def speak(text: str) -> bool:
    """
    Say `text` on the configured Music Assistant speaker as an announcement: music playing there is
    lowered/paused and resumed afterwards, instead of being replaced like a plain tts.speak would do.
    The audio is rendered first with the configured TTS entity and voice, so the voice doesn't depend on
    Music Assistant's own announcement settings.
    Returns False if the speech could not be rendered (the text reply is still usable without audio);
    a failure of the announcement itself only shows up in the logs.
    """
    url = await render(text)
    if url is None:
        return False

    # Music Assistant only answers once the announcement has finished playing (chime included, 10+ seconds),
    # so it runs in the background and the caller gets its reply right away.
    task = asyncio.create_task(announce(url))
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


###############
# VOICE NOTE #
###############
def to_ogg_opus(audio: bytes) -> bytes:
    """Transcode any audio ffmpeg decodes to mono OGG/Opus, the format Telegram shows as a voice note."""
    output = io.BytesIO()

    with av.open(io.BytesIO(audio)) as source, av.open(output, "w", format="ogg") as target:
        # The encoder resamples and re-frames its input to what Opus needs.
        stream = target.add_stream("libopus", rate=48000, layout="mono")
        for frame in source.decode(audio=0):
            frame.pts = None
            target.mux(stream.encode(frame))
        target.mux(stream.encode(None))

    return output.getvalue()


async def synthesize(text: str) -> Optional[bytes]:
    """`text` spoken with the configured voice, as OGG/Opus; None if it could not be rendered."""
    url = await render(text)
    if url is None:
        return None

    try:
        # Absolute URL on Home Assistant: the client's base URL is ignored, its token is still sent.
        response = await home_assistant_client.get(url)
        response.raise_for_status()
        return await asyncio.to_thread(to_ogg_opus, response.content)
    except (httpx.HTTPError, av.FFmpegError, ValueError) as e:
        logger.error(f"Failed to fetch rendered speech ({url}): {e!r}")
        return None
