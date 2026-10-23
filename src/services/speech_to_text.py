from datetime import datetime
from pathlib import Path
import asyncio
import io
import wave

import av
import httpx
import openai

from src.config import Config
from src.logger import logger
from src.services.http_clients import home_assistant_client
from src.services.llm_client import openrouter_client

# 16 kHz / 16-bit / mono PCM is accepted by every Home Assistant STT provider (faster-whisper accepts only this).
SAMPLE_RATE = 16000
STT_METADATA = (
    f"format=wav; codec=pcm; sample_rate={SAMPLE_RATE}; bit_rate=16; channel=1; "
    f"language={Config.HOME_ASSISTANT_STT_LANGUAGE}"
)
FAILED_UPLOADS_DIR = Path(".logs/failed_audio")
NOT_UNDERSTOOD_MESSAGE = "Sorry, I couldn't understand the recording."


##############
# CONVERSION #
##############
def to_pcm(audio: bytes) -> bytes:
    """Decode any container/codec ffmpeg understands (m4a, 3gp, ogg, wav, ...) to raw 16 kHz mono s16 PCM."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm = bytearray()

    def append(frames) -> None:
        for frame in frames:
            # Plane buffers can be padded past the last sample.
            pcm.extend(bytes(frame.planes[0])[: frame.samples * 2])

    with av.open(io.BytesIO(audio)) as container:
        # A recording stopped right after it started has a header but a track with no codec and no samples.
        if not container.streams.audio or container.streams.audio[0].codec_context is None:
            return b""

        for frame in container.decode(audio=0):
            append(resampler.resample(frame))
    append(resampler.resample(None))

    return bytes(pcm)


def to_wav(pcm: bytes) -> bytes:
    """Wrap 16 kHz mono s16 PCM in a WAV header: a small, uniform upload whatever the client recorded."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def save_failed_upload(audio: bytes) -> str:
    FAILED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = FAILED_UPLOADS_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}.bin"
    path.write_bytes(audio)
    return str(path)


###########
# ENGINES #
###########
async def transcribe_with_openrouter(pcm: bytes) -> str:
    """Transcript from OpenRouter's speech-to-text API; raises openai.OpenAIError if the call fails."""
    # Short timeout and a single retry: the local fallback is better than a long wait.
    client = openrouter_client.with_options(timeout=Config.OPENROUTER_STT_TIMEOUT_SECONDS, max_retries=1)
    # Without a language the model detects it, so any language the user speaks is understood.
    language = {"language": Config.OPENROUTER_STT_LANGUAGE} if Config.OPENROUTER_STT_LANGUAGE else {}
    result = await client.audio.transcriptions.create(
        model=Config.OPENROUTER_STT_MODEL,
        file=("recording.wav", to_wav(pcm), "audio/wav"),
        **language,
    )
    return (result.text or "").strip()


async def transcribe_with_home_assistant(pcm: bytes) -> dict:
    """Transcribe with Home Assistant's STT entity. Returns {"text": <transcript>} or {"error": <message>}."""
    try:
        response = await home_assistant_client.post(
            f"/api/stt/{Config.HOME_ASSISTANT_STT_ENTITY}",
            content=pcm,
            headers={"X-Speech-Content": STT_METADATA},
        )
    except httpx.HTTPError as e:
        logger.error(f"Failed to transcribe audio: {e!r}")
        return {"error": f"Failed to transcribe audio: {e!r}"}

    if response.status_code != 200:
        logger.error(f"Failed to transcribe audio: {response.status_code} - {response.text}")
        return {"error": f"Failed to transcribe audio: {response.status_code} - {response.text}"}

    result = response.json()
    text = (result.get("text") or "").strip()
    if result.get("result") != "success" or not text:
        logger.error(f"Speech was not recognized: {result}")
        return {"error": NOT_UNDERSTOOD_MESSAGE}

    return {"text": text}


#################
# TRANSCRIPTION #
#################
async def transcribe(audio: bytes) -> dict:
    """
    Transcribe an audio recording with OpenRouter, or with Home Assistant's STT entity if OpenRouter fails.
    Returns {"text": <transcript>} or {"error": <message>}; a fallback transcript also has a "warning".
    """
    try:
        # Decoding is CPU-bound: keep it off the event loop.
        pcm = await asyncio.to_thread(to_pcm, audio)
    except (av.FFmpegError, ValueError) as e:
        # Keep the upload so the client's recording settings can be diagnosed.
        saved = await asyncio.to_thread(save_failed_upload, audio)
        logger.error(f"Failed to decode audio ({len(audio)} bytes, starts {audio[:16].hex(' ')}, saved to {saved}): {e}")
        return {"error": f"Failed to decode audio: {e}"}

    if not pcm:
        return {"error": "The recording contains no audio."}

    logger.info(f"Transcribing {len(pcm) / (SAMPLE_RATE * 2):.1f}s of audio with {Config.OPENROUTER_STT_MODEL}")

    try:
        text = await transcribe_with_openrouter(pcm)
    except openai.OpenAIError as e:
        logger.warning(f"OpenRouter transcription failed, falling back to {Config.HOME_ASSISTANT_STT_ENTITY}: {e!r}")
        result = await transcribe_with_home_assistant(pcm)
        if "text" in result:
            logger.info(f"Transcript (fallback): '{result['text']}'")
            result["warning"] = (
                f"OpenRouter speech-to-text failed ({type(e).__name__}), "
                f"so the local {Config.HOME_ASSISTANT_STT_ENTITY} fallback was used."
            )
        return result

    # An empty transcript is silence or noise: the fallback would not hear more.
    if not text:
        logger.error("Speech was not recognized: empty transcript")
        return {"error": NOT_UNDERSTOOD_MESSAGE}

    logger.info(f"Transcript: '{text}'")
    return {"text": text}
