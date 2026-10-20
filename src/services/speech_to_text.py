import asyncio
import io

import av
import httpx

from src.config import Config
from src.logger import logger
from src.services.http_clients import home_assistant_client

# 16 kHz / 16-bit / mono PCM is accepted by every Home Assistant STT provider (faster-whisper accepts only this).
SAMPLE_RATE = 16000
STT_METADATA = (
    f"format=wav; codec=pcm; sample_rate={SAMPLE_RATE}; bit_rate=16; channel=1; "
    f"language={Config.HOME_ASSISTANT_STT_LANGUAGE}"
)


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
        for frame in container.decode(audio=0):
            append(resampler.resample(frame))
    append(resampler.resample(None))

    return bytes(pcm)


#################
# TRANSCRIPTION #
#################
async def transcribe(audio: bytes) -> dict:
    """
    Transcribe an audio recording with Home Assistant's STT entity.
    Returns {"text": <transcript>} or {"error": <message>}.
    """
    try:
        # Decoding is CPU-bound: keep it off the event loop.
        pcm = await asyncio.to_thread(to_pcm, audio)
    except (av.FFmpegError, ValueError) as e:
        logger.error(f"Failed to decode audio: {e}")
        return {"error": f"Failed to decode audio: {e}"}

    if not pcm:
        return {"error": "The recording contains no audio."}

    logger.info(f"Transcribing {len(pcm) / (SAMPLE_RATE * 2):.1f}s of audio with {Config.HOME_ASSISTANT_STT_ENTITY}")

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
        return {"error": "Sorry, I couldn't understand the recording."}

    logger.info(f"Transcript: '{text}'")
    return {"text": text}
