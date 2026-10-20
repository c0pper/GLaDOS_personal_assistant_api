from pathlib import Path
import io
import math
import wave

import httpx

from src.config import Config
from src.services import speech_to_text
from src.services.speech_to_text import SAMPLE_RATE, to_pcm, transcribe

STT_PATH = f"/api/stt/{Config.HOME_ASSISTANT_STT_ENTITY}"


def make_wav(seconds: float = 0.5, rate: int = 44100, channels: int = 2) -> bytes:
    """A 440 Hz tone in a format different from the target, so conversion is exercised."""
    frames = bytearray()
    for i in range(int(seconds * rate)):
        sample = int(10000 * math.sin(2 * math.pi * 440 * i / rate)).to_bytes(2, "little", signed=True)
        frames += sample * channels
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))
    return buffer.getvalue()


def test_to_pcm_resamples_to_16k_mono():
    pcm = to_pcm(make_wav(seconds=0.5))

    # 16-bit mono: two bytes per sample. Resampler edges may add or drop a few samples.
    assert abs(len(pcm) / 2 - 0.5 * SAMPLE_RATE) < 100


async def test_transcribe_sends_pcm_with_metadata(fake_http):
    fake_http.on("POST", STT_PATH, httpx.Response(200, json={"text": " Turn on the lights ", "result": "success"}))

    result = await transcribe(make_wav())

    assert result == {"text": "Turn on the lights"}
    request = fake_http.requests_to(STT_PATH)[0]
    assert request.headers["Authorization"] == "Bearer ha-token"
    assert request.headers["X-Speech-Content"] == (
        f"format=wav; codec=pcm; sample_rate=16000; bit_rate=16; channel=1; "
        f"language={Config.HOME_ASSISTANT_STT_LANGUAGE}"
    )
    assert request.content == to_pcm(make_wav())


async def test_transcribe_rejects_undecodable_audio_and_keeps_it(fake_http, monkeypatch, tmp_path):
    monkeypatch.setattr(speech_to_text, "FAILED_UPLOADS_DIR", tmp_path)

    result = await transcribe(b"definitely not audio")

    assert result["error"].startswith("Failed to decode audio:")
    assert fake_http.requests == []
    assert [path.read_bytes() for path in tmp_path.iterdir()] == [b"definitely not audio"]


async def test_transcribe_reports_unrecognized_speech(fake_http):
    fake_http.on("POST", STT_PATH, httpx.Response(200, json={"text": None, "result": "error"}))

    result = await transcribe(make_wav())

    assert result == {"error": "Sorry, I couldn't understand the recording."}


async def test_transcribe_reports_http_errors(fake_http):
    fake_http.on("POST", STT_PATH, httpx.Response(415, text="unsupported"))

    result = await transcribe(make_wav())

    assert result == {"error": "Failed to transcribe audio: 415 - unsupported"}


async def test_empty_recording_is_reported_as_no_audio(fake_http):
    # Real Tasker upload: recording stopped immediately, so the 3gp has a track with no codec or samples.
    empty_recording = (Path(__file__).parent / "fixtures" / "empty_recording.3gp").read_bytes()

    result = await transcribe(empty_recording)

    assert result == {"error": "The recording contains no audio."}
    assert fake_http.requests == []
