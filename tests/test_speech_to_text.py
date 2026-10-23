from pathlib import Path
import io
import math
import wave

import httpx

from src.config import Config
from src.services import speech_to_text
from src.services.speech_to_text import SAMPLE_RATE, to_pcm, transcribe

STT_PATH = f"/api/stt/{Config.HOME_ASSISTANT_STT_ENTITY}"
OPENROUTER_STT_PATH = "/api/v1/audio/transcriptions"


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


def fake_openrouter_stt(fake_http, text: str = "Turn on the lights") -> None:
    fake_http.on("POST", OPENROUTER_STT_PATH, httpx.Response(200, json={"text": text}))


def fail_openrouter_stt(fake_http) -> None:
    fake_http.on("POST", OPENROUTER_STT_PATH, httpx.Response(400, json={"error": {"message": "model unavailable"}}))


def test_to_pcm_resamples_to_16k_mono():
    pcm = to_pcm(make_wav(seconds=0.5))

    # 16-bit mono: two bytes per sample. Resampler edges may add or drop a few samples.
    assert abs(len(pcm) / 2 - 0.5 * SAMPLE_RATE) < 100


##############
# OPENROUTER #
##############
async def test_transcribe_sends_16k_wav_to_openrouter(fake_http):
    fake_openrouter_stt(fake_http, " Turn on the lights ")

    result = await transcribe(make_wav())

    assert result == {"text": "Turn on the lights"}
    request = fake_http.requests_to(OPENROUTER_STT_PATH)[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    body = request.content
    assert f'name="model"\r\n\r\n{Config.OPENROUTER_STT_MODEL}'.encode() in body
    # No language by default: the model detects it (e.g. Italian or English).
    assert b'name="language"' not in body
    # The upload is the decoded 16 kHz mono PCM in a WAV header, whatever the client recorded.
    assert speech_to_text.to_wav(to_pcm(make_wav())) in body
    assert fake_http.requests_to(STT_PATH) == []


async def test_configured_language_is_forced(fake_http, monkeypatch):
    monkeypatch.setattr(Config, "OPENROUTER_STT_LANGUAGE", "it")
    fake_openrouter_stt(fake_http, "Accendi le luci")

    result = await transcribe(make_wav())

    assert result == {"text": "Accendi le luci"}
    assert b'name="language"\r\n\r\nit' in fake_http.requests_to(OPENROUTER_STT_PATH)[0].content


async def test_empty_openrouter_transcript_is_not_understood_without_fallback(fake_http):
    fake_openrouter_stt(fake_http, "  ")

    result = await transcribe(make_wav())

    assert result == {"error": "Sorry, I couldn't understand the recording."}
    assert fake_http.requests_to(STT_PATH) == []


############
# FALLBACK #
############
async def test_openrouter_failure_falls_back_to_home_assistant_with_warning(fake_http):
    fail_openrouter_stt(fake_http)
    fake_http.on("POST", STT_PATH, httpx.Response(200, json={"text": " Turn on the lights ", "result": "success"}))

    result = await transcribe(make_wav())

    assert result == {
        "text": "Turn on the lights",
        "warning": f"OpenRouter speech-to-text failed (BadRequestError), "
                   f"so the local {Config.HOME_ASSISTANT_STT_ENTITY} fallback was used.",
    }
    request = fake_http.requests_to(STT_PATH)[0]
    assert request.headers["Authorization"] == "Bearer ha-token"
    assert request.headers["X-Speech-Content"] == (
        f"format=wav; codec=pcm; sample_rate=16000; bit_rate=16; channel=1; "
        f"language={Config.HOME_ASSISTANT_STT_LANGUAGE}"
    )
    assert request.content == to_pcm(make_wav())


async def test_openrouter_connection_error_falls_back(fake_http):
    def unreachable(request):
        raise httpx.ConnectError("no route", request=request)

    fake_http.on("POST", OPENROUTER_STT_PATH, unreachable)
    fake_http.on("POST", STT_PATH, httpx.Response(200, json={"text": "Hello", "result": "success"}))

    result = await transcribe(make_wav())

    assert result["text"] == "Hello"
    assert "APIConnectionError" in result["warning"]


async def test_fallback_reports_unrecognized_speech(fake_http):
    fail_openrouter_stt(fake_http)
    fake_http.on("POST", STT_PATH, httpx.Response(200, json={"text": None, "result": "error"}))

    result = await transcribe(make_wav())

    assert result == {"error": "Sorry, I couldn't understand the recording."}


async def test_fallback_reports_http_errors(fake_http):
    fail_openrouter_stt(fake_http)
    fake_http.on("POST", STT_PATH, httpx.Response(415, text="unsupported"))

    result = await transcribe(make_wav())

    assert result == {"error": "Failed to transcribe audio: 415 - unsupported"}


############
# DECODING #
############
async def test_transcribe_rejects_undecodable_audio_and_keeps_it(fake_http, monkeypatch, tmp_path):
    monkeypatch.setattr(speech_to_text, "FAILED_UPLOADS_DIR", tmp_path)

    result = await transcribe(b"definitely not audio")

    assert result["error"].startswith("Failed to decode audio:")
    assert fake_http.requests == []
    assert [path.read_bytes() for path in tmp_path.iterdir()] == [b"definitely not audio"]


async def test_empty_recording_is_reported_as_no_audio(fake_http):
    # Real Tasker upload: recording stopped immediately, so the 3gp has a track with no codec or samples.
    empty_recording = (Path(__file__).parent / "fixtures" / "empty_recording.3gp").read_bytes()

    result = await transcribe(empty_recording)

    assert result == {"error": "The recording contains no audio."}
    assert fake_http.requests == []
