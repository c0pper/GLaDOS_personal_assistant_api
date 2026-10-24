from typing import Any, Dict, Literal, Optional
import time

import httpx
from pydantic import BaseModel, Field, ValidationError

from src.config import Config
from src.logger import logger

# A failing Jev must not slow every request down: after a few failures, skip it for a while.
FAILURE_THRESHOLD = 3
COOLDOWN_SECONDS = 60.0


class JevError(Exception):
    """Jev could not answer; the caller should use the normal LLM path."""


########################
# RESPONSE SCHEMAS #
########################
class JevChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class JevNoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class JevUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Optional[float] = None


class JevResponse(BaseModel):
    answers: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    usage: Optional[JevUsage] = None


##########
# CLIENT #
##########
# A single async client (and connection pool) shared by every caller.
_client = httpx.AsyncClient(timeout=Config.JEV_TIMEOUT_SECONDS)

_consecutive_failures = 0
_disabled_until = 0.0


def is_available() -> bool:
    return time.monotonic() >= _disabled_until


def _record_failure() -> None:
    global _consecutive_failures, _disabled_until
    _consecutive_failures += 1
    if _consecutive_failures >= FAILURE_THRESHOLD:
        _disabled_until = time.monotonic() + COOLDOWN_SECONDS
        logger.warning(f"Jev disabled for {COOLDOWN_SECONDS:.0f}s after {_consecutive_failures} failures")


def _record_success() -> None:
    global _consecutive_failures, _disabled_until
    _consecutive_failures = 0
    _disabled_until = 0.0


def reset() -> None:
    """Reset the circuit breaker (tests, or after changing configuration)."""
    global _consecutive_failures, _disabled_until
    _consecutive_failures = 0
    _disabled_until = 0.0


async def decide(state: Any, questions: Dict[str, Any]) -> JevResponse:
    """
    Ask Jev typed questions about `state`.

    Raises JevError on any failure, including the circuit breaker being open.
    """
    if not is_available():
        raise JevError("Jev is temporarily disabled after repeated failures")

    started = time.perf_counter()
    try:
        response = await _client.post(
            Config.JEV_DECISIONS_URL,
            headers={"Authorization": f"Bearer {Config.OPENROUTER_API_KEY}"},
            json={"model": Config.JEV_MODEL, "state": state, "questions": questions},
        )
        response.raise_for_status()
        parsed = JevResponse.model_validate(response.json())
    except (httpx.HTTPError, ValidationError, ValueError) as e:
        _record_failure()
        logger.warning(f"Jev decision failed: {e!r}")
        raise JevError(str(e)) from e

    _record_success()
    elapsed_ms = (time.perf_counter() - started) * 1000
    cost = parsed.usage.cost if parsed.usage else None
    logger.info(f"Jev answered in {elapsed_ms:.0f} ms (model={parsed.model}, cost={cost})")
    return parsed


async def close() -> None:
    await _client.aclose()
