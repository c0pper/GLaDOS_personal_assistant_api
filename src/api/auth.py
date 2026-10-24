import hashlib
import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import Config
from src.logger import logger

# `auto_error=False` so a missing key gets our 401 (with the same message as a wrong one)
# instead of FastAPI's bare 403.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def api_key_fingerprint(key: str) -> str:
    """Non-reversible fingerprint of the shared key, so two processes' logs can be compared without printing it."""
    if not key:
        return "EMPTY (0 chars)"
    return f"sha256:{hashlib.sha256(key.encode()).hexdigest()[:12]} ({len(key)} chars)"


def log_api_key_config() -> None:
    """
    Log which key this process loaded at startup. Two containers that disagree here return 401 to each other;
    the fingerprint (not the key) is enough to spot that. Set GLADOS_LOG_API_KEY=true to print the raw value.
    """
    if os.environ.get("GLADOS_LOG_API_KEY", "").lower() == "true":
        logger.warning(f"GLADOS_API_KEY={Config.GLADOS_API_KEY!r} (raw logging enabled by GLADOS_LOG_API_KEY)")
    else:
        logger.info(f"GLADOS_API_KEY fingerprint: {api_key_fingerprint(Config.GLADOS_API_KEY)}")


async def require_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Reject requests without the shared key; attach to routers so every endpoint is covered."""
    if not api_key or not secrets.compare_digest(api_key.encode(), Config.GLADOS_API_KEY.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
