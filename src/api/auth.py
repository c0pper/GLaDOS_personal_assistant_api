import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import Config

# `auto_error=False` so a missing key gets our 401 (with the same message as a wrong one)
# instead of FastAPI's bare 403.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Reject requests without the shared key; attach to routers so every endpoint is covered."""
    if not api_key or not secrets.compare_digest(api_key.encode(), Config.GLADOS_API_KEY.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
