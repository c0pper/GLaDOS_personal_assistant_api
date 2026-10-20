import httpx

from src.config import Config


# Long-lived clients keep connections alive between requests and enforce timeouts.
home_assistant_client = httpx.AsyncClient(
    base_url=Config.HOME_ASSISTANT_BASE_URL,
    headers={
        "Accept": "application/json",
        "Authorization": f"Bearer {Config.HOME_ASSISTNAT_TOKEN}",
    },
    timeout=Config.HTTP_TIMEOUT_SECONDS,
)

vikunja_client = httpx.AsyncClient(
    # Trailing slash so relative paths like "projects" resolve under /api/v1/.
    base_url=Config.VIKUNJA_BASE_URL.rstrip("/") + "/",
    headers={
        "Accept": "application/json",
        "Authorization": f"Bearer {Config.VIKUNJA_TOKEN}",
    },
    timeout=Config.HTTP_TIMEOUT_SECONDS,
)


async def close_http_clients() -> None:
    await home_assistant_client.aclose()
    await vikunja_client.aclose()
