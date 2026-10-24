import httpx
import pytest

from src.config import Config
from src.main import app

# Every /api route needs the key; one per router proves all routers carry the dependency.
PROTECTED_ROUTES = [
    ("POST", "/api/start", {"message": "Hello"}),
    ("POST", "/api/cache/wipe", None),
    ("DELETE", "/api/conversations/test", None),
]


@pytest.fixture
async def api():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES)
async def test_missing_api_key_is_rejected(api, method, path, body):
    response = await api.request(method, path, json=body)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key."}
    assert response.headers["www-authenticate"] == "ApiKey"


@pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES)
async def test_wrong_api_key_is_rejected(api, method, path, body):
    response = await api.request(method, path, json=body, headers={"X-API-Key": "not-the-key"})

    assert response.status_code == 401


async def test_correct_api_key_is_accepted(api):
    response = await api.post("/api/cache/wipe", headers={"X-API-Key": Config.GLADOS_API_KEY})

    assert response.status_code == 200
    assert response.json() == {"message": "Cache wiped successfully."}


async def test_scheme_is_advertised_in_the_openapi_schema(api):
    schema = (await api.get("/openapi.json")).json()

    assert schema["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
    }


async def test_docs_stay_public(api):
    assert (await api.get("/docs")).status_code == 200
    assert (await api.get("/openapi.json")).status_code == 200
