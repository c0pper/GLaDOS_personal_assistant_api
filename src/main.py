from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from src.api.endpoints.start import router as start_router
from src.api.endpoints.cache import router as cache_router
from src.api.endpoints.conversations import router as conversations_router
from src.logger import logger
from src.services.http_clients import close_http_clients
from src.services.llm_client import openrouter_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_http_clients()
    await openrouter_client.close()
    await logger.complete()


app = FastAPI(lifespan=lifespan)

app.include_router(start_router, prefix="/api")
app.include_router(cache_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
