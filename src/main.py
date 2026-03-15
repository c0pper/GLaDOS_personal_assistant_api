from fastapi import FastAPI
from src.api.endpoints.start import router as start_router
from src.api.endpoints.cache import router as cache_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.include_router(start_router, prefix="/api")
app.include_router(cache_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
