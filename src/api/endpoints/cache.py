from fastapi import APIRouter
from pydantic import BaseModel
from src.services.llm_cache.llm_cache import cache

router = APIRouter()

# Wipe cache endpoint
@router.post("/cache/wipe")
async def wipe_cache():
    cache.wipe()
    return {"message": "Cache wiped successfully."}