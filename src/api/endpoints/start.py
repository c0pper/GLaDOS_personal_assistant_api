from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.agents.orchestrator_agent import route_query
from src.agents.home_assistant_agent import run_home_assistant_agent
from src.agents.glados_responder_agent import get_final_glados_response
from src.agents.vikunja_agent import run_vikunja_agent
from src.logger import logger
from src.services.speech_to_text import transcribe

router = APIRouter()


class StartRequest(BaseModel):
    message: str

class Intent(BaseModel):
    confidence: float
    name: str

class RhasspyRequest(BaseModel):
    entities: List[Any]
    intent: Intent
    raw_text: str
    raw_tokens: List[str]
    recognize_seconds: float
    slots: Dict[str, Any]
    speech_confidence: float
    text: str
    tokens: List[str]
    wakeword_id: Optional[str] = None


@router.post("/start")
async def start(request: StartRequest):
    try:
        return await run_pipeline(request.message)
    except Exception as e:
        # Typically the LLM provider failing even after retries and fallbacks: answer instead of a bare 500.
        logger.exception(f"Pipeline failed: {e!r}")
        return {"message": "Sorry, my brain is temporarily unavailable. Try again in a moment."}


async def run_pipeline(message: str) -> dict:
    route = await route_query(message)

    if route.tool_name == "Home Assistant Tool":
        result = await run_home_assistant_agent(message)
        if result.get("error"):
            return {"message": result["error"]}

        tool_result = result["response"]

    elif route.tool_name == "Vikunja Tool":
        tool_result = await run_vikunja_agent(message)
        if not tool_result:
            return {"message": "run_vikunja_agent returned None."}

    elif route.tool_name == "No Tool":
        # Plain conversation: no tool call needed, GLaDOS answers directly.
        tool_result = None

    else:
        return {"message": "Sorry, I don't know how to help you with that."}

    final_response = await get_final_glados_response(
        user_input=message,
        tool_result=tool_result
    )

    return {"message": final_response}

# The request body is the raw recording (any format ffmpeg decodes, e.g. Tasker's m4a), not a multipart form.
@router.post("/start_audio")
async def start_audio(request: Request):
    audio = await request.body()
    if not audio:
        return {"message": "No audio received."}

    transcript = await transcribe(audio)
    if transcript.get("error"):
        return {"message": transcript["error"], "transcript": None}

    start_response = await start(StartRequest(message=transcript["text"]))
    return {**start_response, "transcript": transcript["text"]}

# rhasspy endpoint accepts any json
@router.post("/start_rhasspy")
async def start_rhasspy(request: Request):
    data = await request.json()

    user_input = data["raw_text"]
    start_request = StartRequest(message=user_input)
    start_response = await start(start_request)
    return start_response
