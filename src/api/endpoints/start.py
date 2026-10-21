from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.agents.orchestrator_agent import route_query
from src.agents.grocy_agent import run_grocy_agent
from src.agents.home_assistant_agent import run_home_assistant_agent
from src.agents.glados_responder_agent import get_final_glados_response
from src.agents.vikunja_agent import run_vikunja_agent
from src.logger import logger
from src.services.speech_to_text import transcribe
from src.services.text_to_speech import speak

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


# Tools backed by an MCP agent: each returns {"response": ...} or {"error": ...}.
MCP_AGENTS = {
    "Home Assistant Tool": run_home_assistant_agent,
    "Grocy Tool": run_grocy_agent,
}


async def run_pipeline(message: str) -> dict:
    route = await route_query(message)

    if route.tool_name in MCP_AGENTS:
        result = await MCP_AGENTS[route.tool_name](message)
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
# Every reply, errors included, is also spoken on the Home Assistant speaker; `spoken` says whether that worked.
@router.post("/start_audio")
async def start_audio(request: Request):
    audio = await request.body()
    logger.info(f"Received {len(audio)} bytes of audio ({request.headers.get('content-type')})")
    if not audio:
        response = {"message": "No audio received.", "transcript": None}
    else:
        transcript = await transcribe(audio)
        if transcript.get("error"):
            response = {"message": transcript["error"], "transcript": None}
        else:
            start_response = await start(StartRequest(message=transcript["text"]))
            response = {**start_response, "transcript": transcript["text"]}

    return {**response, "spoken": await speak(response["message"])}

# rhasspy endpoint accepts any json
@router.post("/start_rhasspy")
async def start_rhasspy(request: Request):
    data = await request.json()

    user_input = data["raw_text"]
    start_request = StartRequest(message=user_input)
    start_response = await start(start_request)
    return start_response
