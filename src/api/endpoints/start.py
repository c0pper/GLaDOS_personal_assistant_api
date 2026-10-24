from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.orchestrator_agent import route_query
from src.agents.grocy_agent import run_grocy_agent
from src.agents.home_assistant_agent import run_home_assistant_agent
from src.agents.glados_responder_agent import get_final_glados_response
from src.agents.vikunja_agent import run_vikunja_agent
from src.agents.web_search_agent import run_web_search_agent
from src.logger import logger
from src.services.conversation_store import conversations
from src.services.speech_to_text import transcribe
from src.services.text_to_speech import speak, synthesize

router = APIRouter()


class StartRequest(BaseModel):
    message: str
    # Clients that want conversation memory (e.g. the Telegram bot, one id per chat) send an id;
    # without one every message is handled on its own.
    conversation_id: Optional[str] = None

class SpeechRequest(BaseModel):
    text: str

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
    history = conversations.history_text(request.conversation_id) if request.conversation_id else None

    try:
        response = await run_pipeline(request.message, history)
    except Exception as e:
        # Typically the LLM provider failing even after retries and fallbacks: answer instead of a bare 500.
        logger.exception(f"Pipeline failed: {e!r}")
        return {"message": "Sorry, my brain is temporarily unavailable. Try again in a moment."}

    if request.conversation_id:
        conversations.add_turn(request.conversation_id, request.message, response["message"])
    return response


# Tools backed by an MCP agent: each returns {"response": ...} or {"error": ...}.
MCP_AGENTS = {
    "Home Assistant Tool": run_home_assistant_agent,
    "Grocy Tool": run_grocy_agent,
    "Web Search Tool": run_web_search_agent,
}


async def run_pipeline(message: str, conversation_history: Optional[str] = None) -> dict:
    route = await route_query(message, conversation_history)
    # Tool agents don't see the conversation: a follow-up ("turn it off") reaches them made self-contained.
    tool_query = route.standalone_query or message
    if tool_query != message:
        logger.info(f"Follow-up rewritten for the tool: '{tool_query}'")

    if route.tool_name in MCP_AGENTS:
        result = await MCP_AGENTS[route.tool_name](tool_query)
        if result.get("error"):
            return {"message": result["error"]}

        tool_result = result["response"]

    elif route.tool_name == "Vikunja Tool":
        tool_result = await run_vikunja_agent(tool_query)
        if not tool_result:
            return {"message": "run_vikunja_agent returned None."}

    elif route.tool_name == "No Tool":
        # Plain conversation: no tool call needed, GLaDOS answers directly.
        tool_result = None

    else:
        return {"message": "Sorry, I don't know how to help you with that."}

    final_response = await get_final_glados_response(
        user_input=message,
        tool_result=tool_result,
        conversation_history=conversation_history,
    )

    return {"message": final_response}

# The request body is the raw recording (any format ffmpeg decodes, e.g. Tasker's m4a), not a multipart form.
# Every reply, errors included, is also spoken on the Home Assistant speaker; `spoken` says whether that worked.
# `conversation_id` (query parameter) works as in /start: the transcript and the reply join that conversation.
@router.post("/start_audio")
async def start_audio(request: Request, conversation_id: Optional[str] = None):
    audio = await request.body()
    logger.info(f"Received {len(audio)} bytes of audio ({request.headers.get('content-type')})")
    # Set when the fallback speech-to-text was used: returned for the client to show, not spoken.
    warning = None
    if not audio:
        response = {"message": "No audio received.", "transcript": None}
    else:
        transcript = await transcribe(audio)
        warning = transcript.get("warning")
        if transcript.get("error"):
            response = {"message": transcript["error"], "transcript": None}
        else:
            start_response = await start(StartRequest(message=transcript["text"], conversation_id=conversation_id))
            response = {**start_response, "transcript": transcript["text"]}

    response["warning"] = warning

    return {**response, "spoken": await speak(response["message"])}

# The text spoken with GLaDOS's voice, as an OGG/Opus voice note (e.g. for the Telegram bot's voice replies).
@router.post("/speech")
async def speech(request: SpeechRequest):
    audio = await synthesize(request.text)
    if audio is None:
        return JSONResponse(status_code=502, content={"message": "Failed to render speech."})
    return Response(content=audio, media_type="audio/ogg")

# rhasspy endpoint accepts any json
@router.post("/start_rhasspy")
async def start_rhasspy(request: Request):
    data = await request.json()

    user_input = data["raw_text"]
    start_request = StartRequest(message=user_input)
    start_response = await start(start_request)
    return start_response
