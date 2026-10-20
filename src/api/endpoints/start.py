from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.agents.orchestrator_agent import route_query
from src.agents.home_assistant_agent import run_home_assistant_agent
from src.agents.glados_responder_agent import get_final_glados_response
from src.agents.vikunja_agent import run_vikunja_agent

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
    route = await route_query(request.message)

    if route.tool_name == "Home Assistant Tool":
        intent_name = route.home_assistant_intent.value if route.home_assistant_intent else None
        result = await run_home_assistant_agent(request.message, intent_name)
        if result.get("error"):
            return {"message": result["error"]}

        tool_result = result.get("speech", {}).get("plain", {}).get("speech")

    elif route.tool_name == "Vikunja Tool":
        tool_result = await run_vikunja_agent(request.message)
        if not tool_result:
            return {"message": "run_vikunja_agent returned None."}

    elif route.tool_name == "No Tool":
        # Plain conversation: no tool call needed, GLaDOS answers directly.
        tool_result = None

    else:
        return {"message": "Sorry, I don't know how to help you with that."}

    final_response = await get_final_glados_response(
        user_input=request.message,
        tool_result=tool_result
    )

    return {"message": final_response}

# rhasspy endpoint accepts any json
@router.post("/start_rhasspy")
async def start_rhasspy(request: Request):
    data = await request.json()

    user_input = data["raw_text"]
    start_request = StartRequest(message=user_input)
    start_response = await start(start_request)
    return start_response
