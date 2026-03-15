from fastapi import APIRouter
from pydantic import BaseModel
from src.agents.orchestrator_agent import get_tool_name
from src.agents.home_assistant_agent import run_home_assistant_agent
from src.agents.glados_responder_agent import get_final_glados_response
from src.agents.vikunja_agent import run_vikunja_agent

router = APIRouter()


class StartRequest(BaseModel):
    message: str


@router.post("/start")
async def start(request: StartRequest):
    tool_name = get_tool_name(request.message)
    if not tool_name:
        return {"message": "Sorry, I don't know how to help you with that."}

    if tool_name == "Home Assistant Tool":
        result = run_home_assistant_agent(request.message)
        if result.get("error"):
            return {"message": result["error"]}

        speech = result["speech"]["plain"]["speech"]
        final_response = get_final_glados_response(
            user_input=request.message, 
            tool_result=speech
        )

        return {"message": final_response}

    if tool_name == "Vikunja Tool":
        result = run_vikunja_agent(request.message)
        if not result:
            return {"message": "run_vikunja_agent returned None."}

        final_response = get_final_glados_response(
            user_input=request.message, 
            tool_result=str(result)
        )

        return {"message": final_response}

    return {"message": f"Sorry, I don't know how to help you with that."}
