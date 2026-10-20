from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent, Tool

from src.config import Config
from src.logger import logger
from src.services.llm_client import fallback_routing, openrouter_client


SYSTEM_PROMPT = (
    "You are an assistant that controls and reads a Home Assistant smart home through the available tools. "
    "Find the right entities with the tools (never guess entity ids), act on them, and verify when useful. "
    "The request may come from speech recognition and contain misheard words: if a name matches nothing after "
    "a couple of searches, stop and say what you could not find instead of searching further. "
    "Answer in a short plain-text sentence stating the outcome or the requested values; "
    "another assistant will rephrase your answer for the user. "
    "Projector related: use switch.projector_power. "
    "For music or radio requests, use strictly musicassistant. "
    "For putting on things on tv, use Stremio: play on Sony TV script with the relevant variables."
)

# Tool results can be huge (search dumps, history); cap what goes back into the prompt.
MAX_TOOL_RESULT_CHARS = 8000


#######
# MCP #
#######
@asynccontextmanager
async def open_mcp_session() -> AsyncIterator[ClientSession]:
    async with streamable_http_client(Config.HA_MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def to_openai_tool(tool: Tool) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema},
    }


def tool_result_to_text(result: CallToolResult) -> str:
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    if result.isError:
        text = f"Tool error: {text}"
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"
    return text


async def call_tool(session: ClientSession, name: str, raw_arguments: str) -> str:
    # The model only sees allowed tools, but never trust the name it sends back.
    if name not in Config.HOME_ASSISTANT_MCP_TOOLS:
        return f"Tool error: '{name}' is not available."

    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Tool error: invalid JSON arguments: {e}"

    logger.info(f"Calling Home Assistant tool {name} with {arguments}")
    return tool_result_to_text(await session.call_tool(name, arguments))


##############
# AGENT LOOP #
##############
async def run_tool_loop(user_query: str) -> str:
    async with open_mcp_session() as session:
        listed = await session.list_tools()
        tools = [to_openai_tool(tool) for tool in listed.tools if tool.name in Config.HOME_ASSISTANT_MCP_TOOLS]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nCurrent date and time: {datetime.now():%Y-%m-%d %H:%M}"},
            {"role": "user", "content": user_query},
        ]

        for _ in range(Config.HOME_ASSISTANT_MAX_TOOL_ROUNDS):
            completion = await openrouter_client.chat.completions.create(
                model=Config.HOME_ASSISTANT_AGENT_MODEL,
                messages=messages,
                tools=tools,
                extra_body=fallback_routing(Config.HOME_ASSISTANT_AGENT_MODEL),
            )
            message = completion.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            messages.append(message.model_dump(include={"role", "content", "tool_calls"}))
            for tool_call in message.tool_calls:
                result = await call_tool(session, tool_call.function.name, tool_call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    raise RuntimeError(f"no answer after {Config.HOME_ASSISTANT_MAX_TOOL_ROUNDS} tool rounds")


async def run_home_assistant_agent(user_query: str) -> dict:
    """
    Let the model act on Home Assistant through its MCP server until it can answer the query.
    Returns {"response": <model answer>} or {"error": <message>}.
    """
    logger.info(f"Running Home Assistant agent with user query: {user_query}")

    try:
        async with asyncio.timeout(Config.HOME_ASSISTANT_TIMEOUT_SECONDS):
            response = await run_tool_loop(user_query)
    except Exception as e:
        # MCP transport errors surface as exception groups, so any failure is reported the same way.
        logger.exception(f"Home Assistant agent failed: {e!r}")
        return {"error": f"Home Assistant agent failed: {e!r}"}

    logger.info(f"Home Assistant agent response: {response}")
    return {"response": response}


#####################
# DEMO EXECUTION #
#####################
if __name__ == "__main__":
    async def main():
        print(await run_home_assistant_agent("What is the temperature right now?"))

    asyncio.run(main())
