from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Tuple
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent, Tool

from src.logger import logger
from src.services.llm_client import fallback_routing, openrouter_client

# Tool results can be huge (search dumps, history); cap what goes back into the prompt.
MAX_TOOL_RESULT_CHARS = 8000


@dataclass(frozen=True)
class McpAgentSpec:
    """Everything that distinguishes one MCP-backed agent (Home Assistant, Grocy, ...) from another."""

    name: str
    mcp_url: str
    model: str
    system_prompt: str
    # Only these tools are offered to (and accepted from) the model; MCP servers also expose admin/destructive
    # ones, and every offered tool definition is sent with each LLM call.
    allowed_tools: Tuple[str, ...]
    max_tool_rounds: int
    timeout_seconds: float


#######
# MCP #
#######
@asynccontextmanager
async def open_mcp_session(url: str) -> AsyncIterator[ClientSession]:
    async with streamable_http_client(url) as (read_stream, write_stream, _):
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


async def call_tool(spec: McpAgentSpec, session: ClientSession, name: str, raw_arguments: str) -> str:
    # The model only sees allowed tools, but never trust the name it sends back.
    if name not in spec.allowed_tools:
        return f"Tool error: '{name}' is not available."

    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Tool error: invalid JSON arguments: {e}"

    logger.info(f"Calling {spec.name} tool {name} with {arguments}")
    return tool_result_to_text(await session.call_tool(name, arguments))


##############
# AGENT LOOP #
##############
async def run_tool_loop(spec: McpAgentSpec, user_query: str) -> str:
    async with open_mcp_session(spec.mcp_url) as session:
        listed = await session.list_tools()
        tools = [to_openai_tool(tool) for tool in listed.tools if tool.name in spec.allowed_tools]

        system_prompt = f"{spec.system_prompt}\nCurrent date and time: {datetime.now():%Y-%m-%d %H:%M}"
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        for _ in range(spec.max_tool_rounds):
            completion = await openrouter_client.chat.completions.create(
                model=spec.model,
                messages=messages,
                tools=tools,
                extra_body=fallback_routing(spec.model),
            )
            message = completion.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            messages.append(message.model_dump(include={"role", "content", "tool_calls"}))
            for tool_call in message.tool_calls:
                result = await call_tool(spec, session, tool_call.function.name, tool_call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    raise RuntimeError(f"no answer after {spec.max_tool_rounds} tool rounds")


async def run_mcp_agent(spec: McpAgentSpec, user_query: str) -> dict:
    """
    Let the model act through the agent's MCP server until it can answer the query.
    Returns {"response": <model answer>} or {"error": <message>}.
    """
    logger.info(f"Running {spec.name} agent with user query: {user_query}")

    try:
        async with asyncio.timeout(spec.timeout_seconds):
            response = await run_tool_loop(spec, user_query)
    except Exception as e:
        # MCP transport errors surface as exception groups, so any failure is reported the same way.
        logger.exception(f"{spec.name} agent failed: {e!r}")
        return {"error": f"{spec.name} agent failed: {e!r}"}

    logger.info(f"{spec.name} agent response: {response}")
    return {"response": response}
