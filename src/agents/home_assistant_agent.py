import asyncio

from src.agents.mcp_agent import McpAgentSpec, run_mcp_agent
from src.config import Config


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

SPEC = McpAgentSpec(
    name="Home Assistant",
    mcp_url=Config.HA_MCP_URL,
    model=Config.HOME_ASSISTANT_AGENT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=Config.HOME_ASSISTANT_MCP_TOOLS,
    max_tool_rounds=Config.HOME_ASSISTANT_MAX_TOOL_ROUNDS,
    timeout_seconds=Config.HOME_ASSISTANT_TIMEOUT_SECONDS,
)


async def run_home_assistant_agent(user_query: str) -> dict:
    """
    Let the model act on Home Assistant through its MCP server until it can answer the query.
    Returns {"response": <model answer>} or {"error": <message>}.
    """
    return await run_mcp_agent(SPEC, user_query)


#####################
# DEMO EXECUTION #
#####################
if __name__ == "__main__":
    async def main():
        print(await run_home_assistant_agent("What is the temperature right now?"))

    asyncio.run(main())
