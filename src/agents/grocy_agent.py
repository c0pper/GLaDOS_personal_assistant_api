import asyncio

from src.agents.mcp_agent import McpAgentSpec, run_mcp_agent
from src.config import Config


SYSTEM_PROMPT = (
    "You are an assistant that manages a household's Grocy instance through the available tools: "
    "pantry and fridge stock, expiring products, the shopping list, recipes, the meal plan and chores. "
    "Find products, recipes and chores with the tools (never guess ids), act on them, and verify when useful. "
    "The request may come from speech recognition and contain misheard words: if a name matches nothing after "
    "a couple of searches, stop and say what you could not find instead of searching further. "
    "Never add, consume or remove more than the user asked for. "
    "Answer in a short plain-text sentence stating the outcome or the requested values; "
    "another assistant will rephrase your answer for the user."
)

SPEC = McpAgentSpec(
    name="Grocy",
    mcp_url=Config.GROCY_MCP_URL,
    model=Config.GROCY_AGENT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=Config.GROCY_MCP_TOOLS,
    max_tool_rounds=Config.GROCY_MAX_TOOL_ROUNDS,
    timeout_seconds=Config.GROCY_TIMEOUT_SECONDS,
)


async def run_grocy_agent(user_query: str) -> dict:
    """
    Let the model act on Grocy through its MCP server until it can answer the query.
    Returns {"response": <model answer>} or {"error": <message>}.
    """
    return await run_mcp_agent(SPEC, user_query)


#####################
# DEMO EXECUTION #
#####################
if __name__ == "__main__":
    async def main():
        print(await run_grocy_agent("What is expiring soon?"))

    asyncio.run(main())
