import asyncio

from src.agents.mcp_agent import McpAgentSpec, run_mcp_agent
from src.config import Config


SYSTEM_PROMPT = (
    "You are an assistant that answers questions by searching the web through the available tools. "
    "Search first, and extract a page when its snippet is not enough; never answer from memory about "
    "current events, prices, schedules or other facts that can change. Use the fewest searches that answer "
    "the question. "
    "Prefer recent and authoritative sources, and say so when they disagree or when the answer is unclear. "
    "The request may come from speech recognition and contain misheard words: search the most likely "
    "reading instead of asking for clarification. "
    "Answer in a short plain-text paragraph with the facts and the dates they refer to; "
    "another assistant will rephrase your answer for the user."
)

SPEC = McpAgentSpec(
    name="Web search",
    mcp_url=Config.TAVILY_MCP_URL,
    model=Config.TAVILY_AGENT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=Config.TAVILY_MCP_TOOLS,
    max_tool_rounds=Config.TAVILY_MAX_TOOL_ROUNDS,
    timeout_seconds=Config.TAVILY_TIMEOUT_SECONDS,
)


async def run_web_search_agent(user_query: str) -> dict:
    """
    Let the model search the web through Tavily's MCP server until it can answer the query.
    Returns {"response": <model answer>} or {"error": <message>}.
    """
    return await run_mcp_agent(SPEC, user_query)


#####################
# DEMO EXECUTION #
#####################
if __name__ == "__main__":
    async def main():
        print(await run_web_search_agent("Who won the last Formula 1 race?"))

    asyncio.run(main())
