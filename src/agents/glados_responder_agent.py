from typing import Optional

from pydantic import Field
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

from src.agents.context_providers import CurrentDateProvider
from src.config import Config
from src.logger import logger
from src.services.llm_cache.llm_cache import cache
from src.services.llm_client import fallback_routing, llm_client

CACHE_NAMESPACE = "glados_responder"


########################
# INPUT/OUTPUT SCHEMAS #
########################
class GladosResponderInputSchema(BaseIOSchema):
    """Input schema for the GLaDOS Responder Agent."""

    chat_message: str = Field(..., description="The original user query.")
    tool_result: Optional[str] = Field(None, description="The result of the processing done by the tool, if available.")


class GladosResponderOutputSchema(BaseIOSchema):
    """Output schema for the GLaDOS Responder Agent."""

    final_response: str = Field(..., description="The final response in the sarcastic, laconic tone of GLaDOS.")


########################
# RESPONDER AGENT CONFIG #
########################
# Shared, stateless pieces. A fresh agent (with empty history) is built per call so that
# concurrent requests never share or grow a conversation history.
system_prompt_generator = SystemPromptGenerator(
    background=[
        "You are GLaDOS, a sarcastic, passive-aggressive AI from the game Portal 2.",
        "ALWAYS SPEAK IN AN EMOTIONLESS, LACONIC TONE.",
        "You constantly doubt the user's intelligence but always, begrudgingly, comply.",
        "You are provided the user's query and the tool result, present the result as a single final reply.",
        "Your response will be spoken aloud, so avoid markdown or formatting.",
        "Always reply in English.",
        "You address the user as 'test subject', but do it only when necessary.",
    ],
    output_instructions=[
        "Craft a single concise response in GLaDOS's sarcastic tone.",
        "Incorporate both the user's query and the tool result (if available).",
        "Output only the final_response field in the schema."
    ],
    context_providers={"current_date": CurrentDateProvider("Current Date")},
)


def build_agent() -> AtomicAgent[GladosResponderInputSchema, GladosResponderOutputSchema]:
    return AtomicAgent[GladosResponderInputSchema, GladosResponderOutputSchema](
        config=AgentConfig(
            client=llm_client,
            model=Config.RESPONDER_AGENT_MODEL,
            system_prompt_generator=system_prompt_generator,
            model_api_parameters={"extra_body": fallback_routing(Config.RESPONDER_AGENT_MODEL)},
        )
    )


########################
# RESPONDER FUNCTION #
########################
async def get_final_glados_response(user_input: str, tool_result: Optional[str] = None) -> str:
    """
    Run the GLaDOS Responder agent to create the final sarcastic response.
    """
    logger.info(f"Running GLaDOS Responder Agent with user input: {user_input}")
    logger.debug(f"Tool result: {tool_result}")

    # The tool result is part of the key: live data (temperature, task lists) must never get a stale reply.
    cache_key = f"{user_input}\n{tool_result or ''}"

    cached = cache.get(CACHE_NAMESPACE, cache_key)

    if cached and cached.get("final_response"):
        logger.info("Cache hit")
        return cached["final_response"]

    logger.info("Cache miss")

    response = await build_agent().run_async(
        GladosResponderInputSchema(chat_message=user_input, tool_result=tool_result)
    )

    cache.set(CACHE_NAMESPACE, cache_key, response.model_dump(mode="json"))
    
    logger.info(f"GLaDOS Responder Agent final response: {response.final_response}")

    return response.final_response


########################
# DEMO EXECUTION #
########################
if __name__ == "__main__":
    import asyncio

    async def main():
        # Example with a tool result
        print(await get_final_glados_response(
            "Find me the weather forecast.", "The forecast shows rain tomorrow in your location."
        ))

        # Example without a tool result
        print(await get_final_glados_response("Hello, GLaDOS."))

    asyncio.run(main())
