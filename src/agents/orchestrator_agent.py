from typing import Literal, Optional

from pydantic import Field, ValidationError
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

from src.agents.context_providers import CurrentDateProvider
from src.agents.home_assistant_agent import IntentName, AvailableIntentsProvider
from src.config import Config
from src.logger import logger
from src.services.llm_cache.llm_cache import cache
from src.services.llm_client import llm_client

CACHE_NAMESPACE = "orchestrator"


########################
# INPUT/OUTPUT SCHEMAS #
########################
class OrchestratorInputSchema(BaseIOSchema):
    """Input schema for the GLaDOS Agent. Contains the user's message."""

    chat_message: str = Field(..., description="The user's input message to be analyzed and classified.")


class OrchestratorOutputSchema(BaseIOSchema):
    """
    Output schema for the GLaDOS Agent.
    Contains the tool to be used and, for Home Assistant queries, the intent to invoke.
    """
    tool_name: Literal['Home Assistant Tool', 'SearXNG Tool', 'Vikunja Tool', 'No Tool'] = Field(
        ...,
        description="The name of the tool that should be used to respond to the query. "
                    "Must be one of the following: 'Home Assistant Tool', 'SearXNG Tool', "
                    "or 'Vikunja Tool'. If the query is a simple conversational message and "
                    "does not require any tool to be used, select 'No Tool'."
    )
    home_assistant_intent: Optional[IntentName] = Field(
        None,
        description="Only when tool_name is 'Home Assistant Tool': the available intent that matches the query. "
                    "Leave null for every other tool."
    )


######################
# Orchestrator AGENT CONFIG #
######################
# Shared, stateless pieces. A fresh agent (with empty history) is built per call so that
# concurrent requests never share or grow a conversation history.
system_prompt_generator = SystemPromptGenerator(
    background=[
        "You are an intent detector.",
        "Your task is to classify the user query and decide which tool, if any, should be used.",
        "Available tools:",
        "- Home Assistant Tool: if the query is about lights, appliances, presence, temperature.",
        "- SearXNG Tool: if the query can be answered with a web search.",
        "- Vikunja Tool: for handling tasks and projects.",
        "- No Tool: if the query is a simple conversational message that does not require any tool.",
    ],
    output_instructions=[
        "Analyze the input and select the most relevant tool or 'No Tool' if none apply.",
        "If the tool is 'Home Assistant Tool', also select exactly one intent from the available intents.",
        "Format output using the defined schema."
    ],
    context_providers={
        "current_date": CurrentDateProvider("Current Date"),
        "available_intents": AvailableIntentsProvider("Available Home Assistant Intents"),
    },
)


def build_agent() -> AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema]:
    return AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema](
        config=AgentConfig(
            client=llm_client,
            model=Config.ORCHESTRATOR_AGENT_MODEL,
            system_prompt_generator=system_prompt_generator,
        )
    )


async def route_query(user_input: str) -> OrchestratorOutputSchema:
    """
    Classify the user message: returns the tool to use and, for Home Assistant, the intent.
    """
    logger.info(f"User query: '{user_input}'")

    cached = cache.get(CACHE_NAMESPACE, user_input)

    if cached:
        try:
            result = OrchestratorOutputSchema.model_validate(cached)
            logger.info(f"Cache hit: {cached}")
            return result
        except ValidationError:
            logger.info("Ignoring cache entry that does not match the current schema")

    logger.info("Cache miss")

    output = await build_agent().run_async(OrchestratorInputSchema(chat_message=user_input))

    cache.set(CACHE_NAMESPACE, user_input, output.model_dump(mode="json"))

    logger.info(f"Fresh result: {output.model_dump(mode='json')}")

    return output


if __name__ == "__main__":
    #####################
    # DEMO EXECUTION #
    #####################
    import asyncio

    async def main():
        # Example 1: Home Assistant query
        print(await route_query("Turn off the living room lights, you incompetent simpleton."))

        # Example 2: SearXNG query
        print(await route_query("Find a recipe for cake. Not that you'd know how to use an oven."))

        # Example 3: Vikunja query
        print(await route_query("Add 'Buy more test subjects' to my to-do list."))

        # Example 4: The "no tool" case
        print(await route_query("Hello, you look like a fat idiot from where I'm standing."))

    asyncio.run(main())
