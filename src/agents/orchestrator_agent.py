from typing import Literal, Optional

from pydantic import Field, ValidationError
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

from src.agents.context_providers import CurrentDateProvider
from src.config import Config
from src.logger import logger
from src.services.llm_cache.llm_cache import cache
from src.services.llm_client import fallback_routing, llm_client

CACHE_NAMESPACE = "orchestrator"


########################
# INPUT/OUTPUT SCHEMAS #
########################
class OrchestratorInputSchema(BaseIOSchema):
    """Input schema for the GLaDOS Agent. Contains the user's message."""

    chat_message: str = Field(..., description="The user's input message to be analyzed and classified.")
    conversation_history: Optional[str] = Field(
        None, description="Earlier exchanges of this conversation, oldest first, if any."
    )


class OrchestratorOutputSchema(BaseIOSchema):
    """
    Output schema for the GLaDOS Agent.
    Contains the tool to be used.
    """
    tool_name: Literal['Home Assistant Tool', 'Grocy Tool', 'SearXNG Tool', 'Vikunja Tool', 'No Tool'] = Field(
        ...,
        description="The name of the tool that should be used to respond to the query. "
                    "Must be one of the following: 'Home Assistant Tool', 'Grocy Tool', 'SearXNG Tool', "
                    "or 'Vikunja Tool'. If the query is a simple conversational message and "
                    "does not require any tool to be used, select 'No Tool'."
    )
    standalone_query: Optional[str] = Field(
        None,
        description="Only when the message relies on the conversation history to be understood "
                    "(e.g. 'turn it off', 'and tomorrow?'): the message rewritten as a self-contained request. "
                    "Otherwise null."
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
        "- Grocy Tool: for groceries and the kitchen: pantry/fridge stock, expiring food, the shopping list, "
        "recipes, the meal plan, and recurring household chores (cleaning, laundry, ...): anything mentioning "
        "chores goes here.",
        "- SearXNG Tool: if the query can be answered with a web search.",
        "- Vikunja Tool: for to-dos, reminders, tasks and projects (but not chores).",
        "- No Tool: if the query is a simple conversational message that does not require any tool.",
        "Earlier exchanges of the conversation may be provided: use them to understand follow-up messages.",
    ],
    output_instructions=[
        "Analyze the input and select the most relevant tool or 'No Tool' if none apply.",
        "If the message is a follow-up that cannot be understood without the earlier exchanges, "
        "fill standalone_query with the request rewritten so the tool can act on it alone.",
        "Format output using the defined schema."
    ],
    context_providers={"current_date": CurrentDateProvider("Current Date")},
)


def build_agent() -> AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema]:
    return AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema](
        config=AgentConfig(
            client=llm_client,
            model=Config.ORCHESTRATOR_AGENT_MODEL,
            system_prompt_generator=system_prompt_generator,
            model_api_parameters={"extra_body": fallback_routing(Config.ORCHESTRATOR_AGENT_MODEL)},
        )
    )


async def route_query(user_input: str, conversation_history: Optional[str] = None) -> OrchestratorOutputSchema:
    """
    Classify the user message: returns the tool to use, and the message made self-contained if it is a follow-up.
    """
    logger.info(f"User query: '{user_input}'")

    # The same message can mean something else after a different conversation ("turn it off").
    cache_key = f"{conversation_history}\n{user_input}" if conversation_history else user_input

    cached = cache.get(CACHE_NAMESPACE, cache_key)

    if cached:
        try:
            result = OrchestratorOutputSchema.model_validate(cached)
            logger.info(f"Cache hit: {cached}")
            return result
        except ValidationError:
            logger.info("Ignoring cache entry that does not match the current schema")

    logger.info("Cache miss")

    output = await build_agent().run_async(
        OrchestratorInputSchema(chat_message=user_input, conversation_history=conversation_history)
    )

    cache.set(CACHE_NAMESPACE, cache_key, output.model_dump(mode="json"))

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
