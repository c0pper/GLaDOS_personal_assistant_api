from enum import Enum
from typing import Optional

from pydantic import Field
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator, BaseDynamicContextProvider
import httpx

from src.config import Config
from src.logger import logger
from src.services.http_clients import home_assistant_client
from src.services.llm_cache.llm_cache import cache
from src.services.llm_client import llm_client

CACHE_NAMESPACE = "home_assistant_intent"


##########################
# CENTRAL INTENTS CONFIG #
##########################
class IntentName(str, Enum):
    GetTemperature = "GetTemperature"
    GetStateMother = "GetStateMother"
    GetStateFather = "GetStateFather"
    TurnOnCeilingLights = "TurnOnCeilingLights"
    TurnOffCeilingLights = "TurnOffCeilingLights"
    TurnOnMoodLights = "TurnOnMoodLights"
    TurnOffMoodLights = "TurnOffMoodLights"
    TurnOnAllBedroomLights = "TurnOnAllBedroomLights"
    TurnOffAllBedroomLights = "TurnOffAllBedroomLights"
    TurnOnKeyboardStrip = "TurnOnKeyboardStrip"
    TurnOffKeyboardStrip = "TurnOffKeyboardStrip"


########################
# INPUT/OUTPUT SCHEMAS #
########################
class HomeAssistantInputSchema(BaseIOSchema):
    """Input schema for Home Assistant Agent."""
    user_query: str = Field(..., description="The user's home automation query to be processed.")


class HomeAssistantOutputSchema(BaseIOSchema):
    """Output schema for Home Assistant Agent."""
    intent_name: IntentName = Field(
        ..., description="The intent name that matches the user query."
    )


#####################
# CONTEXT PROVIDERS #
#####################
class AvailableIntentsProvider(BaseDynamicContextProvider):
    def __init__(self, title):
        super().__init__(title)
        self.intents = [intent.value for intent in IntentName]

    def get_info(self) -> str:
        return "Available Intents:\n" + "\n".join(self.intents)


######################
# HOME AGENT CONFIG #
######################
# Shared, stateless pieces. A fresh agent (with empty history) is built per call so that
# concurrent requests never share or grow a conversation history.
system_prompt_generator = SystemPromptGenerator(
    background=[
        "You are a Home Assistant agent.",
        "Your primary role is to choose the right intent name among those available based on the user query.",
    ],
    output_instructions=[
        "Select exactly one intent name from the available intents.",
        "Format output according to schema."
    ],
    context_providers={"available_intents": AvailableIntentsProvider("Available Intents")},
)


def build_agent() -> AtomicAgent[HomeAssistantInputSchema, HomeAssistantOutputSchema]:
    return AtomicAgent[HomeAssistantInputSchema, HomeAssistantOutputSchema](
        config=AgentConfig(
            client=llm_client,
            model=Config.HOME_ASSISTANT_AGENT_MODEL,
            system_prompt_generator=system_prompt_generator,
        )
    )


#########################
# HOME ASSISTANT METHODS #
#########################
async def invoke_intent(intent_name: str) -> dict:
    """
    Call Home Assistant's intent API with the given intent.
    """
    try:
        response = await home_assistant_client.post("/api/intent/handle", json={"name": intent_name})
    except httpx.HTTPError as e:
        logger.error(f"Failed to invoke intent: {e}")
        return {"error": f"Failed to invoke intent: {e}"}

    if response.status_code in (200, 201):
        result = response.json()
        logger.info(f"Intent invoked successfully: {result}")
        return result

    logger.error(f"Failed to invoke intent: {response.status_code} - {response.text}")
    return {"error": f"Failed to invoke intent: {response.status_code} - {response.text}"}


async def get_intent_name(user_query: str) -> str:
    """
    Run the agent with the user's query.
    """
    logger.info(f"Running Home Assistant Agent with user query: {user_query}")

    cached = cache.get(CACHE_NAMESPACE, user_query)

    if cached and cached.get("intent_name"):
        logger.info("Cache hit")
        return cached["intent_name"]

    logger.info("Cache miss")

    response = await build_agent().run_async(HomeAssistantInputSchema(user_query=user_query))

    result = response.model_dump(mode="json")

    cache.set(CACHE_NAMESPACE, user_query, result)

    return result["intent_name"]


async def run_home_assistant_agent(user_query: str, intent_name: Optional[str] = None) -> dict:
    """
    Invoke the Home Assistant intent for the user's query.
    If the orchestrator already picked the intent, the intent LLM call is skipped.
    """
    if not intent_name:
        intent_name = await get_intent_name(user_query)

    logger.info(f"Invoking intent: {intent_name}")
    return await invoke_intent(intent_name)


#####################
# DEMO EXECUTION #
#####################
if __name__ == "__main__":
    import asyncio

    async def main():
        # Example 1: Turn on mood lights
        print(await run_home_assistant_agent("Turn on the mood lights"))

        # Example 2: Get temperature
        print(await run_home_assistant_agent("What is the temperature right now?"))

        # Example 3: Turn on ceiling lights
        print(await run_home_assistant_agent("Turn on the ceiling lights"))

    asyncio.run(main())
