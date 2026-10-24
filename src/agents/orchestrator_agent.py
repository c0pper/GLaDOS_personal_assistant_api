from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import Field, ValidationError
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

from src.agents.context_providers import CurrentDateProvider
from src.config import Config
from src.logger import logger
from src.services import jev_client
from src.services.llm_cache.llm_cache import cache
from src.services.llm_client import fallback_routing, llm_client

CACHE_NAMESPACE = "orchestrator"

# Single source of truth for the tool descriptions: used by both the LLM prompt and Jev's choice criteria.
TOOL_DESCRIPTIONS = {
    "Home Assistant Tool": "Lights, appliances, presence, temperature, and other home control.",
    "Grocy Tool": "Groceries and the kitchen: pantry/fridge stock, expiring food, the shopping list, "
                  "recipes, the meal plan, and recurring household chores (cleaning, laundry, ...). "
                  "Anything mentioning chores goes here.",
    "Web Search Tool": "Questions that can be answered with a web search.",
    "Vikunja Tool": "To-dos, reminders, tasks and projects (but not chores).",
    "No Tool": "A simple conversational message that does not require any tool.",
}


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
    tool_name: Literal['Home Assistant Tool', 'Grocy Tool', 'Web Search Tool', 'Vikunja Tool', 'No Tool'] = Field(
        ...,
        description="The name of the tool that should be used to respond to the query. "
                    "Must be one of the following: 'Home Assistant Tool', 'Grocy Tool', 'Web Search Tool', "
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
        *(f"- {tool_name}: {description}" for tool_name, description in TOOL_DESCRIPTIONS.items()),
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


def build_jev_request(
    user_input: str, conversation_history: Optional[str]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """The state and typed questions sent to Jev, matching the LLM prompt above."""
    questions: Dict[str, Any] = {
        "tool": {
            "type": "choice",
            "instructions": "Which tool should handle the user's latest message?",
            "criteria": TOOL_DESCRIPTIONS,
        }
    }
    state: Dict[str, Any] = {
        "message": user_input,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }
    if conversation_history:
        state["conversation_history"] = conversation_history
        questions["needs_rewrite"] = {
            "type": "noul",
            "instructions": "Does the latest message depend on the earlier exchanges to be understood?",
            "criteria": {
                "true": "It cannot be understood without the earlier exchanges (e.g. 'turn it off', 'and tomorrow?').",
                "false": "It is understandable on its own.",
            },
        }
    return state, questions


def _choice_answer(response: jev_client.JevResponse, question: str) -> Optional[jev_client.JevChoiceAnswer]:
    answer = response.answers.get(question)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return None
    try:
        return jev_client.JevChoiceAnswer.model_validate(answer)
    except ValidationError:
        return None


async def _route_with_jev(
    user_input: str, conversation_history: Optional[str]
) -> Optional[OrchestratorOutputSchema]:
    """
    Ask Jev to classify the message. Returns None whenever the normal orchestrator should decide:
    Jev unavailable, a malformed answer, low confidence, or a follow-up that needs rewriting.
    """
    state, questions = build_jev_request(user_input, conversation_history)

    try:
        response = await jev_client.decide(state=state, questions=questions)
    except jev_client.JevError:
        return None

    logger.debug(f"Jev answers: {response.answers}")

    tool = _choice_answer(response, "tool")
    if tool is None or tool.choice not in TOOL_DESCRIPTIONS:
        logger.warning(f"Jev returned an unusable tool answer: {response.answers.get('tool')!r}")
        return None

    threshold = Config.JEV_NO_TOOL_MIN_CONFIDENCE if tool.choice == "No Tool" else Config.JEV_MIN_CONFIDENCE
    if tool.confidence is None or tool.confidence < threshold:
        logger.info(f"Jev route '{tool.choice}' is below confidence {threshold} ({tool.confidence}): using the LLM")
        return None

    if conversation_history:
        try:
            probability = jev_client.JevNoulAnswer.model_validate(
                response.answers.get("needs_rewrite")
            ).noul
        except ValidationError:
            logger.warning("Jev did not answer needs_rewrite: using the LLM")
            return None
        if probability > Config.JEV_FOLLOW_UP_THRESHOLD:
            logger.info(
                f"Jev route '{tool.choice}' (confidence {tool.confidence}) needs a rewrite "
                f"(follow-up {probability:.2f}): using the LLM for a self-contained query"
            )
            return None

    logger.info(f"Jev route: {tool.choice} (confidence {tool.confidence})")
    return OrchestratorOutputSchema(tool_name=tool.choice, standalone_query=None)


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

    output = None
    if Config.JEV_ENABLED:
        output = await _route_with_jev(user_input, conversation_history)

    if output is None:
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

        # Example 2: Web search query
        print(await route_query("Find a recipe for cake. Not that you'd know how to use an oven."))

        # Example 3: Vikunja query
        print(await route_query("Add 'Buy more test subjects' to my to-do list."))

        # Example 4: The "no tool" case
        print(await route_query("Hello, you look like a fat idiot from where I'm standing."))

    asyncio.run(main())
