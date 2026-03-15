from typing import Union, Literal
from openai import OpenAI as OpenRouterClient
from pydantic import Field
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator, BaseDynamicContextProvider
from src.config import Config
from src.logger import logger
import instructor
from datetime import datetime
from src.services.llm_cache.llm_cache import cache



def on_parse_error(error):
    """Handle validation errors."""
    print(f"Validation failed: {error}")


def on_completion_kwargs(**kwargs):
    """Log API call details before request."""
    model = kwargs.get("model", "unknown")
    # print(f"Calling model: {model}")


def on_completion_response(response, **kwargs):
    """Process successful responses."""
    if hasattr(response, "usage"):
        # print(f"Tokens used: {response.usage.total_tokens}")
        pass


def on_completion_error(error, **kwargs):
    """Handle API errors."""
    print(f"API error: {type(error).__name__}: {error}")

########################
# INPUT/OUTPUT SCHEMAS #
########################
class OrchestratorInputSchema(BaseIOSchema):
    """Input schema for the GLaDOS Agent. Contains the user's message."""

    chat_message: str = Field(..., description="The user's input message to be analyzed and classified.")


class OrchestratorOutputSchema(BaseIOSchema):
    """
    Updated output schema for the GLaDOS Agent.
    It now contains the name of the tool to be used, including a 'No Tool' option.
    """
    tool_name: Literal['Home Assistant Tool', 'SearXNG Tool', 'Vikunja Tool', 'No Tool'] = Field(
        ...,
        description="The name of the tool that should be used to respond to the query. "
                    "Must be one of the following: 'Home Assistant Tool', 'SearXNG Tool', "
                    "or 'Vikunja Tool'. If the query is a simple conversational message and "
                    "does not require any tool to be used, select 'No Tool'."
    )


#####################
# CONTEXT PROVIDERS #
#####################
class CurrentDateProvider(BaseDynamicContextProvider):
    def __init__(self, title):
        super().__init__(title)
        self.date = datetime.now().strftime("%Y-%m-%d")

    def get_info(self) -> str:
        return f"Current date in format YYYY-MM-DD: {self.date}"


######################
# Orchestrator AGENT CONFIG #
######################
orchestrator_agent_config = AgentConfig(
    client=instructor.from_openai(
        OpenRouterClient(base_url="https://openrouter.ai/api/v1", api_key=Config.OPENROUTER_API_KEY),
    ),
    history=ChatHistory(max_messages=10),
    model = Config.ORCHESTRATOR_AGENT_MODEL,
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
            "Provide only the name of that tool.",
            "Format output using the defined schema."
        ],
    )
)

agent = AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema](config=orchestrator_agent_config)

# Register context providers
agent.register_context_provider("current_date", CurrentDateProvider("Current Date"))

# Register hooks
agent.register_hook("parse:error", on_parse_error)
agent.register_hook("completion:kwargs", on_completion_kwargs)
agent.register_hook("completion:response", on_completion_response)
agent.register_hook("completion:error", on_completion_error)


def get_tool_name(user_input: str) -> str:
    """
    A simple function to demonstrate how to use the updated agent.
    It takes a user message and returns the name of the selected tool.
    """
    logger.info(f"User query: '{user_input}'")
    
    cached = cache.get(user_input)

    if cached:
        if cached.get("tool_name"):
            logger.info("Cache hit")
            return cached["tool_name"]

    logger.info("Cache miss")

    # Run the agent with the user's message
    output = agent.run(OrchestratorInputSchema(chat_message=user_input))

    result = output.model_dump()

    cache.set(user_input, result)

    return result["tool_name"]


if __name__ == "__main__":
    #####################
    # DEMO EXECUTION #
    #####################

    # Example 1: Home Assistant query
    tool_1 = get_tool_name("Turn off the living room lights, you incompetent simpleton.")

    # Example 2: SearXNG query
    tool_2 = get_tool_name("Find a recipe for cake. Not that you'd know how to use an oven.")

    # Example 3: Vikunja query
    tool_3 = get_tool_name("Add 'Buy more test subjects' to my to-do list.")
    
    # Example 4: The new "no tool" case
    tool_4 = get_tool_name("Hello, you look like a fat idiot from where I'm standing.")

