from typing import Optional
from pydantic import Field
from openai import OpenAI as OpenRouterClient
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator, BaseDynamicContextProvider, ChatHistory
from datetime import datetime
import instructor
from src.config import Config
from src.logger import logger
from src.services.llm_cache.llm_cache import cache


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


#####################
# CONTEXT PROVIDERS #
#####################
class CurrentDateProvider(BaseDynamicContextProvider):
    def __init__(self, title):
        super().__init__(title)
        self.date = datetime.now().strftime("%Y-%m-%d")

    def get_info(self) -> str:
        return f"Current date in format YYYY-MM-DD: {self.date}"


########################
# RESPONDER AGENT CONFIG #
########################
glados_responder_config = AgentConfig(
    client=instructor.from_openai(
        OpenRouterClient(base_url="https://openrouter.ai/api/v1", api_key=Config.OPENROUTER_API_KEY),
    ),
    model=Config.RESPONDER_AGENT_MODEL,  
    history=ChatHistory(max_messages=10),
    system_prompt_generator=SystemPromptGenerator(
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
    ),
)

responder_agent = AtomicAgent[GladosResponderInputSchema, GladosResponderOutputSchema](config=glados_responder_config)

responder_agent.register_context_provider("current_date", CurrentDateProvider("Current Date"))


########################
# RESPONDER FUNCTION #
########################
def get_final_glados_response(user_input: str, tool_result: Optional[str] = None) -> str:
    """
    Run the GLaDOS Responder agent to create the final sarcastic response.
    """

    logger.info(f"Running GLaDOS Responder Agent with user input: {user_input} and tool result: {tool_result}")
    
    cached = cache.get(user_input)

    if cached:
        if cached.get("final_response"):
            logger.info("Cache hit")
            return cached["final_response"]
    
    logger.info("Cache miss")

    response = responder_agent.run(
        GladosResponderInputSchema(chat_message=user_input, tool_result=tool_result)
    )

    result = response.model_dump()

    cache.set(user_input, result)

    return response.final_response


########################
# DEMO EXECUTION #
########################
if __name__ == "__main__":
    # Example with a tool result
    user_query_1 = "Find me the weather forecast."
    tool_result_1 = "The forecast shows rain tomorrow in your location."
    print(get_final_glados_response(user_query_1, tool_result_1))

    # Example without a tool result
    user_query_2 = "Hello, GLaDOS."
    print(get_final_glados_response(user_query_2))
