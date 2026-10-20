from openai import AsyncOpenAI
import instructor

from src.config import Config
from src.logger import logger


#########
# HOOKS #
#########
def on_parse_error(error):
    """Log structured output validation errors."""
    logger.error(f"LLM output validation failed: {error}")


def on_completion_error(error, **kwargs):
    """Log LLM API errors."""
    logger.error(f"LLM API error: {type(error).__name__}: {error}")


##########
# CLIENT #
##########
# A single async client (and connection pool) shared by every agent.
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY,
    timeout=Config.LLM_TIMEOUT_SECONDS,
    max_retries=2,
)

llm_client = instructor.from_openai(openrouter_client)

# Hooks are registered on the client, so they are registered once here rather than per agent.
llm_client.on("parse:error", on_parse_error)
llm_client.on("completion:error", on_completion_error)
