from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
    SEARXNG_URL: str = os.environ["SEARXNG_URL"]

    ORCHESTRATOR_AGENT_MODEL: str = "google/gemini-2.5-flash-lite"
    RESPONDER_AGENT_MODEL: str = "openai/gpt-4o-mini"
    GEMINI_AGENT_MODEL: str = "google/gemini-2.5-flash-lite"

    VIKUNJA_BASE_URL: str = os.environ["VIKUNJA_BASE_URL"]
    VIKUNJA_TOKEN: str = os.environ["VIKUNJA_TOKEN"]
    VIKUNJA_AGENT_MODEL: str = "google/gemini-2.5-flash-lite"

    HA_MCP_URL: str = os.environ["HA_MCP_URL"]
    HOME_ASSISTANT_AGENT_MODEL: str = "deepseek/deepseek-v4.1-flash"
    HOME_ASSISTANT_MAX_TOOL_ROUNDS: int = 8
    HOME_ASSISTANT_TIMEOUT_SECONDS: float = 90.0
    # Only everyday control/read tools: the MCP server also exposes admin ones (restart, backups, deletions),
    # and every exposed tool definition is sent with each LLM call.
    HOME_ASSISTANT_MCP_TOOLS: tuple = (
        "ha_search",
        "ha_get_state",
        "ha_get_overview",
        "ha_list_floors_areas",
        "ha_call_service",
        "ha_get_operation_status",
        "ha_list_services",
        "ha_get_history",
        "ha_config_get_calendar_events",
        "ha_get_todo",
    )

    LLM_TIMEOUT_SECONDS: float = 30.0
    HTTP_TIMEOUT_SECONDS: float = 10.0
    VIKUNJA_PROJECTS_TTL_SECONDS: float = 600.0

    LLM_CACHE_DB_PATH: str = os.environ.get("LLM_CACHE_DB_PATH", "db/llm_cache.db")