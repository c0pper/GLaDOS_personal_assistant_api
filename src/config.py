from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    # Shared secret every /api endpoint requires in the X-API-Key header (the Telegram bot sends it too).
    GLADOS_API_KEY: str = os.environ["GLADOS_API_KEY"]
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
    HOME_ASSISTANT_MAX_TOOL_ROUNDS: int = 20
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

    GROCY_MCP_URL: str = os.environ["GROCY_MCP_URL"]
    GROCY_AGENT_MODEL: str = "deepseek/deepseek-v4.1-flash"
    GROCY_MAX_TOOL_ROUNDS: int = 12
    GROCY_TIMEOUT_SECONDS: float = 90.0
    # Everyday kitchen tools only: no generic entity create/update/delete, file, label-printing or
    # clear-shopping-list tools, and no Grocy tasks (to-dos go to Vikunja).
    GROCY_MCP_TOOLS: tuple = (
        "stock_overview_tool",
        "stock_expiring_tool",
        "stock_product_info_tool",
        "stock_search_tool",
        "stock_add_tool",
        "stock_consume_tool",
        "stock_open_tool",
        "stock_transfer_tool",
        "locations_list_tool",
        "shopping_list_view_tool",
        "shopping_list_add_tool",
        "shopping_list_remove_tool",
        "shopping_list_set_amount_tool",
        "shopping_list_set_note_tool",
        "shopping_list_add_missing_tool",
        "recipes_list_tool",
        "recipe_details_tool",
        "recipe_fulfillment_tool",
        "recipe_add_to_shopping_tool",
        "meal_plan_list_tool",
        "meal_plan_summary_tool",
        "meal_plan_add_tool",
        "chores_list_tool",
        "chores_overdue_tool",
        "chore_execute_tool",
        "discover_candidates_tool",
        "calendar_summary_tool",
    )

    # Speech-to-text goes through OpenRouter; if that fails, through Home Assistant's STT API as a fallback
    # (whose language must be one the entity supports).
    OPENROUTER_STT_MODEL: str = os.environ.get("OPENROUTER_STT_MODEL", "openai/gpt-4o-mini-transcribe")
    # ISO-639-1 code to force a language; unset, the model detects it (e.g. Italian or English).
    OPENROUTER_STT_LANGUAGE: Optional[str] = os.environ.get("OPENROUTER_STT_LANGUAGE") or None
    OPENROUTER_STT_TIMEOUT_SECONDS: float = 20.0
    HOME_ASSISTANT_BASE_URL: str = os.environ["HOME_ASSISTANT_BASE_URL"]
    HOME_ASSISTANT_TOKEN: str = os.environ["HOME_ASSISTANT_TOKEN"]
    HOME_ASSISTANT_STT_ENTITY: str = os.environ.get("HOME_ASSISTANT_STT_ENTITY", "stt.faster_whisper")
    HOME_ASSISTANT_STT_LANGUAGE: str = os.environ.get("HOME_ASSISTANT_STT_LANGUAGE", "en")
    STT_TIMEOUT_SECONDS: float = 60.0
    # Replies to voice requests are announced on this Music Assistant player (so music resumes afterwards),
    # rendered with this Home Assistant TTS entity and voice.
    HOME_ASSISTANT_TTS_MEDIA_PLAYER: str = os.environ.get(
        "HOME_ASSISTANT_TTS_MEDIA_PLAYER", "media_player.minisgooglehome_2"
    )
    HOME_ASSISTANT_TTS_ENTITY: str = os.environ.get("HOME_ASSISTANT_TTS_ENTITY", "tts.piper")
    HOME_ASSISTANT_TTS_VOICE: str = os.environ.get("HOME_ASSISTANT_TTS_VOICE", "glados")
    ANNOUNCEMENT_TIMEOUT_SECONDS: float = 180.0

    # OpenRouter retries on these, in order, when an agent's model is rate-limited or down.
    LLM_FALLBACK_MODELS: tuple = ("openai/gpt-4o-mini", "deepseek/deepseek-v4.1-flash")
    LLM_TIMEOUT_SECONDS: float = 30.0
    HTTP_TIMEOUT_SECONDS: float = 10.0
    VIKUNJA_PROJECTS_TTL_SECONDS: float = 600.0

    LLM_CACHE_DB_PATH: str = os.environ.get("LLM_CACHE_DB_PATH", "db/llm_cache.db")

    # Conversation memory for clients that send a conversation_id (e.g. the Telegram bot): the last exchanges
    # are given to the orchestrator and the responder, as long as they are recent enough to still be relevant.
    CONVERSATION_DB_PATH: str = os.environ.get("CONVERSATION_DB_PATH", "db/conversations.db")
    CONVERSATION_MAX_TURNS: int = 10
    CONVERSATION_MAX_AGE_SECONDS: float = 6 * 3600