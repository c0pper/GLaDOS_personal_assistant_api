from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
    SEARXNG_URL: str = os.environ["SEARXNG_URL"]

    ORCHESTRATOR_AGENT_MODEL: str = "mistralai/devstral-small"
    RESPONDER_AGENT_MODEL: str = "openai/gpt-4o-mini"
    GEMINI_AGENT_MODEL: str = "google/gemini-2.5-flash-lite"

    VIKUNJA_BASE_URL: str = os.environ["VIKUNJA_BASE_URL"]
    VIKUNJA_TOKEN: str = os.environ["VIKUNJA_TOKEN"]
    VIKUNJA_AGENT_MODEL: str = "google/gemini-2.5-flash-lite"

    HOME_ASSISTANT_AGENT_MODEL: str = "google/gemini-2.5-flash-lite"
    HOME_ASSISTNAT_TOKEN: str = os.environ["HOME_ASSISTANT_TOKEN"]
    HOME_ASSISTANT_BASE_URL: str = os.environ["HOME_ASSISTANT_BASE_URL"]