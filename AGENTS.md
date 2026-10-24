# AGENTS.md - Agent Coding Guidelines

This file provides guidelines for agentic coding agents working on this codebase.

## Project Overview

This is a FastAPI-based personal assistant API (GLaDOS) that uses atomic-agents framework for orchestrating different tools (Home Assistant, Grocy, Vikunja, Tavily web search). Python 3.14+ required.

---

## Build, Run & Test Commands

### Development
```bash
# Run the API server locally (port 8001)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload

# Run a specific module directly (for testing agents)
cd src && uv run python -m agents.orchestrator_agent
cd src && uv run python -m agents.glados_responder_agent
cd src && uv run python -m agents.home_assistant_agent
```

### Docker
```bash
# Build and run with docker-compose
docker-compose up --build

# Run container only
docker run -d -p 7000:8000 --env-file .env glados_api:0.1.4
```

### Testing
Tests never call real services: `tests/conftest.py` forces fake env vars and a temporary cache DB,
and provides `fake_llm` (replaces `AtomicAgent.run_async`) and `fake_http` (mocks the Home Assistant/Vikunja clients).
```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_orchestrator.py

# Run a single test function
uv run pytest tests/test_orchestrator.py::test_cache_hit_skips_llm

# Telegram bot (separate uv project and container)
cd telegram_bot && uv run pytest
cd telegram_bot && uv run python bot.py   # needs the API running at GLADOS_API_URL (default http://localhost:8001)
```

### Linting & Type Checking (recommended additions)
```bash
# If using ruff
ruff check src/
ruff format src/

# If using mypy
mypy src/
```

---

## Code Style Guidelines

### Import Organization

Organize imports in the following order, separated by blank lines:
1. Standard library imports
2. Third-party imports
3. Local/relative imports (from src.*)

```python
# Standard library
from typing import Optional, List, Literal
from enum import Enum
import json
import os

# Third-party
from fastapi import APIRouter
from pydantic import Field
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
import instructor
import requests

# Local
from src.config import Config
from src.logger import logger
from src.services.llm_cache.llm_cache import cache
```

### Formatting

- Maximum line length: 120 characters
- Use 4 spaces for indentation (no tabs)
- Use blank lines generously to separate logical sections
- Use section comments with `####` borders for major sections

```python
########################
# INPUT/OUTPUT SCHEMAS #
########################
class OrchestratorInputSchema(BaseIOSchema):
    """Input schema for the GLaDOS Agent."""

    chat_message: str = Field(..., description="The user's input message.")


#####################
# CONTEXT PROVIDERS #
#####################
class CurrentDateProvider(BaseDynamicContextProvider):
    ...
```

### Types

- Use Python type hints for all function signatures
- Use Pydantic `Field` with descriptions for schema fields
- Use `Optional[T]` instead of `T | None` for compatibility
- Use `Literal` for enum-like string constants

```python
def get_tool_name(user_input: str) -> str:
    ...

def invoke_intent(intent_name: str) -> Optional[str]:
    ...
```

### Naming Conventions

- **Functions/variables**: snake_case (`get_tool_name`, `user_query`)
- **Classes**: PascalCase (`OrchestratorInputSchema`, `Config`)
- **Constants**: UPPER_SNAKE_CASE (`OPENROUTER_API_KEY`)
- **Enums**: PascalCase with values as PascalCase or UPPER_SNAKE

```python
class IntentName(str, Enum):
    GetTemperature = "GetTemperature"
    TurnOnCeilingLights = "TurnOnCeilingLights"
```

### Pydantic Schemas

All agent input/output schemas should inherit from `BaseIOSchema`:
```python
class OrchestratorInputSchema(BaseIOSchema):
    """Input schema for the GLaDOS Agent."""

    chat_message: str = Field(..., description="The user's input message.")


class OrchestratorOutputSchema(BaseIOSchema):
    """Output schema for the GLaDOS Agent."""

    tool_name: Literal['Home Assistant Tool', 'Web Search Tool', 'No Tool'] = Field(
        ...,
        description="The name of the tool that should be used."
    )
```

### Error Handling

- Use try/except blocks for external API calls
- Log errors with appropriate level (logger.error for failures)
- Return error dictionaries rather than raising for API endpoints
- Handle None cases explicitly

```python
def invoke_intent(intent_name: str) -> Optional[str]:
    url = f"{Config.HOME_ASSISTANT_BASE_URL}/api/intent/handle"
    headers = {"Authorization": f"Bearer {Config.HOME_ASSISTNAT_TOKEN}"}

    try:
        response = requests.post(url, json={"name": intent_name}, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to invoke intent: {e}")
        return {"error": str(e)}
```

### Agent Patterns

This project uses atomic-agents framework. Follow these patterns:

1. **Define Input/Output schemas** using Pydantic + BaseIOSchema
2. **Create context providers** inheriting from `BaseDynamicContextProvider` (shared ones live in `src/agents/context_providers.py`)
3. **Create one module-level `SystemPromptGenerator`** with its `context_providers`
4. **Build a fresh `AtomicAgent` per call** with the shared async `llm_client` (`src/services/llm_client.py`).
   Agents hold mutable history, so a module-level agent would be shared by concurrent requests.
5. **Create async wrapper functions** that `await agent.run_async()`; never make blocking calls in endpoints
6. **External HTTP calls** use the shared `httpx.AsyncClient`s in `src/services/http_clients.py` (keep-alive + timeouts)
7. **Cache** with a per-agent namespace: `cache.get(CACHE_NAMESPACE, key)`; include any live data in the key

```python
system_prompt_generator = SystemPromptGenerator(
    background=[...],
    context_providers={"current_date": CurrentDateProvider("Current Date")},
)

def build_agent() -> AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema]:
    return AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema](
        config=AgentConfig(client=llm_client, model=Config.ORCHESTRATOR_AGENT_MODEL,
                           system_prompt_generator=system_prompt_generator)
    )

async def route_query(user_input: str) -> OrchestratorOutputSchema:
    return await build_agent().run_async(OrchestratorInputSchema(chat_message=user_input))
```

### Logging

Use the provided logger from `src.logger`:
```python
from src.logger import logger

logger.info(f"User query: '{user_input}'")
logger.error(f"Failed to invoke intent: {response.status_code}")
```

### Configuration

Use dataclasses for configuration, loading from environment variables:
```python
from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Config:
    OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
    MODEL: str = "mistralai/devstral-small"
```

### API Endpoints

Use FastAPI routers in `src/api/endpoints/`:
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class StartRequest(BaseModel):
    message: str

@router.post("/start")
async def start(request: StartRequest):
    ...
```

---

## File Structure

```
src/
├── main.py              # FastAPI app entrypoint
├── config.py            # Configuration dataclass
├── logger.py            # Loguru logger setup
├── api/
│   └── endpoints/
│       ├── start.py     # Main chat endpoint (+ /start_audio, /speech)
│       ├── conversations.py # Clear a conversation's memory
│       └── cache.py     # Cache management
├── agents/
│   ├── orchestrator_agent.py
│   ├── glados_responder_agent.py
│   └── home_assistant_agent.py
└── services/
    ├── conversation_store.py # SQLite conversation memory, keyed by the client's conversation_id
    └── llm_cache/
        └── llm_cache.py # SQLite LLM cache
telegram_bot/            # Separate project/container: forwards Telegram messages and voice notes to the API
└── bot.py
```

Conversation memory: `/start` and `/start_audio` take an optional `conversation_id`. With one, recent exchanges are
given to the orchestrator and the responder, and the orchestrator rewrites follow-ups ("turn it off") into a
`standalone_query` for the tool agents, which never see the history. Without one, every message stands alone.

---

## Environment Variables

Required environment variables (see `.env.example` or docker-compose.yml):
- `GLADOS_API_KEY` - Shared secret every `/api/*` endpoint requires in the `X-API-Key` header (the Telegram bot sends it too)
- `OPENROUTER_API_KEY` - API key for LLM calls
- `JEV_ENABLED` / `JEV_MODEL` - optional fast tool routing through TypeSafe's Jev decision model (disabled by default;
  evaluate with `scripts/jev_benchmark.py` before enabling; pin a version instead of the `~typesafe/jev-latest` alias)
- `TAVILY_MCP_URL` - Tavily MCP server URL for web search (streamable HTTP; the query string carries the API key,
  keep it in `.env`)
- `HA_MCP_URL` - Home Assistant MCP server URL (streamable HTTP; the path contains a secret, keep it in `.env`)
- `GROCY_MCP_URL` - Grocy MCP server URL (streamable HTTP)
- `HOME_ASSISTANT_BASE_URL` / `HOME_ASSISTANT_TOKEN` - Home Assistant REST API, used for speech-to-text (`/api/start_audio`)
- `OPENROUTER_STT_MODEL` - OpenRouter speech-to-text model used first by `/api/start_audio` (optional, default `openai/gpt-4o-mini-transcribe`)
- `OPENROUTER_STT_LANGUAGE` - ISO-639-1 code to force the OpenRouter STT language (optional, default unset: auto-detected)
- `HOME_ASSISTANT_STT_ENTITY` / `HOME_ASSISTANT_STT_LANGUAGE` - fallback STT provider used when OpenRouter fails (the response
  then carries a `warning`) and the language it transcribes (optional, default `stt.faster_whisper` / `en`)
- `HOME_ASSISTANT_TTS_MEDIA_PLAYER` / `HOME_ASSISTANT_TTS_ENTITY` / `HOME_ASSISTANT_TTS_VOICE` - Music Assistant player (replies are announcements, so music resumes), TTS entity and voice for spoken replies to `/api/start_audio` (optional, default `media_player.minisgooglehome_2` / `tts.piper` / `glados`)
- `VIKUNJA_BASE_URL` - Vikunja API URL
- `VIKUNJA_TOKEN` - Vikunja API token
- `TELEGRAM_TOKEN` / `TELEGRAM_ALLOWED_CHAT_IDS` - Telegram bot token and the comma-separated chat ids it answers
  (bot container only; `GLADOS_API_URL` points it at the API)
