# AGENTS.md - Agent Coding Guidelines

This file provides guidelines for agentic coding agents working on this codebase.

## Project Overview

This is a FastAPI-based personal assistant API (GLaDOS) that uses atomic-agents framework for orchestrating different tools (Home Assistant, SearXNG, Vikunja). Python 3.14+ required.

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

    tool_name: Literal['Home Assistant Tool', 'SearXNG Tool', 'No Tool'] = Field(
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
│       ├── start.py     # Main chat endpoint
│       └── cache.py     # Cache management
├── agents/
│   ├── orchestrator_agent.py
│   ├── glados_responder_agent.py
│   └── home_assistant_agent.py
└── services/
    └── llm_cache/
        └── llm_cache.py # SQLite LLM cache
```

---

## Environment Variables

Required environment variables (see `.env.example` or docker-compose.yml):
- `OPENROUTER_API_KEY` - API key for LLM calls
- `SEARXNG_URL` - SearXNG instance URL
- `HA_MCP_URL` - Home Assistant MCP server URL (streamable HTTP; the path contains a secret, keep it in `.env`)
- `HOME_ASSISTANT_BASE_URL` / `HOME_ASSISTANT_TOKEN` - Home Assistant REST API, used for speech-to-text (`/api/start_audio`)
- `HOME_ASSISTANT_STT_ENTITY` / `HOME_ASSISTANT_STT_LANGUAGE` - STT provider and a language it supports (optional, default `stt.faster_whisper` / `en`)
- `VIKUNJA_BASE_URL` - Vikunja API URL
- `VIKUNJA_TOKEN` - Vikunja API token
