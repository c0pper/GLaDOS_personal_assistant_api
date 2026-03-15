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

### Testing (no tests currently - recommended to add pytest)
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_orchestrator.py

# Run a single test function
pytest tests/test_orchestrator.py::test_get_tool_name

# Run with coverage
pytest --cov=src --cov-report=html
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
2. **Create context providers** inheriting from `BaseDynamicContextProvider`
3. **Configure AgentConfig** with client, model, history, and system_prompt_generator
4. **Instantiate AtomicAgent** with type hints: `AtomicAgent[InputSchema, OutputSchema]`
5. **Register context providers** and hooks
6. **Create wrapper functions** that call `agent.run()`

```python
agent = AtomicAgent[OrchestratorInputSchema, OrchestratorOutputSchema](config=agent_config)
agent.register_context_provider("current_date", CurrentDateProvider("Current Date"))

def get_tool_name(user_input: str) -> str:
    output = agent.run(OrchestratorInputSchema(chat_message=user_input))
    return output.model_dump()["tool_name"]
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
- `HOME_ASSISTANT_BASE_URL` - Home Assistant URL
- `HOME_ASSISTANT_TOKEN` - Home Assistant token
- `VIKUNJA_BASE_URL` - Vikunja API URL
- `VIKUNJA_TOKEN` - Vikunja API token
