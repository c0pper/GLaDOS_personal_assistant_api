from typing import Any, Callable, Dict, List
from dataclasses import dataclass
import tempfile
import os

# Must run before any `src` import: Config reads the environment at import time.
# Values are forced (not defaulted) so tests can never reach real services or the real cache DB.
os.environ.update({
    "OPENROUTER_API_KEY": "test-key",
    "SEARXNG_URL": "http://searxng.test",
    "VIKUNJA_BASE_URL": "http://vikunja.test/api/v1",
    "VIKUNJA_TOKEN": "vikunja-token",
    "HOME_ASSISTANT_BASE_URL": "http://ha.test",
    "HOME_ASSISTANT_TOKEN": "ha-token",
    "LLM_CACHE_DB_PATH": os.path.join(tempfile.mkdtemp(prefix="glados-tests-"), "llm_cache.db"),
})

import pytest
import httpx
from atomic_agents import AtomicAgent, BaseIOSchema

from src.agents import home_assistant_agent, vikunja_agent
from src.services.llm_cache.llm_cache import cache


###########
# FAKE LLM #
###########
@dataclass
class LLMCall:
    output_schema: type
    user_input: BaseIOSchema
    history_length: int
    system_prompt: str


class FakeLLM:
    """
    Replaces AtomicAgent.run_async. Register a handler per output schema:
    a callable taking the input schema and returning an output schema instance.
    """

    def __init__(self):
        self.handlers: Dict[type, Callable[[BaseIOSchema], BaseIOSchema]] = {}
        self.calls: List[LLMCall] = []

    def on(self, output_schema: type, handler: Callable[[BaseIOSchema], BaseIOSchema]) -> None:
        self.handlers[output_schema] = handler

    def calls_for(self, output_schema: type) -> List[LLMCall]:
        return [call for call in self.calls if call.output_schema is output_schema]

    async def run_async(self, agent: AtomicAgent, user_input: BaseIOSchema) -> BaseIOSchema:
        self.calls.append(LLMCall(
            output_schema=agent.output_schema,
            user_input=user_input,
            history_length=len(agent.history.get_history()),
            system_prompt=agent.system_prompt_generator.generate_prompt(),
        ))
        handler = self.handlers.get(agent.output_schema)
        if handler is None:
            raise AssertionError(f"Unexpected LLM call for {agent.output_schema.__name__}")
        return handler(user_input)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    llm = FakeLLM()

    async def run_async(agent, user_input=None):
        return await llm.run_async(agent, user_input)

    monkeypatch.setattr(AtomicAgent, "run_async", run_async)
    return llm


#############
# FAKE HTTP #
#############
class FakeHTTP:
    """Routes (method, path) to a handler returning an httpx.Response; records every request."""

    def __init__(self):
        self.routes: Dict[tuple, Callable[[httpx.Request], httpx.Response]] = {}
        self.requests: List[httpx.Request] = []

    def on(self, method: str, path: str, handler: Any) -> None:
        # Handler may be a static response or a callable taking the request.
        self.routes[(method, path)] = handler if callable(handler) else (lambda request: handler)

    def requests_to(self, path: str) -> List[httpx.Request]:
        return [request for request in self.requests if request.url.path == path]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.routes.get((request.method, request.url.path))
        if handler is None:
            return httpx.Response(404, text=f"No fake route for {request.method} {request.url.path}")
        return handler(request)


@pytest.fixture
async def fake_http(monkeypatch) -> FakeHTTP:
    http = FakeHTTP()
    transport = httpx.MockTransport(http.handle)

    clients = []
    for module, attribute in [(home_assistant_agent, "home_assistant_client"), (vikunja_agent, "vikunja_client")]:
        original: httpx.AsyncClient = getattr(module, attribute)
        client = httpx.AsyncClient(
            base_url=original.base_url, headers=original.headers, timeout=original.timeout, transport=transport
        )
        monkeypatch.setattr(module, attribute, client)
        clients.append(client)

    yield http

    for client in clients:
        await client.aclose()


##################
# STATE ISOLATION #
##################
@pytest.fixture(autouse=True)
def reset_state():
    cache.wipe()
    provider = vikunja_agent.available_projects_provider
    provider.projects = None
    provider.fetched_at = 0.0
    yield
    cache.wipe()
