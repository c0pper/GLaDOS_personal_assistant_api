from typing import Any, Callable, Dict, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
import tempfile
import asyncio
import json
import os

# Must run before any `src` import: Config reads the environment at import time.
# Values are forced (not defaulted) so tests can never reach real services or the real cache DB.
os.environ.update({
    "OPENROUTER_API_KEY": "test-key",
    "SEARXNG_URL": "http://searxng.test",
    "VIKUNJA_BASE_URL": "http://vikunja.test/api/v1",
    "VIKUNJA_TOKEN": "vikunja-token",
    "HA_MCP_URL": "http://ha-mcp.test/mcp",
    "HOME_ASSISTANT_BASE_URL": "http://ha.test",
    "HOME_ASSISTANT_TOKEN": "ha-token",
    "LLM_CACHE_DB_PATH": os.path.join(tempfile.mkdtemp(prefix="glados-tests-"), "llm_cache.db"),
})

import pytest
import httpx
from atomic_agents import AtomicAgent, BaseIOSchema
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from openai.types.chat import ChatCompletion

from src.agents import home_assistant_agent, vikunja_agent
from src.services import speech_to_text, text_to_speech
from src.config import Config
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
    for module, attribute in [(vikunja_agent, "vikunja_client"), (speech_to_text, "home_assistant_client"),
                              (text_to_speech, "home_assistant_client")]:
        original: httpx.AsyncClient = getattr(module, attribute)
        client = httpx.AsyncClient(
            base_url=original.base_url, headers=original.headers, timeout=original.timeout, transport=transport
        )
        monkeypatch.setattr(module, attribute, client)
        clients.append(client)

    yield http

    # Background announcements still use the fake clients: let them finish before closing.
    await asyncio.gather(*text_to_speech._announcements)
    for client in clients:
        await client.aclose()


###################
# FAKE HA MCP LOOP #
###################
class FakeHomeAssistant:
    """
    Replaces the HA MCP session and the tool-calling LLM.
    `replies` are scripted LLM turns: a string is a final answer, a list of (tool, arguments) is a tool-call turn.
    The last reply repeats once the others are used up.
    """

    def __init__(self):
        self.tools = [Tool(name=name, inputSchema={"type": "object"}) for name in Config.HOME_ASSISTANT_MCP_TOOLS]
        self.tools.append(Tool(name="ha_restart", inputSchema={"type": "object"}))
        self.tool_results: Dict[str, str] = {}
        self.replies: List[Any] = ["Done."]
        self.tool_calls: List[tuple] = []
        self.llm_requests: List[dict] = []

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=self.tools)

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.tool_calls.append((name, arguments))
        return CallToolResult(content=[TextContent(type="text", text=self.tool_results.get(name, "ok"))])

    async def create(self, **request) -> ChatCompletion:
        self.llm_requests.append({**request, "messages": list(request["messages"])})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, str):
            message = {"role": "assistant", "content": reply}
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call-{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(reply)
            ]}
        return ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0, "model": request["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        })


@pytest.fixture
def fake_ha(monkeypatch) -> FakeHomeAssistant:
    ha = FakeHomeAssistant()

    @asynccontextmanager
    async def open_mcp_session():
        yield ha

    # Looked up through `ha.create` at call time so tests can swap it.
    async def create(**request):
        return await ha.create(**request)

    monkeypatch.setattr(home_assistant_agent, "open_mcp_session", open_mcp_session)
    monkeypatch.setattr(home_assistant_agent, "openrouter_client",
                        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    return ha


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
