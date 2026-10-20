import json

import httpx

from src.agents import vikunja_agent
from src.agents.vikunja_agent import (
    VikunjaOutputSchema,
    available_projects_provider,
    create_task,
    get_pending_tasks,
    run_vikunja_agent,
)

PROJECTS_PATH = "/api/v1/projects"
TASKS_PATH = "/api/v1/tasks"
PROJECTS = [{"id": 1, "title": "Personal"}, {"id": 2, "title": "Work"}]


######################
# PROJECTS PROVIDER #
######################
async def test_projects_are_fetched_once_within_ttl(fake_http):
    fake_http.on("GET", PROJECTS_PATH, httpx.Response(200, json=PROJECTS))

    await available_projects_provider.refresh()
    await available_projects_provider.refresh()

    assert len(fake_http.requests_to(PROJECTS_PATH)) == 1
    assert available_projects_provider.get_info() == "Personal (id: 1)\nWork (id: 2)"
    assert fake_http.requests_to(PROJECTS_PATH)[0].headers["Authorization"] == "Bearer vikunja-token"


async def test_projects_are_refetched_after_ttl(fake_http, monkeypatch):
    fake_http.on("GET", PROJECTS_PATH, httpx.Response(200, json=PROJECTS))
    now = [1000.0]
    monkeypatch.setattr(vikunja_agent.time, "monotonic", lambda: now[0])

    await available_projects_provider.refresh()
    now[0] += vikunja_agent.Config.VIKUNJA_PROJECTS_TTL_SECONDS + 1
    await available_projects_provider.refresh()

    assert len(fake_http.requests_to(PROJECTS_PATH)) == 2


async def test_empty_project_list_is_cached_too(fake_http):
    fake_http.on("GET", PROJECTS_PATH, httpx.Response(200, json=[]))

    await available_projects_provider.refresh()
    await available_projects_provider.refresh()

    assert len(fake_http.requests_to(PROJECTS_PATH)) == 1
    assert available_projects_provider.get_info() == "No projects available."


async def test_failed_refresh_keeps_previous_projects(fake_http, monkeypatch):
    fake_http.on("GET", PROJECTS_PATH, httpx.Response(200, json=PROJECTS))
    now = [1000.0]
    monkeypatch.setattr(vikunja_agent.time, "monotonic", lambda: now[0])
    await available_projects_provider.refresh()

    fake_http.on("GET", PROJECTS_PATH, httpx.Response(503))
    now[0] += vikunja_agent.Config.VIKUNJA_PROJECTS_TTL_SECONDS + 1
    await available_projects_provider.refresh()

    assert available_projects_provider.get_info() == "Personal (id: 1)\nWork (id: 2)"


#################
# VIKUNJA CALLS #
#################
async def test_get_pending_tasks_formats_tasks(fake_http):
    fake_http.on("GET", TASKS_PATH, httpx.Response(200, json=[
        {"title": "Buy milk", "description": "2 liters"},
        {"title": "Call mom", "description": ""},
    ]))

    result = await get_pending_tasks()

    assert result == "Pending Tasks: \n\nBuy milk - 2 liters\nCall mom - "
    assert fake_http.requests_to(TASKS_PATH)[0].url.params["filter"] == "done=false"


async def test_get_pending_tasks_failure_returns_none(fake_http):
    fake_http.on("GET", TASKS_PATH, httpx.Response(500))

    assert await get_pending_tasks() is None


async def test_create_task_sends_payload(fake_http):
    fake_http.on("PUT", "/api/v1/projects/1/tasks", httpx.Response(201, json={"id": 7, "title": "Test"}))

    result = await create_task(1, "Test", "desc", "2026-10-21T00:00:00Z")

    assert result == "Task created successfully: {'id': 7, 'title': 'Test'}"
    request = fake_http.requests_to("/api/v1/projects/1/tasks")[0]
    assert json.loads(request.content) == {"title": "Test", "description": "desc", "due_date": "2026-10-21T00:00:00Z"}


async def test_create_task_failure_returns_none(fake_http):
    fake_http.on("PUT", "/api/v1/projects/1/tasks", httpx.Response(400, text="bad"))

    assert await create_task(1, "Test", "desc", "2026-10-21T00:00:00Z") is None


async def test_create_task_transport_error_returns_none(fake_http):
    def raise_error(request):
        raise httpx.ConnectError("refused", request=request)

    fake_http.on("PUT", "/api/v1/projects/1/tasks", raise_error)

    assert await create_task(1, "Test", "desc", "2026-10-21T00:00:00Z") is None


################
# AGENT FLOW #
################
async def test_run_agent_get_tasks(fake_llm, fake_http):
    fake_http.on("GET", PROJECTS_PATH, httpx.Response(200, json=PROJECTS))
    fake_http.on("GET", TASKS_PATH, httpx.Response(200, json=[{"title": "Buy milk", "description": "now"}]))
    fake_llm.on(VikunjaOutputSchema, lambda _: VikunjaOutputSchema(action="get_tasks"))

    result = await run_vikunja_agent("Show me my pending tasks")

    assert result == "Pending Tasks: \n\nBuy milk - now"
    call = fake_llm.calls[0]
    assert "Personal (id: 1)" in call.system_prompt
    assert call.history_length == 0


async def test_run_agent_create_task_defaults_due_date(fake_llm, fake_http):
    fake_http.on("GET", PROJECTS_PATH, httpx.Response(200, json=PROJECTS))
    fake_http.on("PUT", "/api/v1/projects/2/tasks", httpx.Response(200, json={"id": 1}))
    fake_llm.on(VikunjaOutputSchema, lambda _: VikunjaOutputSchema(
        action="create_task", project_id=2, title="Report", description="Q3"
    ))

    result = await run_vikunja_agent("Add 'Report' to Work")

    assert result.startswith("Task created successfully")
    payload = json.loads(fake_http.requests_to("/api/v1/projects/2/tasks")[0].content)
    assert payload["due_date"] == "0001-01-01T00:00:00Z"


async def test_run_agent_failure_returns_agent_output(fake_llm, fake_http):
    fake_http.on("GET", PROJECTS_PATH, httpx.Response(200, json=PROJECTS))
    fake_http.on("GET", TASKS_PATH, httpx.Response(500))
    fake_llm.on(VikunjaOutputSchema, lambda _: VikunjaOutputSchema(action="get_tasks"))

    result = await run_vikunja_agent("Show me my tasks")

    assert json.loads(result)["action"] == "get_tasks"
