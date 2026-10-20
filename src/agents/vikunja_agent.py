from typing import Literal, List, Optional
import asyncio
import time

from pydantic import Field
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator, BaseDynamicContextProvider
import httpx

from src.agents.context_providers import CurrentDateProvider
from src.config import Config
from src.logger import logger
from src.services.http_clients import vikunja_client
from src.services.llm_client import llm_client

########################
# INPUT/OUTPUT SCHEMAS #
########################
class VikunjaInputSchema(BaseIOSchema):
    """Input schema for Vikunja Agent."""
    user_query: str = Field(..., description="The user's task-related query to be processed.")


class VikunjaOutputSchema(BaseIOSchema):
    """Output schema for Vikunja Agent."""
    action: Literal["create_task", "get_tasks"] = Field(
        ..., description="Action type to perform on Vikunja: create a task or retrieve tasks."
    )
    project_id: Optional[int] = Field(
        None, description="ID of the project where the task belongs. Required for create_task."
    )
    title: Optional[str] = Field(None, description="Title of the task (for create_task).")
    description: Optional[str] = Field(None, description="Description of the task (for create_task).")
    due_date: Optional[str] = Field(None, description="Due date of the task in ISO 8601 format (for create_task).")


#####################
# CONTEXT PROVIDERS #
#####################
class AvailableProjectsProvider(BaseDynamicContextProvider):
    """
    Lists Vikunja projects in the prompt.
    get_info() is synchronous, so the list is fetched beforehand with refresh() and kept for a TTL.
    """

    def __init__(self, title):
        super().__init__(title)
        self.projects: Optional[List[dict]] = None
        self.fetched_at: float = 0.0
        self.lock = asyncio.Lock()

    async def refresh(self) -> None:
        """
        Fetch all projects from Vikunja if the cached list is missing or older than the TTL.
        """
        async with self.lock:
            if self.projects is not None and time.monotonic() - self.fetched_at < Config.VIKUNJA_PROJECTS_TTL_SECONDS:
                return

            try:
                response = await vikunja_client.get("projects")
                response.raise_for_status()
            except httpx.HTTPError as e:
                # Keep serving the previous list (if any) rather than failing the request.
                logger.error(f"Failed to fetch Vikunja projects: {e}")
                return

            self.projects = response.json()
            self.fetched_at = time.monotonic()

    def get_info(self) -> str:
        if not self.projects:
            return "No projects available."
        return "\n".join([f"{p['title']} (id: {p['id']})" for p in self.projects])


######################
# VIKUNJA AGENT CONFIG #
######################
available_projects_provider = AvailableProjectsProvider("Available Projects")

# Shared, stateless pieces. A fresh agent (with empty history) is built per call so that
# concurrent requests never share or grow a conversation history.
system_prompt_generator = SystemPromptGenerator(
    background=[
        "You are a helpful assistant for handling tasks in Vikunja projects.",
        "Available actions:",
        "- create_task: use when the query is about creating a new task in a project.",
        "- get_tasks: use when the query is about retrieving tasks (titles, descriptions, due dates, project names).",
        "You must map queries to the appropriate action and extract relevant fields.",
        "Available projects (IDs and names) will be provided in context.",
    ],
    output_instructions=[
        "Decide the action: 'create_task' or 'get_tasks'.",
        "If creating a task, extract project_id, title, description, and due_date (ISO 8601 or default 0001-01-01T00:00:00Z).",
        "If retrieving tasks, only specify action=get_tasks.",
        "Format output according to schema."
    ],
    context_providers={
        "current_date": CurrentDateProvider("Current Date"),
        "available_projects": available_projects_provider,
    },
)


def build_agent() -> AtomicAgent[VikunjaInputSchema, VikunjaOutputSchema]:
    return AtomicAgent[VikunjaInputSchema, VikunjaOutputSchema](
        config=AgentConfig(
            client=llm_client,
            model=Config.VIKUNJA_AGENT_MODEL,
            system_prompt_generator=system_prompt_generator,
        )
    )


#########################
# VIKUNJA AGENT FUNCTION #
#########################

async def create_task(project_id: int, title: str, description: str, due_date: str) -> Optional[str]:
    """
    Create a task in a Vikunja project.

    Args:
        project_id (int): ID of the project where the task should be created.
        title (str): Title of the task.
        description (str): Task description.
        due_date (str): Due date in ISO 8601 format, e.g. '2025-08-18T12:00:00Z'.
    """
    payload = {
        "title": title,
        "description": description,
        "due_date": due_date
    }

    try:
        response = await vikunja_client.put(f"projects/{project_id}/tasks", json=payload)
    except httpx.HTTPError as e:
        logger.error(f"Failed to create task: {e}")
        return None

    if response.status_code in (200, 201):
        result = response.json()
        logger.info(f"Task created successfully: {result}")
        return f"Task created successfully: {result}"

    logger.error(f"Failed to create task: {response.status_code} - {response.text}")
    return None


async def get_pending_tasks() -> Optional[str]:
    """
    Retrieve all pending tasks across Vikunja projects.

    :return: The pending tasks formatted as text, or None on failure.
    """
    try:
        response = await vikunja_client.get("tasks", params={"filter": "done=false"})
    except httpx.HTTPError as e:
        logger.error(f"Failed to retrieve tasks: {e}")
        return None

    if response.status_code == 200:
        formatted_tasks = [f"{task['title']} - {task['description']}" for task in response.json()]
        return "Pending Tasks: \n\n" + "\n".join(formatted_tasks)

    logger.error(f"Failed to retrieve tasks: {response.status_code} - {response.text}")
    return None


async def run_vikunja_agent(user_input: str) -> Optional[str]:
    """
    Process a user query with the Vikunja Agent.
    """
    logger.info(f"Processing user input: {user_input}")

    await available_projects_provider.refresh()

    # Run agent
    result: VikunjaOutputSchema = await build_agent().run_async(VikunjaInputSchema(user_query=user_input))
    action = result.action
    if action == "create_task":
        logger.info(f"Action: create_task - {result.model_dump_json()}")

        response = await create_task(
            project_id=result.project_id,
            title=result.title,
            description=result.description,
            due_date=result.due_date or "0001-01-01T00:00:00Z"
        )
        if response:
            return response
        logger.error("Failed to create task.")

    elif action == "get_tasks":
        logger.info("Action: get_tasks")

        tasks = await get_pending_tasks()
        if tasks:
            logger.debug(f"Retrieved tasks: {tasks}")
            return tasks
        logger.error("Failed to retrieve tasks.")

    return result.model_dump_json()


#####################
# DEMO EXECUTION #
#####################
if __name__ == "__main__":
    async def main():
        # Example 1: Create a task
        print(await run_vikunja_agent("Create a task 'from telegram' in the Personal project"))

        # Example 2: Get tasks
        print(await run_vikunja_agent("Show me all my pending tasks"))

    asyncio.run(main())
