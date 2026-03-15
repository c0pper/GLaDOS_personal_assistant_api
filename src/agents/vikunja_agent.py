from typing import Literal, List, Optional
from datetime import datetime

from openai import OpenAI as OpenRouterClient
from pydantic import Field
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator, BaseDynamicContextProvider
import instructor
import requests

from src.config import Config
from src.logger import logger

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
class CurrentDateProvider(BaseDynamicContextProvider):
    def __init__(self, title):
        super().__init__(title)
        self.date = datetime.now().strftime("%Y-%m-%d")

    def get_info(self) -> str:
        return f"Current date in format YYYY-MM-DD: {self.date}"
    
class AvailableProjectsProvider(BaseDynamicContextProvider):
    def __init__(self, title):
        super().__init__(title)
        self.projects = None


    def get_projects(self):
        """
        Fetch all projects from Vikunja and update self.projects.
        """
        url = f"{Config.VIKUNJA_BASE_URL}/projects"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {Config.VIKUNJA_TOKEN}"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        self.projects = response.json()
        return self.projects

    def get_info(self) -> str:
        if not self.projects:
            self.get_projects()
        if not self.projects:
            return "No projects available."
        return "\n".join([f"{p['title']} (id: {p['id']})" for p in self.projects])


######################
# VIKUNJA AGENT CONFIG #
######################
vikunja_agent_config = AgentConfig(
    client=instructor.from_openai(
        OpenRouterClient(base_url="https://openrouter.ai/api/v1", api_key=Config.OPENROUTER_API_KEY),
    ),
    model=Config.VIKUNJA_AGENT_MODEL,
    system_prompt_generator=SystemPromptGenerator(
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
    )
)

# Instantiate agent
vikunja_agent = AtomicAgent[VikunjaInputSchema, VikunjaOutputSchema](config=vikunja_agent_config)

# Register context providers
vikunja_agent.register_context_provider("current_date", CurrentDateProvider("Current Date"))
vikunja_agent.register_context_provider("available_projects", AvailableProjectsProvider("Available Projects"))


#########################
# VIKUNJA AGENT FUNCTION #
#########################

def create_task(project_id: int, title: str, description: str, due_date: str) -> Optional[str]:
    """
    Create a task in a Vikunja project.

    Args:
        project_id (int): ID of the project where the task should be created.
        title (str): Title of the task.
        description (str): Task description.
        due_date (str): Due date in ISO 8601 format, e.g. '2025-08-18T12:00:00Z'.
    """
    url = f"{Config.VIKUNJA_BASE_URL}/projects/{project_id}/tasks"

    payload = {
        "title": title,
        "description": description,
        "due_date": due_date
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.VIKUNJA_TOKEN}"
    }

    response = requests.put(url, json=payload, headers=headers)

    if response.status_code == 200 or response.status_code == 201:
        logger.info(f"Task created successfully: {response.json()}")
        return f"Task created successfully: {response.json()}"
    else:
        logger.error(f"Failed to create task: {response.status_code} - {response.text}")


def get_pending_tasks() -> Optional[str]:
    """
    Retrieve all tasks from a Vikunja project.
    
    :param project_id: The ID of the project.
    :param token: The authentication token (Bearer).
    :return: JSON response with the tasks.
    """
    url = f"{Config.VIKUNJA_BASE_URL}/tasks?filter=done=false"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {Config.VIKUNJA_TOKEN}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        formatted_tasks = [f"{task['title']} - {task['description']}" for task in response_json]
        return "Pending Tasks: \n\n" + "\n".join(formatted_tasks)
    else:
        logger.error(f"Failed to retrieve tasks: {response.status_code} - {response.text}")
        return None


def run_vikunja_agent(user_input: str) -> Optional[str]:
    """
    Process a user query with the Vikunja Agent.
    available_projects is a list of dicts like [{"id": 1, "title": "General"}, ...]
    """

    logger.info(f"Processing user input: {user_input}")

    # Run agent
    result: VikunjaOutputSchema = vikunja_agent.run(VikunjaInputSchema(user_query=user_input))
    action = result.action
    if action == "create_task":
        logger.info(f"Action: create_task")
        logger.info(f"Project ID: {result.project_id}")
        logger.info(f"Title: {result.title}")
        logger.info(f"Description: {result.description}")
        logger.info(f"Due date: {result.due_date}")

        response = create_task(
            project_id=result.project_id,
            title=result.title,
            description=result.description,
            due_date=result.due_date or "0001-01-01T00:00:00Z"
        )
        if response:
            return response
        else:
            logger.error("Failed to create task.")

    elif action == "get_tasks":
        logger.info(f"Action: get_tasks")

        tasks = get_pending_tasks()
        if tasks:
            logger.info(f"Retrieved tasks: {tasks}")
            return tasks
        else:
            logger.error("Failed to retrieve tasks.")
    return result.model_dump_json()


#####################
# DEMO EXECUTION #
#####################
if __name__ == "__main__":
    # Example 1: Create a task
    output1 = run_vikunja_agent("Create a task 'from telegram' in the Personal project")
    print(output1.model_dump())

    # Example 2: Get tasks
    output2 = run_vikunja_agent("Show me all my pending tasks")
    print(output2.model_dump())