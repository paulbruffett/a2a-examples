import json
import logging
import os
import uuid
from abc import ABC
from collections.abc import AsyncGenerator
from typing import Any, Literal

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCard,
    Part,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import InternalError
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# --- Utils ---

class ServerConfig(BaseModel):
    host: str
    port: int
    transport: str
    url: str


def init_api_key():
    """No-op kept for backward compatibility. API keys are read from env vars by the model backends."""
    pass


def get_mcp_server_config() -> ServerConfig:
    return ServerConfig(
        host='localhost', port=10100, transport='sse',
        url='http://localhost:10100/sse',
    )


# --- Types ---

class PlannerTask(BaseModel):
    id: int
    description: str
    status: (
        Any
        | Literal['input_required', 'completed', 'error', 'pending', 'incomplete', 'todo', 'not_started']
        | None
    ) = 'input_required'


class TripInfo(BaseModel):
    total_budget: str | None = None
    origin: str | None = None
    destination: str | None = None
    type: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    travel_class: str | None = None
    accommodation_type: str | None = None
    room_type: str | None = None
    is_car_rental_required: str | None = None
    type_of_car: str | None = None
    no_of_travellers: str | None = None
    checkin_date: str | None = None
    checkout_date: str | None = None
    car_rental_start_date: str | None = None
    car_rental_end_date: str | None = None

    @model_validator(mode='before')
    @classmethod
    def set_dependent_var(cls, values):
        if isinstance(values, dict):
            if 'start_date' in values:
                values.setdefault('checkin_date', values['start_date'])
                values.setdefault('car_rental_start_date', values['start_date'])
            if 'end_date' in values:
                values.setdefault('checkout_date', values['end_date'])
                values.setdefault('car_rental_end_date', values['end_date'])
        return values


class TaskList(BaseModel):
    original_query: str | None = None
    trip_info: TripInfo | None = None
    tasks: list[PlannerTask] = []


# --- Base Agent ---

class BaseAgent(BaseModel, ABC):
    model_config = {'arbitrary_types_allowed': True, 'extra': 'allow'}
    agent_name: str
    description: str
    content_types: list[str]


# --- Agent Executor ---

class GenericAgentExecutor(AgentExecutor):
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        async for item in self.agent.stream(query, task.context_id, task.id):
            if isinstance(item, tuple):
                stream_resp, _ = item
                payload_type = stream_resp.WhichOneof('payload')
                if payload_type == 'status_update':
                    await event_queue.enqueue_event(stream_resp.status_update)
                elif payload_type == 'artifact_update':
                    await event_queue.enqueue_event(stream_resp.artifact_update)
                continue

            is_task_complete = item['is_task_complete']
            require_user_input = item['require_user_input']

            if is_task_complete:
                if item.get('response_type') == 'data':
                    from google.protobuf import json_format, struct_pb2
                    value = json_format.ParseDict(item['content'], struct_pb2.Value())
                    part = Part(data=value)
                else:
                    part = Part(text=item['content'])
                await updater.add_artifact([part], name=f'{self.agent.agent_name}-result')
                await updater.complete()
                break
            if require_user_input:
                await updater.update_status(
                    TaskState.TASK_STATE_INPUT_REQUIRED,
                    new_agent_text_message(item['content'], task.context_id, task.id),
                    final=True,
                )
                break
            await updater.update_status(
                TaskState.TASK_STATE_WORKING,
                new_agent_text_message(item['content'], task.context_id, task.id),
            )

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        raise InternalError(error=UnsupportedOperationError())


# --- ADK Agent Runner ---

class AgentRunner:
    def __init__(self, app_name: str = 'A2A-MCP', user_id: str = 'user_1'):
        self.session_service = InMemorySessionService()
        self.session = None
        self.app_name = app_name
        self.user_id = user_id

    async def run_stream(self, agent: Agent, query: str, session_id: str) -> AsyncGenerator[Event, None]:
        runner = Runner(agent=agent, app_name=self.app_name, session_service=self.session_service)
        if not session_id:
            session_id = uuid.uuid4().hex
        else:
            self.session = await self.session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=session_id,
            )
        if not self.session:
            self.session = await self.session_service.create_session(
                app_name=self.app_name, user_id=self.user_id, session_id=session_id,
            )
        content = types.Content(role='user', parts=[types.Part(text=query)])
        async for event in runner.run_async(
            user_id=self.user_id, session_id=self.session.id, new_message=content,
        ):
            if event.is_final_response():
                response = ''
                if event.content and event.content.parts and event.content.parts[0].text:
                    response = '\n'.join(p.text for p in event.content.parts if p.text)
                elif event.content and event.content.parts and any(p.function_response for p in event.content.parts):
                    response = next(p.function_response.model_dump() for p in event.content.parts)
                else:
                    response = f'Error in running agent: {agent.name}'
                yield {'type': 'final_result', 'response': response}
            else:
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f'{agent.name}: Processing request...',
                }


# --- App Builder ---

def build_a2a_app(agent: BaseAgent, agent_card_path: str):
    """Build an A2A Starlette app from an agent and its card JSON file."""
    from google.protobuf import json_format
    with open(agent_card_path) as f:
        data = json.load(f)
    card = json_format.ParseDict(data, AgentCard())
    request_handler = DefaultRequestHandler(
        agent_executor=GenericAgentExecutor(agent=agent),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=card, http_handler=request_handler).build()
