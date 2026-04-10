import json
import logging
import uuid
from collections.abc import AsyncIterable
from enum import Enum
from uuid import uuid4

import networkx as nx
from a2a.client import ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from google.protobuf import json_format

import mcp_client
from common import get_mcp_server_config

logger = logging.getLogger(__name__)


class Status(Enum):
    READY = 'READY'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    PAUSED = 'PAUSED'
    INITIALIZED = 'INITIALIZED'


class WorkflowNode:
    """A single node in the workflow graph that finds and invokes an agent."""

    def __init__(self, task: str, node_key: str | None = None, node_label: str | None = None):
        self.id = str(uuid.uuid4())
        self.node_key = node_key
        self.node_label = node_label
        self.task = task
        self.results = None
        self.state = Status.READY

    async def get_planner_resource(self) -> AgentCard | None:
        config = get_mcp_server_config()
        async with mcp_client.init_session(config.host, config.port, config.transport) as session:
            response = await mcp_client.find_resource(session, 'resource://agent_cards/planner_agent')
            data = json.loads(response.contents[0].text)
            return json_format.ParseDict(data['agent_card'][0], AgentCard())

    async def find_agent_for_task(self) -> AgentCard | None:
        config = get_mcp_server_config()
        async with mcp_client.init_session(config.host, config.port, config.transport) as session:
            result = await mcp_client.find_agent(session, self.task)
            agent_card_json = json.loads(result.content[0].text)
            return json_format.ParseDict(agent_card_json, AgentCard())

    async def run_node(self, query: str, task_id: str, context_id: str) -> AsyncIterable[dict[str, any]]:
        agent_card = await self.get_planner_resource() if self.node_key == 'planner' else await self.find_agent_for_task()
        client = await ClientFactory.connect(agent_card)
        message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text=query)],
            message_id=uuid4().hex,
            task_id=task_id,
            context_id=context_id,
        )
        request = SendMessageRequest(message=message)
        async for stream_resp, task in client.send_message(request):
            if stream_resp.WhichOneof('payload') == 'artifact_update':
                self.results = stream_resp.artifact_update.artifact
            yield (stream_resp, task)


class WorkflowGraph:
    """Directed graph of workflow nodes executed in topological order."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.latest_node = None
        self.state = Status.INITIALIZED
        self.paused_node_id = None

    def add_node(self, node) -> None:
        self.graph.add_node(node.id, query=node.task)
        self.nodes[node.id] = node
        self.latest_node = node.id

    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError('Invalid node IDs')
        self.graph.add_edge(from_node_id, to_node_id)

    def set_node_attributes(self, node_id, attr_val) -> None:
        nx.set_node_attributes(self.graph, {node_id: attr_val})

    async def run_workflow(self, start_node_id: str | None = None) -> AsyncIterable[dict[str, any]]:
        if not start_node_id or start_node_id not in self.nodes:
            start_nodes = [n for n, d in self.graph.in_degree() if d == 0]
        else:
            start_nodes = [start_node_id]

        applicable_graph = set()
        for node_id in start_nodes:
            applicable_graph.add(node_id)
            applicable_graph.update(nx.descendants(self.graph, node_id))

        complete_graph = list(nx.topological_sort(self.graph))
        sub_graph = [n for n in complete_graph if n in applicable_graph]
        self.state = Status.RUNNING

        for node_id in sub_graph:
            node = self.nodes[node_id]
            node.state = Status.RUNNING
            query = self.graph.nodes[node_id].get('query')
            task_id = self.graph.nodes[node_id].get('task_id')
            context_id = self.graph.nodes[node_id].get('context_id')
            async for stream_resp, task in node.run_node(query, task_id, context_id):
                if node.state != Status.PAUSED:
                    if stream_resp.WhichOneof('payload') == 'status_update':
                        task_status_event = stream_resp.status_update
                        context_id = task_status_event.context_id
                        if task_status_event.status.state == TaskState.TASK_STATE_INPUT_REQUIRED and context_id:
                            node.state = Status.PAUSED
                            self.state = Status.PAUSED
                            self.paused_node_id = node.id
                    yield (stream_resp, task)
            if self.state == Status.PAUSED:
                break
            if node.state == Status.RUNNING:
                node.state = Status.COMPLETED
        if self.state == Status.RUNNING:
            self.state = Status.COMPLETED
