import json
from collections.abc import AsyncIterable
from enum import Enum
from uuid import uuid4

import httpx
import networkx as nx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from google.protobuf import json_format

import mcp_client
from common import MCP_HOST, MCP_PORT


class Status(Enum):
    READY = 'READY'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    PAUSED = 'PAUSED'
    INITIALIZED = 'INITIALIZED'


class WorkflowNode:
    """A single node in the workflow graph that finds and invokes an agent."""

    def __init__(self, task: str, node_key: str | None = None):
        self.id = uuid4().hex
        self.node_key = node_key
        self.task = task
        self.results = None
        self.state = Status.READY
        self.remote_task_id = None
        self.remote_context_id = None

    async def get_planner_resource(self) -> AgentCard | None:
        print(f'[MCP] Looking up planner resource at mcp://{MCP_HOST}:{MCP_PORT} ...')
        async with mcp_client.init_session(MCP_HOST, MCP_PORT) as session:
            response = await mcp_client.find_resource(session, 'resource://agent_cards/planner_agent')
            data = json.loads(response.contents[0].text)
            card = json_format.ParseDict(data['agent_card'][0], AgentCard())
            endpoint = card.supported_interfaces[0].url if card.supported_interfaces else 'unknown'
            print(f'[MCP] Found planner agent: {card.name} @ {endpoint}')
            return card

    async def find_agent_for_task(self) -> AgentCard | None:
        print(f'[MCP] Searching for agent to handle: "{self.task[:80]}..."')
        async with mcp_client.init_session(MCP_HOST, MCP_PORT) as session:
            result = await mcp_client.find_agent(session, self.task)
            agent_card_json = json.loads(result.content[0].text)
            card = json_format.ParseDict(agent_card_json, AgentCard())
            endpoint = card.supported_interfaces[0].url if card.supported_interfaces else 'unknown'
            print(f'[MCP] Matched agent: {card.name} @ {endpoint}')
            return card

    async def run_node(self, query: str, task_id: str, context_id: str) -> AsyncIterable[dict[str, any]]:
        agent_card = await self.get_planner_resource() if self.node_key == 'planner' else await self.find_agent_for_task()
        print(f'[A2A] Sending to {agent_card.name}: "{query[:80]}..."')
        httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        client = await ClientFactory.connect(
            agent_card,
            client_config=ClientConfig(httpx_client=httpx_client, streaming=True),
        )
        message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text=query)],
            message_id=uuid4().hex,
        )
        if self.remote_task_id:
            message.task_id = self.remote_task_id
        if self.remote_context_id:
            message.context_id = self.remote_context_id
        request = SendMessageRequest(message=message)
        async for stream_resp, task in client.send_message(request):
            payload_type = stream_resp.WhichOneof('payload')
            if payload_type == 'status_update':
                evt = stream_resp.status_update
                if evt.task_id:
                    self.remote_task_id = evt.task_id
                if evt.context_id:
                    self.remote_context_id = evt.context_id
                state_name = TaskState.Name(evt.status.state)
                print(f'[A2A] {agent_card.name} status: {state_name}')
            elif payload_type == 'artifact_update':
                self.results = stream_resp.artifact_update.artifact
                if stream_resp.artifact_update.task_id:
                    self.remote_task_id = stream_resp.artifact_update.task_id
                if stream_resp.artifact_update.context_id:
                    self.remote_context_id = stream_resp.artifact_update.context_id
                print(f'[A2A] Received artifact from {agent_card.name}: {self.results.name}')
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

    def __repr__(self) -> str:
        if not self.nodes:
            return '[Workflow] Graph state: (empty)'
        order = list(nx.topological_sort(self.graph))
        lines = [f'[Workflow] Graph state: ({self.state.value})']
        for i, node_id in enumerate(order):
            node = self.nodes[node_id]
            label = node.node_key or node.id[:8]
            task = node.task[:40] + '...' if len(node.task) > 40 else node.task
            lines.append(f'  [{node.state.value:<9}] {label}: "{task}"')
            if i < len(order) - 1:
                lines.append('       ↓')
        return '\n'.join(lines)

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
        print(f'[Workflow] Running {len(sub_graph)} node(s)')
        print(self)

        for node_id in sub_graph:
            node = self.nodes[node_id]
            node.state = Status.RUNNING
            print(f'[Workflow] Executing node {node.node_key or node.id[:8]}: "{node.task[:60]}..."')
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
                print(self)
        if self.state == Status.RUNNING:
            self.state = Status.COMPLETED
            print('[Workflow] All nodes completed')
