from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, ReadResourceResult


@asynccontextmanager
async def init_session(host, port):
    """Connect to an MCP server via SSE and yield a ClientSession."""
    url = f'http://{host}:{port}/sse'
    async with sse_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream=read_stream, write_stream=write_stream) as session:
            await session.initialize()
            yield session


async def find_agent(session: ClientSession, query: str) -> CallToolResult:
    """Call the 'find_agent' tool on the MCP server."""
    return await session.call_tool(name='find_agent', arguments={'query': query})


async def find_resource(session: ClientSession, resource: str) -> ReadResourceResult:
    """Read a resource from the MCP server."""
    return await session.read_resource(resource)
