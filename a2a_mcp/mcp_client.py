import os
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, ReadResourceResult


@asynccontextmanager
async def init_session(host, port, transport):
    """Connect to an MCP server via SSE or STDIO and yield a ClientSession."""
    if transport == 'sse':
        url = f'http://{host}:{port}/sse'
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream=read_stream, write_stream=write_stream) as session:
                await session.initialize()
                yield session
    elif transport == 'stdio':
        if not os.getenv('GOOGLE_API_KEY'):
            raise ValueError('GOOGLE_API_KEY is not set')
        params = StdioServerParameters(
            command='uv', args=['run', 'a2a-mcp'],
            env={'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')},
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream=read_stream, write_stream=write_stream) as session:
                await session.initialize()
                yield session
    else:
        raise ValueError(f"Unsupported transport: {transport}. Must be 'sse' or 'stdio'.")


async def find_agent(session: ClientSession, query: str) -> CallToolResult:
    """Call the 'find_agent' tool on the MCP server."""
    return await session.call_tool(name='find_agent', arguments={'query': query})


async def find_resource(session: ClientSession, resource: str) -> ReadResourceResult:
    """Read a resource from the MCP server."""
    return await session.read_resource(resource)
