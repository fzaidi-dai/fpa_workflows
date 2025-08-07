"""
ADK Integration Tools for MCP Servers

This module provides integration functions that create Google ADK FunctionTool objects
from custom MCP clients. These functions can be imported by any agent that wants to
use our MCP servers.

The module supports:
- Filesystem MCP server integration with ADK
- Easy-to-use factory functions for creating ADK tools
- Consistent interface for multiple MCP servers
"""

import asyncio
from typing import List

from google.adk.tools.function_tool import FunctionTool
from .fpa_mcp_tools import create_filesystem_client


def create_filesystem_tools() -> List[FunctionTool]:
    """
    Create filesystem tools that connect to our MCP server.

    Returns:
        List of FunctionTool objects for filesystem operations
    """
    # Create MCP client
    client = create_filesystem_client()

    async def write_file_tool(path: str, content: str) -> str:
        """Write content to a file using MCP server."""
        result = await client.call_tool("write_file", {"path": f"/mcp-data/{path}", "content": content})
        if result.get("success"):
            return f"Successfully wrote {result.get('bytes_written', 0)} bytes to {path}"
        else:
            return f"Failed to write file: {result.get('error', 'Unknown error')}"

    async def read_file_tool(path: str) -> str:
        """Read content from a file using MCP server."""
        result = await client.call_tool("read_file", {"path": f"/mcp-data/{path}"})
        if result.get("success"):
            return f"File content:\n{result.get('content', '')}"
        else:
            return f"Failed to read file: {result.get('error', 'Unknown error')}"

    # Create FunctionTool objects
    tools = [
        FunctionTool(func=write_file_tool),
        FunctionTool(func=read_file_tool)
    ]

    return tools


# Future integrations for other MCP servers can be added here:
# def create_database_tools() -> List[FunctionTool]:
#     """Create database tools that connect to our database MCP server."""
#     pass
#
# def create_api_tools() -> List[FunctionTool]:
#     """Create API tools that connect to our API MCP server."""
#     pass
