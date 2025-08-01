"""
Generic MCP Tools Module

This module provides a generic interface for connecting to MCP servers
and creating tools that can be used with Google ADK agents.

The module supports:
- Generic MCP client for any MCP server
- Dynamic tool discovery and creation
- Multiple MCP server configurations
- Error handling and logging
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import httpx
# Import FunctionTool only when needed to avoid circular imports

# Configure logging
logger = logging.getLogger(__name__)

class MCPClient:
    """
    Generic MCP Client for connecting to MCP servers.

    This client can connect to any MCP server and dynamically create
    tools based on the server's available tools.
    """

    def __init__(self, host: str, port: int, service_path: str = "", timeout: int = 30):
        """
        Initialize MCP client.

        Args:
            host: MCP server host (e.g., "mcp-services" or "localhost")
            port: MCP server port (e.g., 3001)
            service_path: Service path for the MCP endpoint (e.g., "/fs_mcp")
            timeout: HTTP request timeout in seconds
        """
        self.host = host
        self.port = port
        self.service_path = service_path
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}{service_path}"
        self.tools_cache: Optional[List[Dict]] = None

        logger.info(f"MCPClient initialized for {self.base_url}")

    async def discover_tools(self) -> List[Dict]:
        """
        Discover available tools from the MCP server.

        Returns:
            List of tool definitions from the MCP server

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response format is invalid
        """
        if self.tools_cache is not None:
            return self.tools_cache

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "method": "tools/list",
                        "params": {}
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()

                if "tools" in result:
                    self.tools_cache = result["tools"]
                    logger.info(f"Discovered {len(self.tools_cache)} tools from {self.base_url}")
                    return self.tools_cache
                else:
                    raise ValueError(f"Invalid tools/list response format: {result}")

        except Exception as e:
            logger.error(f"Failed to discover tools from {self.base_url}: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response format is invalid
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()

                # Extract the actual result from the MCP response format
                if "content" in result and len(result["content"]) > 0:
                    content_text = result["content"][0]["text"]
                    try:
                        return json.loads(content_text)
                    except json.JSONDecodeError:
                        # If it's not JSON, return the text as-is
                        return {"content": content_text}
                else:
                    return {"success": False, "error": "Invalid MCP response format"}

        except Exception as e:
            error_msg = f"MCP tool call failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def _create_tool_function(self, tool_def: Dict) -> Callable:
        """
        Create an async function that wraps an MCP tool call.

        Args:
            tool_def: Tool definition from MCP server

        Returns:
            Async function that calls the MCP tool
        """
        tool_name = tool_def["name"]
        description = tool_def.get("description", "")

        async def tool_function(**kwargs) -> str:
            """Dynamic tool function that calls the MCP server."""
            result = await self.call_tool(tool_name, kwargs)

            # Convert result to string for ADK FunctionTool
            if isinstance(result, dict):
                if result.get("success"):
                    # Return success message or content
                    if "content" in result:
                        return str(result["content"])
                    elif "message" in result:
                        return str(result["message"])
                    else:
                        return json.dumps(result, indent=2)
                else:
                    # Return error message
                    return f"Error: {result.get('error', 'Unknown error')}"
            else:
                return str(result)

        # Set function name and docstring for better debugging
        tool_function.__name__ = f"mcp_{tool_name}"
        tool_function.__doc__ = f"MCP tool: {description}"

        return tool_function

    def create_function_tools(self):
        """
        Create Google ADK FunctionTool objects for all available MCP tools.

        Returns:
            List of FunctionTool objects that can be used with ADK agents

        Raises:
            Exception: If tool discovery fails
        """
        # Import FunctionTool here to avoid circular imports
        from google.adk.tools.function_tool import FunctionTool

        import asyncio

        # Run async discovery in sync context
        try:
            # Use existing event loop if available, otherwise create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't use run_until_complete
                    # This is a limitation - tools should be created in async context
                    logger.warning("Creating tools in sync context with running event loop - this may not work properly")
                    tools_list = []
                else:
                    tools_list = loop.run_until_complete(self.discover_tools())
            except RuntimeError:
                # No event loop running, create one
                tools_list = asyncio.run(self.discover_tools())
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            raise

        # Create FunctionTool objects for each tool
        function_tools = []
        for tool_def in tools_list:
            tool_function = self._create_tool_function(tool_def)
            function_tool = FunctionTool(func=tool_function)
            function_tools.append(function_tool)

        logger.info(f"Created {len(function_tools)} FunctionTool objects")
        return function_tools

# Convenience factory functions

def create_filesystem_tools(host: str = None, port: int = 3001):
    """
    Create filesystem tools that connect to the MCP filesystem server.

    Args:
        host: MCP server host. Defaults to "mcp-services" in development, "localhost" otherwise
        port: MCP server port. Defaults to 3001

    Returns:
        List of FunctionTool objects for filesystem operations
    """
    if host is None:
        # Use Docker service name when running in container, localhost for local development
        host = "mcp-services" if os.getenv("ENVIRONMENT") == "development" else "localhost"

    client = MCPClient(host=host, port=port, service_path="/fs_mcp")
    return client.create_function_tools()

def create_mcp_tools(host: str, port: int, service_path: str = ""):
    """
    Create tools that connect to any MCP server.

    Args:
        host: MCP server host
        port: MCP server port
        service_path: Service path for the MCP endpoint

    Returns:
        List of FunctionTool objects for the MCP server's tools
    """
    client = MCPClient(host=host, port=port, service_path=service_path)
    return client.create_function_tools()

# Configuration management

class MCPConfig:
    """
    MCP Configuration Management.

    Load MCP server configurations from environment variables or config files.
    """

    @staticmethod
    def get_server_config(server_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Dictionary with host, port, and service_path
        """
        # Check environment variables first
        host_env = os.getenv(f"MCP_{server_name.upper()}_HOST")
        port_env = os.getenv(f"MCP_{server_name.upper()}_PORT")
        path_env = os.getenv(f"MCP_{server_name.upper()}_PATH")

        if host_env:
            return {
                "host": host_env,
                "port": int(port_env) if port_env else 3001,
                "service_path": path_env or ""
            }

        # Default configurations
        default_configs = {
            "filesystem": {
                "host": "mcp-services" if os.getenv("ENVIRONMENT") == "development" else "localhost",
                "port": 3001,
                "service_path": "/fs_mcp"
            }
        }

        return default_configs.get(server_name, {
            "host": "localhost",
            "port": 3001,
            "service_path": ""
        })

    @staticmethod
    def create_tools_from_config(server_name: str):
        """
        Create tools based on server configuration.

        Args:
            server_name: Name of the MCP server

        Returns:
            List of FunctionTool objects
        """
        config = MCPConfig.get_server_config(server_name)
        client = MCPClient(
            host=config["host"],
            port=config["port"],
            service_path=config["service_path"]
        )
        return client.create_function_tools()

# Utility functions

async def discover_available_servers(host: str = "localhost", port: int = 3001) -> Dict[str, Dict]:
    """
    Discover available MCP servers by querying the main MCP services endpoint.

    Args:
        host: Host to query for server discovery
        port: Port to query for server discovery

    Returns:
        Dictionary of available servers and their configurations
    """
    # This would require a discovery endpoint on the MCP services container
    # For now, return basic filesystem server info
    return {
        "filesystem": {
            "host": host,
            "port": port,
            "service_path": "/fs_mcp",
            "description": "Filesystem operations server"
        }
    }

# Example usage functions (for documentation)

def example_usage():
    """
    Example usage of the MCP tools module.

    Examples:
        # Simple filesystem tools
        from mcp.mcp_tools import create_filesystem_tools
        tools = create_filesystem_tools()

        # Custom MCP server
        from mcp.mcp_tools import create_mcp_tools
        tools = create_mcp_tools(host="mcp-services", port=3002, service_path="/db_mcp")

        # Configuration-based tools
        from mcp.mcp_tools import MCPConfig
        tools = MCPConfig.create_tools_from_config("filesystem")
    """
    pass
