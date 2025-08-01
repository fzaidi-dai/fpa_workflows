"""
Generic MCP Tools Module

This module provides a generic interface for connecting to MCP servers
and creating tools that can be used with any agent framework.

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
from typing import Any, Dict, List, Optional
import httpx

# Configure logging
logger = logging.getLogger(__name__)

class MCPClient:
    """
    Generic MCP Client for connecting to MCP servers.

    This client can connect to any MCP server and dynamically discover
    and call tools based on the server's available tools.
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

# Convenience factory functions

def create_filesystem_client(host: str = None, port: int = 3001, service_path: str = "/fs_mcp"):
    """
    Create filesystem MCP client that connects to the MCP filesystem server.

    Args:
        host: MCP server host. Defaults to "mcp-services" in development, "localhost" otherwise
        port: MCP server port. Defaults to 3001
        service_path: Service path for the MCP endpoint. Defaults to "/fs_mcp"

    Returns:
        MCPClient instance for filesystem operations
    """
    if host is None:
        # Use Docker service name when running in container, localhost for local development
        host = "mcp-services" if os.getenv("ENVIRONMENT") == "development" else "localhost"

    return MCPClient(host=host, port=port, service_path=service_path)

def create_mcp_client(host: str, port: int, service_path: str = ""):
    """
    Create MCP client that connects to any MCP server.

    Args:
        host: MCP server host
        port: MCP server port
        service_path: Service path for the MCP endpoint

    Returns:
        MCPClient instance for the MCP server
    """
    return MCPClient(host=host, port=port, service_path=service_path)

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

# Example usage functions (for documentation)

def example_usage():
    """
    Example usage of the MCP tools module.

    Examples:
        # Simple filesystem client
        from fpa_mcp_tools import create_filesystem_client
        client = create_filesystem_client()

        # Discover tools
        import asyncio
        tools = asyncio.run(client.discover_tools())

        # Call a tool
        result = asyncio.run(client.call_tool("read_file", {"path": "/data/test.txt"}))

        # Custom MCP server
        from fpa_mcp_tools import create_mcp_client
        client = create_mcp_client(host="mcp-services", port=3002, service_path="/db_mcp")

        # Configuration-based client
        from fpa_mcp_tools import MCPConfig, create_mcp_client
        config = MCPConfig.get_server_config("filesystem")
        client = create_mcp_client(config["host"], config["port"], config["service_path"])
    """
    pass
