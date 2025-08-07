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
from typing import Any, Dict, Union, List, Optional, Callable
from pathlib import Path
import httpx
# Import FunctionTool only when needed to avoid circular imports

# Import FunctionTool at the top level to avoid NameError
try:
    from google.adk.tools.function_tool import FunctionTool
except ImportError:
    FunctionTool = None

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
        Create an async function that wraps an MCP tool call with proper parameter signature.

        Args:
            tool_def: Tool definition from MCP server

        Returns:
            Async function that calls the MCP tool with correct parameter names
        """
        tool_name = tool_def["name"]
        description = tool_def.get("description", "")
        input_schema = tool_def.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Create function signature dynamically based on MCP schema
        import inspect
        from typing import Optional

        # Build parameter list for the function signature
        params = []
        for param_name, param_info in properties.items():
            param_type = str  # Default to string type
            if param_name in required:
                params.append(inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=param_type))
            else:
                params.append(inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                              annotation=Optional[param_type], default=None))

        # Create the function dynamically
        async def tool_function(*args, **kwargs) -> str:
            """Dynamic tool function that calls the MCP server."""
            logger.info(f"Tool {tool_name} called with args: {args}, kwargs: {kwargs}")

            # Bind arguments to parameter names
            sig = inspect.Signature(params)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Prepare arguments for MCP call with path prefixing
            mapped_args = {}
            for param_name, param_value in bound_args.arguments.items():
                if param_value is not None:
                    # Handle path prefixing for filesystem operations
                    if param_name == "path" and param_value:
                        if not str(param_value).startswith("/mcp-data/"):
                            # Add the MCP data prefix for relative paths
                            if str(param_value).startswith("data/") or str(param_value).startswith("scratch_pad/") or str(param_value).startswith("memory/"):
                                param_value = f"/mcp-data/{param_value}"
                            elif param_value in ["data", "scratch_pad", "memory"]:
                                param_value = f"/mcp-data/{param_value}"
                            elif param_value == ".":
                                param_value = "/mcp-data"
                            else:
                                # Default to data directory for simple filenames
                                param_value = f"/mcp-data/data/{param_value}"

                    mapped_args[param_name] = param_value

            logger.info(f"Final mapped_args for {tool_name}: {mapped_args}")
            result = await self.call_tool(tool_name, mapped_args)

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

        # Set the correct signature on the function
        tool_function.__signature__ = inspect.Signature(params)
        tool_function.__name__ = f"mcp_{tool_name}"
        tool_function.__doc__ = f"MCP tool: {description}"

        return tool_function


    def create_function_tools(self,tooldefs: Optional[List[Dict]] = None) -> List[FunctionTool]:
        """
        Create Google ADK FunctionTool objects for all available MCP tools or
        passed tool definitions in toolsDef.

        Returns:
            List of FunctionTool objects that can be used with ADK agents

        Raises:
            Exception: If tool discovery fails
        """
        # Import FunctionTool here to avoid circular imports
        from google.adk.tools.function_tool import FunctionTool

        import asyncio

        # Run async discovery in sync context
        if tooldefs is not None:
            tools_list = tooldefs
            logger.info(f"Using provided tool definitions: {[t['name'] for t in tools_list]}")
        else:
            tools_list = []
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
            logger.info(f"Creating FunctionTool for: {tool_def['name']} with schema: {tool_def.get('inputSchema', {})}")
            tool_function = self._create_tool_function(tool_def)
            function_tool = FunctionTool(func=tool_function)
            function_tools.append(function_tool)

        logger.info(f"Created {len(function_tools)} FunctionTool objects")
        return function_tools


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

    @staticmethod
    def discover_available_servers(config_file: Union[str, Path] = None) -> List[Dict[str, Any]]:
        """
        Discover available MCP servers by reading from configuration file.

        Args:
            config_file: Path to the MCP servers configuration file.
                         Defaults to mcp_tooling/config/mcp_servers.json

        Returns:
            List of server configurations with base_url and service_path
        """
        if config_file is None:
            # Default to the standard config file location
            config_file = Path(__file__).parent / "config" / "mcp_servers.json"

        config_file = Path(config_file)

        if not config_file.exists():
            logger.warning(f"Config file {config_file} not found, returning empty server list")
            return []

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Return the servers list directly
            return config.get("servers", [])

        except Exception as e:
            logger.error(f"Failed to load server configuration from {config_file}: {e}")
            return []

def discover_available_servers(config_file: Union[str, Path] = None) -> List[Dict[str, Any]]:
    """
    Discover available MCP servers by reading from configuration file.

    Args:
        config_file: Path to the MCP servers configuration file.
                     Defaults to mcp_tooling/config/mcp_servers.json

    Returns:
        List of server configurations with base_url and service_path
    """
    if config_file is None:
        # Default to the standard config file location
        config_file = Path(__file__).parent / "config" / "mcp_servers.json"

    config_file = Path(config_file)

    if not config_file.exists():
        logger.warning(f"Config file {config_file} not found, returning empty server list")
        return []

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Return the servers list directly
        return config.get("servers", [])

    except Exception as e:
        logger.error(f"Failed to load server configuration from {config_file}: {e}")
        return []

def create_mcp_tools(server_config: Dict[str, Any], tool_filter: Optional[List[str]] = None):
    """
    Create tools that connect to an MCP server based on server configuration.

    Args:
        server_config: Server configuration dictionary containing base_url and service_path
        tool_filter: Optional list of tool names to include. If None, all tools are included.

    Returns:
        List of FunctionTool objects for the MCP server's tools
    """
    # Extract host and port from base_url
    from urllib.parse import urlparse
    parsed_url = urlparse(server_config["base_url"])
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 80

    # Get service path from config or default to empty string
    service_path = server_config.get("service_path", "")

    client = MCPClient(host=host, port=port, service_path=service_path)

    # If tool_filter is provided, create tools with filtered definitions
    if tool_filter is not None:
        # First discover all tools to get their definitions
        import asyncio
        try:
            # Try to use existing event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to handle this differently
                    # For now, we'll create a task and wait for it
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, client.discover_tools())
                        tools_list = future.result()
                else:
                    tools_list = loop.run_until_complete(client.discover_tools())
            except RuntimeError:
                # No event loop running, create one
                tools_list = asyncio.run(client.discover_tools())

            # Filter tools based on the tool_filter list
            filtered_tools = [tool for tool in tools_list if tool["name"] in tool_filter]
            logger.info(f"Filtered tools: {[t['name'] for t in filtered_tools]} from {[t['name'] for t in tools_list]}")
            return client.create_function_tools(tooldefs=filtered_tools)
        except Exception as e:
            logger.error(f"Failed to create filtered tools: {e}")
            raise
    else:
        # Create all tools
        return client.create_function_tools()
