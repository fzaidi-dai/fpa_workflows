# MCP Tooling Module

This module provides a generic interface for connecting to MCP (Model Context Protocol) servers and creating tools that can be used with any agent framework.

## Features

- **Generic MCP Client**: Connect to any MCP server using pure HTTP communication
- **Dynamic Tool Discovery**: Automatically discover available tools from MCP servers
- **Multiple Server Support**: Easy configuration for multiple MCP servers
- **Error Handling**: Robust error handling and logging
- **Configuration Management**: Environment-based configuration
- **Framework Independent**: No dependencies on specific agent frameworks

## Installation

The module is automatically available as part of the FPA Agents project. No additional installation is required.

## Usage

### Simple Filesystem Client

```python
from mcp_tooling.fpa_mcp_tools import create_filesystem_client

# Create filesystem client with default configuration
client = create_filesystem_client()

# Discover available tools
import asyncio
tools = asyncio.run(client.discover_tools())
print(f"Available tools: {[tool['name'] for tool in tools]}")

# Call specific tools
result = asyncio.run(client.call_tool("read_file", {"path": "/data/test.txt"}))
print(f"File content: {result}")
```

### Generic MCP Server Client

```python
from mcp_tooling.fpa_mcp_tools import MCPClient

# Create client for any MCP server
client = MCPClient(
    host="mcp-services",
    port=3002,
    service_path="/custom_mcp"
)

# Use the client
import asyncio
tools = asyncio.run(client.discover_tools())
result = asyncio.run(client.call_tool("custom_tool", {"param": "value"}))
```

### Configuration-Based Client

```python
from mcp_tooling.fpa_mcp_tools import MCPConfig, MCPClient

# Get server configuration
config = MCPConfig.get_server_config("filesystem")

# Create client from configuration
client = MCPClient(
    host=config["host"],
    port=config["port"],
    service_path=config["service_path"]
)
```

## Environment Variables

The module supports the following environment variables for configuration:

- `ENVIRONMENT`: Set to "development" to use Docker service names
- `MCP_{SERVER_NAME}_HOST`: Custom host for specific MCP server
- `MCP_{SERVER_NAME}_PORT`: Custom port for specific MCP server
- `MCP_{SERVER_NAME}_PATH`: Custom service path for specific MCP server

Example:
```bash
MCP_FILESYSTEM_HOST=localhost
MCP_FILESYSTEM_PORT=3001
MCP_FILESYSTEM_PATH=/fs_mcp
```

## Adding New MCP Servers

To add support for new MCP servers:

1. **Create the MCP server** following the existing patterns
2. **Update docker-compose configuration** to include the new server
3. **Use the generic client** in your agents:

```python
# Example: Database MCP server
from mcp_tooling.fpa_mcp_tools import MCPClient

db_client = MCPClient(
    host="mcp-services",
    port=3002,
    service_path="/db_mcp"
)

# Use with your agent framework
import asyncio
result = asyncio.run(db_client.call_tool("query_database", {"sql": "SELECT * FROM users"}))
```

## Testing

Run the test script to verify the module works correctly:

```bash
cd mcp_tooling
uv run python test_mcp_client.py
```

## Architecture

The module follows a clean architecture:

```
┌─────────────────┐
│   Your Agent    │
└─────────┬───────┘
          │
┌─────────▼───────┐
│  MCP Client     │ ◄── Pure HTTP client, no framework dependencies
└─────────┬───────┘
          │
┌─────────▼───────┐
│   MCP Servers   │ ◄── Filesystem, Database, etc.
└─────────────────┘
```

## Error Handling

The module includes comprehensive error handling:

- **Network errors**: Automatic retry and timeout handling
- **Invalid responses**: Graceful handling of malformed MCP responses
- **Logging**: Detailed logging for debugging

## Performance

- **Async operations**: Non-blocking HTTP requests
- **Connection pooling**: Efficient HTTP client usage
- **Resource limits**: Configurable timeouts and limits

## Security

- **Input validation**: Sanitized tool arguments
- **Error sanitization**: No sensitive information in error messages

## Extending the Module

To add new features:

1. **New MCP client features**: Extend the `MCPClient` class
2. **New factory functions**: Add convenience functions for common patterns
3. **New configuration options**: Extend the `MCPConfig` class

## Troubleshooting

### Common Issues

1. **Connection refused**: Check if MCP services are running
2. **Timeout errors**: Increase timeout values in client configuration
3. **Invalid tool responses**: Verify MCP server is returning correct format

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check MCP server health:

```bash
curl http://localhost:3001/health
```

## Integration with Agent Frameworks

### Google ADK Integration

To use with Google ADK agents, create wrapper functions that convert MCP results to FunctionTool objects:

```python
from mcp_tooling.fpa_mcp_tools import create_filesystem_client
from google.adk.tools.function_tool import FunctionTool

def create_filesystem_tools():
    client = create_filesystem_client()

    async def write_file_tool(path: str, content: str) -> str:
        result = await client.call_tool("write_file", {"path": path, "content": content})
        # Convert result to string for ADK FunctionTool
        if result.get("success"):
            return f"Successfully wrote {result.get('bytes_written', 0)} bytes"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    return [FunctionTool(func=write_file_tool)]
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

## License

This module is part of the FPA Agents project and follows the same license terms.
