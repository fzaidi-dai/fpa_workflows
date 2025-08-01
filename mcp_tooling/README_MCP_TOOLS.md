# MCP Tools Module

This module provides a generic interface for connecting to MCP (Model Context Protocol) servers and creating tools that can be used with Google ADK agents.

## Features

- **Generic MCP Client**: Connect to any MCP server
- **Dynamic Tool Discovery**: Automatically discover available tools from MCP servers
- **Function Tool Creation**: Create Google ADK FunctionTool objects dynamically
- **Multiple Server Support**: Easy configuration for multiple MCP servers
- **Error Handling**: Robust error handling and logging
- **Configuration Management**: Environment-based configuration

## Installation

The module is automatically available as part of the FPA Agents project. No additional installation is required.

## Usage

### Simple Filesystem Tools

```python
from mcp.mcp_tools import create_filesystem_tools

# Create filesystem tools with default configuration
tools = create_filesystem_tools()

# Use tools with ADK agents
agent = LlmAgent(
    # ... other configuration
    tools=tools
)
```

### Generic MCP Server Tools

```python
from mcp.mcp_tools import create_mcp_tools

# Create tools for any MCP server
tools = create_mcp_tools(
    host="mcp-services",
    port=3002,
    service_path="/custom_mcp"
)
```

### Direct MCP Client Usage

```python
from mcp.mcp_tools import MCPClient

# Create MCP client
client = MCPClient(
    host="localhost",
    port=3001,
    service_path="/fs_mcp"
)

# Discover available tools
async def discover_tools():
    tools = await client.discover_tools()
    print(f"Available tools: {tools}")

# Call specific tools
async def call_tool():
    result = await client.call_tool("read_file", {"path": "/data/test.txt"})
    print(f"File content: {result}")
```

### Configuration-Based Tools

```python
from mcp.mcp_tools import MCPConfig

# Create tools based on server configuration
tools = MCPConfig.create_tools_from_config("filesystem")
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
3. **Use the generic tools** in your agents:

```python
# Example: Database MCP server
from mcp.mcp_tools import create_mcp_tools

db_tools = create_mcp_tools(
    host="mcp-services",
    port=3002,
    service_path="/db_mcp"
)

# Use with agent
agent = LlmAgent(
    # ... other configuration
    tools=db_tools
)
```

## Testing

Run the test script to verify the module works correctly:

```bash
cd mcp
python test_mcp_tools.py
```

## Architecture

The module follows a clean architecture:

```
┌─────────────────┐
│   Your Agent    │
└─────────┬───────┘
          │
┌─────────▼───────┐
│  MCP Tools API  │ ◄── Easy to use functions
└─────────┬───────┘
          │
┌─────────▼───────┐
│   MCPClient     │ ◄── Generic MCP client
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
- **Path validation**: Security checks for file operations
- **Logging**: Detailed logging for debugging

## Performance

- **Caching**: Tool discovery results are cached
- **Async operations**: Non-blocking HTTP requests
- **Connection pooling**: Efficient HTTP client usage
- **Resource limits**: Configurable timeouts and limits

## Security

- **Path validation**: Restricted directory access
- **Environment isolation**: Container-based security
- **Input validation**: Sanitized tool arguments
- **Error sanitization**: No sensitive information in error messages

## Extending the Module

To add new features:

1. **New MCP client features**: Extend the `MCPClient` class
2. **New factory functions**: Add convenience functions for common patterns
3. **New configuration options**: Extend the `MCPConfig` class
4. **New utility functions**: Add helper functions for common operations

## Troubleshooting

### Common Issues

1. **Connection refused**: Check if MCP services are running
2. **Timeout errors**: Increase timeout values in client configuration
3. **Permission denied**: Check directory access permissions
4. **Invalid tool responses**: Verify MCP server is returning correct format

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

## Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

## License

This module is part of the FPA Agents project and follows the same license terms.
