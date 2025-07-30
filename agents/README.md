# ADK Agents

This directory contains Google ADK agents for the FPA Agents project.

## Test Agent

The `test_agent.py` demonstrates:
- **LiteLLM Integration**: Uses OpenRouter with the `qwen/qwen3-coder:free` model
- **MCP Tools**: Connects to a remote MCP server via SseServerParams
- **Interactive File Operations**: Handles file reading, writing, and directory operations

### Prerequisites

1. **Environment Variables**: Ensure your `.env` file contains:
   ```
   OPEN_ROUTER_API_KEY=your-openrouter-api-key
   ```

2. **MCP Server**: The agent expects an MCP server running at `http://localhost:8090/sse`
   - Start your MCP server in Docker using the configuration in the `mcp/` folder
   - The server should expose file operation tools

### Usage

#### Via ADK Web Interface (Recommended)

**Using Docker (Recommended for Development):**

1. Start the entire stack with Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Run ADK web interface inside the container:
   ```bash
   docker-compose exec fpa-agents uv run adk web --host 0.0.0.0 --port 8080
   ```

3. Navigate to the web interface at `http://localhost:8080`

4. Select the `test_agent` from the available agents

5. Interact with the agent through the chat interface. Example queries:
   - "Write 'Hello This is a test' to a file called test.txt"
   - "Read the contents of config.json"
   - "List all files in the current directory"
   - "Create a new file with specific content"

**Local Development (Alternative):**

1. Start the ADK web interface locally:
   ```bash
   uv run adk web
   ```

2. Navigate to the web interface (typically `http://localhost:8080`)

#### Standalone Testing (Optional)

For development and testing purposes, you can run the agent directly:

```bash
uv run python agents/test_agent.py
```

This will execute a simple test query to verify the agent is working correctly.

### Agent Configuration

The agent is configured with:
- **Model**: `openrouter/qwen/qwen3-coder:free` via LiteLLM
- **Name**: `test_agent`
- **Tools**: All tools from the connected MCP server
- **Instructions**: Specialized for file operations and user assistance

### Troubleshooting

1. **Missing API Key**: Ensure `OPEN_ROUTER_API_KEY` is set in your `.env` file
2. **MCP Server Connection**: Verify the MCP server is running and accessible at `http://localhost:8090/sse`
3. **Dependencies**: Run `uv add litellm python-dotenv` if packages are missing

### Architecture

```
User Input → ADK Web Interface → test_agent.py → LiteLLM (OpenRouter) → qwen/qwen3-coder:free
                                      ↓
                                 MCPToolset → SseServerParams → MCP Server (Docker) → File Operations
```

The agent acts as a bridge between the user, the language model, and the file system operations provided by the MCP server.
