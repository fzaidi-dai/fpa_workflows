"""
Test Agent using Google ADK with LiteLLM and MCP Tools

This agent demonstrates:
- LiteLLM integration with OpenRouter (qwen/qwen3-coder:free model)
- MCP toolset connection to remote server via SseServerParams
- Interactive file operations through ADK web interface
"""

import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
def validate_environment():
    """Validate that required environment variables are present."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables. "
            "Please ensure your .env file contains this key."
        )
    return openrouter_api_key

# Validate environment on import
validate_environment()

# Create custom filesystem tools that use our MCP server
def create_filesystem_tools():
    """Create filesystem tools that connect to our MCP server."""
    import asyncio
    from mcp_tooling.fpa_mcp_tools import create_filesystem_client
    from google.adk.tools.function_tool import FunctionTool

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

# Define the main agent for ADK web interface
root_agent = LlmAgent(
    model=LiteLlm(model="openrouter/qwen/qwen3-coder:free"),
    name="test_agent",
    instruction=(
        "You are a helpful assistant that can read and write files using the available filesystem tools. "
        "Help users with file operations like reading, writing, and managing files in the data, scratch_pad, and memory directories. "
        "When users ask you to create or modify files, use the write_file tool. When they want to read files, use the read_file tool. "
        "Be clear about what operations you're performing and provide helpful feedback to the user."
    ),
    tools=create_filesystem_tools()
)

# Optional: Add a simple test function for standalone execution
async def test_agent_standalone():
    """
    Test function for standalone agent execution (optional).
    This can be used for testing outside of ADK web interface.
    """
    import asyncio
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types

    try:
        # Create session service
        session_service = InMemorySessionService()

        # Create session
        session = await session_service.create_session(
            app_name="test_agent_app",
            user_id="test_user",
            session_id="test_session"
        )

        # Create runner
        runner = Runner(
            agent=root_agent,
            app_name="test_agent_app",
            session_service=session_service
        )

        # Test query
        test_query = "Write 'Hello This is a test' to a file called test.txt"
        print(f"Testing agent with query: {test_query}")

        # Create message content
        content = types.Content(
            role='user',
            parts=[types.Part(text=test_query)]
        )

        # Run agent
        events_async = runner.run_async(
            session_id=session.id,
            user_id=session.user_id,
            new_message=content
        )

        # Process events
        async for event in events_async:
            print(f"Event: {event}")

    except Exception as e:
        print(f"Error during standalone test: {e}")
        raise

if __name__ == "__main__":
    """
    Standalone execution for testing purposes.
    For normal usage, this agent should be accessed via 'adk web' command.
    """
    import asyncio

    print("Running test agent in standalone mode...")
    print("Note: For normal usage, run 'adk web' and interact with the agent through the web interface.")

    try:
        asyncio.run(test_agent_standalone())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        print("Test completed.")
