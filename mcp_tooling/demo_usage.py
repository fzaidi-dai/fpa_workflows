"""
Demo Usage of MCP Tools Module

This script demonstrates various ways to use the MCP tools module
with different agent frameworks and MCP servers.
"""

import asyncio

async def demo_filesystem_operations():
    """Demonstrate filesystem operations using the MCP client."""
    print("=== Filesystem Operations Demo ===")

    # Import the filesystem client
    from fpa_mcp_tools import create_filesystem_client

    # Create filesystem client
    client = create_filesystem_client()
    print(f"Connected to MCP server at: {client.base_url}")

    # Discover available tools
    tools = await client.discover_tools()
    print(f"Available tools: {[tool['name'] for tool in tools]}")

    # Write a file
    print("\n1. Writing a test file...")
    write_result = await client.call_tool("write_file", {
        "path": "/mcp-data/scratch_pad/demo.txt",
        "content": "Hello from MCP Tools Demo!\nThis file was created using the generic MCP client."
    })
    print(f"Write result: {write_result.get('message', 'Success')}")

    # Read the file back
    print("\n2. Reading the test file...")
    read_result = await client.call_tool("read_file", {
        "path": "/mcp-data/scratch_pad/demo.txt"
    })
    if read_result.get("success"):
        print(f"File content:\n{read_result['content']}")
    else:
        print(f"Error reading file: {read_result.get('error')}")

    # List directory contents
    print("\n3. Listing directory contents...")
    list_result = await client.call_tool("list_directory", {
        "path": "/mcp-data/scratch_pad"
    })
    if list_result.get("success"):
        items = list_result.get("items", [])
        print(f"Directory contents ({len(items)} items):")
        for item in items:
            item_type = item.get("type", "unknown")
            item_name = item.get("name", "unknown")
            print(f"  [{item_type.upper()}] {item_name}")
    else:
        print(f"Error listing directory: {list_result.get('error')}")

async def demo_generic_mcp_client():
    """Demonstrate using the generic MCP client for any server."""
    print("\n=== Generic MCP Client Demo ===")

    from fpa_mcp_tools import MCPClient

    # Create a generic client (this would work with any MCP server)
    client = MCPClient(
        host="mcp-services",
        port=3001,
        service_path="/fs_mcp"
    )

    print(f"Generic client created for: {client.base_url}")

    # Show how this pattern works for different servers
    print("\nThis same pattern works for any MCP server:")
    print("  Database MCP: MCPClient(host='mcp-services', port=3002, service_path='/db_mcp')")
    print("  API MCP:      MCPClient(host='mcp-services', port=3003, service_path='/api_mcp')")
    print("  AI MCP:       MCPClient(host='mcp-services', port=3004, service_path='/ai_mcp')")

async def demo_agent_integration():
    """Demonstrate how to integrate with agent frameworks."""
    print("\n=== Agent Framework Integration Demo ===")

    print("Example 1: Google ADK Integration")
    print("""
from fpa_mcp_tools import create_filesystem_client
from google.adk.tools.function_tool import FunctionTool

def create_filesystem_tools():
    client = create_filesystem_client()

    async def write_file_tool(path: str, content: str) -> str:
        result = await client.call_tool("write_file", {"path": path, "content": content})
        if result.get("success"):
            return f"Successfully wrote {result.get('bytes_written', 0)} bytes"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    return [FunctionTool(func=write_file_tool)]
    """)

    print("Example 2: Custom Agent Integration")
    print("""
from fpa_mcp_tools import MCPClient

class MyAgent:
    def __init__(self):
        self.fs_client = MCPClient("mcp-services", 3001, "/fs_mcp")
        self.db_client = MCPClient("mcp-services", 3002, "/db_mcp")

    async def process_file(self, file_path: str):
        # Use filesystem MCP
        result = await self.fs_client.call_tool("read_file", {"path": file_path})
        return result

    async def query_database(self, sql: str):
        # Use database MCP
        result = await self.db_client.call_tool("execute_query", {"sql": sql})
        return result
    """)

async def main():
    """Run all demos."""
    print("🤖 MCP Tools Module Usage Demo")
    print("=" * 50)

    await demo_filesystem_operations()
    await demo_generic_mcp_client()
    await demo_agent_integration()

    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nKey Benefits:")
    print("  • Framework independent - works with any agent framework")
    print("  • Generic client - works with any MCP server")
    print("  • Easy configuration - environment-based settings")
    print("  • Robust error handling - comprehensive error management")
    print("  • Async support - non-blocking operations")

if __name__ == "__main__":
    asyncio.run(main())
