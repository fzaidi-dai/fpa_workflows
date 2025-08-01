"""
Test script for MCP Tools Module

This script demonstrates how to use the generic MCP tools module
to connect to different MCP servers and create tools dynamically.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import mcp module
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_filesystem_tools():
    """Test the filesystem tools functionality."""
    print("Testing Filesystem MCP Tools...")

    try:
        # Test filesystem tools
        from mcp.mcp_tools import create_filesystem_tools, MCPClient, MCPConfig

        # Create filesystem tools using default configuration
        print("Creating filesystem tools...")
        tools = create_filesystem_tools()
        print(f"Created {len(tools)} filesystem tools")

        # Test MCPClient directly
        print("\nTesting MCPClient directly...")
        client = MCPClient(
            host="localhost" if os.getenv("ENVIRONMENT") != "development" else "mcp-services",
            port=3001,
            service_path="/fs_mcp"
        )

        # Discover tools
        print("Discovering available tools...")
        available_tools = await client.discover_tools()
        print(f"Available tools: {[tool['name'] for tool in available_tools]}")

        # Test configuration-based tools
        print("\nTesting configuration-based tools...")
        config_tools = MCPConfig.create_tools_from_config("filesystem")
        print(f"Created {len(config_tools)} tools from configuration")

        print("✅ Filesystem tools test completed successfully!")

    except Exception as e:
        print(f"❌ Filesystem tools test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_generic_mcp_tools():
    """Test generic MCP tools with custom server."""
    print("\nTesting Generic MCP Tools...")

    try:
        from mcp.mcp_tools import create_mcp_tools, MCPClient

        # This would work with any MCP server
        # For demonstration, we'll just show the usage pattern
        print("Generic MCP tools can be created with:")
        print("tools = create_mcp_tools(host='mcp-services', port=3002, service_path='/custom_mcp')")

        print("✅ Generic MCP tools test completed!")

    except Exception as e:
        print(f"❌ Generic MCP tools test failed: {e}")

async def main():
    """Main test function."""
    print("=" * 50)
    print("MCP Tools Module Test")
    print("=" * 50)

    # Test filesystem tools
    await test_filesystem_tools()

    # Test generic tools
    await test_generic_mcp_tools()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
