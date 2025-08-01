"""
Test script for MCP Client Module

This script demonstrates how to use the generic MCP client module
to connect to MCP servers and call tools.
"""

import asyncio
import os
import sys
from pathlib import Path

async def test_filesystem_client():
    """Test the filesystem client functionality."""
    print("Testing Filesystem MCP Client...")

    try:
        # Test filesystem client
        from fpa_mcp_tools import create_filesystem_client, MCPClient, MCPConfig

        # Create filesystem client using default configuration
        print("Creating filesystem client...")
        client = create_filesystem_client()
        print(f"Created client for {client.base_url}")

        # Test MCPClient directly
        print("\nTesting MCPClient directly...")
        direct_client = MCPClient(
            host="localhost" if os.getenv("ENVIRONMENT") != "development" else "mcp-services",
            port=3001,
            service_path="/fs_mcp"
        )
        print(f"Direct client created for {direct_client.base_url}")

        # Discover tools (this requires the MCP server to be running)
        try:
            print("Discovering available tools...")
            available_tools = await direct_client.discover_tools()
            print(f"Available tools: {[tool['name'] for tool in available_tools]}")
        except Exception as e:
            print(f"Could not discover tools (MCP server may not be running): {e}")

        # Test configuration-based client
        print("\nTesting configuration-based client...")
        config = MCPConfig.get_server_config("filesystem")
        print(f"Configuration: {config}")

        config_client = MCPClient(
            host=config["host"],
            port=config["port"],
            service_path=config["service_path"]
        )
        print(f"Config client created for {config_client.base_url}")

        print("✅ Filesystem client test completed successfully!")

    except Exception as e:
        print(f"❌ Filesystem client test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_generic_mcp_client():
    """Test generic MCP client with custom server."""
    print("\nTesting Generic MCP Client...")

    try:
        from fpa_mcp_tools import MCPClient

        # This would work with any MCP server
        # For demonstration, we'll just show the usage pattern
        print("Generic MCP client can be created with:")
        print("client = MCPClient(host='mcp-services', port=3002, service_path='/custom_mcp')")

        print("✅ Generic MCP client test completed!")

    except Exception as e:
        print(f"❌ Generic MCP client test failed: {e}")

async def main():
    """Main test function."""
    print("=" * 50)
    print("MCP Client Module Test")
    print("=" * 50)

    # Test filesystem client
    await test_filesystem_client()

    # Test generic client
    await test_generic_mcp_client()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
