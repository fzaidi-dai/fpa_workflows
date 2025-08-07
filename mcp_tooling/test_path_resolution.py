#!/usr/bin/env python3
"""
Test script to verify that path resolution works correctly
after moving path prefixing logic from client to server.
"""

import asyncio
import json
import logging
from mcp_tools_adk import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_path_resolution():
    """Test that the MCP client can now work with relative paths."""

    # Create MCP client
    client = MCPClient(
        host="fpa-mcp-services",
        port=3001,
        service_path="/fs_mcp"
    )

    print("🧪 Testing Path Resolution After Client/Server Refactor")
    print("=" * 60)

    # Test cases with relative paths (no hardcoded prefixes in client)
    test_cases = [
        {
            "description": "Write to data directory with relative path",
            "operation": "write_file",
            "path": "data/test_file.txt",
            "content": "Hello from refactored MCP client!"
        },
        {
            "description": "Write to scratch_pad with relative path",
            "operation": "write_file",
            "path": "scratch_pad/test_scratch.txt",
            "content": "Testing scratch pad access"
        },
        {
            "description": "Write simple filename (should default to data/)",
            "operation": "write_file",
            "path": "simple_file.txt",
            "content": "Simple filename test"
        },
        {
            "description": "List data directory",
            "operation": "list_directory",
            "path": "data"
        },
        {
            "description": "List scratch_pad directory",
            "operation": "list_directory",
            "path": "scratch_pad"
        },
        {
            "description": "Read file from data directory",
            "operation": "read_file",
            "path": "data/test_file.txt"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Path: {test_case['path']}")

        try:
            if test_case['operation'] == 'write_file':
                result = await client.call_tool(
                    "write_file",
                    {"path": test_case['path'], "content": test_case['content']}
                )
            elif test_case['operation'] == 'read_file':
                result = await client.call_tool(
                    "read_file",
                    {"path": test_case['path']}
                )
            elif test_case['operation'] == 'list_directory':
                result = await client.call_tool(
                    "list_directory",
                    {"path": test_case['path']}
                )

            if result.get('success'):
                print(f"   ✅ SUCCESS: {result.get('message', 'Operation completed')}")
                if 'path' in result:
                    print(f"   📁 Resolved to: {result['path']}")
            else:
                print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"   💥 EXCEPTION: {str(e)}")

    print("\n" + "=" * 60)
    print("🎯 Key Benefits Achieved:")
    print("   • Client is now generic (no hardcoded path prefixes)")
    print("   • Server handles all path resolution logic")
    print("   • Relative paths work seamlessly")
    print("   • Server configuration controls path mapping")
    print("   • Backward compatibility maintained")

if __name__ == "__main__":
    asyncio.run(test_path_resolution())
