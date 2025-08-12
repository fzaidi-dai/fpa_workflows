#!/usr/bin/env python3
"""
Simple test to debug math MCP server issues
"""

import os
import asyncio
import polars as pl
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import MCP tools
from mcp_tooling.mcp_tools_adk import MCPConfig, create_mcp_tools

def create_simple_test_data():
    """Create a simple test dataset."""
    scratch_pad = Path("scratch_pad")
    scratch_pad.mkdir(exist_ok=True)

    # Simple numbers for testing
    test_numbers = pl.DataFrame({
        "value": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    test_numbers.write_parquet(scratch_pad / "test_numbers.parquet")
    print("✅ Created test_numbers.parquet")
    return "test_numbers.parquet"

async def test_direct_mcp_call():
    """Test direct MCP server call without agent."""
    import httpx

    print("🔍 Testing direct MCP server call...")

    # Test tools/list
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:3002/math_mcp",
                json={"method": "tools/list", "params": {}},
                headers={"Content-Type": "application/json"}
            )
            tools_data = response.json()
            print(f"✅ Found {len(tools_data.get('tools', []))} tools")

            # Test a simple SUM call
            sum_response = await client.post(
                "http://localhost:3002/math_mcp",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": "SUM",
                        "arguments": {"values": "[10,20,30,40,50]"}
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            sum_result = sum_response.json()
            print(f"✅ SUM result: {sum_result}")
            return True

    except Exception as e:
        print(f"❌ Direct MCP call failed: {e}")
        return False

def test_tool_discovery():
    """Test tool discovery mechanism."""
    print("🔍 Testing tool discovery...")

    try:
        # Discover servers
        servers = MCPConfig.discover_available_servers()
        print(f"✅ Discovered servers: {[s['name'] for s in servers]}")

        # Get math server
        math_server = next((s for s in servers if s["name"] == "math_and_aggregation"), None)
        if not math_server:
            print("❌ Math server not found!")
            return False

        print(f"✅ Found math server: {math_server}")

        # Try to create tools
        try:
            math_tools = create_mcp_tools(math_server)
            print(f"✅ Created {len(math_tools)} tools")

            if len(math_tools) > 0:
                print(f"✅ First tool: {math_tools[0].name if hasattr(math_tools[0], 'name') else 'Unknown'}")
                return True
            else:
                print("❌ No tools created")
                return False

        except Exception as e:
            print(f"❌ Tool creation failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Tool discovery failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🧮 Simple Math MCP Server Debug Test")
    print("=" * 40)

    # Create test data
    test_file = create_simple_test_data()

    # Test 1: Direct MCP call
    direct_test = await test_direct_mcp_call()

    # Test 2: Tool discovery
    discovery_test = test_tool_discovery()

    # Summary
    print("\n" + "=" * 40)
    print("📊 DEBUG TEST SUMMARY")
    print("=" * 40)
    print(f"Direct MCP Call: {'✅ PASSED' if direct_test else '❌ FAILED'}")
    print(f"Tool Discovery: {'✅ PASSED' if discovery_test else '❌ FAILED'}")

    if direct_test and discovery_test:
        print("\n🎉 Basic functionality working!")
        return True
    else:
        print("\n⚠️ Issues found - need to debug further")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
