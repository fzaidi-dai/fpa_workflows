#!/usr/bin/env python3
"""
Standalone test for Math and Aggregation MCP Server

This script tests the math server functionality using the same pattern as the agent.py
but specifically focuses on testing math functions.
"""

import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Load environment variables
load_dotenv()

# Import MCP tools
from mcp_tooling.mcp_tools_adk import MCPConfig, create_mcp_tools

def validate_environment():
    """Validate that required environment variables are present."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables. "
            "Please ensure your .env file contains this key."
        )
    return openrouter_api_key

async def test_math_server_unfiltered():
    """Test the math server with all available tools (unfiltered)."""
    print("\n=== Testing Math Server - Unfiltered (All Tools) ===")

    # Discover available servers
    servers = MCPConfig.discover_available_servers()
    print(f"Discovered servers: {[s['name'] for s in servers]}")

    # Get math server
    math_server = next((s for s in servers if s["name"] == "math_and_aggregation"), None)
    if not math_server:
        print("❌ Math server not found!")
        return False

    print(f"✅ Found math server: {math_server['name']} at {math_server['base_url']}{math_server.get('service_path', '')}")

    # Create tools for math server
    math_tools = create_mcp_tools(math_server)
    print(f"✅ Created {len(math_tools)} math tools")

    # Create agent with math tools
    agent = LlmAgent(
        model=LiteLlm(model="openrouter/qwen/qwen3-coder"),
        name="math_test_agent",
        instruction=(
            "You are a math assistant that can perform calculations and aggregations. "
            "Use the available math tools to help users with mathematical operations. "
            "Always show your work and explain the calculations you're performing."
        ),
        tools=math_tools
    )

    # Create session service and session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="math_test_app",
        user_id="test_user",
        session_id="math_test_session"
    )

    # Create runner
    runner = Runner(
        agent=agent,
        app_name="math_test_app",
        session_service=session_service
    )

    # Test queries
    test_queries = [
        "Calculate the sum of numbers 10, 20, 30, 40, 50",
        "Find the average of the numbers 5, 15, 25, 35",
        "Calculate the standard deviation of 2, 4, 6, 8, 10",
        "Find the maximum value among 100, 250, 175, 300, 125"
    ]

    success_count = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        try:
            # Create message content
            content = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )

            # Run agent
            events_async = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            )

            # Process events
            response_parts = []
            async for event in events_async:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

            response = ''.join(response_parts)
            if response.strip():
                print(f"✅ Response: {response[:200]}...")
                success_count += 1
            else:
                print("❌ No response received")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n=== Unfiltered Test Results: {success_count}/{len(test_queries)} successful ===")
    return success_count == len(test_queries)

async def test_math_server_filtered():
    """Test the math server with filtered tools (only basic math operations)."""
    print("\n=== Testing Math Server - Filtered (Basic Math Only) ===")

    # Discover available servers
    servers = MCPConfig.discover_available_servers()

    # Get math server
    math_server = next((s for s in servers if s["name"] == "math_and_aggregation"), None)
    if not math_server:
        print("❌ Math server not found!")
        return False

    # Create filtered tools (only basic operations)
    filtered_tools = create_mcp_tools(
        math_server,
        tool_filter=["SUM", "AVERAGE", "MAX", "MIN"]
    )
    print(f"✅ Created {len(filtered_tools)} filtered math tools")

    # Create agent with filtered tools
    agent = LlmAgent(
        model=LiteLlm(model="openrouter/qwen/qwen3-coder"),
        name="math_filtered_agent",
        instruction=(
            "You are a basic math assistant with limited operations. "
            "You can only perform sum, average, max, and min calculations. "
            "Use only the available tools for calculations."
        ),
        tools=filtered_tools
    )

    # Create session service and session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="math_filtered_app",
        user_id="test_user",
        session_id="math_filtered_session"
    )

    # Create runner
    runner = Runner(
        agent=agent,
        app_name="math_filtered_app",
        session_service=session_service
    )

    # Test queries for filtered tools
    test_queries = [
        "Calculate the sum of 1, 2, 3, 4, 5",
        "Find the average of 10, 20, 30",
        "What's the maximum of 50, 75, 25, 100?",
        "Find the minimum of 8, 3, 12, 1, 9"
    ]

    success_count = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Filtered Test {i}: {query} ---")
        try:
            # Create message content
            content = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )

            # Run agent
            events_async = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            )

            # Process events
            response_parts = []
            async for event in events_async:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

            response = ''.join(response_parts)
            if response.strip():
                print(f"✅ Response: {response[:200]}...")
                success_count += 1
            else:
                print("❌ No response received")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n=== Filtered Test Results: {success_count}/{len(test_queries)} successful ===")
    return success_count == len(test_queries)

async def main():
    """Main test function."""
    print("🧮 Math and Aggregation MCP Server Test")
    print("=" * 50)

    try:
        # Validate environment
        validate_environment()
        print("✅ Environment validation passed")

        # Test unfiltered (all tools)
        unfiltered_success = await test_math_server_unfiltered()

        # Test filtered (basic tools only)
        filtered_success = await test_math_server_filtered()

        # Summary
        print("\n" + "=" * 50)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 50)
        print(f"Unfiltered Test: {'✅ PASSED' if unfiltered_success else '❌ FAILED'}")
        print(f"Filtered Test: {'✅ PASSED' if filtered_success else '❌ FAILED'}")

        if unfiltered_success and filtered_success:
            print("\n🎉 ALL TESTS PASSED! Math MCP Server is working correctly.")
            return True
        else:
            print("\n⚠️  Some tests failed. Please check the server configuration.")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    """Run the math server tests."""
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
