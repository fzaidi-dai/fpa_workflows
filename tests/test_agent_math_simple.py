#!/usr/bin/env python3
"""
Simple agent test for Math MCP Server using localhost URLs
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

async def test_agent_with_math_tools():
    """Test agent using math tools with localhost URLs."""
    print("🤖 Testing Agent with Math MCP Tools")
    print("=" * 50)

    try:
        # Create a custom server config for localhost testing
        math_server = {
            "name": "math_and_aggregation",
            "base_url": "http://localhost:3002",
            "service_path": "/math_mcp",
            "description": "Math and aggregation tools for financial analysis"
        }

        print(f"✅ Using math server: {math_server['name']}")

        # Create tools
        math_tools = create_mcp_tools(math_server)
        print(f"✅ Created {len(math_tools)} math tools")

        # Create agent
        agent = LlmAgent(
            model=LiteLlm(model="openrouter/qwen/qwen3-coder"),
            name="math_test_agent",
            instruction=(
                "You are a mathematical assistant. Use the available math tools to perform calculations. "
                "When given numbers in brackets like [1,2,3], use them directly as input to the tools. "
                "Be concise and show the calculation results."
            ),
            tools=math_tools
        )

        # Create session
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="math_test",
            user_id="test_user",
            session_id="math_session"
        )

        # Create runner
        runner = Runner(
            agent=agent,
            app_name="math_test",
            session_service=session_service
        )

        # Test simple calculations
        test_queries = [
            "Calculate the sum of [10, 20, 30, 40, 50]",
            "Find the average of [100, 200, 300]",
            "What is the square root of [16, 25, 36]?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: {query} ---")

            content = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )

            events_async = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            )

            response_parts = []
            async for event in events_async:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

            response = ''.join(response_parts)
            if response.strip():
                print(f"✅ Agent response: {response[:200]}...")
            else:
                print("❌ No response received")

        print("\n✅ Agent testing completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Simple Agent Math MCP Test")
    print("=" * 40)

    # Validate environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False

    try:
        result = asyncio.run(test_agent_with_math_tools())
        return result
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    """Run the simple agent test."""
    try:
        result = main()
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
