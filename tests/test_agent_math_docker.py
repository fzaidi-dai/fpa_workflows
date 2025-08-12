#!/usr/bin/env python3
"""
Test Agent with Math MCP Tools inside Docker Container

This test demonstrates the agent using math tools through the MCP server
running inside the Docker environment with proper network connectivity.
"""

import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Import MCP tools
from mcp_tooling.mcp_tools_adk import MCPConfig, create_mcp_tools

async def test_math_agent_in_docker():
    """Test agent using math tools inside Docker environment."""
    print("🐳 Testing Math Agent inside Docker Container")
    print("=" * 60)

    try:
        # Discover all available servers
        servers = MCPConfig.discover_available_servers()
        print(f"✅ Discovered {len(servers)} MCP servers:")
        for server in servers:
            print(f"   - {server['name']}: {server['base_url']}")

        # Find the math server
        math_server = next((s for s in servers if s["name"] == "math_and_aggregation"), None)
        if not math_server:
            print("❌ Math server not found in configuration")
            return False

        print(f"✅ Using math server: {math_server['name']} at {math_server['base_url']}")

        # Create math tools
        math_tools = create_mcp_tools(math_server)
        print(f"✅ Created {len(math_tools)} math tools")

        # Create agent with math tools
        agent = LlmAgent(
            model=LiteLlm(model="openrouter/qwen/qwen3-coder"),
            name="math_docker_agent",
            instruction=(
                "You are a mathematical assistant running inside a Docker container. "
                "Use the available math tools to perform calculations on the data provided. "
                "When given numbers in brackets like [1,2,3], use them directly as input to the tools. "
                "Be precise and show the calculation results clearly."
            ),
            tools=math_tools
        )

        # Create session
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="math_docker_test",
            user_id="docker_user",
            session_id="math_docker_session"
        )

        # Create runner
        runner = Runner(
            agent=agent,
            app_name="math_docker_test",
            session_service=session_service
        )

        # Test math calculations
        test_queries = [
            "Calculate the sum of [100, 250, 175, 300, 425]",
            "Find the average of [1000, 1500, 2000, 2500, 3000]",
            "Calculate the square root of [144, 225, 400]",
            "Find the maximum value in [85, 92, 78, 96, 88]"
        ]

        success_count = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Docker Test {i}: {query} ---")

            try:
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
                    print(f"✅ Agent response: {response[:150]}...")
                    success_count += 1
                else:
                    print("❌ No response received")

            except Exception as e:
                print(f"❌ Test {i} failed: {e}")

        print(f"\n=== Docker Math Agent Results: {success_count}/{len(test_queries)} successful ===")

        if success_count == len(test_queries):
            print("\n🎉 ALL DOCKER TESTS PASSED!")
            print("💼 Math MCP Server is fully functional inside Docker!")
            return True
        else:
            print(f"\n⚠️  {len(test_queries) - success_count} tests failed in Docker environment.")
            return False

    except Exception as e:
        print(f"❌ Docker agent test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Docker Math Agent MCP Test")
    print("=" * 40)

    try:
        result = asyncio.run(test_math_agent_in_docker())
        return result
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    """Run the Docker math agent test."""
    try:
        result = main()
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
