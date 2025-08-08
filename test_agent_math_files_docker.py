#!/usr/bin/env python3
"""
Test Agent with Math MCP Tools using Files inside Docker Container

This test demonstrates the agent using math tools with actual data files
through the MCP server running inside the Docker environment.
"""

import asyncio
import polars as pl
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Import MCP tools
from mcp_tooling.mcp_tools_adk import MCPConfig, create_mcp_tools

def create_test_data_files():
    """Create test data files for the agent to work with."""
    print("📊 Creating test data files for agent...")

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create financial revenue data
    revenue_data = pl.DataFrame({
        "revenue": [125000.50, 132000.75, 128500.25, 145000.00, 139500.80, 152000.30, 148750.90, 156200.40]
    })
    revenue_data.write_parquet(data_dir / "monthly_revenue.parquet")

    # Create expense data
    expense_data = pl.DataFrame({
        "expenses": [85000.25, 89500.50, 87200.75, 92000.00, 88750.30, 95500.80, 91200.45, 97800.60]
    })
    expense_data.write_parquet(data_dir / "monthly_expenses.parquet")

    # Create profit margins data
    margins_data = pl.DataFrame({
        "margin_percent": [15.5, 18.2, 16.8, 19.5, 17.3, 20.1, 18.9, 21.2]
    })
    margins_data.write_parquet(data_dir / "profit_margins.parquet")

    print("✅ Created test data files:")
    print("   - data/monthly_revenue.parquet (8 months of revenue)")
    print("   - data/monthly_expenses.parquet (8 months of expenses)")
    print("   - data/profit_margins.parquet (8 months of margins)")

    return True

async def test_agent_with_files_in_docker():
    """Test agent using math tools with actual files inside Docker environment."""
    print("🐳 Testing Math Agent with Files inside Docker Container")
    print("=" * 70)

    try:
        # Create test data files
        if not create_test_data_files():
            print("❌ Failed to create test data files")
            return False

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
            name="math_files_docker_agent",
            instruction=(
                "You are a financial analysis assistant running inside a Docker container. "
                "Use the available math tools to perform calculations on data files. "
                "When given file paths, use them directly with the math tools. "
                "Provide clear financial insights based on the calculations."
            ),
            tools=math_tools
        )

        # Create session
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="math_files_docker_test",
            user_id="docker_user",
            session_id="math_files_docker_session"
        )

        # Create runner
        runner = Runner(
            agent=agent,
            app_name="math_files_docker_test",
            session_service=session_service
        )

        # Test file-based calculations
        test_queries = [
            "Calculate the total revenue from the file data/monthly_revenue.parquet",
            "Find the average monthly expenses from data/monthly_expenses.parquet",
            "What is the maximum profit margin in data/profit_margins.parquet?",
            "Calculate the minimum revenue from data/monthly_revenue.parquet",
            "Find the sum of all expenses in data/monthly_expenses.parquet and save the result to scratch_pad/analysis/total_expenses.parquet"
        ]

        success_count = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Docker File Test {i}: {query} ---")

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
                    print(f"✅ Agent response: {response[:200]}...")
                    success_count += 1
                else:
                    print("❌ No response received")

            except Exception as e:
                print(f"❌ Test {i} failed: {e}")

        print(f"\n=== Docker File Math Agent Results: {success_count}/{len(test_queries)} successful ===")

        # Verify output files were created
        output_file = Path("scratch_pad/analysis/total_expenses.parquet")
        if output_file.exists():
            print("✅ Output file verification: total_expenses.parquet created successfully")
            try:
                df = pl.read_parquet(output_file)
                print(f"   File contains {df.height} rows")
            except Exception as e:
                print(f"   Warning: Could not read output file - {e}")
        else:
            print("❌ Output file verification: total_expenses.parquet not found")

        if success_count == len(test_queries):
            print("\n🎉 ALL DOCKER FILE TESTS PASSED!")
            print("💼 Math MCP Server with file operations is fully functional inside Docker!")
            return True
        else:
            print(f"\n⚠️  {len(test_queries) - success_count} file tests failed in Docker environment.")
            return False

    except Exception as e:
        print(f"❌ Docker file agent test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Docker Math Agent File Operations Test")
    print("=" * 50)

    try:
        result = asyncio.run(test_agent_with_files_in_docker())
        return result
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    """Run the Docker math agent file test."""
    try:
        result = main()
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
