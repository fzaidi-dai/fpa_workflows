#!/usr/bin/env python3
"""
Test script for the read-only agent (root_agent2)

This script tests that the filtered agent only has access to read_file tool
and cannot write files.
"""

import asyncio
import os
from dotenv import load_dotenv
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Load environment variables
load_dotenv()

async def test_read_only_agent():
    """Test the read-only agent functionality."""

    # Import the read-only agent
    from agents.test_agent.agent import root_agent2

    print("Testing read-only agent (root_agent2)...")
    print(f"Agent has {len(root_agent2.tools)} tools available")

    # Print available tools
    for i, tool in enumerate(root_agent2.tools):
        print(f"  Tool {i+1}: {tool.func.__name__}")

    try:
        # Create session service
        session_service = InMemorySessionService()

        # Create session
        session = await session_service.create_session(
            app_name="test_read_only_app",
            user_id="test_user",
            session_id="test_session_readonly"
        )

        # Create runner with read-only agent
        runner = Runner(
            agent=root_agent2,
            app_name="test_read_only_app",
            session_service=session_service
        )

        # Test 1: Try to write a file (should fail or be refused)
        print("\n=== Test 1: Attempting to write a file ===")
        write_query = "Write 'This should not work' to a file called readonly_test.txt"
        print(f"Query: {write_query}")

        content = types.Content(
            role='user',
            parts=[types.Part(text=write_query)]
        )

        events_async = runner.run_async(
            session_id=session.id,
            user_id=session.user_id,
            new_message=content
        )

        async for event in events_async:
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"Response: {part.text.strip()}")
                    elif hasattr(part, 'function_call'):
                        print(f"Function call: {part.function_call.name} with args: {part.function_call.args}")

        # Test 2: Try to read the file we created earlier (should work)
        print("\n=== Test 2: Attempting to read an existing file ===")
        read_query = "Read the contents of the file test.txt"
        print(f"Query: {read_query}")

        content = types.Content(
            role='user',
            parts=[types.Part(text=read_query)]
        )

        events_async = runner.run_async(
            session_id=session.id,
            user_id=session.user_id,
            new_message=content
        )

        async for event in events_async:
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"Response: {part.text.strip()}")
                    elif hasattr(part, 'function_call'):
                        print(f"Function call: {part.function_call.name} with args: {part.function_call.args}")
                    elif hasattr(part, 'function_response'):
                        print(f"Function response: {part.function_response.response}")

    except Exception as e:
        print(f"Error during read-only agent test: {e}")
        raise

if __name__ == "__main__":
    print("Testing read-only agent functionality...")
    try:
        asyncio.run(test_read_only_agent())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        print("Read-only agent test completed.")
