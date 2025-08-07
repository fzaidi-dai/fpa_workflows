"""
Test Agent Package

This package contains test agents that demonstrate:
- LiteLLM integration with OpenRouter
- MCP toolset connection for file operations
- Interactive functionality through ADK web interface
- Both full-access and read-only agents
"""

from .agent import root_agent, root_agent2

__all__ = ["root_agent", "root_agent2"]
