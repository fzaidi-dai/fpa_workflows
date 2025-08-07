"""
MCP Tooling Package

This package provides tools for working with Model Context Protocol (MCP) servers
and integrating them with Google ADK agents.
"""

# Expose key modules and functions
from .fpa_mcp_tools import (
    MCPClient,
    create_filesystem_client,
    create_mcp_client,
    MCPConfig
)

from .adk_integrations import (
    create_filesystem_tools
)

__all__ = [
    'MCPClient',
    'create_filesystem_client',
    'create_mcp_client',
    'MCPConfig',
    'create_filesystem_tools'
]
