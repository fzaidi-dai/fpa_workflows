# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPA Agents is a Financial Planning and Analysis AI-powered workflow system with 200+ Excel-like functions across 19 categories. It combines Google ADK agents with MCP (Model Context Protocol) for tool discovery and execution.

## Key Commands

### Development
```bash
# Install dependencies
uv sync

# Run the agent
uv run python agents/test_agent.py

# Start web interface
uv run adk web

# Run specific tests
uv run pytest tests/test_basic_math_functions.py -v
uv run pytest tests/test_agent_math_docker.py -k "test_name"

# Docker development
./docker-dev.sh build
./docker-dev.sh up
./docker-dev.sh logs
```

### Production
```bash
./deploy-prod.sh build
./deploy-prod.sh start
./deploy-prod.sh health
./deploy-prod.sh backup
```

## Architecture

### Core Components
1. **`/tools/`**: 19 categories of financial/mathematical functions (SUM, VLOOKUP, NPV, etc.)
2. **`/agents/`**: Google ADK-based AI agents using LiteLLM with OpenRouter
3. **`/mcp_tooling/`**: MCP servers for filesystem operations and math computations
4. **`/data/`**: Persistent data storage (CSV, Parquet)
5. **`/scratch_pad/`**: Temporary analysis workspace

### Key Technologies
- **Python 3.13** with `uv` package manager
- **Polars** for data processing (not pandas)
- **Google ADK** for agent framework
- **FastMCP** for MCP server implementation
- **FastAPI** for exposing MCP servers as web service endpoints in Docker
- **LiteLLM** with OpenRouter's free Qwen model

### Tool Categories
Tools are organized in `/tools/` with each category having its own module:
- basic_math_aggregation
- conditional_logic
- lookup_and_reference
- date_and_time_functions
- statistical_and_trend_analysis
- data_transformation_and_pivoting
- financial_and_calendar_operations
- data_validation_and_quality

Each tool follows this pattern:
- Returns FinnOutput with result, data, errors, and metadata
- Uses Polars DataFrames for data operations
- Includes comprehensive error handling
- Has associated tests in `/tests/`

### MCP Integration Pattern
The project uses MCP for tool discovery and execution:
- **Filesystem MCP**: Read/write operations via `mcp_tooling/filesystem_mcp.py`
- **Math MCP**: Computational server via `mcp_tooling/math_mcp_server.py`
- **Generic Client**: Framework-agnostic tool execution via `mcp_tooling/generic_mcp_client.py`

### Testing Approach
- Unit tests for individual functions in `/tests/test_*.py`
- Docker integration tests in `test_agent_math_docker.py`
- MCP server tests in `test_math_mcp_*.py`
- Use `FinnDeps` and `RunContext` for context-aware testing

## Development Guidelines

### When Adding New Tools
1. Add to appropriate category in `/tools/`
2. Follow existing function signature patterns
3. Return FinnOutput with proper error handling
4. Add comprehensive tests
5. Update documentation in `/docs/finn_tools_*.md`

### When Working with Data
- Use Polars, not pandas
- Data files go in `/data/` (persistent) or `/scratch_pad/` (temporary)
- Support both CSV and Parquet formats
- Handle missing data gracefully

### When Modifying Agents
- Agents use Google ADK framework
- Configuration in `agents/test_agent.py`
- MCP connections configured in agent initialization
- Use OpenRouter's free Qwen model by default

### Docker Development
- Development uses `docker-compose.dev.yml`
- Production uses `docker-compose.prod.yml`
- Resource limits enforced in production
- Non-root user execution for security

## Important Notes

- This is a tool-centric system where AI agents orchestrate 200+ financial functions
- Functions mirror Excel/Google Sheets syntax for familiarity
- Production deployment includes SSL, Nginx, and security hardening
- The project targets enterprise financial analysis workflows
- All data processing uses Polars for performance
- MCP provides clean separation between agents and tools
- Always use uv run and other uv commands in this project to manage packages, run comands and other similare tasks instead of directly using Python or pip etc.
