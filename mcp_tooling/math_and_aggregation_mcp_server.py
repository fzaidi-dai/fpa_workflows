#!/usr/bin/env python3
"""
Math and Aggregation MCP Server using FastMCP

This server provides mathematical and aggregation operations through the Model Context Protocol (MCP).
It includes tools for basic math, statistical calculations, and financial computations with Decimal precision.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import uvicorn

# Import standalone math functions to avoid circular imports
from standalone_math_functions import (
    SUM, AVERAGE, MIN, MAX, PRODUCT, MEDIAN, MODE, PERCENTILE,
    POWER, SQRT, EXP, LN, LOG, ABS, SIGN, MOD, ROUND, ROUNDUP, ROUNDDOWN,
    WEIGHTED_AVERAGE, GEOMETRIC_MEAN, HARMONIC_MEAN, CUMSUM, CUMPROD, VARIANCE_WEIGHTED
)

# Simple context classes to avoid dependencies
class SimpleFinnDeps:
    def __init__(self, thread_dir: Path, workspace_dir: Path):
        self.thread_dir = thread_dir
        self.workspace_dir = workspace_dir

class SimpleRunContext:
    def __init__(self, deps: SimpleFinnDeps):
        self.deps = deps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Math and Aggregation Server")
app = FastAPI(title="Math and Aggregation MCP Server", version="1.0.0")

def load_config():
    """Load server configuration from math_and_aggregation.json."""
    config_path = Path(__file__).parent / "config" / "math_and_aggregation.json"
    with open(config_path, 'r') as f:
        return json.load(f)

# Load configuration
CONFIG = load_config()

def create_context() -> SimpleRunContext:
    """Create a RunContext for function execution."""
    thread_dir = Path("/mcp-data/scratch_pad").resolve()
    workspace_dir = Path("/mcp-data").resolve()
    finn_deps = SimpleFinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    return SimpleRunContext(deps=finn_deps)

def parse_values_input(values_str: str) -> Union[List, str, Path]:
    """
    Parse string input into appropriate format for our functions.

    Args:
        values_str: Input string that can be a file path, JSON array, or single number

    Returns:
        Parsed values in appropriate format

    Raises:
        ValueError: If input format is invalid
    """
    # Handle file paths
    if values_str.endswith(('.csv', '.parquet')) or '/' in values_str:
        # If it's a relative path, prefix with /mcp-data/
        if not values_str.startswith('/'):
            return f"/mcp-data/{values_str}"
        else:
            # Already absolute path, use as-is
            return values_str

    # Try to parse as JSON array
    try:
        parsed = json.loads(values_str)
        if isinstance(parsed, list):
            return parsed
        else:
            return [parsed]  # Single number
    except json.JSONDecodeError:
        # Try to parse as single number
        try:
            return [float(values_str)]
        except ValueError:
            raise ValueError(f"Invalid values format: {values_str}. Expected: file path, JSON array, or number")

def parse_numeric_input(value_str: str) -> Union[float, int, Decimal]:
    """Parse string input into numeric value."""
    try:
        # Try integer first
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value: {value_str}")

# Tool wrapper functions that handle ctx parameter internally

def sum_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for SUM function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = SUM(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def average_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for AVERAGE function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = AVERAGE(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def min_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for MIN function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = MIN(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def max_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for MAX function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = MAX(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def product_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for PRODUCT function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = PRODUCT(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def median_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for MEDIAN function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = MEDIAN(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def mode_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for MODE function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = MODE(ctx, parsed_values)
        # Handle both single mode and multiple modes
        if isinstance(result, list):
            result_str = json.dumps([str(r) for r in result])
        else:
            result_str = str(result)
        return {"success": True, "result": result_str}
    except Exception as e:
        return {"success": False, "error": str(e)}

def percentile_tool(values: str, percentile_value: float) -> Dict[str, Any]:
    """MCP wrapper for PERCENTILE function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = PERCENTILE(ctx, parsed_values, percentile_value=percentile_value)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def power_tool(values: str, power: float, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for POWER function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = POWER(ctx, parsed_values, power=power, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def sqrt_tool(values: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for SQRT function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = SQRT(ctx, parsed_values, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def exp_tool(values: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for EXP function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = EXP(ctx, parsed_values, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def ln_tool(values: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for LN function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = LN(ctx, parsed_values, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def log_tool(values: str, base: Optional[float] = None, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for LOG function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = LOG(ctx, parsed_values, base=base, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def abs_tool(values: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for ABS function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = ABS(ctx, parsed_values, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def sign_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for SIGN function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = SIGN(ctx, parsed_values)
        result_str = json.dumps(result)
        return {"success": True, "result": result_str}
    except Exception as e:
        return {"success": False, "error": str(e)}

def mod_tool(dividends: str, divisors: str) -> Dict[str, Any]:
    """MCP wrapper for MOD function."""
    ctx = create_context()
    try:
        parsed_dividends = parse_values_input(dividends)
        parsed_divisors = parse_values_input(divisors)
        result = MOD(ctx, parsed_dividends, divisors=parsed_divisors)
        result_str = json.dumps([str(r) for r in result])
        return {"success": True, "result": result_str}
    except Exception as e:
        return {"success": False, "error": str(e)}

def round_tool(values: str, num_digits: int) -> Dict[str, Any]:
    """MCP wrapper for ROUND function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = ROUND(ctx, parsed_values, num_digits=num_digits)
        result_str = json.dumps([str(r) for r in result])
        return {"success": True, "result": result_str}
    except Exception as e:
        return {"success": False, "error": str(e)}

def roundup_tool(values: str, num_digits: int) -> Dict[str, Any]:
    """MCP wrapper for ROUNDUP function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = ROUNDUP(ctx, parsed_values, num_digits=num_digits)
        result_str = json.dumps([str(r) for r in result])
        return {"success": True, "result": result_str}
    except Exception as e:
        return {"success": False, "error": str(e)}

def rounddown_tool(values: str, num_digits: int) -> Dict[str, Any]:
    """MCP wrapper for ROUNDDOWN function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = ROUNDDOWN(ctx, parsed_values, num_digits=num_digits)
        result_str = json.dumps([str(r) for r in result])
        return {"success": True, "result": result_str}
    except Exception as e:
        return {"success": False, "error": str(e)}

def weighted_average_tool(values: str, weights: str) -> Dict[str, Any]:
    """MCP wrapper for WEIGHTED_AVERAGE function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        parsed_weights = parse_values_input(weights)
        result = WEIGHTED_AVERAGE(ctx, parsed_values, weights=parsed_weights)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def geometric_mean_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for GEOMETRIC_MEAN function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = GEOMETRIC_MEAN(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def harmonic_mean_tool(values: str) -> Dict[str, Any]:
    """MCP wrapper for HARMONIC_MEAN function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = HARMONIC_MEAN(ctx, parsed_values)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def cumsum_tool(values: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for CUMSUM function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = CUMSUM(ctx, parsed_values, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def cumprod_tool(values: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """MCP wrapper for CUMPROD function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        result = CUMPROD(ctx, parsed_values, output_filename=output_filename)

        # Handle both list results and file path results
        if isinstance(result, Path):
            return {"success": True, "result": str(result), "type": "file_path"}
        else:
            result_str = json.dumps([str(r) for r in result])
            return {"success": True, "result": result_str, "type": "array"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def variance_weighted_tool(values: str, weights: str) -> Dict[str, Any]:
    """MCP wrapper for VARIANCE_WEIGHTED function."""
    ctx = create_context()
    try:
        parsed_values = parse_values_input(values)
        parsed_weights = parse_values_input(weights)
        result = VARIANCE_WEIGHTED(ctx, parsed_values, weights=parsed_weights)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Tool name to function mapping
TOOL_FUNCTIONS = {
    "SUM": sum_tool,
    "AVERAGE": average_tool,
    "MIN": min_tool,
    "MAX": max_tool,
    "PRODUCT": product_tool,
    "MEDIAN": median_tool,
    "MODE": mode_tool,
    "PERCENTILE": percentile_tool,
    "POWER": power_tool,
    "SQRT": sqrt_tool,
    "EXP": exp_tool,
    "LN": ln_tool,
    "LOG": log_tool,
    "ABS": abs_tool,
    "SIGN": sign_tool,
    "MOD": mod_tool,
    "ROUND": round_tool,
    "ROUNDUP": roundup_tool,
    "ROUNDDOWN": rounddown_tool,
    "WEIGHTED_AVERAGE": weighted_average_tool,
    "GEOMETRIC_MEAN": geometric_mean_tool,
    "HARMONIC_MEAN": harmonic_mean_tool,
    "CUMSUM": cumsum_tool,
    "CUMPROD": cumprod_tool,
    "VARIANCE_WEIGHTED": variance_weighted_tool,
}

# FastAPI HTTP endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Math and Aggregation MCP Server",
        "version": "1.0.0",
        "available_tools": list(TOOL_FUNCTIONS.keys())
    }

@app.post("/math_mcp")
async def math_mcp_endpoint(request_data: dict):
    """Math and Aggregation MCP tool endpoint."""
    try:
        method = request_data.get("method")
        params = request_data.get("params", {})

        if method == "tools/list":
            # Return tools configuration from JSON file
            return {
                "tools": CONFIG.get("tools", [])
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name not in TOOL_FUNCTIONS:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

            # Call the appropriate tool function
            tool_function = TOOL_FUNCTIONS[tool_name]

            try:
                # Extract arguments and call function
                result = tool_function(**arguments)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            except TypeError as e:
                # Handle missing or invalid arguments
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "success": False,
                                "error": f"Invalid arguments for {tool_name}: {str(e)}"
                            }, indent=2)
                        }
                    ]
                }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    except Exception as e:
        logger.error(f"Error handling MCP request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for MCP communication."""
    return {"message": "Math MCP SSE endpoint - use MCP client to connect"}

def setup_data_directories():
    """Create the necessary data directories if they don't exist."""
    directories = ["/mcp-data", "/mcp-data/scratch_pad", "/mcp-data/scratch_pad/analysis"]
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Math and Aggregation MCP Server...")

    # Setup data directories
    setup_data_directories()

    # Log available tools
    logger.info("Available tools:")
    for tool_name in TOOL_FUNCTIONS.keys():
        logger.info(f"  - {tool_name}")

    # Run the FastAPI server with uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=int(os.getenv("MATH_SERVER_PORT", 3002)),
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
