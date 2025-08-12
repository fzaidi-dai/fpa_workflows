#!/usr/bin/env python3
"""
Enhanced Math and Aggregation MCP Server with Range Support

This server provides mathematical and aggregation operations through the Model Context Protocol (MCP)
with support for range specifications compatible with Google Sheets A1 notation.
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

# Import consolidated sheets-compatible functions
from .sheets_compatible_functions import SheetsCompatibleFunctions

# Initialize the sheets functions
sheets_funcs = SheetsCompatibleFunctions()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Enhanced Math and Aggregation Server")
app = FastAPI(title="Enhanced Math and Aggregation MCP Server", version="2.0.0")

def validate_file_path(file_path: str) -> str:
    """Validate and resolve file path for data operations."""
    # If it's a relative path, prefix with /mcp-data/
    if not file_path.startswith('/'):
        return f"/mcp-data/{file_path}"
    return file_path

def parse_values_input(values_str: str) -> str:
    """
    Parse string input into file path for sheets functions.
    
    Args:
        values_str: Input string that should be a file path
        
    Returns:
        Validated file path
        
    Raises:
        ValueError: If input format is invalid
    """
    # Handle file paths
    if values_str.endswith(('.csv', '.parquet')) or '/' in values_str:
        return validate_file_path(values_str)
    
    # For backward compatibility, try to create a temporary file if it's JSON data
    try:
        parsed = json.loads(values_str)
        if isinstance(parsed, list):
            # Create a temporary DataFrame and save as parquet
            import polars as pl
            import tempfile
            import os
            
            df = pl.DataFrame({"values": parsed})
            temp_dir = "/mcp-data/scratch_pad"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = f"{temp_dir}/temp_data_{hash(values_str)}.parquet"
            df.write_parquet(temp_path)
            return temp_path
        else:
            # Single number - create temp file
            import polars as pl
            import tempfile
            import os
            
            df = pl.DataFrame({"values": [parsed]})
            temp_dir = "/mcp-data/scratch_pad"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = f"{temp_dir}/temp_data_{hash(values_str)}.parquet"
            df.write_parquet(temp_path)
            return temp_path
    except json.JSONDecodeError:
        # Try to parse as single number
        try:
            import polars as pl
            import os
            
            value = float(values_str)
            df = pl.DataFrame({"values": [value]})
            temp_dir = "/mcp-data/scratch_pad"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = f"{temp_dir}/temp_data_{hash(values_str)}.parquet"
            df.write_parquet(temp_path)
            return temp_path
        except ValueError:
            raise ValueError(f"Invalid values format: {values_str}. Expected: file path, JSON array, or number")

def parse_range_spec(range_str: Optional[str]) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Parse range specification string.
    
    Args:
        range_str: Range string in A1 notation or JSON format
        
    Returns:
        Parsed range specification
    """
    if not range_str:
        return None
    
    # Check if it's JSON format
    if range_str.startswith('{'):
        try:
            return json.loads(range_str)
        except json.JSONDecodeError:
            pass
    
    # Otherwise treat as A1 notation
    return range_str

# Enhanced tool wrapper functions with range support

@mcp.tool()
async def sum_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation (e.g., 'A1:C10', 'B:B')"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """
    Calculate sum with optional range specification.
    
    Examples:
        - sum_with_range("data.csv", "A1:A100") - Sum first 100 rows of column A
        - sum_with_range("data.parquet", "B:B", "revenue") - Sum all values in revenue column
    """
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.SUM(data_path, range_spec=range_spec, column=column)
        formula = f"=SUM({range_spec or 'A:A'})"
        return {
            "success": True, 
            "result": str(result), 
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sum_with_range: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def average_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation (e.g., 'A1:C10', 'B:B')"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """
    Calculate average with optional range specification.
    
    Examples:
        - average_with_range("data.csv", "A1:A100") - Average first 100 rows of column A
        - average_with_range("data.parquet", "C2:C50", "price") - Average prices in specified range
    """
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.AVERAGE(data_path, range_spec=range_spec, column=column)
        formula = f"=AVERAGE({range_spec or 'A:A'})"
        return {
            "success": True, 
            "result": str(result), 
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in average_with_range: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def min_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation (e.g., 'A1:C10', 'B:B')"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """
    Find minimum value with optional range specification.
    
    Examples:
        - min_with_range("data.csv", "D:D") - Minimum value in column D
        - min_with_range("data.parquet", "A10:A20") - Minimum in specific row range
    """
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.MIN(data_path, range_spec=range_spec, column=column)
        formula = f"=MIN({range_spec or 'A:A'})"
        return {
            "success": True, 
            "result": str(result), 
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in min_with_range: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def max_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation (e.g., 'A1:C10', 'B:B')"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """
    Find maximum value with optional range specification.
    
    Examples:
        - max_with_range("data.csv", "E:E") - Maximum value in column E
        - max_with_range("data.parquet", "B5:B15") - Maximum in specific range
    """
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.MAX(data_path, range_spec=range_spec, column=column)
        formula = f"=MAX({range_spec or 'A:A'})"
        return {
            "success": True, 
            "result": str(result), 
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in max_with_range: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def median_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation (e.g., 'A1:C10', 'B:B')"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """
    Calculate median with optional range specification.
    
    Examples:
        - median_with_range("data.csv", "F:F") - Median of column F
        - median_with_range("data.parquet", "A1:E100") - Median of specified range
    """
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.MEDIAN(data_path, range_spec=range_spec, column=column)
        formula = f"=MEDIAN({range_spec or 'A:A'})"
        return {
            "success": True, 
            "result": str(result), 
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in median_with_range: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def percentile_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    percentile: float = Field(description="Percentile to calculate (0-100)"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation (e.g., 'A1:C10', 'B:B')"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """
    Calculate percentile with optional range specification.
    
    Examples:
        - percentile_with_range("data.csv", 75, "G:G") - 75th percentile of column G
        - percentile_with_range("data.parquet", 90, "B2:B100") - 90th percentile of range
    """
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.PERCENTILE(data_path, percentile, range_spec=range_spec, column=column)
        formula = f"=PERCENTILE({range_spec or 'A:A'}, {percentile/100})"
        return {
            "success": True, 
            "result": str(result), 
            "formula": formula,
            "percentile": percentile, 
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in percentile_with_range: {e}")
        return {"success": False, "error": str(e)}

# Additional enhanced math functions

@mcp.tool()
async def product_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """Calculate product with optional range specification."""
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.PRODUCT(data_path, range_spec=range_spec, column=column)
        formula = f"=PRODUCT({range_spec or 'A:A'})"
        return {
            "success": True, 
            "result": str(result), 
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in product_with_range: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def power_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    power: float = Field(description="Power to raise values to"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """Calculate power with optional range specification."""
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.POWER(data_path, power, range_spec=range_spec, column=column)
        formula = f"=ARRAYFORMULA(POWER({range_spec or 'A:A'}, {power}))"
        return {
            "success": True, 
            "result": result,  # This returns a list
            "formula": formula,
            "range": range_spec,
            "power": power
        }
    except Exception as e:
        logger.error(f"Error in power_with_range: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sqrt_with_range(
    values: str = Field(description="Input values: file path, JSON array, or number"),
    range_spec: Optional[str] = Field(None, description="Range specification in A1 notation"),
    column: Optional[str] = Field(None, description="Column name for aggregation")
) -> Dict[str, Any]:
    """Calculate square root with optional range specification."""
    try:
        data_path = parse_values_input(values)
        result = sheets_funcs.SQRT(data_path, range_spec=range_spec, column=column)
        formula = f"=ARRAYFORMULA(SQRT({range_spec or 'A:A'}))"
        return {
            "success": True, 
            "result": result,  # This returns a list
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sqrt_with_range: {e}")
        return {"success": False, "error": str(e)}

# Backward compatibility - keep original tool names without range support
@mcp.tool()
async def sum_tool(values: str) -> Dict[str, Any]:
    """Calculate sum (backward compatibility)."""
    return await sum_with_range(values)

@mcp.tool()
async def average_tool(values: str) -> Dict[str, Any]:
    """Calculate average (backward compatibility)."""
    return await average_with_range(values)

@mcp.tool()
async def min_tool(values: str) -> Dict[str, Any]:
    """Find minimum (backward compatibility)."""
    return await min_with_range(values)

@mcp.tool()
async def max_tool(values: str) -> Dict[str, Any]:
    """Find maximum (backward compatibility)."""
    return await max_with_range(values)

# Pydantic models for API
class MathRequest(BaseModel):
    values: str = Field(description="Input values")
    range_spec: Optional[str] = Field(None, description="Range specification")
    column: Optional[str] = Field(None, description="Column name")

class MathResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    range: Optional[str] = None

# FastAPI endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Enhanced Math and Aggregation MCP Server"}

@app.post("/sum", response_model=MathResponse)
async def api_sum(request: MathRequest):
    """Calculate sum via API."""
    result = await sum_with_range(request.values, request.range_spec, request.column)
    return MathResponse(**result)

@app.post("/average", response_model=MathResponse)
async def api_average(request: MathRequest):
    """Calculate average via API."""
    result = await average_with_range(request.values, request.range_spec, request.column)
    return MathResponse(**result)

@app.post("/min", response_model=MathResponse)
async def api_min(request: MathRequest):
    """Find minimum via API."""
    result = await min_with_range(request.values, request.range_spec, request.column)
    return MathResponse(**result)

@app.post("/max", response_model=MathResponse)
async def api_max(request: MathRequest):
    """Find maximum via API."""
    result = await max_with_range(request.values, request.range_spec, request.column)
    return MathResponse(**result)

# Main execution
async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8002, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Enhanced Math and Aggregation MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8002")
    print("API Docs: http://localhost:8002/docs")
    asyncio.run(main())