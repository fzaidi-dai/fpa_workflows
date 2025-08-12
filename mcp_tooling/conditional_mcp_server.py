#!/usr/bin/env python3
"""
Conditional Functions MCP Server

This MCP server provides Google Sheets conditional functions through the Model Context Protocol.
Functions include SUMIF, COUNTIF, AVERAGEIF, SUMIFS with full compatibility to Google Sheets.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
mcp = FastMCP("Conditional Functions Server")
app = FastAPI(title="Conditional Functions MCP Server", version="1.0.0")

def validate_file_path(file_path: str) -> str:
    """Validate and resolve file path for data operations."""
    # If it's a relative path, prefix with /mcp-data/
    if not file_path.startswith('/'):
        return f"/mcp-data/{file_path}"
    return file_path

# ================== CONDITIONAL AGGREGATION FUNCTIONS ==================

@mcp.tool()
async def sumif_tool(
    range_path: str = Field(description="Path to range data file (CSV or Parquet)"),
    criteria: Union[str, float, int] = Field(description="Criteria for summing (e.g., '>100', '=Active', 'Apple')"),
    sum_range_path: Optional[str] = Field(None, description="Optional path to sum range data file")
) -> Dict[str, Any]:
    """
    SUMIF function matching Google Sheets =SUMIF(range, criteria, [sum_range]).
    
    Examples:
        sumif_tool("sales.csv", ">100") - Sum values greater than 100
        sumif_tool("categories.csv", "Electronics", "amounts.csv") - Sum amounts where category is Electronics
        sumif_tool("status.csv", "=Active") - Sum where status equals 'Active'
    """
    try:
        range_data = validate_file_path(range_path)
        sum_range_data = validate_file_path(sum_range_path) if sum_range_path else None
        
        result = sheets_funcs.SUMIF(range_data, criteria, sum_range_data)
        
        if sum_range_path:
            formula = f"=SUMIF(A:A, \"{criteria}\", B:B)"
        else:
            formula = f"=SUMIF(A:A, \"{criteria}\")"
            
        return {
            "success": True,
            "result": float(result),
            "formula": formula,
            "criteria": criteria,
            "range_path": range_path,
            "sum_range_path": sum_range_path
        }
    except Exception as e:
        logger.error(f"Error in sumif_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def countif_tool(
    range_path: str = Field(description="Path to range data file (CSV or Parquet)"),
    criteria: Union[str, float, int] = Field(description="Criteria for counting (e.g., '>=80', 'Active', '*Apple*')")
) -> Dict[str, Any]:
    """
    COUNTIF function matching Google Sheets =COUNTIF(range, criteria).
    
    Examples:
        countif_tool("status.csv", "Active") - Count cells equal to 'Active'
        countif_tool("scores.csv", ">=80") - Count scores >= 80
        countif_tool("products.csv", "*Apple*") - Count cells containing 'Apple'
    """
    try:
        range_data = validate_file_path(range_path)
        result = sheets_funcs.COUNTIF(range_data, criteria)
        formula = f"=COUNTIF(A:A, \"{criteria}\")"
        
        return {
            "success": True,
            "result": int(result),
            "formula": formula,
            "criteria": criteria,
            "range_path": range_path
        }
    except Exception as e:
        logger.error(f"Error in countif_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def averageif_tool(
    range_path: str = Field(description="Path to range data file (CSV or Parquet)"),
    criteria: Union[str, float, int] = Field(description="Criteria for averaging"),
    average_range_path: Optional[str] = Field(None, description="Optional path to average range data file")
) -> Dict[str, Any]:
    """
    AVERAGEIF function matching Google Sheets =AVERAGEIF(range, criteria, [average_range]).
    
    Examples:
        averageif_tool("departments.csv", "Sales", "salaries.csv") - Average salary for Sales department
        averageif_tool("grades.csv", ">=70") - Average of grades >= 70
    """
    try:
        range_data = validate_file_path(range_path)
        average_range_data = validate_file_path(average_range_path) if average_range_path else None
        
        result = sheets_funcs.AVERAGEIF(range_data, criteria, average_range_data)
        
        if average_range_path:
            formula = f"=AVERAGEIF(A:A, \"{criteria}\", B:B)"
        else:
            formula = f"=AVERAGEIF(A:A, \"{criteria}\")"
            
        return {
            "success": True,
            "result": float(result),
            "formula": formula,
            "criteria": criteria,
            "range_path": range_path,
            "average_range_path": average_range_path
        }
    except Exception as e:
        logger.error(f"Error in averageif_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sumifs_tool(
    sum_range_path: str = Field(description="Path to sum range data file"),
    criteria_pairs: List[str] = Field(description="Alternating list of criteria_range_path and criteria (e.g., ['range1.csv', '>100', 'range2.csv', 'Active'])")
) -> Dict[str, Any]:
    """
    SUMIFS function matching Google Sheets =SUMIFS(sum_range, criteria_range1, criteria1, [criteria_range2, criteria2, ...]).
    
    Examples:
        sumifs_tool("amounts.csv", ["status.csv", "Active", "category.csv", "Electronics"])
        - Sum amounts where status is Active AND category is Electronics
    """
    try:
        sum_range_data = validate_file_path(sum_range_path)
        
        # Process criteria pairs
        processed_pairs = []
        for i in range(0, len(criteria_pairs), 2):
            if i + 1 < len(criteria_pairs):
                range_path = validate_file_path(criteria_pairs[i])
                criteria = criteria_pairs[i + 1]
                processed_pairs.extend([range_path, criteria])
        
        result = sheets_funcs.SUMIFS(sum_range_data, *processed_pairs)
        
        # Build formula
        formula_parts = ["=SUMIFS(A:A"]
        for i in range(0, len(criteria_pairs), 2):
            if i + 1 < len(criteria_pairs):
                criteria = criteria_pairs[i + 1]
                formula_parts.append(f", B:B, \"{criteria}\"")
        formula = "".join(formula_parts) + ")"
        
        return {
            "success": True,
            "result": float(result),
            "formula": formula,
            "sum_range_path": sum_range_path,
            "criteria_pairs": criteria_pairs
        }
    except Exception as e:
        logger.error(f"Error in sumifs_tool: {e}")
        return {"success": False, "error": str(e)}

# ================== API MODELS ==================

class SumIfRequest(BaseModel):
    range_path: str
    criteria: Union[str, float, int]
    sum_range_path: Optional[str] = None

class CountIfRequest(BaseModel):
    range_path: str
    criteria: Union[str, float, int]

class AverageIfRequest(BaseModel):
    range_path: str
    criteria: Union[str, float, int]
    average_range_path: Optional[str] = None

class SumIfsRequest(BaseModel):
    sum_range_path: str
    criteria_pairs: List[str]

class ConditionalResponse(BaseModel):
    success: bool
    result: Optional[Union[int, float]] = None
    formula: Optional[str] = None
    error: Optional[str] = None

# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Conditional Functions MCP Server"}

@app.post("/sumif", response_model=ConditionalResponse)
async def api_sumif(request: SumIfRequest):
    """SUMIF function via API."""
    result = await sumif_tool(
        request.range_path,
        request.criteria,
        request.sum_range_path
    )
    return ConditionalResponse(**result)

@app.post("/countif", response_model=ConditionalResponse)
async def api_countif(request: CountIfRequest):
    """COUNTIF function via API."""
    result = await countif_tool(request.range_path, request.criteria)
    return ConditionalResponse(**result)

@app.post("/averageif", response_model=ConditionalResponse)
async def api_averageif(request: AverageIfRequest):
    """AVERAGEIF function via API."""
    result = await averageif_tool(
        request.range_path,
        request.criteria,
        request.average_range_path
    )
    return ConditionalResponse(**result)

@app.post("/sumifs", response_model=ConditionalResponse)
async def api_sumifs(request: SumIfsRequest):
    """SUMIFS function via API."""
    result = await sumifs_tool(request.sum_range_path, request.criteria_pairs)
    return ConditionalResponse(**result)

@app.get("/functions")
async def list_functions():
    """List all available conditional functions."""
    functions = ["SUMIF", "COUNTIF", "AVERAGEIF", "SUMIFS"]
    return {"functions": functions, "count": len(functions)}

@app.get("/criteria-examples")
async def criteria_examples():
    """Get examples of valid criteria formats."""
    return {
        "numeric_criteria": [
            ">100", ">=50", "<200", "<=75", "=50", "<>100"
        ],
        "text_criteria": [
            "Active", "=Complete", "<>Inactive", "*Apple*", "?ohn"
        ],
        "wildcards": {
            "*": "Matches any sequence of characters",
            "?": "Matches any single character"
        },
        "examples": {
            "sumif": "sumif_tool('sales.csv', '>1000')",
            "countif": "countif_tool('status.csv', 'Active')",
            "averageif": "averageif_tool('dept.csv', 'Sales', 'salaries.csv')",
            "sumifs": "sumifs_tool('amounts.csv', ['status.csv', 'Active', 'region.csv', 'North'])"
        }
    }

# ================== MAIN EXECUTION ==================

async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8006, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Conditional Functions MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8006")
    print("API Docs: http://localhost:8006/docs")
    print("\nAvailable functions:")
    print("- SUMIF: Sum values based on criteria")
    print("- COUNTIF: Count cells based on criteria")
    print("- AVERAGEIF: Average values based on criteria")
    print("- SUMIFS: Sum with multiple criteria")
    print("\nCriteria examples: '>100', '=Active', '*Apple*', '>=80'")
    asyncio.run(main())