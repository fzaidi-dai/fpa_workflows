#!/usr/bin/env python3
"""
Lookup Functions MCP Server

This MCP server provides Google Sheets lookup functions through the Model Context Protocol.
Functions include VLOOKUP, HLOOKUP, INDEX, MATCH with full compatibility to Google Sheets.
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
mcp = FastMCP("Lookup Functions Server")
app = FastAPI(title="Lookup Functions MCP Server", version="1.0.0")

def validate_file_path(file_path: str) -> str:
    """Validate and resolve file path for data operations."""
    # If it's a relative path, prefix with /mcp-data/
    if not file_path.startswith('/'):
        return f"/mcp-data/{file_path}"
    return file_path

# ================== LOOKUP FUNCTIONS ==================

@mcp.tool()
async def vlookup_tool(
    lookup_value: Union[str, float, int] = Field(description="Value to look up"),
    table_path: str = Field(description="Path to table data file (CSV or Parquet)"),
    col_index: int = Field(description="Column index to return (1-based)"),
    range_lookup: bool = Field(False, description="FALSE for exact match, TRUE for approximate")
) -> Dict[str, Any]:
    """
    VLOOKUP function matching Google Sheets =VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup]).
    
    Examples:
        vlookup_tool("Product1", "products.csv", 3, False) - Exact match lookup
        vlookup_tool(100, "prices.csv", 2, True) - Approximate match
    """
    try:
        table_data = validate_file_path(table_path)
        result = sheets_funcs.VLOOKUP(lookup_value, table_data, col_index, range_lookup)
        formula = f"=VLOOKUP(\"{lookup_value}\", A:Z, {col_index}, {str(range_lookup).upper()})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "lookup_value": lookup_value,
            "col_index": col_index,
            "range_lookup": range_lookup
        }
    except Exception as e:
        logger.error(f"Error in vlookup_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def hlookup_tool(
    lookup_value: Union[str, float, int] = Field(description="Value to look up"),
    table_path: str = Field(description="Path to table data file (CSV or Parquet)"),
    row_index: int = Field(description="Row index to return (1-based)"),
    range_lookup: bool = Field(False, description="FALSE for exact match, TRUE for approximate")
) -> Dict[str, Any]:
    """
    HLOOKUP function matching Google Sheets =HLOOKUP(lookup_value, table_array, row_index_num, [range_lookup]).
    
    Examples:
        hlookup_tool("Q1", "quarterly_data.csv", 2, False) - Horizontal lookup
    """
    try:
        table_data = validate_file_path(table_path)
        result = sheets_funcs.HLOOKUP(lookup_value, table_data, row_index, range_lookup)
        formula = f"=HLOOKUP(\"{lookup_value}\", 1:100, {row_index}, {str(range_lookup).upper()})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "lookup_value": lookup_value,
            "row_index": row_index,
            "range_lookup": range_lookup
        }
    except Exception as e:
        logger.error(f"Error in hlookup_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def index_tool(
    array_path: str = Field(description="Path to array data file (CSV or Parquet)"),
    row_num: int = Field(description="Row number (1-based)"),
    col_num: Optional[int] = Field(None, description="Column number (1-based, optional)")
) -> Dict[str, Any]:
    """
    INDEX function matching Google Sheets =INDEX(array, row_num, [column_num]).
    
    Examples:
        index_tool("data.csv", 5, 3) - Get value at row 5, column 3
        index_tool("data.csv", 10) - Get entire row 10
    """
    try:
        array_data = validate_file_path(array_path)
        result = sheets_funcs.INDEX(array_data, row_num, col_num)
        
        if col_num:
            formula = f"=INDEX(A:Z, {row_num}, {col_num})"
        else:
            formula = f"=INDEX(A:Z, {row_num})"
            
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "row": row_num,
            "column": col_num
        }
    except Exception as e:
        logger.error(f"Error in index_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def match_tool(
    lookup_value: Union[str, float, int] = Field(description="Value to find"),
    lookup_array: List[Any] = Field(description="Array to search in"),
    match_type: int = Field(0, description="1=less than, 0=exact, -1=greater than")
) -> Dict[str, Any]:
    """
    MATCH function matching Google Sheets =MATCH(lookup_value, lookup_array, [match_type]).
    
    Examples:
        match_tool("Apple", ["Apple", "Banana", "Cherry"], 0) - Find exact position
        match_tool(75, [50, 60, 70, 80, 90], 1) - Find largest value <= 75
    """
    try:
        result = sheets_funcs.MATCH(lookup_value, lookup_array, match_type)
        formula = f"=MATCH(\"{lookup_value}\", array, {match_type})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "position": result,
            "match_type": match_type
        }
    except Exception as e:
        logger.error(f"Error in match_tool: {e}")
        return {"success": False, "error": str(e)}

# ================== API MODELS ==================

class VLookupRequest(BaseModel):
    lookup_value: Union[str, float, int]
    table_path: str
    col_index: int
    range_lookup: bool = False

class HLookupRequest(BaseModel):
    lookup_value: Union[str, float, int]
    table_path: str
    row_index: int
    range_lookup: bool = False

class IndexRequest(BaseModel):
    array_path: str
    row_num: int
    col_num: Optional[int] = None

class MatchRequest(BaseModel):
    lookup_value: Union[str, float, int]
    lookup_array: List[Any]
    match_type: int = 0

class LookupResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    formula: Optional[str] = None
    error: Optional[str] = None

# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Lookup Functions MCP Server"}

@app.post("/vlookup", response_model=LookupResponse)
async def api_vlookup(request: VLookupRequest):
    """VLOOKUP function via API."""
    result = await vlookup_tool(
        request.lookup_value, 
        request.table_path, 
        request.col_index, 
        request.range_lookup
    )
    return LookupResponse(**result)

@app.post("/hlookup", response_model=LookupResponse)
async def api_hlookup(request: HLookupRequest):
    """HLOOKUP function via API."""
    result = await hlookup_tool(
        request.lookup_value, 
        request.table_path, 
        request.row_index, 
        request.range_lookup
    )
    return LookupResponse(**result)

@app.post("/index", response_model=LookupResponse)
async def api_index(request: IndexRequest):
    """INDEX function via API."""
    result = await index_tool(request.array_path, request.row_num, request.col_num)
    return LookupResponse(**result)

@app.post("/match", response_model=LookupResponse)
async def api_match(request: MatchRequest):
    """MATCH function via API."""
    result = await match_tool(request.lookup_value, request.lookup_array, request.match_type)
    return LookupResponse(**result)

@app.get("/functions")
async def list_functions():
    """List all available lookup functions."""
    functions = ["VLOOKUP", "HLOOKUP", "INDEX", "MATCH"]
    return {"functions": functions, "count": len(functions)}

# ================== MAIN EXECUTION ==================

async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8005, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Lookup Functions MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8005")
    print("API Docs: http://localhost:8005/docs")
    print("\nAvailable functions:")
    print("- VLOOKUP: Vertical lookup in tables")
    print("- HLOOKUP: Horizontal lookup in tables") 
    print("- INDEX: Return value at specific position")
    print("- MATCH: Find position of value in array")
    asyncio.run(main())