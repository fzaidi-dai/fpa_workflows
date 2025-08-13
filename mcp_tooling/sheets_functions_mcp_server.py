#!/usr/bin/env python3
"""
Google Sheets Functions MCP Server

This MCP server provides Google Sheets-compatible functions through the Model Context Protocol.
All functions support A1 notation ranges and match Google Sheets behavior exactly.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import uvicorn
import polars as pl

# Import our Sheets-compatible functions
from sheets_compatible_functions import SheetsCompatibleFunctions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
mcp = FastMCP("Google Sheets Functions Server")
app = FastAPI(title="Google Sheets Functions MCP Server", version="1.0.0")
sheets_funcs = SheetsCompatibleFunctions()

# ================== MATH FUNCTIONS ==================

@mcp.tool()
async def sheets_sum(
    data_path: str = Field(description="Path to data file (CSV or Parquet)"),
    range_spec: Optional[str] = Field(None, description="A1 notation range (e.g., 'A1:C10', 'B:B')")
) -> Dict[str, Any]:
    """
    Calculate SUM matching Google Sheets =SUM(range).
    
    Examples:
        sheets_sum("data.csv", "A1:A10") - Sum first 10 rows of column A
        sheets_sum("data.parquet", "B:B") - Sum entire column B
    """
    try:
        result = sheets_funcs.SUM(data_path, range_spec)
        formula = f"=SUM({range_spec or 'A:A'})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_sum: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_average(
    data_path: str = Field(description="Path to data file (CSV or Parquet)"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    Calculate AVERAGE matching Google Sheets =AVERAGE(range).
    """
    try:
        result = sheets_funcs.AVERAGE(data_path, range_spec)
        formula = f"=AVERAGE({range_spec or 'A:A'})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_average: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_count(
    data_path: str = Field(description="Path to data file"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    COUNT numeric cells matching Google Sheets =COUNT(range).
    """
    try:
        result = sheets_funcs.COUNT(data_path, range_spec)
        formula = f"=COUNT({range_spec or 'A:A'})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_count: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_min(
    data_path: str = Field(description="Path to data file"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    Find MIN value matching Google Sheets =MIN(range).
    """
    try:
        result = sheets_funcs.MIN(data_path, range_spec)
        formula = f"=MIN({range_spec or 'A:A'})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_min: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_max(
    data_path: str = Field(description="Path to data file"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    Find MAX value matching Google Sheets =MAX(range).
    """
    try:
        result = sheets_funcs.MAX(data_path, range_spec)
        formula = f"=MAX({range_spec or 'A:A'})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_max: {e}")
        return {"success": False, "error": str(e)}

# ================== LOOKUP FUNCTIONS ==================

@mcp.tool()
async def sheets_vlookup(
    lookup_value: Union[str, float, int] = Field(description="Value to look up"),
    table_path: str = Field(description="Path to table data file"),
    col_index: int = Field(description="Column index to return (1-based)"),
    range_lookup: bool = Field(False, description="FALSE for exact match, TRUE for approximate")
) -> Dict[str, Any]:
    """
    VLOOKUP matching Google Sheets =VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup]).
    
    Examples:
        sheets_vlookup("Product1", "products.csv", 3, False) - Exact match lookup
        sheets_vlookup(100, "prices.csv", 2, True) - Approximate match
    """
    try:
        result = sheets_funcs.VLOOKUP(lookup_value, table_path, col_index, range_lookup)
        formula = f"=VLOOKUP({lookup_value}, A:Z, {col_index}, {range_lookup})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "lookup_value": lookup_value,
            "col_index": col_index
        }
    except Exception as e:
        logger.error(f"Error in sheets_vlookup: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_index(
    array_path: str = Field(description="Path to array data file"),
    row_num: int = Field(description="Row number (1-based)"),
    col_num: Optional[int] = Field(None, description="Column number (1-based)")
) -> Dict[str, Any]:
    """
    INDEX matching Google Sheets =INDEX(array, row_num, [column_num]).
    """
    try:
        result = sheets_funcs.INDEX(array_path, row_num, col_num)
        formula = f"=INDEX(A:Z, {row_num}{f', {col_num}' if col_num else ''})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "row": row_num,
            "column": col_num
        }
    except Exception as e:
        logger.error(f"Error in sheets_index: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_match(
    lookup_value: Union[str, float, int] = Field(description="Value to find"),
    lookup_array: List[Any] = Field(description="Array to search in"),
    match_type: int = Field(0, description="1=less than, 0=exact, -1=greater than")
) -> Dict[str, Any]:
    """
    MATCH matching Google Sheets =MATCH(lookup_value, lookup_array, [match_type]).
    """
    try:
        result = sheets_funcs.MATCH(lookup_value, lookup_array, match_type)
        formula = f"=MATCH({lookup_value}, array, {match_type})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "position": result
        }
    except Exception as e:
        logger.error(f"Error in sheets_match: {e}")
        return {"success": False, "error": str(e)}

# ================== CONDITIONAL AGGREGATION ==================

@mcp.tool()
async def sheets_sumif(
    range_path: str = Field(description="Path to range data file"),
    criteria: Union[str, float, int] = Field(description="Criteria for summing"),
    sum_range_path: Optional[str] = Field(None, description="Optional sum range path")
) -> Dict[str, Any]:
    """
    SUMIF matching Google Sheets =SUMIF(range, criteria, [sum_range]).
    
    Examples:
        sheets_sumif("sales.csv", ">100") - Sum values greater than 100
        sheets_sumif("categories.csv", "Electronics", "amounts.csv") - Sum amounts where category is Electronics
    """
    try:
        result = sheets_funcs.SUMIF(range_path, criteria, sum_range_path)
        formula = f"=SUMIF(A:A, \"{criteria}\"{f', B:B' if sum_range_path else ''})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "criteria": criteria
        }
    except Exception as e:
        logger.error(f"Error in sheets_sumif: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_countif(
    range_path: str = Field(description="Path to range data file"),
    criteria: Union[str, float, int] = Field(description="Criteria for counting")
) -> Dict[str, Any]:
    """
    COUNTIF matching Google Sheets =COUNTIF(range, criteria).
    
    Examples:
        sheets_countif("status.csv", "Active") - Count cells equal to "Active"
        sheets_countif("scores.csv", ">=80") - Count scores >= 80
    """
    try:
        result = sheets_funcs.COUNTIF(range_path, criteria)
        formula = f"=COUNTIF(A:A, \"{criteria}\")"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "criteria": criteria
        }
    except Exception as e:
        logger.error(f"Error in sheets_countif: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_averageif(
    range_path: str = Field(description="Path to range data file"),
    criteria: Union[str, float, int] = Field(description="Criteria for averaging"),
    average_range_path: Optional[str] = Field(None, description="Optional average range path")
) -> Dict[str, Any]:
    """
    AVERAGEIF matching Google Sheets =AVERAGEIF(range, criteria, [average_range]).
    """
    try:
        result = sheets_funcs.AVERAGEIF(range_path, criteria, average_range_path)
        formula = f"=AVERAGEIF(A:A, \"{criteria}\"{f', B:B' if average_range_path else ''})"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "criteria": criteria
        }
    except Exception as e:
        logger.error(f"Error in sheets_averageif: {e}")
        return {"success": False, "error": str(e)}

# ================== TEXT FUNCTIONS ==================

@mcp.tool()
async def sheets_concatenate(
    *texts: str
) -> Dict[str, Any]:
    """
    CONCATENATE matching Google Sheets =CONCATENATE(text1, [text2, ...]).
    """
    try:
        result = sheets_funcs.CONCATENATE(*texts)
        formula = f"=CONCATENATE({', '.join(repr(t) for t in texts)})"
        return {
            "success": True,
            "result": result,
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_concatenate: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_left(
    text: str = Field(description="Text to extract from"),
    num_chars: int = Field(description="Number of characters to extract")
) -> Dict[str, Any]:
    """
    LEFT matching Google Sheets =LEFT(text, num_chars).
    """
    try:
        result = sheets_funcs.LEFT(text, num_chars)
        formula = f"=LEFT(\"{text}\", {num_chars})"
        return {
            "success": True,
            "result": result,
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_left: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_mid(
    text: str = Field(description="Text to extract from"),
    start_num: int = Field(description="Starting position (1-based)"),
    num_chars: int = Field(description="Number of characters to extract")
) -> Dict[str, Any]:
    """
    MID matching Google Sheets =MID(text, start_num, num_chars).
    """
    try:
        result = sheets_funcs.MID(text, start_num, num_chars)
        formula = f"=MID(\"{text}\", {start_num}, {num_chars})"
        return {
            "success": True,
            "result": result,
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_mid: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_substitute(
    text: str = Field(description="Original text"),
    old_text: str = Field(description="Text to replace"),
    new_text: str = Field(description="Replacement text"),
    instance_num: Optional[int] = Field(None, description="Which occurrence to replace")
) -> Dict[str, Any]:
    """
    SUBSTITUTE matching Google Sheets =SUBSTITUTE(text, old_text, new_text, [instance_num]).
    """
    try:
        result = sheets_funcs.SUBSTITUTE(text, old_text, new_text, instance_num)
        formula = f"=SUBSTITUTE(\"{text}\", \"{old_text}\", \"{new_text}\"{f', {instance_num}' if instance_num else ''})"
        return {
            "success": True,
            "result": result,
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_substitute: {e}")
        return {"success": False, "error": str(e)}

# ================== DATE/TIME FUNCTIONS ==================

@mcp.tool()
async def sheets_today() -> Dict[str, Any]:
    """
    TODAY matching Google Sheets =TODAY().
    """
    try:
        result = sheets_funcs.TODAY()
        formula = "=TODAY()"
        return {
            "success": True,
            "result": result.isoformat(),
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_today: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_date(
    year: int = Field(description="Year"),
    month: int = Field(description="Month (1-12)"),
    day: int = Field(description="Day (1-31)")
) -> Dict[str, Any]:
    """
    DATE matching Google Sheets =DATE(year, month, day).
    """
    try:
        result = sheets_funcs.DATE(year, month, day)
        formula = f"=DATE({year}, {month}, {day})"
        return {
            "success": True,
            "result": result.isoformat(),
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_date: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_eomonth(
    start_date: str = Field(description="Start date (YYYY-MM-DD)"),
    months: int = Field(description="Number of months to add/subtract")
) -> Dict[str, Any]:
    """
    EOMONTH matching Google Sheets =EOMONTH(start_date, months).
    Returns the last day of the month.
    """
    try:
        result = sheets_funcs.EOMONTH(start_date, months)
        formula = f"=EOMONTH(\"{start_date}\", {months})"
        return {
            "success": True,
            "result": result.isoformat(),
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_eomonth: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_datedif(
    start_date: str = Field(description="Start date (YYYY-MM-DD)"),
    end_date: str = Field(description="End date (YYYY-MM-DD)"),
    unit: str = Field(description="Unit: Y, M, D, MD, YM, or YD")
) -> Dict[str, Any]:
    """
    DATEDIF matching Google Sheets =DATEDIF(start_date, end_date, unit).
    """
    try:
        result = sheets_funcs.DATEDIF(start_date, end_date, unit)
        formula = f"=DATEDIF(\"{start_date}\", \"{end_date}\", \"{unit}\")"
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "unit": unit
        }
    except Exception as e:
        logger.error(f"Error in sheets_datedif: {e}")
        return {"success": False, "error": str(e)}

# ================== LOGICAL FUNCTIONS ==================

@mcp.tool()
async def sheets_if(
    logical_test: bool = Field(description="Condition to test"),
    value_if_true: Any = Field(description="Value if condition is true"),
    value_if_false: Any = Field(description="Value if condition is false")
) -> Dict[str, Any]:
    """
    IF matching Google Sheets =IF(logical_test, value_if_true, value_if_false).
    """
    try:
        result = sheets_funcs.IF(logical_test, value_if_true, value_if_false)
        formula = f"=IF({logical_test}, {value_if_true}, {value_if_false})"
        return {
            "success": True,
            "result": result,
            "formula": formula
        }
    except Exception as e:
        logger.error(f"Error in sheets_if: {e}")
        return {"success": False, "error": str(e)}

# ================== ARRAY FUNCTIONS ==================

@mcp.tool()
async def sheets_transpose(
    data_path: str = Field(description="Path to data file"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    TRANSPOSE matching Google Sheets =TRANSPOSE(range).
    """
    try:
        result = sheets_funcs.TRANSPOSE(data_path, range_spec)
        # Convert DataFrame to simple structure for JSON serialization
        result_dict = result.to_dict(as_series=False)
        formula = f"=TRANSPOSE({range_spec or 'A:Z'})"
        return {
            "success": True,
            "result": result_dict,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_transpose: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_unique(
    data_path: str = Field(description="Path to data file"),
    range_spec: Optional[str] = Field(None, description="A1 notation range"),
    by_column: bool = Field(False, description="Remove duplicate columns instead of rows"),
    exactly_once: bool = Field(False, description="Return only values that appear exactly once")
) -> Dict[str, Any]:
    """
    UNIQUE matching Google Sheets =UNIQUE(range, [by_column], [exactly_once]).
    """
    try:
        result = sheets_funcs.UNIQUE(data_path, range_spec, by_column, exactly_once)
        result_dict = result.to_dict(as_series=False)
        formula = f"=UNIQUE({range_spec or 'A:Z'}{f', {by_column}' if by_column else ''}{f', {exactly_once}' if exactly_once else ''})"
        return {
            "success": True,
            "result": result_dict,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_unique: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sheets_sort(
    data_path: str = Field(description="Path to data file"),
    sort_column: int = Field(1, description="Column to sort by (1-based)"),
    is_ascending: bool = Field(True, description="Sort ascending (TRUE) or descending (FALSE)"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    SORT matching Google Sheets =SORT(range, sort_column, is_ascending).
    """
    try:
        result = sheets_funcs.SORT(data_path, sort_column, is_ascending, range_spec)
        result_dict = result.to_dict(as_series=False)
        formula = f"=SORT({range_spec or 'A:Z'}, {sort_column}, {is_ascending})"
        return {
            "success": True,
            "result": result_dict,
            "formula": formula,
            "range": range_spec
        }
    except Exception as e:
        logger.error(f"Error in sheets_sort: {e}")
        return {"success": False, "error": str(e)}

# ================== API ENDPOINTS ==================

class FunctionRequest(BaseModel):
    function_name: str
    parameters: Dict[str, Any]

class FunctionResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    formula: Optional[str] = None
    error: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Google Sheets Functions MCP Server"}

@app.post("/execute", response_model=FunctionResponse)
async def execute_function(request: FunctionRequest):
    """Execute any Sheets function via API."""
    try:
        # Map function names to actual functions
        func_name = request.function_name.upper()
        
        if hasattr(sheets_funcs, func_name):
            func = getattr(sheets_funcs, func_name)
            result = func(**request.parameters)
            
            # Generate formula
            if func_name in ["SUM", "AVERAGE", "COUNT", "MIN", "MAX"]:
                range_spec = request.parameters.get("range_spec", "A:A")
                formula = f"={func_name}({range_spec})"
            else:
                formula = f"={func_name}(...)"
            
            return FunctionResponse(
                success=True,
                result=result,
                formula=formula
            )
        else:
            raise ValueError(f"Unknown function: {request.function_name}")
            
    except Exception as e:
        logger.error(f"Error executing {request.function_name}: {e}")
        return FunctionResponse(
            success=False,
            error=str(e)
        )

@app.get("/functions")
async def list_functions():
    """List all available Sheets functions."""
    functions = [
        name for name in dir(sheets_funcs) 
        if not name.startswith("_") and callable(getattr(sheets_funcs, name))
    ]
    return {"functions": functions, "count": len(functions)}

# ================== MAIN EXECUTION ==================

async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8003, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Google Sheets Functions MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8003")
    print("API Docs: http://localhost:8003/docs")
    print("\nAvailable functions match Google Sheets exactly:")
    print("- Math: SUM, AVERAGE, COUNT, MIN, MAX")
    print("- Lookup: VLOOKUP, HLOOKUP, INDEX, MATCH")
    print("- Conditional: SUMIF, COUNTIF, AVERAGEIF, SUMIFS")
    print("- Text: CONCATENATE, LEFT, RIGHT, MID, SUBSTITUTE")
    print("- Date: TODAY, DATE, EOMONTH, DATEDIF")
    print("- Array: TRANSPOSE, UNIQUE, SORT, FILTER")
    print("- Logical: IF, AND, OR, NOT, IFERROR")
    asyncio.run(main())

def main():
    """Run the FastAPI server."""
    import sys
    
    port = 3002  # Default port for sheets functions server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Google Sheets Functions MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()