#!/usr/bin/env python3
"""
Array and Logical Functions MCP Server

This MCP server provides Google Sheets array and logical functions through the Model Context Protocol.
Functions include TRANSPOSE, UNIQUE, SORT, FILTER, IF, AND, OR, NOT with full compatibility to Google Sheets.
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
mcp = FastMCP("Array and Logical Functions Server")
app = FastAPI(title="Array and Logical Functions MCP Server", version="1.0.0")

def validate_file_path(file_path: str) -> str:
    """Validate and resolve file path for data operations."""
    # If it's a relative path, prefix with /mcp-data/
    if not file_path.startswith('/'):
        return f"/mcp-data/{file_path}"
    return file_path

# ================== ARRAY FUNCTIONS ==================

@mcp.tool()
async def transpose_tool(
    data_path: str = Field(description="Path to data file (CSV or Parquet)"),
    range_spec: Optional[str] = Field(None, description="A1 notation range (e.g., 'A1:C10')")
) -> Dict[str, Any]:
    """
    TRANSPOSE function matching Google Sheets =TRANSPOSE(range).
    Flips rows and columns in the data.
    
    Examples:
        transpose_tool("data.csv", "A1:C5") - Transpose specified range
        transpose_tool("matrix.parquet") - Transpose entire dataset
    """
    try:
        data_file = validate_file_path(data_path)
        result_df = sheets_funcs.TRANSPOSE(data_file, range_spec)
        
        # Convert DataFrame to dict for JSON serialization
        result_dict = result_df.to_dict(as_series=False)
        formula = f"=TRANSPOSE({range_spec or 'A:Z'})"
        
        return {
            "success": True,
            "result": result_dict,
            "formula": formula,
            "range": range_spec,
            "rows": len(result_df),
            "columns": len(result_df.columns)
        }
    except Exception as e:
        logger.error(f"Error in transpose_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def unique_tool(
    data_path: str = Field(description="Path to data file (CSV or Parquet)"),
    range_spec: Optional[str] = Field(None, description="A1 notation range"),
    by_column: bool = Field(False, description="Remove duplicate columns instead of rows"),
    exactly_once: bool = Field(False, description="Return only values that appear exactly once")
) -> Dict[str, Any]:
    """
    UNIQUE function matching Google Sheets =UNIQUE(range, [by_column], [exactly_once]).
    Returns unique values from a range.
    
    Examples:
        unique_tool("data.csv", "A:A") - Get unique values from column A
        unique_tool("data.csv", "A1:B100", False, True) - Get values that appear exactly once
    """
    try:
        data_file = validate_file_path(data_path)
        result_df = sheets_funcs.UNIQUE(data_file, range_spec, by_column, exactly_once)
        
        result_dict = result_df.to_dict(as_series=False)
        
        formula_parts = [f"=UNIQUE({range_spec or 'A:Z'})"]
        if by_column or exactly_once:
            formula_parts[0] = f"=UNIQUE({range_spec or 'A:Z'}, {str(by_column).upper()}, {str(exactly_once).upper()})"
        
        return {
            "success": True,
            "result": result_dict,
            "formula": formula_parts[0],
            "range": range_spec,
            "by_column": by_column,
            "exactly_once": exactly_once,
            "unique_count": len(result_df)
        }
    except Exception as e:
        logger.error(f"Error in unique_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def sort_tool(
    data_path: str = Field(description="Path to data file (CSV or Parquet)"),
    sort_column: int = Field(1, description="Column to sort by (1-based)"),
    is_ascending: bool = Field(True, description="Sort ascending (True) or descending (False)"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    SORT function matching Google Sheets =SORT(range, sort_column, is_ascending).
    Sorts data by specified column.
    
    Examples:
        sort_tool("sales.csv", 2, True) - Sort by column 2 ascending
        sort_tool("data.csv", 1, False, "A1:C100") - Sort specified range by column 1 descending
    """
    try:
        data_file = validate_file_path(data_path)
        result_df = sheets_funcs.SORT(data_file, sort_column, is_ascending, range_spec)
        
        result_dict = result_df.to_dict(as_series=False)
        formula = f"=SORT({range_spec or 'A:Z'}, {sort_column}, {str(is_ascending).upper()})"
        
        return {
            "success": True,
            "result": result_dict,
            "formula": formula,
            "range": range_spec,
            "sort_column": sort_column,
            "is_ascending": is_ascending,
            "sorted_rows": len(result_df)
        }
    except Exception as e:
        logger.error(f"Error in sort_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def filter_tool(
    data_path: str = Field(description="Path to data file (CSV or Parquet)"),
    filter_column: str = Field(description="Column name to filter on"),
    filter_value: Union[str, float, int] = Field(description="Value to filter by"),
    range_spec: Optional[str] = Field(None, description="A1 notation range")
) -> Dict[str, Any]:
    """
    FILTER function matching Google Sheets =FILTER(range, condition).
    Filters data based on conditions.
    
    Examples:
        filter_tool("sales.csv", "status", "Active") - Filter where status equals Active
        filter_tool("data.csv", "amount", 100) - Filter where amount equals 100
    """
    try:
        data_file = validate_file_path(data_path)
        
        # Load data and apply range if specified
        import polars as pl
        df = sheets_funcs._load_data(data_file)
        if range_spec:
            df = sheets_funcs.resolver.resolve_range(df, range_spec)
        
        # Apply filter condition
        if filter_column in df.columns:
            result_df = df.filter(pl.col(filter_column) == filter_value)
        else:
            raise ValueError(f"Column '{filter_column}' not found in data")
        
        result_dict = result_df.to_dict(as_series=False)
        formula = f"=FILTER({range_spec or 'A:Z'}, {filter_column}=\"{filter_value}\")"
        
        return {
            "success": True,
            "result": result_dict,
            "formula": formula,
            "range": range_spec,
            "filter_column": filter_column,
            "filter_value": filter_value,
            "filtered_rows": len(result_df)
        }
    except Exception as e:
        logger.error(f"Error in filter_tool: {e}")
        return {"success": False, "error": str(e)}

# ================== LOGICAL FUNCTIONS ==================

@mcp.tool()
async def if_tool(
    logical_test: bool = Field(description="Condition to test (True/False)"),
    value_if_true: Any = Field(description="Value to return if condition is true"),
    value_if_false: Any = Field(description="Value to return if condition is false")
) -> Dict[str, Any]:
    """
    IF function matching Google Sheets =IF(logical_test, value_if_true, value_if_false).
    
    Examples:
        if_tool(True, "Yes", "No") - Returns "Yes"
        if_tool(False, 100, 0) - Returns 0
    """
    try:
        result = sheets_funcs.IF(logical_test, value_if_true, value_if_false)
        formula = f"=IF({str(logical_test).upper()}, {value_if_true}, {value_if_false})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "logical_test": logical_test,
            "value_if_true": value_if_true,
            "value_if_false": value_if_false
        }
    except Exception as e:
        logger.error(f"Error in if_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def and_tool(
    logical_values: List[bool] = Field(description="List of logical values to AND together")
) -> Dict[str, Any]:
    """
    AND function matching Google Sheets =AND(logical1, [logical2, ...]).
    Returns TRUE if all arguments are TRUE.
    
    Examples:
        and_tool([True, True, True]) - Returns True
        and_tool([True, False, True]) - Returns False
    """
    try:
        result = sheets_funcs.AND(*logical_values)
        formula = f"=AND({', '.join(str(v).upper() for v in logical_values)})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "input_values": logical_values,
            "all_true": result
        }
    except Exception as e:
        logger.error(f"Error in and_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def or_tool(
    logical_values: List[bool] = Field(description="List of logical values to OR together")
) -> Dict[str, Any]:
    """
    OR function matching Google Sheets =OR(logical1, [logical2, ...]).
    Returns TRUE if any argument is TRUE.
    
    Examples:
        or_tool([False, False, True]) - Returns True
        or_tool([False, False, False]) - Returns False
    """
    try:
        result = sheets_funcs.OR(*logical_values)
        formula = f"=OR({', '.join(str(v).upper() for v in logical_values)})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "input_values": logical_values,
            "any_true": result
        }
    except Exception as e:
        logger.error(f"Error in or_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def not_tool(
    logical_value: bool = Field(description="Logical value to negate")
) -> Dict[str, Any]:
    """
    NOT function matching Google Sheets =NOT(logical).
    Returns the opposite of the logical value.
    
    Examples:
        not_tool(True) - Returns False
        not_tool(False) - Returns True
    """
    try:
        result = sheets_funcs.NOT(logical_value)
        formula = f"=NOT({str(logical_value).upper()})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "input_value": logical_value,
            "negated_value": result
        }
    except Exception as e:
        logger.error(f"Error in not_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def iferror_tool(
    value: Any = Field(description="Value to test for error"),
    value_if_error: Any = Field(description="Value to return if there's an error")
) -> Dict[str, Any]:
    """
    IFERROR function matching Google Sheets =IFERROR(value, value_if_error).
    Returns value_if_error if value is an error, otherwise returns value.
    
    Examples:
        iferror_tool("valid_value", "Error occurred") - Returns "valid_value"
        iferror_tool(None, "No data") - Returns "No data"
    """
    try:
        result = sheets_funcs.IFERROR(value, value_if_error)
        formula = f"=IFERROR({value}, {value_if_error})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_value": value,
            "error_value": value_if_error,
            "had_error": result == value_if_error
        }
    except Exception as e:
        logger.error(f"Error in iferror_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def isblank_tool(
    value: Any = Field(description="Value to test for blankness")
) -> Dict[str, Any]:
    """
    ISBLANK function matching Google Sheets =ISBLANK(value).
    Returns TRUE if value is empty.
    
    Examples:
        isblank_tool("") - Returns True
        isblank_tool("text") - Returns False
        isblank_tool(None) - Returns True
    """
    try:
        result = sheets_funcs.ISBLANK(value)
        formula = f"=ISBLANK({value})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "tested_value": value,
            "is_blank": result
        }
    except Exception as e:
        logger.error(f"Error in isblank_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def isnumber_tool(
    value: Any = Field(description="Value to test if it's a number")
) -> Dict[str, Any]:
    """
    ISNUMBER function matching Google Sheets =ISNUMBER(value).
    Returns TRUE if value is a number.
    
    Examples:
        isnumber_tool(123) - Returns True
        isnumber_tool("text") - Returns False
        isnumber_tool(45.67) - Returns True
    """
    try:
        result = sheets_funcs.ISNUMBER(value)
        formula = f"=ISNUMBER({value})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "tested_value": value,
            "is_number": result,
            "value_type": type(value).__name__
        }
    except Exception as e:
        logger.error(f"Error in isnumber_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def istext_tool(
    value: Any = Field(description="Value to test if it's text")
) -> Dict[str, Any]:
    """
    ISTEXT function matching Google Sheets =ISTEXT(value).
    Returns TRUE if value is text.
    
    Examples:
        istext_tool("Hello") - Returns True
        istext_tool(123) - Returns False
        istext_tool("") - Returns True (empty string is still text)
    """
    try:
        result = sheets_funcs.ISTEXT(value)
        formula = f"=ISTEXT({value})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "tested_value": value,
            "is_text": result,
            "value_type": type(value).__name__
        }
    except Exception as e:
        logger.error(f"Error in istext_tool: {e}")
        return {"success": False, "error": str(e)}

# ================== API MODELS ==================

class ArrayRequest(BaseModel):
    data_path: str
    range_spec: Optional[str] = None

class TransposeRequest(ArrayRequest):
    pass

class UniqueRequest(ArrayRequest):
    by_column: bool = False
    exactly_once: bool = False

class SortRequest(ArrayRequest):
    sort_column: int = 1
    is_ascending: bool = True

class FilterRequest(ArrayRequest):
    filter_column: str
    filter_value: Union[str, float, int]

class IfRequest(BaseModel):
    logical_test: bool
    value_if_true: Any
    value_if_false: Any

class LogicalRequest(BaseModel):
    logical_values: List[bool]

class NotRequest(BaseModel):
    logical_value: bool

class IferrorRequest(BaseModel):
    value: Any
    value_if_error: Any

class ValueTestRequest(BaseModel):
    value: Any

class ArrayResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    formula: Optional[str] = None
    error: Optional[str] = None

# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Array and Logical Functions MCP Server"}

# Array function endpoints
@app.post("/transpose", response_model=ArrayResponse)
async def api_transpose(request: TransposeRequest):
    """TRANSPOSE function via API."""
    result = await transpose_tool(request.data_path, request.range_spec)
    return ArrayResponse(**result)

@app.post("/unique", response_model=ArrayResponse)
async def api_unique(request: UniqueRequest):
    """UNIQUE function via API."""
    result = await unique_tool(
        request.data_path, 
        request.range_spec, 
        request.by_column, 
        request.exactly_once
    )
    return ArrayResponse(**result)

@app.post("/sort", response_model=ArrayResponse)
async def api_sort(request: SortRequest):
    """SORT function via API."""
    result = await sort_tool(
        request.data_path, 
        request.sort_column, 
        request.is_ascending, 
        request.range_spec
    )
    return ArrayResponse(**result)

@app.post("/filter", response_model=ArrayResponse)
async def api_filter(request: FilterRequest):
    """FILTER function via API."""
    result = await filter_tool(
        request.data_path, 
        request.filter_column, 
        request.filter_value, 
        request.range_spec
    )
    return ArrayResponse(**result)

# Logical function endpoints
@app.post("/if", response_model=ArrayResponse)
async def api_if(request: IfRequest):
    """IF function via API."""
    result = await if_tool(
        request.logical_test, 
        request.value_if_true, 
        request.value_if_false
    )
    return ArrayResponse(**result)

@app.post("/and", response_model=ArrayResponse)
async def api_and(request: LogicalRequest):
    """AND function via API."""
    result = await and_tool(*request.logical_values)
    return ArrayResponse(**result)

@app.post("/or", response_model=ArrayResponse)
async def api_or(request: LogicalRequest):
    """OR function via API."""
    result = await or_tool(*request.logical_values)
    return ArrayResponse(**result)

@app.post("/not", response_model=ArrayResponse)
async def api_not(request: NotRequest):
    """NOT function via API."""
    result = await not_tool(request.logical_value)
    return ArrayResponse(**result)

@app.get("/functions")
async def list_functions():
    """List all available array and logical functions."""
    functions = {
        "array": ["TRANSPOSE", "UNIQUE", "SORT", "FILTER"],
        "logical": ["IF", "AND", "OR", "NOT", "IFERROR", "ISBLANK", "ISNUMBER", "ISTEXT"]
    }
    return {"functions": functions, "total_count": sum(len(f) for f in functions.values())}

@app.get("/examples")
async def function_examples():
    """Get usage examples for array and logical functions."""
    return {
        "array_functions": {
            "transpose": "transpose_tool('matrix.csv', 'A1:C5')",
            "unique": "unique_tool('data.csv', 'A:A')",
            "sort": "sort_tool('sales.csv', 2, True)",
            "filter": "filter_tool('data.csv', 'status', 'Active')"
        },
        "logical_functions": {
            "if": "if_tool(True, 'Yes', 'No')",
            "and": "and_tool(True, True, False)",
            "or": "or_tool(False, True, False)",
            "not": "not_tool(True)",
            "iferror": "iferror_tool('value', 'Error')",
            "isblank": "isblank_tool('')",
            "isnumber": "isnumber_tool(123)",
            "istext": "istext_tool('Hello')"
        }
    }

# ================== MAIN EXECUTION ==================

async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8009, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Array and Logical Functions MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8009")
    print("API Docs: http://localhost:8009/docs")
    print("\nAvailable function categories:")
    print("Array Functions:")
    print("- TRANSPOSE: Flip rows and columns")
    print("- UNIQUE: Get unique values")
    print("- SORT: Sort data by column")
    print("- FILTER: Filter data by conditions")
    print("\nLogical Functions:")
    print("- IF: Conditional logic")
    print("- AND/OR/NOT: Boolean operations")
    print("- IFERROR: Error handling")
    print("- ISBLANK/ISNUMBER/ISTEXT: Type checking")
    asyncio.run(main())