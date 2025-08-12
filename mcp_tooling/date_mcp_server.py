#!/usr/bin/env python3
"""
Date/Time Functions MCP Server

This MCP server provides Google Sheets date and time functions through the Model Context Protocol.
Functions include TODAY, NOW, DATE, YEAR, MONTH, DAY, WEEKDAY, EOMONTH, DATEDIF with full compatibility to Google Sheets.
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
mcp = FastMCP("Date/Time Functions Server")
app = FastAPI(title="Date/Time Functions MCP Server", version="1.0.0")

# ================== DATE/TIME FUNCTIONS ==================

@mcp.tool()
async def today_tool() -> Dict[str, Any]:
    """
    TODAY function matching Google Sheets =TODAY().
    Returns the current date.
    
    Examples:
        today_tool() - Returns current date like "2025-01-15"
    """
    try:
        result = sheets_funcs.TODAY()
        formula = "=TODAY()"
        
        return {
            "success": True,
            "result": result.isoformat(),
            "formula": formula,
            "date_object": str(result)
        }
    except Exception as e:
        logger.error(f"Error in today_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def now_tool() -> Dict[str, Any]:
    """
    NOW function matching Google Sheets =NOW().
    Returns the current date and time.
    
    Examples:
        now_tool() - Returns current datetime like "2025-01-15T14:30:00"
    """
    try:
        result = sheets_funcs.NOW()
        formula = "=NOW()"
        
        return {
            "success": True,
            "result": result.isoformat(),
            "formula": formula,
            "datetime_object": str(result)
        }
    except Exception as e:
        logger.error(f"Error in now_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def date_tool(
    year: int = Field(description="Year (e.g., 2025)"),
    month: int = Field(description="Month (1-12)"),
    day: int = Field(description="Day (1-31)")
) -> Dict[str, Any]:
    """
    DATE function matching Google Sheets =DATE(year, month, day).
    
    Examples:
        date_tool(2025, 12, 25) - Returns "2025-12-25"
        date_tool(2024, 2, 29) - Returns "2024-02-29" (leap year)
    """
    try:
        result = sheets_funcs.DATE(year, month, day)
        formula = f"=DATE({year}, {month}, {day})"
        
        return {
            "success": True,
            "result": result.isoformat(),
            "formula": formula,
            "year": year,
            "month": month,
            "day": day
        }
    except Exception as e:
        logger.error(f"Error in date_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def year_tool(
    date_value: str = Field(description="Date string (YYYY-MM-DD format)")
) -> Dict[str, Any]:
    """
    YEAR function matching Google Sheets =YEAR(date).
    
    Examples:
        year_tool("2025-12-25") - Returns 2025
        year_tool("2024-02-29") - Returns 2024
    """
    try:
        result = sheets_funcs.YEAR(date_value)
        formula = f"=YEAR(\"{date_value}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_date": date_value
        }
    except Exception as e:
        logger.error(f"Error in year_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def month_tool(
    date_value: str = Field(description="Date string (YYYY-MM-DD format)")
) -> Dict[str, Any]:
    """
    MONTH function matching Google Sheets =MONTH(date).
    
    Examples:
        month_tool("2025-12-25") - Returns 12
        month_tool("2024-02-29") - Returns 2
    """
    try:
        result = sheets_funcs.MONTH(date_value)
        formula = f"=MONTH(\"{date_value}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_date": date_value
        }
    except Exception as e:
        logger.error(f"Error in month_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def day_tool(
    date_value: str = Field(description="Date string (YYYY-MM-DD format)")
) -> Dict[str, Any]:
    """
    DAY function matching Google Sheets =DAY(date).
    
    Examples:
        day_tool("2025-12-25") - Returns 25
        day_tool("2024-02-29") - Returns 29
    """
    try:
        result = sheets_funcs.DAY(date_value)
        formula = f"=DAY(\"{date_value}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_date": date_value
        }
    except Exception as e:
        logger.error(f"Error in day_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def weekday_tool(
    date_value: str = Field(description="Date string (YYYY-MM-DD format)"),
    type_num: int = Field(1, description="1=Sun-Sat (1-7), 2=Mon-Sun (1-7), 3=Mon-Sun (0-6)")
) -> Dict[str, Any]:
    """
    WEEKDAY function matching Google Sheets =WEEKDAY(date, [type]).
    
    Examples:
        weekday_tool("2025-01-15", 1) - Returns weekday number (Sun=1, Mon=2, etc.)
        weekday_tool("2025-01-15", 2) - Returns weekday number (Mon=1, Tue=2, etc.)
        weekday_tool("2025-01-15", 3) - Returns weekday number (Mon=0, Tue=1, etc.)
    """
    try:
        result = sheets_funcs.WEEKDAY(date_value, type_num)
        formula = f"=WEEKDAY(\"{date_value}\", {type_num})"
        
        # Add weekday name for clarity
        weekday_names = {
            1: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            2: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            3: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        }
        
        if type_num == 1:
            day_name = weekday_names[1][result - 1]
        elif type_num == 2:
            day_name = weekday_names[2][result - 1]
        else:  # type_num == 3
            day_name = weekday_names[3][result]
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_date": date_value,
            "type_num": type_num,
            "day_name": day_name
        }
    except Exception as e:
        logger.error(f"Error in weekday_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def eomonth_tool(
    start_date: str = Field(description="Start date string (YYYY-MM-DD format)"),
    months: int = Field(description="Number of months to add (positive) or subtract (negative)")
) -> Dict[str, Any]:
    """
    EOMONTH function matching Google Sheets =EOMONTH(start_date, months).
    Returns the last day of the month that is a specified number of months before or after start_date.
    
    Examples:
        eomonth_tool("2025-01-15", 2) - Returns "2025-03-31" (end of March)
        eomonth_tool("2025-03-15", -1) - Returns "2025-02-28" (end of February)
        eomonth_tool("2024-01-15", 1) - Returns "2024-02-29" (leap year February)
    """
    try:
        result = sheets_funcs.EOMONTH(start_date, months)
        formula = f"=EOMONTH(\"{start_date}\", {months})"
        
        return {
            "success": True,
            "result": result.isoformat(),
            "formula": formula,
            "start_date": start_date,
            "months_offset": months
        }
    except Exception as e:
        logger.error(f"Error in eomonth_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def datedif_tool(
    start_date: str = Field(description="Start date string (YYYY-MM-DD format)"),
    end_date: str = Field(description="End date string (YYYY-MM-DD format)"),
    unit: str = Field(description="Unit: Y (years), M (months), D (days), MD (days ignoring months/years), YM (months ignoring years), YD (days ignoring years)")
) -> Dict[str, Any]:
    """
    DATEDIF function matching Google Sheets =DATEDIF(start_date, end_date, unit).
    
    Examples:
        datedif_tool("2020-01-01", "2025-01-01", "Y") - Returns 5 years
        datedif_tool("2024-01-15", "2024-03-20", "M") - Returns 2 months
        datedif_tool("2024-01-01", "2024-01-31", "D") - Returns 30 days
        datedif_tool("2024-01-15", "2024-03-20", "MD") - Returns days difference ignoring months
    """
    try:
        result = sheets_funcs.DATEDIF(start_date, end_date, unit)
        formula = f"=DATEDIF(\"{start_date}\", \"{end_date}\", \"{unit}\")"
        
        unit_descriptions = {
            "Y": "years",
            "M": "months", 
            "D": "days",
            "MD": "days (ignoring months and years)",
            "YM": "months (ignoring years)",
            "YD": "days (ignoring years)"
        }
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "start_date": start_date,
            "end_date": end_date,
            "unit": unit,
            "unit_description": unit_descriptions.get(unit, unit)
        }
    except Exception as e:
        logger.error(f"Error in datedif_tool: {e}")
        return {"success": False, "error": str(e)}

# ================== API MODELS ==================

class DateRequest(BaseModel):
    year: int
    month: int
    day: int

class DateValueRequest(BaseModel):
    date_value: str

class WeekdayRequest(BaseModel):
    date_value: str
    type_num: int = 1

class EomonthRequest(BaseModel):
    start_date: str
    months: int

class DatedifRequest(BaseModel):
    start_date: str
    end_date: str
    unit: str

class DateResponse(BaseModel):
    success: bool
    result: Optional[Union[str, int]] = None
    formula: Optional[str] = None
    error: Optional[str] = None

# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Date/Time Functions MCP Server"}

@app.get("/today", response_model=DateResponse)
async def api_today():
    """TODAY function via API."""
    result = await today_tool()
    return DateResponse(**result)

@app.get("/now", response_model=DateResponse)
async def api_now():
    """NOW function via API."""
    result = await now_tool()
    return DateResponse(**result)

@app.post("/date", response_model=DateResponse)
async def api_date(request: DateRequest):
    """DATE function via API."""
    result = await date_tool(request.year, request.month, request.day)
    return DateResponse(**result)

@app.post("/year", response_model=DateResponse)
async def api_year(request: DateValueRequest):
    """YEAR function via API."""
    result = await year_tool(request.date_value)
    return DateResponse(**result)

@app.post("/month", response_model=DateResponse)
async def api_month(request: DateValueRequest):
    """MONTH function via API."""
    result = await month_tool(request.date_value)
    return DateResponse(**result)

@app.post("/day", response_model=DateResponse)
async def api_day(request: DateValueRequest):
    """DAY function via API."""
    result = await day_tool(request.date_value)
    return DateResponse(**result)

@app.post("/weekday", response_model=DateResponse)
async def api_weekday(request: WeekdayRequest):
    """WEEKDAY function via API."""
    result = await weekday_tool(request.date_value, request.type_num)
    return DateResponse(**result)

@app.post("/eomonth", response_model=DateResponse)
async def api_eomonth(request: EomonthRequest):
    """EOMONTH function via API."""
    result = await eomonth_tool(request.start_date, request.months)
    return DateResponse(**result)

@app.post("/datedif", response_model=DateResponse)
async def api_datedif(request: DatedifRequest):
    """DATEDIF function via API."""
    result = await datedif_tool(request.start_date, request.end_date, request.unit)
    return DateResponse(**result)

@app.get("/functions")
async def list_functions():
    """List all available date/time functions."""
    functions = [
        "TODAY", "NOW", "DATE", "YEAR", "MONTH", "DAY", 
        "WEEKDAY", "EOMONTH", "DATEDIF"
    ]
    return {"functions": functions, "count": len(functions)}

@app.get("/examples")
async def function_examples():
    """Get usage examples for date/time functions."""
    return {
        "current_date_time": {
            "today": "today_tool()",
            "now": "now_tool()"
        },
        "date_creation": {
            "date": "date_tool(2025, 12, 25)"
        },
        "date_extraction": {
            "year": "year_tool('2025-12-25')",
            "month": "month_tool('2025-12-25')", 
            "day": "day_tool('2025-12-25')",
            "weekday": "weekday_tool('2025-01-15', 1)"
        },
        "date_calculations": {
            "eomonth": "eomonth_tool('2025-01-15', 2)",
            "datedif": "datedif_tool('2020-01-01', '2025-01-01', 'Y')"
        },
        "datedif_units": {
            "Y": "Years between dates",
            "M": "Months between dates", 
            "D": "Days between dates",
            "MD": "Days difference ignoring months and years",
            "YM": "Months difference ignoring years",
            "YD": "Days difference ignoring years"
        }
    }

# ================== MAIN EXECUTION ==================

async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8008, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Date/Time Functions MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8008")
    print("API Docs: http://localhost:8008/docs")
    print("\nAvailable functions:")
    print("- TODAY/NOW: Current date and time")
    print("- DATE: Create date from year/month/day")
    print("- YEAR/MONTH/DAY: Extract date components")
    print("- WEEKDAY: Get day of week")
    print("- EOMONTH: End of month calculations")
    print("- DATEDIF: Calculate date differences")
    print("\nDate format: YYYY-MM-DD (e.g., '2025-01-15')")
    asyncio.run(main())