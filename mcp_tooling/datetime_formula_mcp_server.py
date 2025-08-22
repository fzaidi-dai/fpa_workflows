#!/usr/bin/env python3
"""
Date/Time Formula MCP Server - Phase 0 Category-Wise Implementation

This focused MCP server provides date/time formulas with 100% syntax accuracy:
- Current Date/Time: NOW, TODAY
- Date Construction: DATE, YEAR, MONTH, DAY
- Date Calculations: EOMONTH (End of Month)
- Temporal Operations: Date manipulation and formatting

Key Benefits:
- Focused toolset for AI agents (7 date/time tools)
- 100% Formula Accuracy guarantee
- Business-parameter interface
- Specialized for temporal operations

Usage:
    # As MCP Server
    uv run python datetime_formula_mcp_server.py
    
    # As FastAPI Server  
    uv run python datetime_formula_mcp_server.py --port 3038
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import Field
import uvicorn

# Import Formula Builder
from formula_builders import GoogleSheetsFormulaBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("DateTime Formula Builder Server")
app = FastAPI(
    title="DateTime Formula Builder MCP Server", 
    version="1.0.0",
    description="Phase 0 Date/Time Formulas with 100% syntax accuracy"
)


class DateTimeFormulaTools:
    """
    Focused MCP tools for date/time formulas only.
    
    This specialized server handles temporal operations:
    - Current date/time (NOW, TODAY)
    - Date construction (DATE, YEAR, MONTH, DAY)
    - Date calculations (EOMONTH)
    - Temporal analysis and reporting
    """
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        logger.info("📅 DateTimeFormulaTools initialized")
        
        # Get only date/time formulas
        all_formulas = self.formula_builder.get_supported_formulas()
        self.supported_formulas = [f for f in all_formulas if f in [
            'now', 'today', 'date', 'year', 'month', 'day', 'eomonth'
        ]]
        
        logger.info(f"🕐 Supporting {len(self.supported_formulas)} date/time formulas")


# Global instance
datetime_tools = DateTimeFormulaTools()


# ================== CURRENT DATE/TIME TOOLS ==================

@mcp.tool()
async def build_now() -> Dict[str, Any]:
    """
    Build NOW formula with guaranteed syntax accuracy.
    
    NOW returns the current date and time as a serial number.
    Updates every time the sheet recalculates.
    
    Examples:
        build_now() → =NOW()
    """
    try:
        formula = datetime_tools.formula_builder.build_formula('now', {})
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'now',
            'parameters': {}
        }
        
    except Exception as e:
        logger.error(f"Error in build_now: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'now'}


@mcp.tool()
async def build_today() -> Dict[str, Any]:
    """
    Build TODAY formula with guaranteed syntax accuracy.
    
    TODAY returns the current date (without time) as a serial number.
    Updates every time the sheet recalculates.
    
    Examples:
        build_today() → =TODAY()
    """
    try:
        formula = datetime_tools.formula_builder.build_formula('today', {})
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'today',
            'parameters': {}
        }
        
    except Exception as e:
        logger.error(f"Error in build_today: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'today'}


# ================== DATE CONSTRUCTION TOOLS ==================

@mcp.tool()
async def build_date(
    year: str = Field(description="Year value or cell reference"),
    month: str = Field(description="Month value or cell reference (1-12)"),
    day: str = Field(description="Day value or cell reference (1-31)")
) -> Dict[str, Any]:
    """
    Build DATE formula with guaranteed syntax accuracy.
    
    DATE creates a date from separate year, month, and day values.
    
    Examples:
        build_date("2024", "3", "15") → =DATE(2024,3,15)
        build_date("A1", "B1", "C1") → =DATE(A1,B1,C1)
    """
    try:
        formula = datetime_tools.formula_builder.build_formula('date', {
            'year': year,
            'month': month,
            'day': day
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'date',
            'parameters': {
                'year': year,
                'month': month,
                'day': day
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_date: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'date'}


@mcp.tool()
async def build_year(
    date_value: str = Field(description="Date value or cell reference")
) -> Dict[str, Any]:
    """
    Build YEAR formula with guaranteed syntax accuracy.
    
    YEAR extracts the year from a date as a 4-digit number.
    
    Examples:
        build_year("A1") → =YEAR(A1)
        build_year("TODAY()") → =YEAR(TODAY())
    """
    try:
        formula = datetime_tools.formula_builder.build_formula('year', {
            'date_value': date_value
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'year',
            'parameters': {
                'date_value': date_value
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_year: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'year'}


@mcp.tool()
async def build_month(
    date_value: str = Field(description="Date value or cell reference")
) -> Dict[str, Any]:
    """
    Build MONTH formula with guaranteed syntax accuracy.
    
    MONTH extracts the month from a date as a number from 1 to 12.
    
    Examples:
        build_month("A1") → =MONTH(A1)
        build_month("TODAY()") → =MONTH(TODAY())
    """
    try:
        formula = datetime_tools.formula_builder.build_formula('month', {
            'date_value': date_value
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'month',
            'parameters': {
                'date_value': date_value
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_month: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'month'}


@mcp.tool()
async def build_day(
    date_value: str = Field(description="Date value or cell reference")
) -> Dict[str, Any]:
    """
    Build DAY formula with guaranteed syntax accuracy.
    
    DAY extracts the day from a date as a number from 1 to 31.
    
    Examples:
        build_day("A1") → =DAY(A1)
        build_day("TODAY()") → =DAY(TODAY())
    """
    try:
        formula = datetime_tools.formula_builder.build_formula('day', {
            'date_value': date_value
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'day',
            'parameters': {
                'date_value': date_value
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_day: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'day'}


# ================== DATE CALCULATIONS TOOLS ==================

@mcp.tool()
async def build_eomonth(
    start_date: str = Field(description="Starting date value or cell reference"),
    months: int = Field(description="Number of months to add or subtract")
) -> Dict[str, Any]:
    """
    Build EOMONTH formula with guaranteed syntax accuracy.
    
    EOMONTH returns the last day of the month that is a specified number of months 
    before or after a given date.
    
    Examples:
        build_eomonth("A1", 0) → =EOMONTH(A1,0)    # End of current month
        build_eomonth("TODAY()", 3) → =EOMONTH(TODAY(),3)    # End of month 3 months from now
        build_eomonth("B2", -1) → =EOMONTH(B2,-1)   # End of previous month
    """
    try:
        formula = datetime_tools.formula_builder.build_formula('eomonth', {
            'start_date': start_date,
            'months': months
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'eomonth',
            'parameters': {
                'start_date': start_date,
                'months': months
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_eomonth: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'eomonth'}


# ================== SERVER INFO TOOLS ==================

@mcp.tool()
async def get_datetime_capabilities() -> Dict[str, Any]:
    """
    Get all supported date/time formulas and their descriptions.
    """
    try:
        capabilities = {
            'current_datetime': {
                'now': 'Get current date and time',
                'today': 'Get current date (no time)'
            },
            'date_construction': {
                'date': 'Create date from year, month, day values',
                'year': 'Extract year from date',
                'month': 'Extract month from date',
                'day': 'Extract day from date'
            },
            'date_calculations': {
                'eomonth': 'Get end of month date with offset'
            }
        }
        
        use_cases = {
            'now': ['Timestamp creation', 'Log entries', 'Real-time tracking'],
            'today': ['Date comparisons', 'Age calculations', 'Current reporting'],
            'date': ['Data validation', 'Custom date creation', 'Date conversion'],
            'eomonth': ['Month-end reporting', 'Billing cycles', 'Period calculations']
        }
        
        business_examples = {
            'aging_analysis': 'TODAY()-Invoice_Date for payment aging',
            'month_end_close': 'EOMONTH(TODAY(),0) for current month-end',
            'fiscal_year': 'YEAR(DATE(YEAR(TODAY())+1,3,31)) for fiscal year end'
        }
        
        return {
            'success': True,
            'server_name': 'DateTime Formula Builder',
            'total_tools': len(datetime_tools.supported_formulas),
            'categories': capabilities,
            'supported_formulas': datetime_tools.supported_formulas,
            'use_cases': use_cases,
            'business_examples': business_examples
        }
        
    except Exception as e:
        logger.error(f"Error in get_datetime_capabilities: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "DateTime Formula Builder MCP Server",
        "version": "1.0.0",
        "formula_count": len(datetime_tools.supported_formulas),
        "categories": ["current_datetime", "date_construction", "date_calculations"]
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get date/time formula capabilities"""
    return await get_datetime_capabilities()


# ================== MAIN EXECUTION ==================

async def main():
    """Run the DateTime Formula MCP Server"""
    logger.info("🚀 Starting DateTime Formula Builder MCP Server...")
    logger.info("📅 Specialized for Date/Time Formulas")
    logger.info(f"🕐 Supporting {len(datetime_tools.supported_formulas)} datetime tools")
    logger.info("")
    logger.info("🎯 Supported Categories:")
    logger.info("   • Current DateTime: NOW, TODAY")
    logger.info("   • Date Construction: DATE, YEAR, MONTH, DAY")
    logger.info("   • Date Calculations: EOMONTH")
    logger.info("")
    logger.info("✅ 100% Formula Accuracy Guaranteed")
    logger.info("⏰ Perfect for Temporal Analysis & Reporting")
    logger.info("")
    
    # Run MCP server
    await mcp.run()


def run_fastapi_server(port: int = 3038):
    """Run the FastAPI server for HTTP access"""
    logger.info(f"🌐 Starting DateTime Formula FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 3038
        run_fastapi_server(port)
    else:
        # Run as MCP server
        asyncio.run(main())