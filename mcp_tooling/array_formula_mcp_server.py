#!/usr/bin/env python3
"""
Array Formula MCP Server - Phase 0 Category-Wise Implementation

This focused MCP server provides array formulas with 100% syntax accuracy:
- Array Operations: ARRAYFORMULA, TRANSPOSE, UNIQUE
- Data Manipulation: SORT, FILTER, SEQUENCE
- Array Calculations: SUMPRODUCT

Key Benefits:
- Focused toolset for AI agents (7 array tools)
- 100% Formula Accuracy guarantee
- Business-parameter interface
- Specialized for array operations

Usage:
    # As MCP Server
    uv run python array_formula_mcp_server.py
    
    # As FastAPI Server  
    uv run python array_formula_mcp_server.py --port 3034
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
mcp = FastMCP("Array Formula Builder Server")
app = FastAPI(
    title="Array Formula Builder MCP Server", 
    version="1.0.0",
    description="Phase 0 Array Formulas with 100% syntax accuracy"
)


class ArrayFormulaTools:
    """
    Focused MCP tools for array formulas only.
    
    This specialized server handles array operations and manipulations:
    - Array operations (ARRAYFORMULA, TRANSPOSE, UNIQUE)
    - Data manipulation (SORT, FILTER, SEQUENCE)
    - Array calculations (SUMPRODUCT)
    """
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        logger.info("📊 ArrayFormulaTools initialized")
        
        # Get only array formulas
        all_formulas = self.formula_builder.get_supported_formulas()
        self.supported_formulas = [f for f in all_formulas if f in [
            'arrayformula', 'transpose', 'unique', 'sort', 'filter', 'sequence', 'sumproduct'
        ]]
        
        logger.info(f"🔢 Supporting {len(self.supported_formulas)} array formulas")


# Global instance
array_tools = ArrayFormulaTools()


# ================== ARRAY OPERATIONS TOOLS ==================

@mcp.tool()
async def build_arrayformula(
    formula: str = Field(description="The formula to apply as an array formula")
) -> Dict[str, Any]:
    """
    Build ARRAYFORMULA with guaranteed syntax accuracy.
    
    ARRAYFORMULA applies a formula to an entire range of cells automatically.
    
    Examples:
        build_arrayformula("A1:A10*B1:B10") → =ARRAYFORMULA(A1:A10*B1:B10)
        build_arrayformula("IF(A:A<>\"\",A:A*B:B,\"\")") → =ARRAYFORMULA(IF(A:A<>"",A:A*B:B,""))
    """
    try:
        formula_result = array_tools.formula_builder.build_formula('arrayformula', {
            'formula': formula
        })
        
        return {
            'success': True,
            'formula_generated': formula_result,
            'formula_type': 'arrayformula',
            'parameters': {
                'formula': formula
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_arrayformula: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'arrayformula'}


@mcp.tool()
async def build_transpose(
    range_: str = Field(description="Range to transpose (e.g., 'A1:C5')")
) -> Dict[str, Any]:
    """
    Build TRANSPOSE formula with guaranteed syntax accuracy.
    
    TRANSPOSE converts rows to columns and columns to rows.
    
    Examples:
        build_transpose("A1:C5") → =TRANSPOSE(A1:C5)
        build_transpose("Data!B:D") → =TRANSPOSE(Data!B:D)
    """
    try:
        formula = array_tools.formula_builder.build_formula('transpose', {
            'range': range_
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'transpose',
            'parameters': {
                'range': range_
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_transpose: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'transpose'}


@mcp.tool()
async def build_unique(
    range_: str = Field(description="Range to find unique values from"),
    by_col: Optional[bool] = Field(None, description="Compare by columns (True) or rows (False)")
) -> Dict[str, Any]:
    """
    Build UNIQUE formula with guaranteed syntax accuracy.
    
    UNIQUE returns unique values from a range, removing duplicates.
    
    Examples:
        build_unique("A:A") → =UNIQUE(A:A)
        build_unique("A1:C10", True) → =UNIQUE(A1:C10,TRUE)
    """
    try:
        formula = array_tools.formula_builder.build_formula('unique', {
            'range': range_,
            'by_col': by_col
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'unique',
            'parameters': {
                'range': range_,
                'by_col': by_col
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_unique: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'unique'}


# ================== DATA MANIPULATION TOOLS ==================

@mcp.tool()
async def build_sort(
    range_: str = Field(description="Range to sort"),
    sort_column: Optional[int] = Field(None, description="Column number to sort by (1-based)"),
    is_ascending: Optional[bool] = Field(None, description="Sort order: True for ascending, False for descending")
) -> Dict[str, Any]:
    """
    Build SORT formula with guaranteed syntax accuracy.
    
    SORT sorts the rows of a given array by the values in one or more columns.
    
    Examples:
        build_sort("A1:C10") → =SORT(A1:C10)
        build_sort("A:C", 2, False) → =SORT(A:C,2,FALSE)
    """
    try:
        formula = array_tools.formula_builder.build_formula('sort', {
            'range': range_,
            'sort_column': sort_column,
            'is_ascending': is_ascending
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'sort',
            'parameters': {
                'range': range_,
                'sort_column': sort_column,
                'is_ascending': is_ascending
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_sort: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'sort'}


@mcp.tool()
async def build_filter(
    range_: str = Field(description="Range to filter"),
    condition: str = Field(description="Boolean condition for filtering")
) -> Dict[str, Any]:
    """
    Build FILTER formula with guaranteed syntax accuracy.
    
    FILTER returns a filtered subset of a source range based on a condition.
    
    Examples:
        build_filter("A1:C10", "B1:B10>100") → =FILTER(A1:C10,B1:B10>100)
        build_filter("Data!A:C", "Data!B:B=\"Active\"") → =FILTER(Data!A:C,Data!B:B="Active")
    """
    try:
        formula = array_tools.formula_builder.build_formula('filter', {
            'range': range_,
            'condition': condition
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'filter',
            'parameters': {
                'range': range_,
                'condition': condition
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_filter: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'filter'}


@mcp.tool()
async def build_sequence(
    rows: int = Field(description="Number of rows in the sequence"),
    columns: Optional[int] = Field(None, description="Number of columns (default 1)"),
    start: Optional[int] = Field(None, description="Starting value (default 1)"),
    step: Optional[int] = Field(None, description="Step increment (default 1)")
) -> Dict[str, Any]:
    """
    Build SEQUENCE formula with guaranteed syntax accuracy.
    
    SEQUENCE generates a sequence of numbers in an array.
    
    Examples:
        build_sequence(5) → =SEQUENCE(5)
        build_sequence(3, 4, 10, 2) → =SEQUENCE(3,4,10,2)
    """
    try:
        formula = array_tools.formula_builder.build_formula('sequence', {
            'rows': rows,
            'columns': columns,
            'start': start,
            'step': step
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'sequence',
            'parameters': {
                'rows': rows,
                'columns': columns,
                'start': start,
                'step': step
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_sequence: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'sequence'}


# ================== ARRAY CALCULATIONS TOOLS ==================

@mcp.tool()
async def build_sumproduct(
    range1: str = Field(description="First range for multiplication"),
    range2: str = Field(description="Second range for multiplication"),
    range3: Optional[str] = Field(None, description="Third range (optional)"),
    range4: Optional[str] = Field(None, description="Fourth range (optional)")
) -> Dict[str, Any]:
    """
    Build SUMPRODUCT formula with guaranteed syntax accuracy.
    
    SUMPRODUCT multiplies corresponding elements in arrays and returns the sum.
    
    Examples:
        build_sumproduct("A1:A10", "B1:B10") → =SUMPRODUCT(A1:A10,B1:B10)
        build_sumproduct("A:A", "B:B", "C:C") → =SUMPRODUCT(A:A,B:B,C:C)
    """
    try:
        formula = array_tools.formula_builder.build_formula('sumproduct', {
            'range1': range1,
            'range2': range2,
            'range3': range3,
            'range4': range4
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'sumproduct',
            'parameters': {
                'range1': range1,
                'range2': range2,
                'range3': range3,
                'range4': range4
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_sumproduct: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'sumproduct'}


# ================== SERVER INFO TOOLS ==================

@mcp.tool()
async def get_array_capabilities() -> Dict[str, Any]:
    """
    Get all supported array formulas and their descriptions.
    """
    try:
        capabilities = {
            'array_operations': {
                'arrayformula': 'Apply formula to entire range automatically',
                'transpose': 'Convert rows to columns and vice versa',
                'unique': 'Extract unique values from range'
            },
            'data_manipulation': {
                'sort': 'Sort array data by specified columns',
                'filter': 'Filter array data based on conditions',
                'sequence': 'Generate sequence of numbers in array'
            },
            'array_calculations': {
                'sumproduct': 'Multiply corresponding array elements and sum'
            }
        }
        
        use_cases = {
            'arrayformula': ['Bulk calculations', 'Dynamic formulas', 'Range operations'],
            'transpose': ['Data reshaping', 'Matrix operations', 'Report formatting'],
            'filter': ['Data analysis', 'Conditional reporting', 'Dynamic dashboards'],
            'sumproduct': ['Weighted calculations', 'Conditional sums', 'Multi-criteria analysis']
        }
        
        return {
            'success': True,
            'server_name': 'Array Formula Builder',
            'total_tools': len(array_tools.supported_formulas),
            'categories': capabilities,
            'supported_formulas': array_tools.supported_formulas,
            'use_cases': use_cases
        }
        
    except Exception as e:
        logger.error(f"Error in get_array_capabilities: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Array Formula Builder MCP Server",
        "version": "1.0.0",
        "formula_count": len(array_tools.supported_formulas),
        "categories": ["array_operations", "data_manipulation", "array_calculations"]
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get array formula capabilities"""
    return await get_array_capabilities()


# ================== MAIN EXECUTION ==================

async def main():
    """Run the Array Formula MCP Server"""
    logger.info("🚀 Starting Array Formula Builder MCP Server...")
    logger.info("📊 Specialized for Array Formulas")
    logger.info(f"🔢 Supporting {len(array_tools.supported_formulas)} array tools")
    logger.info("")
    logger.info("🎯 Supported Categories:")
    logger.info("   • Array Operations: ARRAYFORMULA, TRANSPOSE, UNIQUE")
    logger.info("   • Data Manipulation: SORT, FILTER, SEQUENCE")
    logger.info("   • Array Calculations: SUMPRODUCT")
    logger.info("")
    logger.info("✅ 100% Formula Accuracy Guaranteed")
    logger.info("")
    
    # Run MCP server
    await mcp.run()


def run_fastapi_server(port: int = 3034):
    """Run the FastAPI server for HTTP access"""
    logger.info(f"🌐 Starting Array Formula FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 3034
        run_fastapi_server(port)
    else:
        # Run as MCP server
        asyncio.run(main())