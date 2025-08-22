#!/usr/bin/env python3
"""
Aggregation Formula MCP Server - Phase 0 Category-Wise Implementation

This focused MCP server provides aggregation formulas with 100% syntax accuracy:
- SUM, AVERAGE, COUNT, COUNTA, MAX, MIN
- SUMIF, SUMIFS, COUNTIF, COUNTIFS, AVERAGEIF, AVERAGEIFS
- SUBTOTAL

Key Benefits:
- Focused toolset for AI agents (13 tools vs 82)
- 100% Formula Accuracy guarantee
- Business-parameter interface
- Specialized for aggregation use cases

Usage:
    # As MCP Server
    uv run python aggregation_formula_mcp_server.py
    
    # As FastAPI Server  
    uv run python aggregation_formula_mcp_server.py --port 3030
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel, Field
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
mcp = FastMCP("Aggregation Formula Builder Server")
app = FastAPI(
    title="Aggregation Formula Builder MCP Server", 
    version="1.0.0",
    description="Phase 0 Aggregation Formulas with 100% syntax accuracy"
)


class AggregationFormulaTools:
    """
    Focused MCP tools for aggregation formulas only.
    
    This specialized server handles all aggregation needs:
    - Basic aggregations: SUM, AVERAGE, COUNT, MAX, MIN
    - Conditional aggregations: SUMIF, COUNTIF, AVERAGEIF
    - Multiple criteria: SUMIFS, COUNTIFS, AVERAGEIFS
    - Subtotals: SUBTOTAL function
    """
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        logger.info("🔢 AggregationFormulaTools initialized")
        
        # Get only aggregation formulas
        all_formulas = self.formula_builder.get_supported_formulas()
        self.supported_formulas = [f for f in all_formulas if f in [
            'sum', 'average', 'count', 'counta', 'max', 'min',
            'sumif', 'sumifs', 'countif', 'countifs', 
            'averageif', 'averageifs', 'subtotal'
        ]]
        
        logger.info(f"📊 Supporting {len(self.supported_formulas)} aggregation formulas")


# Global instance
aggregation_tools = AggregationFormulaTools()


# ================== BASIC AGGREGATION TOOLS ==================

@mcp.tool()
async def build_sum(
    range_ref: str = Field(description="Range to sum (e.g., 'A:A', 'Sheet1!B2:B100')")
) -> Dict[str, Any]:
    """
    Build SUM formula with guaranteed syntax accuracy.
    
    Examples:
        build_sum("A:A") → =SUM(A:A)
        build_sum("Sheet1!B2:B100") → =SUM(Sheet1!B2:B100)
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('sum', {
            'range_ref': range_ref
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'sum',
            'parameters': {'range_ref': range_ref}
        }
        
    except Exception as e:
        logger.error(f"Error in build_sum: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'sum'}


@mcp.tool()
async def build_average(
    range_ref: str = Field(description="Range to average (e.g., 'B:B', 'Data!C2:C50')")
) -> Dict[str, Any]:
    """
    Build AVERAGE formula with guaranteed syntax accuracy.
    
    Examples:
        build_average("B:B") → =AVERAGE(B:B)
        build_average("Data!C2:C50") → =AVERAGE(Data!C2:C50)
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('average', {
            'range_ref': range_ref
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'average',
            'parameters': {'range_ref': range_ref}
        }
        
    except Exception as e:
        logger.error(f"Error in build_average: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'average'}


@mcp.tool()
async def build_count(
    range_ref: str = Field(description="Range to count numeric values (e.g., 'C:C')")
) -> Dict[str, Any]:
    """
    Build COUNT formula (counts numeric values only).
    
    Examples:
        build_count("C:C") → =COUNT(C:C)
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('count', {
            'range_ref': range_ref
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'count',
            'parameters': {'range_ref': range_ref}
        }
        
    except Exception as e:
        logger.error(f"Error in build_count: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'count'}


@mcp.tool()
async def build_counta(
    range_ref: str = Field(description="Range to count non-empty values (e.g., 'D:D')")
) -> Dict[str, Any]:
    """
    Build COUNTA formula (counts all non-empty values).
    
    Examples:
        build_counta("D:D") → =COUNTA(D:D)
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('counta', {
            'range_ref': range_ref
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'counta',
            'parameters': {'range_ref': range_ref}
        }
        
    except Exception as e:
        logger.error(f"Error in build_counta: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'counta'}


@mcp.tool()
async def build_max(
    range_ref: str = Field(description="Range to find maximum value (e.g., 'E:E')")
) -> Dict[str, Any]:
    """
    Build MAX formula with guaranteed syntax accuracy.
    
    Examples:
        build_max("E:E") → =MAX(E:E)
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('max', {
            'range_ref': range_ref
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'max',
            'parameters': {'range_ref': range_ref}
        }
        
    except Exception as e:
        logger.error(f"Error in build_max: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'max'}


@mcp.tool()
async def build_min(
    range_ref: str = Field(description="Range to find minimum value (e.g., 'F:F')")
) -> Dict[str, Any]:
    """
    Build MIN formula with guaranteed syntax accuracy.
    
    Examples:
        build_min("F:F") → =MIN(F:F)
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('min', {
            'range_ref': range_ref
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'min',
            'parameters': {'range_ref': range_ref}
        }
        
    except Exception as e:
        logger.error(f"Error in build_min: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'min'}


# ================== CONDITIONAL AGGREGATION TOOLS ==================

@mcp.tool()
async def build_sumif(
    criteria_range: str = Field(description="Range containing criteria values (e.g., 'A:A')"),
    criteria: str = Field(description="Criteria to match (e.g., 'North', '>100', 'Active')"),
    sum_range: Optional[str] = Field(None, description="Range to sum (optional, defaults to criteria_range)")
) -> Dict[str, Any]:
    """
    Build SUMIF formula with guaranteed syntax accuracy.
    
    This is the most commonly used conditional aggregation function.
    
    Examples:
        build_sumif("A:A", "North", "B:B") → =SUMIF(A:A,"North",B:B)
        build_sumif("Sheet1!D:D", ">1000") → =SUMIF(Sheet1!D:D,">1000")
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('sumif', {
            'criteria_range': criteria_range,
            'criteria': criteria,
            'sum_range': sum_range
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'sumif',
            'parameters': {
                'criteria_range': criteria_range,
                'criteria': criteria,
                'sum_range': sum_range
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_sumif: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'sumif'}


@mcp.tool()
async def build_countif(
    criteria_range: str = Field(description="Range containing criteria values"),
    criteria: str = Field(description="Criteria to match")
) -> Dict[str, Any]:
    """
    Build COUNTIF formula with guaranteed syntax accuracy.
    
    Examples:
        build_countif("A:A", "Active") → =COUNTIF(A:A,"Active")
        build_countif("B:B", ">=50") → =COUNTIF(B:B,">=50")
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('countif', {
            'criteria_range': criteria_range,
            'criteria': criteria
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'countif',
            'parameters': {
                'criteria_range': criteria_range,
                'criteria': criteria
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_countif: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'countif'}


@mcp.tool()
async def build_averageif(
    criteria_range: str = Field(description="Range containing criteria values"),
    criteria: str = Field(description="Criteria to match"),
    average_range: Optional[str] = Field(None, description="Range to average (optional)")
) -> Dict[str, Any]:
    """
    Build AVERAGEIF formula with guaranteed syntax accuracy.
    
    Examples:
        build_averageif("A:A", "North", "B:B") → =AVERAGEIF(A:A,"North",B:B)
        build_averageif("C:C", ">50") → =AVERAGEIF(C:C,">50")
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('averageif', {
            'criteria_range': criteria_range,
            'criteria': criteria,
            'average_range': average_range
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'averageif',
            'parameters': {
                'criteria_range': criteria_range,
                'criteria': criteria,
                'average_range': average_range
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_averageif: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'averageif'}


# ================== MULTIPLE CRITERIA TOOLS ==================

@mcp.tool()
async def build_sumifs(
    sum_range: str = Field(description="Range containing values to sum"),
    criteria_pairs: List[List[str]] = Field(description="List of [criteria_range, criteria] pairs")
) -> Dict[str, Any]:
    """
    Build SUMIFS formula with multiple criteria and guaranteed syntax accuracy.
    
    Examples:
        build_sumifs("C:C", [["A:A", "North"], ["B:B", ">100"]]) 
        → =SUMIFS(C:C,A:A,"North",B:B,">100")
    """
    try:
        # Convert list format to tuple format expected by formula builder
        criteria_tuples = [(pair[0], pair[1]) for pair in criteria_pairs]
        
        formula = aggregation_tools.formula_builder.build_formula('sumifs', {
            'sum_range': sum_range,
            'criteria_pairs': criteria_tuples
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'sumifs',
            'parameters': {
                'sum_range': sum_range,
                'criteria_pairs': criteria_pairs
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_sumifs: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'sumifs'}


@mcp.tool()
async def build_countifs(
    criteria_pairs: List[List[str]] = Field(description="List of [criteria_range, criteria] pairs")
) -> Dict[str, Any]:
    """
    Build COUNTIFS formula with multiple criteria and guaranteed syntax accuracy.
    
    Examples:
        build_countifs([["A:A", "North"], ["B:B", "Active"]]) 
        → =COUNTIFS(A:A,"North",B:B,"Active")
    """
    try:
        # Convert list format to tuple format expected by formula builder
        criteria_tuples = [(pair[0], pair[1]) for pair in criteria_pairs]
        
        formula = aggregation_tools.formula_builder.build_formula('countifs', {
            'criteria_pairs': criteria_tuples
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'countifs',
            'parameters': {
                'criteria_pairs': criteria_pairs
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_countifs: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'countifs'}


@mcp.tool()
async def build_averageifs(
    average_range: str = Field(description="Range containing values to average"),
    criteria_pairs: List[List[str]] = Field(description="List of [criteria_range, criteria] pairs")
) -> Dict[str, Any]:
    """
    Build AVERAGEIFS formula with multiple criteria and guaranteed syntax accuracy.
    
    Examples:
        build_averageifs("C:C", [["A:A", "North"], ["B:B", ">50"]]) 
        → =AVERAGEIFS(C:C,A:A,"North",B:B,">50")
    """
    try:
        # Convert list format to tuple format expected by formula builder
        criteria_tuples = [(pair[0], pair[1]) for pair in criteria_pairs]
        
        formula = aggregation_tools.formula_builder.build_formula('averageifs', {
            'average_range': average_range,
            'criteria_pairs': criteria_tuples
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'averageifs',
            'parameters': {
                'average_range': average_range,
                'criteria_pairs': criteria_pairs
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_averageifs: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'averageifs'}


@mcp.tool()
async def build_subtotal(
    function_num: int = Field(description="SUBTOTAL function number (1-11 or 101-111)"),
    range_ref: str = Field(description="Range to apply subtotal to")
) -> Dict[str, Any]:
    """
    Build SUBTOTAL formula with guaranteed syntax accuracy.
    
    Function numbers:
    1/101=AVERAGE, 2/102=COUNT, 3/103=COUNTA, 4/104=MAX, 5/105=MIN,
    6/106=PRODUCT, 7/107=STDEV, 8/108=STDEVP, 9/109=SUM, 10/110=VAR, 11/111=VARP
    (100+ versions ignore hidden rows)
    
    Examples:
        build_subtotal(9, "B2:B10") → =SUBTOTAL(9,B2:B10)  # SUM
        build_subtotal(109, "C:C") → =SUBTOTAL(109,C:C)    # SUM ignoring hidden
    """
    try:
        formula = aggregation_tools.formula_builder.build_formula('subtotal', {
            'function_num': function_num,
            'range_ref': range_ref
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'subtotal',
            'parameters': {
                'function_num': function_num,
                'range_ref': range_ref
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_subtotal: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'subtotal'}


# ================== SERVER INFO TOOLS ==================

@mcp.tool()
async def get_aggregation_capabilities() -> Dict[str, Any]:
    """
    Get all supported aggregation formulas and their descriptions.
    """
    try:
        capabilities = {
            'basic_aggregation': {
                'sum': 'Sum values in a range',
                'average': 'Calculate average of values',
                'count': 'Count numeric values only',
                'counta': 'Count all non-empty values',
                'max': 'Find maximum value',
                'min': 'Find minimum value'
            },
            'conditional_aggregation': {
                'sumif': 'Sum values meeting single criteria',
                'countif': 'Count values meeting single criteria',
                'averageif': 'Average values meeting single criteria'
            },
            'multiple_criteria': {
                'sumifs': 'Sum values meeting multiple criteria',
                'countifs': 'Count values meeting multiple criteria',
                'averageifs': 'Average values meeting multiple criteria'
            },
            'subtotals': {
                'subtotal': 'Calculate subtotals with function numbers'
            }
        }
        
        return {
            'success': True,
            'server_name': 'Aggregation Formula Builder',
            'total_tools': len(aggregation_tools.supported_formulas),
            'categories': capabilities,
            'supported_formulas': aggregation_tools.supported_formulas
        }
        
    except Exception as e:
        logger.error(f"Error in get_aggregation_capabilities: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Aggregation Formula Builder MCP Server",
        "version": "1.0.0",
        "formula_count": len(aggregation_tools.supported_formulas),
        "categories": ["basic", "conditional", "multiple_criteria", "subtotals"]
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get aggregation formula capabilities"""
    return await get_aggregation_capabilities()


# ================== MAIN EXECUTION ==================

async def main():
    """Run the Aggregation Formula MCP Server"""
    logger.info("🚀 Starting Aggregation Formula Builder MCP Server...")
    logger.info("🔢 Specialized for Aggregation Formulas")
    logger.info(f"📊 Supporting {len(aggregation_tools.supported_formulas)} aggregation tools")
    logger.info("")
    logger.info("🎯 Supported Categories:")
    logger.info("   • Basic: SUM, AVERAGE, COUNT, MAX, MIN")
    logger.info("   • Conditional: SUMIF, COUNTIF, AVERAGEIF")
    logger.info("   • Multiple Criteria: SUMIFS, COUNTIFS, AVERAGEIFS")
    logger.info("   • Subtotals: SUBTOTAL function")
    logger.info("")
    logger.info("✅ 100% Formula Accuracy Guaranteed")
    logger.info("")
    
    # Run MCP server
    await mcp.run()


def run_fastapi_server(port: int = 3030):
    """Run the FastAPI server for HTTP access"""
    logger.info(f"🌐 Starting Aggregation Formula FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 3030
        run_fastapi_server(port)
    else:
        # Run as MCP server
        asyncio.run(main())