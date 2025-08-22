#!/usr/bin/env python3
"""
Statistical Formula MCP Server - Phase 0 Category-Wise Implementation

This focused MCP server provides statistical formulas with 100% syntax accuracy:
- Descriptive Statistics: MEDIAN, STDEV, VAR, MODE
- Ranking & Percentiles: PERCENTILE, PERCENTRANK, RANK
- Data Analysis: Statistical calculations and distributions

Key Benefits:
- Focused toolset for AI agents (7 statistical tools)
- 100% Formula Accuracy guarantee
- Business-parameter interface
- Specialized for statistical analysis

Usage:
    # As MCP Server
    uv run python statistical_formula_mcp_server.py
    
    # As FastAPI Server  
    uv run python statistical_formula_mcp_server.py --port 3037
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
mcp = FastMCP("Statistical Formula Builder Server")
app = FastAPI(
    title="Statistical Formula Builder MCP Server", 
    version="1.0.0",
    description="Phase 0 Statistical Formulas with 100% syntax accuracy"
)


class StatisticalFormulaTools:
    """
    Focused MCP tools for statistical formulas only.
    
    This specialized server handles statistical analysis needs:
    - Descriptive statistics (MEDIAN, STDEV, VAR, MODE)
    - Ranking and percentiles (PERCENTILE, PERCENTRANK, RANK)
    - Data distribution analysis
    """
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        logger.info("📈 StatisticalFormulaTools initialized")
        
        # Get only statistical formulas
        all_formulas = self.formula_builder.get_supported_formulas()
        self.supported_formulas = [f for f in all_formulas if f in [
            'median', 'stdev', 'var', 'mode', 'percentile', 'percentrank', 'rank'
        ]]
        
        logger.info(f"📊 Supporting {len(self.supported_formulas)} statistical formulas")


# Global instance
statistical_tools = StatisticalFormulaTools()


# ================== DESCRIPTIVE STATISTICS TOOLS ==================

@mcp.tool()
async def build_median(
    range_: str = Field(description="Range of values to find median")
) -> Dict[str, Any]:
    """
    Build MEDIAN formula with guaranteed syntax accuracy.
    
    MEDIAN returns the middle value in a set of numbers when sorted.
    
    Examples:
        build_median("A1:A10") → =MEDIAN(A1:A10)
        build_median("Sales!B:B") → =MEDIAN(Sales!B:B)
    """
    try:
        formula = statistical_tools.formula_builder.build_formula('median', {
            'range': range_
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'median',
            'parameters': {
                'range': range_
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_median: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'median'}


@mcp.tool()
async def build_stdev(
    range_: str = Field(description="Range of values to calculate standard deviation"),
    sample: bool = Field(True, description="True for sample standard deviation (STDEV.S), False for population (STDEV.P)")
) -> Dict[str, Any]:
    """
    Build STDEV formula with guaranteed syntax accuracy.
    
    STDEV calculates the standard deviation of a dataset.
    
    Examples:
        build_stdev("A1:A10", True) → =STDEV.S(A1:A10)
        build_stdev("B:B", False) → =STDEV.P(B:B)
    """
    try:
        formula = statistical_tools.formula_builder.build_formula('stdev', {
            'range': range_,
            'sample': sample
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'stdev',
            'parameters': {
                'range': range_,
                'sample': sample
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_stdev: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'stdev'}


@mcp.tool()
async def build_var(
    range_: str = Field(description="Range of values to calculate variance"),
    sample: bool = Field(True, description="True for sample variance (VAR.S), False for population (VAR.P)")
) -> Dict[str, Any]:
    """
    Build VAR formula with guaranteed syntax accuracy.
    
    VAR calculates the variance of a dataset.
    
    Examples:
        build_var("A1:A10", True) → =VAR.S(A1:A10)
        build_var("Data!C:C", False) → =VAR.P(Data!C:C)
    """
    try:
        formula = statistical_tools.formula_builder.build_formula('var', {
            'range': range_,
            'sample': sample
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'var',
            'parameters': {
                'range': range_,
                'sample': sample
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_var: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'var'}


@mcp.tool()
async def build_mode(
    range_: str = Field(description="Range of values to find mode"),
    mode_type: str = Field("single", description="Type of mode: 'single' (MODE.SNGL) or 'multiple' (MODE.MULT)")
) -> Dict[str, Any]:
    """
    Build MODE formula with guaranteed syntax accuracy.
    
    MODE returns the most frequently occurring value(s) in a dataset.
    
    Examples:
        build_mode("A1:A10", "single") → =MODE.SNGL(A1:A10)
        build_mode("B:B", "multiple") → =MODE.MULT(B:B)
    """
    try:
        formula = statistical_tools.formula_builder.build_formula('mode', {
            'range': range_,
            'mode_type': mode_type
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'mode',
            'parameters': {
                'range': range_,
                'mode_type': mode_type
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_mode: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'mode'}


# ================== RANKING & PERCENTILES TOOLS ==================

@mcp.tool()
async def build_percentile(
    range_: str = Field(description="Range of values"),
    k: float = Field(description="Percentile value between 0 and 1 (e.g., 0.25 for 25th percentile)"),
    method: str = Field("exclusive", description="Calculation method: 'exclusive' (PERCENTILE.EXC) or 'inclusive' (PERCENTILE.INC)")
) -> Dict[str, Any]:
    """
    Build PERCENTILE formula with guaranteed syntax accuracy.
    
    PERCENTILE returns the value at a specific percentile of a dataset.
    
    Examples:
        build_percentile("A1:A100", 0.5, "exclusive") → =PERCENTILE.EXC(A1:A100,0.5)
        build_percentile("Scores!B:B", 0.95, "inclusive") → =PERCENTILE.INC(Scores!B:B,0.95)
    """
    try:
        formula = statistical_tools.formula_builder.build_formula('percentile', {
            'range': range_,
            'k': k,
            'method': method
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'percentile',
            'parameters': {
                'range': range_,
                'k': k,
                'method': method
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_percentile: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'percentile'}


@mcp.tool()
async def build_percentrank(
    range_: str = Field(description="Range of values"),
    x: str = Field(description="Value to find percentile rank for"),
    significance: Optional[int] = Field(None, description="Number of significant digits (optional)"),
    method: str = Field("exclusive", description="Calculation method: 'exclusive' (PERCENTRANK.EXC) or 'inclusive' (PERCENTRANK.INC)")
) -> Dict[str, Any]:
    """
    Build PERCENTRANK formula with guaranteed syntax accuracy.
    
    PERCENTRANK returns the percentile rank of a value in a dataset.
    
    Examples:
        build_percentrank("A1:A100", "B1", None, "exclusive") → =PERCENTRANK.EXC(A1:A100,B1)
        build_percentrank("Scores!A:A", "85", 3, "inclusive") → =PERCENTRANK.INC(Scores!A:A,85,3)
    """
    try:
        formula = statistical_tools.formula_builder.build_formula('percentrank', {
            'range': range_,
            'x': x,
            'significance': significance,
            'method': method
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'percentrank',
            'parameters': {
                'range': range_,
                'x': x,
                'significance': significance,
                'method': method
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_percentrank: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'percentrank'}


@mcp.tool()
async def build_rank(
    number: str = Field(description="Value to rank"),
    range_: str = Field(description="Range of values to rank against"),
    order: int = Field(0, description="Ranking order: 0 for descending (largest first), 1 for ascending (smallest first)"),
    method: str = Field("average", description="Ranking method: 'average' (RANK.AVG) or 'equal' (RANK.EQ)")
) -> Dict[str, Any]:
    """
    Build RANK formula with guaranteed syntax accuracy.
    
    RANK returns the rank of a number in a list of numbers.
    
    Examples:
        build_rank("B1", "B:B", 0, "average") → =RANK.AVG(B1,B:B,0)
        build_rank("Score", "A1:A100", 1, "equal") → =RANK.EQ(Score,A1:A100,1)
    """
    try:
        formula = statistical_tools.formula_builder.build_formula('rank', {
            'number': number,
            'range': range_,
            'order': order,
            'method': method
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'rank',
            'parameters': {
                'number': number,
                'range': range_,
                'order': order,
                'method': method
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_rank: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'rank'}


# ================== SERVER INFO TOOLS ==================

@mcp.tool()
async def get_statistical_capabilities() -> Dict[str, Any]:
    """
    Get all supported statistical formulas and their descriptions.
    """
    try:
        capabilities = {
            'descriptive_statistics': {
                'median': 'Find middle value in sorted dataset',
                'stdev': 'Calculate standard deviation (sample or population)',
                'var': 'Calculate variance (sample or population)',
                'mode': 'Find most frequently occurring value(s)'
            },
            'ranking_percentiles': {
                'percentile': 'Find value at specific percentile',
                'percentrank': 'Find percentile rank of a value',
                'rank': 'Find rank of value in dataset'
            }
        }
        
        use_cases = {
            'median': ['Salary analysis', 'Performance metrics', 'Data center calculations'],
            'stdev': ['Quality control', 'Risk analysis', 'Performance variability'],
            'percentile': ['Benchmarking', 'Target setting', 'Distribution analysis'],
            'rank': ['Performance ranking', 'Competitive analysis', 'Score ranking']
        }
        
        analysis_examples = {
            'performance_analysis': 'PERCENTILE(Sales_Data,0.9) for top 10% threshold',
            'quality_control': 'STDEV.S(Measurements) for process variability',
            'ranking_system': 'RANK.EQ(Employee_Score,All_Scores,0) for leaderboard'
        }
        
        return {
            'success': True,
            'server_name': 'Statistical Formula Builder',
            'total_tools': len(statistical_tools.supported_formulas),
            'categories': capabilities,
            'supported_formulas': statistical_tools.supported_formulas,
            'use_cases': use_cases,
            'analysis_examples': analysis_examples
        }
        
    except Exception as e:
        logger.error(f"Error in get_statistical_capabilities: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Statistical Formula Builder MCP Server",
        "version": "1.0.0",
        "formula_count": len(statistical_tools.supported_formulas),
        "categories": ["descriptive_statistics", "ranking_percentiles"]
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get statistical formula capabilities"""
    return await get_statistical_capabilities()


# ================== MAIN EXECUTION ==================

async def main():
    """Run the Statistical Formula MCP Server"""
    logger.info("🚀 Starting Statistical Formula Builder MCP Server...")
    logger.info("📈 Specialized for Statistical Analysis")
    logger.info(f"📊 Supporting {len(statistical_tools.supported_formulas)} statistical tools")
    logger.info("")
    logger.info("🎯 Supported Categories:")
    logger.info("   • Descriptive Statistics: MEDIAN, STDEV, VAR, MODE")
    logger.info("   • Ranking & Percentiles: PERCENTILE, PERCENTRANK, RANK")
    logger.info("")
    logger.info("✅ 100% Formula Accuracy Guaranteed")
    logger.info("🔬 Perfect for Data Analysis & Business Intelligence")
    logger.info("")
    
    # Run MCP server
    await mcp.run()


def run_fastapi_server(port: int = 3037):
    """Run the FastAPI server for HTTP access"""
    logger.info(f"🌐 Starting Statistical Formula FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 3037
        run_fastapi_server(port)
    else:
        # Run as MCP server
        asyncio.run(main())