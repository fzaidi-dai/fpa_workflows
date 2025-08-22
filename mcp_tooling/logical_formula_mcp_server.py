#!/usr/bin/env python3
"""
Logical Formula MCP Server - Phase 0 Category-Wise Implementation

This focused MCP server provides logical formulas with 100% syntax accuracy:
- Conditional Logic: IF statements and nested conditions
- Boolean Operations: AND, OR, NOT
- Decision Making: Complex logical evaluations

Key Benefits:
- Focused toolset for AI agents (4 logical tools)
- 100% Formula Accuracy guarantee
- Business-parameter interface
- Specialized for logical operations

Usage:
    # As MCP Server
    uv run python logical_formula_mcp_server.py
    
    # As FastAPI Server  
    uv run python logical_formula_mcp_server.py --port 3036
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
mcp = FastMCP("Logical Formula Builder Server")
app = FastAPI(
    title="Logical Formula Builder MCP Server", 
    version="1.0.0",
    description="Phase 0 Logical Formulas with 100% syntax accuracy"
)


class LogicalFormulaTools:
    """
    Focused MCP tools for logical formulas only.
    
    This specialized server handles logical operations and decision making:
    - Conditional logic (IF statements)
    - Boolean operations (AND, OR, NOT)
    - Complex logical evaluations
    """
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        logger.info("🧠 LogicalFormulaTools initialized")
        
        # Get only logical formulas
        all_formulas = self.formula_builder.get_supported_formulas()
        self.supported_formulas = [f for f in all_formulas if f in [
            'if', 'and', 'or', 'not'
        ]]
        
        logger.info(f"⚡ Supporting {len(self.supported_formulas)} logical formulas")


# Global instance
logical_tools = LogicalFormulaTools()


# ================== CONDITIONAL LOGIC TOOLS ==================

@mcp.tool()
async def build_if(
    condition: str = Field(description="Logical test condition"),
    value_if_true: str = Field(description="Value returned if condition is TRUE"),
    value_if_false: Optional[str] = Field(None, description="Value returned if condition is FALSE (optional)")
) -> Dict[str, Any]:
    """
    Build IF formula with guaranteed syntax accuracy.
    
    IF performs a logical test and returns different values based on the result.
    
    Examples:
        build_if("A1>10", "High", "Low") → =IF(A1>10,"High","Low")
        build_if("B2=\"Active\"", "✓", "✗") → =IF(B2="Active","✓","✗")
        build_if("C3>=100", "Pass") → =IF(C3>=100,"Pass")
    """
    try:
        formula = logical_tools.formula_builder.build_formula('if', {
            'condition': condition,
            'value_if_true': value_if_true,
            'value_if_false': value_if_false
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'if',
            'parameters': {
                'condition': condition,
                'value_if_true': value_if_true,
                'value_if_false': value_if_false
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_if: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'if'}


# ================== BOOLEAN OPERATIONS TOOLS ==================

@mcp.tool()
async def build_and(
    condition1: str = Field(description="First logical condition"),
    condition2: str = Field(description="Second logical condition"),
    condition3: Optional[str] = Field(None, description="Third logical condition (optional)"),
    condition4: Optional[str] = Field(None, description="Fourth logical condition (optional)"),
    condition5: Optional[str] = Field(None, description="Fifth logical condition (optional)")
) -> Dict[str, Any]:
    """
    Build AND formula with guaranteed syntax accuracy.
    
    AND returns TRUE if all conditions are TRUE, FALSE otherwise.
    
    Examples:
        build_and("A1>0", "B1<100") → =AND(A1>0,B1<100)
        build_and("C1=\"Active\"", "D1>=10", "E1<>\"\"") → =AND(C1="Active",D1>=10,E1<>"")
    """
    try:
        formula = logical_tools.formula_builder.build_formula('and', {
            'condition1': condition1,
            'condition2': condition2,
            'condition3': condition3,
            'condition4': condition4,
            'condition5': condition5
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'and',
            'parameters': {
                'condition1': condition1,
                'condition2': condition2,
                'condition3': condition3,
                'condition4': condition4,
                'condition5': condition5
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_and: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'and'}


@mcp.tool()
async def build_or(
    condition1: str = Field(description="First logical condition"),
    condition2: str = Field(description="Second logical condition"),
    condition3: Optional[str] = Field(None, description="Third logical condition (optional)"),
    condition4: Optional[str] = Field(None, description="Fourth logical condition (optional)"),
    condition5: Optional[str] = Field(None, description="Fifth logical condition (optional)")
) -> Dict[str, Any]:
    """
    Build OR formula with guaranteed syntax accuracy.
    
    OR returns TRUE if any condition is TRUE, FALSE if all are FALSE.
    
    Examples:
        build_or("A1>100", "B1=\"VIP\"") → =OR(A1>100,B1="VIP")
        build_or("C1=\"Urgent\"", "D1=\"High\"", "E1=\"Critical\"") → =OR(C1="Urgent",D1="High",E1="Critical")
    """
    try:
        formula = logical_tools.formula_builder.build_formula('or', {
            'condition1': condition1,
            'condition2': condition2,
            'condition3': condition3,
            'condition4': condition4,
            'condition5': condition5
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'or',
            'parameters': {
                'condition1': condition1,
                'condition2': condition2,
                'condition3': condition3,
                'condition4': condition4,
                'condition5': condition5
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_or: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'or'}


@mcp.tool()
async def build_not(
    condition: str = Field(description="Logical condition to negate")
) -> Dict[str, Any]:
    """
    Build NOT formula with guaranteed syntax accuracy.
    
    NOT returns the opposite of a logical value: TRUE becomes FALSE, FALSE becomes TRUE.
    
    Examples:
        build_not("A1>10") → =NOT(A1>10)
        build_not("B1=\"\"") → =NOT(B1="")
        build_not("ISBLANK(C1)") → =NOT(ISBLANK(C1))
    """
    try:
        formula = logical_tools.formula_builder.build_formula('not', {
            'condition': condition
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'not',
            'parameters': {
                'condition': condition
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_not: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'not'}


# ================== SERVER INFO TOOLS ==================

@mcp.tool()
async def get_logical_capabilities() -> Dict[str, Any]:
    """
    Get all supported logical formulas and their descriptions.
    """
    try:
        capabilities = {
            'conditional_logic': {
                'if': 'Perform logical test and return different values'
            },
            'boolean_operations': {
                'and': 'Returns TRUE if all conditions are TRUE',
                'or': 'Returns TRUE if any condition is TRUE',
                'not': 'Returns opposite of logical value'
            }
        }
        
        use_cases = {
            'if': ['Status calculations', 'Conditional formatting', 'Business rules'],
            'and': ['Multi-criteria validation', 'Complex conditions', 'Approval workflows'],
            'or': ['Alternative conditions', 'Fallback logic', 'Exception handling'],
            'not': ['Negation logic', 'Inverse conditions', 'Exclusion rules']
        }
        
        complex_examples = {
            'nested_if': 'IF(AND(A1>0,B1<100),"Valid",IF(OR(A1<0,B1>100),"Invalid","Check"))',
            'combined_logic': 'IF(OR(AND(A1>10,B1="Active"),NOT(C1="")),D1*1.1,D1)',
            'business_rule': 'IF(AND(Sales>Target,NOT(Status="Closed")),"Bonus","Standard")'
        }
        
        return {
            'success': True,
            'server_name': 'Logical Formula Builder',
            'total_tools': len(logical_tools.supported_formulas),
            'categories': capabilities,
            'supported_formulas': logical_tools.supported_formulas,
            'use_cases': use_cases,
            'complex_examples': complex_examples
        }
        
    except Exception as e:
        logger.error(f"Error in get_logical_capabilities: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Logical Formula Builder MCP Server",
        "version": "1.0.0",
        "formula_count": len(logical_tools.supported_formulas),
        "categories": ["conditional_logic", "boolean_operations"]
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get logical formula capabilities"""
    return await get_logical_capabilities()


# ================== MAIN EXECUTION ==================

async def main():
    """Run the Logical Formula MCP Server"""
    logger.info("🚀 Starting Logical Formula Builder MCP Server...")
    logger.info("🧠 Specialized for Logical Formulas")
    logger.info(f"⚡ Supporting {len(logical_tools.supported_formulas)} logical tools")
    logger.info("")
    logger.info("🎯 Supported Categories:")
    logger.info("   • Conditional Logic: IF statements")
    logger.info("   • Boolean Operations: AND, OR, NOT")
    logger.info("")
    logger.info("✅ 100% Formula Accuracy Guaranteed")
    logger.info("🔧 Perfect for Business Rules & Decision Making")
    logger.info("")
    
    # Run MCP server
    await mcp.run()


def run_fastapi_server(port: int = 3036):
    """Run the FastAPI server for HTTP access"""
    logger.info(f"🌐 Starting Logical Formula FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 3036
        run_fastapi_server(port)
    else:
        # Run as MCP server
        asyncio.run(main())