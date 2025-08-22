#!/usr/bin/env python3
"""
Text Formula MCP Server - Phase 0 Category-Wise Implementation

This focused MCP server provides text formulas with 100% syntax accuracy:
- Text Extraction: LEFT, RIGHT, MID, LEN
- Text Manipulation: CONCATENATE, UPPER, LOWER, TRIM
- String Operations: Text processing and formatting

Key Benefits:
- Focused toolset for AI agents (8 text tools)
- 100% Formula Accuracy guarantee
- Business-parameter interface
- Specialized for text operations

Usage:
    # As MCP Server
    uv run python text_formula_mcp_server.py
    
    # As FastAPI Server  
    uv run python text_formula_mcp_server.py --port 3035
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
mcp = FastMCP("Text Formula Builder Server")
app = FastAPI(
    title="Text Formula Builder MCP Server", 
    version="1.0.0",
    description="Phase 0 Text Formulas with 100% syntax accuracy"
)


class TextFormulaTools:
    """
    Focused MCP tools for text formulas only.
    
    This specialized server handles all text processing needs:
    - Text extraction (LEFT, RIGHT, MID, LEN)
    - Text manipulation (CONCATENATE, UPPER, LOWER, TRIM)
    - String operations and formatting
    """
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        logger.info("📝 TextFormulaTools initialized")
        
        # Get only text formulas
        all_formulas = self.formula_builder.get_supported_formulas()
        self.supported_formulas = [f for f in all_formulas if f in [
            'concatenate', 'left', 'right', 'mid', 'len', 'upper', 'lower', 'trim'
        ]]
        
        logger.info(f"🔤 Supporting {len(self.supported_formulas)} text formulas")


# Global instance
text_tools = TextFormulaTools()


# ================== TEXT EXTRACTION TOOLS ==================

@mcp.tool()
async def build_left(
    text: str = Field(description="Text string or cell reference"),
    num_chars: int = Field(description="Number of characters to extract from left")
) -> Dict[str, Any]:
    """
    Build LEFT formula with guaranteed syntax accuracy.
    
    LEFT extracts a specified number of characters from the left side of text.
    
    Examples:
        build_left("A1", 5) → =LEFT(A1,5)
        build_left("Hello World", 5) → =LEFT("Hello World",5)
    """
    try:
        formula = text_tools.formula_builder.build_formula('left', {
            'text': text,
            'num_chars': num_chars
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'left',
            'parameters': {
                'text': text,
                'num_chars': num_chars
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_left: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'left'}


@mcp.tool()
async def build_right(
    text: str = Field(description="Text string or cell reference"),
    num_chars: int = Field(description="Number of characters to extract from right")
) -> Dict[str, Any]:
    """
    Build RIGHT formula with guaranteed syntax accuracy.
    
    RIGHT extracts a specified number of characters from the right side of text.
    
    Examples:
        build_right("A1", 3) → =RIGHT(A1,3)
        build_right("Hello World", 5) → =RIGHT("Hello World",5)
    """
    try:
        formula = text_tools.formula_builder.build_formula('right', {
            'text': text,
            'num_chars': num_chars
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'right',
            'parameters': {
                'text': text,
                'num_chars': num_chars
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_right: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'right'}


@mcp.tool()
async def build_mid(
    text: str = Field(description="Text string or cell reference"),
    start_pos: int = Field(description="Starting position (1-based)"),
    num_chars: int = Field(description="Number of characters to extract")
) -> Dict[str, Any]:
    """
    Build MID formula with guaranteed syntax accuracy.
    
    MID extracts characters from the middle of text starting at a specific position.
    
    Examples:
        build_mid("A1", 3, 5) → =MID(A1,3,5)
        build_mid("Hello World", 7, 5) → =MID("Hello World",7,5)
    """
    try:
        formula = text_tools.formula_builder.build_formula('mid', {
            'text': text,
            'start_pos': start_pos,
            'num_chars': num_chars
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'mid',
            'parameters': {
                'text': text,
                'start_pos': start_pos,
                'num_chars': num_chars
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_mid: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'mid'}


@mcp.tool()
async def build_len(
    text: str = Field(description="Text string or cell reference")
) -> Dict[str, Any]:
    """
    Build LEN formula with guaranteed syntax accuracy.
    
    LEN returns the number of characters in a text string.
    
    Examples:
        build_len("A1") → =LEN(A1)
        build_len("Hello World") → =LEN("Hello World")
    """
    try:
        formula = text_tools.formula_builder.build_formula('len', {
            'text': text
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'len',
            'parameters': {
                'text': text
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_len: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'len'}


# ================== TEXT MANIPULATION TOOLS ==================

@mcp.tool()
async def build_concatenate(
    text1: str = Field(description="First text string or cell reference"),
    text2: str = Field(description="Second text string or cell reference"),
    text3: Optional[str] = Field(None, description="Third text string (optional)"),
    text4: Optional[str] = Field(None, description="Fourth text string (optional)"),
    text5: Optional[str] = Field(None, description="Fifth text string (optional)")
) -> Dict[str, Any]:
    """
    Build CONCATENATE formula with guaranteed syntax accuracy.
    
    CONCATENATE joins multiple text strings into one string.
    
    Examples:
        build_concatenate("A1", "B1") → =CONCATENATE(A1,B1)
        build_concatenate("Hello", " ", "World") → =CONCATENATE("Hello"," ","World")
    """
    try:
        formula = text_tools.formula_builder.build_formula('concatenate', {
            'text1': text1,
            'text2': text2,
            'text3': text3,
            'text4': text4,
            'text5': text5
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'concatenate',
            'parameters': {
                'text1': text1,
                'text2': text2,
                'text3': text3,
                'text4': text4,
                'text5': text5
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_concatenate: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'concatenate'}


@mcp.tool()
async def build_upper(
    text: str = Field(description="Text string or cell reference to convert to uppercase")
) -> Dict[str, Any]:
    """
    Build UPPER formula with guaranteed syntax accuracy.
    
    UPPER converts text to uppercase letters.
    
    Examples:
        build_upper("A1") → =UPPER(A1)
        build_upper("hello world") → =UPPER("hello world")
    """
    try:
        formula = text_tools.formula_builder.build_formula('upper', {
            'text': text
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'upper',
            'parameters': {
                'text': text
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_upper: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'upper'}


@mcp.tool()
async def build_lower(
    text: str = Field(description="Text string or cell reference to convert to lowercase")
) -> Dict[str, Any]:
    """
    Build LOWER formula with guaranteed syntax accuracy.
    
    LOWER converts text to lowercase letters.
    
    Examples:
        build_lower("A1") → =LOWER(A1)
        build_lower("HELLO WORLD") → =LOWER("HELLO WORLD")
    """
    try:
        formula = text_tools.formula_builder.build_formula('lower', {
            'text': text
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'lower',
            'parameters': {
                'text': text
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_lower: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'lower'}


@mcp.tool()
async def build_trim(
    text: str = Field(description="Text string or cell reference to trim whitespace")
) -> Dict[str, Any]:
    """
    Build TRIM formula with guaranteed syntax accuracy.
    
    TRIM removes leading and trailing spaces from text.
    
    Examples:
        build_trim("A1") → =TRIM(A1)
        build_trim("  hello world  ") → =TRIM("  hello world  ")
    """
    try:
        formula = text_tools.formula_builder.build_formula('trim', {
            'text': text
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'trim',
            'parameters': {
                'text': text
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_trim: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'trim'}


# ================== SERVER INFO TOOLS ==================

@mcp.tool()
async def get_text_capabilities() -> Dict[str, Any]:
    """
    Get all supported text formulas and their descriptions.
    """
    try:
        capabilities = {
            'text_extraction': {
                'left': 'Extract characters from left side of text',
                'right': 'Extract characters from right side of text',
                'mid': 'Extract characters from middle of text',
                'len': 'Get length of text string'
            },
            'text_manipulation': {
                'concatenate': 'Join multiple text strings together',
                'upper': 'Convert text to uppercase',
                'lower': 'Convert text to lowercase',
                'trim': 'Remove leading and trailing spaces'
            }
        }
        
        use_cases = {
            'concatenate': ['Name formatting', 'Address building', 'Report titles'],
            'left': ['Code extraction', 'Prefix isolation', 'Data parsing'],
            'upper': ['Data standardization', 'Header formatting', 'Comparison normalization'],
            'trim': ['Data cleaning', 'Import processing', 'Space removal']
        }
        
        return {
            'success': True,
            'server_name': 'Text Formula Builder',
            'total_tools': len(text_tools.supported_formulas),
            'categories': capabilities,
            'supported_formulas': text_tools.supported_formulas,
            'use_cases': use_cases
        }
        
    except Exception as e:
        logger.error(f"Error in get_text_capabilities: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Text Formula Builder MCP Server",
        "version": "1.0.0",
        "formula_count": len(text_tools.supported_formulas),
        "categories": ["text_extraction", "text_manipulation"]
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get text formula capabilities"""
    return await get_text_capabilities()


# ================== MAIN EXECUTION ==================

async def main():
    """Run the Text Formula MCP Server"""
    logger.info("🚀 Starting Text Formula Builder MCP Server...")
    logger.info("📝 Specialized for Text Formulas")
    logger.info(f"🔤 Supporting {len(text_tools.supported_formulas)} text tools")
    logger.info("")
    logger.info("🎯 Supported Categories:")
    logger.info("   • Text Extraction: LEFT, RIGHT, MID, LEN")
    logger.info("   • Text Manipulation: CONCATENATE, UPPER, LOWER, TRIM")
    logger.info("")
    logger.info("✅ 100% Formula Accuracy Guaranteed")
    logger.info("")
    
    # Run MCP server
    await mcp.run()


def run_fastapi_server(port: int = 3035):
    """Run the FastAPI server for HTTP access"""
    logger.info(f"🌐 Starting Text Formula FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 3035
        run_fastapi_server(port)
    else:
        # Run as MCP server
        asyncio.run(main())