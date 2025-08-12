#!/usr/bin/env python3
"""
Text Functions MCP Server

This MCP server provides Google Sheets text manipulation functions through the Model Context Protocol.
Functions include CONCATENATE, LEFT, RIGHT, MID, LEN, UPPER, LOWER, TRIM, SUBSTITUTE with full compatibility to Google Sheets.
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
mcp = FastMCP("Text Functions Server")
app = FastAPI(title="Text Functions MCP Server", version="1.0.0")

# ================== TEXT MANIPULATION FUNCTIONS ==================

@mcp.tool()
async def concatenate_tool(
    texts: List[str] = Field(description="List of texts to concatenate")
) -> Dict[str, Any]:
    """
    CONCATENATE function matching Google Sheets =CONCATENATE(text1, [text2, ...]).
    
    Examples:
        concatenate_tool(["Hello", " ", "World"]) - Joins texts together
        concatenate_tool(["First", "Second", "Third"]) - Concatenates multiple strings
    """
    try:
        result = sheets_funcs.CONCATENATE(*texts)
        formula = f"=CONCATENATE({', '.join(repr(t) for t in texts)})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "input_texts": texts
        }
    except Exception as e:
        logger.error(f"Error in concatenate_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def left_tool(
    text: str = Field(description="Text to extract from"),
    num_chars: int = Field(description="Number of characters to extract from left")
) -> Dict[str, Any]:
    """
    LEFT function matching Google Sheets =LEFT(text, num_chars).
    
    Examples:
        left_tool("Hello World", 5) - Returns "Hello"
        left_tool("Financial Planning", 9) - Returns "Financial"
    """
    try:
        result = sheets_funcs.LEFT(text, num_chars)
        formula = f"=LEFT(\"{text}\", {num_chars})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text,
            "num_chars": num_chars
        }
    except Exception as e:
        logger.error(f"Error in left_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def right_tool(
    text: str = Field(description="Text to extract from"),
    num_chars: int = Field(description="Number of characters to extract from right")
) -> Dict[str, Any]:
    """
    RIGHT function matching Google Sheets =RIGHT(text, num_chars).
    
    Examples:
        right_tool("Hello World", 5) - Returns "World"
        right_tool("document.pdf", 3) - Returns "pdf"
    """
    try:
        result = sheets_funcs.RIGHT(text, num_chars)
        formula = f"=RIGHT(\"{text}\", {num_chars})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text,
            "num_chars": num_chars
        }
    except Exception as e:
        logger.error(f"Error in right_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def mid_tool(
    text: str = Field(description="Text to extract from"),
    start_num: int = Field(description="Starting position (1-based)"),
    num_chars: int = Field(description="Number of characters to extract")
) -> Dict[str, Any]:
    """
    MID function matching Google Sheets =MID(text, start_num, num_chars).
    
    Examples:
        mid_tool("Hello World", 7, 5) - Returns "World"
        mid_tool("Financial Analysis", 11, 8) - Returns "Analysis"
    """
    try:
        result = sheets_funcs.MID(text, start_num, num_chars)
        formula = f"=MID(\"{text}\", {start_num}, {num_chars})"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text,
            "start_position": start_num,
            "num_chars": num_chars
        }
    except Exception as e:
        logger.error(f"Error in mid_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def len_tool(
    text: str = Field(description="Text to measure length")
) -> Dict[str, Any]:
    """
    LEN function matching Google Sheets =LEN(text).
    
    Examples:
        len_tool("Hello") - Returns 5
        len_tool("Financial Planning Analysis") - Returns 26
    """
    try:
        result = sheets_funcs.LEN(text)
        formula = f"=LEN(\"{text}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "text": text
        }
    except Exception as e:
        logger.error(f"Error in len_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def upper_tool(
    text: str = Field(description="Text to convert to uppercase")
) -> Dict[str, Any]:
    """
    UPPER function matching Google Sheets =UPPER(text).
    
    Examples:
        upper_tool("hello world") - Returns "HELLO WORLD"
        upper_tool("Financial Analysis") - Returns "FINANCIAL ANALYSIS"
    """
    try:
        result = sheets_funcs.UPPER(text)
        formula = f"=UPPER(\"{text}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text
        }
    except Exception as e:
        logger.error(f"Error in upper_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def lower_tool(
    text: str = Field(description="Text to convert to lowercase")
) -> Dict[str, Any]:
    """
    LOWER function matching Google Sheets =LOWER(text).
    
    Examples:
        lower_tool("HELLO WORLD") - Returns "hello world"
        lower_tool("Financial Analysis") - Returns "financial analysis"
    """
    try:
        result = sheets_funcs.LOWER(text)
        formula = f"=LOWER(\"{text}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text
        }
    except Exception as e:
        logger.error(f"Error in lower_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def proper_tool(
    text: str = Field(description="Text to convert to proper case (title case)")
) -> Dict[str, Any]:
    """
    PROPER function matching Google Sheets =PROPER(text).
    
    Examples:
        proper_tool("hello world") - Returns "Hello World"
        proper_tool("FINANCIAL ANALYSIS") - Returns "Financial Analysis"
    """
    try:
        result = sheets_funcs.PROPER(text)
        formula = f"=PROPER(\"{text}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text
        }
    except Exception as e:
        logger.error(f"Error in proper_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def trim_tool(
    text: str = Field(description="Text to remove extra spaces from")
) -> Dict[str, Any]:
    """
    TRIM function matching Google Sheets =TRIM(text).
    Removes leading, trailing, and extra internal spaces.
    
    Examples:
        trim_tool("  Hello   World  ") - Returns "Hello World"
        trim_tool("Financial    Analysis") - Returns "Financial Analysis"
    """
    try:
        result = sheets_funcs.TRIM(text)
        formula = f"=TRIM(\"{text}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text
        }
    except Exception as e:
        logger.error(f"Error in trim_tool: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def substitute_tool(
    text: str = Field(description="Original text"),
    old_text: str = Field(description="Text to replace"),
    new_text: str = Field(description="Replacement text"),
    instance_num: Optional[int] = Field(None, description="Which occurrence to replace (optional)")
) -> Dict[str, Any]:
    """
    SUBSTITUTE function matching Google Sheets =SUBSTITUTE(text, old_text, new_text, [instance_num]).
    
    Examples:
        substitute_tool("Hello World", "World", "Universe") - Returns "Hello Universe"
        substitute_tool("apple apple orange", "apple", "banana", 1) - Returns "banana apple orange"
        substitute_tool("Financial Analysis Report", "Analysis", "Planning") - Returns "Financial Planning Report"
    """
    try:
        result = sheets_funcs.SUBSTITUTE(text, old_text, new_text, instance_num)
        
        if instance_num:
            formula = f"=SUBSTITUTE(\"{text}\", \"{old_text}\", \"{new_text}\", {instance_num})"
        else:
            formula = f"=SUBSTITUTE(\"{text}\", \"{old_text}\", \"{new_text}\")"
        
        return {
            "success": True,
            "result": result,
            "formula": formula,
            "original_text": text,
            "old_text": old_text,
            "new_text": new_text,
            "instance_num": instance_num
        }
    except Exception as e:
        logger.error(f"Error in substitute_tool: {e}")
        return {"success": False, "error": str(e)}

# ================== API MODELS ==================

class ConcatenateRequest(BaseModel):
    texts: List[str]

class LeftRequest(BaseModel):
    text: str
    num_chars: int

class RightRequest(BaseModel):
    text: str
    num_chars: int

class MidRequest(BaseModel):
    text: str
    start_num: int
    num_chars: int

class LenRequest(BaseModel):
    text: str

class CaseRequest(BaseModel):
    text: str

class SubstituteRequest(BaseModel):
    text: str
    old_text: str
    new_text: str
    instance_num: Optional[int] = None

class TextResponse(BaseModel):
    success: bool
    result: Optional[Union[str, int]] = None
    formula: Optional[str] = None
    error: Optional[str] = None

# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Text Functions MCP Server"}

@app.post("/concatenate", response_model=TextResponse)
async def api_concatenate(request: ConcatenateRequest):
    """CONCATENATE function via API."""
    result = await concatenate_tool(*request.texts)
    return TextResponse(**result)

@app.post("/left", response_model=TextResponse)
async def api_left(request: LeftRequest):
    """LEFT function via API."""
    result = await left_tool(request.text, request.num_chars)
    return TextResponse(**result)

@app.post("/right", response_model=TextResponse)
async def api_right(request: RightRequest):
    """RIGHT function via API."""
    result = await right_tool(request.text, request.num_chars)
    return TextResponse(**result)

@app.post("/mid", response_model=TextResponse)
async def api_mid(request: MidRequest):
    """MID function via API."""
    result = await mid_tool(request.text, request.start_num, request.num_chars)
    return TextResponse(**result)

@app.post("/len", response_model=TextResponse)
async def api_len(request: LenRequest):
    """LEN function via API."""
    result = await len_tool(request.text)
    return TextResponse(**result)

@app.post("/upper", response_model=TextResponse)
async def api_upper(request: CaseRequest):
    """UPPER function via API."""
    result = await upper_tool(request.text)
    return TextResponse(**result)

@app.post("/lower", response_model=TextResponse)
async def api_lower(request: CaseRequest):
    """LOWER function via API."""
    result = await lower_tool(request.text)
    return TextResponse(**result)

@app.post("/proper", response_model=TextResponse)
async def api_proper(request: CaseRequest):
    """PROPER function via API."""
    result = await proper_tool(request.text)
    return TextResponse(**result)

@app.post("/trim", response_model=TextResponse)
async def api_trim(request: CaseRequest):
    """TRIM function via API."""
    result = await trim_tool(request.text)
    return TextResponse(**result)

@app.post("/substitute", response_model=TextResponse)
async def api_substitute(request: SubstituteRequest):
    """SUBSTITUTE function via API."""
    result = await substitute_tool(
        request.text, 
        request.old_text, 
        request.new_text, 
        request.instance_num
    )
    return TextResponse(**result)

@app.get("/functions")
async def list_functions():
    """List all available text functions."""
    functions = [
        "CONCATENATE", "LEFT", "RIGHT", "MID", "LEN", 
        "UPPER", "LOWER", "PROPER", "TRIM", "SUBSTITUTE"
    ]
    return {"functions": functions, "count": len(functions)}

@app.get("/examples")
async def function_examples():
    """Get usage examples for text functions."""
    return {
        "concatenate": "concatenate_tool('Hello', ' ', 'World')",
        "left": "left_tool('Financial Analysis', 9)",
        "right": "right_tool('document.pdf', 3)",
        "mid": "mid_tool('Hello World', 7, 5)",
        "len": "len_tool('Financial Planning')",
        "upper": "upper_tool('hello world')",
        "lower": "lower_tool('HELLO WORLD')",
        "proper": "proper_tool('financial analysis')",
        "trim": "trim_tool('  extra   spaces  ')",
        "substitute": "substitute_tool('apple pie', 'apple', 'cherry')"
    }

# ================== MAIN EXECUTION ==================

async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8007, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Text Functions MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8007")
    print("API Docs: http://localhost:8007/docs")
    print("\nAvailable functions:")
    print("- CONCATENATE: Join text strings together")
    print("- LEFT/RIGHT: Extract characters from left/right")
    print("- MID: Extract characters from middle")
    print("- LEN: Get text length")
    print("- UPPER/LOWER/PROPER: Change text case")
    print("- TRIM: Remove extra spaces")
    print("- SUBSTITUTE: Replace text occurrences")
    asyncio.run(main())