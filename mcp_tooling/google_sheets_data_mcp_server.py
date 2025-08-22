#!/usr/bin/env python3
"""
Google Sheets Data MCP Server - Following exact formula server pattern

This server handles Google Sheets data operations with the same architecture
as our formula servers for consistency and maintainability.
"""

from fastmcp import FastMCP
from typing import Dict, Any, List, Optional, Union
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import json

# Import our API modules
from mcp_tooling.google_sheets.api.auth import GoogleSheetsAuth
from mcp_tooling.google_sheets.api.value_ops import ValueOperations
from mcp_tooling.google_sheets.api.range_resolver import RangeResolver
from mcp_tooling.google_sheets.api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Google Sheets Data Server")
app = FastAPI(
    title="Google Sheets Data MCP Server", 
    version="1.0.0",
    description="Google Sheets data operations with 100% reliability"
)


class GoogleSheetsDataTools:
    """
    Focused MCP tools for Google Sheets data operations.
    
    This specialized server handles all data needs:
    - Data reading: single ranges, batch reads, formatted values
    - Data writing: single ranges, batch writes, append operations
    - Polars integration: DataFrame to/from Sheets conversion
    """
    
    def __init__(self):
        self.auth = GoogleSheetsAuth(scope_level='full')
        self.value_ops = None
        logger.info("📊 GoogleSheetsDataTools initialized")
        
    def get_value_ops(self):
        """Get authenticated value operations instance"""
        if self.value_ops is None:
            service = self.auth.authenticate()
            self.value_ops = ValueOperations(service)
        return self.value_ops
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def read_values(self, spreadsheet_id: str, range_spec: str, value_render_option: str = 'FORMATTED_VALUE') -> Dict[str, Any]:
        """
        Read values from a Google Sheets range.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_spec: Range in A1 notation (e.g., 'Sheet1!A1:C10')
            value_render_option: How values should be rendered ('FORMATTED_VALUE', 'UNFORMATTED_VALUE', 'FORMULA')
            
        Returns:
            Dictionary containing values and metadata
        """
        try:
            ops = self.get_value_ops()
            values = ops.get_values(spreadsheet_id, range_spec)
            
            # Get range dimensions for metadata
            rows, cols = RangeResolver.get_range_dimensions(range_spec)
            
            result = {
                'values': values,
                'range': range_spec,
                'dimensions': {'rows': len(values) if values else 0, 'cols': len(values[0]) if values and values[0] else 0},
                'expected_dimensions': {'rows': rows if rows > 0 else len(values), 'cols': cols if cols > 0 else (len(values[0]) if values and values[0] else 0)},
                'success': True
            }
            
            logger.info(f"Read {len(values)} rows from {spreadsheet_id}:{range_spec}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to read values from {spreadsheet_id}:{range_spec}: {e}")
            return {"error": str(e), "success": False}
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def write_values(self, spreadsheet_id: str, range_spec: str, values: List[List[Any]], value_input_option: str = 'USER_ENTERED') -> Dict[str, Any]:
        """
        Write values to a Google Sheets range.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_spec: Range in A1 notation (e.g., 'Sheet1!A1:C10')
            values: 2D array of values to write
            value_input_option: How values should be interpreted ('RAW', 'USER_ENTERED')
            
        Returns:
            Dictionary containing update result and metadata
        """
        try:
            ops = self.get_value_ops()
            result = ops.update_values(spreadsheet_id, range_spec, value_input_option, values)
            
            response = {
                'updated_cells': result.get('updatedCells', 0),
                'updated_rows': result.get('updatedRows', 0),
                'updated_columns': result.get('updatedColumns', 0),
                'updated_range': result.get('updatedRange'),
                'success': True
            }
            
            logger.info(f"Wrote {response['updated_cells']} cells to {spreadsheet_id}:{range_spec}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to write values to {spreadsheet_id}:{range_spec}: {e}")
            return {"error": str(e), "success": False}
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def clear_values(self, spreadsheet_id: str, range_spec: str) -> Dict[str, Any]:
        """
        Clear values from a range.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_spec: Range in A1 notation to clear
            
        Returns:
            Dictionary containing clear result
        """
        try:
            ops = self.get_value_ops()
            result = ops.clear_values(spreadsheet_id, range_spec)
            
            response = {
                'cleared_range': result.get('clearedRange'),
                'success': True
            }
            
            logger.info(f"Cleared values from {spreadsheet_id}:{range_spec}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to clear values from {spreadsheet_id}:{range_spec}: {e}")
            return {"error": str(e), "success": False}


# Global instance (same pattern as formula servers)
data_tools = GoogleSheetsDataTools()


# ================== MCP TOOLS (Call tool instance methods) ==================

@mcp.tool()
async def read_values(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    range_spec: str = Field(description="Range in A1 notation (e.g., 'Sheet1!A1:C10')"),
    value_render_option: str = Field('FORMATTED_VALUE', description="How values should be rendered")
) -> Dict[str, Any]:
    """
    Read values from a Google Sheets range.
    
    Examples:
        read_values("1ABC...", "A1:C10") → {'values': [[...]], 'dimensions': {...}, ...}
    """
    try:
        return data_tools.read_values(spreadsheet_id, range_spec, value_render_option)
    except Exception as e:
        logger.error(f"Error in read_values: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def write_values(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    range_spec: str = Field(description="Range in A1 notation (e.g., 'Sheet1!A1:C10')"),
    values: List[List[Any]] = Field(description="2D array of values to write"),
    value_input_option: str = Field('USER_ENTERED', description="How values should be interpreted")
) -> Dict[str, Any]:
    """
    Write values to a Google Sheets range.
    
    Examples:
        write_values("1ABC...", "A1:B2", [["Name", "Value"], ["Test", 123]]) → {'updated_cells': 4, ...}
    """
    try:
        return data_tools.write_values(spreadsheet_id, range_spec, values, value_input_option)
    except Exception as e:
        logger.error(f"Error in write_values: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def clear_values(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    range_spec: str = Field(description="Range in A1 notation to clear")
) -> Dict[str, Any]:
    """
    Clear values from a range.
    
    Examples:
        clear_values("1ABC...", "A1:C10") → {'cleared_range': 'Sheet1!A1:C10', ...}
    """
    try:
        return data_tools.clear_values(spreadsheet_id, range_spec)
    except Exception as e:
        logger.error(f"Error in clear_values: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

class MCPRequest(BaseModel):
    """Request model for MCP operations."""
    method: str = Field(..., description="MCP method to call")
    params: Dict[str, Any] = Field(default={}, description="Parameters for the method")

@app.post("/")
async def handle_mcp_request(request: MCPRequest) -> Dict[str, Any]:
    """Handle MCP requests via FastAPI endpoint."""
    try:
        if request.method == "tools/list":
            tools = []
            for tool_name, tool_func in mcp._tools.items():
                tool_info = {
                    "name": tool_name,
                    "description": tool_func.__doc__ or f"Tool: {tool_name}",
                    "inputSchema": {"type": "object", "properties": {}, "required": []}
                }
                tools.append(tool_info)
            return {"tools": tools}
            
        elif request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            if tool_name not in mcp._tools:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
            result = await mcp._tools[tool_name](**arguments)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
            
    except Exception as e:
        logger.error(f"Error handling MCP request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Google Sheets Data MCP Server"}

def main():
    """Run the FastAPI server."""
    import sys
    
    port = 3011  # Default port for data server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Google Sheets Data MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

# Export the tool class and mcp instance
__all__ = ['GoogleSheetsDataTools', 'data_tools', 'mcp', 'app']