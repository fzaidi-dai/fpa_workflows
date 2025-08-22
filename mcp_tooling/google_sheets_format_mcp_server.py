#!/usr/bin/env python3
"""
Google Sheets Format MCP Server - Following exact formula server pattern

This server handles Google Sheets formatting operations with the same architecture
as our formula servers for consistency and maintainability.
"""

from fastmcp import FastMCP
from typing import Dict, Any, List, Optional
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import json

# Import our API modules
from mcp_tooling.google_sheets.api.auth import GoogleSheetsAuth
from mcp_tooling.google_sheets.api.batch_ops import BatchOperations
from mcp_tooling.google_sheets.api.range_resolver import RangeResolver
from mcp_tooling.google_sheets.api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Google Sheets Format Server")
app = FastAPI(
    title="Google Sheets Format MCP Server", 
    version="1.0.0",
    description="Google Sheets formatting operations with 100% reliability"
)


class GoogleSheetsFormatTools:
    """
    Focused MCP tools for Google Sheets formatting operations.
    
    This specialized server handles all formatting needs:
    - Cell formatting: background colors, text formatting, borders
    - Conditional formatting: highlight cells based on conditions
    - Format presets: reusable formatting configurations
    """
    
    def __init__(self):
        self.auth = GoogleSheetsAuth(scope_level='full')
        self.batch_ops = None
        logger.info("🎨 GoogleSheetsFormatTools initialized")
        
    def get_batch_ops(self):
        """Get authenticated batch operations instance"""
        if self.batch_ops is None:
            service = self.auth.authenticate()
            self.batch_ops = BatchOperations(service)
        return self.batch_ops
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def apply_cell_formatting(self, spreadsheet_id: str, range_spec: str, formatting: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply formatting to a cell range.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_spec: Range in A1 notation to format
            formatting: Formatting options (backgroundColor, textFormat, etc.)
            
        Returns:
            Dictionary containing formatting result
        """
        try:
            # Convert range to batch update format
            parsed_range = RangeResolver.parse_a1_notation(range_spec)
            
            # Build formatting request
            request = {
                "repeatCell": {
                    "range": {
                        "sheetId": 0,  # TODO: Get actual sheet ID
                        "startRowIndex": parsed_range.start_row,
                        "endRowIndex": parsed_range.end_row + 1 if parsed_range.end_row is not None else parsed_range.start_row + 1,
                        "startColumnIndex": parsed_range.start_col,
                        "endColumnIndex": parsed_range.end_col + 1 if parsed_range.end_col is not None else parsed_range.start_col + 1
                    },
                    "cell": {
                        "userEnteredFormat": formatting
                    },
                    "fields": "userEnteredFormat"
                }
            }
            
            ops = self.get_batch_ops()
            result = ops.batch_update(spreadsheet_id, [request])
            
            response = {
                'range': range_spec,
                'formatting_applied': formatting,
                'success': True
            }
            
            logger.info(f"Applied formatting to {spreadsheet_id}:{range_spec}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to apply formatting to {spreadsheet_id}:{range_spec}: {e}")
            return {"error": str(e), "success": False}
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def apply_conditional_formatting(self, spreadsheet_id: str, range_spec: str, condition: Dict[str, Any], format_style: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply conditional formatting to a range.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_spec: Range in A1 notation to apply conditional formatting
            condition: Condition for formatting (e.g., {'type': 'GREATER_THAN', 'value': 100})
            format_style: Format to apply when condition is met
            
        Returns:
            Dictionary containing conditional formatting result
        """
        try:
            parsed_range = RangeResolver.parse_a1_notation(range_spec)
            
            request = {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [{
                            "sheetId": 0,
                            "startRowIndex": parsed_range.start_row,
                            "endRowIndex": parsed_range.end_row + 1 if parsed_range.end_row is not None else parsed_range.start_row + 1,
                            "startColumnIndex": parsed_range.start_col,
                            "endColumnIndex": parsed_range.end_col + 1 if parsed_range.end_col is not None else parsed_range.start_col + 1
                        }],
                        "booleanRule": {
                            "condition": condition,
                            "format": format_style
                        }
                    },
                    "index": 0
                }
            }
            
            ops = self.get_batch_ops()
            result = ops.batch_update(spreadsheet_id, [request])
            
            response = {
                'range': range_spec,
                'condition': condition,
                'format_style': format_style,
                'success': True
            }
            
            logger.info(f"Applied conditional formatting to {spreadsheet_id}:{range_spec}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to apply conditional formatting to {spreadsheet_id}:{range_spec}: {e}")
            return {"error": str(e), "success": False}
    
    def create_formatting_preset(self, name: str, formatting: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a formatting preset for reuse.
        
        Args:
            name: Name of the preset
            formatting: Formatting configuration
            
        Returns:
            Dictionary containing preset information
        """
        # This would typically be stored in a database or file
        # For now, just return the preset definition
        preset = {
            'name': name,
            'formatting': formatting,
            'created': True
        }
        
        logger.info(f"Created formatting preset '{name}'")
        return preset


# Global instance (same pattern as formula servers)
format_tools = GoogleSheetsFormatTools()


# ================== MCP TOOLS (Call tool instance methods) ==================

@mcp.tool()
async def apply_cell_formatting(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    range_spec: str = Field(description="Range in A1 notation to format"),
    formatting: Dict[str, Any] = Field(description="Formatting options (backgroundColor, textFormat, etc.)")
) -> Dict[str, Any]:
    """
    Apply formatting to a cell range.
    
    Examples:
        apply_cell_formatting("1ABC...", "A1:B5", {"backgroundColor": {"red": 1.0}})
    """
    try:
        return format_tools.apply_cell_formatting(spreadsheet_id, range_spec, formatting)
    except Exception as e:
        logger.error(f"Error in apply_cell_formatting: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def apply_conditional_formatting(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    range_spec: str = Field(description="Range in A1 notation to apply conditional formatting"),
    condition: Dict[str, Any] = Field(description="Condition for formatting"),
    format_style: Dict[str, Any] = Field(description="Format to apply when condition is met")
) -> Dict[str, Any]:
    """
    Apply conditional formatting to a range.
    
    Examples:
        apply_conditional_formatting("1ABC...", "A1:A10", {"type": "NUMBER_GREATER", "values": [{"userEnteredValue": "10"}]}, {"backgroundColor": {"green": 1.0}})
    """
    try:
        return format_tools.apply_conditional_formatting(spreadsheet_id, range_spec, condition, format_style)
    except Exception as e:
        logger.error(f"Error in apply_conditional_formatting: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def create_formatting_preset(
    name: str = Field(description="Name of the preset"),
    formatting: Dict[str, Any] = Field(description="Formatting configuration")
) -> Dict[str, Any]:
    """
    Create a formatting preset for reuse.
    
    Examples:
        create_formatting_preset("header_style", {"backgroundColor": {"blue": 0.8}, "textFormat": {"bold": True}})
    """
    try:
        return format_tools.create_formatting_preset(name, formatting)
    except Exception as e:
        logger.error(f"Error in create_formatting_preset: {e}")
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
    return {"status": "healthy", "service": "Google Sheets Format MCP Server"}

def main():
    """Run the FastAPI server."""
    import sys
    
    port = 3013  # Default port for format server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Google Sheets Format MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

# Export the tool class and mcp instance
__all__ = ['GoogleSheetsFormatTools', 'format_tools', 'mcp', 'app']