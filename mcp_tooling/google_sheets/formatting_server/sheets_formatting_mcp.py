"""Google Sheets Formatting MCP Server - Handles cell formatting and styling"""
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import json

# Import our API modules
from ..api.auth import GoogleSheetsAuth
from ..api.batch_ops import BatchOperations
from ..api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize MCP server and FastAPI app
mcp = FastMCP("Google Sheets Formatting Server")
app = FastAPI(title="Google Sheets Formatting MCP Server", version="1.0.0")

# Global auth and operations instances
auth = GoogleSheetsAuth(scope_level='full')
batch_ops = None

def get_batch_ops():
    """Get authenticated batch operations instance"""
    global batch_ops
    if batch_ops is None:
        service = auth.authenticate()
        batch_ops = BatchOperations(service)
    return batch_ops


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
    
    async def apply_cell_formatting(self, spreadsheet_id: str, range_spec: str, formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Apply formatting to a cell range - Tool class method"""
        return await apply_cell_formatting(spreadsheet_id, range_spec, formatting)
    
    async def apply_conditional_formatting(self, spreadsheet_id: str, range_spec: str, condition: Dict[str, Any], format_style: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conditional formatting to a range - Tool class method"""
        return await apply_conditional_formatting(spreadsheet_id, range_spec, condition, format_style)
    
    async def create_formatting_preset(self, name: str, formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Create a formatting preset for reuse - Tool class method"""
        return await create_formatting_preset(name, formatting)


# Global instance
format_tools = GoogleSheetsFormatTools()

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def apply_cell_formatting(
    spreadsheet_id: str,
    range_spec: str,
    formatting: Dict[str, Any]
) -> Dict[str, Any]:
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
        from ..api.range_resolver import RangeResolver
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
        
        ops = format_tools.get_batch_ops()
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

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def apply_conditional_formatting(
    spreadsheet_id: str,
    range_spec: str,
    condition: Dict[str, Any],
    format_style: Dict[str, Any]
) -> Dict[str, Any]:
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
        from ..api.range_resolver import RangeResolver
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
        
        ops = format_tools.get_batch_ops()
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

@mcp.tool()
async def create_formatting_preset(
    name: str,
    formatting: Dict[str, Any]
) -> Dict[str, Any]:
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

# ================== FASTAPI ENDPOINTS ==================

class MCPRequest(BaseModel):
    """Request model for MCP operations."""
    method: str = Field(..., description="MCP method to call")
    params: Dict[str, Any] = Field(default={}, description="Parameters for the method")

@app.post("/")
async def handle_mcp_request(request: MCPRequest) -> Dict[str, Any]:
    """
    Handle MCP requests via FastAPI endpoint.
    
    This endpoint processes MCP tool calls and returns appropriate responses.
    """
    try:
        if request.method == "tools/list":
            # Return list of available tools
            tools = []
            for tool_name, tool_func in mcp._tools.items():
                tool_info = {
                    "name": tool_name,
                    "description": tool_func.__doc__ or f"Tool: {tool_name}",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                tools.append(tool_info)
            
            return {"tools": tools}
            
        elif request.method == "tools/call":
            # Call a specific tool
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            if tool_name not in mcp._tools:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
            # Call the tool
            result = await mcp._tools[tool_name](**arguments)
            
            # Format response according to MCP protocol
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result)
                    }
                ]
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
            
    except Exception as e:
        logger.error(f"Error handling MCP request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Google Sheets Formatting MCP Server"}

# Main function to run the server
def main():
    """Run the FastAPI server."""
    import sys
    
    port = 3013  # Default port for formatting server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Google Sheets Formatting MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

# Export the mcp instance and app
__all__ = ['mcp', 'app']