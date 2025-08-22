#!/usr/bin/env python3
"""
Google Sheets Structure MCP Server - Following exact formula server pattern

This server handles Google Sheets structure operations with the same architecture
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
from mcp_tooling.google_sheets.api.spreadsheet_ops import SpreadsheetOperations
from mcp_tooling.google_sheets.api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Google Sheets Structure Server")
app = FastAPI(
    title="Google Sheets Structure MCP Server", 
    version="1.0.0",
    description="Google Sheets structure operations with 100% reliability"
)


class GoogleSheetsStructureTools:
    """
    Focused MCP tools for Google Sheets structure operations.
    
    This specialized server handles all structure needs:
    - Spreadsheet management: create, get info, metadata
    - Sheet management: add, delete, duplicate, update properties
    - Authentication: auth status and configuration
    """
    
    def __init__(self):
        self.auth = GoogleSheetsAuth(scope_level='full')
        self.spreadsheet_ops = None
        logger.info("🏗️ GoogleSheetsStructureTools initialized")
        
    def get_spreadsheet_ops(self):
        """Get authenticated spreadsheet operations instance"""
        if self.spreadsheet_ops is None:
            service = self.auth.authenticate()
            self.spreadsheet_ops = SpreadsheetOperations(service)
        return self.spreadsheet_ops
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def get_spreadsheet_info(self, spreadsheet_id: str) -> Dict[str, Any]:
        """
        Get metadata about a spreadsheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            
        Returns:
            Dictionary containing spreadsheet metadata
        """
        try:
            ops = self.get_spreadsheet_ops()
            metadata = ops.get_metadata(spreadsheet_id)
            
            # Extract useful information
            result = {
                'spreadsheet_id': metadata['spreadsheetId'],
                'title': metadata['properties']['title'],
                'sheets': [
                    {
                        'sheet_id': sheet['properties']['sheetId'],
                        'title': sheet['properties']['title'],
                        'grid_properties': sheet['properties'].get('gridProperties', {})
                    }
                    for sheet in metadata['sheets']
                ],
                'spreadsheet_url': metadata['spreadsheetUrl'],
                'success': True
            }
            
            logger.info(f"Retrieved info for spreadsheet: {result['title']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get spreadsheet info for ID '{spreadsheet_id}': {e}")
            return {"error": str(e), "success": False}
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def create_spreadsheet(self, title: str, initial_sheets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new Google Sheets spreadsheet.
        
        Args:
            title: The title for the new spreadsheet
            initial_sheets: Optional list of sheet names to create initially
            
        Returns:
            Dictionary containing spreadsheet_id, spreadsheet_url, and sheet names
        """
        try:
            ops = self.get_spreadsheet_ops()
            
            if initial_sheets:
                result = ops.create_with_sheets(title, initial_sheets)
            else:
                spreadsheet_id = ops.create(title)
                result = {
                    'spreadsheet_id': spreadsheet_id,
                    'spreadsheet_url': f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit',
                    'sheets': ['Sheet1'],  # Default sheet
                    'success': True
                }
            
            logger.info(f"Created spreadsheet '{title}' with ID: {result['spreadsheet_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create spreadsheet '{title}': {e}")
            return {"error": str(e), "success": False}
    
    def get_auth_status(self) -> Dict[str, Any]:
        """
        Get the current authentication status and information.
        
        Returns:
            Dictionary containing authentication details
        """
        try:
            auth_info = self.auth.get_auth_info()
            return {
                "status": "success",
                "auth_info": auth_info
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "auth_info": {"authenticated": False}
            }


# Global instance (same pattern as formula servers)
structure_tools = GoogleSheetsStructureTools()


# ================== MCP TOOLS (Call tool instance methods) ==================

@mcp.tool()
async def get_spreadsheet_info(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet")
) -> Dict[str, Any]:
    """
    Get metadata about a spreadsheet.
    
    Examples:
        get_spreadsheet_info("1ABC...") → {'title': 'My Sheet', 'sheets': [...], ...}
    """
    try:
        return structure_tools.get_spreadsheet_info(spreadsheet_id)
    except Exception as e:
        logger.error(f"Error in get_spreadsheet_info: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def create_spreadsheet(
    title: str = Field(description="The title for the new spreadsheet"),
    initial_sheets: Optional[List[str]] = Field(None, description="Optional list of sheet names to create initially")
) -> Dict[str, Any]:
    """
    Create a new Google Sheets spreadsheet.
    
    Examples:
        create_spreadsheet("My New Sheet", ["Data", "Analysis"]) → {'spreadsheet_id': '1ABC...', ...}
    """
    try:
        return structure_tools.create_spreadsheet(title, initial_sheets)
    except Exception as e:
        logger.error(f"Error in create_spreadsheet: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def get_auth_status() -> Dict[str, Any]:
    """
    Get the current authentication status and information.
    
    Examples:
        get_auth_status() → {'status': 'success', 'auth_info': {...}}
    """
    try:
        return structure_tools.get_auth_status()
    except Exception as e:
        logger.error(f"Error in get_auth_status: {e}")
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
    return {"status": "healthy", "service": "Google Sheets Structure MCP Server"}

def main():
    """Run the FastAPI server."""
    import sys
    
    port = 3010  # Default port for structure server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Google Sheets Structure MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

# Export the tool class and mcp instance
__all__ = ['GoogleSheetsStructureTools', 'structure_tools', 'mcp', 'app']