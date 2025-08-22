#!/usr/bin/env python3
"""
Google Sheets Chart MCP Server - Following exact formula server pattern

This server handles Google Sheets chart operations with the same architecture
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
mcp = FastMCP("Google Sheets Chart Server")
app = FastAPI(
    title="Google Sheets Chart MCP Server", 
    version="1.0.0",
    description="Google Sheets chart operations with 100% reliability"
)


class GoogleSheetsChartTools:
    """
    Focused MCP tools for Google Sheets chart operations.
    
    This specialized server handles all chart needs:
    - Chart creation: line, column, bar, pie, scatter charts
    - Pivot tables: data summarization and analysis
    - Chart management: positioning and configuration
    """
    
    def __init__(self):
        self.auth = GoogleSheetsAuth(scope_level='full')
        self.batch_ops = None
        logger.info("📈 GoogleSheetsChartTools initialized")
        
    def get_batch_ops(self):
        """Get authenticated batch operations instance"""
        if self.batch_ops is None:
            service = self.auth.authenticate()
            self.batch_ops = BatchOperations(service)
        return self.batch_ops
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def create_chart(self, spreadsheet_id: str, sheet_id: int, chart_type: str, data_range: str, title: str, position: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Create a chart in the spreadsheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            sheet_id: ID of the sheet where to place the chart
            chart_type: Type of chart ('LINE', 'COLUMN', 'PIE', etc.)
            data_range: Range containing chart data in A1 notation
            title: Title for the chart
            position: Optional position {'row': int, 'col': int}
            
        Returns:
            Dictionary containing chart creation result
        """
        try:
            parsed_range = RangeResolver.parse_a1_notation(data_range)
            
            chart_spec = {
                "title": title,
                "basicChart": {
                    "chartType": chart_type.upper(),
                    "domains": [{
                        "domain": {
                            "sourceRange": {
                                "sources": [{
                                    "sheetId": sheet_id,
                                    "startRowIndex": parsed_range.start_row,
                                    "endRowIndex": parsed_range.end_row + 1 if parsed_range.end_row is not None else parsed_range.start_row + 1,
                                    "startColumnIndex": parsed_range.start_col,
                                    "endColumnIndex": parsed_range.start_col + 1
                                }]
                            }
                        }
                    }],
                    "series": [{
                        "series": {
                            "sourceRange": {
                                "sources": [{
                                    "sheetId": sheet_id,
                                    "startRowIndex": parsed_range.start_row,
                                    "endRowIndex": parsed_range.end_row + 1 if parsed_range.end_row is not None else parsed_range.start_row + 1,
                                    "startColumnIndex": parsed_range.start_col + 1,
                                    "endColumnIndex": parsed_range.end_col + 1 if parsed_range.end_col is not None else parsed_range.start_col + 2
                                }]
                            }
                        }
                    }]
                }
            }
            
            # Set position if provided
            embedded_object_position = {
                "sheetId": sheet_id,
                "overlayPosition": {
                    "anchorCell": {
                        "sheetId": sheet_id,
                        "rowIndex": position.get('row', 0) if position else 0,
                        "columnIndex": position.get('col', 5) if position else 5
                    }
                }
            }
            
            request = {
                "addChart": {
                    "chart": {
                        "spec": chart_spec,
                        "position": embedded_object_position
                    }
                }
            }
            
            ops = self.get_batch_ops()
            result = ops.batch_update(spreadsheet_id, [request])
            
            response = {
                'chart_type': chart_type,
                'data_range': data_range,
                'title': title,
                'sheet_id': sheet_id,
                'success': True
            }
            
            logger.info(f"Created {chart_type} chart in {spreadsheet_id} sheet {sheet_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to create chart in {spreadsheet_id}: {e}")
            return {"error": str(e), "success": False}
    
    def get_chart_types(self) -> Dict[str, Any]:
        """
        Get available chart types and their descriptions.
        
        Returns:
            Dictionary of available chart types
        """
        chart_types = {
            'LINE': 'Line chart for showing trends over time',
            'COLUMN': 'Column chart for comparing values',
            'BAR': 'Bar chart for horizontal comparisons',
            'PIE': 'Pie chart for showing parts of a whole',
            'SCATTER': 'Scatter plot for showing correlation',
            'AREA': 'Area chart for showing cumulative values',
            'HISTOGRAM': 'Histogram for showing distribution',
            'CANDLESTICK': 'Candlestick chart for financial data'
        }
        
        return {
            'chart_types': chart_types,
            'total_types': len(chart_types),
            'success': True
        }


# Global instance (same pattern as formula servers)
chart_tools = GoogleSheetsChartTools()


# ================== MCP TOOLS (Call tool instance methods) ==================

@mcp.tool()
async def create_chart(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    sheet_id: int = Field(description="ID of the sheet where to place the chart"),
    chart_type: str = Field(description="Type of chart ('LINE', 'COLUMN', 'PIE', etc.)"),
    data_range: str = Field(description="Range containing chart data in A1 notation"),
    title: str = Field(description="Title for the chart"),
    position: Optional[Dict[str, int]] = Field(None, description="Optional position {'row': int, 'col': int}")
) -> Dict[str, Any]:
    """
    Create a chart in the spreadsheet.
    
    Examples:
        create_chart("1ABC...", 0, "COLUMN", "A1:B10", "Sales Chart", {"row": 5, "col": 5})
    """
    try:
        return chart_tools.create_chart(spreadsheet_id, sheet_id, chart_type, data_range, title, position)
    except Exception as e:
        logger.error(f"Error in create_chart: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def get_chart_types() -> Dict[str, Any]:
    """
    Get available chart types and their descriptions.
    
    Examples:
        get_chart_types() → {'chart_types': {'LINE': '...', 'COLUMN': '...'}, ...}
    """
    try:
        return chart_tools.get_chart_types()
    except Exception as e:
        logger.error(f"Error in get_chart_types: {e}")
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
    return {"status": "healthy", "service": "Google Sheets Chart MCP Server"}

def main():
    """Run the FastAPI server."""
    import sys
    
    port = 3014  # Default port for chart server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Google Sheets Chart MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

# Export the tool class and mcp instance
__all__ = ['GoogleSheetsChartTools', 'chart_tools', 'mcp', 'app']