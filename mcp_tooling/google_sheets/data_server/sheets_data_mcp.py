"""Google Sheets Data MCP Server - Handles data reading, writing, and manipulation"""
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional, Union
import logging
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import json

# Import our API modules
from ..api.auth import GoogleSheetsAuth
from ..api.value_ops import ValueOperations
from ..api.range_resolver import RangeResolver
from ..api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize MCP server and FastAPI app
mcp = FastMCP("Google Sheets Data Server")
app = FastAPI(title="Google Sheets Data MCP Server", version="1.0.0")

# Global auth and operations instances
auth = GoogleSheetsAuth(scope_level='full')
value_ops = None

def get_value_ops():
    """Get authenticated value operations instance"""
    global value_ops
    if value_ops is None:
        service = auth.authenticate()
        value_ops = ValueOperations(service)
    return value_ops


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


# Global instance
data_tools = GoogleSheetsDataTools()

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def read_values(
    spreadsheet_id: str,
    range_spec: str,
    value_render_option: str = 'FORMATTED_VALUE'
) -> Dict[str, Any]:
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
        ops = data_tools.get_value_ops()
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

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def write_values(
    spreadsheet_id: str,
    range_spec: str,
    values: List[List[Any]],
    value_input_option: str = 'USER_ENTERED'
) -> Dict[str, Any]:
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
        ops = data_tools.get_value_ops()
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

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def batch_read_values(
    spreadsheet_id: str,
    ranges: List[str]
) -> Dict[str, Any]:
    """
    Read values from multiple ranges in a single request.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        ranges: List of ranges in A1 notation
        
    Returns:
        Dictionary containing values for each range
    """
    try:
        ops = data_tools.get_value_ops()
        results = ops.batch_get_values(spreadsheet_id, ranges)
        
        # Format results by range
        range_data = {}
        for result in results:
            range_name = result.get('range', 'Unknown')
            range_data[range_name] = {
                'values': result.get('values', []),
                'majorDimension': result.get('majorDimension', 'ROWS')
            }
        
        response = {
            'ranges': range_data,
            'total_ranges': len(results),
            'success': True
        }
        
        logger.info(f"Batch read {len(results)} ranges from {spreadsheet_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to batch read values from {spreadsheet_id}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def batch_write_values(
    spreadsheet_id: str,
    data: List[Dict[str, Any]],
    value_input_option: str = 'USER_ENTERED'
) -> Dict[str, Any]:
    """
    Write values to multiple ranges in a single request.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        data: List of {'range': str, 'values': List[List[Any]]} objects
        value_input_option: How values should be interpreted ('RAW', 'USER_ENTERED')
        
    Returns:
        Dictionary containing batch update results
    """
    try:
        ops = data_tools.get_value_ops()
        result = ops.batch_update_values(spreadsheet_id, data, value_input_option)
        
        response = {
            'total_updated_cells': result.get('totalUpdatedCells', 0),
            'total_updated_sheets': result.get('totalUpdatedSheets', 0),
            'responses': result.get('responses', []),
            'success': True
        }
        
        logger.info(f"Batch wrote {response['total_updated_cells']} cells to {spreadsheet_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to batch write values to {spreadsheet_id}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def append_values(
    spreadsheet_id: str,
    range_spec: str,
    values: List[List[Any]],
    value_input_option: str = 'USER_ENTERED'
) -> Dict[str, Any]:
    """
    Append values to a range (adds rows after existing data).
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range in A1 notation where to start appending
        values: 2D array of values to append
        value_input_option: How values should be interpreted ('RAW', 'USER_ENTERED')
        
    Returns:
        Dictionary containing append result and metadata
    """
    try:
        ops = data_tools.get_value_ops()
        result = ops.append_values(spreadsheet_id, range_spec, value_input_option, values)
        
        updates = result.get('updates', {})
        response = {
            'updated_cells': updates.get('updatedCells', 0),
            'updated_rows': updates.get('updatedRows', 0),
            'updated_range': updates.get('updatedRange'),
            'success': True
        }
        
        logger.info(f"Appended {response['updated_cells']} cells to {spreadsheet_id}:{range_spec}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to append values to {spreadsheet_id}:{range_spec}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def clear_values(
    spreadsheet_id: str,
    range_spec: str
) -> Dict[str, Any]:
    """
    Clear values from a range.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range in A1 notation to clear
        
    Returns:
        Dictionary containing clear result
    """
    try:
        ops = data_tools.get_value_ops()
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

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def polars_to_sheets(
    spreadsheet_id: str,
    sheet_name: str,
    dataframe_data: Dict[str, Any],
    start_cell: str = 'A1',
    include_headers: bool = True
) -> Dict[str, Any]:
    """
    Write a Polars DataFrame to Google Sheets.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        sheet_name: Name of the sheet to write to
        dataframe_data: Dictionary representation of DataFrame {'columns': [...], 'data': [[...]]}
        start_cell: Starting cell in A1 notation
        include_headers: Whether to include column headers
        
    Returns:
        Dictionary containing write result and range used
    """
    try:
        # Reconstruct values from dataframe data
        values = []
        if include_headers and 'columns' in dataframe_data:
            values.append(dataframe_data['columns'])
        
        if 'data' in dataframe_data:
            values.extend(dataframe_data['data'])
        
        # Calculate range based on data size
        rows = len(values)
        cols = len(values[0]) if values else 0
        
        # Parse start cell and calculate end range
        parsed_start = RangeResolver.parse_a1_notation(start_cell)
        end_col = parsed_start.start_col + cols - 1
        end_row = parsed_start.start_row + rows - 1
        
        end_col_letter = RangeResolver.index_to_column_letter(end_col)
        range_spec = f"{sheet_name}!{start_cell}:{end_col_letter}{end_row + 1}"
        
        # Write values
        ops = data_tools.get_value_ops()
        result = ops.update_values(spreadsheet_id, range_spec, 'USER_ENTERED', values)
        
        response = {
            'range_used': range_spec,
            'updated_cells': result.get('updatedCells', 0),
            'rows_written': rows,
            'columns_written': cols,
            'success': True
        }
        
        logger.info(f"Wrote Polars DataFrame to {spreadsheet_id}:{range_spec} ({rows}x{cols})")
        return response
        
    except Exception as e:
        logger.error(f"Failed to write Polars DataFrame to {spreadsheet_id}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def sheets_to_polars(
    spreadsheet_id: str,
    range_spec: str,
    headers: bool = True
) -> Dict[str, Any]:
    """
    Read Google Sheets data and format it for Polars DataFrame creation.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range in A1 notation to read
        headers: Whether the first row contains headers
        
    Returns:
        Dictionary with data formatted for Polars DataFrame creation
    """
    try:
        ops = data_tools.get_value_ops()
        values = ops.get_values(spreadsheet_id, range_spec)
        
        if not values:
            return {
                'columns': [],
                'data': [],
                'row_count': 0,
                'success': True
            }
        
        # Process headers and data
        if headers and len(values) > 0:
            columns = values[0]
            data = values[1:] if len(values) > 1 else []
        else:
            # Generate generic column names
            max_cols = max(len(row) for row in values) if values else 0
            columns = [f'column_{i}' for i in range(max_cols)]
            data = values
        
        # Ensure all rows have the same number of columns
        normalized_data = []
        for row in data:
            normalized_row = list(row) + [None] * (len(columns) - len(row))
            normalized_row = normalized_row[:len(columns)]  # Trim if too long
            normalized_data.append(normalized_row)
        
        response = {
            'columns': columns,
            'data': normalized_data,
            'row_count': len(normalized_data),
            'column_count': len(columns),
            'range': range_spec,
            'success': True
        }
        
        logger.info(f"Read {len(normalized_data)} rows x {len(columns)} cols from {spreadsheet_id}:{range_spec}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to read data for Polars from {spreadsheet_id}:{range_spec}: {e}")
        return {"error": str(e), "success": False}

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
    return {"status": "healthy", "service": "Google Sheets Data MCP Server"}

# Main function to run the server
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

# Export the mcp instance and app
__all__ = ['mcp', 'app']