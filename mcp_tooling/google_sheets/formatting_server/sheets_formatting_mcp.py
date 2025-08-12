"""Google Sheets Formatting MCP Server - Handles cell formatting and styling"""
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional
import logging

# Import our API modules
from ..api.auth import GoogleSheetsAuth
from ..api.batch_ops import BatchOperations
from ..api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Google Sheets Formatting Server")

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
        
        ops = get_batch_ops()
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
        
        ops = get_batch_ops()
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

# Export the mcp instance
__all__ = ['mcp']