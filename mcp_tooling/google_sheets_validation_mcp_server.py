#!/usr/bin/env python3
"""
Google Sheets Validation MCP Server - Following exact formula server pattern

This server handles Google Sheets validation operations with the same architecture
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
from mcp_tooling.google_sheets.api.batch_ops import BatchOperations
from mcp_tooling.google_sheets.api.range_resolver import RangeResolver
from mcp_tooling.google_sheets.api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Google Sheets Validation Server")
app = FastAPI(
    title="Google Sheets Validation MCP Server", 
    version="1.0.0",
    description="Google Sheets validation operations with 100% reliability"
)


class GoogleSheetsValidationTools:
    """
    Focused MCP tools for Google Sheets validation operations.
    
    This specialized server handles all validation needs:
    - Data validation: dropdown lists, number ranges, custom rules
    - Data quality checks: empty cells, numeric validation, uniqueness
    - Business rules: custom validation logic and presets
    """
    
    def __init__(self):
        self.auth = GoogleSheetsAuth(scope_level='full')
        self.value_ops = None
        self.batch_ops = None
        logger.info("✅ GoogleSheetsValidationTools initialized")
        
    def get_value_ops(self):
        """Get authenticated value operations instance"""
        if self.value_ops is None:
            service = self.auth.authenticate()
            self.value_ops = ValueOperations(service)
        return self.value_ops
    
    def get_batch_ops(self):
        """Get authenticated batch operations instance"""
        if self.batch_ops is None:
            service = self.auth.authenticate()
            self.batch_ops = BatchOperations(service)
        return self.batch_ops
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def add_data_validation(self, spreadsheet_id: str, range_spec: str, validation_rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add data validation rules to a range.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_spec: Range in A1 notation to apply validation
            validation_rule: Validation rule configuration
            
        Returns:
            Dictionary containing validation result
        """
        try:
            parsed_range = RangeResolver.parse_a1_notation(range_spec)
            
            request = {
                "setDataValidation": {
                    "range": {
                        "sheetId": 0,  # TODO: Get actual sheet ID
                        "startRowIndex": parsed_range.start_row,
                        "endRowIndex": parsed_range.end_row + 1 if parsed_range.end_row is not None else parsed_range.start_row + 1,
                        "startColumnIndex": parsed_range.start_col,
                        "endColumnIndex": parsed_range.end_col + 1 if parsed_range.end_col is not None else parsed_range.start_col + 1
                    },
                    "rule": validation_rule
                }
            }
            
            ops = self.get_batch_ops()
            result = ops.batch_update(spreadsheet_id, [request])
            
            response = {
                'range': range_spec,
                'validation_rule': validation_rule,
                'success': True
            }
            
            logger.info(f"Added data validation to {spreadsheet_id}:{range_spec}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to add data validation to {spreadsheet_id}:{range_spec}: {e}")
            return {"error": str(e), "success": False}
    
    @ErrorHandler.retry_with_backoff(max_retries=3)
    def validate_data_quality(self, spreadsheet_id: str, range_spec: str, checks: List[str]) -> Dict[str, Any]:
        """
        Perform data quality validation on a range.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_spec: Range in A1 notation to validate
            checks: List of validation checks ('not_empty', 'numeric', 'unique', etc.)
            
        Returns:
            Dictionary containing validation results
        """
        try:
            ops = self.get_value_ops()
            values = ops.get_values(spreadsheet_id, range_spec)
            
            validation_results = {
                'range': range_spec,
                'total_cells': sum(len(row) for row in values) if values else 0,
                'checks': {},
                'issues': [],
                'success': True
            }
            
            if not values:
                validation_results['issues'].append('Range is empty')
                return validation_results
            
            # Perform validation checks
            for check in checks:
                if check == 'not_empty':
                    empty_cells = 0
                    for i, row in enumerate(values):
                        for j, cell in enumerate(row):
                            if not cell or str(cell).strip() == '':
                                empty_cells += 1
                    
                    validation_results['checks']['not_empty'] = {
                        'empty_cells': empty_cells,
                        'passed': empty_cells == 0
                    }
                    
                    if empty_cells > 0:
                        validation_results['issues'].append(f'{empty_cells} empty cells found')
                
                elif check == 'numeric':
                    non_numeric_cells = 0
                    for i, row in enumerate(values):
                        for j, cell in enumerate(row):
                            if cell and not str(cell).replace('.', '', 1).replace('-', '', 1).isdigit():
                                non_numeric_cells += 1
                    
                    validation_results['checks']['numeric'] = {
                        'non_numeric_cells': non_numeric_cells,
                        'passed': non_numeric_cells == 0
                    }
                    
                    if non_numeric_cells > 0:
                        validation_results['issues'].append(f'{non_numeric_cells} non-numeric cells found')
                
                elif check == 'unique':
                    all_values = []
                    for row in values:
                        for cell in row:
                            if cell:
                                all_values.append(str(cell))
                    
                    unique_count = len(set(all_values))
                    total_count = len(all_values)
                    duplicates = total_count - unique_count
                    
                    validation_results['checks']['unique'] = {
                        'total_values': total_count,
                        'unique_values': unique_count,
                        'duplicates': duplicates,
                        'passed': duplicates == 0
                    }
                    
                    if duplicates > 0:
                        validation_results['issues'].append(f'{duplicates} duplicate values found')
            
            logger.info(f"Validated data quality for {spreadsheet_id}:{range_spec}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate data quality for {spreadsheet_id}:{range_spec}: {e}")
            return {"error": str(e), "success": False}
    
    def get_validation_examples(self) -> Dict[str, Any]:
        """
        Get examples of validation rules and configurations.
        
        Returns:
            Dictionary containing validation examples
        """
        examples = {
            'data_validation_rules': {
                'dropdown': {
                    'condition': {
                        'type': 'ONE_OF_LIST',
                        'values': [{'userEnteredValue': 'Option 1'}, {'userEnteredValue': 'Option 2'}]
                    },
                    'showCustomUi': True
                },
                'number_range': {
                    'condition': {
                        'type': 'NUMBER_BETWEEN',
                        'values': [{'userEnteredValue': '1'}, {'userEnteredValue': '100'}]
                    },
                    'inputMessage': 'Enter a number between 1 and 100'
                }
            },
            'business_rules': {
                'payback_months_range': {
                    'name': 'Payback months validation',
                    'type': 'range_check',
                    'parameters': {'min': 0, 'max': 60}
                },
                'email_format': {
                    'name': 'Email format validation',
                    'type': 'pattern_match',
                    'parameters': {'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'}
                }
            }
        }
        
        return {
            'examples': examples,
            'success': True
        }


# Global instance (same pattern as formula servers)
validation_tools = GoogleSheetsValidationTools()


# ================== MCP TOOLS (Call tool instance methods) ==================

@mcp.tool()
async def add_data_validation(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    range_spec: str = Field(description="Range in A1 notation to apply validation"),
    validation_rule: Dict[str, Any] = Field(description="Validation rule configuration")
) -> Dict[str, Any]:
    """
    Add data validation rules to a range.
    
    Examples:
        add_data_validation("1ABC...", "A1:A10", {"condition": {"type": "ONE_OF_LIST", "values": [...]}})
    """
    try:
        return validation_tools.add_data_validation(spreadsheet_id, range_spec, validation_rule)
    except Exception as e:
        logger.error(f"Error in add_data_validation: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def validate_data_quality(
    spreadsheet_id: str = Field(description="The ID of the spreadsheet"),
    range_spec: str = Field(description="Range in A1 notation to validate"),
    checks: List[str] = Field(description="List of validation checks ('not_empty', 'numeric', 'unique', etc.)")
) -> Dict[str, Any]:
    """
    Perform data quality validation on a range.
    
    Examples:
        validate_data_quality("1ABC...", "A1:C10", ["not_empty", "numeric"]) → {'checks': {...}, 'issues': [...]}
    """
    try:
        return validation_tools.validate_data_quality(spreadsheet_id, range_spec, checks)
    except Exception as e:
        logger.error(f"Error in validate_data_quality: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool()
async def get_validation_examples() -> Dict[str, Any]:
    """
    Get examples of validation rules and configurations.
    
    Examples:
        get_validation_examples() → {'examples': {'data_validation_rules': {...}, ...}}
    """
    try:
        return validation_tools.get_validation_examples()
    except Exception as e:
        logger.error(f"Error in get_validation_examples: {e}")
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
    return {"status": "healthy", "service": "Google Sheets Validation MCP Server"}

def main():
    """Run the FastAPI server."""
    import sys
    
    port = 3015  # Default port for validation server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Google Sheets Validation MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

# Export the tool class and mcp instance
__all__ = ['GoogleSheetsValidationTools', 'validation_tools', 'mcp', 'app']