"""Google Sheets Validation MCP Server - Handles data validation and business rules"""
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional, Union
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import json

# Import our API modules
from ..api.auth import GoogleSheetsAuth
from ..api.value_ops import ValueOperations
from ..api.batch_ops import BatchOperations
from ..api.range_resolver import RangeResolver
from ..api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize MCP server and FastAPI app
mcp = FastMCP("Google Sheets Validation Server")
app = FastAPI(title="Google Sheets Validation MCP Server", version="1.0.0")

# Global auth and operations instances
auth = GoogleSheetsAuth(scope_level='full')
value_ops = None
batch_ops = None

def get_value_ops():
    """Get authenticated value operations instance"""
    global value_ops
    if value_ops is None:
        service = auth.authenticate()
        value_ops = ValueOperations(service)
    return value_ops

def get_batch_ops():
    """Get authenticated batch operations instance"""
    global batch_ops
    if batch_ops is None:
        service = auth.authenticate()
        batch_ops = BatchOperations(service)
    return batch_ops


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


# Global instance
validation_tools = GoogleSheetsValidationTools()

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def add_data_validation(
    spreadsheet_id: str,
    range_spec: str,
    validation_rule: Dict[str, Any]
) -> Dict[str, Any]:
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
        
        ops = validation_tools.get_batch_ops()
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

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def validate_data_quality(
    spreadsheet_id: str,
    range_spec: str,
    checks: List[str]
) -> Dict[str, Any]:
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
        ops = validation_tools.get_value_ops()
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

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def validate_business_rules(
    spreadsheet_id: str,
    range_spec: str,
    rules: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate custom business rules on data.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range in A1 notation to validate
        rules: List of business rules to validate
        
    Returns:
        Dictionary containing business rule validation results
    """
    try:
        ops = validation_tools.get_value_ops()
        values = ops.get_values(spreadsheet_id, range_spec)
        
        validation_results = {
            'range': range_spec,
            'rules_checked': len(rules),
            'rule_results': [],
            'overall_passed': True,
            'success': True
        }
        
        for rule in rules:
            rule_name = rule.get('name', 'Unnamed rule')
            rule_type = rule.get('type')
            rule_params = rule.get('parameters', {})
            
            rule_result = {
                'name': rule_name,
                'type': rule_type,
                'passed': True,
                'violations': []
            }
            
            if rule_type == 'range_check':
                min_val = rule_params.get('min')
                max_val = rule_params.get('max')
                
                for i, row in enumerate(values):
                    for j, cell in enumerate(row):
                        if cell and str(cell).replace('.', '', 1).replace('-', '', 1).isdigit():
                            val = float(cell)
                            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                                rule_result['violations'].append({
                                    'row': i,
                                    'col': j,
                                    'value': val,
                                    'issue': f'Value {val} outside range [{min_val}, {max_val}]'
                                })
                                rule_result['passed'] = False
            
            elif rule_type == 'pattern_match':
                import re
                pattern = rule_params.get('pattern')
                if pattern:
                    regex = re.compile(pattern)
                    for i, row in enumerate(values):
                        for j, cell in enumerate(row):
                            if cell and not regex.match(str(cell)):
                                rule_result['violations'].append({
                                    'row': i,
                                    'col': j,
                                    'value': cell,
                                    'issue': f'Value does not match pattern {pattern}'
                                })
                                rule_result['passed'] = False
            
            validation_results['rule_results'].append(rule_result)
            if not rule_result['passed']:
                validation_results['overall_passed'] = False
        
        logger.info(f"Validated business rules for {spreadsheet_id}:{range_spec}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Failed to validate business rules for {spreadsheet_id}:{range_spec}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
async def create_validation_preset(
    name: str,
    validation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a validation preset for reuse.
    
    Args:
        name: Name of the validation preset
        validation_config: Validation configuration
        
    Returns:
        Dictionary containing preset information
    """
    preset = {
        'name': name,
        'config': validation_config,
        'created': True
    }
    
    logger.info(f"Created validation preset '{name}'")
    return preset

@mcp.tool()
async def get_validation_examples() -> Dict[str, Any]:
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
    return {"status": "healthy", "service": "Google Sheets Validation MCP Server"}

# Main function to run the server
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

# Export the mcp instance and app
__all__ = ['mcp', 'app']