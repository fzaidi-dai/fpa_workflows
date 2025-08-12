"""Google Sheets Formula MCP Server - Handles formula application and translation"""
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional, Union
import logging

# Import our API modules
from ..api.auth import GoogleSheetsAuth
from ..api.value_ops import ValueOperations
from ..api.formula_translator import FormulaTranslator
from ..api.range_resolver import RangeResolver
from ..api.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Google Sheets Formula Server")

# Global auth and operations instances
auth = GoogleSheetsAuth(scope_level='full')
value_ops = None
formula_translator = FormulaTranslator()

def get_value_ops():
    """Get authenticated value operations instance"""
    global value_ops
    if value_ops is None:
        service = auth.authenticate()
        value_ops = ValueOperations(service)
    return value_ops

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def apply_formula(
    spreadsheet_id: str,
    range_spec: str,
    formula: str
) -> Dict[str, Any]:
    """
    Apply a formula to a cell or range.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range in A1 notation where to apply the formula
        formula: The formula to apply (should start with '=')
        
    Returns:
        Dictionary containing application result
    """
    try:
        # Validate formula syntax
        if not formula_translator.validate_formula_syntax(formula):
            return {
                "error": f"Invalid formula syntax: {formula}",
                "success": False
            }
        
        ops = get_value_ops()
        result = ops.update_values(spreadsheet_id, range_spec, 'USER_ENTERED', [[formula]])
        
        response = {
            'formula': formula,
            'range': range_spec,
            'updated_cells': result.get('updatedCells', 0),
            'success': True
        }
        
        logger.info(f"Applied formula {formula} to {spreadsheet_id}:{range_spec}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to apply formula to {spreadsheet_id}:{range_spec}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def translate_polars_to_formula(
    operation: str,
    range_notation: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Translate a Polars operation to a Google Sheets formula.
    
    Args:
        operation: The Polars operation name (e.g., 'sum', 'mean', 'vlookup')
        range_notation: Range in A1 notation for the operation
        **kwargs: Additional parameters for the operation
        
    Returns:
        Dictionary containing the translated formula
    """
    try:
        formula = formula_translator.polars_to_sheets_formula(
            operation, 
            range_notation, 
            **kwargs
        )
        
        response = {
            'polars_operation': operation,
            'sheets_formula': formula,
            'range': range_notation,
            'parameters': kwargs,
            'success': True
        }
        
        logger.info(f"Translated Polars operation '{operation}' to formula: {formula}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to translate Polars operation '{operation}': {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def apply_polars_operation(
    spreadsheet_id: str,
    range_spec: str,
    operation: str,
    source_range: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply a Polars operation as a Google Sheets formula.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range where to write the formula result
        operation: The Polars operation name
        source_range: Source range for the operation
        **kwargs: Additional parameters for the operation
        
    Returns:
        Dictionary containing application result
    """
    try:
        # Translate operation to formula
        formula = formula_translator.polars_to_sheets_formula(
            operation, 
            source_range, 
            **kwargs
        )
        
        # Apply the formula
        ops = get_value_ops()
        result = ops.update_values(spreadsheet_id, range_spec, 'USER_ENTERED', [[formula]])
        
        response = {
            'polars_operation': operation,
            'sheets_formula': formula,
            'source_range': source_range,
            'result_range': range_spec,
            'updated_cells': result.get('updatedCells', 0),
            'success': True
        }
        
        logger.info(f"Applied Polars operation '{operation}' as formula to {spreadsheet_id}:{range_spec}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to apply Polars operation '{operation}': {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def batch_apply_formulas(
    spreadsheet_id: str,
    formulas: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Apply multiple formulas in a single batch operation.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        formulas: List of {'range': str, 'formula': str} dictionaries
        
    Returns:
        Dictionary containing batch application results
    """
    try:
        # Validate all formulas first
        validated_formulas = []
        for formula_spec in formulas:
            formula = formula_spec['formula']
            if not formula_translator.validate_formula_syntax(formula):
                logger.warning(f"Invalid formula syntax: {formula}")
                continue
            validated_formulas.append(formula_spec)
        
        # Convert to batch update format
        data = [
            {
                'range': formula_spec['range'],
                'values': [[formula_spec['formula']]]
            }
            for formula_spec in validated_formulas
        ]
        
        ops = get_value_ops()
        result = ops.batch_update_values(spreadsheet_id, data, 'USER_ENTERED')
        
        response = {
            'formulas_applied': len(validated_formulas),
            'formulas_rejected': len(formulas) - len(validated_formulas),
            'total_updated_cells': result.get('totalUpdatedCells', 0),
            'success': True
        }
        
        logger.info(f"Batch applied {len(validated_formulas)} formulas to {spreadsheet_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to batch apply formulas to {spreadsheet_id}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
async def get_formula_documentation(
    operation: str
) -> Dict[str, Any]:
    """
    Get documentation for a specific formula or operation.
    
    Args:
        operation: The operation name to get documentation for
        
    Returns:
        Dictionary containing formula documentation
    """
    try:
        doc = formula_translator.get_formula_documentation(operation)
        
        response = {
            'operation': operation,
            'documentation': doc,
            'success': True
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get documentation for '{operation}': {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
async def validate_formula(
    formula: str
) -> Dict[str, Any]:
    """
    Validate Google Sheets formula syntax.
    
    Args:
        formula: The formula to validate
        
    Returns:
        Dictionary containing validation result
    """
    try:
        is_valid = formula_translator.validate_formula_syntax(formula)
        
        response = {
            'formula': formula,
            'is_valid': is_valid,
            'success': True
        }
        
        if not is_valid:
            response['issues'] = []
            if not formula.startswith('='):
                response['issues'].append("Formula must start with '='")
            if formula.count('(') != formula.count(')'):
                response['issues'].append("Unbalanced parentheses")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to validate formula '{formula}': {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def create_array_formula(
    spreadsheet_id: str,
    range_spec: str,
    formula: str
) -> Dict[str, Any]:
    """
    Create an ARRAYFORMULA for applying operations across ranges.
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range where to apply the array formula
        formula: The formula content (without ARRAYFORMULA wrapper)
        
    Returns:
        Dictionary containing array formula result
    """
    try:
        # Wrap in ARRAYFORMULA if not already wrapped
        if not formula.upper().startswith('=ARRAYFORMULA'):
            if formula.startswith('='):
                array_formula = f'=ARRAYFORMULA({formula[1:]})'
            else:
                array_formula = f'=ARRAYFORMULA({formula})'
        else:
            array_formula = formula
        
        ops = get_value_ops()
        result = ops.update_values(spreadsheet_id, range_spec, 'USER_ENTERED', [[array_formula]])
        
        response = {
            'original_formula': formula,
            'array_formula': array_formula,
            'range': range_spec,
            'updated_cells': result.get('updatedCells', 0),
            'success': True
        }
        
        logger.info(f"Created array formula in {spreadsheet_id}:{range_spec}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create array formula in {spreadsheet_id}:{range_spec}: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
async def list_supported_operations() -> Dict[str, Any]:
    """
    List all supported Polars operations and their Google Sheets equivalents.
    
    Returns:
        Dictionary containing all supported operations by category
    """
    try:
        operations = {
            'math': {
                'sum': 'SUM - Sum of values',
                'mean': 'AVERAGE - Average of values', 
                'count': 'COUNT - Count of numeric values',
                'max': 'MAX - Maximum value',
                'min': 'MIN - Minimum value'
            },
            'statistical': {
                'std': 'STDEV - Standard deviation',
                'var': 'VAR - Variance',
                'median': 'MEDIAN - Median value'
            },
            'lookup': {
                'vlookup': 'VLOOKUP - Vertical lookup',
                'hlookup': 'HLOOKUP - Horizontal lookup',
                'index_match': 'INDEX/MATCH - Index and match combination'
            },
            'array': {
                'multiply': 'ARRAYFORMULA with * - Element-wise multiplication',
                'divide': 'ARRAYFORMULA with / - Element-wise division',
                'add': 'ARRAYFORMULA with + - Element-wise addition',
                'subtract': 'ARRAYFORMULA with - - Element-wise subtraction'
            },
            'financial': {
                'npv': 'NPV - Net present value',
                'pmt': 'PMT - Payment calculation',
                'fv': 'FV - Future value',
                'pv': 'PV - Present value'
            }
        }
        
        response = {
            'categories': operations,
            'total_operations': sum(len(ops) for ops in operations.values()),
            'success': True
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list supported operations: {e}")
        return {"error": str(e), "success": False}

@mcp.tool()
@ErrorHandler.retry_with_backoff(max_retries=3)
async def read_formula(
    spreadsheet_id: str,
    range_spec: str
) -> Dict[str, Any]:
    """
    Read the formula from a cell (returns the formula, not the calculated value).
    
    Args:
        spreadsheet_id: The ID of the spreadsheet
        range_spec: Range in A1 notation to read formula from
        
    Returns:
        Dictionary containing the formula
    """
    try:
        ops = get_value_ops()
        # Use FORMULA value render option to get the actual formula
        result = ops.get_values_with_formatting(
            spreadsheet_id, 
            range_spec,
            value_render_option='FORMULA'
        )
        
        values = result.get('values', [])
        formula = values[0][0] if values and values[0] else None
        
        response = {
            'range': range_spec,
            'formula': formula,
            'has_formula': formula is not None and str(formula).startswith('='),
            'success': True
        }
        
        logger.info(f"Read formula from {spreadsheet_id}:{range_spec}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to read formula from {spreadsheet_id}:{range_spec}: {e}")
        return {"error": str(e), "success": False}

# Export the mcp instance
__all__ = ['mcp']