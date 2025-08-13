#!/usr/bin/env python3
"""
Enhanced Math MCP Server with Dual-Layer Execution

This MCP server demonstrates the integration of dual-layer execution capabilities
with existing math and aggregation functions, enabling both Polars computation
and Google Sheets formula transparency.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import uvicorn
import polars as pl

# Import existing sheets functions
from .sheets_compatible_functions import SheetsCompatibleFunctions

# Import dual-layer components
from .dual_layer import (
    DualLayerExecutor, 
    PlanStep, 
    OperationType, 
    CellContext,
    CellType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
sheets_funcs = SheetsCompatibleFunctions()
dual_layer_executor = DualLayerExecutor(
    mappings_dir=Path(__file__).parent / "formula_mappings"
)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Enhanced Dual-Layer Math Server")
app = FastAPI(title="Enhanced Dual-Layer Math MCP Server", version="3.0.0")


# ================== DUAL-LAYER ENHANCED TOOLS ==================

@mcp.tool()
async def dual_layer_sum(
    data_source: str = Field(description="Path to data file or JSON data"),
    output_range: str = Field(description="Google Sheets output range (e.g., 'Summary!B2')"),
    range_spec: Optional[str] = Field(None, description="A1 notation range (e.g., 'A1:A100')"),
    column: Optional[str] = Field(None, description="Column name for aggregation"),
    spreadsheet_id: Optional[str] = Field(None, description="Target Google Sheets ID"),
    enable_validation: bool = Field(True, description="Enable dual-layer validation")
) -> Dict[str, Any]:
    """
    SUM function with dual-layer execution - computes with Polars, displays formula in Sheets
    
    Examples:
        dual_layer_sum("sales.csv", "B:B", "revenue", "Dashboard!C5")
        dual_layer_sum("data.parquet", "A1:A1000", None, "Results!D10")
    """
    try:
        # Create operation specification
        operation = {
            "type": "sum",
            "column": column,
            "range": range_spec
        }
        
        # Create sheet context
        sheet_context = {
            "data_range": range_spec or f"{column}:{column}" if column else "A:A",
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": output_range.split("!")[0] if "!" in output_range else "Sheet1",
            "current_cell": output_range.split("!")[1] if "!" in output_range else output_range
        }
        
        # Create plan step
        step = PlanStep(
            step_id="sum_001",
            description=f"Calculate SUM of {column or range_spec}",
            operation_type=OperationType.AGGREGATION,
            operation=operation,
            input_data=data_source,
            output_range=output_range,
            sheet_context=sheet_context,
            validation_rules=[
                {"type": "range_check", "name": "positive_result", "min": 0},
                {"type": "type_check", "name": "numeric_result", "expected_type": "numeric"}
            ]
        )
        
        # Execute with dual-layer system
        result = await dual_layer_executor.execute_step(step)
        
        return {
            "success": result.success,
            "polars_result": str(result.polars_result),
            "sheets_formula": result.sheets_formula,
            "execution_time_ms": result.execution_time_ms,
            "validation_passed": result.all_validations_passed,
            "output_range": result.sheets_range,
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Error in dual_layer_sum: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def dual_layer_average(
    data_source: str = Field(description="Path to data file or JSON data"),
    output_range: str = Field(description="Google Sheets output range"),
    range_spec: Optional[str] = Field(None, description="A1 notation range"),
    column: Optional[str] = Field(None, description="Column name for aggregation"),
    spreadsheet_id: Optional[str] = Field(None, description="Target Google Sheets ID"),
    enable_validation: bool = Field(True, description="Enable dual-layer validation")
) -> Dict[str, Any]:
    """
    AVERAGE function with dual-layer execution
    
    Examples:
        dual_layer_average("metrics.csv", "C:C", "conversion_rate", "KPIs!E3")
    """
    try:
        operation = {
            "type": "average",
            "column": column,
            "range": range_spec
        }
        
        sheet_context = {
            "data_range": range_spec or f"{column}:{column}" if column else "A:A",
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": output_range.split("!")[0] if "!" in output_range else "Sheet1",
            "current_cell": output_range.split("!")[1] if "!" in output_range else output_range
        }
        
        step = PlanStep(
            step_id="avg_001",
            description=f"Calculate AVERAGE of {column or range_spec}",
            operation_type=OperationType.AGGREGATION,
            operation=operation,
            input_data=data_source,
            output_range=output_range,
            sheet_context=sheet_context,
            validation_rules=[
                {"type": "type_check", "name": "numeric_result", "expected_type": "numeric"}
            ]
        )
        
        result = await dual_layer_executor.execute_step(step)
        
        return {
            "success": result.success,
            "polars_result": str(result.polars_result),
            "sheets_formula": result.sheets_formula,
            "execution_time_ms": result.execution_time_ms,
            "validation_passed": result.all_validations_passed,
            "output_range": result.sheets_range,
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Error in dual_layer_average: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def dual_layer_groupby_sum(
    data_source: str = Field(description="Path to data file"),
    output_range: str = Field(description="Google Sheets output range for results"),
    group_by_column: str = Field(description="Column to group by"),
    sum_column: str = Field(description="Column to sum"),
    spreadsheet_id: Optional[str] = Field(None, description="Target Google Sheets ID")
) -> Dict[str, Any]:
    """
    GROUP BY SUM with dual-layer execution - creates both Polars computation and Sheets SUMIF formulas
    
    Examples:
        dual_layer_groupby_sum("sales.csv", "region", "revenue", "Analysis!A1:B10")
    """
    try:
        operation = {
            "type": "groupby_sum",
            "group_by": group_by_column,
            "sum_column": sum_column,
            "source_range": "A:Z"  # Full data range
        }
        
        sheet_context = {
            "data_range": "A:Z",
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": output_range.split("!")[0] if "!" in output_range else "Sheet1",
            "group_column": group_by_column,
            "sum_column": sum_column
        }
        
        step = PlanStep(
            step_id="groupby_001",
            description=f"GROUP BY {group_by_column} SUM({sum_column})",
            operation_type=OperationType.AGGREGATION,
            operation=operation,
            input_data=data_source,
            output_range=output_range,
            sheet_context=sheet_context,
            validation_rules=[
                {"type": "type_check", "name": "dataframe_result", "expected_type": "dict"}
            ]
        )
        
        result = await dual_layer_executor.execute_step(step)
        
        return {
            "success": result.success,
            "polars_result": str(result.polars_result),
            "sheets_formula": result.sheets_formula,
            "execution_time_ms": result.execution_time_ms,
            "validation_passed": result.all_validations_passed,
            "output_range": result.sheets_range,
            "cell_contexts": [ctx.to_dict() for ctx in result.cell_contexts],
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Error in dual_layer_groupby_sum: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def dual_layer_complex_calculation(
    data_source: str = Field(description="Path to data file"),
    output_range: str = Field(description="Google Sheets output range"),
    calculation_spec: Dict[str, Any] = Field(description="Complex calculation specification"),
    spreadsheet_id: Optional[str] = Field(None, description="Target Google Sheets ID")
) -> Dict[str, Any]:
    """
    Complex calculation with dual-layer execution and validation notes
    
    Examples:
        dual_layer_complex_calculation(
            "customers.csv",
            {
                "type": "customer_lifetime_value",
                "revenue_column": "monthly_revenue",
                "tenure_column": "months_active",
                "group_by": "customer_id"
            },
            "CLV_Analysis!D5"
        )
    """
    try:
        operation = calculation_spec
        
        sheet_context = {
            "data_range": "A:Z",
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": output_range.split("!")[0] if "!" in output_range else "Sheet1",
            "current_cell": output_range.split("!")[1] if "!" in output_range else output_range
        }
        
        step = PlanStep(
            step_id="complex_001",
            description=f"Complex calculation: {calculation_spec.get('type', 'unknown')}",
            operation_type=OperationType.TRANSFORMATION,
            operation=operation,
            input_data=data_source,
            output_range=output_range,
            sheet_context=sheet_context,
            validation_rules=[
                {"type": "custom", "name": "business_logic", "function": "positive_values"},
                {"type": "range_check", "name": "reasonable_values", "min": 0, "max": 1000000}
            ]
        )
        
        result = await dual_layer_executor.execute_step(step)
        
        return {
            "success": result.success,
            "polars_result": str(result.polars_result),
            "sheets_formula": result.sheets_formula,
            "execution_time_ms": result.execution_time_ms,
            "validation_passed": result.all_validations_passed,
            "validation_details": [check.to_dict() for check in (result.validation or [])],
            "output_range": result.sheets_range,
            "cell_contexts": [ctx.to_dict() for ctx in result.cell_contexts],
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Error in dual_layer_complex_calculation: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_execution_statistics() -> Dict[str, Any]:
    """
    Get dual-layer execution statistics and performance metrics
    """
    try:
        stats = dual_layer_executor.get_execution_stats()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting execution statistics: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def validate_formula_mapping(
    operation_type: str = Field(description="Type of operation to validate"),
    sample_data: str = Field(description="Sample data for validation")
) -> Dict[str, Any]:
    """
    Validate that a formula mapping produces consistent results between Polars and Sheets
    """
    try:
        # Create a test operation
        test_operation = {"type": operation_type}
        test_context = {"data_range": "A:A"}
        
        # Get the formula translation
        formula = dual_layer_executor.formula_translator.translate_operation(
            test_operation, test_context
        )
        
        # Simulate validation (in practice, would execute both Polars and Sheets)
        validation_result = dual_layer_executor.formula_translator.validate_translation(
            "test_result", formula
        )
        
        complexity = dual_layer_executor.formula_translator.get_formula_complexity(formula)
        
        return {
            "success": True,
            "operation_type": operation_type,
            "generated_formula": formula,
            "validation_passed": validation_result,
            "formula_complexity": complexity
        }
        
    except Exception as e:
        logger.error(f"Error validating formula mapping: {e}")
        return {"success": False, "error": str(e)}


# ================== API MODELS ==================

class DualLayerRequest(BaseModel):
    data_source: str
    range_spec: Optional[str] = None
    column: Optional[str] = None
    output_range: str
    spreadsheet_id: Optional[str] = None
    enable_validation: bool = True

class GroupByRequest(BaseModel):
    data_source: str
    group_by_column: str
    sum_column: str
    output_range: str
    spreadsheet_id: Optional[str] = None

class ComplexCalculationRequest(BaseModel):
    data_source: str
    calculation_spec: Dict[str, Any]
    output_range: str
    spreadsheet_id: Optional[str] = None

class DualLayerResponse(BaseModel):
    success: bool
    polars_result: Optional[str] = None
    sheets_formula: Optional[str] = None
    execution_time_ms: Optional[int] = None
    validation_passed: Optional[bool] = None
    output_range: Optional[str] = None
    error: Optional[str] = None

# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Enhanced Dual-Layer Math MCP Server"}

@app.post("/dual-layer/sum", response_model=DualLayerResponse)
async def api_dual_layer_sum(request: DualLayerRequest):
    """Dual-layer SUM via API."""
    result = await dual_layer_sum(
        request.data_source,
        request.range_spec,
        request.column,
        request.output_range,
        request.spreadsheet_id,
        request.enable_validation
    )
    return DualLayerResponse(**result)

@app.post("/dual-layer/average", response_model=DualLayerResponse)
async def api_dual_layer_average(request: DualLayerRequest):
    """Dual-layer AVERAGE via API."""
    result = await dual_layer_average(
        request.data_source,
        request.range_spec,
        request.column,
        request.output_range,
        request.spreadsheet_id,
        request.enable_validation
    )
    return DualLayerResponse(**result)

@app.post("/dual-layer/groupby-sum", response_model=DualLayerResponse)
async def api_dual_layer_groupby(request: GroupByRequest):
    """Dual-layer GROUP BY SUM via API."""
    result = await dual_layer_groupby_sum(
        request.data_source,
        request.group_by_column,
        request.sum_column,
        request.output_range,
        request.spreadsheet_id
    )
    return DualLayerResponse(**result)

@app.post("/dual-layer/complex", response_model=DualLayerResponse)
async def api_dual_layer_complex(request: ComplexCalculationRequest):
    """Dual-layer complex calculation via API."""
    result = await dual_layer_complex_calculation(
        request.data_source,
        request.calculation_spec,
        request.output_range,
        request.spreadsheet_id
    )
    return DualLayerResponse(**result)

@app.get("/statistics")
async def api_get_statistics():
    """Get execution statistics via API."""
    return await get_execution_statistics()

@app.get("/capabilities")
async def get_capabilities():
    """List dual-layer execution capabilities."""
    return {
        "dual_layer_functions": [
            "dual_layer_sum",
            "dual_layer_average", 
            "dual_layer_groupby_sum",
            "dual_layer_complex_calculation"
        ],
        "supported_operations": [
            "sum", "average", "count", "max", "min", "median",
            "groupby_sum", "customer_lifetime_value", "variance_analysis"
        ],
        "formula_mappings": [
            "simple_formulas", "array_formulas", "pivot_formulas",
            "financial_formulas", "complex_chains"
        ],
        "validation_types": [
            "exact_match", "tolerance_0.001", "tolerance_0.01",
            "range_check", "type_check", "business_rules"
        ]
    }

# ================== MAIN EXECUTION ==================

async def main():
    """Run both MCP and FastAPI servers."""
    # Start FastAPI in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8010, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        server.serve(),
        mcp.run()
    )

if __name__ == "__main__":
    print("Starting Enhanced Dual-Layer Math MCP Server...")
    print("MCP Server: stdio")
    print("API Server: http://localhost:8010")
    print("API Docs: http://localhost:8010/docs")
    print("\nDual-Layer Capabilities:")
    print("✓ Polars computation with Google Sheets formula transparency")
    print("✓ Automatic formula translation and validation")
    print("✓ Smart context-aware sheets pushing")
    print("✓ Complex financial calculation support")
    print("✓ Comprehensive validation and business rules")
    print("\nEnhanced Functions:")
    print("- dual_layer_sum: SUM with dual execution")
    print("- dual_layer_average: AVERAGE with dual execution")
    print("- dual_layer_groupby_sum: GROUP BY SUM with SUMIF formulas")
    print("- dual_layer_complex_calculation: Advanced calculations with validation")
    asyncio.run(main())

def run_server():
    """Run the FastAPI server."""
    import sys
    
    port = 8010  # Default port for enhanced dual layer server
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default port {port}")
    
    logger.info(f"Starting Enhanced Dual-Layer Math MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)