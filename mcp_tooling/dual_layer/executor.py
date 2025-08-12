"""
Dual-Layer Executor for FPA Agents

Executes computations with Polars for performance while pushing formulas to Google Sheets
for transparency, implementing the core dual-layer architecture.
"""

import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

import polars as pl

from .data_models import (
    PlanStep, StepResult, CellContext, CellType, PolarsExecutor, SheetsPusher,
    ValidationCheck, ValidationResult
)
from .formula_translator import FormulaTranslator
from .sheets_pusher import SmartSheetsPusher
from .validator import DualLayerValidator

# Set up logger
logger = logging.getLogger(__name__)


class DualLayerExecutor:
    """Executes computations with Polars, pushes formulas to Sheets"""
    
    def __init__(self, 
                 mappings_dir: Optional[Union[str, Path]] = None,
                 batch_size: int = 50,
                 enable_validation: bool = True,
                 default_tolerance: float = 0.001):
        """
        Initialize the dual-layer executor
        
        Args:
            mappings_dir: Directory containing formula mappings
            batch_size: Batch size for Google Sheets operations
            enable_validation: Whether to validate Polars vs Sheets consistency
            default_tolerance: Default tolerance for validation
        """
        self.polars_executor = PolarsExecutor()
        self.sheets_pusher = SmartSheetsPusher(batch_size=batch_size)
        self.formula_translator = FormulaTranslator(mappings_dir)
        self.validator = DualLayerValidator(
            default_tolerance=default_tolerance,
            enable_business_rules=enable_validation
        )
        
        self.enable_validation = enable_validation
        self.execution_stats = {
            "steps_executed": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "total_execution_time_ms": 0
        }
    
    async def execute_step(self, step: PlanStep) -> StepResult:
        """
        Execute a single plan step using dual-layer architecture
        
        Args:
            step: The plan step to execute
            
        Returns:
            StepResult with execution details and validation results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Executing step {step.step_id}: {step.description}")
            
            # 1. Compute with Polars
            polars_result = self.polars_executor.execute(
                step.operation,
                step.input_data
            )
            
            # 2. Translate to Sheets formula
            sheets_formula = self.formula_translator.translate_operation(
                step.operation,
                step.sheet_context
            )
            
            # 3. Create cell context for pushing
            cell_contexts = self._create_cell_contexts(
                step, 
                polars_result, 
                sheets_formula
            )
            
            # 4. Push to Sheets
            push_success = await self._push_to_sheets(cell_contexts)
            
            if not push_success:
                raise Exception("Failed to push data to Google Sheets")
            
            # 5. Validate consistency (if enabled)
            validation_checks = []
            if self.enable_validation:
                validation_checks = self.validator.validate_dual_layer(
                    polars_result,
                    sheets_formula,
                    step.validation_rules
                )
            
            # 6. Update statistics
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_stats(validation_checks, execution_time_ms)
            
            # 7. Create result
            result = StepResult(
                step_id=step.step_id,
                success=True,
                polars_result=polars_result,
                sheets_formula=sheets_formula,
                validation=validation_checks,
                execution_time_ms=execution_time_ms,
                sheets_range=step.output_range,
                cell_contexts=cell_contexts
            )
            
            logger.info(f"Step {step.step_id} completed successfully in {execution_time_ms}ms")
            return result
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Step {step.step_id} failed: {str(e)}"
            logger.error(error_msg)
            
            return StepResult(
                step_id=step.step_id,
                success=False,
                polars_result=None,
                error=error_msg,
                execution_time_ms=execution_time_ms
            )
    
    def _create_cell_contexts(self, 
                             step: PlanStep, 
                             polars_result: Any, 
                             sheets_formula: str) -> List[CellContext]:
        """Create cell contexts for different types of data"""
        contexts = []
        
        # Determine cell type based on operation and complexity
        cell_type = self._determine_cell_type(step, sheets_formula)
        
        if cell_type == CellType.SOURCE_DATA:
            # Push raw values only
            contexts.append(CellContext(
                range_spec=step.output_range,
                cell_type=cell_type,
                values=self._format_values_for_sheets(polars_result)
            ))
            
        elif cell_type == CellType.CALCULATION:
            # Push formula only
            contexts.append(CellContext(
                range_spec=step.output_range,
                cell_type=cell_type,
                formula=sheets_formula
            ))
            
        elif cell_type == CellType.COMPLEX_CALCULATION:
            # Push formula with validation note
            contexts.append(CellContext(
                range_spec=step.output_range,
                cell_type=cell_type,
                formula=sheets_formula,
                computed_value=polars_result,
                validation_note=f"Operation: {step.operation.get('type', 'unknown')}"
            ))
            
        elif cell_type == CellType.PIVOT_TABLE:
            # Create pivot table
            pivot_config = self._create_pivot_config(step, polars_result)
            contexts.append(CellContext(
                range_spec=step.output_range,
                cell_type=cell_type,
                pivot_config=pivot_config
            ))
        
        return contexts
    
    def _determine_cell_type(self, step: PlanStep, sheets_formula: str) -> CellType:
        """Determine the appropriate cell type for the step"""
        
        # Check if it's source data (no formula, just values)
        if step.operation.get("type") in ["load_data", "raw_values"]:
            return CellType.SOURCE_DATA
        
        # Check for pivot operations
        if step.operation.get("type") in ["pivot", "groupby_pivot"]:
            return CellType.PIVOT_TABLE
        
        # Check formula complexity
        complexity = self.formula_translator.get_formula_complexity(sheets_formula)
        
        if complexity == "complex":
            return CellType.COMPLEX_CALCULATION
        else:
            return CellType.CALCULATION
    
    def _format_values_for_sheets(self, polars_result: Any) -> List[List[Any]]:
        """Format Polars result for Google Sheets values format"""
        
        if isinstance(polars_result, pl.DataFrame):
            # Convert DataFrame to 2D list
            return polars_result.to_numpy().tolist()
        elif isinstance(polars_result, list):
            # Convert list to column format
            return [[item] for item in polars_result]
        elif isinstance(polars_result, (int, float, str)):
            # Single value
            return [[polars_result]]
        else:
            # Convert to string representation
            return [[str(polars_result)]]
    
    def _create_pivot_config(self, step: PlanStep, polars_result: Any) -> Dict[str, Any]:
        """Create pivot table configuration from operation"""
        operation = step.operation
        
        return {
            "source_range": operation.get("source_range", "A:Z"),
            "rows": operation.get("pivot_rows", []),
            "columns": operation.get("pivot_columns", []),
            "values": operation.get("pivot_values", []),
            "filters": operation.get("pivot_filters", []),
            "value_layout": operation.get("value_layout", "HORIZONTAL")
        }
    
    async def _push_to_sheets(self, cell_contexts: List[CellContext]) -> bool:
        """Push all cell contexts to Google Sheets"""
        success = True
        
        for context in cell_contexts:
            try:
                result = self.sheets_pusher.push_with_context(context)
                success &= result
            except Exception as e:
                logger.error(f"Failed to push context {context.range_spec}: {e}")
                success = False
        
        # Flush any remaining operations
        success &= self.sheets_pusher.flush()
        
        return success
    
    def _update_stats(self, validation_checks: List[ValidationCheck], execution_time_ms: int):
        """Update execution statistics"""
        self.execution_stats["steps_executed"] += 1
        self.execution_stats["total_execution_time_ms"] += execution_time_ms
        
        if validation_checks:
            passed = sum(1 for check in validation_checks if check.result == ValidationResult.PASS)
            failed = sum(1 for check in validation_checks if check.result == ValidationResult.FAIL)
            
            self.execution_stats["validations_passed"] += passed
            self.execution_stats["validations_failed"] += failed
    
    async def execute_plan(self, plan) -> Dict[str, Any]:
        """
        Execute a complete execution plan
        
        Args:
            plan: ExecutionPlan with multiple steps
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Starting execution of plan {plan.plan_id} with {len(plan.steps)} steps")
        
        results = []
        failed_steps = []
        
        for step in plan.steps:
            result = await self.execute_step(step)
            results.append(result)
            
            if not result.success:
                failed_steps.append(result.step_id)
                
                # Stop on failure (could be configurable)
                logger.error(f"Plan execution stopped due to failed step: {result.step_id}")
                break
        
        # Calculate summary
        successful_steps = [r for r in results if r.success]
        total_execution_time = sum(r.execution_time_ms or 0 for r in results)
        
        summary = {
            "plan_id": plan.plan_id,
            "success": len(failed_steps) == 0,
            "total_steps": len(plan.steps),
            "completed_steps": len(results),
            "successful_steps": len(successful_steps),
            "failed_steps": failed_steps,
            "total_execution_time_ms": total_execution_time,
            "results": [result.to_dict() for result in results]
        }
        
        logger.info(f"Plan execution completed. Success: {summary['success']}")
        return summary
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get current execution statistics"""
        stats = self.execution_stats.copy()
        
        # Add calculated metrics
        if stats["steps_executed"] > 0:
            stats["average_execution_time_ms"] = (
                stats["total_execution_time_ms"] / stats["steps_executed"]
            )
            
        total_validations = stats["validations_passed"] + stats["validations_failed"]
        if total_validations > 0:
            stats["validation_pass_rate"] = (
                stats["validations_passed"] / total_validations
            )
        
        # Add pusher stats
        stats["pusher_stats"] = self.sheets_pusher.get_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_stats = {
            "steps_executed": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "total_execution_time_ms": 0
        }
        logger.info("Execution statistics reset")


class BatchExecutor:
    """Optimized executor for batch operations"""
    
    def __init__(self, dual_layer_executor: DualLayerExecutor):
        self.executor = dual_layer_executor
        
    async def execute_batch(self, steps: List[PlanStep]) -> List[StepResult]:
        """Execute multiple steps as a batch for optimization"""
        
        # Group steps by operation type for optimization
        grouped_steps = self._group_steps_by_type(steps)
        
        results = []
        
        for operation_type, step_group in grouped_steps.items():
            logger.info(f"Executing batch of {len(step_group)} {operation_type} operations")
            
            # Execute each step in the group
            for step in step_group:
                result = await self.executor.execute_step(step)
                results.append(result)
        
        return results
    
    def _group_steps_by_type(self, steps: List[PlanStep]) -> Dict[str, List[PlanStep]]:
        """Group steps by operation type for batch optimization"""
        groups = {}
        
        for step in steps:
            operation_type = step.operation.get("type", "unknown")
            
            if operation_type not in groups:
                groups[operation_type] = []
            
            groups[operation_type].append(step)
        
        return groups