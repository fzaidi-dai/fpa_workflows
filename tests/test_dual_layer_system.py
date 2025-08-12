#!/usr/bin/env python3
"""
Test Suite for Dual-Layer Execution System

Tests the integration between Polars computation and Google Sheets formula
translation in the dual-layer architecture.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

import polars as pl

# Import dual-layer components
from mcp_tooling.dual_layer import (
    DualLayerExecutor,
    FormulaTranslator,
    SmartSheetsPusher,
    DualLayerValidator,
    PlanStep,
    StepResult,
    CellContext,
    CellType,
    OperationType
)


class TestDualLayerSystem:
    """Test the dual-layer execution system"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        return pl.DataFrame({
            "product": ["A", "B", "C", "A", "B", "C"],
            "region": ["North", "North", "South", "South", "North", "South"],
            "revenue": [100, 150, 200, 120, 180, 220],
            "units": [10, 15, 20, 12, 18, 22],
            "month": ["Jan", "Jan", "Jan", "Feb", "Feb", "Feb"]
        })
    
    @pytest.fixture
    def temp_data_file(self, sample_data):
        """Create temporary data file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.write_csv(f.name)
            yield f.name
        
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    @pytest.fixture
    def dual_layer_executor(self):
        """Create dual layer executor for testing"""
        # Use test mapping directory if it exists
        mappings_dir = Path(__file__).parent.parent / "mcp_tooling" / "formula_mappings"
        
        return DualLayerExecutor(
            mappings_dir=mappings_dir,
            batch_size=10,
            enable_validation=True
        )
    
    def test_formula_translator_basic(self):
        """Test basic formula translation"""
        translator = FormulaTranslator()
        
        # Test SUM translation
        operation = {"type": "sum", "column": "revenue"}
        context = {"data_range": "C:C"}
        
        formula = translator.translate_operation(operation, context)
        assert formula == "=SUM(C:C)"
        
        # Test AVERAGE translation
        operation = {"type": "average", "column": "revenue"}
        formula = translator.translate_operation(operation, context)
        assert formula == "=AVERAGE(C:C)"
    
    def test_formula_translator_complex(self):
        """Test complex formula translation"""
        translator = FormulaTranslator()
        
        # Test VLOOKUP translation
        operation = {
            "type": "vlookup",
            "lookup_value": "Product A",
            "table_range": "A:D",
            "col_index": 3,
            "range_lookup": False
        }
        context = {}
        
        formula = translator.translate_operation(operation, context)
        expected = '=VLOOKUP("Product A", A:D, 3, FALSE)'
        assert formula == expected
    
    def test_formula_complexity_detection(self):
        """Test formula complexity detection"""
        translator = FormulaTranslator()
        
        # Simple formula
        simple_formula = "=SUM(A:A)"
        complexity = translator.get_formula_complexity(simple_formula)
        assert complexity == "simple"
        
        # Complex formula
        complex_formula = "=ARRAYFORMULA(SUM(IF(A:A>100, B:B*C:C, 0)))"
        complexity = translator.get_formula_complexity(complex_formula)
        assert complexity == "complex"
    
    def test_dual_layer_validator(self):
        """Test dual-layer validation"""
        validator = DualLayerValidator(default_tolerance=0.001)
        
        # Test exact match
        assert validator._values_match(100, 100, 0.001) == True
        assert validator._values_match(100, 101, 0.001) == False
        
        # Test tolerance match
        assert validator._values_match(100.0, 100.0001, 0.001) == True
        assert validator._values_match(100.0, 100.1, 0.001) == False
        
        # Test list match
        list1 = [1, 2, 3]
        list2 = [1.0001, 2.0001, 3.0001]
        assert validator._list_match(list1, list2, 0.001) == True
    
    def test_validation_checks(self):
        """Test validation check system"""
        validator = DualLayerValidator()
        
        # Test range validation
        rule = {"type": "range_check", "min": 0, "max": 100}
        check = validator._check_range_rule(50, rule, "test_range")
        assert check.result.value == "pass"
        
        check = validator._check_range_rule(150, rule, "test_range")
        assert check.result.value == "fail"
        
        # Test type validation
        rule = {"type": "type_check", "expected_type": "numeric"}
        check = validator._check_type_rule(42, rule, "test_type")
        assert check.result.value == "pass"
        
        check = validator._check_type_rule("not a number", rule, "test_type")
        assert check.result.value == "fail"
    
    def test_smart_sheets_pusher(self):
        """Test smart sheets pusher functionality"""
        pusher = SmartSheetsPusher(batch_size=5)
        
        # Test value push
        context = CellContext(
            range_spec="A1:A3",
            cell_type=CellType.SOURCE_DATA,
            values=[[1], [2], [3]]
        )
        
        result = pusher.push_with_context(context)
        assert result == True
        
        # Test formula push
        context = CellContext(
            range_spec="B1",
            cell_type=CellType.CALCULATION,
            formula="=SUM(A1:A3)"
        )
        
        result = pusher.push_with_context(context)
        assert result == True
        
        # Check batch execution
        result = pusher.execute_pending_batch()
        assert result == True
    
    @pytest.mark.asyncio
    async def test_dual_layer_executor_sum(self, dual_layer_executor, temp_data_file):
        """Test dual layer executor with SUM operation"""
        
        # Create plan step
        step = PlanStep(
            step_id="test_sum",
            description="Test SUM operation",
            operation_type=OperationType.AGGREGATION,
            operation={
                "type": "sum",
                "column": "revenue"
            },
            input_data=temp_data_file,
            output_range="Results!B2",
            sheet_context={
                "data_range": "C:C",
                "sheet_name": "Results",
                "current_cell": "B2"
            }
        )
        
        # Execute step
        result = await dual_layer_executor.execute_step(step)
        
        # Verify result
        assert result.success == True
        assert result.polars_result == 970.0  # Sum of sample revenue data
        assert "=SUM(" in result.sheets_formula
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_dual_layer_executor_average(self, dual_layer_executor, temp_data_file):
        """Test dual layer executor with AVERAGE operation"""
        
        step = PlanStep(
            step_id="test_avg",
            description="Test AVERAGE operation",
            operation_type=OperationType.AGGREGATION,
            operation={
                "type": "average",
                "column": "revenue"
            },
            input_data=temp_data_file,
            output_range="Results!B3",
            sheet_context={
                "data_range": "C:C",
                "sheet_name": "Results",
                "current_cell": "B3"
            }
        )
        
        result = await dual_layer_executor.execute_step(step)
        
        assert result.success == True
        assert abs(float(result.polars_result) - 161.67) < 0.1  # Average of sample data
        assert "=AVERAGE(" in result.sheets_formula
    
    @pytest.mark.asyncio 
    async def test_dual_layer_executor_groupby(self, dual_layer_executor, temp_data_file):
        """Test dual layer executor with GROUP BY operation"""
        
        step = PlanStep(
            step_id="test_groupby",
            description="Test GROUP BY SUM operation",
            operation_type=OperationType.AGGREGATION,
            operation={
                "type": "groupby_sum",
                "group_by": "region",
                "sum_column": "revenue"
            },
            input_data=temp_data_file,
            output_range="Results!A5:B7",
            sheet_context={
                "data_range": "A:Z",
                "sheet_name": "Results"
            }
        )
        
        result = await dual_layer_executor.execute_step(step)
        
        assert result.success == True
        assert result.polars_result is not None
        # Should contain SUMIF formula for sheets translation
        assert result.sheets_formula is not None
    
    def test_execution_statistics(self, dual_layer_executor):
        """Test execution statistics tracking"""
        
        initial_stats = dual_layer_executor.get_execution_stats()
        assert initial_stats["steps_executed"] == 0
        
        # Reset stats
        dual_layer_executor.reset_stats()
        reset_stats = dual_layer_executor.get_execution_stats()
        assert reset_stats["steps_executed"] == 0
    
    def test_cell_context_creation(self):
        """Test cell context creation and serialization"""
        
        context = CellContext(
            range_spec="A1:B2",
            cell_type=CellType.COMPLEX_CALCULATION,
            formula="=SUM(A1:A2)*2",
            computed_value=100,
            validation_note="Complex calculation verified"
        )
        
        # Test serialization
        context_dict = context.to_dict()
        assert context_dict["range_spec"] == "A1:B2"
        assert context_dict["cell_type"] == "complex_calculation"
        assert context_dict["formula"] == "=SUM(A1:A2)*2"
        assert context_dict["computed_value"] == 100
    
    def test_business_rule_validation(self):
        """Test business rule validation"""
        validator = DualLayerValidator()
        
        # Test positive values rule
        assert validator._check_positive(100) == True
        assert validator._check_positive(-50) == False
        assert validator._check_positive([10, 20, 30]) == True
        assert validator._check_positive([10, -20, 30]) == False
        
        # Test no nulls rule
        assert validator._check_no_nulls(42) == True
        assert validator._check_no_nulls(None) == False
        assert validator._check_no_nulls([1, 2, 3]) == True
        assert validator._check_no_nulls([1, None, 3]) == False
    
    def test_validation_summary(self):
        """Test validation summary generation"""
        from mcp_tooling.dual_layer.data_models import ValidationCheck, ValidationResult
        
        validator = DualLayerValidator()
        
        checks = [
            ValidationCheck("check1", ValidationResult.PASS, "Passed"),
            ValidationCheck("check2", ValidationResult.PASS, "Passed"),
            ValidationCheck("check3", ValidationResult.FAIL, "Failed"),
            ValidationCheck("check4", ValidationResult.WARNING, "Warning")
        ]
        
        summary = validator.get_validation_summary(checks)
        
        assert summary["total_checks"] == 4
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["warnings"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["all_passed"] == False
        assert summary["has_warnings"] == True


@pytest.mark.asyncio
async def test_integration_workflow():
    """Integration test for complete dual-layer workflow"""
    
    # Create sample data
    data = pl.DataFrame({
        "category": ["A", "B", "A", "B", "C"],
        "value": [100, 200, 150, 250, 300]
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.write_csv(f.name)
        temp_file = f.name
    
    try:
        # Create dual layer executor
        executor = DualLayerExecutor()
        
        # Create execution step
        step = PlanStep(
            step_id="integration_test",
            description="Integration test - sum by category",
            operation_type=OperationType.AGGREGATION,
            operation={
                "type": "sum",
                "column": "value"
            },
            input_data=temp_file,
            output_range="Test!C1",
            sheet_context={
                "data_range": "B:B",
                "sheet_name": "Test",
                "current_cell": "C1"
            },
            validation_rules=[
                {"type": "range_check", "min": 0, "max": 10000},
                {"type": "type_check", "expected_type": "numeric"}
            ]
        )
        
        # Execute
        result = await executor.execute_step(step)
        
        # Verify integration
        assert result.success == True
        assert result.polars_result == 1000.0
        assert "=SUM(" in result.sheets_formula
        assert result.all_validations_passed == True
        
        # Check statistics
        stats = executor.get_execution_stats()
        assert stats["steps_executed"] == 1
        assert stats["validations_passed"] >= 1
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])