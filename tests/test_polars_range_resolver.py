"""Tests for PolarsRangeResolver and enhanced math functions with range support"""
import pytest
import polars as pl
import sys
import os
from pathlib import Path
import tempfile
import json

# Add the mcp_tooling directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_tooling"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_tooling" / "google_sheets" / "api"))

from polars_range_resolver import PolarsRangeResolver
from standalone_math_functions_enhanced import (
    SUM, AVERAGE, MIN, MAX, MEDIAN, MODE, PERCENTILE,
    apply_range_to_dataframe, SimpleRunContext, SimpleFinnDeps
)


class TestPolarsRangeResolver:
    """Test suite for PolarsRangeResolver"""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pl.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [100, 200, 300, 400, 500],
            "D": ["a", "b", "c", "d", "e"]
        })
    
    @pytest.fixture
    def context(self):
        """Create a simple context for testing"""
        thread_dir = Path("/tmp/test_thread")
        workspace_dir = Path("/tmp/test_workspace")
        deps = SimpleFinnDeps(thread_dir, workspace_dir)
        return SimpleRunContext(deps)
    
    def test_sheets_to_polars_cell_range(self):
        """Test conversion of cell range from A1 notation to Polars indices"""
        result = PolarsRangeResolver.sheets_to_polars("A1:C3")
        assert result["row_slice"] == slice(0, 3)
        assert result["col_slice"] == slice(0, 3)
        assert result["type"] == "cell_range"
    
    def test_sheets_to_polars_column_range(self):
        """Test conversion of column range"""
        result = PolarsRangeResolver.sheets_to_polars("B:D")
        assert result["row_slice"] == slice(None)
        assert result["col_slice"] == slice(1, 4)
        assert result["type"] == "column_range"
    
    def test_sheets_to_polars_row_range(self):
        """Test conversion of row range"""
        result = PolarsRangeResolver.sheets_to_polars("2:5")
        assert result["row_slice"] == slice(1, 5)
        assert result["col_slice"] == slice(None)
        assert result["type"] == "row_range"
    
    def test_sheets_to_polars_single_cell(self):
        """Test conversion of single cell"""
        result = PolarsRangeResolver.sheets_to_polars("B3")
        assert result["row_slice"] == slice(2, 3)
        assert result["col_slice"] == slice(1, 2)
        assert result["type"] == "single_cell"
    
    def test_polars_to_sheets(self, sample_df):
        """Test conversion from Polars slice to A1 notation"""
        result = PolarsRangeResolver.polars_to_sheets(
            sample_df,
            row_slice=slice(1, 3),
            col_slice=slice(0, 2)
        )
        assert result == "A2:B3"
    
    def test_resolve_range_with_a1_notation(self, sample_df):
        """Test resolving range with A1 notation"""
        result = PolarsRangeResolver.resolve_range(sample_df, "A1:B3")
        assert len(result) == 3
        assert result.columns == ["A", "B"]
        assert result["A"].to_list() == [1, 2, 3]
        assert result["B"].to_list() == [10, 20, 30]
    
    def test_resolve_range_with_column_spec(self, sample_df):
        """Test resolving range with column specification"""
        result = PolarsRangeResolver.resolve_range(sample_df, "B:B")
        assert len(result) == 5
        assert result.columns == ["B"]
        assert result["B"].to_list() == [10, 20, 30, 40, 50]
    
    def test_resolve_range_with_dict_spec(self, sample_df):
        """Test resolving range with dictionary specification"""
        range_spec = {
            "row_slice": slice(1, 4),
            "columns": ["A", "C"]
        }
        result = PolarsRangeResolver.resolve_range(sample_df, range_spec)
        assert len(result) == 3
        assert result.columns == ["A", "C"]
        assert result["A"].to_list() == [2, 3, 4]
        assert result["C"].to_list() == [200, 300, 400]
    
    def test_create_range_spec_with_rows(self):
        """Test creating range specification with row parameters"""
        # Single row
        spec = PolarsRangeResolver.create_range_spec(rows=2)
        assert spec["row_slice"] == slice(2, 3)
        
        # Row range
        spec = PolarsRangeResolver.create_range_spec(rows=(1, 5))
        assert spec["row_slice"] == slice(1, 5)
        
        # Row slice
        spec = PolarsRangeResolver.create_range_spec(rows=slice(0, 10))
        assert spec["row_slice"] == slice(0, 10)
    
    def test_create_range_spec_with_columns(self):
        """Test creating range specification with column parameters"""
        # Single column
        spec = PolarsRangeResolver.create_range_spec(columns="revenue")
        assert spec["columns"] == ["revenue"]
        
        # Multiple columns
        spec = PolarsRangeResolver.create_range_spec(columns=["A", "B", "C"])
        assert spec["columns"] == ["A", "B", "C"]
        
        # Column range with letters
        spec = PolarsRangeResolver.create_range_spec(columns=("A", "C"))
        assert spec["col_slice"] == slice(0, 3)
    
    def test_expand_range_for_operation(self, sample_df):
        """Test range expansion based on operation type"""
        # Single cell expansion for aggregation
        range_spec = "A1"
        expanded = PolarsRangeResolver.expand_range_for_operation(
            sample_df, range_spec, "sum"
        )
        assert expanded["row_slice"] == slice(None)
        assert expanded["col_slice"] == slice(0, 1)
        
        # Pivot operation validation
        with pytest.raises(ValueError, match="Cannot perform pivot"):
            PolarsRangeResolver.expand_range_for_operation(
                sample_df, "A1", "pivot"
            )
    
    def test_validate_range_for_dataframe(self, sample_df):
        """Test range validation against DataFrame"""
        # Valid range
        is_valid, error = PolarsRangeResolver.validate_range_for_dataframe(
            sample_df, "A1:B3"
        )
        assert is_valid is True
        assert error is None
        
        # Row out of bounds
        is_valid, error = PolarsRangeResolver.validate_range_for_dataframe(
            sample_df, "A1:A10"
        )
        assert is_valid is False
        assert "Row range exceeds" in error
        
        # Column out of bounds
        is_valid, error = PolarsRangeResolver.validate_range_for_dataframe(
            sample_df, "A1:Z1"
        )
        assert is_valid is False
        assert "Column range exceeds" in error
    
    def test_column_letter_conversion(self):
        """Test column letter to index conversion"""
        assert PolarsRangeResolver._col_to_index("A") == 0
        assert PolarsRangeResolver._col_to_index("B") == 1
        assert PolarsRangeResolver._col_to_index("Z") == 25
        assert PolarsRangeResolver._col_to_index("AA") == 26
        assert PolarsRangeResolver._col_to_index("AB") == 27
        assert PolarsRangeResolver._col_to_index("AZ") == 51
        assert PolarsRangeResolver._col_to_index("BA") == 52
        
        # Test reverse conversion
        assert PolarsRangeResolver._index_to_col(0) == "A"
        assert PolarsRangeResolver._index_to_col(1) == "B"
        assert PolarsRangeResolver._index_to_col(25) == "Z"
        assert PolarsRangeResolver._index_to_col(26) == "AA"
        assert PolarsRangeResolver._index_to_col(27) == "AB"
        assert PolarsRangeResolver._index_to_col(51) == "AZ"
        assert PolarsRangeResolver._index_to_col(52) == "BA"


class TestEnhancedMathFunctions:
    """Test suite for enhanced math functions with range support"""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pl.DataFrame({
            "revenue": [100, 200, 300, 400, 500],
            "cost": [50, 100, 150, 200, 250],
            "profit": [50, 100, 150, 200, 250],
            "category": ["A", "B", "A", "B", "A"]
        })
    
    @pytest.fixture
    def context(self):
        """Create a simple context for testing"""
        thread_dir = Path("/tmp/test_thread")
        workspace_dir = Path("/tmp/test_workspace")
        deps = SimpleFinnDeps(thread_dir, workspace_dir)
        return SimpleRunContext(deps)
    
    @pytest.fixture
    def temp_csv_file(self, sample_df):
        """Create a temporary CSV file with sample data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.write_csv(f.name)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_parquet_file(self, sample_df):
        """Create a temporary Parquet file with sample data"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            sample_df.write_parquet(f.name)
            yield f.name
        os.unlink(f.name)
    
    def test_sum_with_range_csv(self, context, temp_csv_file):
        """Test SUM function with range specification on CSV file"""
        # Sum first 3 rows of revenue column
        result = SUM(context, temp_csv_file, range_spec="A1:A3", column="revenue")
        assert result == 600  # 100 + 200 + 300
        
        # Sum all values in cost column
        result = SUM(context, temp_csv_file, range_spec="B:B", column="cost")
        assert result == 750  # 50 + 100 + 150 + 200 + 250
    
    def test_average_with_range_parquet(self, context, temp_parquet_file):
        """Test AVERAGE function with range specification on Parquet file"""
        # Average of first 2 rows of profit column
        result = AVERAGE(context, temp_parquet_file, range_spec="C1:C2", column="profit")
        assert result == 75  # (50 + 100) / 2
        
        # Average of all revenue values
        result = AVERAGE(context, temp_parquet_file, column="revenue")
        assert result == 300  # (100 + 200 + 300 + 400 + 500) / 5
    
    def test_min_max_with_range(self, context, temp_csv_file):
        """Test MIN and MAX functions with range specification"""
        # Min in specific range
        min_result = MIN(context, temp_csv_file, range_spec="A2:A4", column="revenue")
        assert min_result == 200  # Min of [200, 300, 400]
        
        # Max in specific range
        max_result = MAX(context, temp_csv_file, range_spec="A2:A4", column="revenue")
        assert max_result == 400  # Max of [200, 300, 400]
    
    def test_median_with_range(self, context, temp_csv_file):
        """Test MEDIAN function with range specification"""
        # Median of first 3 rows
        result = MEDIAN(context, temp_csv_file, range_spec="1:3", column="cost")
        assert result == 100  # Median of [50, 100, 150]
    
    def test_percentile_with_range(self, context, temp_csv_file):
        """Test PERCENTILE function with range specification"""
        # 75th percentile of profit column
        result = PERCENTILE(context, temp_csv_file, 75, range_spec="C:C", column="profit")
        assert result == 200  # 75th percentile of [50, 100, 150, 200, 250]
    
    def test_apply_range_to_dataframe(self, sample_df):
        """Test apply_range_to_dataframe helper function"""
        # Extract values from specific range and column
        values = apply_range_to_dataframe(sample_df, "A1:A3", "revenue")
        assert values == [100, 200, 300]
        
        # Extract full DataFrame slice
        df_slice = apply_range_to_dataframe(sample_df, "A1:B3")
        assert isinstance(df_slice, pl.DataFrame)
        assert len(df_slice) == 3
        assert df_slice.columns == ["revenue", "cost"]
    
    def test_mode_with_range(self, context):
        """Test MODE function with range specification"""
        # Create test data with repeated values
        test_data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        result = MODE(context, test_data)
        assert result == 4  # Most common value
        
        # Test with DataFrame
        df = pl.DataFrame({"values": test_data})
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.write_csv(f.name)
            try:
                result = MODE(context, f.name, column="values")
                assert result == 4
            finally:
                os.unlink(f.name)
    
    def test_error_handling(self, context):
        """Test error handling for invalid inputs"""
        # Invalid file path
        with pytest.raises(Exception):
            SUM(context, "nonexistent.csv")
        
        # Invalid range specification
        with pytest.raises(ValueError):
            range_spec = PolarsRangeResolver.sheets_to_polars("INVALID")
        
        # Empty values
        with pytest.raises(ValueError):
            AVERAGE(context, [])


def test_integration_with_complex_range():
    """Integration test with complex range specifications"""
    # Create a larger test DataFrame
    df = pl.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "Sales": [1000, 1500, 1200, 1800, 2000, 2200],
        "Costs": [800, 900, 850, 1000, 1100, 1200],
        "Profit": [200, 600, 350, 800, 900, 1000]
    })
    
    # Test various range operations
    # 1. Q1 Sales total (first 3 months)
    q1_slice = PolarsRangeResolver.resolve_range(df, "B1:B3")
    q1_sales = q1_slice["Sales"].sum()
    assert q1_sales == 3700  # 1000 + 1500 + 1200
    
    # 2. Q2 Average profit
    q2_slice = PolarsRangeResolver.resolve_range(df, "D4:D6")
    q2_avg_profit = q2_slice["Profit"].mean()
    assert q2_avg_profit == 900  # (800 + 900 + 1000) / 3
    
    # 3. Full column operations
    costs_column = PolarsRangeResolver.resolve_range(df, "C:C")
    total_costs = costs_column["Costs"].sum()
    assert total_costs == 5850


if __name__ == "__main__":
    # Run specific test if needed
    pytest.main([__file__, "-v"])