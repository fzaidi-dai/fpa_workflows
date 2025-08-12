"""
Test Suite for Excel Style Miscellaneous Utility Functions

This test suite validates all functions in the excel_style_misc_utils module,
ensuring they work correctly with various input types and edge cases.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import polars as pl
import tempfile
import os
import platform
import sys
from unittest.mock import Mock

# Import the functions to test
from tools.core_data_and_math_utils.excel_style_misc_utils.excel_style_misc_utils import (
    FORMULATEXT,
    TRANSPOSE,
    CELL,
    INFO,
    N,
    T,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)


class MockRunContext:
    """Mock RunContext for testing"""
    def __init__(self):
        self.deps = Mock()
        # Create temporary directories for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.deps.analysis_dir = self.temp_dir / "analysis"
        self.deps.data_dir = self.temp_dir / "data"
        self.deps.analysis_dir.mkdir(exist_ok=True)
        self.deps.data_dir.mkdir(exist_ok=True)


@pytest.fixture
def mock_ctx():
    """Fixture providing mock run context"""
    return MockRunContext()


class TestFORMULATEXT:
    """Test cases for FORMULATEXT function"""

    def test_formulatext_string_reference(self, mock_ctx):
        """Test FORMULATEXT with string cell reference"""
        result = FORMULATEXT(mock_ctx, "A1")
        assert result == "=SUM(A1:Z1)"

    def test_formulatext_dict_with_formula(self, mock_ctx):
        """Test FORMULATEXT with dictionary containing formula"""
        reference = {"formula": "=AVERAGE(C1:C5)"}
        result = FORMULATEXT(mock_ctx, reference)
        assert result == "=AVERAGE(C1:C5)"

    def test_formulatext_dict_with_cell(self, mock_ctx):
        """Test FORMULATEXT with dictionary containing cell reference"""
        reference = {"cell": "B2"}
        result = FORMULATEXT(mock_ctx, reference)
        assert result == "=AVERAGE(B2:B2)"

    def test_formulatext_generic_reference(self, mock_ctx):
        """Test FORMULATEXT with generic reference"""
        result = FORMULATEXT(mock_ctx, 123)
        assert result == "=FORMULA_FOR(123)"

    def test_formulatext_none_reference(self, mock_ctx):
        """Test FORMULATEXT with None reference raises error"""
        with pytest.raises(ValidationError):
            FORMULATEXT(mock_ctx, None)


class TestTRANSPOSE:
    """Test cases for TRANSPOSE function"""

    def test_transpose_2d_list(self, mock_ctx):
        """Test TRANSPOSE with 2D list"""
        array = [[1, 2, 3], [4, 5, 6]]
        result_path = TRANSPOSE(mock_ctx, array, output_filename="test_transpose.parquet")

        # Verify file was created
        assert result_path.exists()

        # Load and verify transposed data
        df = pl.read_parquet(result_path)
        assert df.shape == (3, 2)  # Original was 2x3, transposed is 3x2

        # Check columns (each original row becomes a column) - now converted to strings
        assert df["col_0"].to_list() == ["1", "2", "3"]  # First row of original data
        assert df["col_1"].to_list() == ["4", "5", "6"]  # Second row of original data

    def test_transpose_dataframe(self, mock_ctx):
        """Test TRANSPOSE with Polars DataFrame"""
        df = pl.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result_path = TRANSPOSE(mock_ctx, df, output_filename="test_transpose_df.parquet")

        # Verify file was created
        assert result_path.exists()

        # Load and verify transposed data
        transposed_df = pl.read_parquet(result_path)
        assert transposed_df.shape == (3, 2)  # Original was 2x3, transposed is 3x2

    def test_transpose_file_input(self, mock_ctx):
        """Test TRANSPOSE with file input"""
        # Create test data file
        test_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        test_file = mock_ctx.deps.data_dir / "test_input.parquet"
        test_df.write_parquet(test_file)

        result_path = TRANSPOSE(mock_ctx, test_file, output_filename="test_transpose_file.parquet")

        # Verify file was created
        assert result_path.exists()

        # Load and verify transposed data
        transposed_df = pl.read_parquet(result_path)
        assert transposed_df.shape == (2, 2)  # Original was 2x2, transposed is 2x2

    def test_transpose_invalid_array(self, mock_ctx):
        """Test TRANSPOSE with invalid array structure"""
        with pytest.raises(ValidationError):
            TRANSPOSE(mock_ctx, [1, 2, 3], output_filename="test.parquet")  # Not 2D

    def test_transpose_inconsistent_rows(self, mock_ctx):
        """Test TRANSPOSE with inconsistent row lengths"""
        array = [[1, 2, 3], [4, 5]]  # Different row lengths
        with pytest.raises(ValidationError):
            TRANSPOSE(mock_ctx, array, output_filename="test.parquet")


class TestCELL:
    """Test cases for CELL function"""

    def test_cell_address(self, mock_ctx):
        """Test CELL with address info_type"""
        result = CELL(mock_ctx, "address", "A1")
        assert result == "$A1"

    def test_cell_row(self, mock_ctx):
        """Test CELL with row info_type"""
        result = CELL(mock_ctx, "row", "B5")
        assert result == 5

    def test_cell_col(self, mock_ctx):
        """Test CELL with col info_type"""
        result = CELL(mock_ctx, "col", "C3")
        assert result == 3

    def test_cell_col_multiple_letters(self, mock_ctx):
        """Test CELL with multi-letter column reference"""
        result = CELL(mock_ctx, "col", "AA1")
        assert result == 27  # AA = 26 + 1

    def test_cell_type_numeric(self, mock_ctx):
        """Test CELL with type info_type for numeric value"""
        result = CELL(mock_ctx, "type", 123.45)
        assert result == "v"

    def test_cell_type_text(self, mock_ctx):
        """Test CELL with type info_type for text value"""
        result = CELL(mock_ctx, "type", "hello")
        assert result == "l"

    def test_cell_type_boolean(self, mock_ctx):
        """Test CELL with type info_type for boolean value"""
        result = CELL(mock_ctx, "type", True)
        assert result == "v"

    def test_cell_type_none(self, mock_ctx):
        """Test CELL with type info_type for None value"""
        result = CELL(mock_ctx, "type", None)
        assert result == "b"

    def test_cell_contents(self, mock_ctx):
        """Test CELL with contents info_type"""
        result = CELL(mock_ctx, "contents", "test_value")
        assert result == "test_value"

    def test_cell_format(self, mock_ctx):
        """Test CELL with format info_type"""
        result = CELL(mock_ctx, "format", "A1")
        assert result == "G"

    def test_cell_width(self, mock_ctx):
        """Test CELL with width info_type"""
        result = CELL(mock_ctx, "width", "A1")
        assert result == 8

    def test_cell_filename(self, mock_ctx):
        """Test CELL with filename info_type"""
        result = CELL(mock_ctx, "filename", "A1")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cell_unsupported_info_type(self, mock_ctx):
        """Test CELL with unsupported info_type"""
        with pytest.raises(ValidationError):
            CELL(mock_ctx, "unsupported", "A1")

    def test_cell_missing_reference(self, mock_ctx):
        """Test CELL with missing reference for address"""
        with pytest.raises(ConfigurationError):
            CELL(mock_ctx, "address", None)


class TestINFO:
    """Test cases for INFO function"""

    def test_info_version(self, mock_ctx):
        """Test INFO with version type_text"""
        result = INFO(mock_ctx, "version")
        assert result.startswith("Python")
        assert sys.version.split()[0] in result

    def test_info_system(self, mock_ctx):
        """Test INFO with system type_text"""
        result = INFO(mock_ctx, "system")
        assert result == platform.system()

    def test_info_release(self, mock_ctx):
        """Test INFO with release type_text"""
        result = INFO(mock_ctx, "release")
        assert result == platform.release()

    def test_info_machine(self, mock_ctx):
        """Test INFO with machine type_text"""
        result = INFO(mock_ctx, "machine")
        assert result == platform.machine()

    def test_info_processor(self, mock_ctx):
        """Test INFO with processor type_text"""
        result = INFO(mock_ctx, "processor")
        assert isinstance(result, str)

    def test_info_platform(self, mock_ctx):
        """Test INFO with platform type_text"""
        result = INFO(mock_ctx, "platform")
        assert result == platform.platform()

    def test_info_node(self, mock_ctx):
        """Test INFO with node type_text"""
        result = INFO(mock_ctx, "node")
        assert result == platform.node()

    def test_info_architecture(self, mock_ctx):
        """Test INFO with architecture type_text"""
        result = INFO(mock_ctx, "architecture")
        assert isinstance(result, str)

    def test_info_python_version(self, mock_ctx):
        """Test INFO with python_version type_text"""
        result = INFO(mock_ctx, "python_version")
        assert result == platform.python_version()

    def test_info_directory(self, mock_ctx):
        """Test INFO with directory type_text"""
        result = INFO(mock_ctx, "directory")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_info_recalc(self, mock_ctx):
        """Test INFO with recalc type_text"""
        result = INFO(mock_ctx, "recalc")
        assert result == "Automatic"

    def test_info_origin(self, mock_ctx):
        """Test INFO with origin type_text"""
        result = INFO(mock_ctx, "origin")
        assert result == "$A$1"

    def test_info_unsupported_type(self, mock_ctx):
        """Test INFO with unsupported type_text"""
        with pytest.raises(ValidationError):
            INFO(mock_ctx, "unsupported_type")

    def test_info_case_insensitive(self, mock_ctx):
        """Test INFO is case insensitive"""
        result1 = INFO(mock_ctx, "VERSION")
        result2 = INFO(mock_ctx, "version")
        assert result1 == result2


class TestN:
    """Test cases for N function"""

    def test_n_boolean_true(self, mock_ctx):
        """Test N with True boolean"""
        result = N(mock_ctx, True)
        assert result == Decimal('1')

    def test_n_boolean_false(self, mock_ctx):
        """Test N with False boolean"""
        result = N(mock_ctx, False)
        assert result == Decimal('0')

    def test_n_integer(self, mock_ctx):
        """Test N with integer"""
        result = N(mock_ctx, 42)
        assert result == Decimal('42')

    def test_n_float(self, mock_ctx):
        """Test N with float"""
        result = N(mock_ctx, 3.14)
        assert result == Decimal('3.14')

    def test_n_decimal(self, mock_ctx):
        """Test N with Decimal"""
        input_val = Decimal('123.45')
        result = N(mock_ctx, input_val)
        assert result == input_val

    def test_n_numeric_string(self, mock_ctx):
        """Test N with numeric string"""
        result = N(mock_ctx, "123.45")
        assert result == Decimal('123.45')

    def test_n_percentage_string(self, mock_ctx):
        """Test N with percentage string"""
        result = N(mock_ctx, "50%")
        assert result == Decimal('0.5')  # 50% = 0.5 in decimal

    def test_n_formatted_string(self, mock_ctx):
        """Test N with formatted numeric string"""
        result = N(mock_ctx, "$1,234.56")
        assert result == Decimal('1234.56')

    def test_n_non_numeric_string(self, mock_ctx):
        """Test N with non-numeric string"""
        result = N(mock_ctx, "hello")
        assert result == Decimal('0')

    def test_n_none(self, mock_ctx):
        """Test N with None"""
        result = N(mock_ctx, None)
        assert result == Decimal('0')

    def test_n_list(self, mock_ctx):
        """Test N with list"""
        result = N(mock_ctx, [1, 2, 3])
        assert result == Decimal('0')

    def test_n_dataframe(self, mock_ctx):
        """Test N with DataFrame"""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = N(mock_ctx, df)
        assert result == Decimal('0')

    def test_n_empty_string(self, mock_ctx):
        """Test N with empty string"""
        result = N(mock_ctx, "")
        assert result == Decimal('0')


class TestT:
    """Test cases for T function"""

    def test_t_string(self, mock_ctx):
        """Test T with string"""
        result = T(mock_ctx, "Hello World")
        assert result == "Hello World"

    def test_t_integer(self, mock_ctx):
        """Test T with integer"""
        result = T(mock_ctx, 123)
        assert result == ""

    def test_t_float(self, mock_ctx):
        """Test T with float"""
        result = T(mock_ctx, 3.14)
        assert result == ""

    def test_t_decimal(self, mock_ctx):
        """Test T with Decimal"""
        result = T(mock_ctx, Decimal('123.45'))
        assert result == ""

    def test_t_boolean_true(self, mock_ctx):
        """Test T with True boolean"""
        result = T(mock_ctx, True)
        assert result == ""

    def test_t_boolean_false(self, mock_ctx):
        """Test T with False boolean"""
        result = T(mock_ctx, False)
        assert result == ""

    def test_t_none(self, mock_ctx):
        """Test T with None"""
        result = T(mock_ctx, None)
        assert result == ""

    def test_t_list(self, mock_ctx):
        """Test T with list"""
        result = T(mock_ctx, [1, 2, 3])
        assert result == ""

    def test_t_dataframe(self, mock_ctx):
        """Test T with DataFrame"""
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = T(mock_ctx, df)
        assert result == ""

    def test_t_empty_string(self, mock_ctx):
        """Test T with empty string"""
        result = T(mock_ctx, "")
        assert result == ""

    def test_t_non_numeric_object(self, mock_ctx):
        """Test T with non-numeric object that converts to string"""
        class CustomObject:
            def __str__(self):
                return "custom_object"

        result = T(mock_ctx, CustomObject())
        assert result == "custom_object"


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_transpose_empty_array(self, mock_ctx):
        """Test TRANSPOSE with empty array"""
        with pytest.raises(ValidationError):
            TRANSPOSE(mock_ctx, [], output_filename="test.parquet")

    def test_cell_case_insensitive(self, mock_ctx):
        """Test CELL is case insensitive for info_type"""
        result1 = CELL(mock_ctx, "ADDRESS", "A1")
        result2 = CELL(mock_ctx, "address", "A1")
        assert result1 == result2

    def test_n_very_large_number(self, mock_ctx):
        """Test N with very large number"""
        large_num = "999999999999999999999999999999"
        result = N(mock_ctx, large_num)
        assert result == Decimal(large_num)

    def test_t_numeric_string_object(self, mock_ctx):
        """Test T with object that converts to numeric string"""
        class NumericObject:
            def __str__(self):
                return "123.45"

        result = T(mock_ctx, NumericObject())
        assert result == ""  # Should return empty string for numeric-looking strings


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_n_and_t_conversion_consistency(self, mock_ctx):
        """Test that N and T functions are consistent in their conversions"""
        test_values = [
            123,
            "hello",
            True,
            False,
            None,
            3.14,
            "123.45"
        ]

        for value in test_values:
            n_result = N(mock_ctx, value)
            t_result = T(mock_ctx, value)

            # If T returns empty string, N should return a number
            # If T returns a string, N should return 0 (for non-numeric strings)
            if t_result == "":
                assert isinstance(n_result, Decimal)
            else:
                # For string values that T preserves, N should handle appropriately
                assert isinstance(t_result, str)

    def test_cell_and_info_system_info(self, mock_ctx):
        """Test that CELL and INFO provide consistent system information"""
        cell_filename = CELL(mock_ctx, "filename", "A1")
        info_directory = INFO(mock_ctx, "directory")

        # Both should return string paths
        assert isinstance(cell_filename, str)
        assert isinstance(info_directory, str)

    def test_transpose_with_mixed_data_types(self, mock_ctx):
        """Test TRANSPOSE with mixed data types"""
        # Create DataFrame with mixed types
        df = pl.DataFrame({
            "numbers": [1, 2, 3],
            "strings": ["a", "b", "c"],
            "booleans": [True, False, True]
        })

        result_path = TRANSPOSE(mock_ctx, df, output_filename="mixed_transpose.parquet")
        assert result_path.exists()

        # Load and verify structure
        transposed_df = pl.read_parquet(result_path)
        assert transposed_df.shape == (3, 3)  # 3 columns became 3 rows


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
