"""
Additional Useful Functions

These functions further aid in analysis, documentation, or advanced computations.
All functions use Decimal precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, Union
from pathlib import Path
import polars as pl
import platform
import sys
import os
from tools.tool_exceptions import (
    FPABaseException,
    RetryAfterCorrectionError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)
from tools.toolset_utils import load_df, save_df_to_analysis_dir

# Set decimal precision for financial calculations
getcontext().prec = 28


class FPABaseException(Exception):
    """Base exception for FP&A operations"""
    pass


class RetryAfterCorrectionError(FPABaseException):
    """Error that can be resolved by correcting input data or parameters"""
    def __init__(self, message: str, correction_hint: str):
        self.correction_hint = correction_hint
        super().__init__(message)


class ValidationError(FPABaseException):
    """Input validation failed - data structure or business rule violation"""
    pass


class CalculationError(FPABaseException):
    """Mathematical or financial calculation error - likely data issue"""
    pass


class ConfigurationError(FPABaseException):
    """Function configuration or parameter error - code issue"""
    pass


class DataQualityError(RetryAfterCorrectionError):
    """Data quality issues that can be corrected"""
    pass


def _convert_to_decimal(value: Any) -> Decimal:
    """
    Safely convert value to Decimal with proper error handling.

    Args:
        value: Value to convert

    Returns:
        Decimal: Converted value

    Raises:
        DataQualityError: If conversion fails
    """
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (ValueError, TypeError, OverflowError) as e:
        raise DataQualityError(
            f"Cannot convert value to Decimal: {str(e)}",
            "Ensure value is a valid numeric type"
        )


def FORMULATEXT(run_context: Any, reference: Any) -> str:
    """
    Returns the formula in a referenced cell as text, which can help in auditing or documentation.

    Since this is a simulation environment, this function returns a placeholder indicating
    the formula text functionality for the given reference.

    Args:
        run_context: RunContext object for file operations
        reference: Cell reference or identifier

    Returns:
        str: Text string representing formula (simulated)

    Raises:
        ValidationError: If reference is invalid

    Example:
        >>> FORMULATEXT(ctx, "A1")
        "=SUM(B1:B10)"
        >>> FORMULATEXT(ctx, {"cell": "B2", "formula": "=AVERAGE(C1:C5)"})
        "=AVERAGE(C1:C5)"
    """
    try:
        if reference is None:
            raise ValidationError("Reference cannot be None")

        # Simulate formula text extraction based on reference type
        if isinstance(reference, str):
            # Simple cell reference
            return f"=SUM({reference}:Z{reference[1:] if len(reference) > 1 else '1'})"
        elif isinstance(reference, dict) and "formula" in reference:
            # Dictionary with formula key
            return reference["formula"]
        elif isinstance(reference, dict) and "cell" in reference:
            # Dictionary with cell reference
            cell = reference["cell"]
            return f"=AVERAGE({cell}:{cell})"
        else:
            # Generic reference
            return f"=FORMULA_FOR({str(reference)})"

    except Exception as e:
        raise ValidationError(f"Invalid reference for FORMULATEXT: {str(e)}")


def TRANSPOSE(run_context: Any, array: Union[list[list[Any]], pl.DataFrame, str, Path], *, output_filename: str) -> Path:
    """
    Converts rows to columns or vice versa, useful for rearranging data.

    This function transposes a 2D array or DataFrame, swapping rows and columns.
    Essential for financial data restructuring and matrix operations.

    Args:
        run_context: RunContext object for file operations
        array: 2D array, DataFrame, or file path to transpose
        output_filename: Filename to save transposed results as parquet file

    Returns:
        Path: Path to saved transposed data file

    Raises:
        ValidationError: If array structure is invalid
        DataQualityError: If data cannot be transposed

    Example:
        >>> data = [[1, 2, 3], [4, 5, 6]]
        >>> TRANSPOSE(ctx, data, output_filename="transposed.parquet")
        Path("scratch_pad/analysis/transposed.parquet")

        >>> df = pl.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        >>> TRANSPOSE(ctx, df, output_filename="transposed_df.parquet")
        Path("scratch_pad/analysis/transposed_df.parquet")
    """
    try:
        # Handle file path input
        if isinstance(array, (str, Path)):
            df = load_df(run_context, array)
        elif isinstance(array, pl.DataFrame):
            df = array
        elif isinstance(array, list):
            # Validate 2D list structure
            if not array or not isinstance(array[0], list):
                raise ValidationError("Array must be a 2D list structure")

            # Check all rows have same length
            row_length = len(array[0])
            for i, row in enumerate(array):
                if len(row) != row_length:
                    raise ValidationError(f"All rows must have same length. Row {i} has length {len(row)}, expected {row_length}")

            # Convert to DataFrame for easier transposition
            # Create column names
            col_names = [f"col_{i}" for i in range(row_length)]
            df = pl.DataFrame({col_names[i]: [row[i] for row in array] for i in range(row_length)})
        else:
            raise ValidationError(f"Unsupported array type: {type(array)}")

        # Perform transpose operation
        # Convert all data to string to handle mixed types
        original_columns = df.columns
        num_rows = df.height
        num_cols = len(original_columns)

        # Create transposed data structure
        # Each original row becomes a column in the transposed data
        final_transposed_data = {}

        for row_idx in range(num_rows):
            col_data = []
            for col_name in original_columns:
                # Convert each value to string to handle mixed types
                value = df[col_name][row_idx]
                col_data.append(str(value) if value is not None else "")
            final_transposed_data[f"col_{row_idx}"] = col_data

        transposed_df = pl.DataFrame(final_transposed_data, strict=False)

        # Save results to file
        return save_df_to_analysis_dir(run_context, transposed_df, output_filename)

    except ValidationError:
        # Re-raise ValidationError as-is
        raise
    except Exception as e:
        raise DataQualityError(
            f"Failed to transpose array: {str(e)}",
            "Ensure array is a valid 2D structure with consistent dimensions"
        )


def CELL(run_context: Any, info_type: str, reference: Any | None = None) -> Any:
    """
    Return information about cell formatting, location, or contents.

    This function simulates Excel's CELL function by returning various types of
    information about a cell or the current environment.

    Args:
        run_context: RunContext object for file operations
        info_type: Type of information to return
        reference: Cell reference (optional)

    Returns:
        Any: Various types depending on info_type

    Raises:
        ValidationError: If info_type is not supported
        ConfigurationError: If reference is required but not provided

    Example:
        >>> CELL(ctx, "address", "A1")
        "$A$1"
        >>> CELL(ctx, "row", "B5")
        5
        >>> CELL(ctx, "col", "C3")
        3
        >>> CELL(ctx, "type", 123.45)
        "v"
    """
    try:
        info_type = info_type.lower()

        if info_type == "address":
            if reference is None:
                raise ConfigurationError("Reference required for address info_type")
            if isinstance(reference, str):
                # Return absolute reference format
                return f"${reference}"
            return f"${str(reference)}"

        elif info_type == "row":
            if reference is None:
                raise ConfigurationError("Reference required for row info_type")
            if isinstance(reference, str) and len(reference) > 1:
                # Extract row number from reference like "B5"
                row_part = ''.join(filter(str.isdigit, reference))
                return int(row_part) if row_part else 1
            return 1

        elif info_type == "col":
            if reference is None:
                raise ConfigurationError("Reference required for col info_type")
            if isinstance(reference, str):
                # Extract column number from reference like "C3"
                col_part = ''.join(filter(str.isalpha, reference))
                if col_part:
                    # Convert column letter to number (A=1, B=2, etc.)
                    col_num = 0
                    for char in col_part.upper():
                        col_num = col_num * 26 + (ord(char) - ord('A') + 1)
                    return col_num
            return 1

        elif info_type == "type":
            if reference is None:
                return "b"  # blank
            elif isinstance(reference, (int, float, Decimal)):
                return "v"  # value
            elif isinstance(reference, str):
                return "l"  # label (text)
            elif isinstance(reference, bool):
                return "v"  # value
            else:
                return "v"  # default to value

        elif info_type == "contents":
            if reference is None:
                return ""
            return str(reference)

        elif info_type == "format":
            # Return general format
            return "G"

        elif info_type == "width":
            # Return default column width
            return 8

        elif info_type == "filename":
            # Return current working directory info
            return str(Path.cwd())

        else:
            raise ValidationError(f"Unsupported info_type: {info_type}")

    except Exception as e:
        if isinstance(e, (ValidationError, ConfigurationError)):
            raise
        raise ValidationError(f"Error in CELL function: {str(e)}")


def INFO(run_context: Any, type_text: str) -> str:
    """
    Return information about operating environment.

    This function provides system and environment information useful for
    financial analysis documentation and system compatibility checks.

    Args:
        run_context: RunContext object for file operations
        type_text: Type of information to return

    Returns:
        str: Text string with system info

    Raises:
        ValidationError: If type_text is not supported

    Example:
        >>> INFO(ctx, "version")
        "Python 3.11.0"
        >>> INFO(ctx, "system")
        "Darwin"
        >>> INFO(ctx, "release")
        "22.1.0"
    """
    try:
        type_text = type_text.lower()

        if type_text == "version":
            return f"Python {sys.version.split()[0]}"

        elif type_text == "system":
            return platform.system()

        elif type_text == "release":
            return platform.release()

        elif type_text == "machine":
            return platform.machine()

        elif type_text == "processor":
            return platform.processor() or "Unknown"

        elif type_text == "platform":
            return platform.platform()

        elif type_text == "node":
            return platform.node()

        elif type_text == "architecture":
            return str(platform.architecture())

        elif type_text == "python_version":
            return platform.python_version()

        elif type_text == "python_build":
            return str(platform.python_build())

        elif type_text == "python_compiler":
            return platform.python_compiler()

        elif type_text == "python_implementation":
            return platform.python_implementation()

        elif type_text == "directory":
            return str(Path.cwd())

        elif type_text == "memavail":
            # Return available memory info (simplified)
            try:
                import psutil
                return f"{psutil.virtual_memory().available // (1024**3)} GB"
            except ImportError:
                return "Memory info unavailable"

        elif type_text == "memused":
            # Return used memory info (simplified)
            try:
                import psutil
                return f"{psutil.virtual_memory().used // (1024**3)} GB"
            except ImportError:
                return "Memory info unavailable"

        elif type_text == "totmem":
            # Return total memory info (simplified)
            try:
                import psutil
                return f"{psutil.virtual_memory().total // (1024**3)} GB"
            except ImportError:
                return "Memory info unavailable"

        elif type_text == "numfile":
            # Return number of open files (simplified)
            return "File count unavailable"

        elif type_text == "recalc":
            # Return recalculation mode (always automatic in this context)
            return "Automatic"

        elif type_text == "origin":
            # Return origin address
            return "$A$1"

        elif type_text == "osversion":
            return platform.version()

        else:
            raise ValidationError(f"Unsupported type_text: {type_text}")

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Error in INFO function: {str(e)}")


def N(run_context: Any, value: Any) -> Decimal:
    """
    Convert value to number using Decimal precision.

    This function converts various data types to numeric values, following Excel's N function behavior.
    Essential for financial calculations where consistent numeric conversion is required.

    Args:
        run_context: RunContext object for file operations
        value: Value to convert

    Returns:
        Decimal: Numeric value or 0

    Raises:
        DataQualityError: If conversion results in invalid numeric value

    Example:
        >>> N(ctx, True)
        Decimal('1')
        >>> N(ctx, False)
        Decimal('0')
        >>> N(ctx, "123.45")
        Decimal('123.45')
        >>> N(ctx, "text")
        Decimal('0')
        >>> N(ctx, None)
        Decimal('0')
    """
    try:
        # Handle None/null values
        if value is None:
            return Decimal('0')

        # Handle boolean values
        if isinstance(value, bool):
            return Decimal('1') if value else Decimal('0')

        # Handle numeric values
        if isinstance(value, (int, float, Decimal)):
            return _convert_to_decimal(value)

        # Handle string values
        if isinstance(value, str):
            # Try to convert string to number
            try:
                # Handle empty string
                if not value.strip():
                    return Decimal('0')

                # Remove common formatting characters
                cleaned_value = value.strip().replace(',', '').replace('$', '')

                # Handle percentage conversion
                if '%' in value:
                    cleaned_value = cleaned_value.replace('%', '')
                    numeric_val = _convert_to_decimal(cleaned_value)
                    return numeric_val / Decimal('100')

                return _convert_to_decimal(cleaned_value)
            except (ValueError, TypeError, Exception):
                # If string cannot be converted to number, return 0
                return Decimal('0')

        # Handle list/array - return 0 (Excel behavior)
        if isinstance(value, (list, tuple)):
            return Decimal('0')

        # Handle DataFrame/Series - return 0
        if isinstance(value, (pl.DataFrame, pl.Series)):
            return Decimal('0')

        # For any other type, try direct conversion or return 0
        try:
            return _convert_to_decimal(value)
        except:
            return Decimal('0')

    except Exception as e:
        raise DataQualityError(
            f"Error converting value to number in N function: {str(e)}",
            "Ensure value is convertible to numeric format"
        )


def T(run_context: Any, value: Any) -> str:
    """
    Convert value to text.

    This function converts various data types to text strings, following Excel's T function behavior.
    Useful for financial reporting where consistent text conversion is required.

    Args:
        run_context: RunContext object for file operations
        value: Value to convert

    Returns:
        str: Text string or empty string

    Raises:
        DataQualityError: If conversion fails

    Example:
        >>> T(ctx, 123)
        ""
        >>> T(ctx, "Hello")
        "Hello"
        >>> T(ctx, True)
        ""
        >>> T(ctx, None)
        ""
    """
    try:
        # Handle None/null values
        if value is None:
            return ""

        # Handle string values - return as-is
        if isinstance(value, str):
            return value

        # Handle numeric values - return empty string (Excel behavior)
        if isinstance(value, (int, float, Decimal)):
            return ""

        # Handle boolean values - return empty string (Excel behavior)
        if isinstance(value, bool):
            return ""

        # Handle list/array - return empty string
        if isinstance(value, (list, tuple)):
            return ""

        # Handle DataFrame/Series - return empty string
        if isinstance(value, (pl.DataFrame, pl.Series)):
            return ""

        # For any other type, try string conversion
        try:
            converted = str(value)
            # If it looks like a number, return empty string (Excel behavior)
            try:
                float(converted)
                return ""
            except ValueError:
                return converted
        except:
            return ""

    except Exception as e:
        raise DataQualityError(
            f"Error converting value to text in T function: {str(e)}",
            "Ensure value is convertible to text format"
        )
