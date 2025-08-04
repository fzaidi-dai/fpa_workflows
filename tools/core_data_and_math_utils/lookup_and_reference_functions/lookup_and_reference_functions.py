"""
Lookup & Reference Functions

These functions are invaluable when you need to retrieve data from a table or array dynamically.
All functions use Decimal precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, Union, Optional
import polars as pl
import numpy as np
import re
from functools import lru_cache
from pathlib import Path
from tools.tool_exceptions import (
    FPABaseException,
    RetryAfterCorrectionError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)
from decimal import InvalidOperation
import numbers
from tools.toolset_utils import load_df

# Set decimal precision for financial calculations
getcontext().prec = 28

# Performance optimization: Cache compiled regex patterns and table conversions
_TABLE_CACHE = {}
_COLUMN_CACHE = {}


def _validate_table_array(table_array: Any, function_name: str, run_context: Any = None) -> pl.DataFrame:
    """
    Standardized input validation for table array data.

    Args:
        table_array: Input table data to validate
        function_name: Name of calling function for error messages
        run_context: RunContext for file operations (optional)

    Returns:
        pl.DataFrame: Validated Polars DataFrame

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid structure
    """
    try:
        # Handle file path input
        if isinstance(table_array, (str, Path)):
            if run_context is None:
                raise ValidationError(f"RunContext required for file input in {function_name}")
            df = load_df(run_context, table_array)
            df = df.with_columns(
                pl.col(pl.NUMERIC_DTYPES).map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
            )
            return df

        if isinstance(table_array, pl.DataFrame):
            if table_array.is_empty():
                raise ValidationError(f"Table array cannot be empty for {function_name}")
            df = table_array.with_columns(
                pl.col(pl.NUMERIC_DTYPES).map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
            )
            return df

        elif isinstance(table_array, list):
            if not table_array:
                raise ValidationError(f"Table array cannot be empty for {function_name}")
            if isinstance(table_array[0], list):
                row_lengths = [len(row) for row in table_array]
                if len(set(row_lengths)) > 1:
                    raise DataQualityError(
                        f"All rows in table array must have same length for {function_name}",
                        "Ensure all rows have consistent number of columns"
                    )
                max_cols = max(row_lengths) if row_lengths else 0
                columns = [f"col_{i}" for i in range(max_cols)]
                df = pl.DataFrame(table_array, schema=columns, orient="row")
                df = df.with_columns(
                    pl.col(pl.NUMERIC_DTYPES).map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
                )
                return df
            else:
                df = pl.DataFrame({"col_0": table_array}, strict=False)
                df = df.with_columns(
                    pl.col(pl.NUMERIC_DTYPES).map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
                )
                return df

        elif isinstance(table_array, np.ndarray):
            if table_array.size == 0:
                raise ValidationError(f"Table array cannot be empty for {function_name}")
            if table_array.ndim == 1:
                df = pl.DataFrame({"col_0": table_array.tolist()}, strict=False)
                df = df.with_columns(
                    pl.col(pl.NUMERIC_DTYPES).map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
                )
                return df
            elif table_array.ndim == 2:
                columns = [f"col_{i}" for i in range(table_array.shape[1])]
                df = pl.DataFrame(table_array.tolist(), schema=columns, orient="row")
                df = df.with_columns(
                    pl.col(pl.NUMERIC_DTYPES).map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
                )
                return df
            else:
                raise ValidationError(f"Table array must be 1D or 2D for {function_name}")
        else:
            raise ValidationError(f"Unsupported table array type for {function_name}: {type(table_array)}")
    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid table array structure in {function_name}: {str(e)}",
            "Ensure table array is a valid 2D structure with consistent dimensions"
        )


def _validate_lookup_array(lookup_array: Any, function_name: str) -> pl.Series:
    """
    Standardized input validation for lookup array data.

    Args:
        lookup_array: Input lookup array to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.Series: Validated Polars Series

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid values
    """
    try:
        if isinstance(lookup_array, pl.Series):
            if lookup_array.is_empty():
                raise ValidationError(f"Lookup array cannot be empty for {function_name}")
            lookup_series = lookup_array
            if lookup_series.dtype in pl.NUMERIC_DTYPES:
                lookup_series = lookup_series.map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
            return lookup_series
        elif isinstance(lookup_array, (list, np.ndarray)):
            if len(lookup_array) == 0:
                raise ValidationError(f"Lookup array cannot be empty for {function_name}")
            series = pl.Series(lookup_array, strict=False)
            if series.dtype in pl.NUMERIC_DTYPES:
                series = series.map_elements(_convert_to_decimal_cached, return_dtype=pl.Object)
            return series
        else:
            raise ValidationError(f"Unsupported lookup array type for {function_name}: {type(lookup_array)}")
    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid lookup array values in {function_name}: {str(e)}",
            "Ensure all values are valid data types"
        )


def _validate_index_bounds(index: int, max_value: int, function_name: str, index_name: str = "index") -> None:
    """
    Validate index is within bounds.

    Args:
        index: Index value to validate
        max_value: Maximum allowed value (exclusive)
        function_name: Name of calling function
        index_name: Name of the index parameter

    Raises:
        ValidationError: If index is out of bounds
    """
    if not isinstance(index, int):
        raise ValidationError(f"{index_name} must be an integer for {function_name}")
    if index < 1 or index > max_value:
        raise ValidationError(f"{index_name} must be between 1 and {max_value} for {function_name}")


@lru_cache(maxsize=512)
def _convert_to_decimal_cached(value: Any) -> Decimal:
    """
    Safely convert value to Decimal with caching for performance.

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
        if value is None:
            return Decimal('0')
        return Decimal(str(value))
    except (ValueError, TypeError, OverflowError, InvalidOperation) as e:
        raise DataQualityError(
            f"Cannot convert value to Decimal: {str(e)}",
            "Ensure value is a valid numeric type"
        )


def _to_decimal_if_numeric(value: Any) -> Any:
    if isinstance(value, numbers.Number) and not isinstance(value, Decimal):
        try:
            return _convert_to_decimal_cached(value)
        except DataQualityError:
            return value
    elif isinstance(value, str):
        try:
            return _convert_to_decimal_cached(value)
        except DataQualityError:
            return value
    return value


def _find_match_position(lookup_value: Any, lookup_series: pl.Series, match_type: int) -> int:
    """
    Find position of lookup value in series based on match type.

    Args:
        lookup_value: Value to find
        lookup_series: Series to search in
        match_type: 0=exact, 1=largest <=, -1=smallest >=

    Returns:
        int: Position (1-based) of match

    Raises:
        CalculationError: If no match found
    """
    try:
        is_object = lookup_series.dtype == pl.Object
        if is_object:
            values = lookup_series.to_list()
            if match_type == 0:
                for idx, val in enumerate(values):
                    if val == lookup_value:
                        return idx + 1
                raise CalculationError(f"Exact match not found for value: {lookup_value}")
            elif match_type == 1:
                max_idx = -1
                for idx, val in enumerate(values):
                    if val <= lookup_value:
                        max_idx = idx
                if max_idx == -1:
                    raise CalculationError(f"No value less than or equal to: {lookup_value}")
                return max_idx + 1
            elif match_type == -1:
                min_idx = -1
                for idx, val in enumerate(reversed(values)):
                    if val >= lookup_value:
                        min_idx = len(values) - 1 - idx
                        break
                if min_idx == -1:
                    raise CalculationError(f"No value greater than or equal to: {lookup_value}")
                return min_idx + 1
            else:
                raise ValidationError(f"Invalid match_type: {match_type}. Must be -1, 0, or 1")
        else:
            if match_type == 0:
                mask = lookup_series == lookup_value
                matches = mask.arg_true()
                if matches.is_empty():
                    raise CalculationError(f"Exact match not found for value: {lookup_value}")
                return int(matches[0]) + 1
            elif match_type == 1:
                mask = lookup_series <= lookup_value
                if not mask.any():
                    raise CalculationError(f"No value less than or equal to: {lookup_value}")
                true_indices = mask.arg_true()
                return int(true_indices[-1]) + 1
            elif match_type == -1:
                mask = lookup_series >= lookup_value
                if not mask.any():
                    raise CalculationError(f"No value greater than or equal to: {lookup_value}")
                true_indices = mask.arg_true()
                return int(true_indices[0]) + 1
            else:
                raise ValidationError(f"Invalid match_type: {match_type}. Must be -1, 0, or 1")
    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError)):
            raise
        raise CalculationError(f"Error finding match position: {str(e)}")


def VLOOKUP(
    run_context: Any,
    lookup_value: Any,
    table_array: Union[list[list[Any]], pl.DataFrame, np.ndarray, str, Path],
    *,
    col_index: int,
    range_lookup: bool = False
) -> Any:
    """
    Search for a value in the first column of a table and return a value in the same row from a specified column.
    """
    df = _validate_table_array(table_array, "VLOOKUP", run_context)
    if df.width == 0:
        raise ValidationError("Table array must have at least one column for VLOOKUP")
    _validate_index_bounds(col_index, df.width, "VLOOKUP", "col_index")
    try:
        lookup_column = df.get_column(df.columns[0])
        if range_lookup:
            match_type = 1
            position = _find_match_position(lookup_value, lookup_column, match_type)
        else:
            match_type = 0
            position = _find_match_position(lookup_value, lookup_column, match_type)
        result_column = df.get_column(df.columns[col_index - 1])
        result_value = result_column[position - 1]
        return _to_decimal_if_numeric(result_value)
    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError, DataQualityError)):
            raise
        raise CalculationError(f"VLOOKUP calculation failed: {str(e)}")


def HLOOKUP(
    run_context: Any,
    lookup_value: Any,
    table_array: Union[list[list[Any]], pl.DataFrame, np.ndarray, str, Path],
    *,
    row_index: int,
    range_lookup: bool = False
) -> Any:
    """
    Search for a value in the first row of a table and return a value in the same column from a specified row.
    """
    df = _validate_table_array(table_array, "HLOOKUP", run_context)
    if df.height == 0:
        raise ValidationError("Table array must have at least one row for HLOOKUP")
    _validate_index_bounds(row_index, df.height, "HLOOKUP", "row_index")
    try:
        first_row = df.row(0)
        lookup_series = pl.Series(first_row, strict=False)
        if range_lookup:
            match_type = 1
            position = _find_match_position(lookup_value, lookup_series, match_type)
        else:
            match_type = 0
            position = _find_match_position(lookup_value, lookup_series, match_type)
        result_row = df.row(row_index - 1)
        result_value = result_row[position - 1]
        return _to_decimal_if_numeric(result_value)
    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError, DataQualityError)):
            raise
        raise CalculationError(f"HLOOKUP calculation failed: {str(e)}")


def INDEX(
    run_context: Any,
    array: Union[list[list[Any]], list[Any], pl.DataFrame, pl.Series, np.ndarray, str, Path],
    row_num: int,
    *,
    column_num: Optional[int] = None
) -> Any:
    """
    Return a value at a given position in an array or table.
    """
    try:
        # Handle file path input
        if isinstance(array, (str, Path)):
            if run_context is None:
                raise ValidationError("RunContext required for file input in INDEX")
            array = load_df(run_context, array)

        if isinstance(array, pl.DataFrame):
            df = array
            if df.is_empty():
                raise ValidationError("Array cannot be empty for INDEX")
            _validate_index_bounds(row_num, df.height, "INDEX", "row_num")
            if column_num is None:
                if df.width > 1:
                    raise ValidationError("column_num is required for multi-column DataFrame in INDEX")
                return _to_decimal_if_numeric(df.item(row_num - 1, 0))
            else:
                _validate_index_bounds(column_num, df.width, "INDEX", "column_num")
                return _to_decimal_if_numeric(df.item(row_num - 1, column_num - 1))
        elif isinstance(array, pl.Series):
            if array.is_empty():
                raise ValidationError("Array cannot be empty for INDEX")
            _validate_index_bounds(row_num, len(array), "INDEX", "row_num")
            if column_num is not None:
                raise ValidationError("column_num should not be specified for 1D Series in INDEX")
            return _to_decimal_if_numeric(array[row_num - 1])
        elif isinstance(array, list):
            if not array:
                raise ValidationError("Array cannot be empty for INDEX")
            if isinstance(array[0], list):
                _validate_index_bounds(row_num, len(array), "INDEX", "row_num")
                if column_num is None:
                    if len(array[0]) > 1:
                        raise ValidationError("column_num is required for 2D array in INDEX")
                    return _to_decimal_if_numeric(array[row_num - 1][0])
                else:
                    row = array[row_num - 1]
                    _validate_index_bounds(column_num, len(row), "INDEX", "column_num")
                    return _to_decimal_if_numeric(row[column_num - 1])
            else:
                _validate_index_bounds(row_num, len(array), "INDEX", "row_num")
                if column_num is not None:
                    raise ValidationError("column_num should not be specified for 1D array in INDEX")
                return _to_decimal_if_numeric(array[row_num - 1])
        elif isinstance(array, np.ndarray):
            if array.size == 0:
                raise ValidationError("Array cannot be empty for INDEX")
            if array.ndim == 1:
                _validate_index_bounds(row_num, len(array), "INDEX", "row_num")
                if column_num is not None:
                    raise ValidationError("column_num should not be specified for 1D array in INDEX")
                return _to_decimal_if_numeric(array[row_num - 1])
            elif array.ndim == 2:
                _validate_index_bounds(row_num, array.shape[0], "INDEX", "row_num")
                if column_num is None:
                    if array.shape[1] > 1:
                        raise ValidationError("column_num is required for 2D array in INDEX")
                    return _to_decimal_if_numeric(array[row_num - 1, 0])
                else:
                    _validate_index_bounds(column_num, array.shape[1], "INDEX", "column_num")
                    return _to_decimal_if_numeric(array[row_num - 1, column_num - 1])
            else:
                raise ValidationError("Array must be 1D or 2D for INDEX")
        else:
            raise ValidationError(f"Unsupported array type for INDEX: {type(array)}")
    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError, DataQualityError)):
            raise
        raise CalculationError(f"INDEX calculation failed: {str(e)}")


def MATCH(
    run_context: Any,
    lookup_value: Any,
    lookup_array: Union[list[Any], pl.Series, np.ndarray, str, Path],
    *,
    match_type: int = 0
) -> int:
    """
    Find the relative position of an item in an array.
    """
    # Handle file path input
    if isinstance(lookup_array, (str, Path)):
        if run_context is None:
            raise ValidationError("RunContext required for file input in MATCH")
        lookup_array = load_df(run_context, lookup_array)

    lookup_series = _validate_lookup_array(lookup_array, "MATCH")
    if match_type not in [-1, 0, 1]:
        raise ValidationError("match_type must be -1, 0, or 1 for MATCH")
    try:
        position = _find_match_position(lookup_value, lookup_series, match_type)
        return position
    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError)):
            raise
        raise CalculationError(f"MATCH calculation failed: {str(e)}")


def XLOOKUP(
    run_context: Any,
    lookup_value: Any,
    lookup_array: Union[list[Any], pl.Series, np.ndarray, str, Path],
    return_array: Union[list[Any], pl.Series, np.ndarray, str, Path],
    *,
    if_not_found: Optional[Any] = None,
    match_mode: int = 0,
    search_mode: int = 1
) -> Any:
    """
    Modern, flexible lookup function replacing VLOOKUP/HLOOKUP with enhanced capabilities.
    """
    # Handle file path input
    if isinstance(lookup_array, (str, Path)):
        if run_context is None:
            raise ValidationError("RunContext required for file input in XLOOKUP")
        lookup_array = load_df(run_context, lookup_array)

    if isinstance(return_array, (str, Path)):
        if run_context is None:
            raise ValidationError("RunContext required for file input in XLOOKUP")
        return_array = load_df(run_context, return_array)

    lookup_series = _validate_lookup_array(lookup_array, "XLOOKUP")
    return_series = _validate_lookup_array(return_array, "XLOOKUP")
    if len(lookup_series) != len(return_series):
        raise ValidationError("lookup_array and return_array must have the same length for XLOOKUP")
    if match_mode not in [-1, 0, 1, 2]:
        raise ValidationError("match_mode must be -1, 0, 1, or 2 for XLOOKUP")
    if search_mode not in [-2, -1, 1, 2]:
        raise ValidationError("search_mode must be -2, -1, 1, or 2 for XLOOKUP")
    try:
        if match_mode == 2:
            pattern = str(lookup_value).replace('*', '.*').replace('?', '.')
            regex_pattern = f'^{pattern}$'
            for i, value in enumerate(lookup_series.to_list()):
                if re.match(regex_pattern, str(value)):
                    return _to_decimal_if_numeric(return_series[i])
            # Also check if lookup_value pattern matches any of the lookup_array values
            for i, value in enumerate(lookup_series.to_list()):
                value_pattern = str(value).replace('*', '.*').replace('?', '.')
                value_regex = f'^{value_pattern}$'
                if re.match(value_regex, str(lookup_value)):
                    return _to_decimal_if_numeric(return_series[i])
            if if_not_found is not None:
                return if_not_found
            else:
                raise CalculationError(f"Wildcard match not found for value: {lookup_value}")
        else:
            if search_mode == -1:
                search_indices = range(len(lookup_series) - 1, -1, -1)
            else:
                search_indices = range(len(lookup_series))
            for i in search_indices:
                value = lookup_series[i]
                if match_mode == 0:
                    if value == lookup_value:
                        return _to_decimal_if_numeric(return_series[i])
                elif match_mode == -1:
                    if value <= lookup_value:
                        return _to_decimal_if_numeric(return_series[i])
                elif match_mode == 1:
                    if value >= lookup_value:
                        return _to_decimal_if_numeric(return_series[i])
            if if_not_found is not None:
                return if_not_found
            else:
                raise CalculationError(f"Match not found for value: {lookup_value}")
    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError)):
            raise
        raise CalculationError(f"XLOOKUP calculation failed: {str(e)}")


def LOOKUP(
    run_context: Any,
    lookup_value: Any,
    lookup_vector: Union[list[Any], pl.Series, np.ndarray, str, Path],
    *,
    result_vector: Optional[Union[list[Any], pl.Series, np.ndarray, str, Path]] = None
) -> Any:
    """
    Simple lookup function (vector form) that finds the largest value less than or equal to lookup_value.
    """
    # Handle file path input
    if isinstance(lookup_vector, (str, Path)):
        if run_context is None:
            raise ValidationError("RunContext required for file input in LOOKUP")
        lookup_vector = load_df(run_context, lookup_vector)

    if result_vector is not None and isinstance(result_vector, (str, Path)):
        if run_context is None:
            raise ValidationError("RunContext required for file input in LOOKUP")
        result_vector = load_df(run_context, result_vector)

    lookup_series = _validate_lookup_array(lookup_vector, "LOOKUP")
    if result_vector is not None:
        result_series = _validate_lookup_array(result_vector, "LOOKUP")
        if len(lookup_series) != len(result_series):
            raise ValidationError("lookup_vector and result_vector must have the same length for LOOKUP")
    else:
        result_series = lookup_series
    try:
        position = _find_match_position(lookup_value, lookup_series, match_type=1)
        return _to_decimal_if_numeric(result_series[position - 1])
    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError)):
            raise
        raise CalculationError(f"LOOKUP calculation failed: {str(e)}")


def CHOOSE(index_num: int, *values: Any) -> Any:
    """
    Return a value from a list based on index number.
    """
    if not values:
        raise ValidationError("At least one value must be provided for CHOOSE")
    if not isinstance(index_num, int):
        raise ValidationError("index_num must be an integer for CHOOSE")
    if index_num < 1 or index_num > len(values):
        raise ValidationError(f"index_num must be between 1 and {len(values)} for CHOOSE")
    try:
        return _to_decimal_if_numeric(values[index_num - 1])
    except Exception as e:
        raise CalculationError(f"CHOOSE calculation failed: {str(e)}")


def OFFSET(
    run_context: Any,
    reference: Union[list[list[Any]], pl.DataFrame, str, Path],
    rows: int,
    cols: int,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
    output_filename: Optional[str] = None
) -> Union[Any, pl.DataFrame, Path]:
    """
    Create dynamic ranges based on reference point with offset.
    """
    df = _validate_table_array(reference, "OFFSET", run_context)
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise ValidationError("rows and cols must be integers for OFFSET")
    if height is None:
        height = 1
    if width is None:
        width = 1
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValidationError("height and width must be integers for OFFSET")
    if height < 1 or width < 1:
        raise ValidationError("height and width must be positive for OFFSET")
    try:
        start_row = rows
        start_col = cols
        end_row = start_row + height
        end_col = start_col + width
        if start_row < 0 or start_col < 0:
            raise ValidationError("Offset position cannot be negative for OFFSET")
        if end_row > df.height or end_col > df.width:
            raise ValidationError("Offset range exceeds array bounds for OFFSET")
        if height == 1 and width == 1:
            result = _to_decimal_if_numeric(df.item(start_row, start_col))
            # For scalar results, if output_filename is provided, create a single-value DataFrame
            if output_filename is not None:
                result_df = pl.DataFrame({"value": [result]})
                from tools.toolset_utils import save_df_to_analysis_dir
                return save_df_to_analysis_dir(run_context, result_df, output_filename)
            return result
        else:
            # Updated block: enable allow_object to handle Object dtype values
            result_df = df.slice(start_row, height).select(df.columns[start_col:end_col])
            # For DataFrame results, if output_filename is provided, save to file
            if output_filename is not None:
                from tools.toolset_utils import save_df_to_analysis_dir
                return save_df_to_analysis_dir(run_context, result_df, output_filename)
            return result_df
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise CalculationError(f"OFFSET calculation failed: {str(e)}")


def INDIRECT(ref_text: str, *, a1_style: bool = True) -> str:
    """
    Create references based on text strings (simplified implementation).
    """
    if not isinstance(ref_text, str):
        raise ValidationError("ref_text must be a string for INDIRECT")
    if not ref_text.strip():
        raise ValidationError("ref_text cannot be empty for INDIRECT")
    try:
        return ref_text.strip()
    except Exception as e:
        raise CalculationError(f"INDIRECT calculation failed: {str(e)}")


def ADDRESS(
    row_num: int,
    column_num: int,
    *,
    abs_num: int = 1,
    a1: bool = True,
    sheet_text: Optional[str] = None
) -> str:
    """
    Create cell address as text.
    """
    if not isinstance(row_num, int) or not isinstance(column_num, int):
        raise ValidationError("row_num and column_num must be integers for ADDRESS")
    if row_num < 1 or column_num < 1:
        raise ValidationError("row_num and column_num must be positive for ADDRESS")
    if abs_num not in [1, 2, 3, 4]:
        raise ValidationError("abs_num must be 1, 2, 3, or 4 for ADDRESS")
    try:
        if a1:
            col_letter = ""
            col = column_num
            while col > 0:
                col -= 1
                col_letter = chr(65 + (col % 26)) + col_letter
                col //= 26
            if abs_num == 1:
                address = f"${col_letter}${row_num}"
            elif abs_num == 2:
                address = f"{col_letter}${row_num}"
            elif abs_num == 3:
                address = f"${col_letter}{row_num}"
            else:
                address = f"{col_letter}{row_num}"
        else:
            if abs_num == 1:
                address = f"R{row_num}C{column_num}"
            elif abs_num == 2:
                address = f"R{row_num}C[{column_num}]"
            elif abs_num == 3:
                address = f"R[{row_num}]C{column_num}"
            else:
                address = f"R[{row_num}]C[{column_num}]"
        if sheet_text:
            address = f"{sheet_text}!{address}"
        return address
    except Exception as e:
        raise CalculationError(f"ADDRESS calculation failed: {str(e)}")


def ROW(
    run_context: Any,
    reference: Optional[Union[list[list[Any]], pl.DataFrame, str, Path]] = None,
    *,
    output_filename: Optional[str] = None
) -> Union[int, list[int], Path]:
    """
    Return row number of reference.
    """
    try:
        if reference is None:
            result = 1
            if output_filename is not None:
                result_df = pl.DataFrame({"row_number": [result]})
                from tools.toolset_utils import save_df_to_analysis_dir
                return save_df_to_analysis_dir(run_context, result_df, output_filename)
            return result

        # Handle original input type to determine proper return format
        if isinstance(reference, pl.DataFrame):
            if reference.height == 1:
                result = [1]
            else:
                result = list(range(1, reference.height + 1))
        else:
            df = _validate_table_array(reference, "ROW", run_context)
            if df.height == 1:
                result = [1]
            else:
                result = list(range(1, df.height + 1))

        # For array results, if output_filename is provided, save to file
        if output_filename is not None:
            result_df = pl.DataFrame({"row_numbers": result})
            from tools.toolset_utils import save_df_to_analysis_dir
            return save_df_to_analysis_dir(run_context, result_df, output_filename)
        return result
    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"ROW calculation failed: {str(e)}")


def COLUMN(
    run_context: Any,
    reference: Optional[Union[list[list[Any]], pl.DataFrame, str, Path]] = None,
    *,
    output_filename: Optional[str] = None
) -> Union[int, list[int], Path]:
    """
    Return column number of reference.
    """
    try:
        if reference is None:
            result = 1
            if output_filename is not None:
                result_df = pl.DataFrame({"column_number": [result]})
                from tools.toolset_utils import save_df_to_analysis_dir
                return save_df_to_analysis_dir(run_context, result_df, output_filename)
            return result

        # Handle original input type to determine proper return format
        if isinstance(reference, pl.DataFrame):
            if reference.width == 1:
                result = [1]
            else:
                result = list(range(1, reference.width + 1))
        else:
            df = _validate_table_array(reference, "COLUMN", run_context)
            if df.width == 1:
                result = [1]
            else:
                result = list(range(1, df.width + 1))

        # For array results, if output_filename is provided, save to file
        if output_filename is not None:
            result_df = pl.DataFrame({"column_numbers": result})
            from tools.toolset_utils import save_df_to_analysis_dir
            return save_df_to_analysis_dir(run_context, result_df, output_filename)
        return result
    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"COLUMN calculation failed: {str(e)}")


def ROWS(
    run_context: Any,
    array: Union[list[list[Any]], pl.DataFrame, np.ndarray, str, Path],
    *,
    output_filename: Optional[str] = None
) -> Union[int, Path]:
    """
    Return number of rows in reference.
    """
    try:
        df = _validate_table_array(array, "ROWS", run_context)
        result = df.height
        # For scalar results, if output_filename is provided, create a single-value DataFrame
        if output_filename is not None:
            result_df = pl.DataFrame({"row_count": [result]})
            from tools.toolset_utils import save_df_to_analysis_dir
            return save_df_to_analysis_dir(run_context, result_df, output_filename)
        return result
    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"ROWS calculation failed: {str(e)}")


def COLUMNS(
    run_context: Any,
    array: Union[list[list[Any]], pl.DataFrame, np.ndarray, str, Path],
    *,
    output_filename: Optional[str] = None
) -> Union[int, Path]:
    """
    Return number of columns in reference.
    """
    try:
        df = _validate_table_array(array, "COLUMNS", run_context)
        result = df.width
        # For scalar results, if output_filename is provided, create a single-value DataFrame
        if output_filename is not None:
            result_df = pl.DataFrame({"column_count": [result]})
            from tools.toolset_utils import save_df_to_analysis_dir
            return save_df_to_analysis_dir(run_context, result_df, output_filename)
        return result
    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"COLUMNS calculation failed: {str(e)}")
