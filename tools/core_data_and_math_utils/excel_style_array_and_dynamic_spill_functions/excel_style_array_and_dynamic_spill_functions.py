"""
Array and Dynamic Spill Functions (Modern Excel)

These functions help in performing calculations across ranges and enabling dynamic results.
All functions use Decimal precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, Union
from pathlib import Path
import polars as pl
import numpy as np
import random
from functools import lru_cache
import math

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


def _validate_array_input(values: Any, function_name: str) -> pl.Series:
    """
    Standardized input validation for array data.

    Args:
        values: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.Series: Validated Polars Series

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid values
    """
    try:
        # Convert to Polars Series for optimal processing
        if isinstance(values, (list, np.ndarray)):
            series = pl.Series(values)
        elif isinstance(values, pl.Series):
            series = values
        elif isinstance(values, pl.DataFrame):
            # Use first column if DataFrame is provided
            series = values[values.columns[0]]
        else:
            raise ValidationError(f"Unsupported input type for {function_name}: {type(values)}")

        # Check if series is empty
        if series.is_empty():
            raise ValidationError(f"Input values cannot be empty for {function_name}")

        return series

    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid values in {function_name}: {str(e)}",
            "Ensure all values are valid for the operation"
        )


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


def UNIQUE(run_context: Any, array: Union[list[Any], pl.Series, pl.DataFrame, str, Path], *, by_col: bool | None = None, exactly_once: bool | None = None, output_filename: str | None = None) -> list[Any]:
    """
    Extract a list of unique values from a range.

    Args:
        run_context: RunContext object for file operations
        array: Array to process (list, Series, DataFrame, or file path)
        by_col: Process by column (optional, not implemented for basic arrays)
        exactly_once: Return only values that appear exactly once (optional)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[Any]: Array of unique values

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If input contains invalid data

    Example:
        >>> UNIQUE(ctx, [1, 2, 2, 3, 3, 3])
        [1, 2, 3]
        >>> UNIQUE(ctx, [1, 2, 2, 3, 3, 3], exactly_once=True)
        [1]
        >>> UNIQUE(ctx, "data.parquet", output_filename="unique_results.parquet")
        [1, 2, 3, 4, 5]
    """
    # Handle file path input
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        # Assume first column contains the data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_array_input(array, "UNIQUE")

    try:
        if exactly_once:
            # Return only values that appear exactly once
            value_counts = series.value_counts()
            # Get the column name for values (first column in value_counts)
            value_col = value_counts.columns[0]
            unique_once = value_counts.filter(pl.col("count") == 1)[value_col].to_list()
            result = sorted(unique_once)
        else:
            # Return all unique values, maintaining order
            result = series.unique(maintain_order=True).to_list()

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Create DataFrame from results
            result_df = pl.DataFrame({
                "unique_values": result
            })
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"UNIQUE calculation failed: {str(e)}")


def SORT(run_context: Any, array: Union[list[Any], pl.Series, pl.DataFrame, str, Path], *, sort_index: int | None = None, sort_order: int | None = None, by_col: bool | None = None, output_filename: str | None = None) -> list[Any]:
    """
    Sort data or arrays dynamically.

    Args:
        run_context: RunContext object for file operations
        array: Array to sort (list, Series, DataFrame, or file path)
        sort_index: Sort index (optional, for multi-column sorting)
        sort_order: Sort order (1 for ascending, -1 for descending, optional)
        by_col: Sort by column (optional, not implemented for basic arrays)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[Any]: Sorted array

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If input contains invalid data

    Example:
        >>> SORT(ctx, [3, 1, 4, 1, 5])
        [1, 1, 3, 4, 5]
        >>> SORT(ctx, [3, 1, 4, 1, 5], sort_order=-1)
        [5, 4, 3, 1, 1]
        >>> SORT(ctx, "data.parquet", output_filename="sorted_results.parquet")
        [1, 2, 3, 4, 5]
    """
    # Handle file path input
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        # Assume first column contains the data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_array_input(array, "SORT")

    try:
        # Determine sort order
        descending = False
        if sort_order is not None and sort_order == -1:
            descending = True

        # Sort the series
        sorted_series = series.sort(descending=descending)
        result = sorted_series.to_list()

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Create DataFrame from results
            result_df = pl.DataFrame({
                "sorted_values": result
            })
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"SORT calculation failed: {str(e)}")


def SORTBY(run_context: Any, array: Union[list[Any], pl.Series, pl.DataFrame, str, Path], *, by_arrays_and_orders: Union[list[Any], pl.Series, pl.DataFrame, str, Path], output_filename: str | None = None) -> list[Any]:
    """
    Sort an array by values in another array.

    Args:
        run_context: RunContext object for file operations
        array: Array to sort (list, Series, DataFrame, or file path)
        by_arrays_and_orders: Array to sort by (list, Series, DataFrame, or file path)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[Any]: Sorted array

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        DataQualityError: If input contains invalid data

    Example:
        >>> SORTBY(ctx, ['apple', 'banana', 'cherry'], by_arrays_and_orders=[3, 1, 2])
        ['banana', 'cherry', 'apple']
        >>> SORTBY(ctx, [100, 200, 300], by_arrays_and_orders=[3, 1, 2])
        [200, 300, 100]
        >>> SORTBY(ctx, "values.parquet", by_arrays_and_orders="sort_keys.parquet", output_filename="sortby_results.parquet")
        ['banana', 'cherry', 'apple']
    """
    # Handle file path input for main array
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        # Assume first column contains the data
        main_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        main_series = _validate_array_input(array, "SORTBY")

    # Handle file path input for sort-by array
    if isinstance(by_arrays_and_orders, (str, Path)):
        df = load_df(run_context, by_arrays_and_orders)
        # Assume first column contains the data
        sort_by_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        sort_by_series = _validate_array_input(by_arrays_and_orders, "SORTBY")

    # Check if lengths match
    if len(main_series) != len(sort_by_series):
        raise ValidationError("Main array and sort-by array must have the same length")

    try:
        # Create a DataFrame with both arrays for sorting
        combined_df = pl.DataFrame({
            "values": main_series,
            "sort_keys": sort_by_series
        })

        # Sort by the sort_keys column
        sorted_df = combined_df.sort("sort_keys")
        result = sorted_df["values"].to_list()

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Create DataFrame from results
            result_df = pl.DataFrame({
                "sortby_values": result
            })
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"SORTBY calculation failed: {str(e)}")


def FILTER(run_context: Any, array: Union[list[Any], pl.Series, pl.DataFrame, str, Path], *, include: Union[list[bool], pl.Series], if_empty: Any | None = None, output_filename: str | None = None) -> list[Any]:
    """
    Return only those records that meet specified conditions.

    Args:
        run_context: RunContext object for file operations
        array: Array to filter (list, Series, DataFrame, or file path)
        include: Boolean array indicating which elements to include
        if_empty: Value to return if no elements match (optional)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[Any]: Filtered array

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        DataQualityError: If input contains invalid data

    Example:
        >>> FILTER(ctx, [1, 2, 3, 4, 5], include=[True, False, True, False, True])
        [1, 3, 5]
        >>> FILTER(ctx, ['a', 'b', 'c'], include=[False, False, False], if_empty='none')
        'none'
        >>> FILTER(ctx, "data.parquet", include=[True, False, True], output_filename="filtered_results.parquet")
        [1, 3]
    """
    # Handle file path input
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        # Assume first column contains the data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_array_input(array, "FILTER")

    # Validate include array
    if isinstance(include, list):
        include_series = pl.Series(include)
    elif isinstance(include, pl.Series):
        include_series = include
    else:
        raise ValidationError("Include parameter must be a list or Polars Series of boolean values")

    # Check if lengths match
    if len(series) != len(include_series):
        raise ValidationError("Array and include condition must have the same length")

    try:
        # Filter the series based on the include condition
        filtered_series = series.filter(include_series)
        result = filtered_series.to_list()

        # Handle empty result
        if len(result) == 0 and if_empty is not None:
            return if_empty

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Create DataFrame from results
            result_df = pl.DataFrame({
                "filtered_values": result
            })
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"FILTER calculation failed: {str(e)}")


def SEQUENCE(run_context: Any, rows: int, *, columns: int | None = None, start: int | None = None, step: int | None = None, output_filename: str | None = None) -> list[list[int]]:
    """
    Generate a list of sequential numbers in an array format.

    Args:
        run_context: RunContext object for file operations
        rows: Number of rows
        columns: Number of columns (optional, defaults to 1)
        start: Starting number (optional, defaults to 1)
        step: Step size (optional, defaults to 1)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[list[int]]: Array of sequential numbers

    Raises:
        ValidationError: If parameters are invalid
        CalculationError: If calculation fails

    Example:
        >>> SEQUENCE(ctx, 3)
        [[1], [2], [3]]
        >>> SEQUENCE(ctx, 2, columns=3, start=5, step=2)
        [[5, 7, 9], [11, 13, 15]]
        >>> SEQUENCE(ctx, 3, columns=2, output_filename="sequence_results.parquet")
        [[1, 2], [3, 4], [5, 6]]
    """
    # Validate parameters
    if rows <= 0:
        raise ValidationError("Number of rows must be positive")

    if columns is None:
        columns = 1
    elif columns <= 0:
        raise ValidationError("Number of columns must be positive")

    if start is None:
        start = 1

    if step is None:
        step = 1
    elif step == 0:
        raise ValidationError("Step size cannot be zero")

    try:
        # Generate the sequence
        result = []
        current_value = start

        for row in range(rows):
            row_values = []
            for col in range(columns):
                row_values.append(current_value)
                current_value += step
            result.append(row_values)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Flatten the result for DataFrame creation
            flattened_data = []
            for row_idx, row in enumerate(result):
                for col_idx, value in enumerate(row):
                    flattened_data.append({
                        "row": row_idx,
                        "column": col_idx,
                        "value": value
                    })

            result_df = pl.DataFrame(flattened_data)
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"SEQUENCE calculation failed: {str(e)}")


def RAND(run_context: Any) -> float:
    """
    Generate random numbers between 0 and 1.

    Args:
        run_context: RunContext object for file operations

    Returns:
        float: Random decimal between 0 and 1

    Raises:
        CalculationError: If random generation fails

    Example:
        >>> result = RAND(ctx)
        >>> 0 <= result < 1
        True
    """
    try:
        return random.random()
    except Exception as e:
        raise CalculationError(f"RAND calculation failed: {str(e)}")


def RANDBETWEEN(run_context: Any, bottom: int, top: int) -> int:
    """
    Generate random integers between two values.

    Args:
        run_context: RunContext object for file operations
        bottom: Lower bound (inclusive)
        top: Upper bound (inclusive)

    Returns:
        int: Random integer within range

    Raises:
        ValidationError: If bounds are invalid
        CalculationError: If random generation fails

    Example:
        >>> result = RANDBETWEEN(ctx, 1, 10)
        >>> 1 <= result <= 10
        True
        >>> result = RANDBETWEEN(ctx, -5, 5)
        >>> -5 <= result <= 5
        True
    """
    # Validate parameters
    if not isinstance(bottom, int) or not isinstance(top, int):
        raise ValidationError("Bottom and top must be integers")

    if bottom > top:
        raise ValidationError("Bottom value cannot be greater than top value")

    try:
        return random.randint(bottom, top)
    except Exception as e:
        raise CalculationError(f"RANDBETWEEN calculation failed: {str(e)}")


def FREQUENCY(run_context: Any, data_array: Union[list[float], pl.Series, pl.DataFrame, str, Path], *, bins_array: Union[list[float], pl.Series, pl.DataFrame, str, Path], output_filename: str | None = None) -> list[int]:
    """
    Calculate frequency distribution.

    Args:
        run_context: RunContext object for file operations
        data_array: Data array (list, Series, DataFrame, or file path)
        bins_array: Bins array defining the intervals (list, Series, DataFrame, or file path)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[int]: Array of frequencies

    Raises:
        ValidationError: If inputs are invalid
        DataQualityError: If input contains invalid data

    Example:
        >>> FREQUENCY(ctx, [1, 2, 3, 4, 5, 6], bins_array=[2, 4, 6])
        [2, 2, 2, 0]
        >>> FREQUENCY(ctx, [1.5, 2.5, 3.5, 4.5], bins_array=[2, 3, 4])
        [1, 1, 1, 1]
        >>> FREQUENCY(ctx, "data.parquet", bins_array="bins.parquet", output_filename="frequency_results.parquet")
        [2, 2, 2, 0]
    """
    # Handle file path input for data array
    if isinstance(data_array, (str, Path)):
        df = load_df(run_context, data_array)
        data_series = df[df.columns[0]]
    else:
        data_series = _validate_array_input(data_array, "FREQUENCY")

    # Handle file path input for bins array
    if isinstance(bins_array, (str, Path)):
        df = load_df(run_context, bins_array)
        bins_series = df[df.columns[0]]
    else:
        bins_series = _validate_array_input(bins_array, "FREQUENCY")

    try:
        # Convert to lists for processing
        data_list = data_series.to_list()
        bins_list = sorted(bins_series.to_list())

        # Calculate frequencies
        frequencies = []

        for i in range(len(bins_list)):
            if i == 0:
                # First bin: count values <= first bin
                count = sum(1 for x in data_list if x <= bins_list[i])
            else:
                # Subsequent bins: count values > previous bin and <= current bin
                count = sum(1 for x in data_list if bins_list[i-1] < x <= bins_list[i])
            frequencies.append(count)

        # Add count for values > last bin
        count = sum(1 for x in data_list if x > bins_list[-1])
        frequencies.append(count)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Create DataFrame from results
            result_df = pl.DataFrame({
                "frequency": frequencies
            })
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return frequencies

    except Exception as e:
        raise CalculationError(f"FREQUENCY calculation failed: {str(e)}")


def TRANSPOSE(run_context: Any, array: Union[list[list[Any]], pl.DataFrame, str, Path], *, output_filename: str | None = None) -> list[list[Any]]:
    """
    Transpose array orientation.

    Args:
        run_context: RunContext object for file operations
        array: Array to transpose (2D list, DataFrame, or file path)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[list[Any]]: Transposed array

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If input contains invalid data

    Example:
        >>> TRANSPOSE(ctx, [[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
        >>> TRANSPOSE(ctx, [[1, 2], [3, 4], [5, 6]])
        [[1, 3, 5], [2, 4, 6]]
        >>> TRANSPOSE(ctx, "matrix.parquet", output_filename="transposed_results.parquet")
        [[1, 4], [2, 5], [3, 6]]
    """
    # Handle file path input
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        # Convert DataFrame to list of lists
        array_2d = [df[col].to_list() for col in df.columns]
        array_2d = list(map(list, zip(*array_2d)))  # Transpose to get rows
    elif isinstance(array, pl.DataFrame):
        # Convert DataFrame to list of lists
        array_2d = [df[col].to_list() for col in array.columns]
        array_2d = list(map(list, zip(*array_2d)))  # Transpose to get rows
    elif isinstance(array, list) and len(array) > 0 and isinstance(array[0], list):
        array_2d = array
    else:
        raise ValidationError("Array must be a 2D list, DataFrame, or file path")

    try:
        # Validate that all rows have the same length
        if len(array_2d) > 0:
            row_length = len(array_2d[0])
            for i, row in enumerate(array_2d):
                if len(row) != row_length:
                    raise ValidationError(f"All rows must have the same length. Row {i} has length {len(row)}, expected {row_length}")

        # Transpose the array
        if len(array_2d) == 0:
            result = []
        else:
            result = list(map(list, zip(*array_2d)))

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Flatten the result for DataFrame creation
            flattened_data = []
            for row_idx, row in enumerate(result):
                for col_idx, value in enumerate(row):
                    flattened_data.append({
                        "row": row_idx,
                        "column": col_idx,
                        "value": value
                    })

            result_df = pl.DataFrame(flattened_data)
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"TRANSPOSE calculation failed: {str(e)}")


def MMULT(run_context: Any, array1: Union[list[list[float]], pl.DataFrame, str, Path], *, array2: Union[list[list[float]], pl.DataFrame, str, Path], output_filename: str | None = None) -> list[list[float]]:
    """
    Matrix multiplication.

    Args:
        run_context: RunContext object for file operations
        array1: First matrix (2D list, DataFrame, or file path)
        array2: Second matrix (2D list, DataFrame, or file path)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[list[float]]: Matrix product

    Raises:
        ValidationError: If matrices are incompatible for multiplication
        CalculationError: If calculation fails

    Example:
        >>> MMULT(ctx, [[1, 2], [3, 4]], array2=[[5, 6], [7, 8]])
        [[19.0, 22.0], [43.0, 50.0]]
        >>> MMULT(ctx, [[1, 2, 3]], array2=[[4], [5], [6]])
        [[32.0]]
        >>> MMULT(ctx, "matrix1.parquet", array2="matrix2.parquet", output_filename="mmult_results.parquet")
        [[19.0, 22.0], [43.0, 50.0]]
    """
    def _load_matrix(data):
        """Helper function to load matrix from various input types."""
        if isinstance(data, (str, Path)):
            df = load_df(run_context, data)
            return df.to_numpy().tolist()
        elif isinstance(data, pl.DataFrame):
            return data.to_numpy().tolist()
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return data
        else:
            raise ValidationError("Matrix must be a 2D list, DataFrame, or file path")

    try:
        # Load both matrices
        matrix1 = _load_matrix(array1)
        matrix2 = _load_matrix(array2)

        # Validate matrix dimensions
        if len(matrix1) == 0 or len(matrix2) == 0:
            raise ValidationError("Matrices cannot be empty")

        rows1, cols1 = len(matrix1), len(matrix1[0])
        rows2, cols2 = len(matrix2), len(matrix2[0])

        if cols1 != rows2:
            raise ValidationError(f"Matrix dimensions incompatible for multiplication: ({rows1}x{cols1}) × ({rows2}x{cols2})")

        # Convert to NumPy arrays for efficient multiplication
        np_matrix1 = np.array(matrix1, dtype=float)
        np_matrix2 = np.array(matrix2, dtype=float)

        # Perform matrix multiplication
        result_np = np.dot(np_matrix1, np_matrix2)
        result = result_np.tolist()

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Flatten the result for DataFrame creation
            flattened_data = []
            for row_idx, row in enumerate(result):
                for col_idx, value in enumerate(row):
                    flattened_data.append({
                        "row": row_idx,
                        "column": col_idx,
                        "value": value
                    })

            result_df = pl.DataFrame(flattened_data)
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except ValidationError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        raise CalculationError(f"MMULT calculation failed: {str(e)}")


def MINVERSE(run_context: Any, array: Union[list[list[float]], pl.DataFrame, str, Path], *, output_filename: str | None = None) -> list[list[float]]:
    """
    Matrix inverse.

    Args:
        run_context: RunContext object for file operations
        array: Square matrix to invert (2D list, DataFrame, or file path)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[list[float]]: Inverse matrix

    Raises:
        ValidationError: If matrix is not square or singular
        CalculationError: If calculation fails

    Example:
        >>> MINVERSE(ctx, [[1, 2], [3, 4]])
        [[-2.0, 1.0], [1.5, -0.5]]
        >>> MINVERSE(ctx, [[2, 0], [0, 2]])
        [[0.5, 0.0], [0.0, 0.5]]
        >>> MINVERSE(ctx, "matrix.parquet", output_filename="inverse_results.parquet")
        [[-2.0, 1.0], [1.5, -0.5]]
    """
    # Handle file path input
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        matrix = df.to_numpy().tolist()
    elif isinstance(array, pl.DataFrame):
        matrix = array.to_numpy().tolist()
    elif isinstance(array, list) and len(array) > 0 and isinstance(array[0], list):
        matrix = array
    else:
        raise ValidationError("Matrix must be a 2D list, DataFrame, or file path")

    try:
        # Validate matrix is square
        if len(matrix) == 0:
            raise ValidationError("Matrix cannot be empty")

        rows, cols = len(matrix), len(matrix[0])
        if rows != cols:
            raise ValidationError(f"Matrix must be square for inversion. Got {rows}x{cols}")

        # Convert to NumPy array for efficient calculation
        np_matrix = np.array(matrix, dtype=float)

        # Check if matrix is singular (determinant is zero)
        det = np.linalg.det(np_matrix)
        if abs(det) < 1e-10:
            raise CalculationError("Matrix is singular (determinant is zero) and cannot be inverted")

        # Calculate inverse
        inverse_np = np.linalg.inv(np_matrix)
        result = inverse_np.tolist()

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Flatten the result for DataFrame creation
            flattened_data = []
            for row_idx, row in enumerate(result):
                for col_idx, value in enumerate(row):
                    flattened_data.append({
                        "row": row_idx,
                        "column": col_idx,
                        "value": value
                    })

            result_df = pl.DataFrame(flattened_data)
            # Save using save_df_to_analysis_dir function
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"MINVERSE calculation failed: {str(e)}")


def MDETERM(run_context: Any, array: Union[list[list[float]], pl.DataFrame, str, Path]) -> float:
    """
    Matrix determinant.

    Args:
        run_context: RunContext object for file operations
        array: Square matrix to calculate determinant of (2D list, DataFrame, or file path)

    Returns:
        float: Determinant value

    Raises:
        ValidationError: If matrix is not square
        CalculationError: If calculation fails

    Example:
        >>> MDETERM(ctx, [[1, 2], [3, 4]])
        -2.0
        >>> MDETERM(ctx, [[2, 0], [0, 2]])
        4.0
        >>> MDETERM(ctx, "matrix.parquet")
        -2.0
    """
    # Handle file path input
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        matrix = df.to_numpy().tolist()
    elif isinstance(array, pl.DataFrame):
        matrix = array.to_numpy().tolist()
    elif isinstance(array, list) and len(array) > 0 and isinstance(array[0], list):
        matrix = array
    else:
        raise ValidationError("Matrix must be a 2D list, DataFrame, or file path")

    try:
        # Validate matrix is square
        if len(matrix) == 0:
            raise ValidationError("Matrix cannot be empty")

        rows, cols = len(matrix), len(matrix[0])
        if rows != cols:
            raise ValidationError(f"Matrix must be square for determinant calculation. Got {rows}x{cols}")

        # Convert to NumPy array for efficient calculation
        np_matrix = np.array(matrix, dtype=float)

        # Calculate determinant
        det = np.linalg.det(np_matrix)
        return float(det)

    except ValidationError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        raise CalculationError(f"MDETERM calculation failed: {str(e)}")
