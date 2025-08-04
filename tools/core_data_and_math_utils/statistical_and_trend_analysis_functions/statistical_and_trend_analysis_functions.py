"""
Statistical & Trend Analysis Functions

These functions support forecasting and risk analysis by uncovering trends and relationships in data.
All functions use Decimal precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, Union
from functools import lru_cache
import polars as pl
import numpy as np
from scipy import stats
import math
from pathlib import Path
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


def _validate_numeric_input(values: Any, function_name: str) -> pl.Series:
    """
    Standardized input validation for numeric data.

    Args:
        values: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.Series: Validated Polars Series

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid numeric values
    """
    try:
        # Convert to Polars Series for optimal processing
        if isinstance(values, (list, np.ndarray)):
            series = pl.Series(values)
        elif isinstance(values, pl.Series):
            series = values
        else:
            raise ValidationError(f"Unsupported input type for {function_name}: {type(values)}")

        # Check if series is empty
        if series.is_empty():
            raise ValidationError(f"Input values cannot be empty for {function_name}")

        # Check for null values
        if series.null_count() > 0:
            raise DataQualityError(
                f"Input contains null values for {function_name}",
                "Remove or replace null values with appropriate numeric values"
            )

        return series

    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid numeric values in {function_name}: {str(e)}",
            "Ensure all values are numeric (int, float, Decimal)"
        )


@lru_cache(maxsize=1024)
def _convert_to_decimal(value: Any) -> Decimal:
    """
    Safely convert value to Decimal with proper error handling.
    Uses LRU cache for performance optimization with frequently converted values.

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


def STDEV_P(run_context: Any, values: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate the standard deviation for a full population using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        values: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Population standard deviation

    Raises:
        ValidationError: If input is empty or invalid type
        DataQualityError: If input contains non-numeric values

    Example:
        >>> STDEV_P(ctx, [2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('2.0')
        >>> STDEV_P(ctx, "data_file.parquet")
        Decimal('1.58113883008419')
    """
    # Handle file path input
    if isinstance(values, (str, Path)):
        df = load_df(run_context, values)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_numeric_input(values, "STDEV_P")

    try:
        # Use Polars std with ddof=0 for population standard deviation
        polars_std = series.std(ddof=0)

        # Handle null result
        if polars_std is None:
            raise CalculationError("Cannot calculate standard deviation of empty dataset")

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(polars_std)

        return result

    except Exception as e:
        raise CalculationError(f"STDEV_P calculation failed: {str(e)}")


def STDEV_S(run_context: Any, values: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate the standard deviation for a sample using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        values: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Sample standard deviation

    Raises:
        ValidationError: If input is empty or invalid type
        DataQualityError: If input contains non-numeric values

    Example:
        >>> STDEV_S(ctx, [2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('2.1380899352993')
        >>> STDEV_S(ctx, "data_file.parquet")
        Decimal('1.58113883008419')
    """
    # Handle file path input
    if isinstance(values, (str, Path)):
        df = load_df(run_context, values)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_numeric_input(values, "STDEV_S")

    try:
        # Use Polars std with ddof=1 for sample standard deviation (default)
        polars_std = series.std(ddof=1)

        # Handle null result
        if polars_std is None:
            raise CalculationError("Cannot calculate standard deviation of empty dataset")

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(polars_std)

        return result

    except Exception as e:
        raise CalculationError(f"STDEV_S calculation failed: {str(e)}")


def VAR_P(run_context: Any, values: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate variance for a population using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        values: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Population variance

    Raises:
        ValidationError: If input is empty or invalid type
        DataQualityError: If input contains non-numeric values

    Example:
        >>> VAR_P(ctx, [2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('4.0')
        >>> VAR_P(ctx, "data_file.parquet")
        Decimal('2.5')
    """
    # Handle file path input
    if isinstance(values, (str, Path)):
        df = load_df(run_context, values)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_numeric_input(values, "VAR_P")

    try:
        # Use Polars var with ddof=0 for population variance
        polars_var = series.var(ddof=0)

        # Handle null result
        if polars_var is None:
            raise CalculationError("Cannot calculate variance of empty dataset")

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(polars_var)

        return result

    except Exception as e:
        raise CalculationError(f"VAR_P calculation failed: {str(e)}")


def VAR_S(run_context: Any, values: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate variance for a sample using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        values: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Sample variance

    Raises:
        ValidationError: If input is empty or invalid type
        DataQualityError: If input contains non-numeric values

    Example:
        >>> VAR_S(ctx, [2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('4.571428571428571')
        >>> VAR_S(ctx, "data_file.parquet")
        Decimal('2.5')
    """
    # Handle file path input
    if isinstance(values, (str, Path)):
        df = load_df(run_context, values)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_numeric_input(values, "VAR_S")

    try:
        # Use Polars var with ddof=1 for sample variance (default)
        polars_var = series.var(ddof=1)

        # Handle null result
        if polars_var is None:
            raise CalculationError("Cannot calculate variance of empty dataset")

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(polars_var)

        return result

    except Exception as e:
        raise CalculationError(f"VAR_S calculation failed: {str(e)}")


def MEDIAN(run_context: Any, values: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Determine the middle value in a dataset using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        values: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Median value

    Raises:
        ValidationError: If input is empty or invalid type
        DataQualityError: If input contains non-numeric values

    Example:
        >>> MEDIAN(ctx, [1, 2, 3, 4, 5])
        Decimal('3')
        >>> MEDIAN(ctx, [1, 2, 3, 4])
        Decimal('2.5')
        >>> MEDIAN(ctx, "data_file.parquet")
        Decimal('3')
    """
    # Handle file path input
    if isinstance(values, (str, Path)):
        df = load_df(run_context, values)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_numeric_input(values, "MEDIAN")

    try:
        # Use Polars median aggregation for performance
        polars_median = series.median()

        # Handle null result
        if polars_median is None:
            raise CalculationError("Cannot calculate median of empty dataset")

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(polars_median)

        return result

    except Exception as e:
        raise CalculationError(f"MEDIAN calculation failed: {str(e)}")


def MODE(run_context: Any, values: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Union[Decimal, list[Decimal]]:
    """
    Find the most frequently occurring value in a dataset using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        values: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal or list[Decimal]: Most frequent value(s)

    Raises:
        ValidationError: If input is empty or invalid type
        DataQualityError: If input contains non-numeric values

    Example:
        >>> MODE(ctx, [1, 2, 2, 3, 3, 3])
        Decimal('3')
        >>> MODE(ctx, [1, 1, 2, 2, 3])
        [Decimal('1'), Decimal('2')]
        >>> MODE(ctx, "data_file.parquet")
        Decimal('3')
    """
    # Handle file path input
    if isinstance(values, (str, Path)):
        df = load_df(run_context, values)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_numeric_input(values, "MODE")

    try:
        # Convert to list for processing
        values_list = series.to_list()

        # Count frequency of each value
        frequency_map = {}
        for value in values_list:
            decimal_value = _convert_to_decimal(value)
            frequency_map[decimal_value] = frequency_map.get(decimal_value, 0) + 1

        # Find maximum frequency
        max_frequency = max(frequency_map.values())

        # Find all values with maximum frequency
        modes = [value for value, freq in frequency_map.items() if freq == max_frequency]

        # Sort modes for consistent output
        modes.sort()

        # Return single mode or list of modes
        if len(modes) == 1:
            return modes[0]
        else:
            return modes

    except Exception as e:
        raise CalculationError(f"MODE calculation failed: {str(e)}")


def CORREL(run_context: Any, range1: Union[list[float], pl.Series, np.ndarray, str, Path], *, range2: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Measure the correlation between two datasets using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        range1: First range of values (float, Decimal, Polars Series, NumPy array, or file path)
        range2: Second range of values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Correlation coefficient (-1 to 1)

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        DataQualityError: If input contains non-numeric values

    Example:
        >>> CORREL(ctx, [1, 2, 3, 4, 5], range2=[2, 4, 6, 8, 10])
        Decimal('1.0')
        >>> CORREL(ctx, [1, 2, 3, 4, 5], range2=[5, 4, 3, 2, 1])
        Decimal('-1.0')
        >>> CORREL(ctx, "data1.parquet", range2="data2.parquet")
        Decimal('0.8')
    """
    # Handle file path input for range1
    if isinstance(range1, (str, Path)):
        df = load_df(run_context, range1)
        # Assume first column contains the numeric data
        series1 = df[df.columns[0]]
    else:
        # Input validation for direct data
        series1 = _validate_numeric_input(range1, "CORREL")

    # Handle file path input for range2
    if isinstance(range2, (str, Path)):
        df = load_df(run_context, range2)
        # Assume first column contains the numeric data
        series2 = df[df.columns[0]]
    else:
        # Input validation for direct data
        series2 = _validate_numeric_input(range2, "CORREL")

    # Check if lengths match
    if len(series1) != len(series2):
        raise ValidationError("Range1 and range2 must have the same length")

    try:
        # Create DataFrame for correlation calculation
        df = pl.DataFrame({
            "x": series1,
            "y": series2
        })

        # Use Polars correlation function
        correlation_matrix = df.corr()
        correlation_value = correlation_matrix.item(0, 1)  # Get correlation between x and y

        # Handle null result
        if correlation_value is None:
            raise CalculationError("Cannot calculate correlation - insufficient variance in data")

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(correlation_value)

        return result

    except Exception as e:
        raise CalculationError(f"CORREL calculation failed: {str(e)}")


def COVARIANCE_P(run_context: Any, range1: Union[list[float], pl.Series, np.ndarray, str, Path], *, range2: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate covariance for a population using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        range1: First range of values (float, Decimal, Polars Series, NumPy array, or file path)
        range2: Second range of values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Population covariance

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        DataQualityError: If input contains non-numeric values

    Example:
        >>> COVARIANCE_P(ctx, [1, 2, 3, 4, 5], range2=[2, 4, 6, 8, 10])
        Decimal('4.0')
        >>> COVARIANCE_P(ctx, "data1.parquet", range2="data2.parquet")
        Decimal('2.5')
    """
    # Handle file path input for range1
    if isinstance(range1, (str, Path)):
        df = load_df(run_context, range1)
        # Assume first column contains the numeric data
        series1 = df[df.columns[0]]
    else:
        # Input validation for direct data
        series1 = _validate_numeric_input(range1, "COVARIANCE_P")

    # Handle file path input for range2
    if isinstance(range2, (str, Path)):
        df = load_df(run_context, range2)
        # Assume first column contains the numeric data
        series2 = df[df.columns[0]]
    else:
        # Input validation for direct data
        series2 = _validate_numeric_input(range2, "COVARIANCE_P")

    # Check if lengths match
    if len(series1) != len(series2):
        raise ValidationError("Range1 and range2 must have the same length")

    try:
        # Convert to lists for processing
        values1 = series1.to_list()
        values2 = series2.to_list()

        # Convert to Decimal
        decimal_values1 = [_convert_to_decimal(v) for v in values1]
        decimal_values2 = [_convert_to_decimal(v) for v in values2]

        # Calculate means
        n = len(decimal_values1)
        mean1 = sum(decimal_values1) / n
        mean2 = sum(decimal_values2) / n

        # Calculate population covariance: Σ((xi - μx)(yi - μy)) / N
        covariance = Decimal('0')
        for x, y in zip(decimal_values1, decimal_values2):
            covariance += (x - mean1) * (y - mean2)

        result = covariance / n

        return result

    except Exception as e:
        raise CalculationError(f"COVARIANCE_P calculation failed: {str(e)}")


def COVARIANCE_S(run_context: Any, range1: Union[list[float], pl.Series, np.ndarray, str, Path], *, range2: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate covariance for a sample using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        range1: First range of values (float, Decimal, Polars Series, NumPy array, or file path)
        range2: Second range of values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Sample covariance

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        DataQualityError: If input contains non-numeric values

    Example:
        >>> COVARIANCE_S(ctx, [1, 2, 3, 4, 5], range2=[2, 4, 6, 8, 10])
        Decimal('5.0')
        >>> COVARIANCE_S(ctx, "data1.parquet", range2="data2.parquet")
        Decimal('3.333333333333333333333333333')
    """
    # Handle file path input for range1
    if isinstance(range1, (str, Path)):
        df = load_df(run_context, range1)
        # Assume first column contains the numeric data
        series1 = df[df.columns[0]]
    else:
        # Input validation for direct data
        series1 = _validate_numeric_input(range1, "COVARIANCE_S")

    # Handle file path input for range2
    if isinstance(range2, (str, Path)):
        df = load_df(run_context, range2)
        # Assume first column contains the numeric data
        series2 = df[df.columns[0]]
    else:
        # Input validation for direct data
        series2 = _validate_numeric_input(range2, "COVARIANCE_S")

    # Check if lengths match
    if len(series1) != len(series2):
        raise ValidationError("Range1 and range2 must have the same length")

    # Check minimum sample size
    if len(series1) < 2:
        raise ValidationError("Sample covariance requires at least 2 data points")

    try:
        # Convert to lists for processing
        values1 = series1.to_list()
        values2 = series2.to_list()

        # Convert to Decimal
        decimal_values1 = [_convert_to_decimal(v) for v in values1]
        decimal_values2 = [_convert_to_decimal(v) for v in values2]

        # Calculate means
        n = len(decimal_values1)
        mean1 = sum(decimal_values1) / n
        mean2 = sum(decimal_values2) / n

        # Calculate sample covariance: Σ((xi - μx)(yi - μy)) / (N-1)
        covariance = Decimal('0')
        for x, y in zip(decimal_values1, decimal_values2):
            covariance += (x - mean1) * (y - mean2)

        result = covariance / (n - 1)

        return result

    except Exception as e:
        raise CalculationError(f"COVARIANCE_S calculation failed: {str(e)}")


def TREND(run_context: Any, known_y: Union[list[float], pl.Series, np.ndarray, str, Path], *, known_x: Union[list[float], pl.Series, np.ndarray, str, Path, None] = None, new_x: Union[list[float], pl.Series, np.ndarray, str, Path, None] = None, const: bool = True, output_filename: str) -> Path:
    """
    Predict future values based on linear trends using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        known_y: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_x: Known x values (optional, defaults to 1, 2, 3, ...)
        new_x: New x values for prediction (optional, defaults to continuation of known_x)
        const: Whether to force intercept through zero (default True)
        output_filename: Filename to save prediction results

    Returns:
        Path: Path to saved prediction results

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If regression calculation fails

    Example:
        >>> TREND(ctx, [1, 2, 3, 4, 5], known_x=[1, 2, 3, 4, 5], new_x=[6, 7, 8], output_filename="trend_results.parquet")
        Path('/path/to/analysis/trend_results.parquet')
    """
    # Handle file path input for known_y
    if isinstance(known_y, (str, Path)):
        df = load_df(run_context, known_y)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_y, "TREND")

    # Handle known_x
    if known_x is None:
        # Default to 1, 2, 3, ... sequence
        x_series = pl.Series(range(1, len(y_series) + 1))
    elif isinstance(known_x, (str, Path)):
        df = load_df(run_context, known_x)
        x_series = df[df.columns[0]]
    else:
        x_series = _validate_numeric_input(known_x, "TREND")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_x and known_y must have the same length")

    # Handle new_x
    if new_x is None:
        # Default to continuation of known_x sequence
        last_x = x_series[-1]
        new_x_series = pl.Series([last_x + i for i in range(1, len(y_series) + 1)])
    elif isinstance(new_x, (str, Path)):
        df = load_df(run_context, new_x)
        new_x_series = df[df.columns[0]]
    else:
        new_x_series = _validate_numeric_input(new_x, "TREND")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()
        new_x_array = new_x_series.to_numpy()

        # Perform linear regression
        if const:
            # Standard linear regression with intercept
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
            predictions = slope * new_x_array + intercept
        else:
            # Force regression through origin (no intercept)
            slope = np.sum(x_array * y_array) / np.sum(x_array * x_array)
            predictions = slope * new_x_array

        # Convert predictions to Decimal
        decimal_predictions = [_convert_to_decimal(pred) for pred in predictions]

        # Create DataFrame from results
        result_df = pl.DataFrame({
            "x_values": new_x_series,
            "trend_predictions": decimal_predictions
        })

        # Save using save_df_to_analysis_dir function
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        raise CalculationError(f"TREND calculation failed: {str(e)}")


def FORECAST(run_context: Any, new_x: float, *, known_y: Union[list[float], pl.Series, np.ndarray, str, Path], known_x: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Predict a future value based on linear regression using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        new_x: New x value for prediction
        known_y: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_x: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Single predicted value

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If regression calculation fails

    Example:
        >>> FORECAST(ctx, 6, known_y=[1, 2, 3, 4, 5], known_x=[1, 2, 3, 4, 5])
        Decimal('6.0')
        >>> FORECAST(ctx, 10, known_y="y_data.parquet", known_x="x_data.parquet")
        Decimal('12.5')
    """
    # Handle file path input for known_y
    if isinstance(known_y, (str, Path)):
        df = load_df(run_context, known_y)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_y, "FORECAST")

    # Handle file path input for known_x
    if isinstance(known_x, (str, Path)):
        df = load_df(run_context, known_x)
        # Assume first column contains the numeric data
        x_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        x_series = _validate_numeric_input(known_x, "FORECAST")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_x and known_y must have the same length")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)

        # Calculate prediction
        prediction = slope * new_x + intercept

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(prediction)

        return result

    except Exception as e:
        raise CalculationError(f"FORECAST calculation failed: {str(e)}")


def FORECAST_LINEAR(run_context: Any, new_x: float, *, known_y: Union[list[float], pl.Series, np.ndarray, str, Path], known_x: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Predict a future value based on linear regression (newer version).
    This is an alias for FORECAST function.

    Args:
        run_context: RunContext object for file operations
        new_x: New x value for prediction
        known_y: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_x: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Single predicted value

    Example:
        >>> FORECAST_LINEAR(ctx, 6, known_y=[1, 2, 3, 4, 5], known_x=[1, 2, 3, 4, 5])
        Decimal('6.0')
    """
    return FORECAST(run_context, new_x, known_y=known_y, known_x=known_x)


def GROWTH(run_context: Any, known_y: Union[list[float], pl.Series, np.ndarray, str, Path], *, known_x: Union[list[float], pl.Series, np.ndarray, str, Path, None] = None, new_x: Union[list[float], pl.Series, np.ndarray, str, Path, None] = None, const: bool = True, output_filename: str) -> Path:
    """
    Forecast exponential growth trends using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        known_y: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_x: Known x values (optional, defaults to 1, 2, 3, ...)
        new_x: New x values for prediction (optional, defaults to continuation of known_x)
        const: Whether to include constant term (default True)
        output_filename: Filename to save prediction results

    Returns:
        Path: Path to saved prediction results

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If regression calculation fails

    Example:
        >>> GROWTH(ctx, [1, 2, 4, 8, 16], known_x=[1, 2, 3, 4, 5], new_x=[6, 7, 8], output_filename="growth_results.parquet")
        Path('/path/to/analysis/growth_results.parquet')
    """
    # Handle file path input for known_y
    if isinstance(known_y, (str, Path)):
        df = load_df(run_context, known_y)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_y, "GROWTH")

    # Handle known_x
    if known_x is None:
        # Default to 1, 2, 3, ... sequence
        x_series = pl.Series(range(1, len(y_series) + 1))
    elif isinstance(known_x, (str, Path)):
        df = load_df(run_context, known_x)
        x_series = df[df.columns[0]]
    else:
        x_series = _validate_numeric_input(known_x, "GROWTH")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_x and known_y must have the same length")

    # Handle new_x
    if new_x is None:
        # Default to continuation of known_x sequence
        last_x = x_series[-1]
        new_x_series = pl.Series([last_x + i for i in range(1, len(y_series) + 1)])
    elif isinstance(new_x, (str, Path)):
        df = load_df(run_context, new_x)
        new_x_series = df[df.columns[0]]
    else:
        new_x_series = _validate_numeric_input(new_x, "GROWTH")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()
        new_x_array = new_x_series.to_numpy()

        # Check for non-positive y values
        if np.any(y_array <= 0):
            raise CalculationError("GROWTH requires all y values to be positive")

        # Transform to logarithmic space for exponential regression
        log_y = np.log(y_array)

        # Perform linear regression on log-transformed data
        if const:
            # Standard linear regression with intercept
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, log_y)
            log_predictions = slope * new_x_array + intercept
        else:
            # Force regression through origin (no intercept)
            slope = np.sum(x_array * log_y) / np.sum(x_array * x_array)
            log_predictions = slope * new_x_array

        # Transform back from logarithmic space
        predictions = np.exp(log_predictions)

        # Convert predictions to Decimal
        decimal_predictions = [_convert_to_decimal(pred) for pred in predictions]

        # Create DataFrame from results
        result_df = pl.DataFrame({
            "x_values": new_x_series,
            "growth_predictions": decimal_predictions
        })

        # Save using save_df_to_analysis_dir function
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        raise CalculationError(f"GROWTH calculation failed: {str(e)}")


def SLOPE(run_context: Any, known_ys: Union[list[float], pl.Series, np.ndarray, str, Path], *, known_xs: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate slope of linear regression line using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        known_ys: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_xs: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Slope of regression line

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If regression calculation fails

    Example:
        >>> SLOPE(ctx, [1, 2, 3, 4, 5], known_xs=[1, 2, 3, 4, 5])
        Decimal('1.0')
        >>> SLOPE(ctx, "y_data.parquet", known_xs="x_data.parquet")
        Decimal('2.5')
    """
    # Handle file path input for known_ys
    if isinstance(known_ys, (str, Path)):
        df = load_df(run_context, known_ys)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_ys, "SLOPE")

    # Handle file path input for known_xs
    if isinstance(known_xs, (str, Path)):
        df = load_df(run_context, known_xs)
        # Assume first column contains the numeric data
        x_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        x_series = _validate_numeric_input(known_xs, "SLOPE")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_xs and known_ys must have the same length")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(slope)

        return result

    except Exception as e:
        raise CalculationError(f"SLOPE calculation failed: {str(e)}")


def INTERCEPT(run_context: Any, known_ys: Union[list[float], pl.Series, np.ndarray, str, Path], *, known_xs: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate y-intercept of linear regression line using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        known_ys: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_xs: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: Y-intercept of regression line

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If regression calculation fails

    Example:
        >>> INTERCEPT(ctx, [2, 4, 6, 8, 10], known_xs=[1, 2, 3, 4, 5])
        Decimal('0.0')
        >>> INTERCEPT(ctx, "y_data.parquet", known_xs="x_data.parquet")
        Decimal('1.5')
    """
    # Handle file path input for known_ys
    if isinstance(known_ys, (str, Path)):
        df = load_df(run_context, known_ys)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_ys, "INTERCEPT")

    # Handle file path input for known_xs
    if isinstance(known_xs, (str, Path)):
        df = load_df(run_context, known_xs)
        # Assume first column contains the numeric data
        x_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        x_series = _validate_numeric_input(known_xs, "INTERCEPT")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_xs and known_ys must have the same length")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(intercept)

        return result

    except Exception as e:
        raise CalculationError(f"INTERCEPT calculation failed: {str(e)}")


def RSQ(run_context: Any, known_ys: Union[list[float], pl.Series, np.ndarray, str, Path], *, known_xs: Union[list[float], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Calculate R-squared of linear regression using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        known_ys: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_xs: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

    Returns:
        Decimal: R-squared value (coefficient of determination)

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If regression calculation fails

    Example:
        >>> RSQ(ctx, [1, 2, 3, 4, 5], known_xs=[1, 2, 3, 4, 5])
        Decimal('1.0')
        >>> RSQ(ctx, "y_data.parquet", known_xs="x_data.parquet")
        Decimal('0.95')
    """
    # Handle file path input for known_ys
    if isinstance(known_ys, (str, Path)):
        df = load_df(run_context, known_ys)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_ys, "RSQ")

    # Handle file path input for known_xs
    if isinstance(known_xs, (str, Path)):
        df = load_df(run_context, known_xs)
        # Assume first column contains the numeric data
        x_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        x_series = _validate_numeric_input(known_xs, "RSQ")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_xs and known_ys must have the same length")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)

        # R-squared is the square of the correlation coefficient
        r_squared = r_value ** 2

        # Convert to Decimal for financial precision
        result = _convert_to_decimal(r_squared)

        return result

    except Exception as e:
        raise CalculationError(f"RSQ calculation failed: {str(e)}")


def LINEST(run_context: Any, known_ys: Union[list[float], pl.Series, np.ndarray, str, Path], *, known_xs: Union[list[float], pl.Series, np.ndarray, str, Path, None] = None, const: bool = True, stats_flag: bool = False, output_filename: str) -> Path:
    """
    Calculate linear regression statistics using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        known_ys: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_xs: Known x values (optional, defaults to 1, 2, 3, ...)
        const: Whether to include constant term (default True)
        stats_flag: Whether to include additional statistics (default False)
        output_filename: Filename to save regression statistics

    Returns:
        Path: Path to saved regression statistics

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If regression calculation fails

    Example:
        >>> LINEST(ctx, [1, 2, 3, 4, 5], known_xs=[1, 2, 3, 4, 5], output_filename="linest_results.parquet")
        Path('/path/to/analysis/linest_results.parquet')
    """
    # Handle file path input for known_ys
    if isinstance(known_ys, (str, Path)):
        df = load_df(run_context, known_ys)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_ys, "LINEST")

    # Handle known_xs
    if known_xs is None:
        # Default to 1, 2, 3, ... sequence
        x_series = pl.Series(range(1, len(y_series) + 1))
    elif isinstance(known_xs, (str, Path)):
        df = load_df(run_context, known_xs)
        x_series = df[df.columns[0]]
    else:
        x_series = _validate_numeric_input(known_xs, "LINEST")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_xs and known_ys must have the same length")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)

        # Calculate additional statistics if requested
        if stats_flag:
            # Calculate residuals and additional statistics
            predictions = slope * x_array + intercept
            residuals = y_array - predictions

            # Standard error of slope and intercept
            n = len(x_array)
            x_mean = np.mean(x_array)
            ss_xx = np.sum((x_array - x_mean) ** 2)
            mse = np.sum(residuals ** 2) / (n - 2)

            se_slope = np.sqrt(mse / ss_xx)
            se_intercept = np.sqrt(mse * (1/n + x_mean**2/ss_xx))

            # F-statistic
            ss_reg = np.sum((predictions - np.mean(y_array)) ** 2)
            f_stat = ss_reg / mse if mse > 0 else 0

            # Degrees of freedom
            df_reg = 1
            df_res = n - 2

            result_data = {
                "statistic": ["slope", "intercept", "se_slope", "se_intercept", "r_squared", "std_error", "f_statistic", "df_residual"],
                "value": [
                    _convert_to_decimal(slope),
                    _convert_to_decimal(intercept),
                    _convert_to_decimal(se_slope),
                    _convert_to_decimal(se_intercept),
                    _convert_to_decimal(r_value ** 2),
                    _convert_to_decimal(np.sqrt(mse)),
                    _convert_to_decimal(f_stat),
                    _convert_to_decimal(df_res)
                ]
            }
        else:
            # Basic statistics only
            result_data = {
                "statistic": ["slope", "intercept"],
                "value": [
                    _convert_to_decimal(slope),
                    _convert_to_decimal(intercept)
                ]
            }

        # Create DataFrame from results
        result_df = pl.DataFrame(result_data)

        # Save using save_df_to_analysis_dir function
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        raise CalculationError(f"LINEST calculation failed: {str(e)}")


def LOGEST(run_context: Any, known_ys: Union[list[float], pl.Series, np.ndarray, str, Path], *, known_xs: Union[list[float], pl.Series, np.ndarray, str, Path, None] = None, const: bool = True, stats_flag: bool = False, output_filename: str) -> Path:
    """
    Calculate exponential regression statistics using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        known_ys: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
        known_xs: Known x values (optional, defaults to 1, 2, 3, ...)
        const: Whether to include constant term (default True)
        stats_flag: Whether to include additional statistics (default False)
        output_filename: Filename to save regression statistics

    Returns:
        Path: Path to saved regression statistics

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If regression calculation fails

    Example:
        >>> LOGEST(ctx, [1, 2, 4, 8, 16], known_xs=[1, 2, 3, 4, 5], output_filename="logest_results.parquet")
        Path('/path/to/analysis/logest_results.parquet')
    """
    # Handle file path input for known_ys
    if isinstance(known_ys, (str, Path)):
        df = load_df(run_context, known_ys)
        # Assume first column contains the numeric data
        y_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        y_series = _validate_numeric_input(known_ys, "LOGEST")

    # Handle known_xs
    if known_xs is None:
        # Default to 1, 2, 3, ... sequence
        x_series = pl.Series(range(1, len(y_series) + 1))
    elif isinstance(known_xs, (str, Path)):
        df = load_df(run_context, known_xs)
        x_series = df[df.columns[0]]
    else:
        x_series = _validate_numeric_input(known_xs, "LOGEST")

    # Check if lengths match
    if len(x_series) != len(y_series):
        raise ValidationError("known_xs and known_ys must have the same length")

    try:
        # Convert to numpy arrays for scipy
        x_array = x_series.to_numpy()
        y_array = y_series.to_numpy()

        # Check for non-positive y values
        if np.any(y_array <= 0):
            raise CalculationError("LOGEST requires all y values to be positive")

        # Transform to logarithmic space for exponential regression
        log_y = np.log(y_array)

        # Perform linear regression on log-transformed data
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, log_y)

        # Transform back to exponential parameters
        # y = a * b^x, where ln(y) = ln(a) + x*ln(b)
        # So: a = exp(intercept), b = exp(slope)
        exp_base = np.exp(slope)
        exp_constant = np.exp(intercept)

        # Calculate additional statistics if requested
        if stats_flag:
            # Calculate residuals and additional statistics in log space
            log_predictions = slope * x_array + intercept
            log_residuals = log_y - log_predictions

            # Standard error of slope and intercept
            n = len(x_array)
            x_mean = np.mean(x_array)
            ss_xx = np.sum((x_array - x_mean) ** 2)
            mse = np.sum(log_residuals ** 2) / (n - 2)

            se_slope = np.sqrt(mse / ss_xx)
            se_intercept = np.sqrt(mse * (1/n + x_mean**2/ss_xx))

            # F-statistic
            ss_reg = np.sum((log_predictions - np.mean(log_y)) ** 2)
            f_stat = ss_reg / mse if mse > 0 else 0

            # Degrees of freedom
            df_res = n - 2

            result_data = {
                "statistic": ["exp_base", "exp_constant", "se_slope", "se_intercept", "r_squared", "std_error", "f_statistic", "df_residual"],
                "value": [
                    _convert_to_decimal(exp_base),
                    _convert_to_decimal(exp_constant),
                    _convert_to_decimal(se_slope),
                    _convert_to_decimal(se_intercept),
                    _convert_to_decimal(r_value ** 2),
                    _convert_to_decimal(np.sqrt(mse)),
                    _convert_to_decimal(f_stat),
                    _convert_to_decimal(df_res)
                ]
            }
        else:
            # Basic statistics only
            result_data = {
                "statistic": ["exp_base", "exp_constant"],
                "value": [
                    _convert_to_decimal(exp_base),
                    _convert_to_decimal(exp_constant)
                ]
            }

        # Create DataFrame from results
        result_df = pl.DataFrame(result_data)

        # Save using save_df_to_analysis_dir function
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        raise CalculationError(f"LOGEST calculation failed: {str(e)}")


def RANK(run_context: Any, number: float, *, ref: Union[list[float], pl.Series, np.ndarray, str, Path], order: int = 0) -> int:
    """
    Calculate rank of number in array using 1-based ranking.

    Args:
        run_context: RunContext object for file operations
        number: Number to rank
        ref: Reference array (float, Decimal, Polars Series, NumPy array, or file path)
        order: Sort order (0 = descending, 1 = ascending, default 0)

    Returns:
        int: Rank of number (1-based)

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If ranking calculation fails

    Example:
        >>> RANK(ctx, 85, ref=[100, 85, 90, 75, 95], order=0)
        4
        >>> RANK(ctx, 85, ref=[100, 85, 90, 75, 95], order=1)
        2
        >>> RANK(ctx, 85, ref="data_file.parquet", order=0)
        3
    """
    # Handle file path input for ref
    if isinstance(ref, (str, Path)):
        df = load_df(run_context, ref)
        # Assume first column contains the numeric data
        ref_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        ref_series = _validate_numeric_input(ref, "RANK")

    try:
        # Convert number to Decimal for consistency
        target_number = _convert_to_decimal(number)

        # Convert series to list for processing
        ref_list = ref_series.to_list()
        decimal_ref = [_convert_to_decimal(v) for v in ref_list]

        # Check if number exists in reference array
        if target_number not in decimal_ref:
            raise CalculationError(f"Number {target_number} not found in reference array")

        # Sort the reference array based on order
        if order == 0:
            # Descending order (default Excel behavior)
            sorted_ref = sorted(decimal_ref, reverse=True)
        else:
            # Ascending order
            sorted_ref = sorted(decimal_ref)

        # Find the rank (1-based)
        rank = sorted_ref.index(target_number) + 1

        return rank

    except Exception as e:
        raise CalculationError(f"RANK calculation failed: {str(e)}")


def PERCENTRANK(run_context: Any, array: Union[list[float], pl.Series, np.ndarray, str, Path], *, x: float, significance: int = 3) -> Decimal:
    """
    Calculate percentile rank using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        array: Array of values (float, Decimal, Polars Series, NumPy array, or file path)
        x: Value to rank
        significance: Number of significant digits (default 3)

    Returns:
        Decimal: Percentile rank (0 to 1)

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If calculation fails

    Example:
        >>> PERCENTRANK(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], x=7)
        Decimal('0.667')
        >>> PERCENTRANK(ctx, "data_file.parquet", x=85, significance=4)
        Decimal('0.7500')
    """
    # Handle file path input for array
    if isinstance(array, (str, Path)):
        df = load_df(run_context, array)
        # Assume first column contains the numeric data
        array_series = df[df.columns[0]]
    else:
        # Input validation for direct data
        array_series = _validate_numeric_input(array, "PERCENTRANK")

    try:
        # Convert x to Decimal for consistency
        target_value = _convert_to_decimal(x)

        # Convert series to list for processing
        array_list = array_series.to_list()
        decimal_array = [_convert_to_decimal(v) for v in array_list]

        # Sort the array
        sorted_array = sorted(decimal_array)
        n = len(sorted_array)

        # Find the position of x in the sorted array
        if target_value < sorted_array[0]:
            percentile_rank = Decimal('0')
        elif target_value > sorted_array[-1]:
            percentile_rank = Decimal('1')
        else:
            # Count values less than x
            count_less = sum(1 for v in sorted_array if v < target_value)
            # Count values equal to x
            count_equal = sum(1 for v in sorted_array if v == target_value)

            # Calculate percentile rank using Excel's method
            # PERCENTRANK = (count_less + 0.5 * count_equal) / n
            percentile_rank = (Decimal(count_less) + Decimal('0.5') * Decimal(count_equal)) / Decimal(n)

        # Round to specified significance
        quantizer = Decimal('0.1') ** significance
        result = percentile_rank.quantize(quantizer)

        return result

    except Exception as e:
        raise CalculationError(f"PERCENTRANK calculation failed: {str(e)}")
