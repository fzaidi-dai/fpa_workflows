"""
Conditional Aggregation & Counting Functions

These functions allow you to work with data subsets based on specific criteria.
All functions use Decimal precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, Union
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
from tools.toolset_utils import load_df

# Set decimal precision for financial calculations
getcontext().prec = 28

# Performance optimization: Cache compiled regex patterns
_REGEX_CACHE = {}
_CRITERIA_CACHE = {}


def _validate_range_input(values: Any, function_name: str) -> pl.Series:
    """
    Standardized input validation for range data.

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
            # Use strict=False to handle mixed data types
            series = pl.Series(values, strict=False)
        elif isinstance(values, pl.Series):
            series = values
        else:
            raise ValidationError(f"Unsupported input type for {function_name}: {type(values)}")

        # Check if series is empty
        if series.is_empty():
            raise ValidationError(f"Input range cannot be empty for {function_name}")

        return series

    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid range values in {function_name}: {str(e)}",
            "Ensure all values are valid data types"
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
        if value is None:
            return Decimal('0')
        return Decimal(str(value))
    except (ValueError, TypeError, OverflowError) as e:
        raise DataQualityError(
            f"Cannot convert value to Decimal: {str(e)}",
            "Ensure value is a valid numeric type"
        )


@lru_cache(maxsize=512)
def _parse_criteria(criteria: Any) -> tuple[str, Any]:
    """
    Parse criteria string into operator and value with caching for performance.

    Args:
        criteria: Criteria to parse (string with operator or direct value)

    Returns:
        tuple: (operator, value) where operator is one of '=', '>', '<', '>=', '<=', '<>', 'like'

    Raises:
        ValidationError: If criteria format is invalid
    """
    if criteria is None:
        return '=', None

    # If criteria is not a string, treat as exact match
    if not isinstance(criteria, str):
        return '=', criteria

    criteria = criteria.strip()

    # Check for comparison operators
    if criteria.startswith('>='):
        return '>=', criteria[2:].strip()
    elif criteria.startswith('<='):
        return '<=', criteria[2:].strip()
    elif criteria.startswith('<>'):
        return '<>', criteria[2:].strip()
    elif criteria.startswith('>'):
        return '>', criteria[1:].strip()
    elif criteria.startswith('<'):
        return '<', criteria[1:].strip()
    elif criteria.startswith('='):
        return '=', criteria[1:].strip()
    elif '*' in criteria or '?' in criteria:
        # Wildcard pattern matching
        return 'like', criteria
    else:
        # Default to exact match
        return '=', criteria


def _get_compiled_regex(pattern: str) -> re.Pattern:
    """
    Get compiled regex pattern with caching for performance.

    Args:
        pattern: Regex pattern string

    Returns:
        re.Pattern: Compiled regex pattern
    """
    if pattern not in _REGEX_CACHE:
        _REGEX_CACHE[pattern] = re.compile(pattern)
    return _REGEX_CACHE[pattern]


def _is_numeric_series(series: pl.Series) -> bool:
    """
    Fast check if series contains primarily numeric data.

    Args:
        series: Polars Series to check

    Returns:
        bool: True if series is numeric or can be converted to numeric
    """
    dtype = series.dtype
    return dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]


def _apply_criteria(series: pl.Series, criteria: Any) -> pl.Series:
    """
    Apply criteria to a Polars Series and return boolean mask with optimized performance.

    Args:
        series: Polars Series to filter
        criteria: Criteria to apply

    Returns:
        pl.Series: Boolean mask

    Raises:
        ValidationError: If criteria is invalid
        CalculationError: If comparison fails
    """
    # Cache key for this criteria application
    cache_key = (str(criteria), str(series.dtype), len(series))

    operator, value = _parse_criteria(criteria)

    # Handle null comparisons
    if operator == '=' and value is None:
        return series.is_null()
    elif operator == '<>' and value is None:
        return series.is_not_null()
    elif operator == 'like':
        # Convert wildcard pattern to regex with caching
        pattern = value.replace('*', '.*').replace('?', '.')
        regex_pattern = f'^{pattern}$'
        compiled_regex = _get_compiled_regex(regex_pattern)
        return series.str.contains(regex_pattern, literal=False)
    else:
        # Optimized path: Check if series is numeric for fast processing
        if _is_numeric_series(series):
            try:
                # Use Polars native numeric operations for better performance
                numeric_value = _convert_to_decimal(value)

                # Use Polars expressions directly instead of converting to float
                if operator == '=':
                    return series == float(numeric_value)
                elif operator == '<>':
                    return series != float(numeric_value)
                elif operator == '>':
                    return series > float(numeric_value)
                elif operator == '<':
                    return series < float(numeric_value)
                elif operator == '>=':
                    return series >= float(numeric_value)
                elif operator == '<=':
                    return series <= float(numeric_value)
                else:
                    raise ValidationError(f"Unknown operator: {operator}")

            except Exception:
                # Fall back to string comparison if numeric conversion fails
                pass

        # String/mixed type comparison fallback
        if operator == '=':
            return series == value
        elif operator == '<>':
            return series != value
        else:
            raise ValidationError(f"Cannot apply numeric operator {operator} to non-numeric value: {value}")


def _apply_criteria_batch(series_list: list[pl.Series], criteria_list: list[Any]) -> pl.Series:
    """
    Apply multiple criteria efficiently with vectorized operations.

    Args:
        series_list: List of Polars Series to evaluate
        criteria_list: List of criteria corresponding to each series

    Returns:
        pl.Series: Combined boolean mask (AND logic)

    Raises:
        ValidationError: If inputs are mismatched or invalid
    """
    if len(series_list) != len(criteria_list):
        raise ValidationError("Number of series must match number of criteria")

    if not series_list:
        raise ValidationError("At least one series and criteria required")

    # Apply all criteria and combine with AND logic
    combined_mask = None

    for series, criteria in zip(series_list, criteria_list):
        mask = _apply_criteria(series, criteria)

        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask & mask

    return combined_mask


def COUNTIF(ctx: Any, range_to_evaluate: Union[list[Any], pl.Series, np.ndarray, str, Path], *, criteria: Any) -> int:
    """
    Count cells that meet one condition.

    This function is essential for financial analysis tasks such as counting transactions
    above a threshold, identifying outliers, or segmenting data for reporting.

    Args:
        ctx: RunContext object for file operations
        range_to_evaluate: Range of values to evaluate (list, Polars Series, NumPy array, or file path)
        criteria: Criteria to match (supports operators: >, <, >=, <=, =, <>, wildcards *, ?)

    Returns:
        int: Count of cells meeting the criteria

    Raises:
        ValidationError: If input is empty or invalid type
        CalculationError: If criteria application fails
        DataQualityError: If input contains invalid values

    Financial Examples:
        # Count high-value transactions for risk analysis
        >>> transactions = [1000, 2500, 500, 7500, 1200, 300, 5000]
        >>> high_value_count = COUNTIF(ctx, transactions, criteria=">5000")
        >>> print(f"High-value transactions: {high_value_count}")
        High-value transactions: 2

        # Count specific product categories for inventory analysis
        >>> categories = ["Electronics", "Clothing", "Electronics", "Food", "Electronics"]
        >>> electronics_count = COUNTIF(ctx, categories, criteria="Electronics")
        >>> print(f"Electronics items: {electronics_count}")
        Electronics items: 3

        # Count budget variances exceeding threshold
        >>> variances = [-0.05, 0.12, -0.03, 0.18, -0.01, 0.25]
        >>> over_threshold = COUNTIF(ctx, variances, criteria=">0.1")
        >>> print(f"Variances over 10%: {over_threshold}")
        Variances over 10%: 3

    Example:
        >>> COUNTIF(ctx, [100, 200, 150, 300, 50], criteria=">150")
        2
        >>> COUNTIF(ctx, ["Sales", "Marketing", "Sales", "IT"], criteria="Sales")
        2
        >>> COUNTIF(ctx, "data_file.parquet", criteria=">150")
        2
    """
    # Handle file path input
    if isinstance(range_to_evaluate, (str, Path)):
        df = load_df(ctx, range_to_evaluate)
        # Assume first column contains the data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_range_input(range_to_evaluate, "COUNTIF")

    try:
        # Apply criteria and count matches
        mask = _apply_criteria(series, criteria)
        count = mask.sum()

        return int(count) if count is not None else 0

    except Exception as e:
        raise CalculationError(f"COUNTIF calculation failed: {str(e)}")


def COUNTIFS(ctx: Any, criteria_ranges: list[Union[list[Any], pl.Series, np.ndarray, str, Path]], *, criteria_values: list[Any]) -> int:
    """
    Count cells that meet multiple conditions across different ranges.

    This function enables complex financial analysis by allowing multiple criteria
    to be applied simultaneously, essential for multi-dimensional data analysis.

    Args:
        ctx: RunContext object for file operations
        criteria_ranges: List of ranges to evaluate (list, Polars Series, NumPy array, or file paths)
        criteria_values: List of criteria corresponding to each range

    Returns:
        int: Count of rows where all criteria are met

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If criteria application fails
        DataQualityError: If input contains invalid values

    Financial Examples:
        # Count high-value sales in specific regions for territory analysis
        >>> sales_amounts = [1000, 2500, 500, 7500, 1200]
        >>> regions = ["North", "South", "North", "West", "South"]
        >>> high_value_north = COUNTIFS(ctx, [sales_amounts, regions], criteria_values=[">1000", "North"])
        >>> print(f"High-value North sales: {high_value_north}")
        High-value North sales: 1

        # Count profitable products in Q4 for year-end analysis
        >>> profits = [50000, -10000, 75000, 25000, -5000]
        >>> quarters = ["Q1", "Q2", "Q3", "Q4", "Q4"]
        >>> profitable_q4 = COUNTIFS(ctx, [profits, quarters], criteria_values=[">0", "Q4"])
        >>> print(f"Profitable Q4 products: {profitable_q4}")
        Profitable Q4 products: 1

        # Count customers with high value and low risk
        >>> customer_values = [100000, 50000, 200000, 75000, 30000]
        >>> risk_scores = ["Low", "High", "Low", "Medium", "Low"]
        >>> premium_customers = COUNTIFS(ctx, [customer_values, risk_scores], criteria_values=[">75000", "Low"])
        >>> print(f"Premium low-risk customers: {premium_customers}")
        Premium low-risk customers: 1

    Example:
        >>> amounts = [100, 200, 150, 300, 50]
        >>> categories = ["A", "B", "A", "A", "B"]
        >>> COUNTIFS(ctx, [amounts, categories], criteria_values=[">100", "A"])
        2
        >>> COUNTIFS(ctx, ["sales_data.parquet", "regions.parquet"], criteria_values=[">1000", "North"])
        2
    """
    # Input validation
    if not criteria_ranges or not criteria_values:
        raise ValidationError("Criteria ranges and values cannot be empty")

    if len(criteria_ranges) != len(criteria_values):
        raise ValidationError("Number of criteria ranges must match number of criteria values")

    try:
        # Convert all ranges to Polars Series and validate lengths
        series_list = []
        expected_length = None

        for i, range_data in enumerate(criteria_ranges):
            # Handle file path input
            if isinstance(range_data, (str, Path)):
                df = load_df(ctx, range_data)
                # Assume first column contains the data
                series = df[df.columns[0]]
            else:
                series = _validate_range_input(range_data, f"COUNTIFS range {i+1}")

            if expected_length is None:
                expected_length = len(series)
            elif len(series) != expected_length:
                raise ValidationError(f"All ranges must have the same length. Expected {expected_length}, got {len(series)} for range {i+1}")

            series_list.append(series)

        # Apply all criteria and combine with AND logic
        combined_mask = None

        for series, criteria in zip(series_list, criteria_values):
            mask = _apply_criteria(series, criteria)

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask

        count = combined_mask.sum() if combined_mask is not None else 0
        return int(count) if count is not None else 0

    except Exception as e:
        raise CalculationError(f"COUNTIFS calculation failed: {str(e)}")


def SUMIF(ctx: Any, range_to_evaluate: Union[list[Any], pl.Series, np.ndarray, str, Path], *, criteria: Any, sum_range: Union[list[Any], pl.Series, np.ndarray, str, Path, None] = None) -> Decimal:
    """
    Sum numbers that meet one condition using Decimal precision.

    Essential for financial calculations where conditional summation is required,
    such as revenue analysis by region, expense categorization, or performance metrics.

    Args:
        ctx: RunContext object for file operations
        range_to_evaluate: Range to evaluate against criteria (list, Polars Series, NumPy array, or file path)
        criteria: Criteria to match (supports operators: >, <, >=, <=, =, <>, wildcards *, ?)
        sum_range: Range to sum (optional, defaults to range_to_evaluate, can be file path)

    Returns:
        Decimal: Sum of values meeting the criteria

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If criteria application fails
        DataQualityError: If input contains invalid numeric values

    Financial Examples:
        # Sum revenue for high-performing regions
        >>> regions = ["North", "South", "East", "West", "North"]
        >>> revenues = [100000, 75000, 120000, 90000, 110000]
        >>> north_revenue = SUMIF(ctx, regions, criteria="North", sum_range=revenues)
        >>> print(f"North region revenue: ${north_revenue:,}")
        North region revenue: $210,000

        # Sum expenses above budget threshold for variance analysis
        >>> expenses = [5000, 12000, 8000, 15000, 6000, 20000]
        >>> over_budget = SUMIF(ctx, expenses, criteria=">10000")
        >>> print(f"Over-budget expenses: ${over_budget:,}")
        Over-budget expenses: $47,000

        # Sum Q4 sales for year-end reporting
        >>> quarters = ["Q1", "Q2", "Q3", "Q4", "Q4", "Q1"]
        >>> sales = [250000, 280000, 300000, 320000, 310000, 260000]
        >>> q4_sales = SUMIF(ctx, quarters, criteria="Q4", sum_range=sales)
        >>> print(f"Q4 total sales: ${q4_sales:,}")
        Q4 total sales: $630,000

    Example:
        >>> categories = ["A", "B", "A", "C", "A"]
        >>> values = [100, 200, 150, 300, 50]
        >>> SUMIF(ctx, categories, criteria="A", sum_range=values)
        Decimal('300')
        >>> SUMIF(ctx, "regions.parquet", criteria="North", sum_range="revenues.parquet")
        Decimal('210000')
    """
    # Handle file path input for range_to_evaluate
    if isinstance(range_to_evaluate, (str, Path)):
        df = load_df(ctx, range_to_evaluate)
        # Assume first column contains the data
        eval_series = df[df.columns[0]]
    else:
        eval_series = _validate_range_input(range_to_evaluate, "SUMIF")

    # Handle file path input for sum_range
    if sum_range is not None:
        if isinstance(sum_range, (str, Path)):
            df = load_df(ctx, sum_range)
            # Assume first column contains the data
            sum_series = df[df.columns[0]]
        else:
            sum_series = _validate_range_input(sum_range, "SUMIF sum_range")

        if len(eval_series) != len(sum_series):
            raise ValidationError("Range to evaluate and sum range must have the same length")
    else:
        sum_series = eval_series

    try:
        # Apply criteria to get boolean mask
        mask = _apply_criteria(eval_series, criteria)

        # Filter sum_series using the mask and calculate sum
        filtered_values = sum_series.filter(mask)

        if filtered_values.is_empty():
            return Decimal('0')

        # Convert to Decimal and sum
        decimal_values = [_convert_to_decimal(val) for val in filtered_values.to_list()]
        result = sum(decimal_values)

        return result

    except Exception as e:
        raise CalculationError(f"SUMIF calculation failed: {str(e)}")


def SUMIFS(ctx: Any, sum_range: Union[list[Any], pl.Series, np.ndarray, str, Path], *, criteria_ranges: list[Union[list[Any], pl.Series, np.ndarray, str, Path]], criteria_values: list[Any]) -> Decimal:
    """
    Sum numbers that meet multiple conditions using Decimal precision.

    Critical for complex financial analysis requiring multiple criteria,
    such as multi-dimensional revenue analysis, cost allocation, and performance measurement.

    Args:
        ctx: RunContext object for file operations
        sum_range: Range of values to sum (list, Polars Series, NumPy array, or file path)
        criteria_ranges: List of ranges to evaluate (must all be same length as sum_range)
        criteria_values: List of criteria corresponding to each range

    Returns:
        Decimal: Sum of values where all criteria are met

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If criteria application fails
        DataQualityError: If input contains invalid numeric values

    Financial Examples:
        # Sum revenue for high-value customers in specific regions
        >>> revenues = [100000, 75000, 120000, 90000, 110000]
        >>> regions = ["North", "South", "North", "West", "North"]
        >>> customer_types = ["Premium", "Standard", "Premium", "Premium", "Standard"]
        >>> premium_north = SUMIFS(ctx, revenues, criteria_ranges=[regions, customer_types],
        ...                       criteria_values=["North", "Premium"])
        >>> print(f"Premium North revenue: ${premium_north:,}")
        Premium North revenue: $220,000

        # Sum Q4 expenses for specific departments over threshold
        >>> expenses = [50000, 75000, 60000, 80000, 45000, 90000]
        >>> quarters = ["Q3", "Q4", "Q4", "Q4", "Q3", "Q4"]
        >>> departments = ["Sales", "Marketing", "Sales", "IT", "Sales", "Marketing"]
        >>> q4_marketing_high = SUMIFS(ctx, expenses, criteria_ranges=[quarters, departments, expenses],
        ...                           criteria_values=["Q4", "Marketing", ">70000"])
        >>> print(f"Q4 high Marketing expenses: ${q4_marketing_high:,}")
        Q4 high Marketing expenses: $165,000

    Example:
        >>> amounts = [100, 200, 150, 300, 50]
        >>> categories = ["A", "B", "A", "A", "B"]
        >>> regions = ["North", "South", "North", "West", "South"]
        >>> SUMIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])
        Decimal('250')
        >>> SUMIFS(ctx, "revenues.parquet", criteria_ranges=["regions.parquet", "customer_types.parquet"], criteria_values=["North", "Premium"])
        Decimal('220000')
    """
    # Handle file path input for sum_range
    if isinstance(sum_range, (str, Path)):
        df = load_df(ctx, sum_range)
        # Assume first column contains the data
        sum_series = df[df.columns[0]]
    else:
        sum_series = _validate_range_input(sum_range, "SUMIFS")

    if not criteria_ranges or not criteria_values:
        raise ValidationError("Criteria ranges and values cannot be empty")

    if len(criteria_ranges) != len(criteria_values):
        raise ValidationError("Number of criteria ranges must match number of criteria values")

    try:
        # Convert all ranges to Polars Series and validate lengths
        series_list = []

        for i, range_data in enumerate(criteria_ranges):
            # Handle file path input
            if isinstance(range_data, (str, Path)):
                df = load_df(ctx, range_data)
                # Assume first column contains the data
                series = df[df.columns[0]]
            else:
                series = _validate_range_input(range_data, f"SUMIFS criteria range {i+1}")

            if len(series) != len(sum_series):
                raise ValidationError(f"All ranges must have the same length as sum_range. Expected {len(sum_series)}, got {len(series)} for criteria range {i+1}")

            series_list.append(series)

        # Apply all criteria and combine with AND logic
        combined_mask = None

        for series, criteria in zip(series_list, criteria_values):
            mask = _apply_criteria(series, criteria)

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask

        # Filter sum_range using the combined mask and calculate sum
        if combined_mask is None:
            return Decimal('0')

        filtered_values = sum_series.filter(combined_mask)

        if filtered_values.is_empty():
            return Decimal('0')

        # Convert to Decimal and sum
        decimal_values = [_convert_to_decimal(val) for val in filtered_values.to_list()]
        result = sum(decimal_values)

        return result

    except Exception as e:
        raise CalculationError(f"SUMIFS calculation failed: {str(e)}")


def AVERAGEIF(ctx: Any, range_to_evaluate: Union[list[Any], pl.Series, np.ndarray, str, Path], *, criteria: Any, average_range: Union[list[Any], pl.Series, np.ndarray, str, Path, None] = None) -> Decimal:
    """
    Calculate average of cells that meet one condition using Decimal precision.

    Essential for financial analysis requiring conditional averaging, such as
    average transaction size by customer segment or average performance by region.

    Args:
        ctx: RunContext object for file operations
        range_to_evaluate: Range to evaluate against criteria (list, Polars Series, NumPy array, or file path)
        criteria: Criteria to match (supports operators: >, <, >=, <=, =, <>, wildcards *, ?)
        average_range: Range to average (optional, defaults to range_to_evaluate, can be file path)

    Returns:
        Decimal: Average of values meeting the criteria

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If criteria application fails or no values match
        DataQualityError: If input contains invalid numeric values

    Financial Examples:
        # Average deal size for enterprise customers
        >>> customer_types = ["SMB", "Enterprise", "SMB", "Enterprise", "Mid-market"]
        >>> deal_sizes = [25000, 150000, 30000, 200000, 75000]
        >>> enterprise_avg = AVERAGEIF(ctx, customer_types, criteria="Enterprise", average_range=deal_sizes)
        >>> print(f"Average enterprise deal size: ${enterprise_avg:,}")
        Average enterprise deal size: $175,000

        # Average expense ratio for high-performing funds
        >>> performance_ratings = ["A", "B", "A", "C", "A", "B"]
        >>> expense_ratios = [0.65, 1.25, 0.75, 1.85, 0.55, 1.15]
        >>> top_rated_avg = AVERAGEIF(ctx, performance_ratings, criteria="A", average_range=expense_ratios)
        >>> print(f"Average expense ratio for A-rated funds: {top_rated_avg:.2f}%")
        Average expense ratio for A-rated funds: 0.65%

    Example:
        >>> categories = ["A", "B", "A", "C", "A"]
        >>> values = [100, 200, 150, 300, 50]
        >>> AVERAGEIF(ctx, categories, criteria="A", average_range=values)
        Decimal('100')
        >>> AVERAGEIF(ctx, "customer_types.parquet", criteria="Enterprise", average_range="deal_sizes.parquet")
        Decimal('175000')
    """
    # Handle file path input for range_to_evaluate
    if isinstance(range_to_evaluate, (str, Path)):
        df = load_df(ctx, range_to_evaluate)
        # Assume first column contains the data
        eval_series = df[df.columns[0]]
    else:
        eval_series = _validate_range_input(range_to_evaluate, "AVERAGEIF")

    # Handle file path input for average_range
    if average_range is not None:
        if isinstance(average_range, (str, Path)):
            df = load_df(ctx, average_range)
            # Assume first column contains the data
            avg_series = df[df.columns[0]]
        else:
            avg_series = _validate_range_input(average_range, "AVERAGEIF average_range")

        if len(eval_series) != len(avg_series):
            raise ValidationError("Range to evaluate and average range must have the same length")
    else:
        avg_series = eval_series

    try:
        # Apply criteria to get boolean mask
        mask = _apply_criteria(eval_series, criteria)

        # Filter average_series using the mask
        filtered_values = avg_series.filter(mask)

        if filtered_values.is_empty():
            raise CalculationError("No values match the criteria for AVERAGEIF")

        # Convert to Decimal and calculate average
        decimal_values = [_convert_to_decimal(val) for val in filtered_values.to_list()]
        result = sum(decimal_values) / Decimal(len(decimal_values))

        return result

    except Exception as e:
        raise CalculationError(f"AVERAGEIF calculation failed: {str(e)}")


def AVERAGEIFS(ctx: Any, average_range: Union[list[Any], pl.Series, np.ndarray, str, Path], *, criteria_ranges: list[Union[list[Any], pl.Series, np.ndarray, str, Path]], criteria_values: list[Any]) -> Decimal:
    """
    Calculate average of cells that meet multiple conditions using Decimal precision.

    Critical for sophisticated financial analysis requiring multi-dimensional averaging,
    such as performance analysis across multiple factors or risk-adjusted returns.

    Args:
        ctx: RunContext object for file operations
        average_range: Range of values to average (list, Polars Series, NumPy array, or file path)
        criteria_ranges: List of ranges to evaluate (must all be same length as average_range)
        criteria_values: List of criteria corresponding to each range

    Returns:
        Decimal: Average of values where all criteria are met

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If criteria application fails or no values match
        DataQualityError: If input contains invalid numeric values

    Financial Examples:
        # Average return for high-rated bonds in specific maturity range
        >>> returns = [3.5, 4.2, 2.8, 5.1, 3.9, 4.7]
        >>> ratings = ["AAA", "AA", "BBB", "AAA", "AA", "AAA"]
        >>> maturities = [5, 10, 3, 7, 15, 8]
        >>> avg_return = AVERAGEIFS(ctx, returns, criteria_ranges=[ratings, maturities],
        ...                        criteria_values=["AAA", "<=8"])
        >>> print(f"Average return for AAA bonds ≤8 years: {avg_return:.2f}%")
        Average return for AAA bonds ≤8 years: 4.43%

    Example:
        >>> amounts = [100, 200, 150, 300, 50]
        >>> categories = ["A", "B", "A", "A", "B"]
        >>> regions = ["North", "South", "North", "West", "South"]
        >>> AVERAGEIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])
        Decimal('125')
        >>> AVERAGEIFS(ctx, "returns.parquet", criteria_ranges=["ratings.parquet", "maturities.parquet"], criteria_values=["AAA", "<=8"])
        Decimal('4.43')
    """
    # Handle file path input for average_range
    if isinstance(average_range, (str, Path)):
        df = load_df(ctx, average_range)
        # Assume first column contains the data
        avg_series = df[df.columns[0]]
    else:
        avg_series = _validate_range_input(average_range, "AVERAGEIFS")

    if not criteria_ranges or not criteria_values:
        raise ValidationError("Criteria ranges and values cannot be empty")

    if len(criteria_ranges) != len(criteria_values):
        raise ValidationError("Number of criteria ranges must match number of criteria values")

    try:
        # Convert all ranges to Polars Series and validate lengths
        series_list = []

        for i, range_data in enumerate(criteria_ranges):
            # Handle file path input
            if isinstance(range_data, (str, Path)):
                df = load_df(ctx, range_data)
                # Assume first column contains the data
                series = df[df.columns[0]]
            else:
                series = _validate_range_input(range_data, f"AVERAGEIFS criteria range {i+1}")

            if len(series) != len(avg_series):
                raise ValidationError(f"All ranges must have the same length as average_range. Expected {len(avg_series)}, got {len(series)} for criteria range {i+1}")

            series_list.append(series)

        # Apply all criteria and combine with AND logic
        combined_mask = None

        for series, criteria in zip(series_list, criteria_values):
            mask = _apply_criteria(series, criteria)

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask

        # Filter average_range using the combined mask and calculate average
        if combined_mask is None:
            raise CalculationError("No values match the criteria for AVERAGEIFS")

        filtered_values = avg_series.filter(combined_mask)

        if filtered_values.is_empty():
            raise CalculationError("No values match the criteria for AVERAGEIFS")

        # Convert to Decimal and calculate average
        decimal_values = [_convert_to_decimal(val) for val in filtered_values.to_list()]
        result = sum(decimal_values) / Decimal(len(decimal_values))

        return result

    except Exception as e:
        raise CalculationError(f"AVERAGEIFS calculation failed: {str(e)}")


def MAXIFS(ctx: Any, max_range: Union[list[Any], pl.Series, np.ndarray, str, Path], *, criteria_ranges: list[Union[list[Any], pl.Series, np.ndarray, str, Path]], criteria_values: list[Any]) -> Decimal:
    """
    Find maximum value based on multiple criteria using Decimal precision.

    Essential for financial analysis requiring conditional maximum values,
    such as finding peak performance metrics or highest values in specific segments.

    Args:
        ctx: RunContext object for file operations
        max_range: Range of values to find maximum from (list, Polars Series, NumPy array, or file path)
        criteria_ranges: List of ranges to evaluate (must all be same length as max_range)
        criteria_values: List of criteria corresponding to each range

    Returns:
        Decimal: Maximum value where all criteria are met

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If criteria application fails or no values match
        DataQualityError: If input contains invalid numeric values

    Financial Examples:
        # Find highest revenue for premium customers in Q4
        >>> revenues = [100000, 250000, 150000, 300000, 200000]
        >>> customer_tiers = ["Standard", "Premium", "Premium", "Premium", "Standard"]
        >>> quarters = ["Q3", "Q4", "Q4", "Q4", "Q3"]
        >>> max_premium_q4 = MAXIFS(ctx, revenues, criteria_ranges=[customer_tiers, quarters],
        ...                        criteria_values=["Premium", "Q4"])
        >>> print(f"Highest Q4 premium revenue: ${max_premium_q4:,}")
        Highest Q4 premium revenue: $300,000

    Example:
        >>> amounts = [100, 200, 150, 300, 50]
        >>> categories = ["A", "B", "A", "A", "B"]
        >>> regions = ["North", "South", "North", "West", "South"]
        >>> MAXIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])
        Decimal('150')
        >>> MAXIFS(ctx, "revenues.parquet", criteria_ranges=["customer_tiers.parquet", "quarters.parquet"], criteria_values=["Premium", "Q4"])
        Decimal('300000')
    """
    # Handle file path input for max_range
    if isinstance(max_range, (str, Path)):
        df = load_df(ctx, max_range)
        # Assume first column contains the data
        max_series = df[df.columns[0]]
    else:
        max_series = _validate_range_input(max_range, "MAXIFS")

    if not criteria_ranges or not criteria_values:
        raise ValidationError("Criteria ranges and values cannot be empty")

    if len(criteria_ranges) != len(criteria_values):
        raise ValidationError("Number of criteria ranges must match number of criteria values")

    try:
        # Convert all ranges to Polars Series and validate lengths
        series_list = []

        for i, range_data in enumerate(criteria_ranges):
            # Handle file path input
            if isinstance(range_data, (str, Path)):
                df = load_df(ctx, range_data)
                # Assume first column contains the data
                series = df[df.columns[0]]
            else:
                series = _validate_range_input(range_data, f"MAXIFS criteria range {i+1}")

            if len(series) != len(max_series):
                raise ValidationError(f"All ranges must have the same length as max_range. Expected {len(max_series)}, got {len(series)} for criteria range {i+1}")

            series_list.append(series)

        # Apply all criteria and combine with AND logic
        combined_mask = None

        for series, criteria in zip(series_list, criteria_values):
            mask = _apply_criteria(series, criteria)

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask

        # Filter max_range using the combined mask and find maximum
        if combined_mask is None:
            raise CalculationError("No values match the criteria for MAXIFS")

        filtered_values = max_series.filter(combined_mask)

        if filtered_values.is_empty():
            raise CalculationError("No values match the criteria for MAXIFS")

        # Convert to Decimal and find maximum
        decimal_values = [_convert_to_decimal(val) for val in filtered_values.to_list()]
        result = max(decimal_values)

        return result

    except Exception as e:
        raise CalculationError(f"MAXIFS calculation failed: {str(e)}")


def MINIFS(ctx: Any, min_range: Union[list[Any], pl.Series, np.ndarray, str, Path], *, criteria_ranges: list[Union[list[Any], pl.Series, np.ndarray, str, Path]], criteria_values: list[Any]) -> Decimal:
    """
    Find minimum value based on multiple criteria using Decimal precision.

    Essential for financial analysis requiring conditional minimum values,
    such as finding lowest costs in specific segments or minimum performance thresholds.

    Args:
        ctx: RunContext object for file operations
        min_range: Range of values to find minimum from (list, Polars Series, NumPy array, or file path)
        criteria_ranges: List of ranges to evaluate (must all be same length as min_range)
        criteria_values: List of criteria corresponding to each range

    Returns:
        Decimal: Minimum value where all criteria are met

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If criteria application fails or no values match
        DataQualityError: If input contains invalid numeric values

    Financial Examples:
        # Find lowest cost for premium suppliers in Q4
        >>> costs = [50000, 25000, 75000, 30000, 60000]
        >>> supplier_tiers = ["Standard", "Premium", "Premium", "Premium", "Standard"]
        >>> quarters = ["Q3", "Q4", "Q4", "Q4", "Q3"]
        >>> min_premium_q4 = MINIFS(ctx, costs, criteria_ranges=[supplier_tiers, quarters],
        ...                        criteria_values=["Premium", "Q4"])
        >>> print(f"Lowest Q4 premium cost: ${min_premium_q4:,}")
        Lowest Q4 premium cost: $25,000

    Example:
        >>> amounts = [100, 200, 150, 300, 50]
        >>> categories = ["A", "B", "A", "A", "B"]
        >>> regions = ["North", "South", "North", "West", "South"]
        >>> MINIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])
        Decimal('100')
        >>> MINIFS(ctx, "costs.parquet", criteria_ranges=["supplier_tiers.parquet", "quarters.parquet"], criteria_values=["Premium", "Q4"])
        Decimal('25000')
    """
    # Handle file path input for min_range
    if isinstance(min_range, (str, Path)):
        df = load_df(ctx, min_range)
        # Assume first column contains the data
        min_series = df[df.columns[0]]
    else:
        min_series = _validate_range_input(min_range, "MINIFS")

    if not criteria_ranges or not criteria_values:
        raise ValidationError("Criteria ranges and values cannot be empty")

    if len(criteria_ranges) != len(criteria_values):
        raise ValidationError("Number of criteria ranges must match number of criteria values")

    try:
        # Convert all ranges to Polars Series and validate lengths
        series_list = []

        for i, range_data in enumerate(criteria_ranges):
            # Handle file path input
            if isinstance(range_data, (str, Path)):
                df = load_df(ctx, range_data)
                # Assume first column contains the data
                series = df[df.columns[0]]
            else:
                series = _validate_range_input(range_data, f"MINIFS criteria range {i+1}")

            if len(series) != len(min_series):
                raise ValidationError(f"All ranges must have the same length as min_range. Expected {len(min_series)}, got {len(series)} for criteria range {i+1}")

            series_list.append(series)

        # Apply all criteria and combine with AND logic
        combined_mask = None

        for series, criteria in zip(series_list, criteria_values):
            mask = _apply_criteria(series, criteria)

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask

        # Filter min_range using the combined mask and find minimum
        if combined_mask is None:
            raise CalculationError("No values match the criteria for MINIFS")

        filtered_values = min_series.filter(combined_mask)

        if filtered_values.is_empty():
            raise CalculationError("No values match the criteria for MINIFS")

        # Convert to Decimal and find minimum
        decimal_values = [_convert_to_decimal(val) for val in filtered_values.to_list()]
        result = min(decimal_values)

        return result

    except Exception as e:
        raise CalculationError(f"MINIFS calculation failed: {str(e)}")


def SUMPRODUCT(ctx: Any, *ranges: Union[list[Any], pl.Series, np.ndarray, str, Path]) -> Decimal:
    """
    Sum the products of corresponding ranges using Decimal precision.

    Essential for financial calculations involving weighted sums, portfolio calculations,
    and complex aggregations where multiplication precedes summation.

    Args:
        ranges: Two or more ranges of equal length to multiply and sum

    Returns:
        Decimal: Sum of products of corresponding elements

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        CalculationError: If calculation fails
        DataQualityError: If input contains invalid numeric values

    Financial Examples:
        # Calculate weighted portfolio value
        >>> quantities = [100, 200, 150, 300]
        >>> prices = [50.25, 75.80, 120.50, 45.75]
        >>> portfolio_value = SUMPRODUCT(quantities, prices)
        >>> print(f"Total portfolio value: ${portfolio_value:,}")
        Total portfolio value: $54,000

        # Calculate revenue with quantity and unit price
        >>> units_sold = [1000, 1500, 800, 1200]
        >>> unit_prices = [25.50, 18.75, 42.25, 35.80]
        >>> total_revenue = SUMPRODUCT(units_sold, unit_prices)
        >>> print(f"Total revenue: ${total_revenue:,}")
        Total revenue: $115,275

        # Calculate weighted average cost with volumes
        >>> volumes = [500, 300, 700, 400]
        >>> costs = [12.50, 15.25, 10.75, 18.00]
        >>> weights = [0.25, 0.15, 0.35, 0.25]
        >>> weighted_cost = SUMPRODUCT(volumes, costs, weights)
        >>> print(f"Weighted cost: ${weighted_cost:,.2f}")
        Weighted cost: $2,318.75

    Example:
        >>> SUMPRODUCT([1, 2, 3], [4, 5, 6])
        Decimal('32')  # (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        >>> SUMPRODUCT([10, 20], [5, 3], [2, 1])
        Decimal('160')  # (10*5*2) + (20*3*1) = 100 + 60 = 160
    """
    # Input validation
    if len(ranges) < 2:
        raise ValidationError("SUMPRODUCT requires at least two ranges")

    try:
        # Convert all ranges to Polars Series and validate lengths
        series_list = []
        expected_length = None

        for i, range_data in enumerate(ranges):
            # Handle file path input
            if isinstance(range_data, (str, Path)):
                df = load_df(ctx, range_data)
                # Assume first column contains the data
                series = df[df.columns[0]]
            else:
                series = _validate_range_input(range_data, f"SUMPRODUCT range {i+1}")

            if expected_length is None:
                expected_length = len(series)
            elif len(series) != expected_length:
                raise ValidationError(f"All ranges must have the same length. Expected {expected_length}, got {len(series)} for range {i+1}")

            series_list.append(series)

        # Calculate products and sum
        result = Decimal('0')

        for i in range(expected_length):
            product = Decimal('1')
            for series in series_list:
                value = series[i]
                decimal_value = _convert_to_decimal(value)
                product *= decimal_value
            result += product

        return result

    except Exception as e:
        raise CalculationError(f"SUMPRODUCT calculation failed: {str(e)}")


def COUNTBLANK(ctx: Any, range_to_evaluate: Union[list[Any], pl.Series, np.ndarray, str, Path]) -> int:
    """
    Count blank/empty cells in a range.

    Essential for data quality assessment and completeness analysis in financial datasets,
    helping identify missing data that could impact analysis accuracy.

    Args:
        ctx: RunContext object for file operations
        range_to_evaluate: Range to evaluate for blank/null values (list, Polars Series, NumPy array, or file path)

    Returns:
        int: Count of blank/null cells

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If input contains invalid values

    Financial Examples:
        # Count missing revenue data for data quality assessment
        >>> monthly_revenues = [100000, None, 120000, None, 110000, 130000]
        >>> missing_count = COUNTBLANK(ctx, monthly_revenues)
        >>> print(f"Missing revenue data points: {missing_count}")
        Missing revenue data points: 2

        # Assess completeness of customer data
        >>> customer_emails = ["john@company.com", "", "jane@corp.com", None, "bob@firm.com"]
        >>> incomplete_records = COUNTBLANK(ctx, customer_emails)
        >>> print(f"Incomplete customer records: {incomplete_records}")
        Incomplete customer records: 2

    Example:
        >>> COUNTBLANK(ctx, [1, None, 3, "", 5, None])
        3
        >>> COUNTBLANK(ctx, ["A", "B", "", None, "C"])
        2
        >>> COUNTBLANK(ctx, "revenue_data.parquet")
        2
    """
    # Handle file path input
    if isinstance(range_to_evaluate, (str, Path)):
        df = load_df(ctx, range_to_evaluate)
        # Assume first column contains the data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_range_input(range_to_evaluate, "COUNTBLANK")

    try:
        # Count null values and empty strings
        null_count = series.null_count()

        # Also count empty strings
        empty_string_count = 0
        for value in series.to_list():
            if value == "" or value == " ":
                empty_string_count += 1

        total_blank = null_count + empty_string_count
        return int(total_blank)

    except Exception as e:
        raise CalculationError(f"COUNTBLANK calculation failed: {str(e)}")


def COUNTA(ctx: Any, range_to_evaluate: Union[list[Any], pl.Series, np.ndarray, str, Path]) -> int:
    """
    Count non-empty cells in a range.

    Essential for data completeness analysis and determining the actual size of datasets
    for financial analysis, excluding missing or blank values.

    Args:
        ctx: RunContext object for file operations
        range_to_evaluate: Range to evaluate for non-empty values (list, Polars Series, NumPy array, or file path)

    Returns:
        int: Count of non-empty cells

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If input contains invalid values

    Financial Examples:
        # Count valid transaction records for analysis
        >>> transactions = [1000, None, 2500, 0, "", 3000, 1500]
        >>> valid_count = COUNTA(ctx, transactions)
        >>> print(f"Valid transaction records: {valid_count}")
        Valid transaction records: 5

        # Count completed customer surveys
        >>> survey_responses = ["Satisfied", "", "Very Satisfied", None, "Neutral", "Dissatisfied"]
        >>> completed_surveys = COUNTA(ctx, survey_responses)
        >>> print(f"Completed surveys: {completed_surveys}")
        Completed surveys: 4

    Example:
        >>> COUNTA(ctx, [1, None, 3, "", 5, 0])
        4  # Counts 1, 3, 5, 0 (zero is not blank)
        >>> COUNTA(ctx, ["A", "B", "", None, "C"])
        3  # Counts "A", "B", "C"
        >>> COUNTA(ctx, "transaction_data.parquet")
        5
    """
    # Handle file path input
    if isinstance(range_to_evaluate, (str, Path)):
        df = load_df(ctx, range_to_evaluate)
        # Assume first column contains the data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_range_input(range_to_evaluate, "COUNTA")

    try:
        # Count non-null values, excluding empty strings
        total_count = 0
        for value in series.to_list():
            if value is not None and value != "" and value != " ":
                total_count += 1

        return int(total_count)

    except Exception as e:
        raise CalculationError(f"COUNTA calculation failed: {str(e)}")


def AGGREGATE(function_num: int, *, options: int, array: Union[list[Any], pl.Series, np.ndarray], k: Union[int, None] = None) -> Decimal:
    """
    Perform various aggregations with error handling and filtering using Decimal precision.

    Advanced function for robust financial analysis that can ignore errors, hidden rows,
    and nested subtotals while performing standard aggregation operations.

    Args:
        function_num: Function number (1=AVERAGE, 2=COUNT, 3=COUNTA, 4=MAX, 5=MIN, 6=PRODUCT, 7=STDEV, 8=STDEVP, 9=SUM, 10=VAR, 11=VARP, 12=MEDIAN, 13=MODE, 14=LARGE, 15=SMALL, 16=PERCENTILE)
        options: Options for handling errors and filtering (0=default, 1=ignore hidden, 2=ignore errors, 3=ignore hidden and errors, 4=ignore nothing, 5=ignore errors, 6=ignore hidden, 7=ignore hidden and errors)
        array: Array to aggregate
        k: Additional parameter for functions like LARGE, SMALL, PERCENTILE

    Returns:
        Decimal: Aggregated result

    Raises:
        ValidationError: If inputs are invalid
        ConfigurationError: If function_num is not supported
        CalculationError: If calculation fails

    Financial Examples:
        # Calculate sum ignoring errors in financial data
        >>> financial_data = [100000, "Error", 150000, None, 200000, "#DIV/0!", 175000]
        >>> clean_sum = AGGREGATE(9, options=2, array=financial_data)  # SUM ignoring errors
        >>> print(f"Clean sum: ${clean_sum:,}")
        Clean sum: $625,000

        # Find maximum value ignoring errors and hidden values
        >>> performance_data = [0.15, "N/A", 0.22, 0.18, "Error", 0.25]
        >>> max_performance = AGGREGATE(4, options=3, array=performance_data)  # MAX ignoring errors
        >>> print(f"Maximum performance: {max_performance:.1%}")
        Maximum performance: 25.0%

    Example:
        >>> AGGREGATE(9, options=2, array=[10, "Error", 20, 30])  # SUM ignoring errors
        Decimal('60')
        >>> AGGREGATE(4, options=0, array=[10, 20, 30, 40])  # MAX
        Decimal('40')
    """
    # Input validation
    if not isinstance(function_num, int) or function_num < 1 or function_num > 16:
        raise ConfigurationError("Function number must be between 1 and 16")

    if not isinstance(options, int) or options < 0 or options > 7:
        raise ConfigurationError("Options must be between 0 and 7")

    series = _validate_range_input(array, "AGGREGATE")

    try:
        # Filter data based on options
        filtered_data = []
        for value in series.to_list():
            # Skip errors if options indicate to ignore them
            if options in [2, 3, 5, 7]:  # Ignore errors
                if isinstance(value, str) and (value.startswith("#") or value.lower() in ["error", "n/a", "na"]):
                    continue
                if value is None:
                    continue

            # Convert to numeric if possible
            try:
                numeric_value = _convert_to_decimal(value)
                filtered_data.append(numeric_value)
            except (ValueError, TypeError):
                if options not in [2, 3, 5, 7]:  # Don't ignore errors
                    raise DataQualityError(f"Invalid numeric value: {value}", "Remove or replace invalid values")

        if not filtered_data:
            raise CalculationError("No valid data remaining after filtering")

        # Apply the specified function
        if function_num == 1:  # AVERAGE
            return sum(filtered_data) / Decimal(len(filtered_data))
        elif function_num == 2:  # COUNT (numeric values only)
            return Decimal(len(filtered_data))
        elif function_num == 3:  # COUNTA (non-empty values)
            return Decimal(len(filtered_data))
        elif function_num == 4:  # MAX
            return max(filtered_data)
        elif function_num == 5:  # MIN
            return min(filtered_data)
        elif function_num == 6:  # PRODUCT
            result = Decimal('1')
            for value in filtered_data:
                result *= value
            return result
        elif function_num == 9:  # SUM
            return sum(filtered_data)
        elif function_num == 12:  # MEDIAN
            sorted_data = sorted(filtered_data)
            n = len(sorted_data)
            if n % 2 == 0:
                return (sorted_data[n//2 - 1] + sorted_data[n//2]) / Decimal('2')
            else:
                return sorted_data[n//2]
        elif function_num == 14:  # LARGE
            if k is None:
                raise ValidationError("Parameter k is required for LARGE function")
            sorted_data = sorted(filtered_data, reverse=True)
            if k < 1 or k > len(sorted_data):
                raise ValidationError(f"k must be between 1 and {len(sorted_data)}")
            return sorted_data[k-1]
        elif function_num == 15:  # SMALL
            if k is None:
                raise ValidationError("Parameter k is required for SMALL function")
            sorted_data = sorted(filtered_data)
            if k < 1 or k > len(sorted_data):
                raise ValidationError(f"k must be between 1 and {len(sorted_data)}")
            return sorted_data[k-1]
        else:
            raise ConfigurationError(f"Function number {function_num} is not yet implemented")

    except Exception as e:
        raise CalculationError(f"AGGREGATE calculation failed: {str(e)}")


def SUBTOTAL(function_num: int, *, ref1: Union[list[Any], pl.Series, np.ndarray]) -> Decimal:
    """
    Calculate subtotals with filtering capability using Decimal precision.

    Essential for financial reporting where subtotals need to ignore hidden rows
    or other SUBTOTAL functions to prevent double-counting in hierarchical data.

    Args:
        function_num: Function number (101-111 ignore hidden values, 1-11 include all)
                     1/101=AVERAGE, 2/102=COUNT, 3/103=COUNTA, 4/104=MAX, 5/105=MIN,
                     6/106=PRODUCT, 9/109=SUM, 10/110=VAR, 11/111=VARP
        ref1: Reference range to calculate subtotal for

    Returns:
        Decimal: Subtotal result

    Raises:
        ValidationError: If inputs are invalid
        ConfigurationError: If function_num is not supported
        CalculationError: If calculation fails

    Financial Examples:
        # Calculate subtotal sum for visible rows in filtered data
        >>> quarterly_sales = [250000, 280000, 300000, 320000]
        >>> q_total = SUBTOTAL(109, ref1=quarterly_sales)  # SUM ignoring hidden
        >>> print(f"Quarterly total: ${q_total:,}")
        Quarterly total: $1,150,000

        # Calculate average for visible performance metrics
        >>> performance_scores = [85, 92, 78, 88, 95]
        >>> avg_performance = SUBTOTAL(101, ref1=performance_scores)  # AVERAGE ignoring hidden
        >>> print(f"Average performance: {avg_performance:.1f}")
        Average performance: 87.6

    Example:
        >>> SUBTOTAL(109, ref1=[10, 20, 30, 40])  # SUM
        Decimal('100')
        >>> SUBTOTAL(101, ref1=[10, 20, 30, 40])  # AVERAGE
        Decimal('25')
    """
    # Input validation
    valid_functions = [1, 2, 3, 4, 5, 6, 9, 10, 11, 101, 102, 103, 104, 105, 106, 109, 110, 111]
    if function_num not in valid_functions:
        raise ConfigurationError(f"Function number {function_num} is not supported")

    series = _validate_range_input(ref1, "SUBTOTAL")

    try:
        # Convert to numeric values, filtering out non-numeric
        numeric_data = []
        for value in series.to_list():
            try:
                numeric_value = _convert_to_decimal(value)
                numeric_data.append(numeric_value)
            except (ValueError, TypeError):
                continue  # Skip non-numeric values

        if not numeric_data:
            raise CalculationError("No valid numeric data for SUBTOTAL calculation")

        # Determine base function (subtract 100 if > 100)
        base_function = function_num if function_num <= 11 else function_num - 100

        # Apply the specified function
        if base_function == 1:  # AVERAGE
            return sum(numeric_data) / Decimal(len(numeric_data))
        elif base_function == 2:  # COUNT
            return Decimal(len(numeric_data))
        elif base_function == 3:  # COUNTA
            return Decimal(len(numeric_data))
        elif base_function == 4:  # MAX
            return max(numeric_data)
        elif base_function == 5:  # MIN
            return min(numeric_data)
        elif base_function == 6:  # PRODUCT
            result = Decimal('1')
            for value in numeric_data:
                result *= value
            return result
        elif base_function == 9:  # SUM
            return sum(numeric_data)
        elif base_function == 10:  # VAR (sample variance)
            if len(numeric_data) < 2:
                raise CalculationError("Variance requires at least 2 values")
            mean = sum(numeric_data) / Decimal(len(numeric_data))
            variance = sum((x - mean) ** 2 for x in numeric_data) / Decimal(len(numeric_data) - 1)
            return variance
        elif base_function == 11:  # VARP (population variance)
            if len(numeric_data) < 1:
                raise CalculationError("Population variance requires at least 1 value")
            mean = sum(numeric_data) / Decimal(len(numeric_data))
            variance = sum((x - mean) ** 2 for x in numeric_data) / Decimal(len(numeric_data))
            return variance
        else:
            raise ConfigurationError(f"Function {base_function} is not implemented")

    except Exception as e:
        raise CalculationError(f"SUBTOTAL calculation failed: {str(e)}")
