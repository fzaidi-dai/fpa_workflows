"""
Data Filtering & Selection Functions

These functions provide atomic operations for filtering and selecting data subsets.
All functions use Polars for optimal performance and are optimized for AI agent integration.
"""

from typing import Any, Union, Dict
from pathlib import Path
from datetime import datetime, date
import polars as pl
from tools.tool_exceptions import (
    FPABaseException,
    RetryAfterCorrectionError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)
from tools.toolset_utils import load_df, save_df_to_analysis_dir


def _validate_dataframe_input(df: pl.DataFrame, function_name: str) -> pl.DataFrame:
    """
    Standardized input validation for DataFrame data.

    Args:
        df: Input DataFrame to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.DataFrame: Validated Polars DataFrame

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If DataFrame is empty
    """
    if not isinstance(df, pl.DataFrame):
        raise ValidationError(f"Input must be a Polars DataFrame for {function_name}")

    if df.is_empty():
        raise DataQualityError(
            f"Input DataFrame is empty for {function_name}",
            "Provide a DataFrame with at least one row of data"
        )

    return df


def _validate_column_exists(df: pl.DataFrame, column: str, function_name: str) -> None:
    """
    Validate that a column exists in the DataFrame.

    Args:
        df: DataFrame to check
        column: Column name to validate
        function_name: Name of calling function for error messages

    Raises:
        ValidationError: If column doesn't exist
    """
    if column not in df.columns:
        raise ValidationError(
            f"Column '{column}' not found in DataFrame for {function_name}. "
            f"Available columns: {df.columns}"
        )


def _parse_date_string(date_str: str, function_name: str) -> pl.Expr:
    """
    Parse date string into Polars date expression.

    Args:
        date_str: Date string to parse
        function_name: Name of calling function for error messages

    Returns:
        pl.Expr: Polars date expression

    Raises:
        DataQualityError: If date string is invalid
    """
    try:
        # Try to parse as ISO format first (YYYY-MM-DD)
        if len(date_str) == 10 and date_str.count('-') == 2:
            year, month, day = map(int, date_str.split('-'))
            return pl.date(year, month, day)
        else:
            # Try to parse with datetime and convert
            parsed_date = datetime.fromisoformat(date_str).date()
            return pl.date(parsed_date.year, parsed_date.month, parsed_date.day)
    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid date format '{date_str}' in {function_name}: {str(e)}",
            "Use ISO format (YYYY-MM-DD) or valid datetime string"
        )


def _create_operator_expression(column: str, operator: str, value: Any, function_name: str) -> pl.Expr:
    """
    Create Polars expression for comparison operations.

    Args:
        column: Column name
        operator: Comparison operator string
        value: Value to compare against
        function_name: Name of calling function for error messages

    Returns:
        pl.Expr: Polars filter expression

    Raises:
        ConfigurationError: If operator is not supported
    """
    col_expr = pl.col(column)

    operator_map = {
        '>': lambda c, v: c > v,
        '<': lambda c, v: c < v,
        '>=': lambda c, v: c >= v,
        '<=': lambda c, v: c <= v,
        '==': lambda c, v: c == v,
        '!=': lambda c, v: c != v,
        '=': lambda c, v: c == v,  # Alternative equality
    }

    if operator not in operator_map:
        raise ConfigurationError(
            f"Unsupported operator '{operator}' in {function_name}. "
            f"Supported operators: {list(operator_map.keys())}"
        )

    try:
        return operator_map[operator](col_expr, value)
    except Exception as e:
        raise DataQualityError(
            f"Error creating filter expression for {column} {operator} {value} in {function_name}: {str(e)}",
            "Ensure value type is compatible with column data type"
        )


def FILTER_BY_DATE_RANGE(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    date_column: str,
    start_date: str,
    end_date: str,
    output_filename: str
) -> Path:
    """
    Filter DataFrame by date range using Polars optimized date operations.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to filter or file path
        date_column: Name of the date column
        start_date: Start date (ISO format YYYY-MM-DD)
        end_date: End date (ISO format YYYY-MM-DD)
        output_filename: Filename to save filtered results

    Returns:
        Path: Path to saved filtered DataFrame

    Raises:
        ValidationError: If inputs are invalid
        DataQualityError: If date formats are invalid or column doesn't exist

    Example:
        >>> result_path = FILTER_BY_DATE_RANGE(
        ...     ctx, transactions_df,
        ...     date_column='transaction_date',
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31',
        ...     output_filename='filtered_transactions.parquet'
        ... )
    """
    # Load DataFrame if file path provided
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "FILTER_BY_DATE_RANGE")
    _validate_column_exists(df, date_column, "FILTER_BY_DATE_RANGE")

    try:
        # Parse date strings to Polars date expressions
        start_date_expr = _parse_date_string(start_date, "FILTER_BY_DATE_RANGE")
        end_date_expr = _parse_date_string(end_date, "FILTER_BY_DATE_RANGE")

        # Validate date range
        start_dt = datetime.fromisoformat(start_date).date()
        end_dt = datetime.fromisoformat(end_date).date()
        if start_dt > end_dt:
            raise ValidationError("Start date must be before or equal to end date")

        # Core filtering operation using Polars is_between for optimal performance
        # First ensure the date column is properly parsed as date type
        filtered_df = df.with_columns(
            pl.col(date_column).str.to_date(format="%Y-%m-%d", strict=False).alias(date_column)
        ).filter(
            pl.col(date_column).is_between(start_date_expr, end_date_expr)
        )

        # Validate results
        if filtered_df.is_empty():
            raise DataQualityError(
                f"No records found in date range {start_date} to {end_date}",
                "Check date range and ensure data exists in the specified period"
            )

        # Save results to analysis directory
        return save_df_to_analysis_dir(run_context, filtered_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Date range filtering failed: {str(e)}")


def FILTER_BY_VALUE(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    column: str,
    operator: str,
    value: Any,
    output_filename: str
) -> Path:
    """
    Filter DataFrame by column values using comparison operators.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to filter or file path
        column: Column name to filter on
        operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
        value: Value to compare against
        output_filename: Filename to save filtered results

    Returns:
        Path: Path to saved filtered DataFrame

    Raises:
        ValidationError: If inputs are invalid
        ConfigurationError: If operator is not supported
        DataQualityError: If filtering produces no results

    Example:
        >>> result_path = FILTER_BY_VALUE(
        ...     ctx, sales_df,
        ...     column='amount',
        ...     operator='>',
        ...     value=1000,
        ...     output_filename='high_value_sales.parquet'
        ... )
    """
    # Load DataFrame if file path provided
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "FILTER_BY_VALUE")
    _validate_column_exists(df, column, "FILTER_BY_VALUE")

    try:
        # Create filter expression
        filter_expr = _create_operator_expression(column, operator, value, "FILTER_BY_VALUE")

        # Core filtering operation
        filtered_df = df.filter(filter_expr)

        # Validate results
        if filtered_df.is_empty():
            raise DataQualityError(
                f"No records found matching condition: {column} {operator} {value}",
                "Adjust filter criteria or check data values"
            )

        # Save results to analysis directory
        return save_df_to_analysis_dir(run_context, filtered_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Value filtering failed: {str(e)}")


def FILTER_BY_MULTIPLE_CONDITIONS(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    conditions_dict: Dict[str, Any],
    output_filename: str
) -> Path:
    """
    Filter DataFrame by multiple conditions using AND logic.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to filter or file path
        conditions_dict: Dictionary of conditions {column: value} or {column: 'operator:value'}
        output_filename: Filename to save filtered results

    Returns:
        Path: Path to saved filtered DataFrame

    Raises:
        ValidationError: If inputs are invalid
        ConfigurationError: If condition format is invalid
        DataQualityError: If filtering produces no results

    Example:
        >>> result_path = FILTER_BY_MULTIPLE_CONDITIONS(
        ...     ctx, df,
        ...     conditions_dict={'region': 'North', 'sales': '>:1000', 'status': 'active'},
        ...     output_filename='filtered_data.parquet'
        ... )
    """
    # Load DataFrame if file path provided
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "FILTER_BY_MULTIPLE_CONDITIONS")

    if not conditions_dict:
        raise ValidationError("Conditions dictionary cannot be empty")

    try:
        filter_expressions = []

        for column, condition in conditions_dict.items():
            # Validate column exists
            _validate_column_exists(df, column, "FILTER_BY_MULTIPLE_CONDITIONS")

            # Parse condition
            if isinstance(condition, str) and ':' in condition:
                # Format: 'operator:value' (e.g., '>:1000')
                operator, value_str = condition.split(':', 1)
                # Try to convert value to appropriate type
                try:
                    # Try int first, then float, then keep as string
                    if value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
                        value = int(value_str)
                    elif '.' in value_str:
                        value = float(value_str)
                    else:
                        value = value_str
                except ValueError:
                    value = value_str
            else:
                # Direct equality comparison
                operator = '=='
                value = condition

            # Create filter expression
            filter_expr = _create_operator_expression(column, operator, value, "FILTER_BY_MULTIPLE_CONDITIONS")
            filter_expressions.append(filter_expr)

        # Combine all conditions with AND logic
        combined_filter = filter_expressions[0]
        for expr in filter_expressions[1:]:
            combined_filter = combined_filter & expr

        # Core filtering operation
        filtered_df = df.filter(combined_filter)

        # Validate results
        if filtered_df.is_empty():
            raise DataQualityError(
                f"No records found matching all conditions: {conditions_dict}",
                "Relax filter criteria or check data values"
            )

        # Save results to analysis directory
        return save_df_to_analysis_dir(run_context, filtered_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Multiple conditions filtering failed: {str(e)}")


def TOP_N(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    column: str,
    n: int,
    ascending: bool = False,
    output_filename: str
) -> Path:
    """
    Select top N records by value using Polars optimized top_k operation.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to select from or file path
        column: Column to sort by
        n: Number of records to select
        ascending: Sort order (False for descending/top values, True for ascending)
        output_filename: Filename to save selected results

    Returns:
        Path: Path to saved DataFrame with top N records

    Raises:
        ValidationError: If inputs are invalid
        DataQualityError: If n is larger than available data

    Example:
        >>> result_path = TOP_N(
        ...     ctx, customers_df,
        ...     column='revenue',
        ...     n=10,
        ...     ascending=False,
        ...     output_filename='top_customers.parquet'
        ... )
    """
    # Load DataFrame if file path provided
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "TOP_N")
    _validate_column_exists(df, column, "TOP_N")

    if not isinstance(n, int) or n <= 0:
        raise ValidationError("n must be a positive integer")

    if n > len(df):
        raise DataQualityError(
            f"Requested {n} records but DataFrame only has {len(df)} rows",
            f"Reduce n to {len(df)} or less"
        )

    try:
        if ascending:
            # For ascending order, we want the smallest values (bottom_k)
            selected_df = df.bottom_k(n, by=column)
        else:
            # For descending order, we want the largest values (top_k)
            selected_df = df.top_k(n, by=column)

        # Save results to analysis directory
        return save_df_to_analysis_dir(run_context, selected_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"TOP_N selection failed: {str(e)}")


def BOTTOM_N(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    column: str,
    n: int,
    output_filename: str
) -> Path:
    """
    Select bottom N records by value using Polars optimized bottom_k operation.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to select from or file path
        column: Column to sort by
        n: Number of records to select
        output_filename: Filename to save selected results

    Returns:
        Path: Path to saved DataFrame with bottom N records

    Raises:
        ValidationError: If inputs are invalid
        DataQualityError: If n is larger than available data

    Example:
        >>> result_path = BOTTOM_N(
        ...     ctx, products_df,
        ...     column='profit_margin',
        ...     n=5,
        ...     output_filename='lowest_margin_products.parquet'
        ... )
    """
    # Load DataFrame if file path provided
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "BOTTOM_N")
    _validate_column_exists(df, column, "BOTTOM_N")

    if not isinstance(n, int) or n <= 0:
        raise ValidationError("n must be a positive integer")

    if n > len(df):
        raise DataQualityError(
            f"Requested {n} records but DataFrame only has {len(df)} rows",
            f"Reduce n to {len(df)} or less"
        )

    try:
        # Use Polars bottom_k for optimal performance
        selected_df = df.bottom_k(n, by=column)

        # Save results to analysis directory
        return save_df_to_analysis_dir(run_context, selected_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"BOTTOM_N selection failed: {str(e)}")


def SAMPLE_DATA(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    n_samples: int,
    random_state: int | None = None,
    output_filename: str
) -> Path:
    """
    Sample random records from DataFrame using Polars optimized sampling.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to sample from or file path
        n_samples: Number of samples to take
        random_state: Random state for reproducibility (optional)
        output_filename: Filename to save sampled results

    Returns:
        Path: Path to saved DataFrame with sampled records

    Raises:
        ValidationError: If inputs are invalid
        DataQualityError: If n_samples is larger than available data

    Example:
        >>> result_path = SAMPLE_DATA(
        ...     ctx, large_dataset_df,
        ...     n_samples=1000,
        ...     random_state=42,
        ...     output_filename='sample_data.parquet'
        ... )
    """
    # Load DataFrame if file path provided
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "SAMPLE_DATA")

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValidationError("n_samples must be a positive integer")

    if n_samples > len(df):
        raise DataQualityError(
            f"Requested {n_samples} samples but DataFrame only has {len(df)} rows",
            f"Reduce n_samples to {len(df)} or less, or use sampling with replacement"
        )

    try:
        # Use Polars sample method for optimal performance
        if random_state is not None:
            sampled_df = df.sample(n=n_samples, seed=random_state)
        else:
            sampled_df = df.sample(n=n_samples)

        # Save results to analysis directory
        return save_df_to_analysis_dir(run_context, sampled_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"Data sampling failed: {str(e)}")
