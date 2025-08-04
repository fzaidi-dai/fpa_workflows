"""
Comparison & Ranking Functions

Functions for comparing values and creating rankings.
All functions use Decimal precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, List, Dict, Union
from pathlib import Path
import polars as pl
import numpy as np
from scipy import stats

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


def _validate_dataframe_input(df: Any, function_name: str) -> pl.DataFrame:
    """
    Standardized input validation for DataFrame data.

    Args:
        df: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.DataFrame: Validated Polars DataFrame

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains issues
    """
    if not isinstance(df, pl.DataFrame):
        raise ValidationError(f"Input must be a Polars DataFrame for {function_name}")

    if df.is_empty():
        raise ValidationError(f"Input DataFrame cannot be empty for {function_name}")

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


def RANK_BY_COLUMN(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    column: str,
    ascending: bool = False,
    method: str = 'average',
    output_filename: str
) -> Path:
    """
    Rank records by column values with financial precision.

    This function is essential for financial analysis tasks such as ranking investments
    by performance, customers by revenue, or products by profitability. The ranking
    provides a clear ordinal position for each record based on the specified metric.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to rank (Polars DataFrame or file path)
        column: Column to rank by
        ascending: Sort order (default False for descending - highest values get rank 1)
        method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal')
        output_filename: Filename to save results as parquet file

    Returns:
        Path: Path to saved results file

    Raises:
        ValidationError: If input is invalid or column doesn't exist
        DataQualityError: If column contains non-numeric values
        ConfigurationError: If method is invalid

    Financial Examples:
        # Rank investment portfolios by annual return
        >>> portfolio_df = pl.DataFrame({
        ...     "portfolio_id": ["A", "B", "C", "D"],
        ...     "annual_return": [0.12, 0.08, 0.15, 0.06]
        ... })
        >>> result_path = RANK_BY_COLUMN(ctx, portfolio_df,
        ...                             column="annual_return",
        ...                             ascending=False,
        ...                             method="dense",
        ...                             output_filename="portfolio_rankings.parquet")

        # Rank customers by total revenue (descending)
        >>> customer_df = pl.DataFrame({
        ...     "customer_id": [1, 2, 3, 4, 5],
        ...     "total_revenue": [50000, 75000, 30000, 90000, 45000]
        ... })
        >>> result_path = RANK_BY_COLUMN(ctx, customer_df,
        ...                             column="total_revenue",
        ...                             ascending=False,
        ...                             method="average",
        ...                             output_filename="customer_rankings.parquet")

    Method Options:
        - 'average': Average rank of tied values
        - 'min': Minimum rank of tied values
        - 'max': Maximum rank of tied values
        - 'dense': Dense ranking (no gaps in rank sequence)
        - 'ordinal': Ordinal ranking (ties broken arbitrarily)
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "RANK_BY_COLUMN")
    _validate_column_exists(df, column, "RANK_BY_COLUMN")

    # Validate method parameter
    valid_methods = ['average', 'min', 'max', 'dense', 'ordinal']
    if method not in valid_methods:
        raise ConfigurationError(f"Invalid ranking method '{method}'. Must be one of: {valid_methods}")

    try:
        # Check if column contains numeric data
        if not df[column].dtype.is_numeric():
            raise DataQualityError(
                f"Column '{column}' must contain numeric values for ranking",
                "Ensure the ranking column contains only numeric data"
            )

        # Check for null values
        if df[column].null_count() > 0:
            raise DataQualityError(
                f"Column '{column}' contains null values",
                "Remove or replace null values before ranking"
            )

        # Core ranking operation using Polars
        result_df = df.with_columns(
            pl.col(column)
            .rank(method=method, descending=not ascending)
            .alias(f"{column}_rank")
        )

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Ranking calculation failed: {str(e)}")


def PERCENTILE_RANK(
    run_context: Any,
    series: Union[pl.Series, List[Union[float, int]], np.ndarray, str, Path],
    *,
    method: str = 'average',
    output_filename: str
) -> Path:
    """
    Calculate percentile rank for each value in financial datasets.

    Percentile rank is crucial in finance for understanding relative performance,
    risk assessment, and benchmarking. It shows what percentage of values fall
    below each observation, providing context for performance evaluation.

    Args:
        run_context: RunContext object for file operations
        series: Series to rank (Polars Series, list, NumPy array, or file path)
        method: Ranking method for ties ('average', 'min', 'max', 'dense', 'ordinal')
        output_filename: Filename to save results as parquet file

    Returns:
        Path: Path to saved results file containing percentile ranks

    Raises:
        ValidationError: If input is empty or invalid type
        DataQualityError: If input contains non-numeric values

    Financial Examples:
        # Calculate percentile ranks for stock returns
        >>> returns = [0.05, 0.12, -0.03, 0.08, 0.15, 0.02, 0.10]
        >>> result_path = PERCENTILE_RANK(ctx, returns,
        ...                              method='average',
        ...                              output_filename="return_percentiles.parquet")
        # Result shows what percentage of returns fall below each value

        # Analyze credit scores relative to portfolio
        >>> credit_scores = [720, 680, 750, 620, 800, 690, 740]
        >>> result_path = PERCENTILE_RANK(ctx, credit_scores,
        ...                              method='dense',
        ...                              output_filename="credit_percentiles.parquet")

        # File input example
        >>> result_path = PERCENTILE_RANK(ctx, "monthly_revenues.parquet",
        ...                              method='average',
        ...                              output_filename="revenue_percentiles.parquet")

    Mathematical Note:
        Percentile rank = (rank - 1) / (n - 1) * 100
        Where rank is the position and n is the total count
        Values range from 0 to 100, representing percentage below each value
    """
    # Handle file path input
    if isinstance(series, (str, Path)):
        df = load_df(run_context, series)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    elif isinstance(series, (list, np.ndarray)):
        series = pl.Series(series)
    elif not isinstance(series, pl.Series):
        raise ValidationError(f"Unsupported input type for PERCENTILE_RANK: {type(series)}")

    # Input validation
    if series.is_empty():
        raise ValidationError("Input series cannot be empty for PERCENTILE_RANK")

    if series.null_count() > 0:
        raise DataQualityError(
            "Input contains null values for PERCENTILE_RANK",
            "Remove or replace null values with appropriate numeric values"
        )

    try:
        # Check if series contains numeric data
        if not series.dtype.is_numeric():
            raise DataQualityError(
                "Input must contain numeric values for percentile ranking",
                "Ensure all values are numeric (int, float, Decimal)"
            )

        # Calculate ranks using Polars
        ranks = series.rank(method=method, descending=False)
        n = len(series)

        # Calculate percentile ranks: (rank - 1) / (n - 1) * 100
        # Handle edge case where n = 1
        if n == 1:
            percentile_ranks = pl.Series([50.0])  # Single value gets 50th percentile
        else:
            percentile_ranks = ((ranks - 1) / (n - 1) * 100)

        # Create result DataFrame
        result_df = pl.DataFrame({
            "value": series,
            "rank": ranks,
            "percentile_rank": percentile_ranks
        })

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"Percentile rank calculation failed: {str(e)}")


def COMPARE_PERIODS(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    value_column: str,
    period_column: str,
    periods_to_compare: List[str],
    output_filename: str
) -> Path:
    """
    Compare financial values between specified periods with variance analysis.

    This function is essential for period-over-period analysis in financial reporting,
    budget variance analysis, and performance tracking. It calculates both absolute
    and percentage changes between periods, providing comprehensive comparison metrics.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame containing period data (Polars DataFrame or file path)
        value_column: Column containing values to compare
        period_column: Column containing period identifiers
        periods_to_compare: List of exactly 2 period identifiers to compare
        output_filename: Filename to save results as parquet file

    Returns:
        Path: Path to saved results file containing period comparison

    Raises:
        ValidationError: If input is invalid or periods don't exist
        DataQualityError: If value column contains non-numeric data
        ConfigurationError: If periods_to_compare doesn't contain exactly 2 periods

    Financial Examples:
        # Compare quarterly revenue year-over-year
        >>> quarterly_data = pl.DataFrame({
        ...     "quarter": ["Q1-2023", "Q2-2023", "Q1-2024", "Q2-2024"],
        ...     "revenue": [1000000, 1200000, 1100000, 1350000],
        ...     "region": ["North", "North", "North", "North"]
        ... })
        >>> result_path = COMPARE_PERIODS(ctx, quarterly_data,
        ...                              value_column="revenue",
        ...                              period_column="quarter",
        ...                              periods_to_compare=["Q1-2023", "Q1-2024"],
        ...                              output_filename="yoy_comparison.parquet")

        # Compare monthly expenses vs budget
        >>> monthly_data = pl.DataFrame({
        ...     "month": ["Jan", "Feb", "Mar"],
        ...     "actual_expense": [50000, 52000, 48000],
        ...     "budget_expense": [51000, 50000, 49000],
        ...     "department": ["Sales", "Sales", "Sales"]
        ... })
        >>> result_path = COMPARE_PERIODS(ctx, monthly_data,
        ...                              value_column="actual_expense",
        ...                              period_column="month",
        ...                              periods_to_compare=["Jan", "Mar"],
        ...                              output_filename="expense_comparison.parquet")

    Output Columns:
        - All original columns from both periods
        - {value_column}_variance: Absolute difference (Period2 - Period1)
        - {value_column}_variance_pct: Percentage change ((Period2 - Period1) / Period1 * 100)
        - comparison_type: Description of the comparison performed
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "COMPARE_PERIODS")
    _validate_column_exists(df, value_column, "COMPARE_PERIODS")
    _validate_column_exists(df, period_column, "COMPARE_PERIODS")

    # Validate periods_to_compare parameter
    if not isinstance(periods_to_compare, list) or len(periods_to_compare) != 2:
        raise ConfigurationError("periods_to_compare must be a list of exactly 2 period identifiers")

    period1, period2 = periods_to_compare

    try:
        # Check if value column contains numeric data
        if not df[value_column].dtype.is_numeric():
            raise DataQualityError(
                f"Column '{value_column}' must contain numeric values for comparison",
                "Ensure the value column contains only numeric data"
            )

        # Check if both periods exist in the data
        available_periods = df[period_column].unique().to_list()
        missing_periods = [p for p in periods_to_compare if p not in available_periods]
        if missing_periods:
            raise ValidationError(
                f"Periods {missing_periods} not found in column '{period_column}'. "
                f"Available periods: {available_periods}"
            )

        # Filter data for the two periods
        period1_data = df.filter(pl.col(period_column) == period1)
        period2_data = df.filter(pl.col(period_column) == period2)

        if period1_data.is_empty():
            raise DataQualityError(
                f"No data found for period '{period1}'",
                "Ensure the period identifier exists in the dataset"
            )

        if period2_data.is_empty():
            raise DataQualityError(
                f"No data found for period '{period2}'",
                "Ensure the period identifier exists in the dataset"
            )

        # Get other columns for joining (exclude period and value columns)
        # Also exclude any numeric columns that might be other value columns
        join_columns = []
        for col in df.columns:
            if col not in [period_column, value_column]:
                # Only include non-numeric columns as join columns (categorical/grouping columns)
                if not df[col].dtype.is_numeric():
                    join_columns.append(col)

        if join_columns:
            # Remove the period column from both datasets before joining
            period1_join_data = period1_data.drop(period_column)
            period2_join_data = period2_data.drop(period_column)

            # Join on other columns to match corresponding records
            comparison_df = period1_join_data.join(
                period2_join_data,
                on=join_columns,
                how="inner",
                suffix="_period2"
            )

            # Rename columns for clarity
            # The original value column becomes period1, the suffixed one is already period2
            comparison_df = comparison_df.rename({
                value_column: f"{value_column}_period1"
            })

            # The period2 column should already have the suffix
            if f"{value_column}_period2" not in comparison_df.columns:
                raise CalculationError(f"Expected column '{value_column}_period2' not found after join")
        else:
            # If no join columns, assume single values per period
            if len(period1_data) > 1 or len(period2_data) > 1:
                raise ValidationError(
                    "Multiple records found per period with no grouping columns. "
                    "Add grouping columns or aggregate data first."
                )

            # Create comparison DataFrame
            comparison_df = pl.DataFrame({
                f"{value_column}_period1": [period1_data[value_column][0]],
                f"{value_column}_period2": [period2_data[value_column][0]],
                "period1": [period1],
                "period2": [period2]
            })

        # Calculate variance metrics using Decimal precision
        period1_values = comparison_df[f"{value_column}_period1"]
        period2_values = comparison_df[f"{value_column}_period2"]

        # Calculate absolute variance (Period2 - Period1)
        variance = period2_values - period1_values

        # Calculate percentage variance ((Period2 - Period1) / Period1 * 100)
        # Handle division by zero
        variance_pct = pl.when(period1_values != 0).\
            then((variance / period1_values) * 100).\
            otherwise(None)

        # Add variance columns to result
        result_df = comparison_df.with_columns([
            variance.alias(f"{value_column}_variance"),
            variance_pct.alias(f"{value_column}_variance_pct"),
            pl.lit(f"{period1} vs {period2}").alias("comparison_type")
        ])

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Period comparison calculation failed: {str(e)}")


def VARIANCE_FROM_TARGET(
    run_context: Any,
    actual_values: Union[pl.Series, List[Union[float, int]], np.ndarray, str, Path],
    *,
    target_values: Union[pl.Series, List[Union[float, int]], np.ndarray, str, Path],
    output_filename: str
) -> Path:
    """
    Calculate variance from target values with financial precision.

    This function is fundamental for budget variance analysis, performance tracking,
    and financial control systems. It calculates both absolute and percentage
    variances, providing comprehensive variance analysis for financial management.

    Args:
        run_context: RunContext object for file operations
        actual_values: Actual values achieved (Series, list, NumPy array, or file path)
        target_values: Target/budget values (Series, list, NumPy array, or file path)
        output_filename: Filename to save results as parquet file

    Returns:
        Path: Path to saved results file containing variance analysis

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        DataQualityError: If inputs contain non-numeric values

    Financial Examples:
        # Budget variance analysis for departments
        >>> actual_expenses = [45000, 52000, 38000, 61000]
        >>> budget_expenses = [50000, 50000, 40000, 60000]
        >>> result_path = VARIANCE_FROM_TARGET(ctx, actual_expenses,
        ...                                   target_values=budget_expenses,
        ...                                   output_filename="budget_variance.parquet")

        # Sales performance vs targets
        >>> actual_sales = [120000, 95000, 110000, 135000]
        >>> target_sales = [100000, 100000, 100000, 100000]
        >>> result_path = VARIANCE_FROM_TARGET(ctx, actual_sales,
        ...                                   target_values=target_sales,
        ...                                   output_filename="sales_variance.parquet")

        # File input example
        >>> result_path = VARIANCE_FROM_TARGET(ctx, "actual_revenue.parquet",
        ...                                   target_values="budget_revenue.parquet",
        ...                                   output_filename="revenue_variance.parquet")

    Output Columns:
        - actual_value: Original actual values
        - target_value: Original target values
        - absolute_variance: Actual - Target (positive = favorable for revenue, negative = unfavorable)
        - variance_percentage: (Actual - Target) / Target * 100
        - variance_type: "Favorable" or "Unfavorable" classification

    Financial Interpretation:
        - Positive variance: Actual > Target (favorable for revenue, unfavorable for costs)
        - Negative variance: Actual < Target (unfavorable for revenue, favorable for costs)
        - Percentage variance: Relative magnitude of variance for comparison across different scales
    """
    # Handle file path inputs
    if isinstance(actual_values, (str, Path)):
        df = load_df(run_context, actual_values)
        actual_series = df[df.columns[0]]
    elif isinstance(actual_values, (list, np.ndarray)):
        actual_series = pl.Series(actual_values)
    elif isinstance(actual_values, pl.Series):
        actual_series = actual_values
    else:
        raise ValidationError(f"Unsupported input type for actual_values: {type(actual_values)}")

    if isinstance(target_values, (str, Path)):
        df = load_df(run_context, target_values)
        target_series = df[df.columns[0]]
    elif isinstance(target_values, (list, np.ndarray)):
        target_series = pl.Series(target_values)
    elif isinstance(target_values, pl.Series):
        target_series = target_values
    else:
        raise ValidationError(f"Unsupported input type for target_values: {type(target_values)}")

    # Input validation
    if actual_series.is_empty() or target_series.is_empty():
        raise ValidationError("Input series cannot be empty for VARIANCE_FROM_TARGET")

    if len(actual_series) != len(target_series):
        raise ValidationError("Actual and target values must have the same length")

    if actual_series.null_count() > 0 or target_series.null_count() > 0:
        raise DataQualityError(
            "Input contains null values for VARIANCE_FROM_TARGET",
            "Remove or replace null values with appropriate numeric values"
        )

    try:
        # Check if series contain numeric data
        if not actual_series.dtype.is_numeric() or not target_series.dtype.is_numeric():
            raise DataQualityError(
                "Input must contain numeric values for variance calculation",
                "Ensure all values are numeric (int, float, Decimal)"
            )

        # Create a temporary DataFrame to perform calculations
        temp_df = pl.DataFrame({
            "actual_value": actual_series,
            "target_value": target_series,
        })

        # Calculate variance metrics using Polars expressions
        result_df = temp_df.with_columns([
            # Calculate absolute variance (Actual - Target)
            (pl.col("actual_value") - pl.col("target_value")).alias("absolute_variance"),

            # Calculate percentage variance ((Actual - Target) / Target * 100)
            # Handle division by zero
            pl.when(pl.col("target_value") != 0)
            .then(((pl.col("actual_value") - pl.col("target_value")) / pl.col("target_value")) * 100)
            .otherwise(None)
            .alias("variance_percentage"),

            # Classify variance type
            pl.when((pl.col("actual_value") - pl.col("target_value")) > 0)
            .then(pl.lit("Above Target"))
            .when((pl.col("actual_value") - pl.col("target_value")) < 0)
            .then(pl.lit("Below Target"))
            .otherwise(pl.lit("On Target"))
            .alias("variance_type")
        ])

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"Variance calculation failed: {str(e)}")


def RANK_CORRELATION(
    run_context: Any,
    series1: Union[pl.Series, List[Union[float, int]], np.ndarray, str, Path],
    *,
    series2: Union[pl.Series, List[Union[float, int]], np.ndarray, str, Path]
) -> float:
    """
    Calculate Spearman rank correlation coefficient between two series.

    Spearman rank correlation is crucial in finance for measuring monotonic relationships
    between variables without assuming linear relationships. It's particularly valuable
    for analyzing relationships between rankings, ordinal data, or non-normally
    distributed financial metrics.

    Args:
        run_context: RunContext object for file operations
        series1: First series for correlation (Series, list, NumPy array, or file path)
        series2: Second series for correlation (Series, list, NumPy array, or file path)

    Returns:
        float: Spearman rank correlation coefficient (-1 to 1)

    Raises:
        ValidationError: If inputs are invalid or mismatched lengths
        DataQualityError: If inputs contain non-numeric values or insufficient data
        CalculationError: If correlation calculation fails

    Financial Examples:
        # Analyze relationship between credit scores and default rates
        >>> credit_scores = [720, 680, 750, 620, 800, 690, 740, 660, 780, 710]
        >>> default_rates = [0.02, 0.08, 0.01, 0.15, 0.005, 0.06, 0.015, 0.12, 0.008, 0.03]
        >>> correlation = RANK_CORRELATION(ctx, credit_scores, series2=default_rates)
        >>> print(f"Credit score vs default rate correlation: {correlation:.3f}")
        # Expected: Strong negative correlation (higher scores, lower default rates)

        # Compare investment performance rankings across different periods
        >>> q1_returns = [0.12, 0.08, 0.15, 0.06, 0.10, 0.14, 0.09, 0.11]
        >>> q2_returns = [0.10, 0.09, 0.13, 0.07, 0.12, 0.11, 0.08, 0.14]
        >>> correlation = RANK_CORRELATION(ctx, q1_returns, series2=q2_returns)
        >>> print(f"Q1 vs Q2 performance correlation: {correlation:.3f}")

        # Analyze relationship between company size and profitability rankings
        >>> revenue_millions = [100, 250, 75, 500, 150, 300, 80, 400]
        >>> profit_margins = [0.15, 0.12, 0.18, 0.10, 0.14, 0.11, 0.16, 0.09]
        >>> correlation = RANK_CORRELATION(ctx, revenue_millions, series2=profit_margins)
        >>> print(f"Size vs profitability correlation: {correlation:.3f}")

        # File input example
        >>> correlation = RANK_CORRELATION(ctx, "portfolio_returns.parquet",
        ...                               series2="benchmark_returns.parquet")

    Mathematical Note:
        Spearman's ρ = 1 - (6 * Σd²) / (n * (n² - 1))
        Where d is the difference between ranks and n is the number of observations

        Interpretation:
        - ρ = 1: Perfect positive monotonic relationship
        - ρ = 0: No monotonic relationship
        - ρ = -1: Perfect negative monotonic relationship
        - |ρ| > 0.7: Strong relationship
        - 0.3 < |ρ| < 0.7: Moderate relationship
        - |ρ| < 0.3: Weak relationship
    """
    # Handle file path inputs
    if isinstance(series1, (str, Path)):
        df = load_df(run_context, series1)
        series1 = df[df.columns[0]]
    elif isinstance(series1, (list, np.ndarray)):
        series1 = pl.Series(series1)
    elif not isinstance(series1, pl.Series):
        raise ValidationError(f"Unsupported input type for series1: {type(series1)}")

    if isinstance(series2, (str, Path)):
        df = load_df(run_context, series2)
        series2 = df[df.columns[0]]
    elif isinstance(series2, (list, np.ndarray)):
        series2 = pl.Series(series2)
    elif not isinstance(series2, pl.Series):
        raise ValidationError(f"Unsupported input type for series2: {type(series2)}")

    # Input validation
    if series1.is_empty() or series2.is_empty():
        raise ValidationError("Input series cannot be empty for RANK_CORRELATION")

    if len(series1) != len(series2):
        raise ValidationError("Both series must have the same length for correlation calculation")

    if len(series1) < 3:
        raise DataQualityError(
            "At least 3 observations required for meaningful correlation calculation",
            "Provide more data points for reliable correlation analysis"
        )

    if series1.null_count() > 0 or series2.null_count() > 0:
        raise DataQualityError(
            "Input contains null values for RANK_CORRELATION",
            "Remove or replace null values with appropriate numeric values"
        )

    try:
        # Check if series contain numeric data
        if not series1.dtype.is_numeric() or not series2.dtype.is_numeric():
            raise DataQualityError(
                "Input must contain numeric values for correlation calculation",
                "Ensure all values are numeric (int, float, Decimal)"
            )

        # Convert to NumPy arrays for SciPy compatibility
        array1 = series1.to_numpy()
        array2 = series2.to_numpy()

        # Check for constant series (no variation)
        if np.var(array1) == 0 or np.var(array2) == 0:
            raise DataQualityError(
                "Cannot calculate correlation for constant series (no variation)",
                "Ensure both series have varying values"
            )

        # Calculate Spearman rank correlation using SciPy
        correlation_result = stats.spearmanr(array1, array2)
        correlation_coefficient = correlation_result.statistic

        # Handle NaN result
        if np.isnan(correlation_coefficient):
            raise CalculationError("Correlation calculation resulted in NaN")

        return float(correlation_coefficient)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, CalculationError)):
            raise
        raise CalculationError(f"Rank correlation calculation failed: {str(e)}")
