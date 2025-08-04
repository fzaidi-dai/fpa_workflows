"""
Data Validation & Quality Functions

These functions ensure data integrity and quality for financial analysis.
All functions use Polars for optimal performance and are optimized for AI agent integration.
"""

from typing import Any, List, Dict, Union
from pathlib import Path
from decimal import Decimal, getcontext
import polars as pl
import numpy as np
from scipy import stats
from datetime import datetime, date
import re

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


def _validate_dataframe_input(data: Any, function_name: str) -> pl.DataFrame:
    """
    Standardized input validation for DataFrame data.

    Args:
        data: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.DataFrame: Validated Polars DataFrame

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains issues
    """
    if not isinstance(data, pl.DataFrame):
        raise ValidationError(f"Input must be a Polars DataFrame for {function_name}, got {type(data)}")

    if data.is_empty():
        raise ValidationError(f"Input DataFrame cannot be empty for {function_name}")

    return data


def _validate_series_input(data: Any, function_name: str) -> pl.Series:
    """
    Standardized input validation for Series data.

    Args:
        data: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.Series: Validated Polars Series

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains issues
    """
    try:
        # Convert to Polars Series for optimal processing
        if isinstance(data, (list, np.ndarray)):
            series = pl.Series(data)
        elif isinstance(data, pl.Series):
            series = data
        elif isinstance(data, pl.DataFrame):
            if len(data.columns) != 1:
                raise ValidationError(f"DataFrame input must have exactly one column for {function_name}")
            series = data[data.columns[0]]
        else:
            raise ValidationError(f"Unsupported input type for {function_name}: {type(data)}")

        # Check if series is empty
        if series.is_empty():
            raise ValidationError(f"Input values cannot be empty for {function_name}")

        return series

    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid data in {function_name}: {str(e)}",
            "Ensure input data is in a valid format"
        )


def CHECK_DUPLICATES(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    columns_to_check: List[str],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Identify duplicate records in dataset using Polars efficient duplicate detection.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to check for duplicates (DataFrame or file path)
        columns_to_check: List of column names to check for duplicates
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with duplicate flags or path to saved file

    Raises:
        ValidationError: If input is invalid or columns don't exist
        DataQualityError: If data contains issues

    Financial Examples:
        # Check for duplicate transaction IDs in financial data
        >>> transactions = pl.DataFrame({
        ...     "transaction_id": ["T001", "T002", "T001", "T003"],
        ...     "amount": [100, 200, 100, 300],
        ...     "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"]
        ... })
        >>> result = CHECK_DUPLICATES(ctx, transactions, columns_to_check=["transaction_id"])
        >>> # Returns DataFrame with is_duplicate column flagging duplicates

        # Check for duplicate customer records
        >>> customers = pl.DataFrame({
        ...     "customer_id": ["C001", "C002", "C001"],
        ...     "name": ["John", "Jane", "John"],
        ...     "email": ["john@email.com", "jane@email.com", "john@email.com"]
        ... })
        >>> result = CHECK_DUPLICATES(ctx, customers, columns_to_check=["customer_id", "email"])
        >>> # Flags records that are duplicates based on both customer_id and email

    Example:
        >>> CHECK_DUPLICATES(ctx, transactions_df, columns_to_check=['transaction_id'])
        >>> CHECK_DUPLICATES(ctx, "transactions.parquet", columns_to_check=['transaction_id', 'amount'])
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "CHECK_DUPLICATES")

    # Validate columns exist
    missing_columns = [col for col in columns_to_check if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Columns not found in DataFrame: {missing_columns}")

    if not columns_to_check:
        raise ValidationError("At least one column must be specified for duplicate checking")

    try:
        # Use Polars efficient duplicate detection
        # Create duplicate flags using is_duplicated on specified columns
        result_df = df.with_columns([
            pl.struct(columns_to_check).is_duplicated().alias("is_duplicate")
        ])

        # Add duplicate count for each group
        duplicate_counts = df.group_by(columns_to_check).agg(
            pl.len().alias("duplicate_count")
        )

        # Join back to get counts
        result_df = result_df.join(
            duplicate_counts,
            on=columns_to_check,
            how="left"
        )

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Duplicate detection failed: {str(e)}")


def VALIDATE_DATES(
    run_context: Any,
    date_series: Union[pl.Series, List[str], str, Path],
    *,
    min_date: str,
    max_date: str,
    output_filename: str | None = None
) -> Union[pl.Series, Path]:
    """
    Validate date formats and ranges using Polars date parsing capabilities.

    Args:
        run_context: RunContext object for file operations
        date_series: Date series to validate (Series, list, or file path)
        min_date: Minimum acceptable date (ISO format: YYYY-MM-DD)
        max_date: Maximum acceptable date (ISO format: YYYY-MM-DD)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.Series or Path: Series with validation flags or path to saved file

    Raises:
        ValidationError: If input is invalid or date bounds are invalid
        DataQualityError: If date parsing fails

    Financial Examples:
        # Validate transaction dates are within fiscal year
        >>> transaction_dates = ["2024-01-15", "2024-06-30", "2023-12-31", "2024-12-31"]
        >>> valid_flags = VALIDATE_DATES(ctx, transaction_dates,
        ...                             min_date="2024-01-01", max_date="2024-12-31")
        >>> # Returns [True, True, False, True] - 2023 date is invalid

        # Validate employee hire dates
        >>> hire_dates = ["2020-03-15", "2025-01-01", "invalid_date", "2022-07-20"]
        >>> valid_flags = VALIDATE_DATES(ctx, hire_dates,
        ...                             min_date="2020-01-01", max_date="2024-12-31")
        >>> # Returns [True, False, False, True] - future date and invalid format flagged

    Example:
        >>> VALIDATE_DATES(ctx, date_column, min_date='2020-01-01', max_date='2025-12-31')
        >>> VALIDATE_DATES(ctx, "dates.parquet", min_date='2020-01-01', max_date='2025-12-31')
    """
    # Handle file path input
    if isinstance(date_series, (str, Path)):
        df = load_df(run_context, date_series)
        # Assume first column contains the date data
        date_series = df[df.columns[0]]

    # Input validation
    if isinstance(date_series, list):
        date_series = pl.Series(date_series)

    date_series = _validate_series_input(date_series, "VALIDATE_DATES")

    # Validate date bounds
    try:
        min_date_parsed = datetime.strptime(min_date, "%Y-%m-%d").date()
        max_date_parsed = datetime.strptime(max_date, "%Y-%m-%d").date()

        if min_date_parsed >= max_date_parsed:
            raise ValidationError("min_date must be less than max_date")

    except ValueError as e:
        raise ValidationError(f"Invalid date format in bounds (use YYYY-MM-DD): {str(e)}")

    try:
        # Convert to string if not already
        if date_series.dtype != pl.Utf8:
            date_series = date_series.cast(pl.Utf8)

        # Parse dates with error handling
        # Use Polars' str.to_date with error handling
        parsed_dates = date_series.str.to_date(format="%Y-%m-%d", strict=False)

        # Create validation flags
        # Valid if: 1) Successfully parsed, 2) Within date range
        is_valid_format = parsed_dates.is_not_null()

        # Check date range for successfully parsed dates
        min_date_pl = pl.lit(min_date_parsed)
        max_date_pl = pl.lit(max_date_parsed)

        is_in_range = parsed_dates.is_between(min_date_pl, max_date_pl, closed="both")

        # Combine validations: valid format AND in range
        validation_flags = is_valid_format & is_in_range.fill_null(False)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            result_df = pl.DataFrame({
                "date_validation_flags": validation_flags
            })
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return validation_flags

    except Exception as e:
        raise DataQualityError(
            f"Date validation failed: {str(e)}",
            "Ensure dates are in YYYY-MM-DD format and within valid ranges"
        )


def CHECK_NUMERIC_RANGE(
    run_context: Any,
    numeric_series: Union[pl.Series, List[Union[int, float]], str, Path],
    *,
    min_value: float,
    max_value: float,
    output_filename: str | None = None
) -> Union[pl.Series, Path]:
    """
    Validate numeric values within expected ranges using Polars efficient range checking.

    Args:
        run_context: RunContext object for file operations
        numeric_series: Numeric series to validate (Series, list, or file path)
        min_value: Minimum acceptable value (inclusive)
        max_value: Maximum acceptable value (inclusive)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.Series or Path: Series with validation flags or path to saved file

    Raises:
        ValidationError: If input is invalid or range bounds are invalid
        DataQualityError: If data contains non-numeric values

    Financial Examples:
        # Validate revenue amounts are positive and reasonable
        >>> revenues = [1000000, 2500000, -50000, 15000000]
        >>> valid_flags = CHECK_NUMERIC_RANGE(ctx, revenues, min_value=0, max_value=10000000)
        >>> # Returns [True, True, False, False] - negative and too large values flagged

        # Validate interest rates are within reasonable bounds
        >>> interest_rates = [0.025, 0.045, 0.15, -0.01, 0.08]
        >>> valid_flags = CHECK_NUMERIC_RANGE(ctx, interest_rates, min_value=0.0, max_value=0.12)
        >>> # Returns [True, True, False, False, True] - negative and too high rates flagged

        # Validate employee ages
        >>> ages = [25, 45, 67, 150, 16]
        >>> valid_flags = CHECK_NUMERIC_RANGE(ctx, ages, min_value=18, max_value=70)
        >>> # Returns [True, True, True, False, False] - unrealistic and underage flagged

    Example:
        >>> CHECK_NUMERIC_RANGE(ctx, revenue_column, min_value=0, max_value=1000000)
        >>> CHECK_NUMERIC_RANGE(ctx, "financial_data.parquet", min_value=0, max_value=1000000)
    """
    # Handle file path input
    if isinstance(numeric_series, (str, Path)):
        df = load_df(run_context, numeric_series)
        # Assume first column contains the numeric data
        numeric_series = df[df.columns[0]]

    # Input validation
    if isinstance(numeric_series, list):
        numeric_series = pl.Series(numeric_series)

    numeric_series = _validate_series_input(numeric_series, "CHECK_NUMERIC_RANGE")

    # Validate range bounds
    if min_value >= max_value:
        raise ValidationError("min_value must be less than max_value")

    try:
        # Ensure series is numeric
        if not numeric_series.dtype.is_numeric():
            # Try to cast to numeric
            try:
                numeric_series = numeric_series.cast(pl.Float64)
            except Exception:
                raise DataQualityError(
                    "Series contains non-numeric values that cannot be converted",
                    "Ensure all values are numeric or can be converted to numeric"
                )

        # Use Polars efficient range checking
        validation_flags = numeric_series.is_between(min_value, max_value, closed="both")

        # Handle null values - they are considered invalid
        validation_flags = validation_flags.fill_null(False)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            result_df = pl.DataFrame({
                "numeric_range_validation_flags": validation_flags
            })
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return validation_flags

    except Exception as e:
        raise DataQualityError(
            f"Numeric range validation failed: {str(e)}",
            "Ensure all values are numeric and within reasonable bounds"
        )


def OUTLIER_DETECTION(
    run_context: Any,
    numeric_series: Union[pl.Series, List[Union[int, float]], str, Path],
    *,
    method: str,
    threshold: float,
    output_filename: str | None = None
) -> Union[pl.Series, Path]:
    """
    Detect statistical outliers using IQR or z-score methods with SciPy integration.

    Args:
        run_context: RunContext object for file operations
        numeric_series: Numeric series to analyze (Series, list, or file path)
        method: Detection method ('iqr' or 'z-score')
        threshold: Detection threshold (1.5 for IQR, 2-3 for z-score typically)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.Series or Path: Series with outlier flags or path to saved file

    Raises:
        ValidationError: If input is invalid or method is unsupported
        CalculationError: If statistical calculation fails
        DataQualityError: If data contains insufficient valid values

    Financial Examples:
        # Detect outlier transactions using IQR method
        >>> transaction_amounts = [100, 150, 200, 175, 10000, 125, 180, 50000]
        >>> outlier_flags = OUTLIER_DETECTION(ctx, transaction_amounts, method='iqr', threshold=1.5)
        >>> # Returns [False, False, False, False, True, False, False, True] - large amounts flagged

        # Detect outlier stock returns using z-score
        >>> daily_returns = [0.02, -0.01, 0.03, 0.15, -0.02, 0.01, -0.25, 0.04]
        >>> outlier_flags = OUTLIER_DETECTION(ctx, daily_returns, method='z-score', threshold=2.0)
        >>> # Returns flags for returns more than 2 standard deviations from mean

        # Detect outlier revenue figures
        >>> monthly_revenues = [1000000, 1100000, 950000, 5000000, 1050000, 980000]
        >>> outlier_flags = OUTLIER_DETECTION(ctx, monthly_revenues, method='iqr', threshold=1.5)
        >>> # Flags the 5M revenue as an outlier

    Example:
        >>> OUTLIER_DETECTION(ctx, sales_data, method='iqr', threshold=1.5)
        >>> OUTLIER_DETECTION(ctx, "returns.parquet", method='z-score', threshold=2.5)
    """
    # Handle file path input
    if isinstance(numeric_series, (str, Path)):
        df = load_df(run_context, numeric_series)
        # Assume first column contains the numeric data
        numeric_series = df[df.columns[0]]

    # Input validation
    if isinstance(numeric_series, list):
        numeric_series = pl.Series(numeric_series)

    numeric_series = _validate_series_input(numeric_series, "OUTLIER_DETECTION")

    # Validate method
    if method not in ['iqr', 'z-score']:
        raise ValidationError("Method must be 'iqr' or 'z-score'")

    # Validate threshold
    if threshold <= 0:
        raise ValidationError("Threshold must be positive")

    try:
        # Ensure series is numeric and remove nulls for calculation
        if not numeric_series.dtype.is_numeric():
            try:
                numeric_series = numeric_series.cast(pl.Float64)
            except Exception:
                raise DataQualityError(
                    "Series contains non-numeric values that cannot be converted",
                    "Ensure all values are numeric"
                )

        # Remove null values for statistical calculations
        clean_series = numeric_series.drop_nulls()

        if len(clean_series) < 3:
            raise DataQualityError(
                "Insufficient valid data points for outlier detection (need at least 3)",
                "Provide more data points or clean the dataset"
            )

        # Convert to numpy for SciPy calculations
        values = clean_series.to_numpy()

        if method == 'iqr':
            # IQR method using SciPy
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            # Create outlier flags for original series (including nulls)
            outlier_flags = (numeric_series < lower_bound) | (numeric_series > upper_bound)

        elif method == 'z-score':
            # Z-score method using SciPy
            z_scores = np.abs(stats.zscore(values, nan_policy='omit'))

            # Map z-scores back to original series
            # Create a mapping from clean values to z-scores
            value_to_zscore = dict(zip(values, z_scores))

            # Apply z-score threshold to original series
            outlier_flags = numeric_series.map_elements(
                lambda x: value_to_zscore.get(x, 0) > threshold if x is not None else False,
                return_dtype=pl.Boolean
            )

        # Handle null values - they are not considered outliers
        outlier_flags = outlier_flags.fill_null(False)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            result_df = pl.DataFrame({
                "outlier_flags": outlier_flags
            })
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return outlier_flags

    except DataQualityError:
        # Re-raise DataQualityError as-is
        raise
    except Exception as e:
        raise CalculationError(f"Outlier detection failed: {str(e)}")


def COMPLETENESS_CHECK(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path]
) -> Dict[str, float]:
    """
    Check data completeness by column using Polars efficient null counting.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to check for completeness (DataFrame or file path)

    Returns:
        Dict[str, float]: Dictionary with column names and completeness percentages (0-100)

    Raises:
        ValidationError: If input is invalid
        CalculationError: If completeness calculation fails

    Financial Examples:
        # Check completeness of customer data
        >>> customer_df = pl.DataFrame({
        ...     "customer_id": ["C001", "C002", "C003", "C004"],
        ...     "name": ["John", "Jane", None, "Bob"],
        ...     "email": ["john@email.com", None, None, "bob@email.com"],
        ...     "revenue": [1000, 2000, 1500, None]
        ... })
        >>> completeness = COMPLETENESS_CHECK(ctx, customer_df)
        >>> # Returns: {"customer_id": 100.0, "name": 75.0, "email": 50.0, "revenue": 75.0}

        # Check completeness of financial statements
        >>> financial_df = pl.DataFrame({
        ...     "period": ["Q1", "Q2", "Q3", "Q4"],
        ...     "revenue": [1000000, 1100000, None, 1200000],
        ...     "expenses": [800000, 900000, 850000, 950000],
        ...     "profit": [200000, 200000, None, 250000]
        ... })
        >>> completeness = COMPLETENESS_CHECK(ctx, financial_df)
        >>> # Returns completeness percentage for each financial metric

    Example:
        >>> COMPLETENESS_CHECK(ctx, financial_data_df)
        >>> COMPLETENESS_CHECK(ctx, "customer_data.parquet")
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "COMPLETENESS_CHECK")

    try:
        # Use Polars efficient null counting
        total_rows = len(df)

        if total_rows == 0:
            raise CalculationError("Cannot calculate completeness for empty DataFrame")

        completeness_dict = {}

        # Calculate completeness for each column
        for column in df.columns:
            null_count = df[column].null_count()
            non_null_count = total_rows - null_count
            completeness_percentage = (non_null_count / total_rows) * 100.0
            completeness_dict[column] = round(completeness_percentage, 2)

        return completeness_dict

    except Exception as e:
        raise CalculationError(f"Completeness check failed: {str(e)}")


def CONSISTENCY_CHECK(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    consistency_rules: Dict[str, List[str]],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Check data consistency across related fields using configurable business rules.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to check for consistency (DataFrame or file path)
        consistency_rules: Rules for consistency checking (e.g., {'total': ['subtotal', 'tax']})
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with consistency flags or path to saved file

    Raises:
        ValidationError: If input is invalid or rules reference non-existent columns
        CalculationError: If consistency calculation fails

    Financial Examples:
        # Check if total equals sum of components
        >>> invoice_df = pl.DataFrame({
        ...     "subtotal": [100.0, 200.0, 150.0],
        ...     "tax": [10.0, 20.0, 15.0],
        ...     "total": [110.0, 220.0, 160.0]  # Last one is inconsistent
        ... })
        >>> rules = {"total": ["subtotal", "tax"]}
        >>> result = CONSISTENCY_CHECK(ctx, invoice_df, consistency_rules=rules)
        >>> # Returns DataFrame with is_consistent_total column

        # Check budget vs actual consistency
        >>> budget_df = pl.DataFrame({
        ...     "budget_revenue": [1000000, 1100000, 1200000],
        ...     "actual_revenue": [950000, 1150000, 1180000],
        ...     "variance": [50000, -50000, 20000]  # Should be budget - actual
        ... })
        >>> rules = {"variance": ["budget_revenue", "actual_revenue"]}  # variance = budget - actual
        >>> result = CONSISTENCY_CHECK(ctx, budget_df, consistency_rules=rules)

        # Check balance sheet equation: Assets = Liabilities + Equity
        >>> balance_sheet = pl.DataFrame({
        ...     "assets": [1000000, 1500000, 2000000],
        ...     "liabilities": [600000, 900000, 1200000],
        ...     "equity": [400000, 600000, 800000]
        ... })
        >>> rules = {"assets": ["liabilities", "equity"]}
        >>> result = CONSISTENCY_CHECK(ctx, balance_sheet, consistency_rules=rules)

    Example:
        >>> CONSISTENCY_CHECK(ctx, df, consistency_rules={'total': ['subtotal', 'tax']})
        >>> CONSISTENCY_CHECK(ctx, "invoices.parquet", consistency_rules={'total': ['subtotal', 'tax']})
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "CONSISTENCY_CHECK")

    if not consistency_rules:
        raise ValidationError("At least one consistency rule must be provided")

    # Validate that all referenced columns exist
    all_referenced_columns = set()
    for target_col, component_cols in consistency_rules.items():
        all_referenced_columns.add(target_col)
        all_referenced_columns.update(component_cols)

    missing_columns = [col for col in all_referenced_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Columns referenced in rules not found in DataFrame: {missing_columns}")

    try:
        result_df = df.clone()

        # Apply each consistency rule
        for target_column, component_columns in consistency_rules.items():
            # Validate component columns are numeric
            for col in component_columns + [target_column]:
                if not df[col].dtype.is_numeric():
                    try:
                        df = df.with_columns(pl.col(col).cast(pl.Float64))
                    except Exception:
                        raise DataQualityError(
                            f"Column '{col}' contains non-numeric values that cannot be converted",
                            f"Ensure column '{col}' contains only numeric values"
                        )

            # Calculate expected value (sum of components)
            expected_expr = pl.lit(0.0)
            for component_col in component_columns:
                expected_expr = expected_expr + pl.col(component_col).fill_null(0)

            # Check consistency with tolerance for floating point precision
            tolerance = 1e-10
            actual_value = pl.col(target_column).fill_null(0)

            consistency_flag = (
                (actual_value - expected_expr).abs() <= tolerance
            ).alias(f"is_consistent_{target_column}")

            result_df = result_df.with_columns(consistency_flag)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Consistency check failed: {str(e)}")
