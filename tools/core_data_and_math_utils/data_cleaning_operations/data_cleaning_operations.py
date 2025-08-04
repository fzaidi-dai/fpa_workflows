"""
Data Cleaning Operations Functions

Functions for cleaning and standardizing financial data.
All functions use Polars for high-performance data processing and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, List, Dict, Union
from pathlib import Path
import polars as pl
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


def _validate_series_input(data: Any, function_name: str) -> pl.Series:
    """
    Standardized input validation for series data.

    Args:
        data: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.Series: Validated Polars Series

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid values
    """
    try:
        # Convert to Polars Series for optimal processing
        if isinstance(data, (list, tuple)):
            series = pl.Series(data)
        elif isinstance(data, pl.Series):
            series = data
        else:
            raise ValidationError(f"Unsupported input type for {function_name}: {type(data)}")

        # Check if series is empty
        if series.is_empty():
            raise ValidationError(f"Input data cannot be empty for {function_name}")

        return series

    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid data in {function_name}: {str(e)}",
            "Ensure all values are valid for the operation"
        )


def STANDARDIZE_CURRENCY(
    run_context: Any,
    currency_series: Union[List[str], pl.Series, str, Path],
    *,
    target_format: str,
    output_filename: str
) -> Path:
    """
    Standardize currency formats for financial data consistency.

    This function converts various currency representations into a standardized format,
    essential for financial reporting, analysis, and regulatory compliance.

    Args:
        run_context: RunContext object for file operations
        currency_series: Series of currency values in various formats (List, Polars Series, or file path)
        target_format: Target currency format (e.g., 'USD', 'EUR', 'GBP', 'JPY')
        output_filename: Filename to save standardized currency results

    Returns:
        Path: Path to the saved parquet file containing standardized currency values

    Raises:
        ValidationError: If input is empty or target_format is invalid
        DataQualityError: If currency values cannot be parsed
        ConfigurationError: If target_format is not supported

    Financial Examples:
        # Standardize mixed currency formats to USD
        >>> mixed_currencies = ["$1,234.56", "USD 1234.56", "1234.56 USD", "1,234.56"]
        >>> result_path = STANDARDIZE_CURRENCY(ctx, mixed_currencies, target_format="USD", output_filename="std_currency.parquet")

        # Standardize European currencies to EUR
        >>> eur_currencies = ["€1.234,56", "EUR 1234.56", "1234.56 EUR"]
        >>> result_path = STANDARDIZE_CURRENCY(ctx, eur_currencies, target_format="EUR", output_filename="eur_std.parquet")

        # File input example
        >>> result_path = STANDARDIZE_CURRENCY(ctx, "currency_data.csv", target_format="USD", output_filename="standardized.parquet")

    Business Use Cases:
        - Multi-currency financial reporting consolidation
        - Regulatory compliance for international transactions
        - Data preparation for financial analysis and modeling
        - Standardization for accounting system integration
    """
    # Handle file path input
    if isinstance(currency_series, (str, Path)):
        df = load_df(run_context, currency_series)
        # Assume first column contains the currency data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_series_input(currency_series, "STANDARDIZE_CURRENCY")

    # Validate target format
    supported_formats = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY']
    if target_format not in supported_formats:
        raise ConfigurationError(f"Unsupported target format: {target_format}. Supported: {supported_formats}")

    try:
        # Currency symbol mapping
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CAD': 'C$',
            'AUD': 'A$',
            'CHF': 'CHF',
            'CNY': '¥'
        }

        # Core cleaning and standardization logic
        df_work = pl.DataFrame({"currency": series})

        # Step 1: Extract numeric value from various currency formats
        # Remove common currency symbols and codes
        df_work = df_work.with_columns([
            pl.col("currency")
            .str.replace_all(r'[€$£¥]', '')  # Remove currency symbols
            .str.replace_all(r'\b(USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY)\b', '')  # Remove currency codes
            .str.replace_all(r'[,\s]', '')  # Remove commas and spaces
            .str.strip_chars()  # Remove leading/trailing whitespace
            .alias("numeric_value")
        ])

        # Step 2: Validate and convert to decimal
        df_work = df_work.with_columns([
            pl.col("numeric_value")
            .cast(pl.Float64, strict=False)
            .alias("amount")
        ])

        # Check for parsing failures
        null_count = df_work.filter(pl.col("amount").is_null()).height
        if null_count > 0:
            raise DataQualityError(
                f"Failed to parse {null_count} currency values",
                "Ensure all currency values contain valid numeric amounts"
            )

        # Step 3: Format according to target currency
        target_symbol = currency_symbols[target_format]

        if target_format in ['USD', 'GBP', 'CAD', 'AUD']:
            # Format: $1,234.56
            df_work = df_work.with_columns([
                pl.col("amount")
                .map_elements(lambda x: f"{target_symbol}{x:,.2f}", return_dtype=pl.Utf8)
                .alias("standardized_currency")
            ])
        elif target_format == 'EUR':
            # Format: €1.234,56 (European format)
            df_work = df_work.with_columns([
                pl.col("amount")
                .map_elements(lambda x: f"€{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'), return_dtype=pl.Utf8)
                .alias("standardized_currency")
            ])
        elif target_format in ['JPY', 'CNY']:
            # Format: ¥1,234 (no decimal places for yen)
            df_work = df_work.with_columns([
                pl.col("amount")
                .map_elements(lambda x: f"{target_symbol}{int(x):,}", return_dtype=pl.Utf8)
                .alias("standardized_currency")
            ])
        else:  # CHF
            # Format: CHF 1,234.56
            df_work = df_work.with_columns([
                pl.col("amount")
                .map_elements(lambda x: f"CHF {x:,.2f}", return_dtype=pl.Utf8)
                .alias("standardized_currency")
            ])

        # Create result DataFrame
        result_df = df_work.select("standardized_currency")

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Currency standardization failed: {str(e)}")


def CLEAN_NUMERIC(
    run_context: Any,
    mixed_series: Union[List[str], pl.Series, str, Path],
    *,
    output_filename: str
) -> Path:
    """
    Clean numeric data by removing non-numeric characters and converting to proper numeric format.

    Essential for preparing financial data that contains mixed formatting, currency symbols,
    or other non-numeric characters that prevent mathematical operations.

    Args:
        run_context: RunContext object for file operations
        mixed_series: Series with mixed data containing numbers and non-numeric characters (List, Polars Series, or file path)
        output_filename: Filename to save cleaned numeric results

    Returns:
        Path: Path to the saved parquet file containing cleaned numeric values

    Raises:
        ValidationError: If input is empty or invalid
        DataQualityError: If values cannot be converted to numeric format

    Financial Examples:
        # Clean financial data with currency symbols and formatting
        >>> mixed_data = ["$1,234.56", "€987.65", "¥1000", "(500.00)", "2,345.67%"]
        >>> result_path = CLEAN_NUMERIC(ctx, mixed_data, output_filename="clean_numbers.parquet")

        # Clean accounting data with parentheses for negatives
        >>> accounting_data = ["1,234.56", "(567.89)", "2,345.67", "(1,000.00)"]
        >>> result_path = CLEAN_NUMERIC(ctx, accounting_data, output_filename="clean_accounting.parquet")

        # File input example
        >>> result_path = CLEAN_NUMERIC(ctx, "mixed_financial_data.csv", output_filename="cleaned.parquet")

    Business Use Cases:
        - Preparing imported financial data for analysis
        - Cleaning data from various accounting systems
        - Standardizing numeric formats across data sources
        - Converting text-based financial reports to numeric format
    """
    # Handle file path input
    if isinstance(mixed_series, (str, Path)):
        df = load_df(run_context, mixed_series)
        # Assume first column contains the mixed data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_series_input(mixed_series, "CLEAN_NUMERIC")

    try:
        # Core cleaning logic
        df_work = pl.DataFrame({"mixed_data": series})

        # Step 1: Handle accounting format negatives (parentheses)
        df_work = df_work.with_columns([
            pl.col("mixed_data")
            .str.replace_all(r'^\((.*)\)$', r'-$1')  # Convert (123.45) to -123.45
            .alias("step1")
        ])

        # Step 2: Remove currency symbols and non-numeric characters
        df_work = df_work.with_columns([
            pl.col("step1")
            .str.replace_all(r'[€$£¥%,\s]', '')  # Remove currency symbols, %, commas, spaces
            .str.replace_all(r'\b(USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY)\b', '')  # Remove currency codes
            .str.strip_chars()  # Remove leading/trailing whitespace
            .alias("step2")
        ])

        # Step 3: Handle empty strings and convert to numeric
        df_work = df_work.with_columns([
            pl.when(pl.col("step2") == "")
            .then(None)
            .otherwise(pl.col("step2"))
            .alias("step3")
        ])

        # Step 4: Convert to decimal with proper validation
        df_work = df_work.with_columns([
            pl.col("step3")
            .cast(pl.Float64, strict=False)
            .alias("cleaned_numeric")
        ])

        # Check for parsing failures (excluding intentional nulls from empty strings)
        original_non_empty = df_work.filter(pl.col("mixed_data").str.strip_chars() != "").height
        successful_conversions = df_work.filter(pl.col("cleaned_numeric").is_not_null()).height

        if successful_conversions == 0 and original_non_empty > 0:
            raise DataQualityError(
                "No values could be converted to numeric format",
                "Ensure input contains valid numeric data with or without formatting characters"
            )

        # Create result DataFrame with high precision for financial accuracy
        result_df = df_work.select("cleaned_numeric")

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"Numeric cleaning failed: {str(e)}")


def NORMALIZE_NAMES(
    run_context: Any,
    name_series: Union[List[str], pl.Series, str, Path],
    *,
    normalization_rules: Dict[str, str],
    output_filename: str
) -> Path:
    """
    Normalize company/customer names for consistent identification and reporting.

    Critical for financial data integrity, customer relationship management,
    and regulatory reporting where entity names must be standardized.

    Args:
        run_context: RunContext object for file operations
        name_series: Series of names to normalize (List, Polars Series, or file path)
        normalization_rules: Dictionary mapping variations to standard forms
        output_filename: Filename to save normalized names

    Returns:
        Path: Path to the saved parquet file containing normalized names

    Raises:
        ValidationError: If input is empty or normalization_rules is invalid
        DataQualityError: If names cannot be processed

    Financial Examples:
        # Standardize company names for financial reporting
        >>> company_names = ["Apple Inc.", "Apple Incorporated", "APPLE INC", "apple inc."]
        >>> rules = {"incorporated": "Inc.", "corporation": "Corp.", "company": "Co."}
        >>> result_path = NORMALIZE_NAMES(ctx, company_names, normalization_rules=rules, output_filename="std_companies.parquet")

        # Normalize customer names for CRM integration
        >>> customer_names = ["John Smith Jr.", "JOHN SMITH JR", "john smith junior"]
        >>> rules = {"junior": "Jr.", "senior": "Sr.", "incorporated": "Inc."}
        >>> result_path = NORMALIZE_NAMES(ctx, customer_names, normalization_rules=rules, output_filename="std_customers.parquet")

        # File input example
        >>> result_path = NORMALIZE_NAMES(ctx, "entity_names.csv", normalization_rules=rules, output_filename="normalized.parquet")

    Business Use Cases:
        - Customer data deduplication and master data management
        - Regulatory reporting with standardized entity names
        - Financial consolidation across subsidiaries
        - Vendor management and procurement standardization
    """
    # Handle file path input
    if isinstance(name_series, (str, Path)):
        df = load_df(run_context, name_series)
        # Assume first column contains the name data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_series_input(name_series, "NORMALIZE_NAMES")

    # Validate normalization rules
    if not isinstance(normalization_rules, dict):
        raise ValidationError("Normalization rules must be a dictionary")

    if not normalization_rules:
        raise ValidationError("Normalization rules cannot be empty")

    try:
        # Core normalization logic
        df_work = pl.DataFrame({"original_name": series})

        # Step 1: Basic cleaning - trim whitespace and normalize case
        df_work = df_work.with_columns([
            pl.col("original_name")
            .str.strip_chars()
            .str.to_titlecase()  # Convert to Title Case
            .alias("step1")
        ])

        # Step 2: Apply normalization rules
        normalized_col = pl.col("step1")

        for pattern, replacement in normalization_rules.items():
            # Case-insensitive replacement
            normalized_col = normalized_col.str.replace_all(
                f"(?i)\\b{re.escape(pattern)}\\b",
                replacement
            )

        df_work = df_work.with_columns([
            normalized_col.alias("step2")
        ])

        # Step 3: Additional common business name standardizations
        df_work = df_work.with_columns([
            pl.col("step2")
            # Standardize common business suffixes
            .str.replace_all(r'\bInc\b\.?', 'Inc.')
            .str.replace_all(r'\bCorp\b\.?', 'Corp.')
            .str.replace_all(r'\bLtd\b\.?', 'Ltd.')
            .str.replace_all(r'\bLlc\b\.?', 'LLC')
            .str.replace_all(r'\bCo\b\.?', 'Co.')
            # Remove extra spaces
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .alias("normalized_name")
        ])

        # Create result DataFrame
        result_df = df_work.select("normalized_name")

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"Name normalization failed: {str(e)}")


def REMOVE_DUPLICATES(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    subset_columns: List[str],
    keep_method: str,
    output_filename: str
) -> Path:
    """
    Remove duplicate records with configurable options for financial data integrity.

    Essential for maintaining data quality in financial systems, preventing
    double-counting in reports, and ensuring accurate financial analysis.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to process (Polars DataFrame or file path)
        subset_columns: List of column names to check for duplicates
        keep_method: Method to keep records ('first', 'last', 'none')
        output_filename: Filename to save deduplicated results

    Returns:
        Path: Path to the saved parquet file containing deduplicated data

    Raises:
        ValidationError: If input parameters are invalid
        ConfigurationError: If keep_method or subset_columns are invalid
        DataQualityError: If DataFrame processing fails

    Financial Examples:
        # Remove duplicate transactions
        >>> transactions = pl.DataFrame({
        ...     "transaction_id": ["T001", "T002", "T001", "T003"],
        ...     "amount": [100.0, 200.0, 100.0, 300.0],
        ...     "date": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-03"]
        ... })
        >>> result_path = REMOVE_DUPLICATES(ctx, transactions, subset_columns=["transaction_id"], keep_method="first", output_filename="unique_transactions.parquet")

        # Remove duplicate customer records based on multiple columns
        >>> customers = pl.DataFrame({
        ...     "customer_id": ["C001", "C002", "C001", "C003"],
        ...     "name": ["John Doe", "Jane Smith", "John Doe", "Bob Johnson"],
        ...     "email": ["john@email.com", "jane@email.com", "john@email.com", "bob@email.com"]
        ... })
        >>> result_path = REMOVE_DUPLICATES(ctx, customers, subset_columns=["customer_id", "email"], keep_method="last", output_filename="unique_customers.parquet")

        # File input example
        >>> result_path = REMOVE_DUPLICATES(ctx, "financial_data.csv", subset_columns=["account_id", "date"], keep_method="first", output_filename="deduplicated.parquet")

    Business Use Cases:
        - Transaction deduplication in payment processing
        - Customer master data management
        - Financial report accuracy and compliance
        - Data warehouse ETL processes
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        dataframe = load_df(run_context, df)
    elif isinstance(df, pl.DataFrame):
        dataframe = df
    else:
        raise ValidationError(f"Unsupported DataFrame type: {type(df)}")

    # Validate input parameters
    if dataframe.is_empty():
        raise ValidationError("Input DataFrame cannot be empty")

    if not isinstance(subset_columns, list) or not subset_columns:
        raise ValidationError("subset_columns must be a non-empty list")

    # Validate subset columns exist in DataFrame
    missing_columns = [col for col in subset_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValidationError(f"Columns not found in DataFrame: {missing_columns}")

    # Validate keep_method
    valid_methods = ['first', 'last', 'none']
    if keep_method not in valid_methods:
        raise ConfigurationError(f"Invalid keep_method: {keep_method}. Valid options: {valid_methods}")

    try:
        # Core deduplication logic
        if keep_method == 'none':
            # Remove all duplicates, keep none
            # First identify duplicates
            duplicate_mask = dataframe.is_duplicated(subset=subset_columns)
            result_df = dataframe.filter(~duplicate_mask)
        else:
            # Use Polars unique method with keep parameter
            keep_param = keep_method  # 'first' or 'last'
            result_df = dataframe.unique(subset=subset_columns, keep=keep_param)

        # Validate result
        if result_df.is_empty() and not dataframe.is_empty():
            if keep_method == 'none':
                # This is expected if all rows were duplicates
                pass
            else:
                raise DataQualityError(
                    "Deduplication resulted in empty DataFrame",
                    "Check if subset_columns contain valid data for deduplication"
                )

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, ConfigurationError, DataQualityError)):
            raise
        raise CalculationError(f"Duplicate removal failed: {str(e)}")


def STANDARDIZE_DATES(
    run_context: Any,
    date_series: Union[List[str], pl.Series, str, Path],
    *,
    target_format: str,
    output_filename: str
) -> Path:
    """
    Convert various date formats to a standardized format for consistent financial reporting.

    Critical for financial data integrity, time-series analysis, and regulatory
    compliance where consistent date formatting is required.

    Args:
        run_context: RunContext object for file operations
        date_series: Series of dates in various formats (List, Polars Series, or file path)
        target_format: Target date format string (e.g., '%Y-%m-%d', '%m/%d/%Y')
        output_filename: Filename to save standardized dates

    Returns:
        Path: Path to the saved parquet file containing standardized dates

    Raises:
        ValidationError: If input is empty or target_format is invalid
        DataQualityError: If dates cannot be parsed
        ConfigurationError: If target_format is malformed

    Financial Examples:
        # Standardize mixed date formats for financial reporting
        >>> mixed_dates = ["01/15/2023", "2023-01-15", "15-Jan-2023", "January 15, 2023"]
        >>> result_path = STANDARDIZE_DATES(ctx, mixed_dates, target_format="%Y-%m-%d", output_filename="std_dates.parquet")

        # Convert to US format for regulatory reporting
        >>> iso_dates = ["2023-01-15", "2023-02-20", "2023-03-25"]
        >>> result_path = STANDARDIZE_DATES(ctx, iso_dates, target_format="%m/%d/%Y", output_filename="us_dates.parquet")

        # File input example
        >>> result_path = STANDARDIZE_DATES(ctx, "transaction_dates.csv", target_format="%Y-%m-%d", output_filename="standardized.parquet")

    Business Use Cases:
        - Financial report standardization across systems
        - Regulatory compliance for date formatting requirements
        - Data warehouse ETL date normalization
        - Cross-border financial data integration
    """
    # Handle file path input
    if isinstance(date_series, (str, Path)):
        df = load_df(run_context, date_series)
        # Assume first column contains the date data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        series = _validate_series_input(date_series, "STANDARDIZE_DATES")

    # Validate target format
    if not isinstance(target_format, str) or not target_format.strip():
        raise ValidationError("target_format must be a non-empty string")

    # Test target format validity
    try:
        from datetime import datetime
        test_date = datetime(2023, 1, 15)
        test_date.strftime(target_format)
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"Invalid target_format: {target_format}. Error: {str(e)}")

    try:
        # Core date standardization logic
        df_work = pl.DataFrame({"date_string": series})

        # Common date format patterns to try
        date_patterns = [
            "%Y-%m-%d",           # 2023-01-15
            "%m/%d/%Y",           # 01/15/2023
            "%d/%m/%Y",           # 15/01/2023
            "%Y/%m/%d",           # 2023/01/15
            "%d-%m-%Y",           # 15-01-2023
            "%d-%b-%Y",           # 15-Jan-2023
            "%B %d, %Y",          # January 15, 2023
            "%b %d, %Y",          # Jan 15, 2023
            "%d %B %Y",           # 15 January 2023
            "%d %b %Y",           # 15 Jan 2023
            "%Y%m%d",             # 20230115
            "%m-%d-%Y",           # 01-15-2023
            "%Y.%m.%d",           # 2023.01.15
            "%d.%m.%Y",           # 15.01.2023
        ]

        # Try to parse dates with different patterns
        parsed_dates = None
        successful_pattern = None

        for pattern in date_patterns:
            try:
                # Attempt to parse with current pattern
                test_parsed = df_work.with_columns([
                    pl.col("date_string").str.to_date(pattern, strict=False).alias("parsed_date")
                ])

                # Check how many dates were successfully parsed
                success_count = test_parsed.filter(pl.col("parsed_date").is_not_null()).height

                if success_count > 0:
                    # If this pattern works for some dates, use it
                    if parsed_dates is None or success_count > parsed_dates.filter(pl.col("parsed_date").is_not_null()).height:
                        parsed_dates = test_parsed
                        successful_pattern = pattern

                        # If all dates parsed successfully, we're done
                        if success_count == df_work.height:
                            break

            except Exception:
                # Pattern didn't work, try next one
                continue

        if parsed_dates is None:
            raise DataQualityError(
                "No date format pattern could parse the input dates",
                f"Ensure dates are in recognizable formats. Tried patterns: {date_patterns}"
            )

        # Check parsing success rate - allow some failures but not complete failure
        null_count = parsed_dates.filter(pl.col("parsed_date").is_null()).height
        success_rate = (df_work.height - null_count) / df_work.height

        if success_rate < 0.5:  # Less than 50% success rate
            raise DataQualityError(
                f"Failed to parse {null_count} out of {df_work.height} dates using pattern {successful_pattern}",
                "Ensure most dates are in consistent, recognizable formats"
            )

        # Convert to target format, handling nulls gracefully
        result_df = parsed_dates.with_columns([
            pl.when(pl.col("parsed_date").is_not_null())
            .then(pl.col("parsed_date").dt.to_string(target_format))
            .otherwise(None)
            .alias("standardized_date")
        ]).select("standardized_date")

        # Save results to file
        return save_df_to_analysis_dir(run_context, result_df, output_filename)

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Date standardization failed: {str(e)}")
