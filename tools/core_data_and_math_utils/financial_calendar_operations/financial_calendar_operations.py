"""
Financial Calendar Operations Functions

Functions for handling financial calendars and periods.
All functions use proper financial date handling and are optimized for AI agent integration.
"""

from typing import Any, List, Union
import datetime
from pathlib import Path
import polars as pl
from tools.tool_exceptions import (
    FPABaseException,
    RetryAfterCorrectionError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)
from tools.toolset_utils import load_df


def _parse_date_input(date_input: Union[datetime.date, str]) -> datetime.date:
    """
    Parse date input into datetime.date object.

    Args:
        date_input: Date as datetime.date or string

    Returns:
        datetime.date: Parsed date

    Raises:
        ValidationError: If date cannot be parsed
    """
    if isinstance(date_input, datetime.date):
        return date_input
    elif isinstance(date_input, str):
        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    return datetime.datetime.strptime(date_input, fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date string: {date_input}")
        except ValueError as e:
            raise ValidationError(f"Invalid date format: {str(e)}")
    else:
        raise ValidationError(f"Date must be datetime.date or string, got {type(date_input)}")


def FISCAL_YEAR(run_context: Any, date: Union[datetime.date, str], *, fiscal_year_start_month: int) -> int:
    """
    Convert calendar date to fiscal year.

    A fiscal year is a 12-month period used for financial reporting that may not align
    with the calendar year. This function determines which fiscal year a given date falls into
    based on the specified fiscal year start month.

    Args:
        run_context: RunContext object for file operations
        date: Date to convert (datetime.date or string)
        fiscal_year_start_month: Fiscal year start month (1-12, where 1=January)

    Returns:
        int: Fiscal year

    Raises:
        ValidationError: If inputs are invalid
        ConfigurationError: If fiscal_year_start_month is out of range

    Financial Examples:
        # Standard fiscal year starting in April (common in many countries)
        >>> FISCAL_YEAR(ctx, datetime.date(2024, 3, 15), fiscal_year_start_month=4)
        2023  # March 2024 is in FY 2023-24

        >>> FISCAL_YEAR(ctx, datetime.date(2024, 5, 15), fiscal_year_start_month=4)
        2024  # May 2024 is in FY 2024-25

        # US Federal fiscal year starting in October
        >>> FISCAL_YEAR(ctx, datetime.date(2024, 9, 30), fiscal_year_start_month=10)
        2024  # September 2024 is in FY 2024

        >>> FISCAL_YEAR(ctx, datetime.date(2024, 10, 1), fiscal_year_start_month=10)
        2025  # October 2024 is in FY 2025

    Example:
        >>> FISCAL_YEAR(ctx, '2024-03-15', fiscal_year_start_month=4)
        2023
    """
    # Validate fiscal year start month
    if not isinstance(fiscal_year_start_month, int) or not (1 <= fiscal_year_start_month <= 12):
        raise ConfigurationError("fiscal_year_start_month must be an integer between 1 and 12")

    # Parse date input
    parsed_date = _parse_date_input(date)

    try:
        # Core calculation
        # For fiscal year naming, we use the year when the fiscal year starts
        # For example, FY 2024 for April start means Apr 2024 - Mar 2025
        if parsed_date.month >= fiscal_year_start_month:
            # Date is in the fiscal year that starts in the same calendar year
            return parsed_date.year
        else:
            # Date is in the fiscal year that started in the previous calendar year
            return parsed_date.year - 1

    except Exception as e:
        raise CalculationError(f"FISCAL_YEAR calculation failed: {str(e)}")


def FISCAL_QUARTER(run_context: Any, date: Union[datetime.date, str], *, fiscal_year_start_month: int) -> str:
    """
    Convert date to fiscal quarter.

    Determines which fiscal quarter (Q1, Q2, Q3, Q4) a given date falls into
    based on the specified fiscal year start month. Fiscal quarters are 3-month
    periods within the fiscal year.

    Args:
        run_context: RunContext object for file operations
        date: Date to convert (datetime.date or string)
        fiscal_year_start_month: Fiscal year start month (1-12, where 1=January)

    Returns:
        str: Fiscal quarter ('Q1', 'Q2', 'Q3', 'Q4')

    Raises:
        ValidationError: If inputs are invalid
        ConfigurationError: If fiscal_year_start_month is out of range

    Financial Examples:
        # Fiscal year starting in April (Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar)
        >>> FISCAL_QUARTER(ctx, datetime.date(2024, 5, 15), fiscal_year_start_month=4)
        'Q1'  # May is in Q1 of fiscal year

        >>> FISCAL_QUARTER(ctx, datetime.date(2024, 8, 15), fiscal_year_start_month=4)
        'Q2'  # August is in Q2 of fiscal year

        >>> FISCAL_QUARTER(ctx, datetime.date(2024, 2, 15), fiscal_year_start_month=4)
        'Q4'  # February is in Q4 of fiscal year (previous calendar year's fiscal year)

        # Calendar year fiscal year (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
        >>> FISCAL_QUARTER(ctx, datetime.date(2024, 3, 15), fiscal_year_start_month=1)
        'Q1'  # March is in Q1 of calendar fiscal year

    Example:
        >>> FISCAL_QUARTER(ctx, '2024-03-15', fiscal_year_start_month=4)
        'Q4'
    """
    # Validate fiscal year start month
    if not isinstance(fiscal_year_start_month, int) or not (1 <= fiscal_year_start_month <= 12):
        raise ConfigurationError("fiscal_year_start_month must be an integer between 1 and 12")

    # Parse date input
    parsed_date = _parse_date_input(date)

    try:
        # Calculate fiscal month (0-11, where 0 is the first month of fiscal year)
        fiscal_month = (parsed_date.month - fiscal_year_start_month) % 12

        # Determine quarter based on fiscal month
        if 0 <= fiscal_month <= 2:
            return 'Q1'
        elif 3 <= fiscal_month <= 5:
            return 'Q2'
        elif 6 <= fiscal_month <= 8:
            return 'Q3'
        else:  # 9 <= fiscal_month <= 11
            return 'Q4'

    except Exception as e:
        raise CalculationError(f"FISCAL_QUARTER calculation failed: {str(e)}")


def BUSINESS_DAYS_BETWEEN(run_context: Any, start_date: Union[datetime.date, str], end_date: Union[datetime.date, str], *, holidays_list: List[Union[datetime.date, str]] | None = None) -> int:
    """
    Calculate business days between dates.

    Calculates the number of business days (weekdays excluding weekends and holidays)
    between two dates. This is crucial for financial calculations involving settlement
    dates, working day calculations, and business timeline planning.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (datetime.date or string)
        end_date: End date (datetime.date or string)
        holidays_list: Optional list of holiday dates to exclude (datetime.date or strings)

    Returns:
        int: Number of business days between dates (inclusive of start, exclusive of end)

    Raises:
        ValidationError: If inputs are invalid
        CalculationError: If date calculation fails

    Financial Examples:
        # Basic business days calculation (Monday to Friday)
        >>> BUSINESS_DAYS_BETWEEN(ctx, datetime.date(2024, 1, 1), datetime.date(2024, 1, 8))
        5  # Excludes weekend days

        # With holidays
        >>> holidays = [datetime.date(2024, 1, 1)]  # New Year's Day
        >>> BUSINESS_DAYS_BETWEEN(ctx, datetime.date(2024, 1, 1), datetime.date(2024, 1, 8), holidays_list=holidays)
        4  # Excludes weekend days and New Year's Day

        # Settlement date calculations
        >>> trade_date = datetime.date(2024, 3, 15)  # Friday
        >>> settlement_date = datetime.date(2024, 3, 19)  # Tuesday (T+2 settlement)
        >>> BUSINESS_DAYS_BETWEEN(ctx, trade_date, settlement_date)
        2  # 2 business days for settlement

    Example:
        >>> BUSINESS_DAYS_BETWEEN(ctx, '2024-01-01', '2024-01-31', holidays_list=['2024-01-15'])
        22
    """
    # Parse date inputs
    start_parsed = _parse_date_input(start_date)
    end_parsed = _parse_date_input(end_date)

    # Validate date order
    if start_parsed > end_parsed:
        raise ValidationError("start_date must be less than or equal to end_date")

    try:
        # Convert to Polars date range
        date_range = pl.date_range(start_parsed, end_parsed, "1d", eager=True)

        # Create Series and filter for business days
        date_series = pl.Series("dates", date_range)
        business_days = date_series.filter(date_series.dt.weekday() <= 5)  # Monday=1, Sunday=7

        # Exclude holidays if provided
        if holidays_list:
            # Parse holiday dates
            parsed_holidays = [_parse_date_input(holiday) for holiday in holidays_list]
            holiday_series = pl.Series("holidays", parsed_holidays)

            # Filter out holidays
            business_days = business_days.filter(~business_days.is_in(holiday_series))

        # Return count (subtract 1 to exclude end date, making it exclusive)
        result = len(business_days) - 1
        return max(0, result)  # Ensure non-negative result

    except Exception as e:
        raise CalculationError(f"BUSINESS_DAYS_BETWEEN calculation failed: {str(e)}")


def END_OF_PERIOD(run_context: Any, date: Union[datetime.date, str], *, period_type: str) -> datetime.date:
    """
    Get end date of period (month, quarter, year).

    Returns the last date of the specified period (month, quarter, or year) that contains
    the given date. This is essential for financial reporting periods, accrual calculations,
    and period-end financial analysis.

    Args:
        run_context: RunContext object for file operations
        date: Date to convert (datetime.date or string)
        period_type: Type of period ('month', 'quarter', 'year')

    Returns:
        datetime.date: End date of the period

    Raises:
        ValidationError: If inputs are invalid
        ConfigurationError: If period_type is not supported
        CalculationError: If date calculation fails

    Financial Examples:
        # Month-end for accrual calculations
        >>> END_OF_PERIOD(ctx, datetime.date(2024, 2, 15), period_type='month')
        datetime.date(2024, 2, 29)  # February 2024 (leap year)

        # Quarter-end for quarterly reporting
        >>> END_OF_PERIOD(ctx, datetime.date(2024, 2, 15), period_type='quarter')
        datetime.date(2024, 3, 31)  # Q1 2024 ends March 31

        # Year-end for annual reporting
        >>> END_OF_PERIOD(ctx, datetime.date(2024, 6, 15), period_type='year')
        datetime.date(2024, 12, 31)  # Calendar year 2024 ends December 31

        # Edge cases
        >>> END_OF_PERIOD(ctx, datetime.date(2024, 12, 31), period_type='month')
        datetime.date(2024, 12, 31)  # Already at month end

    Example:
        >>> END_OF_PERIOD(ctx, '2024-03-15', period_type='quarter')
        datetime.date(2024, 3, 31)
    """
    # Validate period type
    valid_periods = ['month', 'quarter', 'year']
    if period_type not in valid_periods:
        raise ConfigurationError(f"period_type must be one of {valid_periods}, got '{period_type}'")

    # Parse date input
    parsed_date = _parse_date_input(date)

    try:
        # Convert to Polars Series for date operations
        date_series = pl.Series("date", [parsed_date])

        if period_type == 'month':
            # Use Polars month_end functionality
            result_series = date_series.dt.month_end()

        elif period_type == 'quarter':
            # Calculate quarter end manually
            quarter = ((parsed_date.month - 1) // 3) + 1
            if quarter == 1:  # Q1: Jan-Mar
                end_month = 3
            elif quarter == 2:  # Q2: Apr-Jun
                end_month = 6
            elif quarter == 3:  # Q3: Jul-Sep
                end_month = 9
            else:  # Q4: Oct-Dec
                end_month = 12

            # Create date for last day of quarter end month
            quarter_end_date = datetime.date(parsed_date.year, end_month, 1)
            # Get the actual last day of that month
            quarter_end_series = pl.Series("quarter_end", [quarter_end_date])
            result_series = quarter_end_series.dt.month_end()

        elif period_type == 'year':
            # Year end is always December 31
            year_end_date = datetime.date(parsed_date.year, 12, 31)
            result_series = pl.Series("year_end", [year_end_date])

        # Extract the result
        result_date = result_series[0]
        return result_date

    except Exception as e:
        raise CalculationError(f"END_OF_PERIOD calculation failed: {str(e)}")


def PERIOD_OVERLAP(run_context: Any, start1: Union[datetime.date, str], end1: Union[datetime.date, str], start2: Union[datetime.date, str], end2: Union[datetime.date, str]) -> int:
    """
    Calculate overlap between two periods in days.

    Determines the number of days that two date periods overlap. This is crucial for
    financial analysis involving overlapping contracts, revenue recognition periods,
    lease calculations, and project timeline analysis.

    Args:
        run_context: RunContext object for file operations
        start1: First period start date (datetime.date or string)
        end1: First period end date (datetime.date or string)
        start2: Second period start date (datetime.date or string)
        end2: Second period end date (datetime.date or string)

    Returns:
        int: Number of overlapping days (0 if no overlap)

    Raises:
        ValidationError: If inputs are invalid or date ranges are invalid
        CalculationError: If overlap calculation fails

    Financial Examples:
        # Contract overlap analysis
        >>> contract1_start = datetime.date(2024, 1, 1)
        >>> contract1_end = datetime.date(2024, 6, 30)
        >>> contract2_start = datetime.date(2024, 4, 1)
        >>> contract2_end = datetime.date(2024, 9, 30)
        >>> PERIOD_OVERLAP(ctx, contract1_start, contract1_end, contract2_start, contract2_end)
        91  # April 1 to June 30, 2024 (91 days overlap)

        # Revenue recognition periods
        >>> service_period = (datetime.date(2024, 1, 1), datetime.date(2024, 3, 31))
        >>> billing_period = (datetime.date(2024, 2, 1), datetime.date(2024, 4, 30))
        >>> PERIOD_OVERLAP(ctx, *service_period, *billing_period)
        59  # February 1 to March 31, 2024

        # No overlap case
        >>> PERIOD_OVERLAP(ctx, datetime.date(2024, 1, 1), datetime.date(2024, 1, 31),
        ...                 datetime.date(2024, 2, 1), datetime.date(2024, 2, 29))
        0  # No overlap between January and February

        # Adjacent periods (touching but not overlapping)
        >>> PERIOD_OVERLAP(ctx, datetime.date(2024, 1, 1), datetime.date(2024, 1, 31),
        ...                 datetime.date(2024, 1, 31), datetime.date(2024, 2, 29))
        1  # One day overlap (January 31)

    Example:
        >>> PERIOD_OVERLAP(ctx, '2024-01-01', '2024-06-30', '2024-04-01', '2024-09-30')
        91
    """
    # Parse all date inputs
    start1_parsed = _parse_date_input(start1)
    end1_parsed = _parse_date_input(end1)
    start2_parsed = _parse_date_input(start2)
    end2_parsed = _parse_date_input(end2)

    # Validate date ranges
    if start1_parsed > end1_parsed:
        raise ValidationError("First period: start1 must be less than or equal to end1")
    if start2_parsed > end2_parsed:
        raise ValidationError("Second period: start2 must be less than or equal to end2")

    try:
        # Calculate overlap using max of starts and min of ends
        overlap_start = max(start1_parsed, start2_parsed)
        overlap_end = min(end1_parsed, end2_parsed)

        # If overlap_start > overlap_end, there's no overlap
        if overlap_start > overlap_end:
            return 0

        # Calculate the number of days in overlap (inclusive of both start and end)
        overlap_days = (overlap_end - overlap_start).days + 1

        return max(0, overlap_days)  # Ensure non-negative result

    except Exception as e:
        raise CalculationError(f"PERIOD_OVERLAP calculation failed: {str(e)}")
