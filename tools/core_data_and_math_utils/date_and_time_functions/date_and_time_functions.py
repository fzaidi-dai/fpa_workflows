"""
Date & Time Functions

Essential for forecasting, scheduling cash flows, and working with time series data.
All functions use proper date/time handling for financial accuracy and are optimized for AI agent integration.
"""

from typing import Any, Union
import datetime
import calendar
from decimal import Decimal, getcontext
from pathlib import Path
import polars as pl
from dateutil.relativedelta import relativedelta

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


def _parse_date_input(date_input: Union[datetime.date, datetime.datetime, str]) -> datetime.date:
    """
    Parse various date input formats into datetime.date.

    Args:
        date_input: Date in various formats

    Returns:
        datetime.date: Parsed date

    Raises:
        DataQualityError: If date cannot be parsed
    """
    if isinstance(date_input, datetime.date):
        return date_input
    elif isinstance(date_input, datetime.datetime):
        return date_input.date()
    elif isinstance(date_input, str):
        # Try common date formats
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%Y%m%d'
        ]

        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_input, fmt).date()
            except ValueError:
                continue

        raise DataQualityError(
            f"Cannot parse date string: {date_input}",
            "Use ISO format (YYYY-MM-DD) or common formats like MM/DD/YYYY"
        )
    else:
        raise DataQualityError(
            f"Invalid date type: {type(date_input)}",
            "Provide date as datetime.date, datetime.datetime, or string"
        )


def _parse_time_input(time_input: Union[datetime.time, datetime.datetime, str]) -> datetime.time:
    """
    Parse various time input formats into datetime.time.

    Args:
        time_input: Time in various formats

    Returns:
        datetime.time: Parsed time

    Raises:
        DataQualityError: If time cannot be parsed
    """
    if isinstance(time_input, datetime.time):
        return time_input
    elif isinstance(time_input, datetime.datetime):
        return time_input.time()
    elif isinstance(time_input, str):
        # Try common time formats
        formats = [
            '%H:%M:%S',
            '%H:%M',
            '%I:%M:%S %p',
            '%I:%M %p'
        ]

        for fmt in formats:
            try:
                return datetime.datetime.strptime(time_input, fmt).time()
            except ValueError:
                continue

        raise DataQualityError(
            f"Cannot parse time string: {time_input}",
            "Use formats like HH:MM:SS, HH:MM, or HH:MM AM/PM"
        )
    else:
        raise DataQualityError(
            f"Invalid time type: {type(time_input)}",
            "Provide time as datetime.time, datetime.datetime, or string"
        )


def TODAY(run_context: Any) -> datetime.date:
    """
    Return the current date.

    Args:
        run_context: RunContext object for file operations

    Returns:
        datetime.date: Current date

    Example:
        >>> TODAY(ctx)
        datetime.date(2025, 1, 8)
    """
    return datetime.date.today()


def NOW(run_context: Any) -> datetime.datetime:
    """
    Return the current date and time.

    Args:
        run_context: RunContext object for file operations

    Returns:
        datetime.datetime: Current date and time

    Example:
        >>> NOW(ctx)
        datetime.datetime(2025, 1, 8, 14, 30, 45, 123456)
    """
    return datetime.datetime.now()


def DATE(run_context: Any, year: int, month: int, day: int) -> datetime.date:
    """
    Construct a date from year, month, and day components.

    Args:
        run_context: RunContext object for file operations
        year: Year (e.g., 2025)
        month: Month (1-12)
        day: Day (1-31)

    Returns:
        datetime.date: Constructed date

    Raises:
        ValidationError: If date components are invalid
        CalculationError: If date is invalid (e.g., Feb 30)

    Example:
        >>> DATE(ctx, 2025, 4, 15)
        datetime.date(2025, 4, 15)
    """
    try:
        # Validate input ranges
        if not isinstance(year, int) or year < 1:
            raise ValidationError("Year must be a positive integer")
        if not isinstance(month, int) or not (1 <= month <= 12):
            raise ValidationError("Month must be an integer between 1 and 12")
        if not isinstance(day, int) or not (1 <= day <= 31):
            raise ValidationError("Day must be an integer between 1 and 31")

        return datetime.date(year, month, day)
    except ValueError as e:
        raise CalculationError(f"Invalid date: {year}-{month}-{day}. {str(e)}")


def YEAR(run_context: Any, date: Union[datetime.date, str, Path]) -> int:
    """
    Extract the year from a date.

    Args:
        run_context: RunContext object for file operations
        date: Date to extract year from (date object, string, or file path)

    Returns:
        int: Year component

    Raises:
        DataQualityError: If date cannot be parsed

    Example:
        >>> YEAR(ctx, datetime.date(2025, 4, 15))
        2025
        >>> YEAR(ctx, "2025-04-15")
        2025
    """
    # Handle file path input
    if isinstance(date, (str, Path)):
        file_path = Path(date)
        if file_path.exists() or (isinstance(date, str) and not any(c in date for c in ['-', '/', '\\']) and '.' in date):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, date)
                # Assume first column contains the date data
                date_value = df[df.columns[0]][0]
                parsed_date = _parse_date_input(date_value)
            except:
                # If loading fails, treat as date string
                parsed_date = _parse_date_input(date)
        else:
            parsed_date = _parse_date_input(date)
    else:
        parsed_date = _parse_date_input(date)

    return parsed_date.year


def MONTH(run_context: Any, date: Union[datetime.date, str, Path]) -> int:
    """
    Extract the month from a date.

    Args:
        run_context: RunContext object for file operations
        date: Date to extract month from (date object, string, or file path)

    Returns:
        int: Month component (1-12)

    Raises:
        DataQualityError: If date cannot be parsed

    Example:
        >>> MONTH(ctx, datetime.date(2025, 4, 15))
        4
        >>> MONTH(ctx, "2025-04-15")
        4
    """
    # Handle file path input
    if isinstance(date, (str, Path)):
        file_path = Path(date)
        if file_path.exists() or (isinstance(date, str) and not any(c in date for c in ['-', '/', '\\']) and '.' in date):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, date)
                # Assume first column contains the date data
                date_value = df[df.columns[0]][0]
                parsed_date = _parse_date_input(date_value)
            except:
                # If loading fails, treat as date string
                parsed_date = _parse_date_input(date)
        else:
            parsed_date = _parse_date_input(date)
    else:
        parsed_date = _parse_date_input(date)

    return parsed_date.month


def DAY(run_context: Any, date: Union[datetime.date, str, Path]) -> int:
    """
    Extract the day from a date.

    Args:
        run_context: RunContext object for file operations
        date: Date to extract day from (date object, string, or file path)

    Returns:
        int: Day component (1-31)

    Raises:
        DataQualityError: If date cannot be parsed

    Example:
        >>> DAY(ctx, datetime.date(2025, 4, 15))
        15
        >>> DAY(ctx, "2025-04-15")
        15
    """
    # Handle file path input
    if isinstance(date, (str, Path)):
        file_path = Path(date)
        if file_path.exists() or (isinstance(date, str) and not any(c in date for c in ['-', '/', '\\']) and '.' in date):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, date)
                # Assume first column contains the date data
                date_value = df[df.columns[0]][0]
                parsed_date = _parse_date_input(date_value)
            except:
                # If loading fails, treat as date string
                parsed_date = _parse_date_input(date)
        else:
            parsed_date = _parse_date_input(date)
    else:
        parsed_date = _parse_date_input(date)

    return parsed_date.day


def EDATE(run_context: Any, start_date: Union[datetime.date, str, Path], months: int) -> datetime.date:
    """
    Calculate a date a given number of months before or after a specified date.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (date object, string, or file path)
        months: Number of months to add (positive) or subtract (negative)

    Returns:
        datetime.date: Calculated date

    Raises:
        DataQualityError: If date cannot be parsed
        ValidationError: If months is not an integer

    Example:
        >>> EDATE(ctx, datetime.date(2025, 1, 15), 3)
        datetime.date(2025, 4, 15)
        >>> EDATE(ctx, "2025-01-15", -2)
        datetime.date(2024, 11, 15)
    """
    # Handle file path input
    if isinstance(start_date, (str, Path)):
        file_path = Path(start_date)
        if file_path.exists() or (isinstance(start_date, str) and not any(c in start_date for c in ['-', '/', '\\']) and '.' in start_date):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, start_date)
                # Assume first column contains the date data
                date_value = df[df.columns[0]][0]
                parsed_date = _parse_date_input(date_value)
            except:
                # If loading fails, treat as date string
                parsed_date = _parse_date_input(start_date)
        else:
            parsed_date = _parse_date_input(start_date)
    else:
        parsed_date = _parse_date_input(start_date)

    if not isinstance(months, int):
        raise ValidationError("Months must be an integer")

    try:
        result_date = parsed_date + relativedelta(months=months)
        return result_date
    except Exception as e:
        raise CalculationError(f"Error calculating EDATE: {str(e)}")


def EOMONTH(run_context: Any, start_date: Union[datetime.date, str, Path], months: int) -> datetime.date:
    """
    Find the end of the month for a given date.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (date object, string, or file path)
        months: Number of months to add (positive) or subtract (negative)

    Returns:
        datetime.date: End of month date

    Raises:
        DataQualityError: If date cannot be parsed
        ValidationError: If months is not an integer

    Example:
        >>> EOMONTH(ctx, datetime.date(2025, 1, 15), 0)
        datetime.date(2025, 1, 31)
        >>> EOMONTH(ctx, "2025-01-15", 2)
        datetime.date(2025, 3, 31)
    """
    # Handle file path input
    if isinstance(start_date, (str, Path)):
        file_path = Path(start_date)
        if file_path.exists() or (isinstance(start_date, str) and not any(c in start_date for c in ['-', '/', '\\']) and '.' in start_date):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, start_date)
                # Assume first column contains the date data
                date_value = df[df.columns[0]][0]
                parsed_date = _parse_date_input(date_value)
            except:
                # If loading fails, treat as date string
                parsed_date = _parse_date_input(start_date)
        else:
            parsed_date = _parse_date_input(start_date)
    else:
        parsed_date = _parse_date_input(start_date)

    if not isinstance(months, int):
        raise ValidationError("Months must be an integer")

    try:
        # Add months to get target month
        target_date = parsed_date + relativedelta(months=months)
        # Get last day of that month
        last_day = calendar.monthrange(target_date.year, target_date.month)[1]
        return datetime.date(target_date.year, target_date.month, last_day)
    except Exception as e:
        raise CalculationError(f"Error calculating EOMONTH: {str(e)}")


def DATEDIF(run_context: Any, start_date: Union[datetime.date, str, Path], end_date: Union[datetime.date, str, Path], unit: str) -> int:
    """
    Calculate the difference between two dates.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (date object, string, or file path)
        end_date: End date (date object, string, or file path)
        unit: Unit of difference ("Y" for years, "M" for months, "D" for days)

    Returns:
        int: Difference in specified unit

    Raises:
        DataQualityError: If dates cannot be parsed
        ValidationError: If unit is invalid

    Example:
        >>> DATEDIF(ctx, datetime.date(2024, 1, 1), datetime.date(2025, 1, 1), "Y")
        1
        >>> DATEDIF(ctx, "2024-01-01", "2024-04-01", "M")
        3
    """
    # Handle file path input for start_date
    if isinstance(start_date, (str, Path)) and Path(start_date).exists():
        df = load_df(run_context, start_date)
        date_value = df[df.columns[0]][0]
        parsed_start = _parse_date_input(date_value)
    else:
        parsed_start = _parse_date_input(start_date)

    # Handle file path input for end_date
    if isinstance(end_date, (str, Path)) and Path(end_date).exists():
        df = load_df(run_context, end_date)
        date_value = df[df.columns[0]][0]
        parsed_end = _parse_date_input(date_value)
    else:
        parsed_end = _parse_date_input(end_date)

    unit = unit.upper()
    if unit not in ["Y", "M", "D"]:
        raise ValidationError("Unit must be 'Y' (years), 'M' (months), or 'D' (days)")

    if parsed_start > parsed_end:
        raise ValidationError("Start date must be before or equal to end date")

    try:
        if unit == "D":
            return (parsed_end - parsed_start).days
        elif unit == "M":
            return (parsed_end.year - parsed_start.year) * 12 + (parsed_end.month - parsed_start.month)
        elif unit == "Y":
            return parsed_end.year - parsed_start.year
    except Exception as e:
        raise CalculationError(f"Error calculating DATEDIF: {str(e)}")


def YEARFRAC(run_context: Any, start_date: Union[datetime.date, str, Path], end_date: Union[datetime.date, str, Path], basis: int | None = None) -> Decimal:
    """
    Calculate the fraction of a year between two dates.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (date object, string, or file path)
        end_date: End date (date object, string, or file path)
        basis: Day count basis (0=30/360 US, 1=Actual/Actual, 2=Actual/360, 3=Actual/365, 4=30/360 European)

    Returns:
        Decimal: Fraction of year between dates

    Raises:
        DataQualityError: If dates cannot be parsed
        ValidationError: If basis is invalid

    Example:
        >>> YEARFRAC(ctx, datetime.date(2024, 1, 1), datetime.date(2024, 7, 1), 1)
        Decimal('0.4972677595628415')
    """
    # Handle file path input for start_date
    if isinstance(start_date, (str, Path)) and Path(start_date).exists():
        df = load_df(run_context, start_date)
        date_value = df[df.columns[0]][0]
        parsed_start = _parse_date_input(date_value)
    else:
        parsed_start = _parse_date_input(start_date)

    # Handle file path input for end_date
    if isinstance(end_date, (str, Path)) and Path(end_date).exists():
        df = load_df(run_context, end_date)
        date_value = df[df.columns[0]][0]
        parsed_end = _parse_date_input(date_value)
    else:
        parsed_end = _parse_date_input(end_date)

    if basis is None:
        basis = 0

    if basis not in [0, 1, 2, 3, 4]:
        raise ValidationError("Basis must be 0, 1, 2, 3, or 4")

    if parsed_start > parsed_end:
        raise ValidationError("Start date must be before or equal to end date")

    try:
        if basis == 0:  # 30/360 US (NASD)
            # Simplified 30/360 calculation
            days = (parsed_end.year - parsed_start.year) * 360 + \
                   (parsed_end.month - parsed_start.month) * 30 + \
                   (parsed_end.day - parsed_start.day)
            return Decimal(days) / Decimal('360')

        elif basis == 1:  # Actual/Actual
            days = (parsed_end - parsed_start).days
            # Use actual days in year(s)
            if parsed_start.year == parsed_end.year:
                year_days = 366 if calendar.isleap(parsed_start.year) else 365
                return Decimal(days) / Decimal(year_days)
            else:
                # Complex calculation for multiple years
                total_fraction = Decimal('0')
                current_date = parsed_start

                while current_date.year < parsed_end.year:
                    year_end = datetime.date(current_date.year, 12, 31)
                    year_days = 366 if calendar.isleap(current_date.year) else 365
                    days_in_year = (year_end - current_date).days + 1
                    total_fraction += Decimal(days_in_year) / Decimal(year_days)
                    current_date = datetime.date(current_date.year + 1, 1, 1)

                # Add final year
                if current_date <= parsed_end:
                    year_days = 366 if calendar.isleap(parsed_end.year) else 365
                    days_in_final_year = (parsed_end - current_date).days
                    total_fraction += Decimal(days_in_final_year) / Decimal(year_days)

                return total_fraction

        elif basis == 2:  # Actual/360
            days = (parsed_end - parsed_start).days
            return Decimal(days) / Decimal('360')

        elif basis == 3:  # Actual/365
            days = (parsed_end - parsed_start).days
            return Decimal(days) / Decimal('365')

        elif basis == 4:  # 30/360 European
            # Similar to basis 0 but with European conventions
            days = (parsed_end.year - parsed_start.year) * 360 + \
                   (parsed_end.month - parsed_start.month) * 30 + \
                   (parsed_end.day - parsed_start.day)
            return Decimal(days) / Decimal('360')

    except Exception as e:
        raise CalculationError(f"Error calculating YEARFRAC: {str(e)}")


def WORKDAY(run_context: Any, start_date: Union[datetime.date, str, Path], days: int, holidays: list[datetime.date] | None = None) -> datetime.date:
    """
    Return a future or past date excluding weekends and holidays.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (date object, string, or file path)
        days: Number of working days to add (positive) or subtract (negative)
        holidays: Optional list of holiday dates to exclude

    Returns:
        datetime.date: Calculated working day

    Raises:
        DataQualityError: If date cannot be parsed
        ValidationError: If days is not an integer

    Example:
        >>> WORKDAY(ctx, datetime.date(2025, 1, 1), 5)
        datetime.date(2025, 1, 8)
    """
    # Handle file path input
    if isinstance(start_date, (str, Path)) and Path(start_date).exists():
        df = load_df(run_context, start_date)
        date_value = df[df.columns[0]][0]
        parsed_date = _parse_date_input(date_value)
    else:
        parsed_date = _parse_date_input(start_date)

    if not isinstance(days, int):
        raise ValidationError("Days must be an integer")

    if holidays is None:
        holidays = []

    try:
        current_date = parsed_date
        remaining_days = abs(days)
        direction = 1 if days >= 0 else -1

        while remaining_days > 0:
            current_date += datetime.timedelta(days=direction)

            # Check if it's a weekday (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                # Check if it's not a holiday
                if current_date not in holidays:
                    remaining_days -= 1

        return current_date

    except Exception as e:
        raise CalculationError(f"Error calculating WORKDAY: {str(e)}")


def NETWORKDAYS(run_context: Any, start_date: Union[datetime.date, str, Path], end_date: Union[datetime.date, str, Path], holidays: list[datetime.date] | None = None) -> int:
    """
    Count working days between two dates.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (date object, string, or file path)
        end_date: End date (date object, string, or file path)
        holidays: Optional list of holiday dates to exclude

    Returns:
        int: Number of working days

    Raises:
        DataQualityError: If dates cannot be parsed

    Example:
        >>> NETWORKDAYS(ctx, datetime.date(2025, 1, 1), datetime.date(2025, 1, 10))
        8
    """
    # Handle file path input for start_date
    if isinstance(start_date, (str, Path)) and Path(start_date).exists():
        df = load_df(run_context, start_date)
        date_value = df[df.columns[0]][0]
        parsed_start = _parse_date_input(date_value)
    else:
        parsed_start = _parse_date_input(start_date)

    # Handle file path input for end_date
    if isinstance(end_date, (str, Path)) and Path(end_date).exists():
        df = load_df(run_context, end_date)
        date_value = df[df.columns[0]][0]
        parsed_end = _parse_date_input(date_value)
    else:
        parsed_end = _parse_date_input(end_date)

    if holidays is None:
        holidays = []

    if parsed_start > parsed_end:
        raise ValidationError("Start date must be before or equal to end date")

    try:
        workdays = 0
        current_date = parsed_start

        while current_date <= parsed_end:
            # Check if it's a weekday (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                # Check if it's not a holiday
                if current_date not in holidays:
                    workdays += 1

            current_date += datetime.timedelta(days=1)

        return workdays

    except Exception as e:
        raise CalculationError(f"Error calculating NETWORKDAYS: {str(e)}")


def DATE_RANGE(run_context: Any, start_date: Union[datetime.date, str, Path], end_date: Union[datetime.date, str, Path], frequency: str, *, output_filename: str | None = None) -> list[datetime.date]:
    """
    Generate a series of dates between a start and end date with a specified frequency, essential for creating financial model timelines.

    Args:
        run_context: RunContext object for file operations
        start_date: Start date (date object, string, or file path)
        end_date: End date (date object, string, or file path)
        frequency: Frequency ('D' for daily, 'W' for weekly, 'M' for month-end, 'Q' for quarter-end, 'Y' for year-end)
        output_filename: Optional filename to save results as parquet file

    Returns:
        list[datetime.date]: Series of dates

    Raises:
        DataQualityError: If dates cannot be parsed
        ValidationError: If frequency is invalid

    Example:
        >>> DATE_RANGE(ctx, "2025-01-01", "2025-03-31", "M")
        [datetime.date(2025, 1, 31), datetime.date(2025, 2, 28), datetime.date(2025, 3, 31)]
    """
    # Handle file path input for start_date
    if isinstance(start_date, (str, Path)) and Path(start_date).exists():
        df = load_df(run_context, start_date)
        date_value = df[df.columns[0]][0]
        parsed_start = _parse_date_input(date_value)
    else:
        parsed_start = _parse_date_input(start_date)

    # Handle file path input for end_date
    if isinstance(end_date, (str, Path)) and Path(end_date).exists():
        df = load_df(run_context, end_date)
        date_value = df[df.columns[0]][0]
        parsed_end = _parse_date_input(date_value)
    else:
        parsed_end = _parse_date_input(end_date)

    frequency = frequency.upper()
    if frequency not in ['D', 'W', 'M', 'Q', 'Y']:
        raise ValidationError("Frequency must be 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly), or 'Y' (yearly)")

    if parsed_start > parsed_end:
        raise ValidationError("Start date must be before or equal to end date")

    try:
        dates = []
        current_date = parsed_start

        if frequency == 'D':
            while current_date <= parsed_end:
                dates.append(current_date)
                current_date += datetime.timedelta(days=1)

        elif frequency == 'W':
            while current_date <= parsed_end:
                dates.append(current_date)
                current_date += datetime.timedelta(weeks=1)

        elif frequency == 'M':
            while current_date <= parsed_end:
                # Get end of current month
                last_day = calendar.monthrange(current_date.year, current_date.month)[1]
                month_end = datetime.date(current_date.year, current_date.month, last_day)
                if month_end <= parsed_end:
                    dates.append(month_end)
                current_date = current_date + relativedelta(months=1)
                current_date = datetime.date(current_date.year, current_date.month, 1)

        elif frequency == 'Q':
            # Start from first quarter end after start date
            quarter_months = [3, 6, 9, 12]
            current_year = current_date.year
            current_month = current_date.month

            # Find next quarter end
            next_quarter_month = None
            for qm in quarter_months:
                if qm >= current_month:
                    next_quarter_month = qm
                    break

            if next_quarter_month is None:
                next_quarter_month = 3
                current_year += 1

            while True:
                quarter_end = datetime.date(current_year, next_quarter_month,
                                          calendar.monthrange(current_year, next_quarter_month)[1])
                if quarter_end > parsed_end:
                    break
                dates.append(quarter_end)

                # Move to next quarter
                if next_quarter_month == 12:
                    next_quarter_month = 3
                    current_year += 1
                else:
                    next_quarter_month += 3

        elif frequency == 'Y':
            current_year = current_date.year
            while True:
                year_end = datetime.date(current_year, 12, 31)
                if year_end > parsed_end:
                    break
                dates.append(year_end)
                current_year += 1

        # Save results to file if output_filename is provided
        if output_filename is not None:
            # Create DataFrame from results
            result_df = pl.DataFrame({
                "date_range": dates
            })
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return dates

    except Exception as e:
        raise CalculationError(f"Error generating DATE_RANGE: {str(e)}")


def WEEKDAY(run_context: Any, serial_number: Union[datetime.date, str, Path], return_type: int | None = None) -> int:
    """
    Return day of week as number.

    Args:
        run_context: RunContext object for file operations
        serial_number: Date (date object, string, or file path)
        return_type: Return type (1=Sunday=1 to Saturday=7, 2=Monday=1 to Sunday=7, 3=Monday=0 to Sunday=6)

    Returns:
        int: Day of week number

    Raises:
        DataQualityError: If date cannot be parsed
        ValidationError: If return_type is invalid

    Example:
        >>> WEEKDAY(ctx, datetime.date(2025, 1, 8))  # Wednesday
        4
    """
    # Handle file path input
    if isinstance(serial_number, (str, Path)):
        file_path = Path(serial_number)
        if file_path.exists() or (isinstance(serial_number, str) and not any(c in serial_number for c in ['-', '/', '\\']) and '.' in serial_number):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, serial_number)
                # Assume first column contains the date data
                date_value = df[df.columns[0]][0]
                parsed_date = _parse_date_input(date_value)
            except:
                # If loading fails, treat as date string
                parsed_date = _parse_date_input(serial_number)
        else:
            parsed_date = _parse_date_input(serial_number)
    else:
        parsed_date = _parse_date_input(serial_number)

    if return_type is None:
        return_type = 1

    if return_type not in [1, 2, 3]:
        raise ValidationError("Return type must be 1, 2, or 3")

    try:
        # Python weekday: Monday=0, Sunday=6
        python_weekday = parsed_date.weekday()

        if return_type == 1:  # Sunday=1, Monday=2, ..., Saturday=7
            return (python_weekday + 2) % 7 or 7
        elif return_type == 2:  # Monday=1, Tuesday=2, ..., Sunday=7
            return python_weekday + 1
        elif return_type == 3:  # Monday=0, Tuesday=1, ..., Sunday=6
            return python_weekday

    except Exception as e:
        raise CalculationError(f"Error calculating WEEKDAY: {str(e)}")


def QUARTER(run_context: Any, date: Union[datetime.date, str, Path]) -> int:
    """
    Extract quarter from date.

    Args:
        run_context: RunContext object for file operations
        date: Date to extract quarter from (date object, string, or file path)

    Returns:
        int: Quarter (1-4)

    Raises:
        DataQualityError: If date cannot be parsed

    Example:
        >>> QUARTER(ctx, datetime.date(2024, 7, 15))
        3
    """
    # Handle file path input
    if isinstance(date, (str, Path)):
        file_path = Path(date)
        if file_path.exists() or (isinstance(date, str) and not any(c in date for c in ['-', '/', '\\']) and '.' in date):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, date)
                # Assume first column contains the date data
                date_value = df[df.columns[0]][0]
                parsed_date = _parse_date_input(date_value)
            except:
                # If loading fails, treat as date string
                parsed_date = _parse_date_input(date)
        else:
            parsed_date = _parse_date_input(date)
    else:
        parsed_date = _parse_date_input(date)

    return (parsed_date.month - 1) // 3 + 1


def TIME(run_context: Any, hour: int, minute: int, second: int) -> datetime.time:
    """
    Create time value from hours, minutes, seconds.

    Args:
        run_context: RunContext object for file operations
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second (0-59)

    Returns:
        datetime.time: Time value

    Raises:
        ValidationError: If time components are invalid
        CalculationError: If time is invalid

    Example:
        >>> TIME(ctx, 14, 30, 0)
        datetime.time(14, 30)
    """
    try:
        # Validate input ranges
        if not isinstance(hour, int) or not (0 <= hour <= 23):
            raise ValidationError("Hour must be an integer between 0 and 23")
        if not isinstance(minute, int) or not (0 <= minute <= 59):
            raise ValidationError("Minute must be an integer between 0 and 59")
        if not isinstance(second, int) or not (0 <= second <= 59):
            raise ValidationError("Second must be an integer between 0 and 59")

        return datetime.time(hour, minute, second)
    except ValueError as e:
        raise CalculationError(f"Invalid time: {hour}:{minute}:{second}. {str(e)}")


def HOUR(run_context: Any, serial_number: Union[datetime.time, datetime.datetime, str, Path]) -> int:
    """
    Extract hour from time.

    Args:
        run_context: RunContext object for file operations
        serial_number: Time value (time object, datetime object, string, or file path)

    Returns:
        int: Hour (0-23)

    Raises:
        DataQualityError: If time cannot be parsed

    Example:
        >>> HOUR(ctx, datetime.time(14, 30, 0))
        14
    """
    # Handle file path input
    if isinstance(serial_number, (str, Path)):
        file_path = Path(serial_number)
        if file_path.exists() or (isinstance(serial_number, str) and not any(c in serial_number for c in ['-', '/', '\\']) and '.' in serial_number):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, serial_number)
                # Assume first column contains the time data
                time_value = df[df.columns[0]][0]
                parsed_time = _parse_time_input(time_value)
            except:
                # If loading fails, treat as time string
                parsed_time = _parse_time_input(serial_number)
        else:
            parsed_time = _parse_time_input(serial_number)
    else:
        parsed_time = _parse_time_input(serial_number)

    return parsed_time.hour


def MINUTE(run_context: Any, serial_number: Union[datetime.time, datetime.datetime, str, Path]) -> int:
    """
    Extract minute from time.

    Args:
        run_context: RunContext object for file operations
        serial_number: Time value (time object, datetime object, string, or file path)

    Returns:
        int: Minute (0-59)

    Raises:
        DataQualityError: If time cannot be parsed

    Example:
        >>> MINUTE(ctx, datetime.time(14, 30, 45))
        30
    """
    # Handle file path input
    if isinstance(serial_number, (str, Path)):
        file_path = Path(serial_number)
        if file_path.exists() or (isinstance(serial_number, str) and not any(c in serial_number for c in ['-', '/', '\\']) and '.' in serial_number):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, serial_number)
                # Assume first column contains the time data
                time_value = df[df.columns[0]][0]
                parsed_time = _parse_time_input(time_value)
            except:
                # If loading fails, treat as time string
                parsed_time = _parse_time_input(serial_number)
        else:
            parsed_time = _parse_time_input(serial_number)
    else:
        parsed_time = _parse_time_input(serial_number)

    return parsed_time.minute


def SECOND(run_context: Any, serial_number: Union[datetime.time, datetime.datetime, str, Path]) -> int:
    """
    Extract second from time.

    Args:
        run_context: RunContext object for file operations
        serial_number: Time value (time object, datetime object, string, or file path)

    Returns:
        int: Second (0-59)

    Raises:
        DataQualityError: If time cannot be parsed

    Example:
        >>> SECOND(ctx, datetime.time(14, 30, 45))
        45
    """
    # Handle file path input
    if isinstance(serial_number, (str, Path)):
        file_path = Path(serial_number)
        if file_path.exists() or (isinstance(serial_number, str) and not any(c in serial_number for c in ['-', '/', '\\']) and '.' in serial_number):
            # This looks like a filename, try to load it
            try:
                df = load_df(run_context, serial_number)
                # Assume first column contains the time data
                time_value = df[df.columns[0]][0]
                parsed_time = _parse_time_input(time_value)
            except:
                # If loading fails, treat as time string
                parsed_time = _parse_time_input(serial_number)
        else:
            parsed_time = _parse_time_input(serial_number)
    else:
        parsed_time = _parse_time_input(serial_number)

    return parsed_time.second
