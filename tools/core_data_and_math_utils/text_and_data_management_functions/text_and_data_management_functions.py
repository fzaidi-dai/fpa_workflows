"""
Text & Data Management Functions

Useful for generating labels, combining text, and cleaning up data reports.
All functions use proper validation and are optimized for AI agent integration.
"""

from typing import Any, Union
from decimal import Decimal, getcontext
from pathlib import Path
import polars as pl
import re
import unicodedata
from datetime import datetime, date

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


def _validate_text_input(value: Any, function_name: str) -> str:
    """
    Standardized input validation for text data.

    Args:
        value: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        str: Validated string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid values
    """
    if value is None:
        raise ValidationError(f"Input cannot be None for {function_name}")

    try:
        return str(value)
    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Cannot convert input to string for {function_name}: {str(e)}",
            "Ensure input is a valid text or numeric value"
        )


def _validate_integer_input(value: Any, function_name: str, min_value: int = None, max_value: int = None) -> int:
    """
    Standardized input validation for integer parameters.

    Args:
        value: Input value to validate
        function_name: Name of calling function for error messages
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)

    Returns:
        int: Validated integer

    Raises:
        ValidationError: If input is invalid
    """
    try:
        int_value = int(value)

        if min_value is not None and int_value < min_value:
            raise ValidationError(f"Value must be >= {min_value} for {function_name}")

        if max_value is not None and int_value > max_value:
            raise ValidationError(f"Value must be <= {max_value} for {function_name}")

        return int_value
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid integer value for {function_name}: {str(e)}")


def CONCAT(run_context: Any, *texts: Union[str, int, float, Path]) -> str:
    """
    Merge text strings together (modern version).

    Args:
        run_context: RunContext object for file operations
        texts: Text strings to concatenate (supports file paths)

    Returns:
        str: Combined text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If file processing fails

    Example:
        >>> CONCAT(ctx, "Hello", " ", "World")
        'Hello World'
        >>> CONCAT(ctx, "Revenue: $", 1000, " Million")
        'Revenue: $1000 Million'
        >>> CONCAT(ctx, "data.csv")  # File input
        'concatenated content from file'
    """
    if not texts:
        return ""

    try:
        result_parts = []

        for text in texts:
            # Handle file path input
            if isinstance(text, (str, Path)) and Path(text).suffix in ['.csv', '.parquet']:
                try:
                    df = load_df(run_context, text)
                    # Concatenate all values from first column
                    if len(df) > 0:
                        first_col = df[df.columns[0]]
                        text_values = [_validate_text_input(val, "CONCAT") for val in first_col.to_list()]
                        result_parts.extend(text_values)
                    continue
                except FileNotFoundError:
                    # If file not found, treat as regular text
                    pass

            # Regular text processing
            validated_text = _validate_text_input(text, "CONCAT")
            result_parts.append(validated_text)

        return "".join(result_parts)

    except Exception as e:
        raise DataQualityError(
            f"Error in CONCAT operation: {str(e)}",
            "Ensure all inputs are valid text, numbers, or file paths"
        )


def CONCATENATE(run_context: Any, *texts: Union[str, int, float, Path]) -> str:
    """
    Merge text strings together (legacy version).

    Args:
        run_context: RunContext object for file operations
        texts: Text strings to concatenate (supports file paths)

    Returns:
        str: Combined text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If file processing fails

    Example:
        >>> CONCATENATE(ctx, "Q1", " ", "2024")
        'Q1 2024'
        >>> CONCATENATE(ctx, "Budget: ", 50000)
        'Budget: 50000'
        >>> CONCATENATE(ctx, "data.csv")  # File input
        'concatenated content from file'
    """
    # Legacy version - same implementation as CONCAT for compatibility
    return CONCAT(run_context, *texts)


def TEXT(run_context: Any, value: Union[str, int, float, Decimal, datetime, date, Path], *, format_text: str) -> str:
    """
    Format numbers or dates as text with a specified format.

    Args:
        run_context: RunContext object for file operations
        value: Value to format (number, date, or file path)
        format_text: Format text (Excel-style format codes)

    Returns:
        str: Formatted text string

    Raises:
        ValidationError: If format is invalid
        DataQualityError: If value cannot be formatted

    Financial Examples:
        >>> TEXT(ctx, 0.125, format_text="0.00%")
        '12.50%'
        >>> TEXT(ctx, 1234567.89, format_text="$#,##0.00")
        '$1,234,567.89'
        >>> TEXT(ctx, Decimal('0.0825'), format_text="0.000%")
        '8.250%'
        >>> TEXT(ctx, datetime(2024, 3, 15), format_text="yyyy-mm-dd")
        '2024-03-15'
    """
    # Handle file path input
    if isinstance(value, (str, Path)) and Path(str(value)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, value)
            if len(df) > 0:
                # Use first value from first column
                value = df[df.columns[0]][0]
        except FileNotFoundError:
            # If file not found, treat as regular text
            pass

    try:
        # Handle different format patterns
        format_lower = format_text.lower()

        # Percentage formatting
        if "%" in format_text:
            if isinstance(value, (int, float, Decimal)):
                decimal_value = Decimal(str(value))
                percentage = decimal_value * 100

                # Count decimal places in format
                if "0.00%" in format_text:
                    return f"{percentage:.2f}%"
                elif "0.000%" in format_text:
                    return f"{percentage:.3f}%"
                elif "0.0%" in format_text:
                    return f"{percentage:.1f}%"
                else:
                    return f"{percentage:.0f}%"

        # Currency formatting
        if "$" in format_text and "#,##0" in format_text:
            if isinstance(value, (int, float, Decimal)):
                decimal_value = Decimal(str(value))
                if ".00" in format_text:
                    return f"${decimal_value:,.2f}"
                else:
                    return f"${decimal_value:,.0f}"

        # Date formatting
        if isinstance(value, (datetime, date)):
            if "yyyy-mm-dd" in format_lower:
                return value.strftime("%Y-%m-%d")
            elif "mm/dd/yyyy" in format_lower:
                return value.strftime("%m/%d/%Y")
            elif "dd-mmm-yyyy" in format_lower:
                return value.strftime("%d-%b-%Y")

        # Number formatting with decimals
        if isinstance(value, (int, float, Decimal)):
            decimal_value = Decimal(str(value))
            if "0.00" in format_text:
                return f"{decimal_value:.2f}"
            elif "0.000" in format_text:
                return f"{decimal_value:.3f}"
            elif "0.0" in format_text:
                return f"{decimal_value:.1f}"
            elif "#,##0" in format_text:
                return f"{decimal_value:,.0f}"

        # Default: convert to string
        return _validate_text_input(value, "TEXT")

    except Exception as e:
        raise DataQualityError(
            f"Cannot format value with TEXT function: {str(e)}",
            "Ensure value and format are compatible"
        )


def LEFT(run_context: Any, text: Union[str, Path], *, num_chars: int) -> str:
    """
    Extract characters from the left side of a text string.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path
        num_chars: Number of characters to extract

    Returns:
        str: Text substring

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If text processing fails

    Example:
        >>> LEFT(ctx, "Financial Planning", num_chars=9)
        'Financial'
        >>> LEFT(ctx, "AAPL-2024-Q1", num_chars=4)
        'AAPL'
        >>> LEFT(ctx, "data.csv", num_chars=5)  # File input
        'first'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                # Use first value from first column
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            # If file not found, treat as regular text
            pass

    validated_text = _validate_text_input(text, "LEFT")
    validated_num_chars = _validate_integer_input(num_chars, "LEFT", min_value=0)

    try:
        return validated_text[:validated_num_chars]
    except Exception as e:
        raise DataQualityError(
            f"Error extracting left characters: {str(e)}",
            "Ensure text is valid and num_chars is non-negative"
        )


def RIGHT(run_context: Any, text: Union[str, Path], *, num_chars: int) -> str:
    """
    Extract characters from the right side of a text string.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path
        num_chars: Number of characters to extract

    Returns:
        str: Text substring

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If text processing fails

    Example:
        >>> RIGHT(ctx, "Financial Planning", num_chars=8)
        'Planning'
        >>> RIGHT(ctx, "AAPL-2024-Q1", num_chars=2)
        'Q1'
        >>> RIGHT(ctx, "data.csv", num_chars=3)  # File input
        'ast'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                # Use first value from first column
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            # If file not found, treat as regular text
            pass

    validated_text = _validate_text_input(text, "RIGHT")
    validated_num_chars = _validate_integer_input(num_chars, "RIGHT", min_value=0)

    try:
        if validated_num_chars == 0:
            return ""
        return validated_text[-validated_num_chars:]
    except Exception as e:
        raise DataQualityError(
            f"Error extracting right characters: {str(e)}",
            "Ensure text is valid and num_chars is non-negative"
        )


def MID(run_context: Any, text: Union[str, Path], *, start_num: int, num_chars: int) -> str:
    """
    Extract characters from the middle of a text string.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path
        start_num: Starting position (1-based)
        num_chars: Number of characters to extract

    Returns:
        str: Text substring

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If text processing fails

    Example:
        >>> MID(ctx, "Financial Planning", start_num=11, num_chars=8)
        'Planning'
        >>> MID(ctx, "AAPL-2024-Q1", start_num=6, num_chars=4)
        '2024'
        >>> MID(ctx, "data.csv", start_num=2, num_chars=3)  # File input
        'ata'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                # Use first value from first column
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            # If file not found, treat as regular text
            pass

    validated_text = _validate_text_input(text, "MID")
    validated_start_num = _validate_integer_input(start_num, "MID", min_value=1)
    validated_num_chars = _validate_integer_input(num_chars, "MID", min_value=0)

    try:
        # Convert to 0-based indexing
        start_index = validated_start_num - 1
        end_index = start_index + validated_num_chars
        return validated_text[start_index:end_index]
    except Exception as e:
        raise DataQualityError(
            f"Error extracting middle characters: {str(e)}",
            "Ensure text is valid and positions are within bounds"
        )


def LEN(run_context: Any, text: Union[str, Path]) -> int:
    """
    Count the number of characters in a text string.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path

    Returns:
        int: Character count

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text processing fails

    Example:
        >>> LEN(ctx, "Financial Planning")
        18
        >>> LEN(ctx, "AAPL")
        4
        >>> LEN(ctx, "data.csv")  # File input
        10
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                # Use first value from first column
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            # If file not found, treat as regular text
            pass

    validated_text = _validate_text_input(text, "LEN")

    try:
        return len(validated_text)
    except Exception as e:
        raise DataQualityError(
            f"Error calculating text length: {str(e)}",
            "Ensure text is valid"
        )


def FIND(run_context: Any, find_text: Union[str, Path], within_text: Union[str, Path], start_num: int | None = None) -> int:
    """
    Locate one text string within another (case-sensitive).

    Args:
        run_context: RunContext object for file operations
        find_text: Text to find or file path
        within_text: Text to search within or file path
        start_num: Starting position (1-based, optional)

    Returns:
        int: Position (1-based) or -1 if not found

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If text processing fails

    Example:
        >>> FIND(ctx, "Plan", "Financial Planning")
        11
        >>> FIND(ctx, "plan", "Financial Planning")  # Case-sensitive
        -1
        >>> FIND(ctx, "2024", "AAPL-2024-Q1", start_num=1)
        6
    """
    # Handle file path input for find_text
    if isinstance(find_text, (str, Path)) and Path(str(find_text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, find_text)
            if len(df) > 0:
                find_text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    # Handle file path input for within_text
    if isinstance(within_text, (str, Path)) and Path(str(within_text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, within_text)
            if len(df) > 0:
                within_text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_find_text = _validate_text_input(find_text, "FIND")
    validated_within_text = _validate_text_input(within_text, "FIND")

    if start_num is not None:
        validated_start_num = _validate_integer_input(start_num, "FIND", min_value=1)
        start_index = validated_start_num - 1
    else:
        start_index = 0

    try:
        # Case-sensitive search
        position = validated_within_text.find(validated_find_text, start_index)
        return position + 1 if position != -1 else -1
    except Exception as e:
        raise DataQualityError(
            f"Error in FIND operation: {str(e)}",
            "Ensure both text strings are valid"
        )


def SEARCH(run_context: Any, find_text: Union[str, Path], within_text: Union[str, Path], start_num: int | None = None) -> int:
    """
    Locate one text string within another (not case-sensitive).

    Args:
        run_context: RunContext object for file operations
        find_text: Text to find or file path
        within_text: Text to search within or file path
        start_num: Starting position (1-based, optional)

    Returns:
        int: Position (1-based) or -1 if not found

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If text processing fails

    Example:
        >>> SEARCH(ctx, "plan", "Financial Planning")
        11
        >>> SEARCH(ctx, "PLAN", "Financial Planning")  # Case-insensitive
        11
        >>> SEARCH(ctx, "q1", "AAPL-2024-Q1", start_num=1)
        11
    """
    # Handle file path input for find_text
    if isinstance(find_text, (str, Path)) and Path(str(find_text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, find_text)
            if len(df) > 0:
                find_text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    # Handle file path input for within_text
    if isinstance(within_text, (str, Path)) and Path(str(within_text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, within_text)
            if len(df) > 0:
                within_text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_find_text = _validate_text_input(find_text, "SEARCH")
    validated_within_text = _validate_text_input(within_text, "SEARCH")

    if start_num is not None:
        validated_start_num = _validate_integer_input(start_num, "SEARCH", min_value=1)
        start_index = validated_start_num - 1
    else:
        start_index = 0

    try:
        # Case-insensitive search
        position = validated_within_text.lower().find(validated_find_text.lower(), start_index)
        return position + 1 if position != -1 else -1
    except Exception as e:
        raise DataQualityError(
            f"Error in SEARCH operation: {str(e)}",
            "Ensure both text strings are valid"
        )


def REPLACE(run_context: Any, old_text: Union[str, Path], *, start_num: int, num_chars: int, new_text: str) -> str:
    """
    Replace a portion of a text string with another text string.

    Args:
        run_context: RunContext object for file operations
        old_text: Original text or file path
        start_num: Starting position (1-based)
        num_chars: Number of characters to replace
        new_text: New text

    Returns:
        str: Modified text string

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If text processing fails

    Example:
        >>> REPLACE(ctx, "Financial Planning", start_num=11, num_chars=8, new_text="Analysis")
        'Financial Analysis'
        >>> REPLACE(ctx, "AAPL-2023-Q1", start_num=6, num_chars=4, new_text="2024")
        'AAPL-2024-Q1'
    """
    # Handle file path input
    if isinstance(old_text, (str, Path)) and Path(str(old_text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, old_text)
            if len(df) > 0:
                old_text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_old_text = _validate_text_input(old_text, "REPLACE")
    validated_start_num = _validate_integer_input(start_num, "REPLACE", min_value=1)
    validated_num_chars = _validate_integer_input(num_chars, "REPLACE", min_value=0)
    validated_new_text = _validate_text_input(new_text, "REPLACE")

    try:
        # Convert to 0-based indexing
        start_index = validated_start_num - 1
        end_index = start_index + validated_num_chars

        # Replace the specified portion
        result = validated_old_text[:start_index] + validated_new_text + validated_old_text[end_index:]
        return result
    except Exception as e:
        raise DataQualityError(
            f"Error in REPLACE operation: {str(e)}",
            "Ensure positions are within text bounds"
        )


def SUBSTITUTE(run_context: Any, text: Union[str, Path], *, old_text: str, new_text: str, instance_num: int | None = None) -> str:
    """
    Replace occurrences of old text with new text.

    Args:
        run_context: RunContext object for file operations
        text: Original text or file path
        old_text: Text to replace
        new_text: New text
        instance_num: Instance number to replace (optional, replaces all if None)

    Returns:
        str: Modified text string

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If text processing fails

    Example:
        >>> SUBSTITUTE(ctx, "Financial Planning and Financial Analysis", old_text="Financial", new_text="Business")
        'Business Planning and Business Analysis'
        >>> SUBSTITUTE(ctx, "Q1-Q1-Q1", old_text="Q1", new_text="Q2", instance_num=2)
        'Q1-Q2-Q1'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_text = _validate_text_input(text, "SUBSTITUTE")
    validated_old_text = _validate_text_input(old_text, "SUBSTITUTE")
    validated_new_text = _validate_text_input(new_text, "SUBSTITUTE")

    if instance_num is not None:
        validated_instance_num = _validate_integer_input(instance_num, "SUBSTITUTE", min_value=1)

    try:
        if instance_num is None:
            # Replace all occurrences
            return validated_text.replace(validated_old_text, validated_new_text)
        else:
            # Replace specific instance
            parts = validated_text.split(validated_old_text)
            if len(parts) <= validated_instance_num:
                # Instance doesn't exist
                return validated_text

            # Reconstruct with replacement at specific instance
            result = validated_old_text.join(parts[:validated_instance_num])
            result += validated_new_text
            result += validated_old_text.join(parts[validated_instance_num:])
            return result
    except Exception as e:
        raise DataQualityError(
            f"Error in SUBSTITUTE operation: {str(e)}",
            "Ensure all text parameters are valid"
        )


def TRIM(run_context: Any, text: Union[str, Path]) -> str:
    """
    Remove extra spaces from text.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path

    Returns:
        str: Cleaned text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text processing fails

    Example:
        >>> TRIM(ctx, "  Extra   Spaces  ")
        'Extra Spaces'
        >>> TRIM(ctx, "  Financial Planning  ")
        'Financial Planning'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_text = _validate_text_input(text, "TRIM")

    try:
        # Remove leading/trailing spaces and collapse multiple spaces
        return re.sub(r'\s+', ' ', validated_text.strip())
    except Exception as e:
        raise DataQualityError(
            f"Error in TRIM operation: {str(e)}",
            "Ensure text is valid"
        )


def CLEAN(run_context: Any, text: Union[str, Path]) -> str:
    """
    Remove non-printable characters.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path

    Returns:
        str: Cleaned text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text processing fails

    Example:
        >>> CLEAN(ctx, "Financial\x00Planning\x01")
        'FinancialPlanning'
        >>> CLEAN(ctx, "Clean\tText\n")
        'Clean\tText\n'  # Keeps printable whitespace
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_text = _validate_text_input(text, "CLEAN")

    try:
        # Remove non-printable characters but keep normal whitespace
        cleaned = ""
        for char in validated_text:
            if unicodedata.category(char)[0] != 'C' or char in '\t\n\r ':
                cleaned += char
        return cleaned
    except Exception as e:
        raise DataQualityError(
            f"Error in CLEAN operation: {str(e)}",
            "Ensure text is valid"
        )


def UPPER(run_context: Any, text: Union[str, Path]) -> str:
    """
    Convert text to uppercase.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path

    Returns:
        str: Uppercase text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text processing fails

    Example:
        >>> UPPER(ctx, "hello world")
        'HELLO WORLD'
        >>> UPPER(ctx, "Financial Planning")
        'FINANCIAL PLANNING'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_text = _validate_text_input(text, "UPPER")

    try:
        return validated_text.upper()
    except Exception as e:
        raise DataQualityError(
            f"Error in UPPER operation: {str(e)}",
            "Ensure text is valid"
        )


def LOWER(run_context: Any, text: Union[str, Path]) -> str:
    """
    Convert text to lowercase.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path

    Returns:
        str: Lowercase text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text processing fails

    Example:
        >>> LOWER(ctx, "HELLO WORLD")
        'hello world'
        >>> LOWER(ctx, "Financial Planning")
        'financial planning'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_text = _validate_text_input(text, "LOWER")

    try:
        return validated_text.lower()
    except Exception as e:
        raise DataQualityError(
            f"Error in LOWER operation: {str(e)}",
            "Ensure text is valid"
        )


def PROPER(run_context: Any, text: Union[str, Path]) -> str:
    """
    Convert text to proper case.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path

    Returns:
        str: Proper case text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text processing fails

    Example:
        >>> PROPER(ctx, "hello world")
        'Hello World'
        >>> PROPER(ctx, "financial planning")
        'Financial Planning'
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_text = _validate_text_input(text, "PROPER")

    try:
        return validated_text.title()
    except Exception as e:
        raise DataQualityError(
            f"Error in PROPER operation: {str(e)}",
            "Ensure text is valid"
        )


def VALUE(run_context: Any, text: Union[str, Path]) -> Decimal:
    """
    Convert text to number using Decimal precision.

    Args:
        run_context: RunContext object for file operations
        text: Text string or file path

    Returns:
        Decimal: Numeric value

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text cannot be converted to number

    Financial Examples:
        >>> VALUE(ctx, "123.45")
        Decimal('123.45')
        >>> VALUE(ctx, "$1,234.56")
        Decimal('1234.56')
        >>> VALUE(ctx, "12.5%")
        Decimal('0.125')
        >>> VALUE(ctx, "(500)")  # Negative in parentheses
        Decimal('-500')
    """
    # Handle file path input
    if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
        try:
            df = load_df(run_context, text)
            if len(df) > 0:
                text = df[df.columns[0]][0]
        except FileNotFoundError:
            pass

    validated_text = _validate_text_input(text, "VALUE")

    try:
        # Clean the text for numeric conversion
        cleaned_text = validated_text.strip()

        # Handle percentage
        if cleaned_text.endswith('%'):
            numeric_part = cleaned_text[:-1].strip()
            return Decimal(numeric_part) / 100

        # Handle currency symbols
        cleaned_text = re.sub(r'[$€£¥]', '', cleaned_text)

        # Handle thousands separators
        cleaned_text = cleaned_text.replace(',', '')

        # Handle negative numbers in parentheses
        if cleaned_text.startswith('(') and cleaned_text.endswith(')'):
            cleaned_text = '-' + cleaned_text[1:-1]

        # Convert to Decimal
        return Decimal(cleaned_text)

    except (ValueError, TypeError, Exception) as e:
        raise DataQualityError(
            f"Cannot convert text to number: {str(e)}",
            "Ensure text represents a valid number"
        )


def TEXTJOIN(run_context: Any, delimiter: str, ignore_empty: bool, *texts: Union[str, int, float, Path]) -> str:
    """
    Join text strings with delimiter.

    Args:
        run_context: RunContext object for file operations
        delimiter: Delimiter string
        ignore_empty: Ignore empty values
        texts: Text strings to join (supports file paths)

    Returns:
        str: Combined text string

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If text processing fails

    Example:
        >>> TEXTJOIN(ctx, ", ", True, "Apple", "", "Banana", "Cherry")
        'Apple, Banana, Cherry'
        >>> TEXTJOIN(ctx, " | ", False, "Q1", "Q2", "Q3", "Q4")
        'Q1 | Q2 | Q3 | Q4'
        >>> TEXTJOIN(ctx, ",", True, "data.csv")  # File input
        'value1,value2,value3'
    """
    if not texts:
        return ""

    try:
        validated_delimiter = _validate_text_input(delimiter, "TEXTJOIN")
        text_parts = []

        for text in texts:
            # Handle file path input
            if isinstance(text, (str, Path)) and Path(str(text)).suffix in ['.csv', '.parquet']:
                try:
                    df = load_df(run_context, text)
                    # Join all values from first column
                    if len(df) > 0:
                        first_col = df[df.columns[0]]
                        file_values = [_validate_text_input(val, "TEXTJOIN") for val in first_col.to_list()]
                        text_parts.extend(file_values)
                    continue
                except FileNotFoundError:
                    # If file not found, treat as regular text
                    pass

            # Regular text processing
            validated_text = _validate_text_input(text, "TEXTJOIN")

            # Handle empty values based on ignore_empty flag
            if ignore_empty and not validated_text.strip():
                continue

            text_parts.append(validated_text)

        return validated_delimiter.join(text_parts)

    except Exception as e:
        raise DataQualityError(
            f"Error in TEXTJOIN operation: {str(e)}",
            "Ensure all inputs are valid text, numbers, or file paths"
        )
