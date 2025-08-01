"""
Conditional Aggregation & Counting Functions

These functions allow you to work with data subsets based on specific criteria.
"""

from typing import Any


def SUMIF(range_to_evaluate: list[Any], criteria: Any, sum_range: list[Any] | None = None) -> float:
    """
    Sum numbers that meet one condition.

    Args:
        range_to_evaluate: Range to evaluate
        criteria: Criteria to match
        sum_range: Sum range (optional)

    Returns:
        Single numeric value

    Example:
        SUMIF(A1:A10, ">100", B1:B10)
    """
    raise NotImplementedError("SUMIF function not yet implemented")


def SUMIFS(sum_range: list[Any], criteria_ranges: list[list[Any]], criteria_values: list[Any]) -> float:
    """
    Sum numbers that meet multiple conditions.

    Args:
        sum_range: Sum range
        criteria_ranges: Criteria ranges
        criteria_values: Criteria values

    Returns:
        Single numeric value

    Example:
        SUMIFS(C1:C10, [A1:A10, B1:B10], [">100", "Sales"])
    """
    raise NotImplementedError("SUMIFS function not yet implemented")


def COUNTIF(range_to_evaluate: list[Any], criteria: Any) -> int:
    """
    Count cells that meet one condition.

    Args:
        range_to_evaluate: Range to evaluate
        criteria: Criteria to match

    Returns:
        Integer count

    Example:
        COUNTIF(A1:A10, ">100")
    """
    raise NotImplementedError("COUNTIF function not yet implemented")


def COUNTIFS(criteria_ranges: list[list[Any]], criteria_values: list[Any]) -> int:
    """
    Count cells that meet multiple conditions.

    Args:
        criteria_ranges: Criteria ranges
        criteria_values: Criteria values (pairs)

    Returns:
        Integer count

    Example:
        COUNTIFS([A1:A10, B1:B10], [">100", "Sales"])
    """
    raise NotImplementedError("COUNTIFS function not yet implemented")


def AVERAGEIF(range_to_evaluate: list[Any], criteria: Any, average_range: list[Any] | None = None) -> float:
    """
    Calculate average of cells that meet one condition.

    Args:
        range_to_evaluate: Range to evaluate
        criteria: Criteria to match
        average_range: Average range (optional)

    Returns:
        Single numeric value

    Example:
        AVERAGEIF(A1:A10, ">100", B1:B10)
    """
    raise NotImplementedError("AVERAGEIF function not yet implemented")


def AVERAGEIFS(average_range: list[Any], criteria_ranges: list[list[Any]], criteria_values: list[Any]) -> float:
    """
    Calculate average of cells that meet multiple conditions.

    Args:
        average_range: Average range
        criteria_ranges: Criteria ranges
        criteria_values: Criteria values

    Returns:
        Single numeric value

    Example:
        AVERAGEIFS(C1:C10, [A1:A10, B1:B10], [">100", "Sales"])
    """
    raise NotImplementedError("AVERAGEIFS function not yet implemented")


def MAXIFS(max_range: list[Any], criteria_ranges: list[list[Any]], criteria_values: list[Any]) -> float:
    """
    Find maximum value based on multiple criteria.

    Args:
        max_range: Max range
        criteria_ranges: Criteria ranges
        criteria_values: Criteria values

    Returns:
        Single numeric value

    Example:
        MAXIFS(C1:C10, [A1:A10, B1:B10], [">100", "Sales"])
    """
    raise NotImplementedError("MAXIFS function not yet implemented")


def MINIFS(min_range: list[Any], criteria_ranges: list[list[Any]], criteria_values: list[Any]) -> float:
    """
    Find minimum value based on multiple criteria.

    Args:
        min_range: Min range
        criteria_ranges: Criteria ranges
        criteria_values: Criteria values

    Returns:
        Single numeric value

    Example:
        MINIFS(C1:C10, [A1:A10, B1:B10], [">100", "Sales"])
    """
    raise NotImplementedError("MINIFS function not yet implemented")


def SUMPRODUCT(range1: list[Any], range2: list[Any], *additional_ranges: list[Any]) -> float:
    """
    Sum the products of corresponding ranges.

    Args:
        range1: First range
        range2: Second range
        additional_ranges: Additional ranges (optional)

    Returns:
        Single numeric value

    Example:
        SUMPRODUCT(A1:A10, B1:B10)
    """
    raise NotImplementedError("SUMPRODUCT function not yet implemented")


def AGGREGATE(function_num: int, options: int, array: list[Any], k: Any | None = None) -> float:
    """
    Perform various aggregations with error handling and filtering.

    Args:
        function_num: Function number
        options: Options
        array: Array to aggregate
        k: Additional parameter (optional)

    Returns:
        Single numeric value

    Example:
        AGGREGATE(1, 5, A1:A10)  # Sum ignoring errors
    """
    raise NotImplementedError("AGGREGATE function not yet implemented")


def SUBTOTAL(function_num: int, ref1: list[Any], *additional_refs: list[Any]) -> float:
    """
    Calculate subtotals with filtering capability.

    Args:
        function_num: Function number
        ref1: First reference
        additional_refs: Additional references (optional)

    Returns:
        Single numeric value

    Example:
        SUBTOTAL(109, A1:A10)  # Sum of visible cells
    """
    raise NotImplementedError("SUBTOTAL function not yet implemented")


def COUNTBLANK(range_to_evaluate: list[Any]) -> int:
    """
    Count blank/empty cells in a range.

    Args:
        range_to_evaluate: Range to evaluate

    Returns:
        Integer count

    Example:
        COUNTBLANK(A1:A10)
    """
    raise NotImplementedError("COUNTBLANK function not yet implemented")


def COUNTA(range_to_evaluate: list[Any]) -> int:
    """
    Count non-empty cells in a range.

    Args:
        range_to_evaluate: Range to evaluate

    Returns:
        Integer count

    Example:
        COUNTA(A1:A10)
    """
    raise NotImplementedError("COUNTA function not yet implemented")
