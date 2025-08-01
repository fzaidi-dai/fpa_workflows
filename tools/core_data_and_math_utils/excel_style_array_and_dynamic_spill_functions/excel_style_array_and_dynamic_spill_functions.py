"""
Array and Dynamic Spill Functions (Modern Excel)

These functions help in performing calculations across ranges and enabling dynamic results.
"""

from typing import Any, List


def UNIQUE(array: List[Any], by_col: bool | None = None, exactly_once: bool | None = None) -> List[Any]:
    """
    Extract a list of unique values from a range.

    Args:
        array: Array to process
        by_col: Process by column (optional)
        exactly_once: Return only values that appear exactly once (optional)

    Returns:
        Array of unique values

    Example:
        UNIQUE(range)
    """
    raise NotImplementedError("UNIQUE function not yet implemented")


def SORT(array: List[Any], sort_index: int | None = None, sort_order: int | None = None, by_col: bool | None = None) -> List[Any]:
    """
    Sort data or arrays dynamically.

    Args:
        array: Array to sort
        sort_index: Sort index (optional)
        sort_order: Sort order (optional)
        by_col: Sort by column (optional)

    Returns:
        Sorted array

    Example:
        SORT(range)
    """
    raise NotImplementedError("SORT function not yet implemented")


def SORTBY(array: List[Any], *by_arrays_and_orders: Any) -> List[Any]:
    """
    Sort an array by values in another array.

    Args:
        array: Array to sort
        by_arrays_and_orders: Arrays and sort orders (alternating)

    Returns:
        Sorted array

    Example:
        SORTBY(array, by_array)
    """
    raise NotImplementedError("SORTBY function not yet implemented")


def FILTER(array: List[Any], include: Any, if_empty: Any | None = None) -> List[Any]:
    """
    Return only those records that meet specified conditions.

    Args:
        array: Array to filter
        include: Include condition
        if_empty: Value if empty (optional)

    Returns:
        Filtered array

    Example:
        FILTER(range, condition)
    """
    raise NotImplementedError("FILTER function not yet implemented")


def SEQUENCE(rows: int, columns: int | None = None, start: int | None = None, step: int | None = None) -> List[List[int]]:
    """
    Generate a list of sequential numbers in an array format.

    Args:
        rows: Number of rows
        columns: Number of columns (optional)
        start: Starting number (optional)
        step: Step size (optional)

    Returns:
        Array of sequential numbers

    Example:
        SEQUENCE(rows, [columns], [start], [step])
    """
    raise NotImplementedError("SEQUENCE function not yet implemented")


def RAND() -> float:
    """
    Generate random numbers between 0 and 1.

    Args:
        No parameters

    Returns:
        Random decimal between 0 and 1

    Example:
        RAND()
    """
    raise NotImplementedError("RAND function not yet implemented")


def RANDBETWEEN(bottom: int, top: int) -> int:
    """
    Generate random integers between two values.

    Args:
        bottom: Lower bound
        top: Upper bound

    Returns:
        Random integer within range

    Example:
        RANDBETWEEN(lower, upper)
    """
    raise NotImplementedError("RANDBETWEEN function not yet implemented")


def FREQUENCY(data_array: List[float], bins_array: List[float]) -> List[int]:
    """
    Calculate frequency distribution.

    Args:
        data_array: Data array
        bins_array: Bins array

    Returns:
        Array of frequencies

    Example:
        FREQUENCY(A1:A100, C1:C10)
    """
    raise NotImplementedError("FREQUENCY function not yet implemented")


def TRANSPOSE(array: List[List[Any]]) -> List[List[Any]]:
    """
    Transpose array orientation.

    Args:
        array: Array to transpose

    Returns:
        Transposed array

    Example:
        TRANSPOSE(A1:E5)
    """
    raise NotImplementedError("TRANSPOSE function not yet implemented")


def MMULT(array1: List[List[float]], array2: List[List[float]]) -> List[List[float]]:
    """
    Matrix multiplication.

    Args:
        array1: First array
        array2: Second array

    Returns:
        Matrix product

    Example:
        MMULT(A1:B3, D1:E2)
    """
    raise NotImplementedError("MMULT function not yet implemented")


def MINVERSE(array: List[List[float]]) -> List[List[float]]:
    """
    Matrix inverse.

    Args:
        array: Array to invert

    Returns:
        Inverse matrix

    Example:
        MINVERSE(A1:B2)
    """
    raise NotImplementedError("MINVERSE function not yet implemented")


def MDETERM(array: List[List[float]]) -> float:
    """
    Matrix determinant.

    Args:
        array: Array to calculate determinant of

    Returns:
        Float (determinant)

    Example:
        MDETERM(A1:B2)
    """
    raise NotImplementedError("MDETERM function not yet implemented")
