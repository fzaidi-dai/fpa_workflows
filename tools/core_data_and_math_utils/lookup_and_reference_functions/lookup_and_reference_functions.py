"""
Lookup & Reference Functions

These are invaluable when you need to retrieve data from a table or array dynamically.
"""

from typing import Any


def VLOOKUP(lookup_value: Any, table_array: list[list[Any]], col_index: int, range_lookup: bool | None = None) -> Any:
    """
    Search for a value in a vertical range.

    Args:
        lookup_value: Value to look up
        table_array: Table array to search in
        col_index: Column index to return
        range_lookup: Range lookup (optional)

    Returns:
        Single value from specified column

    Example:
        VLOOKUP(lookup_value, table_array, col_index, [range_lookup])
    """
    raise NotImplementedError("VLOOKUP function not yet implemented")


def HLOOKUP(lookup_value: Any, table_array: list[list[Any]], row_index: int, range_lookup: bool | None = None) -> Any:
    """
    Search for a value in a horizontal range.

    Args:
        lookup_value: Value to look up
        table_array: Table array to search in
        row_index: Row index to return
        range_lookup: Range lookup (optional)

    Returns:
        Single value from specified row

    Example:
        HLOOKUP(lookup_value, table_array, row_index, [range_lookup])
    """
    raise NotImplementedError("HLOOKUP function not yet implemented")


def INDEX(array: list[list[Any]], row_num: int, column_num: int | None = None) -> Any:
    """
    Return a value at a given position in an array.

    Args:
        array: Array to search in
        row_num: Row number
        column_num: Column number (optional)

    Returns:
        Single value at specified position

    Example:
        INDEX(return_range, row_num, [column_num])
    """
    raise NotImplementedError("INDEX function not yet implemented")


def MATCH(lookup_value: Any, lookup_array: list[Any], match_type: int) -> int:
    """
    Find the relative position of an item in an array.

    Args:
        lookup_value: Value to look up
        lookup_array: Array to search in
        match_type: Match type

    Returns:
        Integer position

    Example:
        MATCH(lookup_value, lookup_range, 0)
    """
    raise NotImplementedError("MATCH function not yet implemented")


def XLOOKUP(lookup_value: Any, lookup_array: list[Any], return_array: list[Any], if_not_found: Any | None = None) -> Any:
    """
    Modern, flexible lookup function replacing VLOOKUP/HLOOKUP.

    Args:
        lookup_value: Value to look up
        lookup_array: Array to search in
        return_array: Array to return values from
        if_not_found: Value to return if not found (optional)

    Returns:
        Value from return array or if_not_found value

    Example:
        XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found])
    """
    raise NotImplementedError("XLOOKUP function not yet implemented")


def OFFSET(reference: Any, rows: int, cols: int, height: int | None = None, width: int | None = None) -> Any:
    """
    Create dynamic ranges based on reference point.

    Args:
        reference: Reference point
        rows: Number of rows to offset
        cols: Number of columns to offset
        height: Height (optional)
        width: Width (optional)

    Returns:
        Range reference

    Example:
        OFFSET(reference, rows, cols, [height], [width])
    """
    raise NotImplementedError("OFFSET function not yet implemented")


def INDIRECT(ref_text: str, a1_style: bool | None = None) -> Any:
    """
    Create references based on text strings.

    Args:
        ref_text: Reference text
        a1_style: A1 style (optional)

    Returns:
        Range reference

    Example:
        INDIRECT(ref_text)
    """
    raise NotImplementedError("INDIRECT function not yet implemented")


def CHOOSE(index_num: int, *values: Any) -> Any:
    """
    Return a value from a list based on index number.

    Args:
        index_num: Index number
        values: Values to choose from

    Returns:
        Selected value

    Example:
        CHOOSE(index_num, value1, value2, …)
    """
    raise NotImplementedError("CHOOSE function not yet implemented")


def LOOKUP(lookup_value: Any, lookup_vector: list[Any], result_vector: list[Any] | None = None) -> Any:
    """
    Simple lookup function (vector or array form).

    Args:
        lookup_value: Value to look up
        lookup_vector: Lookup vector
        result_vector: Result vector (optional)

    Returns:
        Single value

    Example:
        LOOKUP(lookup_value, lookup_vector, result_vector)
    """
    raise NotImplementedError("LOOKUP function not yet implemented")


def ADDRESS(row_num: int, column_num: int, abs_num: int | None = None, a1: bool | None = None, sheet_text: str | None = None) -> str:
    """
    Create cell address as text.

    Args:
        row_num: Row number
        column_num: Column number
        abs_num: Absolute number (optional)
        a1: A1 style (optional)
        sheet_text: Sheet text (optional)

    Returns:
        Text string (cell address)

    Example:
        ADDRESS(1, 1, 1, TRUE, "Sheet1")
    """
    raise NotImplementedError("ADDRESS function not yet implemented")


def ROW(reference: Any | None = None) -> int | list[int]:
    """
    Return row number of reference.

    Args:
        reference: Reference (optional)

    Returns:
        Integer or array of integers

    Example:
        ROW(A5)
    """
    raise NotImplementedError("ROW function not yet implemented")


def COLUMN(reference: Any | None = None) -> int | list[int]:
    """
    Return column number of reference.

    Args:
        reference: Reference (optional)

    Returns:
        Integer or array of integers

    Example:
        COLUMN(B1)
    """
    raise NotImplementedError("COLUMN function not yet implemented")


def ROWS(array: list[list[Any]]) -> int:
    """
    Return number of rows in reference.

    Args:
        array: Array to evaluate

    Returns:
        Integer

    Example:
        ROWS(A1:A10)
    """
    raise NotImplementedError("ROWS function not yet implemented")


def COLUMNS(array: list[list[Any]]) -> int:
    """
    Return number of columns in reference.

    Args:
        array: Array to evaluate

    Returns:
        Integer

    Example:
        COLUMNS(A1:E1)
    """
    raise NotImplementedError("COLUMNS function not yet implemented")
