"""Enhanced Range Resolver with Polars DataFrame integration

This module provides unified range handling between Google Sheets A1 notation
and Polars DataFrame operations, enabling seamless translation between the two.
"""
import re
from typing import Union, Tuple, Optional, List, Dict, Any
import polars as pl
from pathlib import Path


class PolarsRangeResolver:
    """Unified range handling for Polars DataFrames and Google Sheets"""
    
    @staticmethod
    def sheets_to_polars(range_spec: str) -> Dict[str, Any]:
        """
        Convert Google Sheets A1 notation to Polars slice indices.
        
        Args:
            range_spec: A1 notation range (e.g., "A1:C10", "B:B", "5:10")
            
        Returns:
            Dictionary with row and column slice information
            
        Examples:
            "A1:C10" -> {"row_slice": slice(0, 10), "col_slice": slice(0, 3)}
            "B:B" -> {"row_slice": slice(None), "col_slice": slice(1, 2)}
            "5:10" -> {"row_slice": slice(4, 10), "col_slice": slice(None)}
        """
        # Handle full column references (e.g., "A:C")
        col_pattern = r'^([A-Z]+):([A-Z]+)$'
        col_match = re.match(col_pattern, range_spec)
        if col_match:
            start_col = PolarsRangeResolver._col_to_index(col_match.group(1))
            end_col = PolarsRangeResolver._col_to_index(col_match.group(2)) + 1
            return {
                "row_slice": slice(None),
                "col_slice": slice(start_col, end_col),
                "type": "column_range"
            }
        
        # Handle full row references (e.g., "5:10")
        row_pattern = r'^(\d+):(\d+)$'
        row_match = re.match(row_pattern, range_spec)
        if row_match:
            start_row = int(row_match.group(1)) - 1  # Convert to 0-based
            end_row = int(row_match.group(2))
            return {
                "row_slice": slice(start_row, end_row),
                "col_slice": slice(None),
                "type": "row_range"
            }
        
        # Handle standard cell ranges (e.g., "A1:C10")
        cell_pattern = r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$'
        cell_match = re.match(cell_pattern, range_spec)
        if cell_match:
            start_col = PolarsRangeResolver._col_to_index(cell_match.group(1))
            start_row = int(cell_match.group(2)) - 1
            end_col = PolarsRangeResolver._col_to_index(cell_match.group(3)) + 1
            end_row = int(cell_match.group(4))
            return {
                "row_slice": slice(start_row, end_row),
                "col_slice": slice(start_col, end_col),
                "type": "cell_range"
            }
        
        # Handle single cell (e.g., "A1")
        single_cell_pattern = r'^([A-Z]+)(\d+)$'
        single_match = re.match(single_cell_pattern, range_spec)
        if single_match:
            col = PolarsRangeResolver._col_to_index(single_match.group(1))
            row = int(single_match.group(2)) - 1
            return {
                "row_slice": slice(row, row + 1),
                "col_slice": slice(col, col + 1),
                "type": "single_cell"
            }
        
        raise ValueError(f"Invalid range specification: {range_spec}")
    
    @staticmethod
    def polars_to_sheets(df: pl.DataFrame, 
                        row_slice: Optional[slice] = None, 
                        col_slice: Optional[slice] = None) -> str:
        """
        Convert Polars DataFrame slice to A1 notation.
        
        Args:
            df: Polars DataFrame
            row_slice: Row slice specification
            col_slice: Column slice specification
            
        Returns:
            A1 notation range string
        """
        # Determine actual row range
        if row_slice is None:
            start_row = 1
            end_row = len(df)
        else:
            start_row = (row_slice.start or 0) + 1
            end_row = row_slice.stop if row_slice.stop is not None else len(df)
        
        # Determine actual column range
        if col_slice is None:
            start_col = 0
            end_col = len(df.columns)
        else:
            start_col = col_slice.start or 0
            end_col = col_slice.stop if col_slice.stop is not None else len(df.columns)
        
        # Convert to A1 notation
        start_col_letter = PolarsRangeResolver._index_to_col(start_col)
        end_col_letter = PolarsRangeResolver._index_to_col(end_col - 1)
        
        return f"{start_col_letter}{start_row}:{end_col_letter}{end_row}"
    
    @staticmethod
    def resolve_range(df: pl.DataFrame, 
                     range_spec: Optional[Union[str, Dict[str, Any]]] = None) -> pl.DataFrame:
        """
        Apply range specification to DataFrame.
        
        Args:
            df: Input Polars DataFrame
            range_spec: Range specification (A1 notation string or dict)
            
        Returns:
            Sliced DataFrame according to range specification
        """
        if range_spec is None:
            return df
        
        # Handle string range specs (A1 notation)
        if isinstance(range_spec, str):
            range_info = PolarsRangeResolver.sheets_to_polars(range_spec)
        else:
            range_info = range_spec
        
        # Apply row slicing
        if range_info.get("row_slice") is not None:
            row_slice = range_info["row_slice"]
            if row_slice.start is not None or row_slice.stop is not None:
                start = row_slice.start or 0
                stop = row_slice.stop or len(df)
                df = df.slice(start, stop - start)
        
        # Apply column slicing
        if range_info.get("col_slice") is not None:
            col_slice = range_info["col_slice"]
            if col_slice.start is not None or col_slice.stop is not None:
                start = col_slice.start or 0
                stop = col_slice.stop or len(df.columns)
                columns = df.columns[start:stop]
                df = df.select(columns)
        
        # Handle column names if specified
        if "columns" in range_info:
            df = df.select(range_info["columns"])
        
        return df
    
    @staticmethod
    def create_range_spec(
        rows: Optional[Union[int, Tuple[int, int], slice]] = None,
        columns: Optional[Union[str, List[str], Tuple[str, str], slice]] = None
    ) -> Dict[str, Any]:
        """
        Create a range specification from flexible input formats.
        
        Args:
            rows: Row specification (single int, tuple of ints, or slice)
            columns: Column specification (single name, list of names, tuple, or slice)
            
        Returns:
            Range specification dictionary
        """
        spec = {}
        
        # Handle row specification
        if rows is not None:
            if isinstance(rows, int):
                spec["row_slice"] = slice(rows, rows + 1)
            elif isinstance(rows, tuple) and len(rows) == 2:
                spec["row_slice"] = slice(rows[0], rows[1])
            elif isinstance(rows, slice):
                spec["row_slice"] = rows
            else:
                raise ValueError(f"Invalid row specification: {rows}")
        
        # Handle column specification
        if columns is not None:
            if isinstance(columns, str):
                spec["columns"] = [columns]
            elif isinstance(columns, list):
                spec["columns"] = columns
            elif isinstance(columns, tuple) and len(columns) == 2:
                # Assume these are column indices or letters
                if isinstance(columns[0], str):
                    start_idx = PolarsRangeResolver._col_to_index(columns[0])
                    end_idx = PolarsRangeResolver._col_to_index(columns[1]) + 1
                else:
                    start_idx = columns[0]
                    end_idx = columns[1] + 1
                spec["col_slice"] = slice(start_idx, end_idx)
            elif isinstance(columns, slice):
                spec["col_slice"] = columns
            else:
                raise ValueError(f"Invalid column specification: {columns}")
        
        return spec
    
    @staticmethod
    def _col_to_index(col_letter: str) -> int:
        """Convert column letter(s) to 0-based index"""
        result = 0
        for char in col_letter:
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result - 1
    
    @staticmethod
    def _index_to_col(index: int) -> str:
        """Convert 0-based index to column letter(s)"""
        result = ""
        index += 1  # Convert to 1-based
        while index > 0:
            index -= 1
            result = chr(index % 26 + ord('A')) + result
            index //= 26
        return result
    
    @staticmethod
    def expand_range_for_operation(
        df: pl.DataFrame,
        range_spec: Optional[Union[str, Dict[str, Any]]],
        operation_type: str
    ) -> Dict[str, Any]:
        """
        Expand or adjust range based on operation type.
        
        Args:
            df: Input DataFrame
            range_spec: Original range specification
            operation_type: Type of operation (e.g., "sum", "pivot", "filter")
            
        Returns:
            Adjusted range specification for the operation
        """
        if range_spec is None:
            return {"row_slice": slice(None), "col_slice": slice(None)}
        
        # Parse range if string
        if isinstance(range_spec, str):
            range_info = PolarsRangeResolver.sheets_to_polars(range_spec)
        else:
            range_info = range_spec.copy()
        
        # Adjust based on operation type
        if operation_type in ["sum", "average", "count"]:
            # Aggregation operations typically work on full columns
            if range_info.get("type") == "single_cell":
                # Expand single cell to column
                col_idx = range_info["col_slice"].start
                range_info["col_slice"] = slice(col_idx, col_idx + 1)
                range_info["row_slice"] = slice(None)
        
        elif operation_type == "pivot":
            # Pivot operations need full data range
            if range_info.get("type") == "single_cell":
                # Can't pivot on single cell
                raise ValueError("Cannot perform pivot operation on single cell")
        
        elif operation_type == "filter":
            # Filter operations typically apply to all columns
            if range_info.get("col_slice") and range_info["col_slice"].stop:
                # Keep column restriction but ensure we have all necessary columns
                pass
        
        return range_info
    
    @staticmethod
    def validate_range_for_dataframe(
        df: pl.DataFrame,
        range_spec: Union[str, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a range specification is valid for the given DataFrame.
        
        Args:
            df: DataFrame to validate against
            range_spec: Range specification to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if isinstance(range_spec, str):
                range_info = PolarsRangeResolver.sheets_to_polars(range_spec)
            else:
                range_info = range_spec
            
            # Check row bounds
            if range_info.get("row_slice"):
                row_slice = range_info["row_slice"]
                if row_slice.stop and row_slice.stop > len(df):
                    return False, f"Row range exceeds DataFrame size ({len(df)} rows)"
            
            # Check column bounds
            if range_info.get("col_slice"):
                col_slice = range_info["col_slice"]
                if col_slice.stop and col_slice.stop > len(df.columns):
                    return False, f"Column range exceeds DataFrame size ({len(df.columns)} columns)"
            
            # Check column names
            if range_info.get("columns"):
                missing_cols = set(range_info["columns"]) - set(df.columns)
                if missing_cols:
                    return False, f"Columns not found in DataFrame: {missing_cols}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)