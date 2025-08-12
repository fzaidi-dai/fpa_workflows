"""Range resolver for A1 notation and Polars integration"""
import re
import polars as pl
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass

@dataclass
class ParsedRange:
    """Parsed A1 notation range"""
    sheet_name: Optional[str]
    start_col: int  # 0-based
    start_row: int  # 0-based  
    end_col: Optional[int]
    end_row: Optional[int]

class RangeResolver:
    """Handle A1 notation conversion for Google Sheets"""
    
    @staticmethod
    def column_letter_to_index(column: str) -> int:
        """Convert column letter to 0-based index (A=0, B=1, etc.)"""
        result = 0
        for char in column:
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result - 1
    
    @staticmethod
    def index_to_column_letter(index: int) -> str:
        """Convert 0-based index to column letter"""
        result = ""
        index += 1
        while index > 0:
            index -= 1
            result = chr(index % 26 + ord('A')) + result
            index //= 26
        return result
    
    @staticmethod
    def parse_a1_notation(range_string: str) -> ParsedRange:
        """Parse A1 notation like 'Sheet1!A1:C10' or 'A2:E'"""
        if not range_string or not isinstance(range_string, str):
            raise ValueError(f"Invalid range specification: {range_string}")
        
        # Handle sheet name
        sheet_name = None
        if '!' in range_string:
            sheet_part, range_part = range_string.split('!', 1)
            sheet_name = sheet_part.strip("'")
        else:
            range_part = range_string
        
        if not range_part:
            raise ValueError(f"Invalid range specification: {range_string}")
        
        # Parse range part (e.g., "A1:C10" or "A1")
        if ':' in range_part:
            start, end = range_part.split(':')
        else:
            start = end = range_part
        
        # Parse start position
        match = re.match(r'([A-Z]+)(\d+)?', start.upper())
        if not match:
            raise ValueError(f"Invalid A1 notation: {range_string}")
        
        start_col = RangeResolver.column_letter_to_index(match.group(1))
        start_row = int(match.group(2)) - 1 if match.group(2) else 0
        
        # Parse end position
        if end != start:
            match = re.match(r'([A-Z]+)(\d+)?', end.upper())
            if not match:
                raise ValueError(f"Invalid A1 notation: {range_string}")
            end_col = RangeResolver.column_letter_to_index(match.group(1))
            end_row = int(match.group(2)) - 1 if match.group(2) else None
        else:
            end_col = start_col
            end_row = start_row
        
        return ParsedRange(sheet_name, start_col, start_row, end_col, end_row)
    
    @staticmethod
    def dataframe_to_a1_range(df: pl.DataFrame, 
                             start_cell: str = "A1",
                             sheet_name: Optional[str] = None) -> str:
        """Convert DataFrame dimensions to A1 range notation"""
        parsed = RangeResolver.parse_a1_notation(start_cell)
        
        end_col = parsed.start_col + len(df.columns) - 1
        end_row = parsed.start_row + len(df) - 1
        
        start_notation = f"{RangeResolver.index_to_column_letter(parsed.start_col)}{parsed.start_row + 1}"
        end_notation = f"{RangeResolver.index_to_column_letter(end_col)}{end_row + 1}"
        
        range_str = f"{start_notation}:{end_notation}"
        
        if sheet_name:
            if ' ' in sheet_name:
                sheet_name = f"'{sheet_name}'"
            range_str = f"{sheet_name}!{range_str}"
        
        return range_str
    
    @staticmethod
    def apply_range_to_dataframe(df: pl.DataFrame, range_spec: str) -> pl.DataFrame:
        """Apply Google Sheets range to Polars DataFrame"""
        parsed = RangeResolver.parse_a1_notation(range_spec)
        
        # Apply row slicing
        if parsed.end_row is not None:
            df = df[parsed.start_row:parsed.end_row + 1]
        else:
            df = df[parsed.start_row:]
        
        # Apply column slicing
        cols = df.columns
        if parsed.end_col is not None:
            selected_cols = cols[parsed.start_col:min(parsed.end_col + 1, len(cols))]
        else:
            selected_cols = cols[parsed.start_col:]
        
        return df.select(selected_cols)
    
    @staticmethod
    def expand_range(range_spec: str, rows: int, cols: int) -> str:
        """Expand a range by specified rows and columns"""
        parsed = RangeResolver.parse_a1_notation(range_spec)
        
        # Calculate new end position
        new_end_row = (parsed.end_row if parsed.end_row is not None else parsed.start_row) + rows
        new_end_col = (parsed.end_col if parsed.end_col is not None else parsed.start_col) + cols
        
        # Build new range string
        start = f"{RangeResolver.index_to_column_letter(parsed.start_col)}{parsed.start_row + 1}"
        end = f"{RangeResolver.index_to_column_letter(new_end_col)}{new_end_row + 1}"
        
        range_str = f"{start}:{end}"
        if parsed.sheet_name:
            if ' ' in parsed.sheet_name:
                sheet_name = f"'{parsed.sheet_name}'"
            else:
                sheet_name = parsed.sheet_name
            range_str = f"{sheet_name}!{range_str}"
        
        return range_str
    
    @staticmethod
    def get_range_dimensions(range_spec: str) -> Tuple[int, int]:
        """Get the number of rows and columns in a range"""
        parsed = RangeResolver.parse_a1_notation(range_spec)
        
        if parsed.end_row is not None:
            rows = parsed.end_row - parsed.start_row + 1
        else:
            rows = -1  # Unbounded
        
        if parsed.end_col is not None:
            cols = parsed.end_col - parsed.start_col + 1
        else:
            cols = -1  # Unbounded
        
        return (rows, cols)