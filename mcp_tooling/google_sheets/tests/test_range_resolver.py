"""Unit tests for RangeResolver module"""
import pytest
import polars as pl
from mcp_tooling.google_sheets.api.range_resolver import RangeResolver, ParsedRange

class TestRangeResolver:
    """Test cases for RangeResolver functionality"""
    
    def test_column_letter_to_index(self):
        """Test column letter to index conversion"""
        assert RangeResolver.column_letter_to_index('A') == 0
        assert RangeResolver.column_letter_to_index('B') == 1
        assert RangeResolver.column_letter_to_index('Z') == 25
        assert RangeResolver.column_letter_to_index('AA') == 26
        assert RangeResolver.column_letter_to_index('AB') == 27
        assert RangeResolver.column_letter_to_index('AZ') == 51
        assert RangeResolver.column_letter_to_index('BA') == 52
    
    def test_index_to_column_letter(self):
        """Test index to column letter conversion"""
        assert RangeResolver.index_to_column_letter(0) == 'A'
        assert RangeResolver.index_to_column_letter(1) == 'B'
        assert RangeResolver.index_to_column_letter(25) == 'Z'
        assert RangeResolver.index_to_column_letter(26) == 'AA'
        assert RangeResolver.index_to_column_letter(27) == 'AB'
        assert RangeResolver.index_to_column_letter(51) == 'AZ'
        assert RangeResolver.index_to_column_letter(52) == 'BA'
    
    def test_parse_a1_notation_simple(self):
        """Test parsing simple A1 notation"""
        parsed = RangeResolver.parse_a1_notation('A1')
        assert parsed.sheet_name is None
        assert parsed.start_col == 0
        assert parsed.start_row == 0
        assert parsed.end_col == 0
        assert parsed.end_row == 0
    
    def test_parse_a1_notation_range(self):
        """Test parsing A1 notation ranges"""
        parsed = RangeResolver.parse_a1_notation('A1:C10')
        assert parsed.sheet_name is None
        assert parsed.start_col == 0
        assert parsed.start_row == 0
        assert parsed.end_col == 2
        assert parsed.end_row == 9
    
    def test_parse_a1_notation_with_sheet(self):
        """Test parsing A1 notation with sheet names"""
        parsed = RangeResolver.parse_a1_notation('Sheet1!A1:C10')
        assert parsed.sheet_name == 'Sheet1'
        assert parsed.start_col == 0
        assert parsed.start_row == 0
        assert parsed.end_col == 2
        assert parsed.end_row == 9
    
    def test_parse_a1_notation_quoted_sheet(self):
        """Test parsing A1 notation with quoted sheet names"""
        parsed = RangeResolver.parse_a1_notation("'My Sheet'!A1:C10")
        assert parsed.sheet_name == 'My Sheet'
        assert parsed.start_col == 0
        assert parsed.start_row == 0
        assert parsed.end_col == 2
        assert parsed.end_row == 9
    
    def test_dataframe_to_a1_range(self):
        """Test converting DataFrame dimensions to A1 range"""
        # Create a test DataFrame
        df = pl.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        
        # Test default start cell
        result = RangeResolver.dataframe_to_a1_range(df)
        assert result == 'A1:C3'
        
        # Test with sheet name
        result = RangeResolver.dataframe_to_a1_range(df, sheet_name='Data')
        assert result == 'Data!A1:C3'
        
        # Test with sheet name containing spaces
        result = RangeResolver.dataframe_to_a1_range(df, sheet_name='My Data')
        assert result == "'My Data'!A1:C3"
        
        # Test with custom start cell
        result = RangeResolver.dataframe_to_a1_range(df, start_cell='B2')
        assert result == 'B2:D4'
    
    def test_apply_range_to_dataframe(self):
        """Test applying range specifications to DataFrames"""
        # Create test DataFrame
        df = pl.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10],
            'C': [11, 12, 13, 14, 15],
            'D': [16, 17, 18, 19, 20]
        })
        
        # Test row and column slicing
        result = RangeResolver.apply_range_to_dataframe(df, 'B2:C4')
        expected_df = pl.DataFrame({
            'B': [7, 8, 9],
            'C': [12, 13, 14]
        })
        assert result.equals(expected_df)
        
        # Test single column
        result = RangeResolver.apply_range_to_dataframe(df, 'A1:A3')
        expected_df = pl.DataFrame({
            'A': [1, 2, 3]
        })
        assert result.equals(expected_df)
    
    def test_get_range_dimensions(self):
        """Test getting range dimensions"""
        rows, cols = RangeResolver.get_range_dimensions('A1:C10')
        assert rows == 10
        assert cols == 3
        
        rows, cols = RangeResolver.get_range_dimensions('B2:E5')
        assert rows == 4
        assert cols == 4
        
        # Single cell
        rows, cols = RangeResolver.get_range_dimensions('A1')
        assert rows == 1
        assert cols == 1
    
    def test_expand_range(self):
        """Test expanding ranges"""
        result = RangeResolver.expand_range('A1:C3', 2, 1)
        assert result == 'A1:D5'
        
        result = RangeResolver.expand_range('Sheet1!B2:D4', 1, 2)
        assert result == 'Sheet1!B2:F5'
        
        # Test with quoted sheet name
        result = RangeResolver.expand_range("'My Sheet'!A1:B2", 0, 1)
        assert result == "'My Sheet'!A1:C2"
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test invalid A1 notation
        with pytest.raises(ValueError):
            RangeResolver.parse_a1_notation('invalid_range')
        
        # Test empty DataFrame
        df = pl.DataFrame()
        result = RangeResolver.dataframe_to_a1_range(df)
        # Should handle empty DataFrame gracefully
        assert 'A1' in result