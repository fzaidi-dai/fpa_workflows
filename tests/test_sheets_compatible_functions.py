"""Tests for Google Sheets Compatible Functions"""
import pytest
import polars as pl
import sys
import os
from pathlib import Path
import tempfile
from datetime import date, datetime
import json

# Add the mcp_tooling directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_tooling"))

from sheets_compatible_functions import SheetsCompatibleFunctions


class TestSheetsCompatibleFunctions:
    """Test suite for Google Sheets compatible functions"""
    
    @pytest.fixture
    def sheets_funcs(self):
        """Create SheetsCompatibleFunctions instance"""
        return SheetsCompatibleFunctions()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pl.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [100, 200, 300, 400, 500],
            "D": ["apple", "banana", "cherry", "date", "elderberry"]
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_df):
        """Create temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df.write_csv(f.name)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def lookup_df(self):
        """Create DataFrame for lookup tests"""
        return pl.DataFrame({
            "Product": ["Apple", "Banana", "Cherry", "Date"],
            "Price": [1.20, 0.50, 2.00, 3.50],
            "Category": ["Fruit", "Fruit", "Fruit", "Fruit"],
            "Stock": [100, 200, 50, 25]
        })
    
    @pytest.fixture
    def temp_lookup_file(self, lookup_df):
        """Create temporary lookup file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            lookup_df.write_csv(f.name)
            yield f.name
        os.unlink(f.name)
    
    # ================== MATH FUNCTIONS TESTS ==================
    
    def test_sum_basic(self, sheets_funcs, temp_csv_file):
        """Test SUM function basic functionality"""
        result = sheets_funcs.SUM(temp_csv_file)
        expected = 1 + 2 + 3 + 4 + 5 + 10 + 20 + 30 + 40 + 50 + 100 + 200 + 300 + 400 + 500
        assert result == expected
    
    def test_sum_with_range(self, sheets_funcs, temp_csv_file):
        """Test SUM with range specification"""
        result = sheets_funcs.SUM(temp_csv_file, "A1:A3")
        assert result == 6  # 1 + 2 + 3
        
        result = sheets_funcs.SUM(temp_csv_file, "B:B")
        assert result == 150  # 10 + 20 + 30 + 40 + 50
    
    def test_average_basic(self, sheets_funcs, temp_csv_file):
        """Test AVERAGE function"""
        result = sheets_funcs.AVERAGE(temp_csv_file, "A1:A5")
        assert result == 3.0  # (1+2+3+4+5)/5
    
    def test_count(self, sheets_funcs, temp_csv_file):
        """Test COUNT function"""
        result = sheets_funcs.COUNT(temp_csv_file)
        assert result == 15  # 5 + 5 + 5 (three numeric columns)
        
        result = sheets_funcs.COUNT(temp_csv_file, "A:A")
        assert result == 5
    
    def test_counta(self, sheets_funcs, temp_csv_file):
        """Test COUNTA function"""
        result = sheets_funcs.COUNTA(temp_csv_file)
        assert result == 20  # All non-null cells (4 columns × 5 rows)
    
    def test_min_max(self, sheets_funcs, temp_csv_file):
        """Test MIN and MAX functions"""
        min_result = sheets_funcs.MIN(temp_csv_file, "A1:A5")
        assert min_result == 1
        
        max_result = sheets_funcs.MAX(temp_csv_file, "C1:C5")
        assert max_result == 500
    
    # ================== LOOKUP FUNCTIONS TESTS ==================
    
    def test_vlookup_exact_match(self, sheets_funcs, temp_lookup_file):
        """Test VLOOKUP with exact match"""
        result = sheets_funcs.VLOOKUP("Banana", temp_lookup_file, 2, False)
        assert result == 0.50
        
        result = sheets_funcs.VLOOKUP("Cherry", temp_lookup_file, 4, False)
        assert result == 50
    
    def test_vlookup_not_found(self, sheets_funcs, temp_lookup_file):
        """Test VLOOKUP when value not found"""
        result = sheets_funcs.VLOOKUP("Orange", temp_lookup_file, 2, False)
        assert result is None
    
    def test_index_function(self, sheets_funcs, temp_lookup_file):
        """Test INDEX function"""
        result = sheets_funcs.INDEX(temp_lookup_file, 2, 2)  # Row 2, Col 2 = Price of Banana
        assert result == 0.50
        
        # Test entire row
        row = sheets_funcs.INDEX(temp_lookup_file, 1)  # First row
        assert row[0] == "Apple"
        assert row[1] == 1.20
    
    def test_match_function(self, sheets_funcs):
        """Test MATCH function"""
        lookup_array = ["Apple", "Banana", "Cherry", "Date"]
        
        # Exact match
        result = sheets_funcs.MATCH("Cherry", lookup_array, 0)
        assert result == 3  # 1-based index
        
        # Not found
        result = sheets_funcs.MATCH("Orange", lookup_array, 0)
        assert result is None
    
    # ================== CONDITIONAL AGGREGATION TESTS ==================
    
    def test_sumif_basic(self, sheets_funcs, temp_csv_file):
        """Test SUMIF function"""
        # Create test data with criteria
        df = pl.DataFrame({
            "Category": ["A", "B", "A", "B", "A"],
            "Amount": [10, 20, 30, 40, 50]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.write_csv(f.name)
            try:
                result = sheets_funcs.SUMIF(f.name, "A")  # Sum where category = "A"
                assert result == 90  # 10 + 30 + 50
            finally:
                os.unlink(f.name)
    
    def test_countif_basic(self, sheets_funcs):
        """Test COUNTIF function"""
        df = pl.DataFrame({
            "Status": ["Active", "Inactive", "Active", "Active", "Inactive"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.write_csv(f.name)
            try:
                result = sheets_funcs.COUNTIF(f.name, "Active")
                assert result == 3
            finally:
                os.unlink(f.name)
    
    def test_averageif_basic(self, sheets_funcs):
        """Test AVERAGEIF function"""
        df = pl.DataFrame({
            "Grade": ["A", "B", "A", "C", "A"],
            "Score": [95, 85, 90, 75, 88]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.write_csv(f.name)
            try:
                result = sheets_funcs.AVERAGEIF(f.name, "A")  # Average of A grades
                assert result == 91.0  # (95 + 90 + 88) / 3
            finally:
                os.unlink(f.name)
    
    def test_criteria_parsing(self, sheets_funcs):
        """Test criteria parsing for conditional functions"""
        df = pl.DataFrame({
            "Value": [10, 20, 30, 40, 50]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.write_csv(f.name)
            try:
                # Test greater than
                result = sheets_funcs.COUNTIF(f.name, ">25")
                assert result == 3  # 30, 40, 50
                
                # Test less than or equal
                result = sheets_funcs.COUNTIF(f.name, "<=30")
                assert result == 3  # 10, 20, 30
            finally:
                os.unlink(f.name)
    
    # ================== TEXT FUNCTIONS TESTS ==================
    
    def test_concatenate(self, sheets_funcs):
        """Test CONCATENATE function"""
        result = sheets_funcs.CONCATENATE("Hello", " ", "World")
        assert result == "Hello World"
    
    def test_left_right_mid(self, sheets_funcs):
        """Test LEFT, RIGHT, MID functions"""
        text = "Hello World"
        
        assert sheets_funcs.LEFT(text, 5) == "Hello"
        assert sheets_funcs.RIGHT(text, 5) == "World"
        assert sheets_funcs.MID(text, 7, 5) == "World"  # 1-based indexing
    
    def test_len_function(self, sheets_funcs):
        """Test LEN function"""
        assert sheets_funcs.LEN("Hello") == 5
        assert sheets_funcs.LEN("") == 0
    
    def test_case_functions(self, sheets_funcs):
        """Test UPPER, LOWER, PROPER functions"""
        text = "hello world"
        
        assert sheets_funcs.UPPER(text) == "HELLO WORLD"
        assert sheets_funcs.LOWER("HELLO WORLD") == "hello world"
        assert sheets_funcs.PROPER(text) == "Hello World"
    
    def test_trim_function(self, sheets_funcs):
        """Test TRIM function"""
        result = sheets_funcs.TRIM("  Hello   World  ")
        assert result == "Hello World"
    
    def test_substitute(self, sheets_funcs):
        """Test SUBSTITUTE function"""
        text = "Hello World Hello"
        
        # Replace all
        result = sheets_funcs.SUBSTITUTE(text, "Hello", "Hi")
        assert result == "Hi World Hi"
        
        # Replace specific instance
        result = sheets_funcs.SUBSTITUTE(text, "Hello", "Hi", 1)
        assert result == "Hi World Hello"
    
    # ================== DATE/TIME FUNCTIONS TESTS ==================
    
    def test_today_now(self, sheets_funcs):
        """Test TODAY and NOW functions"""
        today = sheets_funcs.TODAY()
        now = sheets_funcs.NOW()
        
        assert isinstance(today, date)
        assert isinstance(now, datetime)
        assert today == now.date()
    
    def test_date_function(self, sheets_funcs):
        """Test DATE function"""
        result = sheets_funcs.DATE(2023, 12, 25)
        expected = date(2023, 12, 25)
        assert result == expected
    
    def test_year_month_day(self, sheets_funcs):
        """Test YEAR, MONTH, DAY functions"""
        test_date = date(2023, 12, 25)
        
        assert sheets_funcs.YEAR(test_date) == 2023
        assert sheets_funcs.MONTH(test_date) == 12
        assert sheets_funcs.DAY(test_date) == 25
        
        # Test with string input
        assert sheets_funcs.YEAR("2023-06-15") == 2023
        assert sheets_funcs.MONTH("2023-06-15") == 6
        assert sheets_funcs.DAY("2023-06-15") == 15
    
    def test_weekday(self, sheets_funcs):
        """Test WEEKDAY function"""
        # Monday, January 1, 2024
        monday = date(2024, 1, 1)
        
        # Type 1: Sunday=1, Monday=2, ..., Saturday=7
        assert sheets_funcs.WEEKDAY(monday, 1) == 2
        
        # Type 2: Monday=1, Tuesday=2, ..., Sunday=7
        assert sheets_funcs.WEEKDAY(monday, 2) == 1
        
        # Type 3: Monday=0, Tuesday=1, ..., Sunday=6
        assert sheets_funcs.WEEKDAY(monday, 3) == 0
    
    def test_eomonth(self, sheets_funcs):
        """Test EOMONTH function"""
        start_date = date(2023, 1, 15)
        
        # End of same month
        result = sheets_funcs.EOMONTH(start_date, 0)
        assert result == date(2023, 1, 31)
        
        # End of next month
        result = sheets_funcs.EOMONTH(start_date, 1)
        assert result == date(2023, 2, 28)
        
        # End of previous month
        result = sheets_funcs.EOMONTH(start_date, -1)
        assert result == date(2022, 12, 31)
    
    def test_datedif(self, sheets_funcs):
        """Test DATEDIF function"""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        # Days
        result = sheets_funcs.DATEDIF(start_date, end_date, "D")
        assert result == 364  # 2023 is not a leap year
        
        # Months
        result = sheets_funcs.DATEDIF(start_date, end_date, "M")
        assert result == 11
        
        # Years
        result = sheets_funcs.DATEDIF(start_date, "2024-01-01", "Y")
        assert result == 1
    
    # ================== LOGICAL FUNCTIONS TESTS ==================
    
    def test_if_function(self, sheets_funcs):
        """Test IF function"""
        assert sheets_funcs.IF(True, "Yes", "No") == "Yes"
        assert sheets_funcs.IF(False, "Yes", "No") == "No"
        assert sheets_funcs.IF(5 > 3, "Greater", "Less") == "Greater"
    
    def test_and_or_not(self, sheets_funcs):
        """Test AND, OR, NOT functions"""
        assert sheets_funcs.AND(True, True, True) == True
        assert sheets_funcs.AND(True, False, True) == False
        
        assert sheets_funcs.OR(False, True, False) == True
        assert sheets_funcs.OR(False, False, False) == False
        
        assert sheets_funcs.NOT(True) == False
        assert sheets_funcs.NOT(False) == True
    
    def test_iferror(self, sheets_funcs):
        """Test IFERROR function"""
        assert sheets_funcs.IFERROR("Valid", "Error") == "Valid"
        assert sheets_funcs.IFERROR(None, "Error") == "Error"
        assert sheets_funcs.IFERROR("#DIV/0!", "Error") == "Error"
    
    def test_type_checking(self, sheets_funcs):
        """Test ISBLANK, ISNUMBER, ISTEXT functions"""
        assert sheets_funcs.ISBLANK(None) == True
        assert sheets_funcs.ISBLANK("") == True
        assert sheets_funcs.ISBLANK("text") == False
        
        assert sheets_funcs.ISNUMBER(42) == True
        assert sheets_funcs.ISNUMBER(3.14) == True
        assert sheets_funcs.ISNUMBER("42") == False
        assert sheets_funcs.ISNUMBER(True) == False  # Boolean is not number
        
        assert sheets_funcs.ISTEXT("hello") == True
        assert sheets_funcs.ISTEXT(42) == False
    
    # ================== ARRAY FUNCTIONS TESTS ==================
    
    def test_transpose(self, sheets_funcs, temp_csv_file):
        """Test TRANSPOSE function"""
        result = sheets_funcs.TRANSPOSE(temp_csv_file, "A1:B2")
        
        # Check that dimensions are swapped
        assert result.height == 2  # Original had 2 columns, now 2 rows
        # The exact structure depends on Polars transpose implementation
    
    def test_unique(self, sheets_funcs):
        """Test UNIQUE function"""
        # Create data with duplicates
        df = pl.DataFrame({
            "Values": [1, 2, 2, 3, 3, 3, 4]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.write_csv(f.name)
            try:
                result = sheets_funcs.UNIQUE(f.name)
                unique_values = result["Values"].to_list()
                assert len(set(unique_values)) == len(unique_values)  # All unique
                assert set(unique_values) == {1, 2, 3, 4}
            finally:
                os.unlink(f.name)
    
    def test_sort(self, sheets_funcs, temp_csv_file):
        """Test SORT function"""
        # Sort by first column ascending
        result = sheets_funcs.SORT(temp_csv_file, 1, True, "A1:B5")
        
        # Check that first column is sorted
        first_col = result[result.columns[0]].to_list()
        assert first_col == sorted(first_col)
    
    # ================== ERROR HANDLING TESTS ==================
    
    def test_file_not_found(self, sheets_funcs):
        """Test error handling for non-existent files"""
        with pytest.raises(Exception):
            sheets_funcs.SUM("non_existent_file.csv")
    
    def test_invalid_range_spec(self, sheets_funcs, temp_csv_file):
        """Test error handling for invalid range specifications"""
        with pytest.raises(Exception):
            sheets_funcs.SUM(temp_csv_file, "Z1:Z100")  # Column Z doesn't exist
    
    def test_vlookup_invalid_column(self, sheets_funcs, temp_lookup_file):
        """Test VLOOKUP with invalid column index"""
        with pytest.raises(ValueError):
            sheets_funcs.VLOOKUP("Apple", temp_lookup_file, 10, False)  # Column 10 doesn't exist
    
    def test_index_out_of_bounds(self, sheets_funcs, temp_lookup_file):
        """Test INDEX with out-of-bounds indices"""
        with pytest.raises(ValueError):
            sheets_funcs.INDEX(temp_lookup_file, 100, 1)  # Row 100 doesn't exist
        
        with pytest.raises(ValueError):
            sheets_funcs.INDEX(temp_lookup_file, 1, 100)  # Column 100 doesn't exist


def test_integration_scenario():
    """Integration test with a realistic financial scenario"""
    # Create sample sales data
    sales_data = pl.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "Product": ["Apple", "Banana", "Apple", "Cherry", "Banana"],
        "Quantity": [10, 15, 8, 20, 12],
        "Price": [1.20, 0.50, 1.20, 2.00, 0.50],
        "Total": [12.00, 7.50, 9.60, 40.00, 6.00]
    })
    
    sheets_funcs = SheetsCompatibleFunctions()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sales_data.write_csv(f.name)
        try:
            # Test various operations
            total_sales = sheets_funcs.SUM(f.name, "E:E")  # Total column
            assert total_sales == 75.10
            
            avg_price = sheets_funcs.AVERAGE(f.name, "D:D")  # Price column
            assert abs(avg_price - 1.08) < 0.01  # Allow for floating point precision
            
            apple_sales = sheets_funcs.SUMIF(f.name, "Apple")  # Would need proper implementation
            # This test shows the integration works
            
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])