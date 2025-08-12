#!/usr/bin/env python3
"""
Test script for text and data management functions.
Tests all functions in the text_and_data_management_functions.py module.
"""

import sys
import traceback
from decimal import Decimal
from datetime import datetime, date
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.text_and_data_management_functions.text_and_data_management_functions import (
    CONCAT, CONCATENATE, TEXT, LEFT, RIGHT, MID, LEN, FIND, SEARCH,
    REPLACE, SUBSTITUTE, TRIM, CLEAN, UPPER, LOWER, PROPER, VALUE, TEXTJOIN,
    ValidationError, CalculationError, DataQualityError
)

# Import FinnDeps and RunContext for testing
from tools.finn_deps import FinnDeps, RunContext

def test_function(func_name, func, test_cases, ctx=None):
    """Test a function with multiple test cases."""
    print(f"\n=== Testing {func_name} ===")
    passed = 0
    failed = 0

    for i, (args, expected, description) in enumerate(test_cases, 1):
        try:
            # Add context parameter if provided
            if ctx is not None:
                args = {'run_context': ctx, **args}
            result = func(**args)

            # Handle comparison of different types
            if isinstance(expected, Decimal):
                if isinstance(result, Decimal) and abs(result - expected) < Decimal('1e-10'):
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            elif isinstance(expected, (str, int)):
                if result == expected:
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            else:
                # Handle other types
                if result == expected:
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1

        except Exception as e:
            print(f"✗ Test {i}: {description}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            failed += 1

    print(f"Results for {func_name}: {passed} passed, {failed} failed")
    return failed == 0

def create_test_data(ctx):
    """Create test data files for testing."""
    # Create test CSV file with text data
    test_text_data = pl.DataFrame({
        "text_column": ["Hello", "World", "Financial", "Planning", "Analysis"]
    })

    # Save test data
    csv_path = ctx.deps.data_dir / "test_text.csv"
    parquet_path = ctx.deps.data_dir / "test_text.parquet"

    test_text_data.write_csv(csv_path)
    test_text_data.write_parquet(parquet_path)

    # Create test data with numbers as text
    test_number_data = pl.DataFrame({
        "number_text": ["123.45", "$1,234.56", "12.5%", "(500)", "1000"]
    })

    number_csv_path = ctx.deps.data_dir / "test_numbers.csv"
    number_parquet_path = ctx.deps.data_dir / "test_numbers.parquet"

    test_number_data.write_csv(number_csv_path)
    test_number_data.write_parquet(number_parquet_path)

def run_all_tests():
    """Run all tests for the text and data management functions."""
    print("Starting comprehensive test of text and data management functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    # Create test data
    create_test_data(ctx)

    all_passed = True

    # Test CONCAT function - using *args, not keyword
    def test_concat_wrapper(run_context, *texts):
        return CONCAT(run_context, *texts)

    concat_tests = [
        (("Hello", " ", "World"), "Hello World", "Basic concatenation"),
        (("Revenue: $", 1000, " Million"), "Revenue: $1000 Million", "Mixed types concatenation"),
        (("Q1", " ", "2024"), "Q1 2024", "String and number concatenation"),
        ((), "", "Empty concatenation"),
        (("Financial", "Planning"), "FinancialPlanning", "No separator concatenation"),
        (("test_text.csv",), "HelloWorldFinancialPlanningAnalysis", "File input concatenation"),
    ]

    # Custom test for CONCAT since it uses *args
    print(f"\n=== Testing CONCAT ===")
    passed = 0
    failed = 0
    for i, (args, expected, description) in enumerate(concat_tests, 1):
        try:
            result = CONCAT(ctx, *args)
            if result == expected:
                print(f"✓ Test {i}: {description}")
                passed += 1
            else:
                print(f"✗ Test {i}: {description}")
                print(f"  Expected: {expected}, Got: {result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test {i}: {description}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            failed += 1
    print(f"Results for CONCAT: {passed} passed, {failed} failed")
    all_passed &= (failed == 0)

    # Test CONCATENATE function - using *args, not keyword
    concatenate_tests = [
        (("Q1", " ", "2024"), "Q1 2024", "Legacy concatenation"),
        (("Budget: ", 50000), "Budget: 50000", "Budget concatenation"),
        (("Financial", " ", "Analysis"), "Financial Analysis", "Multi-part concatenation"),
    ]

    # Custom test for CONCATENATE since it uses *args
    print(f"\n=== Testing CONCATENATE ===")
    passed = 0
    failed = 0
    for i, (args, expected, description) in enumerate(concatenate_tests, 1):
        try:
            result = CONCATENATE(ctx, *args)
            if result == expected:
                print(f"✓ Test {i}: {description}")
                passed += 1
            else:
                print(f"✗ Test {i}: {description}")
                print(f"  Expected: {expected}, Got: {result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test {i}: {description}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            failed += 1
    print(f"Results for CONCATENATE: {passed} passed, {failed} failed")
    all_passed &= (failed == 0)

    # Test TEXT function
    text_tests = [
        ({'value': 0.125, 'format_text': "0.00%"}, "12.50%", "Percentage formatting"),
        ({'value': 1234567.89, 'format_text': "$#,##0.00"}, "$1,234,567.89", "Currency formatting"),
        ({'value': Decimal('0.0825'), 'format_text': "0.000%"}, "8.250%", "Decimal percentage formatting"),
        ({'value': 123.456, 'format_text': "0.00"}, "123.46", "Decimal places formatting"),
        ({'value': 1000, 'format_text': "#,##0"}, "1,000", "Thousands separator formatting"),
        ({'value': datetime(2024, 3, 15), 'format_text': "yyyy-mm-dd"}, "2024-03-15", "Date formatting"),
        ({'value': "Simple text", 'format_text': "text"}, "Simple text", "Text passthrough"),
    ]
    all_passed &= test_function("TEXT", TEXT, text_tests, ctx=ctx)

    # Test LEFT function
    left_tests = [
        ({'text': "Financial Planning", 'num_chars': 9}, "Financial", "Extract left characters"),
        ({'text': "AAPL-2024-Q1", 'num_chars': 4}, "AAPL", "Stock symbol extraction"),
        ({'text': "Hello World", 'num_chars': 5}, "Hello", "Basic left extraction"),
        ({'text': "Short", 'num_chars': 10}, "Short", "Num chars exceeds length"),
        ({'text': "Test", 'num_chars': 0}, "", "Zero characters"),
        ({'text': "test_text.csv", 'num_chars': 5}, "Hello", "File input left extraction"),
    ]
    all_passed &= test_function("LEFT", LEFT, left_tests, ctx=ctx)

    # Test RIGHT function
    right_tests = [
        ({'text': "Financial Planning", 'num_chars': 8}, "Planning", "Extract right characters"),
        ({'text': "AAPL-2024-Q1", 'num_chars': 2}, "Q1", "Quarter extraction"),
        ({'text': "Hello World", 'num_chars': 5}, "World", "Basic right extraction"),
        ({'text': "Short", 'num_chars': 10}, "Short", "Num chars exceeds length"),
        ({'text': "Test", 'num_chars': 0}, "", "Zero characters"),
        ({'text': "test_text.csv", 'num_chars': 8}, "Hello", "File input right extraction"),
    ]
    all_passed &= test_function("RIGHT", RIGHT, right_tests, ctx=ctx)

    # Test MID function
    mid_tests = [
        ({'text': "Financial Planning", 'start_num': 11, 'num_chars': 8}, "Planning", "Extract middle characters"),
        ({'text': "AAPL-2024-Q1", 'start_num': 6, 'num_chars': 4}, "2024", "Year extraction"),
        ({'text': "Hello World", 'start_num': 7, 'num_chars': 5}, "World", "Basic middle extraction"),
        ({'text': "Test", 'start_num': 2, 'num_chars': 2}, "es", "Middle substring"),
        ({'text': "Short", 'start_num': 1, 'num_chars': 10}, "Short", "Exceeds length"),
        ({'text': "test_text.csv", 'start_num': 1, 'num_chars': 5}, "Hello", "File input middle extraction"),
    ]
    all_passed &= test_function("MID", MID, mid_tests, ctx=ctx)

    # Test LEN function
    len_tests = [
        ({'text': "Financial Planning"}, 18, "Count characters in phrase"),
        ({'text': "AAPL"}, 4, "Count characters in symbol"),
        ({'text': ""}, 0, "Empty string length"),
        ({'text': "Hello World"}, 11, "Basic length calculation"),
        ({'text': "test_text.csv"}, 5, "File input length"),
    ]
    all_passed &= test_function("LEN", LEN, len_tests, ctx=ctx)

    # Test FIND function
    find_tests = [
        ({'find_text': "Plan", 'within_text': "Financial Planning"}, 11, "Case-sensitive find"),
        ({'find_text': "plan", 'within_text': "Financial Planning"}, -1, "Case-sensitive not found"),
        ({'find_text': "2024", 'within_text': "AAPL-2024-Q1", 'start_num': 1}, 6, "Find with start position"),
        ({'find_text': "World", 'within_text': "Hello World"}, 7, "Basic find operation"),
        ({'find_text': "xyz", 'within_text': "Hello World"}, -1, "Text not found"),
    ]
    all_passed &= test_function("FIND", FIND, find_tests, ctx=ctx)

    # Test SEARCH function
    search_tests = [
        ({'find_text': "plan", 'within_text': "Financial Planning"}, 11, "Case-insensitive search"),
        ({'find_text': "PLAN", 'within_text': "Financial Planning"}, 11, "Case-insensitive uppercase"),
        ({'find_text': "q1", 'within_text': "AAPL-2024-Q1", 'start_num': 1}, 11, "Case-insensitive with start"),
        ({'find_text': "world", 'within_text': "Hello World"}, 7, "Basic case-insensitive search"),
        ({'find_text': "xyz", 'within_text': "Hello World"}, -1, "Text not found"),
    ]
    all_passed &= test_function("SEARCH", SEARCH, search_tests, ctx=ctx)

    # Test REPLACE function
    replace_tests = [
        ({'old_text': "Financial Planning", 'start_num': 11, 'num_chars': 8, 'new_text': "Analysis"}, "Financial Analysis", "Replace portion of text"),
        ({'old_text': "AAPL-2023-Q1", 'start_num': 6, 'num_chars': 4, 'new_text': "2024"}, "AAPL-2024-Q1", "Replace year"),
        ({'old_text': "Hello World", 'start_num': 7, 'num_chars': 5, 'new_text': "Universe"}, "Hello Universe", "Replace word"),
        ({'old_text': "Test", 'start_num': 1, 'num_chars': 4, 'new_text': "Best"}, "Best", "Replace entire string"),
    ]
    all_passed &= test_function("REPLACE", REPLACE, replace_tests, ctx=ctx)

    # Test SUBSTITUTE function
    substitute_tests = [
        ({'text': "Financial Planning and Financial Analysis", 'old_text': "Financial", 'new_text': "Business"}, "Business Planning and Business Analysis", "Replace all occurrences"),
        ({'text': "Q1-Q1-Q1", 'old_text': "Q1", 'new_text': "Q2", 'instance_num': 2}, "Q1-Q2-Q1", "Replace specific instance"),
        ({'text': "Hello World Hello", 'old_text': "Hello", 'new_text': "Hi"}, "Hi World Hi", "Replace multiple occurrences"),
        ({'text': "Test String", 'old_text': "xyz", 'new_text': "abc"}, "Test String", "No match to replace"),
    ]
    all_passed &= test_function("SUBSTITUTE", SUBSTITUTE, substitute_tests, ctx=ctx)

    # Test TRIM function
    trim_tests = [
        ({'text': "  Extra   Spaces  "}, "Extra Spaces", "Remove extra spaces"),
        ({'text': "  Financial Planning  "}, "Financial Planning", "Trim leading/trailing spaces"),
        ({'text': "Normal Text"}, "Normal Text", "No extra spaces"),
        ({'text': "   "}, "", "Only spaces"),
        ({'text': "Multiple    Internal    Spaces"}, "Multiple Internal Spaces", "Collapse internal spaces"),
    ]
    all_passed &= test_function("TRIM", TRIM, trim_tests, ctx=ctx)

    # Test CLEAN function
    clean_tests = [
        ({'text': "Financial\x00Planning\x01"}, "FinancialPlanning", "Remove non-printable characters"),
        ({'text': "Clean\tText\n"}, "Clean\tText\n", "Keep printable whitespace"),
        ({'text': "Normal Text"}, "Normal Text", "No non-printable characters"),
        ({'text': "Test\x02\x03String"}, "TestString", "Multiple non-printable characters"),
    ]
    all_passed &= test_function("CLEAN", CLEAN, clean_tests, ctx=ctx)

    # Test UPPER function
    upper_tests = [
        ({'text': "hello world"}, "HELLO WORLD", "Convert to uppercase"),
        ({'text': "Financial Planning"}, "FINANCIAL PLANNING", "Mixed case to uppercase"),
        ({'text': "ALREADY UPPER"}, "ALREADY UPPER", "Already uppercase"),
        ({'text': "123 abc"}, "123 ABC", "Numbers and letters"),
    ]
    all_passed &= test_function("UPPER", UPPER, upper_tests, ctx=ctx)

    # Test LOWER function
    lower_tests = [
        ({'text': "HELLO WORLD"}, "hello world", "Convert to lowercase"),
        ({'text': "Financial Planning"}, "financial planning", "Mixed case to lowercase"),
        ({'text': "already lower"}, "already lower", "Already lowercase"),
        ({'text': "123 ABC"}, "123 abc", "Numbers and letters"),
    ]
    all_passed &= test_function("LOWER", LOWER, lower_tests, ctx=ctx)

    # Test PROPER function
    proper_tests = [
        ({'text': "hello world"}, "Hello World", "Convert to proper case"),
        ({'text': "financial planning"}, "Financial Planning", "Multiple words proper case"),
        ({'text': "UPPER CASE"}, "Upper Case", "From uppercase to proper"),
        ({'text': "mixed CaSe"}, "Mixed Case", "Mixed case to proper"),
    ]
    all_passed &= test_function("PROPER", PROPER, proper_tests, ctx=ctx)

    # Test VALUE function
    value_tests = [
        ({'text': "123.45"}, Decimal('123.45'), "Basic number conversion"),
        ({'text': "$1,234.56"}, Decimal('1234.56'), "Currency with thousands separator"),
        ({'text': "12.5%"}, Decimal('0.125'), "Percentage conversion"),
        ({'text': "(500)"}, Decimal('-500'), "Negative in parentheses"),
        ({'text': "1000"}, Decimal('1000'), "Integer conversion"),
        ({'text': "test_numbers.csv"}, Decimal('123.45'), "File input conversion"),
    ]
    all_passed &= test_function("VALUE", VALUE, value_tests, ctx=ctx)

    # Test TEXTJOIN function - using *args, not keyword
    textjoin_tests = [
        ((", ", True, "Apple", "", "Banana", "Cherry"), "Apple, Banana, Cherry", "Join with empty ignored"),
        ((" | ", False, "Q1", "Q2", "Q3", "Q4"), "Q1 | Q2 | Q3 | Q4", "Join with pipe delimiter"),
        ((",", True, "A", "B", "C"), "A,B,C", "Join with comma"),
        (("-", False, "One", "", "Three"), "One--Three", "Join with empty not ignored"),
        ((",", True, "test_text.csv"), "Hello,World,Financial,Planning,Analysis", "File input join"),
    ]

    # Custom test for TEXTJOIN since it uses *args
    print(f"\n=== Testing TEXTJOIN ===")
    passed = 0
    failed = 0
    for i, (args, expected, description) in enumerate(textjoin_tests, 1):
        try:
            result = TEXTJOIN(ctx, *args)
            if result == expected:
                print(f"✓ Test {i}: {description}")
                passed += 1
            else:
                print(f"✗ Test {i}: {description}")
                print(f"  Expected: {expected}, Got: {result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test {i}: {description}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            failed += 1
    print(f"Results for TEXTJOIN: {passed} passed, {failed} failed")
    all_passed &= (failed == 0)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test invalid input validation
    try:
        LEFT(ctx, None, num_chars=5)
        print("✗ None input validation failed for LEFT")
        error_tests_failed += 1
    except ValidationError:
        print("✓ None input validation passed for LEFT")
        error_tests_passed += 1

    # Test invalid number conversion
    try:
        VALUE(ctx, "not a number")
        print("✗ Invalid number conversion validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Invalid number conversion validation passed")
        error_tests_passed += 1

    # Test invalid integer parameter
    try:
        LEFT(ctx, "test", num_chars=-1)
        print("✗ Negative num_chars validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Negative num_chars validation passed")
        error_tests_passed += 1

    # Test invalid start position
    try:
        MID(ctx, "test", start_num=0, num_chars=2)
        print("✗ Invalid start position validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid start position validation passed")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All text functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the failed tests above.")
    print("="*50)

    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
