#!/usr/bin/env python3
"""
Test script for data cleaning operations functions.
Tests all functions in the data_cleaning_operations.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.data_cleaning_operations.data_cleaning_operations import (
    STANDARDIZE_CURRENCY, CLEAN_NUMERIC, NORMALIZE_NAMES, REMOVE_DUPLICATES, STANDARDIZE_DATES,
    ValidationError, CalculationError, DataQualityError, ConfigurationError
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

            # Handle comparison of Path objects (for functions that return file paths)
            if isinstance(expected, Path):
                if isinstance(result, Path) and result.exists():
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: Path object that exists, Got: {result}")
                    failed += 1
            elif isinstance(expected, str) and "path" in expected.lower():
                # Handle string path expectations
                if isinstance(result, Path) and result.exists():
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: Valid path, Got: {result}")
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

def create_test_data_files(ctx):
    """Create test data files for file input testing."""
    # Create test currency data
    currency_data = pl.DataFrame({
        "currency": ["$1,234.56", "USD 2345.67", "€3,456.78", "¥4567", "£5,678.90"]
    })
    currency_path = ctx.deps.data_dir / "test_currency.csv"
    currency_data.write_csv(currency_path)

    # Create test mixed numeric data
    mixed_data = pl.DataFrame({
        "mixed": ["$1,234.56", "(567.89)", "€2,345.67", "¥3000", "4,567.89%"]
    })
    mixed_path = ctx.deps.data_dir / "test_mixed.csv"
    mixed_data.write_csv(mixed_path)

    # Create test names data
    names_data = pl.DataFrame({
        "names": ["Apple Inc.", "Apple Incorporated", "MICROSOFT CORP", "microsoft corporation", "Google LLC"]
    })
    names_path = ctx.deps.data_dir / "test_names.csv"
    names_data.write_csv(names_path)

    # Create test duplicate data
    duplicate_data = pl.DataFrame({
        "id": ["A001", "A002", "A001", "A003", "A002"],
        "name": ["John", "Jane", "John", "Bob", "Jane"],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0]
    })
    duplicate_path = ctx.deps.data_dir / "test_duplicates.csv"
    duplicate_data.write_csv(duplicate_path)

    # Create test dates data - use consistent format for file test
    dates_data = pl.DataFrame({
        "dates": ["01/15/2023", "01/20/2023", "01/25/2023", "01/30/2023", "02/05/2023"]
    })
    dates_path = ctx.deps.data_dir / "test_dates.csv"
    dates_data.write_csv(dates_path)

    return {
        "currency": currency_path,
        "mixed": mixed_path,
        "names": names_path,
        "duplicates": duplicate_path,
        "dates": dates_path
    }

def run_all_tests():
    """Run all tests for the data cleaning operations functions."""
    print("Starting comprehensive test of data cleaning operations functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    # Create test data files
    test_files = create_test_data_files(ctx)

    all_passed = True

    # Test STANDARDIZE_CURRENCY function
    currency_tests = [
        ({
            'currency_series': ["$1,234.56", "USD 2345.67", "1234.56"],
            'target_format': 'USD',
            'output_filename': 'currency_test1.parquet'
        }, "valid_path", "Basic USD currency standardization"),

        ({
            'currency_series': ["€1.234,56", "EUR 2345.67", "3456.78"],
            'target_format': 'EUR',
            'output_filename': 'currency_test2.parquet'
        }, "valid_path", "EUR currency standardization"),

        ({
            'currency_series': ["¥1234", "JPY 2345", "3456"],
            'target_format': 'JPY',
            'output_filename': 'currency_test3.parquet'
        }, "valid_path", "JPY currency standardization"),

        ({
            'currency_series': pl.Series(["$1,234.56", "USD 2345.67"]),
            'target_format': 'USD',
            'output_filename': 'currency_test4.parquet'
        }, "valid_path", "Currency standardization with Polars Series"),

        ({
            'currency_series': str(test_files["currency"]),
            'target_format': 'USD',
            'output_filename': 'currency_test5.parquet'
        }, "valid_path", "Currency standardization with file input"),
    ]
    all_passed &= test_function("STANDARDIZE_CURRENCY", STANDARDIZE_CURRENCY, currency_tests, ctx=ctx)

    # Test CLEAN_NUMERIC function
    numeric_tests = [
        ({
            'mixed_series': ["$1,234.56", "€987.65", "¥1000", "(500.00)"],
            'output_filename': 'numeric_test1.parquet'
        }, "valid_path", "Basic numeric cleaning"),

        ({
            'mixed_series': ["1,234.56", "(567.89)", "2,345.67%", "USD 1000"],
            'output_filename': 'numeric_test2.parquet'
        }, "valid_path", "Accounting format and percentage cleaning"),

        ({
            'mixed_series': pl.Series(["$1,234.56", "(567.89)", "2,345.67"]),
            'output_filename': 'numeric_test3.parquet'
        }, "valid_path", "Numeric cleaning with Polars Series"),

        ({
            'mixed_series': str(test_files["mixed"]),
            'output_filename': 'numeric_test4.parquet'
        }, "valid_path", "Numeric cleaning with file input"),
    ]
    all_passed &= test_function("CLEAN_NUMERIC", CLEAN_NUMERIC, numeric_tests, ctx=ctx)

    # Test NORMALIZE_NAMES function
    names_tests = [
        ({
            'name_series': ["Apple Inc.", "Apple Incorporated", "APPLE INC"],
            'normalization_rules': {"incorporated": "Inc.", "corporation": "Corp."},
            'output_filename': 'names_test1.parquet'
        }, "valid_path", "Basic name normalization"),

        ({
            'name_series': ["Microsoft Corp", "MICROSOFT CORPORATION", "microsoft corp."],
            'normalization_rules': {"corporation": "Corp.", "company": "Co."},
            'output_filename': 'names_test2.parquet'
        }, "valid_path", "Corporation name normalization"),

        ({
            'name_series': pl.Series(["Google LLC", "google llc", "GOOGLE LLC"]),
            'normalization_rules': {"limited liability company": "LLC"},
            'output_filename': 'names_test3.parquet'
        }, "valid_path", "Name normalization with Polars Series"),

        ({
            'name_series': str(test_files["names"]),
            'normalization_rules': {"incorporated": "Inc.", "corporation": "Corp."},
            'output_filename': 'names_test4.parquet'
        }, "valid_path", "Name normalization with file input"),
    ]
    all_passed &= test_function("NORMALIZE_NAMES", NORMALIZE_NAMES, names_tests, ctx=ctx)

    # Test REMOVE_DUPLICATES function
    duplicates_tests = [
        ({
            'df': pl.DataFrame({
                "id": ["A001", "A002", "A001", "A003"],
                "name": ["John", "Jane", "John", "Bob"],
                "amount": [100.0, 200.0, 150.0, 300.0]
            }),
            'subset_columns': ["id"],
            'keep_method': 'first',
            'output_filename': 'duplicates_test1.parquet'
        }, "valid_path", "Remove duplicates keeping first"),

        ({
            'df': pl.DataFrame({
                "id": ["A001", "A002", "A001", "A003"],
                "name": ["John", "Jane", "John", "Bob"],
                "amount": [100.0, 200.0, 150.0, 300.0]
            }),
            'subset_columns': ["id"],
            'keep_method': 'last',
            'output_filename': 'duplicates_test2.parquet'
        }, "valid_path", "Remove duplicates keeping last"),

        ({
            'df': pl.DataFrame({
                "id": ["A001", "A002", "A001", "A003"],
                "name": ["John", "Jane", "John", "Bob"],
                "amount": [100.0, 200.0, 150.0, 300.0]
            }),
            'subset_columns': ["id", "name"],
            'keep_method': 'first',
            'output_filename': 'duplicates_test3.parquet'
        }, "valid_path", "Remove duplicates on multiple columns"),

        ({
            'df': str(test_files["duplicates"]),
            'subset_columns': ["id"],
            'keep_method': 'first',
            'output_filename': 'duplicates_test4.parquet'
        }, "valid_path", "Remove duplicates with file input"),
    ]
    all_passed &= test_function("REMOVE_DUPLICATES", REMOVE_DUPLICATES, duplicates_tests, ctx=ctx)

    # Test STANDARDIZE_DATES function
    dates_tests = [
        ({
            'date_series': ["01/15/2023", "01/20/2023", "01/25/2023"],
            'target_format': '%Y-%m-%d',
            'output_filename': 'dates_test1.parquet'
        }, "valid_path", "Basic date standardization to ISO format"),

        ({
            'date_series': ["2023-01-15", "2023-02-20", "2023-03-25"],
            'target_format': '%m/%d/%Y',
            'output_filename': 'dates_test2.parquet'
        }, "valid_path", "Date standardization to US format"),

        ({
            'date_series': ["January 15, 2023", "February 20, 2023", "March 25, 2023"],
            'target_format': '%Y-%m-%d',
            'output_filename': 'dates_test3.parquet'
        }, "valid_path", "Date standardization from text format"),

        ({
            'date_series': pl.Series(["01/15/2023", "01/20/2023", "01/25/2023"]),
            'target_format': '%Y-%m-%d',
            'output_filename': 'dates_test4.parquet'
        }, "valid_path", "Date standardization with Polars Series"),

        ({
            'date_series': str(test_files["dates"]),
            'target_format': '%m/%d/%Y',
            'output_filename': 'dates_test5.parquet'
        }, "valid_path", "Date standardization with file input"),
    ]
    all_passed &= test_function("STANDARDIZE_DATES", STANDARDIZE_DATES, dates_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test empty input validation for STANDARDIZE_CURRENCY
    try:
        STANDARDIZE_CURRENCY(ctx, [], target_format="USD", output_filename="test.parquet")
        print("✗ Empty input validation failed for STANDARDIZE_CURRENCY")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty input validation passed for STANDARDIZE_CURRENCY")
        error_tests_passed += 1

    # Test invalid target format for STANDARDIZE_CURRENCY
    try:
        STANDARDIZE_CURRENCY(ctx, ["$100"], target_format="INVALID", output_filename="test.parquet")
        print("✗ Invalid target format validation failed for STANDARDIZE_CURRENCY")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Invalid target format validation passed for STANDARDIZE_CURRENCY")
        error_tests_passed += 1

    # Test empty normalization rules for NORMALIZE_NAMES
    try:
        NORMALIZE_NAMES(ctx, ["Apple Inc."], normalization_rules={}, output_filename="test.parquet")
        print("✗ Empty normalization rules validation failed for NORMALIZE_NAMES")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty normalization rules validation passed for NORMALIZE_NAMES")
        error_tests_passed += 1

    # Test invalid keep method for REMOVE_DUPLICATES
    try:
        test_df = pl.DataFrame({"id": ["A", "B"], "name": ["X", "Y"]})
        REMOVE_DUPLICATES(ctx, test_df, subset_columns=["id"], keep_method="invalid", output_filename="test.parquet")
        print("✗ Invalid keep method validation failed for REMOVE_DUPLICATES")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Invalid keep method validation passed for REMOVE_DUPLICATES")
        error_tests_passed += 1

    # Test invalid date format for STANDARDIZE_DATES
    try:
        STANDARDIZE_DATES(ctx, ["2023-01-15"], target_format="invalid_format", output_filename="test.parquet")
        print("✗ Invalid date format validation failed for STANDARDIZE_DATES")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Invalid date format validation passed for STANDARDIZE_DATES")
        error_tests_passed += 1

    # Test unparseable currency values
    try:
        STANDARDIZE_CURRENCY(ctx, ["not_a_currency", "also_not_currency"], target_format="USD", output_filename="test.parquet")
        print("✗ Unparseable currency validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Unparseable currency validation passed")
        error_tests_passed += 1

    # Test unparseable dates
    try:
        STANDARDIZE_DATES(ctx, ["not_a_date", "also_not_date"], target_format="%Y-%m-%d", output_filename="test.parquet")
        print("✗ Unparseable dates validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Unparseable dates validation passed")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Test edge cases
    print("\n=== Testing Edge Cases ===")
    edge_tests_passed = 0
    edge_tests_failed = 0

    # Test single value inputs
    try:
        result = STANDARDIZE_CURRENCY(ctx, ["$100.00"], target_format="USD", output_filename="edge_test1.parquet")
        if isinstance(result, Path) and result.exists():
            print("✓ Single value currency standardization passed")
            edge_tests_passed += 1
        else:
            print("✗ Single value currency standardization failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ Single value currency standardization failed: {e}")
        edge_tests_failed += 1

    # Test mixed valid/invalid data for CLEAN_NUMERIC
    try:
        result = CLEAN_NUMERIC(ctx, ["$100.00", "", "€200.50", "invalid"], output_filename="edge_test2.parquet")
        if isinstance(result, Path) and result.exists():
            print("✓ Mixed valid/invalid numeric cleaning passed")
            edge_tests_passed += 1
        else:
            print("✗ Mixed valid/invalid numeric cleaning failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ Mixed valid/invalid numeric cleaning failed: {e}")
        edge_tests_failed += 1

    # Test DataFrame with no duplicates
    try:
        no_dup_df = pl.DataFrame({
            "id": ["A001", "A002", "A003"],
            "name": ["John", "Jane", "Bob"]
        })
        result = REMOVE_DUPLICATES(ctx, no_dup_df, subset_columns=["id"], keep_method="first", output_filename="edge_test3.parquet")
        if isinstance(result, Path) and result.exists():
            print("✓ No duplicates DataFrame processing passed")
            edge_tests_passed += 1
        else:
            print("✗ No duplicates DataFrame processing failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ No duplicates DataFrame processing failed: {e}")
        edge_tests_failed += 1

    print(f"Edge case tests: {edge_tests_passed} passed, {edge_tests_failed} failed")
    all_passed &= (edge_tests_failed == 0)

    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All data cleaning functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the failed tests above.")
    print("="*50)

    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
