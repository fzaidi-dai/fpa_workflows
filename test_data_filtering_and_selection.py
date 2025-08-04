#!/usr/bin/env python3
"""
Test script for data filtering and selection functions.
Tests all functions in the data_filtering_and_selection.py module.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime, date
import polars as pl
import numpy as np

# Import the functions to test
from tools.core_data_and_math_utils.data_filtering_and_selection.data_filtering_and_selection import (
    FILTER_BY_DATE_RANGE, FILTER_BY_VALUE, FILTER_BY_MULTIPLE_CONDITIONS,
    TOP_N, BOTTOM_N, SAMPLE_DATA,
    ValidationError, CalculationError, DataQualityError, ConfigurationError
)

# Import FinnDeps and RunContext for testing
from tools.finn_deps import FinnDeps, RunContext


def create_test_data():
    """Create test datasets for filtering and selection operations."""

    # Create sample sales data with dates
    sales_data = pl.DataFrame({
        "transaction_date": [
            "2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-25",
            "2024-06-30", "2024-07-12", "2024-08-18", "2024-09-22", "2024-10-08"
        ],
        "amount": [1500, 2300, 800, 3200, 1200, 2800, 950, 4100, 1800, 2600],
        "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
        "product": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
        "status": ["active", "active", "pending", "active", "active", "pending", "active", "active", "pending", "active"]
    })

    # Create customer data for ranking tests
    customer_data = pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "revenue": [50000, 75000, 30000, 95000, 45000, 85000, 25000, 65000, 55000, 40000],
        "profit_margin": [0.15, 0.22, 0.08, 0.28, 0.12, 0.25, 0.05, 0.18, 0.20, 0.10],
        "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
        "tier": ["Gold", "Platinum", "Silver", "Platinum", "Gold", "Platinum", "Bronze", "Gold", "Gold", "Silver"]
    })

    # Create large dataset for sampling tests
    large_data = pl.DataFrame({
        "id": range(1, 1001),
        "value": np.random.normal(100, 20, 1000),
        "category": [f"Cat_{i % 5}" for i in range(1000)],
        "score": np.random.uniform(0, 100, 1000)
    })

    return sales_data, customer_data, large_data


def save_test_data(sales_data, customer_data, large_data):
    """Save test data to files for file input testing."""

    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Save as both CSV and Parquet
    sales_data.write_csv("data/sales_test.csv")
    sales_data.write_parquet("data/sales_test.parquet")

    customer_data.write_csv("data/customer_test.csv")
    customer_data.write_parquet("data/customer_test.parquet")

    # Save a subset of large data for testing
    large_data.head(100).write_csv("data/large_test.csv")
    large_data.head(100).write_parquet("data/large_test.parquet")


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

            # Handle Path comparison for file output functions
            if isinstance(expected, Path):
                if isinstance(result, Path) and result.exists():
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: Path that exists, Got: {result}")
                    failed += 1
            elif isinstance(expected, str) and expected == "file_exists":
                # Check if result is a Path and file exists
                if isinstance(result, Path) and result.exists():
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: File to exist, Got: {result}")
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


def run_all_tests():
    """Run all tests for the data filtering and selection functions."""
    print("Starting comprehensive test of data filtering and selection functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    # Create and save test data
    sales_data, customer_data, large_data = create_test_data()
    save_test_data(sales_data, customer_data, large_data)

    all_passed = True

    # Test FILTER_BY_DATE_RANGE function
    filter_date_tests = [
        ({
            'df': sales_data,
            'date_column': 'transaction_date',
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'output_filename': 'date_range_results.parquet'
        }, "file_exists", "Filter by Q1 date range"),
        ({
            'df': sales_data,
            'date_column': 'transaction_date',
            'start_date': '2024-06-01',
            'end_date': '2024-08-31',
            'output_filename': 'date_range_summer.parquet'
        }, "file_exists", "Filter by summer date range"),
        ({
            'df': "data/sales_test.csv",
            'date_column': 'transaction_date',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'output_filename': 'date_range_full_year.parquet'
        }, "file_exists", "Filter with CSV file input"),
        ({
            'df': "data/sales_test.parquet",
            'date_column': 'transaction_date',
            'start_date': '2024-09-01',
            'end_date': '2024-12-31',
            'output_filename': 'date_range_q4.parquet'
        }, "file_exists", "Filter with Parquet file input"),
    ]
    all_passed &= test_function("FILTER_BY_DATE_RANGE", FILTER_BY_DATE_RANGE, filter_date_tests, ctx=ctx)

    # Test FILTER_BY_VALUE function
    filter_value_tests = [
        ({
            'df': sales_data,
            'column': 'amount',
            'operator': '>',
            'value': 2000,
            'output_filename': 'high_value_sales.parquet'
        }, "file_exists", "Filter sales > 2000"),
        ({
            'df': sales_data,
            'column': 'amount',
            'operator': '<=',
            'value': 1500,
            'output_filename': 'low_value_sales.parquet'
        }, "file_exists", "Filter sales <= 1500"),
        ({
            'df': sales_data,
            'column': 'region',
            'operator': '==',
            'value': 'North',
            'output_filename': 'north_sales.parquet'
        }, "file_exists", "Filter by region equality"),
        ({
            'df': customer_data,
            'column': 'profit_margin',
            'operator': '>=',
            'value': 0.20,
            'output_filename': 'high_margin_customers.parquet'
        }, "file_exists", "Filter by profit margin"),
        ({
            'df': "data/customer_test.csv",
            'column': 'revenue',
            'operator': '>',
            'value': 50000,
            'output_filename': 'high_revenue_customers.parquet'
        }, "file_exists", "Filter with CSV file input"),
    ]
    all_passed &= test_function("FILTER_BY_VALUE", FILTER_BY_VALUE, filter_value_tests, ctx=ctx)

    # Test FILTER_BY_MULTIPLE_CONDITIONS function
    multi_condition_tests = [
        ({
            'df': sales_data,
            'conditions_dict': {'region': 'North', 'status': 'active'},
            'output_filename': 'north_active_sales.parquet'
        }, "file_exists", "Filter by region and status"),
        ({
            'df': sales_data,
            'conditions_dict': {'amount': '>:2000', 'region': 'South'},
            'output_filename': 'south_high_sales.parquet'
        }, "file_exists", "Filter with operator syntax"),
        ({
            'df': customer_data,
            'conditions_dict': {'revenue': '>:50000', 'profit_margin': '>=:0.15', 'tier': 'Gold'},
            'output_filename': 'premium_customers.parquet'
        }, "file_exists", "Multiple numeric and string conditions"),
        ({
            'df': sales_data,
            'conditions_dict': {'amount': '>=:1000', 'amount': '<=:3000'},
            'output_filename': 'mid_range_sales.parquet'
        }, "file_exists", "Range conditions (note: this will use last condition due to dict key collision)"),
        ({
            'df': "data/sales_test.parquet",
            'conditions_dict': {'region': 'West', 'product': 'C'},
            'output_filename': 'west_product_c.parquet'
        }, "file_exists", "Filter with Parquet file input"),
    ]
    all_passed &= test_function("FILTER_BY_MULTIPLE_CONDITIONS", FILTER_BY_MULTIPLE_CONDITIONS, multi_condition_tests, ctx=ctx)

    # Test TOP_N function
    top_n_tests = [
        ({
            'df': customer_data,
            'column': 'revenue',
            'n': 3,
            'ascending': False,
            'output_filename': 'top_3_revenue.parquet'
        }, "file_exists", "Top 3 customers by revenue"),
        ({
            'df': customer_data,
            'column': 'profit_margin',
            'n': 5,
            'ascending': False,
            'output_filename': 'top_5_margin.parquet'
        }, "file_exists", "Top 5 customers by profit margin"),
        ({
            'df': sales_data,
            'column': 'amount',
            'n': 2,
            'ascending': True,
            'output_filename': 'bottom_2_sales.parquet'
        }, "file_exists", "Bottom 2 sales (ascending=True)"),
        ({
            'df': "data/customer_test.csv",
            'column': 'revenue',
            'n': 4,
            'ascending': False,
            'output_filename': 'top_4_csv.parquet'
        }, "file_exists", "Top N with CSV file input"),
        ({
            'df': "data/customer_test.parquet",
            'column': 'profit_margin',
            'n': 3,
            'ascending': False,
            'output_filename': 'top_3_parquet.parquet'
        }, "file_exists", "Top N with Parquet file input"),
    ]
    all_passed &= test_function("TOP_N", TOP_N, top_n_tests, ctx=ctx)

    # Test BOTTOM_N function
    bottom_n_tests = [
        ({
            'df': customer_data,
            'column': 'revenue',
            'n': 3,
            'output_filename': 'bottom_3_revenue.parquet'
        }, "file_exists", "Bottom 3 customers by revenue"),
        ({
            'df': customer_data,
            'column': 'profit_margin',
            'n': 2,
            'output_filename': 'bottom_2_margin.parquet'
        }, "file_exists", "Bottom 2 customers by profit margin"),
        ({
            'df': sales_data,
            'column': 'amount',
            'n': 4,
            'output_filename': 'bottom_4_sales.parquet'
        }, "file_exists", "Bottom 4 sales by amount"),
        ({
            'df': "data/customer_test.csv",
            'column': 'revenue',
            'n': 2,
            'output_filename': 'bottom_2_csv.parquet'
        }, "file_exists", "Bottom N with CSV file input"),
        ({
            'df': "data/customer_test.parquet",
            'column': 'profit_margin',
            'n': 3,
            'output_filename': 'bottom_3_parquet.parquet'
        }, "file_exists", "Bottom N with Parquet file input"),
    ]
    all_passed &= test_function("BOTTOM_N", BOTTOM_N, bottom_n_tests, ctx=ctx)

    # Test SAMPLE_DATA function
    sample_tests = [
        ({
            'df': customer_data,
            'n_samples': 5,
            'random_state': 42,
            'output_filename': 'sample_5_customers.parquet'
        }, "file_exists", "Sample 5 customers with seed"),
        ({
            'df': sales_data,
            'n_samples': 3,
            'random_state': 123,
            'output_filename': 'sample_3_sales.parquet'
        }, "file_exists", "Sample 3 sales with different seed"),
        ({
            'df': customer_data,
            'n_samples': 7,
            'output_filename': 'sample_7_no_seed.parquet'
        }, "file_exists", "Sample without random state"),
        ({
            'df': "data/large_test.csv",
            'n_samples': 20,
            'random_state': 456,
            'output_filename': 'sample_20_csv.parquet'
        }, "file_exists", "Sample with CSV file input"),
        ({
            'df': "data/large_test.parquet",
            'n_samples': 15,
            'random_state': 789,
            'output_filename': 'sample_15_parquet.parquet'
        }, "file_exists", "Sample with Parquet file input"),
    ]
    all_passed &= test_function("SAMPLE_DATA", SAMPLE_DATA, sample_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test empty DataFrame validation
    try:
        empty_df = pl.DataFrame()
        FILTER_BY_VALUE(ctx, empty_df, column='test', operator='>', value=1, output_filename='test.parquet')
        print("✗ Empty DataFrame validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Empty DataFrame validation passed")
        error_tests_passed += 1

    # Test invalid column name
    try:
        FILTER_BY_VALUE(ctx, sales_data, column='nonexistent', operator='>', value=1, output_filename='test.parquet')
        print("✗ Invalid column validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid column validation passed")
        error_tests_passed += 1

    # Test invalid operator
    try:
        FILTER_BY_VALUE(ctx, sales_data, column='amount', operator='invalid', value=1, output_filename='test.parquet')
        print("✗ Invalid operator validation failed")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Invalid operator validation passed")
        error_tests_passed += 1

    # Test invalid date format
    try:
        FILTER_BY_DATE_RANGE(ctx, sales_data, date_column='transaction_date',
                            start_date='invalid-date', end_date='2024-12-31',
                            output_filename='test.parquet')
        print("✗ Invalid date format validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Invalid date format validation passed")
        error_tests_passed += 1

    # Test invalid date range (start > end)
    try:
        FILTER_BY_DATE_RANGE(ctx, sales_data, date_column='transaction_date',
                            start_date='2024-12-31', end_date='2024-01-01',
                            output_filename='test.parquet')
        print("✗ Invalid date range validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid date range validation passed")
        error_tests_passed += 1

    # Test n larger than DataFrame size
    try:
        TOP_N(ctx, sales_data, column='amount', n=100, output_filename='test.parquet')
        print("✗ N too large validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ N too large validation passed")
        error_tests_passed += 1

    # Test negative n
    try:
        SAMPLE_DATA(ctx, sales_data, n_samples=-5, output_filename='test.parquet')
        print("✗ Negative n validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Negative n validation passed")
        error_tests_passed += 1

    # Test empty conditions dictionary
    try:
        FILTER_BY_MULTIPLE_CONDITIONS(ctx, sales_data, conditions_dict={}, output_filename='test.parquet')
        print("✗ Empty conditions validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty conditions validation passed")
        error_tests_passed += 1

    # Test filter that returns no results
    try:
        FILTER_BY_VALUE(ctx, sales_data, column='amount', operator='>', value=999999, output_filename='test.parquet')
        print("✗ No results validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ No results validation passed")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Test edge cases
    print("\n=== Testing Edge Cases ===")
    edge_tests_passed = 0
    edge_tests_failed = 0

    # Test single row DataFrame
    try:
        single_row_df = sales_data.head(1)
        result = TOP_N(ctx, single_row_df, column='amount', n=1, output_filename='single_row_test.parquet')
        if isinstance(result, Path) and result.exists():
            print("✓ Single row DataFrame test passed")
            edge_tests_passed += 1
        else:
            print("✗ Single row DataFrame test failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ Single row DataFrame test failed: {e}")
        edge_tests_failed += 1

    # Test exact date match
    try:
        result = FILTER_BY_DATE_RANGE(ctx, sales_data, date_column='transaction_date',
                                    start_date='2024-01-15', end_date='2024-01-15',
                                    output_filename='exact_date_test.parquet')
        if isinstance(result, Path) and result.exists():
            print("✓ Exact date match test passed")
            edge_tests_passed += 1
        else:
            print("✗ Exact date match test failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ Exact date match test failed: {e}")
        edge_tests_failed += 1

    # Test sampling entire dataset
    try:
        result = SAMPLE_DATA(ctx, customer_data, n_samples=len(customer_data),
                           random_state=42, output_filename='full_sample_test.parquet')
        if isinstance(result, Path) and result.exists():
            print("✓ Full dataset sampling test passed")
            edge_tests_passed += 1
        else:
            print("✗ Full dataset sampling test failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ Full dataset sampling test failed: {e}")
        edge_tests_failed += 1

    print(f"Edge case tests: {edge_tests_passed} passed, {edge_tests_failed} failed")
    all_passed &= (edge_tests_failed == 0)

    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the failed tests above.")
    print("="*50)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
