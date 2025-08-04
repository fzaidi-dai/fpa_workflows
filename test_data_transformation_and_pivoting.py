#!/usr/bin/env python3
"""
Test script for data transformation and pivoting functions.
Tests all functions in the data_transformation_and_pivoting.py module.
"""

import sys
import traceback
from pathlib import Path
import polars as pl
import numpy as np

# Import the functions to test
from tools.core_data_and_math_utils.data_transformation_and_pivoting.data_transformation_and_pivoting import (
    PIVOT_TABLE, UNPIVOT, GROUP_BY, CROSS_TAB, GROUP_BY_AGG, STACK, UNSTACK,
    MERGE, CONCAT, FILL_FORWARD, INTERPOLATE,
    ValidationError, CalculationError, ConfigurationError, DataQualityError
)

# Import FinnDeps and RunContext for testing
from tools.finn_deps import FinnDeps, RunContext


def create_test_data():
    """Create test DataFrames for testing."""
    # Sales data for testing
    sales_df = pl.DataFrame({
        "region": ["North", "South", "North", "South", "East", "East"],
        "product": ["A", "A", "B", "B", "A", "B"],
        "quarter": ["Q1", "Q1", "Q1", "Q1", "Q2", "Q2"],
        "revenue": [100, 150, 200, 120, 180, 160],
        "units": [10, 15, 20, 12, 18, 16],
        "cost": [80, 120, 160, 96, 144, 128]
    })

    # Customer data for testing merges
    customer_df = pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "region": ["North", "South", "East", "West", "North"],
        "segment": ["Premium", "Standard", "Premium", "Standard", "Premium"]
    })

    # Orders data for testing merges
    orders_df = pl.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "customer_id": [1, 2, 1, 3, 2],
        "amount": [500, 300, 750, 400, 600],
        "date": ["2023-01-15", "2023-01-20", "2023-02-10", "2023-02-15", "2023-03-01"]
    })

    # Wide format data for unpivoting
    wide_df = pl.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "Q1": [100, 150, 200],
        "Q2": [120, 180, 220],
        "Q3": [110, 160, 210],
        "Q4": [130, 170, 230]
    })

    # Data with nulls for fill/interpolate testing
    null_df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "value1": [10.0, None, 30.0, None, 50.0, 60.0],
        "value2": [100.0, 200.0, None, 400.0, None, 600.0],
        "category": ["A", "A", "B", "B", "C", "C"]
    })

    return sales_df, customer_df, orders_df, wide_df, null_df


def test_function(func_name, func, test_cases, ctx=None):
    """Test a function with multiple test cases."""
    print(f"\n=== Testing {func_name} ===")
    passed = 0
    failed = 0

    for i, (args, expected_type, description) in enumerate(test_cases, 1):
        try:
            # Add context parameter if provided
            if ctx is not None:
                args = {'run_context': ctx, **args}
            result = func(**args)

            # Check if result is expected type
            if expected_type == "DataFrame":
                if isinstance(result, pl.DataFrame):
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected DataFrame, Got: {type(result)}")
                    failed += 1
            elif expected_type == "Path":
                if isinstance(result, Path):
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected Path, Got: {type(result)}")
                    failed += 1
            else:
                print(f"✓ Test {i}: {description}")
                passed += 1

        except Exception as e:
            print(f"✗ Test {i}: {description}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            failed += 1

    print(f"Results for {func_name}: {passed} passed, {failed} failed")
    return failed == 0


def run_all_tests():
    """Run all tests for the data transformation and pivoting functions."""
    print("Starting comprehensive test of data transformation and pivoting functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    # Create test data
    sales_df, customer_df, orders_df, wide_df, null_df = create_test_data()

    # Save test data to files for file input testing
    sales_df.write_csv("scratch_pad/analysis/sales_test.csv")
    sales_df.write_parquet("scratch_pad/analysis/sales_test.parquet")
    customer_df.write_csv("scratch_pad/analysis/customer_test.csv")
    orders_df.write_parquet("scratch_pad/analysis/orders_test.parquet")
    wide_df.write_parquet("scratch_pad/analysis/wide_test.parquet")
    null_df.write_parquet("scratch_pad/analysis/null_test.parquet")

    all_passed = True

    # Test PIVOT_TABLE function
    pivot_tests = [
        ({'df': sales_df, 'index_cols': ['region'], 'value_cols': ['revenue'], 'agg_func': 'sum'},
         "DataFrame", "Basic pivot table with sum aggregation"),
        ({'df': sales_df, 'index_cols': ['region', 'product'], 'value_cols': ['revenue', 'units'], 'agg_func': 'mean'},
         "DataFrame", "Multi-index pivot with mean aggregation"),
        ({'df': "sales_test.csv", 'index_cols': ['region'], 'value_cols': ['revenue'], 'agg_func': 'sum'},
         "DataFrame", "Pivot table with CSV file input"),
        ({'df': sales_df, 'index_cols': ['region'], 'value_cols': ['revenue'], 'agg_func': 'sum', 'output_filename': 'pivot_results.parquet'},
         "Path", "Pivot table with file output"),
    ]
    all_passed &= test_function("PIVOT_TABLE", PIVOT_TABLE, pivot_tests, ctx=ctx)

    # Test UNPIVOT function
    unpivot_tests = [
        ({'df': wide_df, 'identifier_cols': ['customer_id', 'name'], 'value_cols': ['Q1', 'Q2', 'Q3', 'Q4']},
         "DataFrame", "Basic unpivot operation"),
        ({'df': wide_df, 'identifier_cols': ['customer_id'], 'value_cols': ['Q1', 'Q2']},
         "DataFrame", "Partial unpivot with subset of columns"),
        ({'df': "wide_test.parquet", 'identifier_cols': ['customer_id'], 'value_cols': ['Q1', 'Q2'], 'output_filename': 'unpivot_results.parquet'},
         "Path", "Unpivot with file input and output"),
    ]
    all_passed &= test_function("UNPIVOT", UNPIVOT, unpivot_tests, ctx=ctx)

    # Test GROUP_BY function
    groupby_tests = [
        ({'df': sales_df, 'grouping_cols': ['region'], 'agg_func': 'sum'},
         "DataFrame", "Basic group by with sum"),
        ({'df': sales_df, 'grouping_cols': ['region', 'product'], 'agg_func': 'mean'},
         "DataFrame", "Multi-column group by with mean"),
        ({'df': "sales_test.parquet", 'grouping_cols': ['region'], 'agg_func': 'count', 'output_filename': 'groupby_results.parquet'},
         "Path", "Group by with file input and output"),
    ]
    all_passed &= test_function("GROUP_BY", GROUP_BY, groupby_tests, ctx=ctx)

    # Test CROSS_TAB function
    crosstab_tests = [
        ({'df': sales_df, 'row_vars': ['region'], 'col_vars': ['product'], 'values': ['revenue']},
         "DataFrame", "Basic cross-tabulation"),
        ({'df': sales_df, 'row_vars': ['region'], 'col_vars': ['quarter'], 'values': ['units']},
         "DataFrame", "Cross-tab with different variables"),
        ({'df': "sales_test.csv", 'row_vars': ['region'], 'col_vars': ['product'], 'values': ['revenue'], 'output_filename': 'crosstab_results.parquet'},
         "Path", "Cross-tab with file input and output"),
    ]
    all_passed &= test_function("CROSS_TAB", CROSS_TAB, crosstab_tests, ctx=ctx)

    # Test GROUP_BY_AGG function
    groupby_agg_tests = [
        ({'df': sales_df, 'group_by_cols': ['region'], 'agg_dict': {'revenue': 'sum', 'units': 'mean'}},
         "DataFrame", "Group by with multiple aggregations"),
        ({'df': sales_df, 'group_by_cols': ['region', 'product'], 'agg_dict': {'revenue': 'sum', 'cost': 'min', 'units': 'max'}},
         "DataFrame", "Multi-column group by with different aggregations"),
        ({'df': "sales_test.parquet", 'group_by_cols': ['region'], 'agg_dict': {'revenue': 'sum'}, 'output_filename': 'groupby_agg_results.parquet'},
         "Path", "Group by agg with file input and output"),
    ]
    all_passed &= test_function("GROUP_BY_AGG", GROUP_BY_AGG, groupby_agg_tests, ctx=ctx)

    # Test STACK function
    stack_tests = [
        ({'df': wide_df, 'columns_to_stack': ['Q1', 'Q2', 'Q3', 'Q4']},
         "DataFrame", "Basic stack operation"),
        ({'df': wide_df, 'columns_to_stack': ['Q1', 'Q2']},
         "DataFrame", "Partial stack operation"),
        ({'df': "wide_test.parquet", 'columns_to_stack': ['Q1', 'Q2'], 'output_filename': 'stack_results.parquet'},
         "Path", "Stack with file input and output"),
    ]
    all_passed &= test_function("STACK", STACK, stack_tests, ctx=ctx)

    # Create stacked data for unstack testing
    stacked_df = wide_df.unpivot(
        index=['customer_id', 'name'],
        on=['Q1', 'Q2'],
        variable_name='quarter',
        value_name='amount'
    )

    # Test UNSTACK function
    unstack_tests = [
        ({'df': stacked_df, 'level_to_unstack': 'quarter'},
         "DataFrame", "Basic unstack operation"),
        ({'df': stacked_df, 'level_to_unstack': 'quarter', 'output_filename': 'unstack_results.parquet'},
         "Path", "Unstack with file output"),
    ]
    all_passed &= test_function("UNSTACK", UNSTACK, unstack_tests, ctx=ctx)

    # Test MERGE function
    merge_tests = [
        ({'left_df': orders_df, 'right_df': customer_df, 'join_keys': 'customer_id', 'join_type': 'inner'},
         "DataFrame", "Inner join on single key"),
        ({'left_df': orders_df, 'right_df': customer_df, 'join_keys': 'customer_id', 'join_type': 'left'},
         "DataFrame", "Left join on single key"),
        ({'left_df': "orders_test.parquet", 'right_df': "customer_test.csv", 'join_keys': 'customer_id', 'join_type': 'inner', 'output_filename': 'merge_results.parquet'},
         "Path", "Merge with file inputs and output"),
    ]
    all_passed &= test_function("MERGE", MERGE, merge_tests, ctx=ctx)

    # Test CONCAT function
    # Create additional DataFrames for concatenation
    df1 = sales_df.head(3)
    df2 = sales_df.tail(3)
    df3 = customer_df.select(['customer_id', 'name'])
    df4 = customer_df.select(['region', 'segment'])

    concat_tests = [
        ({'dataframes': [df1, df2], 'axis': 0},
         "DataFrame", "Vertical concatenation"),
        ({'dataframes': [df3, df4], 'axis': 1},
         "DataFrame", "Horizontal concatenation"),
        ({'dataframes': ["sales_test.csv", "sales_test.parquet"], 'axis': 0, 'output_filename': 'concat_results.parquet'},
         "Path", "Concat with file inputs and output"),
    ]
    all_passed &= test_function("CONCAT", CONCAT, concat_tests, ctx=ctx)

    # Test FILL_FORWARD function
    fill_forward_tests = [
        ({'df': null_df},
         "DataFrame", "Basic forward fill"),
        ({'df': "null_test.parquet", 'output_filename': 'fill_forward_results.parquet'},
         "Path", "Fill forward with file input and output"),
    ]
    all_passed &= test_function("FILL_FORWARD", FILL_FORWARD, fill_forward_tests, ctx=ctx)

    # Test INTERPOLATE function
    interpolate_tests = [
        ({'df': null_df, 'method': 'linear'},
         "DataFrame", "Linear interpolation"),
        ({'df': "null_test.parquet", 'method': 'linear', 'output_filename': 'interpolate_results.parquet'},
         "Path", "Interpolate with file input and output"),
    ]
    all_passed &= test_function("INTERPOLATE", INTERPOLATE, interpolate_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test empty DataFrame validation
    try:
        empty_df = pl.DataFrame()
        PIVOT_TABLE(ctx, empty_df, index_cols=['region'], value_cols=['revenue'], agg_func='sum')
        print("✗ Empty DataFrame validation failed for PIVOT_TABLE")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty DataFrame validation passed for PIVOT_TABLE")
        error_tests_passed += 1

    # Test missing columns validation
    try:
        UNPIVOT(ctx, sales_df, identifier_cols=['nonexistent'], value_cols=['Q1'])
        print("✗ Missing columns validation failed for UNPIVOT")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Missing columns validation passed for UNPIVOT")
        error_tests_passed += 1

    # Test invalid aggregation function
    try:
        GROUP_BY(ctx, sales_df, grouping_cols=['region'], agg_func='invalid_func')
        print("✗ Invalid aggregation function validation failed for GROUP_BY")
        error_tests_failed += 1
    except (ConfigurationError, CalculationError):
        print("✓ Invalid aggregation function validation passed for GROUP_BY")
        error_tests_passed += 1

    # Test invalid join type
    try:
        MERGE(ctx, orders_df, customer_df, join_keys='customer_id', join_type='invalid_join')
        print("✗ Invalid join type validation failed for MERGE")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid join type validation passed for MERGE")
        error_tests_passed += 1

    # Test invalid axis for concatenation
    try:
        CONCAT(ctx, [sales_df, customer_df], axis=2)
        print("✗ Invalid axis validation failed for CONCAT")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid axis validation passed for CONCAT")
        error_tests_passed += 1

    # Test invalid interpolation method
    try:
        INTERPOLATE(ctx, null_df, method='invalid_method')
        print("✗ Invalid interpolation method validation failed for INTERPOLATE")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid interpolation method validation passed for INTERPOLATE")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Test specific functionality
    print("\n=== Testing Specific Functionality ===")
    functionality_tests_passed = 0
    functionality_tests_failed = 0

    try:
        # Test pivot table result structure
        pivot_result = PIVOT_TABLE(ctx, sales_df, index_cols=['region'], value_cols=['revenue'], agg_func='sum')
        if 'region' in pivot_result.columns and 'revenue_sum' in pivot_result.columns:
            print("✓ Pivot table structure validation passed")
            functionality_tests_passed += 1
        else:
            print("✗ Pivot table structure validation failed")
            functionality_tests_failed += 1
    except Exception as e:
        print(f"✗ Pivot table structure test failed: {e}")
        functionality_tests_failed += 1

    try:
        # Test unpivot result structure
        unpivot_result = UNPIVOT(ctx, wide_df, identifier_cols=['customer_id'], value_cols=['Q1', 'Q2'])
        expected_cols = {'customer_id', 'variable', 'value'}
        if expected_cols.issubset(set(unpivot_result.columns)):
            print("✓ Unpivot result structure validation passed")
            functionality_tests_passed += 1
        else:
            print("✗ Unpivot result structure validation failed")
            functionality_tests_failed += 1
    except Exception as e:
        print(f"✗ Unpivot structure test failed: {e}")
        functionality_tests_failed += 1

    try:
        # Test merge result
        merge_result = MERGE(ctx, orders_df, customer_df, join_keys='customer_id', join_type='inner')
        if len(merge_result) > 0 and 'customer_id' in merge_result.columns:
            print("✓ Merge operation validation passed")
            functionality_tests_passed += 1
        else:
            print("✗ Merge operation validation failed")
            functionality_tests_failed += 1
    except Exception as e:
        print(f"✗ Merge operation test failed: {e}")
        functionality_tests_failed += 1

    try:
        # Test fill forward effectiveness
        fill_result = FILL_FORWARD(ctx, null_df)
        original_nulls = null_df.null_count().sum_horizontal()[0]
        result_nulls = fill_result.null_count().sum_horizontal()[0]
        if result_nulls < original_nulls:
            print("✓ Fill forward effectiveness validation passed")
            functionality_tests_passed += 1
        else:
            print("✗ Fill forward effectiveness validation failed")
            functionality_tests_failed += 1
    except Exception as e:
        print(f"✗ Fill forward effectiveness test failed: {e}")
        functionality_tests_failed += 1

    print(f"Functionality tests: {functionality_tests_passed} passed, {functionality_tests_failed} failed")
    all_passed &= (functionality_tests_failed == 0)

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
