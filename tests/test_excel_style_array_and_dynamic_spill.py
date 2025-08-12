#!/usr/bin/env python3
"""
Test script for Excel-style array and dynamic spill functions.
Tests all functions in the excel_style_array_and_dynamic_spill_functions.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.excel_style_array_and_dynamic_spill_functions.excel_style_array_and_dynamic_spill_functions import (
    UNIQUE, SORT, SORTBY, FILTER, SEQUENCE, RAND, RANDBETWEEN, FREQUENCY,
    TRANSPOSE, MMULT, MINVERSE, MDETERM,
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
            if isinstance(expected, list):
                # Handle list comparisons (for functions that return lists)
                if isinstance(result, list):
                    if len(result) == len(expected):
                        # Use different tolerances for different types of comparisons
                        def compare_values(r, e):
                            if isinstance(r, list) and isinstance(e, list):
                                # Handle nested lists (for matrix operations)
                                if len(r) != len(e):
                                    return False
                                return all(compare_values(ri, ei) for ri, ei in zip(r, e))
                            elif isinstance(r, (int, float, Decimal)) and isinstance(e, (int, float, Decimal)):
                                # Use more relaxed tolerance for matrix operations
                                tolerance = 1e-2 if func_name in ['MINVERSE', 'MMULT'] else 1e-3
                                return abs(float(r) - float(e)) < tolerance
                            else:
                                return r == e

                        if all(compare_values(r, e) for r, e in zip(result, expected)):
                            print(f"✓ Test {i}: {description}")
                            passed += 1
                        else:
                            print(f"✗ Test {i}: {description}")
                            print(f"  Expected: {expected}, Got: {result}")
                            failed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected length {len(expected)}, Got length {len(result)}")
                        failed += 1
                elif isinstance(result, Path):
                    # Handle file output case
                    if isinstance(expected, Path):
                        if result.exists() and expected == result:
                            print(f"✓ Test {i}: {description}")
                            passed += 1
                        else:
                            print(f"✗ Test {i}: {description}")
                            print(f"  Expected file: {expected}, Got: {result}")
                            failed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected list: {expected}, Got file path: {result}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected list: {expected}, Got single value: {result}")
                    failed += 1
            elif isinstance(expected, (int, float)):
                # Handle numeric comparisons
                if isinstance(result, (int, float)):
                    if abs(result - expected) < 1e-6:
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected: {expected}, Got: {result}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected numeric: {expected}, Got: {result}")
                    failed += 1
            elif isinstance(expected, Path):
                # Handle file path comparisons
                if isinstance(result, Path):
                    if result.exists() and expected == result:
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected file: {expected}, Got: {result}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected file path: {expected}, Got: {result}")
                    failed += 1
            else:
                # Handle other types (string, etc.)
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
    """Run all tests for the Excel-style array and dynamic spill functions."""
    print("Starting comprehensive test of Excel-style array and dynamic spill functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    all_passed = True

    # Test UNIQUE function
    unique_tests = [
        ({'array': [1, 2, 2, 3, 3, 3]}, [1, 2, 3], "Basic unique values"),
        ({'array': [1, 2, 2, 3, 3, 3], 'exactly_once': True}, [1], "Values that appear exactly once"),
        ({'array': ['apple', 'banana', 'apple', 'cherry']}, ['apple', 'banana', 'cherry'], "Unique strings"),
        ({'array': pl.Series([1, 2, 2, 3, 3, 3])}, [1, 2, 3], "Unique with Polars Series"),
        ({'array': np.array([1, 2, 2, 3, 3, 3])}, [1, 2, 3], "Unique with NumPy array"),
        ({'array': "test_data.csv", 'output_filename': "unique_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/unique_results.parquet"), "Unique with CSV file input and output"),
        ({'array': "test_data.parquet", 'output_filename': "unique_results2.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/unique_results2.parquet"), "Unique with Parquet file input and output"),
    ]
    all_passed &= test_function("UNIQUE", UNIQUE, unique_tests, ctx=ctx)

    # Test SORT function
    sort_tests = [
        ({'array': [3, 1, 4, 1, 5]}, [1, 1, 3, 4, 5], "Basic ascending sort"),
        ({'array': [3, 1, 4, 1, 5], 'sort_order': -1}, [5, 4, 3, 1, 1], "Descending sort"),
        ({'array': ['cherry', 'apple', 'banana']}, ['apple', 'banana', 'cherry'], "Sort strings"),
        ({'array': pl.Series([3, 1, 4, 1, 5])}, [1, 1, 3, 4, 5], "Sort with Polars Series"),
        ({'array': np.array([3, 1, 4, 1, 5])}, [1, 1, 3, 4, 5], "Sort with NumPy array"),
        ({'array': "test_data.csv", 'output_filename': "sort_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/sort_results.parquet"), "Sort with CSV file input and output"),
        ({'array': "test_data.parquet", 'output_filename': "sort_results2.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/sort_results2.parquet"), "Sort with Parquet file input and output"),
    ]
    all_passed &= test_function("SORT", SORT, sort_tests, ctx=ctx)

    # Test SORTBY function
    sortby_tests = [
        ({'array': ['apple', 'banana', 'cherry'], 'by_arrays_and_orders': [3, 1, 2]}, ['banana', 'cherry', 'apple'], "Sort strings by numeric keys"),
        ({'array': [100, 200, 300], 'by_arrays_and_orders': [3, 1, 2]}, [200, 300, 100], "Sort numbers by numeric keys"),
        ({'array': ['c', 'a', 'b'], 'by_arrays_and_orders': ['z', 'x', 'y']}, ['a', 'b', 'c'], "Sort by string keys"),
        ({'array': pl.Series(['apple', 'banana', 'cherry']), 'by_arrays_and_orders': pl.Series([3, 1, 2])}, ['banana', 'cherry', 'apple'], "SORTBY with Polars Series"),
        ({'array': "test_data.csv", 'by_arrays_and_orders': [5, 4, 3, 2, 1], 'output_filename': "sortby_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/sortby_results.parquet"), "SORTBY with CSV file input and output"),
    ]
    all_passed &= test_function("SORTBY", SORTBY, sortby_tests, ctx=ctx)

    # Test FILTER function
    filter_tests = [
        ({'array': [1, 2, 3, 4, 5], 'include': [True, False, True, False, True]}, [1, 3, 5], "Basic filtering"),
        ({'array': ['a', 'b', 'c'], 'include': [False, False, False], 'if_empty': 'none'}, 'none', "Filter with empty result"),
        ({'array': [10, 20, 30, 40], 'include': [True, True, False, True]}, [10, 20, 40], "Filter numbers"),
        ({'array': pl.Series([1, 2, 3, 4, 5]), 'include': pl.Series([True, False, True, False, True])}, [1, 3, 5], "Filter with Polars Series"),
        ({'array': "test_data.csv", 'include': [True, False, True, False, True], 'output_filename': "filter_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/filter_results.parquet"), "Filter with CSV file input and output"),
    ]
    all_passed &= test_function("FILTER", FILTER, filter_tests, ctx=ctx)

    # Test SEQUENCE function
    sequence_tests = [
        ({'rows': 3}, [[1], [2], [3]], "Basic sequence"),
        ({'rows': 2, 'columns': 3, 'start': 5, 'step': 2}, [[5, 7, 9], [11, 13, 15]], "Sequence with custom parameters"),
        ({'rows': 3, 'columns': 2}, [[1, 2], [3, 4], [5, 6]], "Sequence with multiple columns"),
        ({'rows': 4, 'start': 10, 'step': 5}, [[10], [15], [20], [25]], "Sequence with custom start and step"),
        ({'rows': 2, 'columns': 2, 'output_filename': "sequence_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/sequence_results.parquet"), "Sequence with output file"),
    ]
    all_passed &= test_function("SEQUENCE", SEQUENCE, sequence_tests, ctx=ctx)

    # Test RAND function
    rand_tests = [
        ({}, "range_check", "Random number between 0 and 1"),
    ]

    # Special handling for RAND since it returns random values
    print(f"\n=== Testing RAND ===")
    rand_passed = 0
    rand_failed = 0

    for i in range(5):  # Test multiple times
        try:
            result = RAND(ctx)
            if 0 <= result < 1:
                print(f"✓ Test {i+1}: Random value {result} is in valid range [0, 1)")
                rand_passed += 1
            else:
                print(f"✗ Test {i+1}: Random value {result} is outside valid range [0, 1)")
                rand_failed += 1
        except Exception as e:
            print(f"✗ Test {i+1}: Error: {type(e).__name__}: {str(e)}")
            rand_failed += 1

    print(f"Results for RAND: {rand_passed} passed, {rand_failed} failed")
    all_passed &= (rand_failed == 0)

    # Test RANDBETWEEN function
    randbetween_tests = [
        ({'bottom': 1, 'top': 10}, "range_check", "Random integer between 1 and 10"),
        ({'bottom': -5, 'top': 5}, "range_check", "Random integer between -5 and 5"),
        ({'bottom': 100, 'top': 100}, 100, "Random integer with same bounds"),
    ]

    # Special handling for RANDBETWEEN since it returns random values
    print(f"\n=== Testing RANDBETWEEN ===")
    randbetween_passed = 0
    randbetween_failed = 0

    # Test range 1-10
    for i in range(5):
        try:
            result = RANDBETWEEN(ctx, 1, 10)
            if 1 <= result <= 10:
                print(f"✓ Test {i+1}: Random value {result} is in valid range [1, 10]")
                randbetween_passed += 1
            else:
                print(f"✗ Test {i+1}: Random value {result} is outside valid range [1, 10]")
                randbetween_failed += 1
        except Exception as e:
            print(f"✗ Test {i+1}: Error: {type(e).__name__}: {str(e)}")
            randbetween_failed += 1

    # Test same bounds
    try:
        result = RANDBETWEEN(ctx, 100, 100)
        if result == 100:
            print(f"✓ Test 6: Same bounds test passed: {result}")
            randbetween_passed += 1
        else:
            print(f"✗ Test 6: Same bounds test failed: expected 100, got {result}")
            randbetween_failed += 1
    except Exception as e:
        print(f"✗ Test 6: Error: {type(e).__name__}: {str(e)}")
        randbetween_failed += 1

    print(f"Results for RANDBETWEEN: {randbetween_passed} passed, {randbetween_failed} failed")
    all_passed &= (randbetween_failed == 0)

    # Test FREQUENCY function
    frequency_tests = [
        ({'data_array': [1, 2, 3, 4, 5, 6], 'bins_array': [2, 4, 6]}, [2, 2, 2, 0], "Basic frequency distribution"),
        ({'data_array': [1.5, 2.5, 3.5, 4.5], 'bins_array': [2, 3, 4]}, [1, 1, 1, 1], "Frequency with decimals"),
        ({'data_array': [1, 1, 2, 2, 3, 3], 'bins_array': [1, 2, 3]}, [2, 2, 2, 0], "Frequency with duplicates"),
        ({'data_array': pl.Series([1, 2, 3, 4, 5, 6]), 'bins_array': pl.Series([2, 4, 6])}, [2, 2, 2, 0], "Frequency with Polars Series"),
        ({'data_array': "test_data.csv", 'bins_array': [2, 4], 'output_filename': "frequency_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/frequency_results.parquet"), "Frequency with CSV file input and output"),
    ]
    all_passed &= test_function("FREQUENCY", FREQUENCY, frequency_tests, ctx=ctx)

    # Test TRANSPOSE function
    transpose_tests = [
        ({'array': [[1, 2, 3], [4, 5, 6]]}, [[1, 4], [2, 5], [3, 6]], "Basic transpose"),
        ({'array': [[1, 2], [3, 4], [5, 6]]}, [[1, 3, 5], [2, 4, 6]], "Transpose 3x2 matrix"),
        ({'array': [[1]]}, [[1]], "Transpose 1x1 matrix"),
        ({'array': [[1, 2, 3, 4]]}, [[1], [2], [3], [4]], "Transpose row vector"),
        ({'array': [[1, 2, 3], [4, 5, 6]], 'output_filename': "transpose_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/transpose_results.parquet"), "Transpose with output file"),
    ]
    all_passed &= test_function("TRANSPOSE", TRANSPOSE, transpose_tests, ctx=ctx)

    # Test MMULT function
    mmult_tests = [
        ({'array1': [[1, 2], [3, 4]], 'array2': [[5, 6], [7, 8]]}, [[19.0, 22.0], [43.0, 50.0]], "Basic matrix multiplication"),
        ({'array1': [[1, 2, 3]], 'array2': [[4], [5], [6]]}, [[32.0]], "Row vector × column vector"),
        ({'array1': [[1, 0], [0, 1]], 'array2': [[5, 6], [7, 8]]}, [[5.0, 6.0], [7.0, 8.0]], "Identity matrix multiplication"),
        ({'array1': [[2, 0], [0, 3]], 'array2': [[1, 2], [3, 4]]}, [[2.0, 4.0], [9.0, 12.0]], "Diagonal matrix multiplication"),
        ({'array1': [[1, 2], [3, 4]], 'array2': [[5, 6], [7, 8]], 'output_filename': "mmult_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/mmult_results.parquet"), "Matrix multiplication with output file"),
    ]
    all_passed &= test_function("MMULT", MMULT, mmult_tests, ctx=ctx)

    # Test MINVERSE function
    minverse_tests = [
        ({'array': [[1, 2], [3, 4]]}, [[-2.0, 1.0], [1.5, -0.5]], "Basic matrix inverse"),
        ({'array': [[2, 0], [0, 2]]}, [[0.5, 0.0], [0.0, 0.5]], "Diagonal matrix inverse"),
        ({'array': [[1, 0], [0, 1]]}, [[1.0, 0.0], [0.0, 1.0]], "Identity matrix inverse"),
        ({'array': [[4, 7], [2, 6]]}, [[0.6, -0.7], [-0.2, 0.4]], "Another matrix inverse"),
        ({'array': [[1, 2], [3, 4]], 'output_filename': "minverse_results.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/minverse_results.parquet"), "Matrix inverse with output file"),
    ]
    all_passed &= test_function("MINVERSE", MINVERSE, minverse_tests, ctx=ctx)

    # Test MDETERM function
    mdeterm_tests = [
        ({'array': [[1, 2], [3, 4]]}, -2.0, "Basic determinant"),
        ({'array': [[2, 0], [0, 2]]}, 4.0, "Diagonal matrix determinant"),
        ({'array': [[1, 0], [0, 1]]}, 1.0, "Identity matrix determinant"),
        ({'array': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, 0.0, "Singular matrix determinant"),
        ({'array': [[2, 1], [1, 2]]}, 3.0, "Another determinant"),
    ]
    all_passed &= test_function("MDETERM", MDETERM, mdeterm_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test empty input validation
    try:
        UNIQUE(ctx, [])
        print("✗ Empty input validation failed for UNIQUE")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty input validation passed for UNIQUE")
        error_tests_passed += 1

    # Test invalid RANDBETWEEN bounds
    try:
        RANDBETWEEN(ctx, 10, 5)
        print("✗ Invalid bounds validation failed for RANDBETWEEN")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid bounds validation passed for RANDBETWEEN")
        error_tests_passed += 1

    # Test mismatched array lengths for SORTBY
    try:
        SORTBY(ctx, [1, 2, 3], by_arrays_and_orders=[1, 2])
        print("✗ Mismatched lengths validation failed for SORTBY")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Mismatched lengths validation passed for SORTBY")
        error_tests_passed += 1

    # Test singular matrix for MINVERSE
    try:
        MINVERSE(ctx, [[1, 2], [2, 4]])
        print("✗ Singular matrix validation failed for MINVERSE")
        error_tests_failed += 1
    except CalculationError:
        print("✓ Singular matrix validation passed for MINVERSE")
        error_tests_passed += 1

    # Test non-square matrix for MDETERM
    try:
        MDETERM(ctx, [[1, 2, 3], [4, 5, 6]])
        print("✗ Non-square matrix validation failed for MDETERM")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Non-square matrix validation passed for MDETERM")
        error_tests_passed += 1

    # Test incompatible matrices for MMULT
    try:
        MMULT(ctx, [[1, 2]], array2=[[1], [2], [3]])
        print("✗ Incompatible matrices validation failed for MMULT")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Incompatible matrices validation passed for MMULT")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

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
