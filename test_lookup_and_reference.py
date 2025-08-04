#!/usr/bin/env python3
"""
Test script for lookup and reference functions.
Tests all functions in the lookup_and_reference_functions.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.lookup_and_reference_functions.lookup_and_reference_functions import (
    VLOOKUP, HLOOKUP, INDEX, MATCH, XLOOKUP, LOOKUP, CHOOSE, OFFSET, INDIRECT,
    ADDRESS, ROW, COLUMN, ROWS, COLUMNS,
    ValidationError, CalculationError, DataQualityError, ConfigurationError
)

from tools.finn_deps import FinnDeps, RunContext

def test_function(func_name, func, test_cases):
    """Test a function with multiple test cases."""
    print(f"\n=== Testing {func_name} ===")
    passed = 0
    failed = 0

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path("scratch_pad").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    for i, (args, expected, description) in enumerate(test_cases, 1):
        try:
            # Add context to all function calls
            result = func(ctx, **args)

            # Handle comparison of different types for numeric values
            if isinstance(expected, (int, float)):
                # Convert both to Decimal for comparison if they're numeric
                if isinstance(result, (int, float, Decimal)):
                    expected_decimal = Decimal(str(expected))
                    if isinstance(result, Decimal):
                        result_decimal = result
                    else:
                        result_decimal = Decimal(str(result))
                    if abs(result_decimal - expected_decimal) < Decimal('1e-10'):
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected: {expected}, Got: {result}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            elif isinstance(expected, Decimal):
                # If result is numeric but not Decimal, convert it.
                if not isinstance(result, Decimal) and isinstance(result, (int, float)):
                    result = Decimal(str(result))
                # Now compare as Decimals
                if isinstance(result, Decimal) and abs(result - expected) < Decimal('1e-10'):
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            elif isinstance(expected, list):
                if isinstance(result, list) and len(result) == len(expected):
                    match = True
                    for a, b in zip(result, expected):
                        # Convert numeric values to Decimal for comparison
                        if isinstance(b, (int, float)):
                            expected_decimal = Decimal(str(b))
                            if isinstance(a, (int, float, Decimal)):
                                if isinstance(a, Decimal):
                                    result_decimal = a
                                else:
                                    result_decimal = Decimal(str(a))
                                if abs(result_decimal - expected_decimal) >= Decimal('1e-10'):
                                    match = False
                                    break
                            else:
                                match = False
                                break
                        elif isinstance(b, Decimal):
                            if not isinstance(a, Decimal) and isinstance(a, (int, float)):
                                a = Decimal(str(a))
                            if not (isinstance(a, Decimal) and abs(a - b) < Decimal('1e-10')):
                                match = False
                                break
                        else:
                            if a != b:
                                match = False
                                break
                    if match:
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected: {expected}, Got: {result}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            else:
                # For other types, use equality
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
    """Run all tests for the lookup and reference functions."""
    print("Starting comprehensive test of lookup and reference functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path("scratch_pad").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    all_passed = True

    # Test VLOOKUP function
    vlookup_tests = [
        ({'lookup_value': "B", 'table_array': [["A", 1], ["B", 2], ["C", 3]], 'col_index': 2}, 2, "Basic exact match"),
        ({'lookup_value': "CUST001", 'table_array': [["CUST001", "Premium", 0.95], ["CUST002", "Standard", 1.00]], 'col_index': 3}, 0.95, "Customer pricing lookup"),
        ({'lookup_value': "EMP002", 'table_array': [["EMP001", "John", 75000], ["EMP002", "Jane", 55000]], 'col_index': 3}, 55000, "Employee salary lookup"),
        ({'lookup_value': 75, 'table_array': [[50, "Low"], [100, "High"]], 'col_index': 2, 'range_lookup': True}, "Low", "Range lookup - approximate match"),
        ({'lookup_value': "A", 'table_array': pl.DataFrame([["A", 10], ["B", 20]], strict=False, orient="row"), 'col_index': 2}, 10, "Polars DataFrame input"),
        ({'lookup_value': 1, 'table_array': np.array([[1, 100], [2, 200]]), 'col_index': 2}, 100, "NumPy array input"),
    ]
    all_passed &= test_function("VLOOKUP", VLOOKUP, vlookup_tests)

    # Test HLOOKUP function
    hlookup_tests = [
        ({'lookup_value': "B", 'table_array': [["A", "B", "C"], [1, 2, 3]], 'row_index': 2}, 2, "Basic exact match"),
        ({'lookup_value': "Q2 2024", 'table_array': [["Q1 2024", "Q2 2024", "Q3 2024"], [1000000, 1100000, 1200000]], 'row_index': 2}, 1100000, "Quarterly revenue lookup"),
        ({'lookup_value': "Mar", 'table_array': [["Jan", "Feb", "Mar"], [100, 105, 110]], 'row_index': 2}, 110, "Monthly budget lookup"),
        ({'lookup_value': "B", 'table_array': [["A", "C", "E"], [10, 30, 50]], 'row_index': 2, 'range_lookup': True}, 10, "Range lookup - approximate match"),
        ({'lookup_value': "X", 'table_array': pl.DataFrame([["X", "Y"], [100, 200]], strict=False, orient="row"), 'row_index': 2}, 100, "Polars DataFrame input"),
    ]
    all_passed &= test_function("HLOOKUP", HLOOKUP, hlookup_tests)

    # Test INDEX function
    index_tests = [
        ({'array': [[1, 2], [3, 4]], 'row_num': 2, 'column_num': 1}, 3, "2D array - row 2, col 1"),
        ({'array': [["Revenue", 1000000], ["COGS", 600000]], 'row_num': 1, 'column_num': 2}, 1000000, "Financial data extraction"),
        ({'array': [10, 20, 30], 'row_num': 2}, 20, "1D array - position 2"),
        ({'array': [50000, 52000, 48000], 'row_num': 3}, 48000, "Budget array extraction"),
        ({'array': pl.DataFrame([[1, 2], [3, 4]], strict=False, orient="row"), 'row_num': 1, 'column_num': 2}, 2, "Polars DataFrame"),
        ({'array': pl.Series([100, 200, 300]), 'row_num': 2}, 200, "Polars Series"),
        ({'array': np.array([[1, 2], [3, 4]]), 'row_num': 2, 'column_num': 2}, 4, "NumPy 2D array"),
        ({'array': np.array([10, 20, 30]), 'row_num': 1}, 10, "NumPy 1D array"),
    ]
    all_passed &= test_function("INDEX", INDEX, index_tests)

    # Test MATCH function
    match_tests = [
        ({'lookup_value': "B", 'lookup_array': ["A", "B", "C"], 'match_type': 0}, 2, "Exact match - text"),
        ({'lookup_value': 20, 'lookup_array': [10, 20, 30], 'match_type': 0}, 2, "Exact match - number"),
        ({'lookup_value': "Apr", 'lookup_array': ["Jan", "Feb", "Mar", "Apr"], 'match_type': 0}, 4, "Month position lookup"),
        ({'lookup_value': 25, 'lookup_array': [10, 20, 30, 40], 'match_type': 1}, 2, "Largest value <= lookup_value"),
        ({'lookup_value': 15, 'lookup_array': [40, 30, 20, 10], 'match_type': -1}, 3, "Smallest value >= lookup_value"),
        ({'lookup_value': "X", 'lookup_array': pl.Series(["X", "Y", "Z"]), 'match_type': 0}, 1, "Polars Series input"),
        ({'lookup_value': 100, 'lookup_array': np.array([50, 100, 150]), 'match_type': 0}, 2, "NumPy array input"),
    ]
    all_passed &= test_function("MATCH", MATCH, match_tests)

    # Test XLOOKUP function
    xlookup_tests = [
        ({'lookup_value': "B", 'lookup_array': ["A", "B", "C"], 'return_array': [1, 2, 3]}, 2, "Basic exact match"),
        ({'lookup_value': "D", 'lookup_array': ["A", "B", "C"], 'return_array': [1, 2, 3], 'if_not_found': "Not Found"}, "Not Found", "Not found fallback"),
        ({'lookup_value': "CUST005", 'lookup_array': ["CUST001", "CUST002"], 'return_array': [50000, 75000], 'if_not_found': 10000}, 10000, "Customer credit limit fallback"),
        ({'lookup_value': "PROD-A123", 'lookup_array': ["PROD-A*", "PROD-B*"], 'return_array': [100, 150], 'match_mode': 2}, 100, "Wildcard pattern matching"),
        ({'lookup_value': 75, 'lookup_array': [50, 100, 150], 'return_array': ["Low", "Medium", "High"], 'match_mode': -1}, "Low", "Next smallest match"),
        ({'lookup_value': 75, 'lookup_array': [50, 100, 150], 'return_array': ["Low", "Medium", "High"], 'match_mode': 1}, "Medium", "Next largest match"),
        ({'lookup_value': "B", 'lookup_array': ["A", "B", "B", "C"], 'return_array': [1, 2, 3, 4], 'search_mode': -1}, 3, "Search from last to first"),
        ({'lookup_value': "X", 'lookup_array': pl.Series(["X", "Y"]), 'return_array': pl.Series([10, 20])}, 10, "Polars Series input"),
    ]
    all_passed &= test_function("XLOOKUP", XLOOKUP, xlookup_tests)

    # Test LOOKUP function
    lookup_tests = [
        ({'lookup_value': 7, 'lookup_vector': [1, 5, 10], 'result_vector': ["Low", "Medium", "High"]}, "Medium", "Tax bracket style lookup"),
        ({'lookup_value': 15, 'lookup_vector': [1, 5, 10, 20]}, 10, "Simple lookup without result vector"),
        ({'lookup_value': 75000, 'lookup_vector': [0, 50000, 100000], 'result_vector': [0.10, 0.25, 0.35]}, 0.25, "Tax rate lookup"),
        ({'lookup_value': 60000, 'lookup_vector': [0, 25000, 50000, 100000], 'result_vector': [0.02, 0.03, 0.04, 0.05]}, 0.04, "Commission tier lookup"),
        ({'lookup_value': 85, 'lookup_vector': [0, 60, 70, 80, 90], 'result_vector': ["F", "D", "C", "B", "A"]}, "B", "Performance rating lookup"),
        ({'lookup_value': 5, 'lookup_vector': pl.Series([1, 3, 7]), 'result_vector': pl.Series([10, 30, 70])}, 30, "Polars Series input"),
    ]
    all_passed &= test_function("LOOKUP", LOOKUP, lookup_tests)

    # Test CHOOSE function
    print(f"\n=== Testing CHOOSE ===")
    choose_passed = 0
    choose_failed = 0
    choose_test_cases = [
        (2, ["Apple", "Banana", "Cherry"], "Banana", "Basic fruit selection"),
        (1, [100, 200, 300], 100, "Numeric selection"),
        (2, ["Conservative", "Moderate", "Aggressive"], "Moderate", "Financial scenario selection"),
        (3, ["Operations", "Marketing", "R&D", "Admin"], "R&D", "Budget category selection"),
        (4, [0.95, 1.02, 1.08, 1.15], 1.15, "Performance metric selection"),
        (1, ["Single"], "Single", "Single value selection"),
        (5, ["A", "B", "C", "D", "E"], "E", "Last value selection"),
    ]
    for i, (index_num, values, expected, description) in enumerate(choose_test_cases, 1):
        try:
            result = CHOOSE(index_num, *values)
            # Handle Decimal comparison for numeric values
            if isinstance(expected, (int, float)) and isinstance(result, Decimal):
                expected_decimal = Decimal(str(expected))
                if abs(result - expected_decimal) < Decimal('1e-10'):
                    print(f"✓ Test {i}: {description}")
                    choose_passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    choose_failed += 1
            elif result == expected:
                print(f"✓ Test {i}: {description}")
                choose_passed += 1
            else:
                print(f"✗ Test {i}: {description}")
                print(f"  Expected: {expected}, Got: {result}")
                choose_failed += 1
        except Exception as e:
            print(f"✗ Test {i}: {description}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            choose_failed += 1
    print(f"Results for CHOOSE: {choose_passed} passed, {choose_failed} failed")
    all_passed &= (choose_failed == 0)

    # Test OFFSET function
    offset_tests = [
        ({'reference': [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'rows': 1, 'cols': 1}, 5, "Single value offset"),
        ({'reference': [[50000, 52000], [45000, 47000]], 'rows': 1, 'cols': 1}, 47000, "Budget value offset"),
        ({'reference': pl.DataFrame([[10, 20], [30, 40]], strict=False), 'rows': 1, 'cols': 0}, 20, "Polars DataFrame offset"),
    ]
    print(f"\n=== Testing OFFSET ===")
    offset_passed = 0
    offset_failed = 0
    for i, (args, expected, description) in enumerate(offset_tests, 1):
        try:
            result = OFFSET(ctx, **args)
            if not isinstance(expected, pl.DataFrame) and isinstance(expected, (int, float)):
                expected_value = Decimal(str(expected))
            else:
                expected_value = expected
            if isinstance(expected_value, pl.DataFrame):
                if isinstance(result, pl.DataFrame):
                    print(f"✓ Test {i}: {description}")
                    offset_passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected DataFrame, Got: {type(result)}")
                    offset_failed += 1
            else:
                if isinstance(result, Decimal) and abs(result - expected_value) < Decimal('1e-10'):
                    print(f"✓ Test {i}: {description}")
                    offset_passed += 1
                elif result == expected_value:
                    print(f"✓ Test {i}: {description}")
                    offset_passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected_value}, Got: {result}")
                    offset_failed += 1
        except Exception as e:
            print(f"✗ Test {i}: {description}")
            print(f"  Error: {type(e).__name__}: {str(e)}")
            offset_failed += 1
    print(f"Results for OFFSET: {offset_passed} passed, {offset_failed} failed")
    all_passed &= (offset_failed == 0)

    # Test INDIRECT function
    def test_indirect():
        print(f"\n=== Testing INDIRECT ===")
        passed = 0
        failed = 0
        indirect_tests = [
            ("A1", {}, "A1", "Simple cell reference"),
            ("Sheet1!B2", {}, "Sheet1!B2", "Sheet reference"),
            ("A1:C10", {}, "A1:C10", "Range reference"),
            ("  A1  ", {}, "A1", "Reference with whitespace"),
            ("B2", {"a1_style": True}, "B2", "A1 style reference"),
            ("R2C2", {"a1_style": False}, "R2C2", "R1C1 style reference"),
        ]
        for i, (ref_text, kwargs, expected, description) in enumerate(indirect_tests, 1):
            try:
                result = INDIRECT(ref_text, **kwargs)
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
        print(f"Results for INDIRECT: {passed} passed, {failed} failed")
        return failed == 0
    all_passed &= test_indirect()

    # Test ADDRESS function
    def test_address():
        print(f"\n=== Testing ADDRESS ===")
        passed = 0
        failed = 0
        address_tests = [
            ((1, 1), {}, "$A$1", "Basic absolute address"),
            ((5, 3), {}, "$C$5", "Absolute address C5"),
            ((10, 2), {"abs_num": 4}, "B10", "Relative address"),
            ((5, 3), {"abs_num": 2}, "C$5", "Mixed reference - absolute row"),
            ((5, 3), {"abs_num": 3}, "$C5", "Mixed reference - absolute column"),
            ((1, 1), {"sheet_text": "Budget"}, "Budget!$A$1", "Sheet reference"),
            ((10, 26), {}, "$Z$10", "Column Z address"),
            ((1, 27), {}, "$AA$1", "Column AA address"),
            ((5, 3), {"a1": False}, "R5C3", "R1C1 absolute"),
            ((5, 3), {"abs_num": 4, "a1": False}, "R[5]C[3]", "R1C1 relative"),
        ]
        for i, (args, kwargs, expected, description) in enumerate(address_tests, 1):
            try:
                result = ADDRESS(*args, **kwargs)
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
        print(f"Results for ADDRESS: {passed} passed, {failed} failed")
        return failed == 0
    all_passed &= test_address()

    # Test ROW function
    row_tests = [
        ({'reference': None}, 1, "No reference provided"),
        ({'reference': [["Value"]]}, [1], "Single row reference"),
        ({'reference': [["A"], ["B"], ["C"]]}, [1, 2, 3], "Multiple rows reference"),
        ({'reference': [["Revenue", 1000000], ["COGS", 600000]]}, [1, 2], "Financial data rows"),
        ({'reference': pl.DataFrame([["A"], ["B"]], schema=["col_0"], orient="row")}, [1, 2], "Polars DataFrame rows"),
    ]
    all_passed &= test_function("ROW", ROW, row_tests)

    # Test COLUMN function
    column_tests = [
        ({'reference': None}, 1, "No reference provided"),
        ({'reference': [["A"], ["B"]]}, [1], "Single column reference"),
        ({'reference': [["Q1", "Q2", "Q3", "Q4"]]}, [1, 2, 3, 4], "Multiple columns reference"),
        ({'reference': [["A", "B", "C"]]}, [1, 2, 3], "Three columns reference"),
        ({'reference': pl.DataFrame([["A", "B", "C"]], schema=["col_0", "col_1", "col_2"], orient="row")}, [1, 2, 3], "Polars DataFrame columns"),
    ]
    all_passed &= test_function("COLUMN", COLUMN, column_tests)

    # Test ROWS function
    rows_tests = [
        ({'array': [[1, 2], [3, 4], [5, 6]]}, 3, "Count rows in 2D array"),
        ({'array': [["A"], ["B"]]}, 2, "Count rows in simple array"),
        ({'array': [["Revenue", 1000000], ["COGS", 600000], ["Gross Profit", 400000]]}, 3, "Financial statement line items"),
        ({'array': [[50000], [52000], [48000], [55000]]}, 4, "Budget periods count"),
        ({'array': pl.DataFrame([[1, 2], [3, 4]], schema=["col_0", "col_1"], orient="row")}, 2, "Polars DataFrame rows"),
        ({'array': np.array([[1, 2], [3, 4], [5, 6]])}, 3, "NumPy array rows"),
    ]
    all_passed &= test_function("ROWS", ROWS, rows_tests)

    # Test COLUMNS function
    columns_tests = [
        ({'array': [[1, 2, 3], [4, 5, 6]]}, 3, "Count columns in 2D array"),
        ({'array': [["A"], ["B"]]}, 1, "Count columns in single column"),
        ({'array': [["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"], [1000000, 1100000, 1200000, 1300000]]}, 4, "Quarterly periods count"),
        ({'array': [[100, 200, 300]]}, 3, "Budget matrix width"),
        ({'array': pl.DataFrame([[1, 2, 3]], schema=["col_0", "col_1", "col_2"], orient="row")}, 3, "Polars DataFrame columns"),
        ({'array': np.array([[1, 2], [3, 4]])}, 2, "NumPy array columns"),
    ]
    all_passed &= test_function("COLUMNS", COLUMNS, columns_tests)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0
    try:
        VLOOKUP(ctx, "A", [], col_index=1)
        print("✗ Empty table validation failed for VLOOKUP")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty table validation passed for VLOOKUP")
        error_tests_passed += 1
    try:
        VLOOKUP(ctx, "A", [["A", 1]], col_index=3)
        print("✗ Column index bounds validation failed for VLOOKUP")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Column index bounds validation passed for VLOOKUP")
        error_tests_passed += 1
    try:
        VLOOKUP(ctx, "Z", [["A", 1], ["B", 2]], col_index=2, range_lookup=False)
        print("✗ Exact match not found validation failed for VLOOKUP")
        error_tests_failed += 1
    except CalculationError:
        print("✓ Exact match not found validation passed for VLOOKUP")
        error_tests_passed += 1
    try:
        MATCH(ctx, "A", ["A", "B"], match_type=5)
        print("✗ Invalid match type validation failed for MATCH")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid match type validation passed for MATCH")
        error_tests_passed += 1
    try:
        XLOOKUP(ctx, "A", ["A", "B"], [1])
        print("✗ Array length mismatch validation failed for XLOOKUP")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Array length mismatch validation passed for XLOOKUP")
        error_tests_passed += 1
    try:
        CHOOSE(5, "A", "B", "C")
        print("✗ Index out of bounds validation failed for CHOOSE")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Index out of bounds validation passed for CHOOSE")
        error_tests_passed += 1
    try:
        OFFSET(ctx, [[1, 2], [3, 4]], -1, 0)
        print("✗ Negative offset validation failed for OFFSET")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Negative offset validation passed for OFFSET")
        error_tests_passed += 1
    try:
        INDIRECT("")
        print("✗ Empty reference text validation failed for INDIRECT")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty reference text validation passed for INDIRECT")
        error_tests_passed += 1
    try:
        ADDRESS(0, 1)
        print("✗ Invalid row number validation failed for ADDRESS")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid row number validation passed for ADDRESS")
        error_tests_passed += 1
    try:
        VLOOKUP(ctx, "A", [["A", 1], ["B"]], col_index=2)
        print("✗ Inconsistent row lengths validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Inconsistent row lengths validation passed")
        error_tests_passed += 1
    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Test advanced scenarios
    print("\n=== Testing Advanced Scenarios ===")
    advanced_tests_passed = 0
    advanced_tests_failed = 0
    try:
        lookup_array = ["Jan", "Feb", "Mar", "Apr"]
        return_array = [100, 105, 110, 108]
        position = MATCH(ctx, "Mar", lookup_array, match_type=0)
        result = INDEX(ctx, return_array, position)
        if abs(Decimal(str(result)) - Decimal('110')) < Decimal('1e-10'):
            print("✓ INDEX + MATCH combination passed")
            advanced_tests_passed += 1
        else:
            print(f"✗ INDEX + MATCH combination failed: expected 110, got {result}")
            advanced_tests_failed += 1
    except Exception as e:
        print(f"✗ INDEX + MATCH combination error: {e}")
        advanced_tests_failed += 1
    try:
        large_table = [[f"ID{i:05d}", f"Value{i}", i * 100] for i in range(50000)]
        start_time = __import__('time').time()
        result = VLOOKUP(ctx, "ID25000", large_table, col_index=3)
        end_time = __import__('time').time()
        if result == 2500000 and (end_time - start_time) < 0.1:
            print(f"✓ Large dataset VLOOKUP performance passed: {result} in {end_time - start_time:.3f}s")
            advanced_tests_passed += 1
        else:
            print(f"✗ Large dataset VLOOKUP performance failed: {result} in {end_time - start_time:.3f}s")
            advanced_tests_failed += 1
    except Exception as e:
        print(f"✗ Large dataset VLOOKUP performance error: {e}")
        advanced_tests_failed += 1
    try:
        product_codes = ["PROD-A*", "PROD-B*", "PROD-C*"]
        prices = [100, 150, 200]
        result = XLOOKUP(ctx, "PROD-A123", product_codes, prices, match_mode=2)
        if result == 100:
            print("✓ XLOOKUP wildcard matching passed")
            advanced_tests_passed += 1
        else:
            print(f"✗ XLOOKUP wildcard matching failed: expected 100, got {result}")
            advanced_tests_failed += 1
    except Exception as e:
        print(f"✗ XLOOKUP wildcard matching error: {e}")
        advanced_tests_failed += 1
    try:
        employee_table = [
            ["EMP001", "Sales", "Manager", 75000],
            ["EMP002", "Sales", "Rep", 45000],
            ["EMP003", "IT", "Manager", 85000],
            ["EMP004", "IT", "Analyst", 55000]
        ]
        emp_id = "EMP003"
        salary = VLOOKUP(ctx, emp_id, employee_table, col_index=4)
        if salary == 85000:
            print("✓ Complex financial lookup scenario passed")
            advanced_tests_passed += 1
        else:
            print(f"✗ Complex financial lookup scenario failed: expected 85000, got {salary}")
            advanced_tests_failed += 1
    except Exception as e:
        print(f"✗ Complex financial lookup scenario error: {e}")
        advanced_tests_failed += 1
    try:
        address = ADDRESS(1, 702)
        expected = "$ZZ$1"
        if address == expected:
            print("✓ Large column number ADDRESS conversion passed")
            advanced_tests_passed += 1
        else:
            print(f"✗ Large column number ADDRESS conversion failed: expected {expected}, got {address}")
            advanced_tests_failed += 1
    except Exception as e:
        print(f"✗ Large column number ADDRESS conversion error: {e}")
        advanced_tests_failed += 1
    print(f"Advanced scenario tests: {advanced_tests_passed} passed, {advanced_tests_failed} failed")
    all_passed &= (advanced_tests_failed == 0)

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The lookup and reference functions are working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the failed tests and fix any issues.")
        return False

if __name__ == "__main__":
    print("Lookup and Reference Functions Test Suite")
    print("=" * 60)
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {type(e).__name__}: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
