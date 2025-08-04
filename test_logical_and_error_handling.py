#!/usr/bin/env python3
"""
Test script for logical and error handling functions.
Tests all functions in the logical_and_error_handling_functions.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np

# Import the functions to test
from tools.core_data_and_math_utils.logical_and_error_handling_functions.logical_and_error_handling_functions import (
    IF, IFS, AND, OR, NOT, XOR, IFERROR, IFNA, ISERROR, ISBLANK, ISNUMBER, ISTEXT, SWITCH,
    ValidationError, CalculationError, DataQualityError, ConfigurationError
)

from tools.finn_deps import FinnDeps, RunContext
from pathlib import Path

# Create global test context
thread_dir = Path("scratch_pad").resolve()
workspace_dir = Path("scratch_pad").resolve()
finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
ctx = RunContext(deps=finn_deps)

def test_function(func_name, func, test_cases, needs_context=True):
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
            # Add context to function calls that need it
            if needs_context:
                result = func(ctx, **args)
            else:
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
            elif isinstance(expected, (int, float)):
                if isinstance(result, (int, float, Decimal)):
                    if isinstance(result, Decimal):
                        result_val = float(result)
                    else:
                        result_val = result
                    if abs(result_val - expected) < 1e-10:
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
            elif isinstance(expected, list):
                if isinstance(result, list) and len(result) == len(expected):
                    match = True
                    for a, b in zip(result, expected):
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
            elif isinstance(expected, pl.Series):
                if isinstance(result, pl.Series):
                    if result.equals(expected):
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected: {expected.to_list()}, Got: {result.to_list()}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected Series, Got: {type(result)}")
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
    """Run all tests for the logical and error handling functions."""
    print("Starting comprehensive test of logical and error handling functions...")

    all_passed = True

    # Test IF function
    if_tests = [
        ({'logical_test': True, 'value_if_true': "Yes", 'value_if_false': "No"}, "Yes", "Basic true condition"),
        ({'logical_test': False, 'value_if_true': "Yes", 'value_if_false': "No"}, "No", "Basic false condition"),
        ({'logical_test': 100 > 50, 'value_if_true': "High", 'value_if_false': "Low"}, "High", "Numeric comparison true"),
        ({'logical_test': 10 > 50, 'value_if_true': "High", 'value_if_false': "Low"}, "Low", "Numeric comparison false"),
        ({'logical_test': 1, 'value_if_true': "Non-zero", 'value_if_false': "Zero"}, "Non-zero", "Non-zero number as true"),
        ({'logical_test': 0, 'value_if_true': "Non-zero", 'value_if_false': "Zero"}, "Zero", "Zero as false"),
        ({'logical_test': "true", 'value_if_true': "String True", 'value_if_false': "String False"}, "String True", "String 'true' as true"),
        ({'logical_test': "", 'value_if_true': "Non-empty", 'value_if_false': "Empty"}, "Empty", "Empty string as false"),
        ({'logical_test': None, 'value_if_true': "Not None", 'value_if_false': "None"}, "None", "None as false"),
        ({'logical_test': "hello", 'value_if_true': "Text", 'value_if_false': "No Text"}, "Text", "Non-empty string as true"),
        ({'logical_test': Decimal('5'), 'value_if_true': "Decimal", 'value_if_false': "Not Decimal"}, "Decimal", "Decimal as true"),
        ({'logical_test': Decimal('0'), 'value_if_true': "Decimal", 'value_if_false': "Zero Decimal"}, "Zero Decimal", "Zero Decimal as false"),
    ]
    all_passed &= test_function("IF", IF, if_tests)

    # Test IF with Polars Series
    print(f"\n=== Testing IF with Polars Series ===")
    if_series_passed = 0
    if_series_failed = 0
    try:
        # Test with boolean Series
        bool_series = pl.Series([True, False, True, False])
        result = IF(ctx, bool_series, "Yes", "No")
        expected = pl.Series(["Yes", "No", "Yes", "No"])
        if isinstance(result, pl.Series) and result.equals(expected):
            print("✓ Test 1: Boolean Series IF")
            if_series_passed += 1
        else:
            print("✗ Test 1: Boolean Series IF")
            print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
            if_series_failed += 1
    except Exception as e:
        print(f"✗ Test 1: Boolean Series IF - Error: {e}")
        if_series_failed += 1

    try:
        # Test with numeric Series
        numeric_series = pl.Series([1, 0, 5, -2])
        result = IF(ctx, numeric_series, "Truthy", "Falsy")
        expected = pl.Series(["Truthy", "Falsy", "Truthy", "Truthy"])
        if isinstance(result, pl.Series) and result.equals(expected):
            print("✓ Test 2: Numeric Series IF")
            if_series_passed += 1
        else:
            print("✗ Test 2: Numeric Series IF")
            print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
            if_series_failed += 1
    except Exception as e:
        print(f"✗ Test 2: Numeric Series IF - Error: {e}")
        if_series_failed += 1

    print(f"Results for IF with Series: {if_series_passed} passed, {if_series_failed} failed")
    all_passed &= (if_series_failed == 0)

    # Test IF with lists
    print(f"\n=== Testing IF with Lists ===")
    if_list_passed = 0
    if_list_failed = 0
    try:
        # Test with boolean list
        bool_list = [True, False, True]
        result = IF(ctx, bool_list, "Yes", "No")
        expected = ["Yes", "No", "Yes"]
        if result == expected:
            print("✓ Test 1: Boolean list IF")
            if_list_passed += 1
        else:
            print("✗ Test 1: Boolean list IF")
            print(f"  Expected: {expected}, Got: {result}")
            if_list_failed += 1
    except Exception as e:
        print(f"✗ Test 1: Boolean list IF - Error: {e}")
        if_list_failed += 1

    print(f"Results for IF with Lists: {if_list_passed} passed, {if_list_failed} failed")
    all_passed &= (if_list_failed == 0)

    # Test IFS function
    def test_ifs():
        print(f"\n=== Testing IFS ===")
        passed = 0
        failed = 0

        test_cases = [
            ((False, "A", True, "B", False, "C"), "B", "First true condition"),
            ((10 > 100, "High", 10 > 5, "Medium", True, "Low"), "Medium", "Numeric comparison"),
            ((False, "A", False, "B", True, "C"), "C", "Last condition true"),
            ((True, "First"), "First", "Single condition"),
            ((80 >= 90, "A", 80 >= 80, "B", 80 >= 70, "C", True, "F"), "B", "Grade assignment"),
            ((750 >= 800, "AAA", 750 >= 750, "AA", 750 >= 700, "A", True, "Below Grade"), "AA", "Credit rating"),
        ]

        for i, (args, expected, description) in enumerate(test_cases, 1):
            try:
                result = IFS(ctx, *args)
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

        print(f"Results for IFS: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_ifs()

    # Test AND function
    def test_and():
        print(f"\n=== Testing AND ===")
        passed = 0
        failed = 0

        test_cases = [
            ((True, True, True), True, "All true"),
            ((True, False, True), False, "One false"),
            ((False, False, False), False, "All false"),
            ((10 > 5, 20 > 15, 30 > 25), True, "All numeric comparisons true"),
            ((10 > 5, 20 > 25, 30 > 25), False, "One numeric comparison false"),
            ((1, 2, 3), True, "All non-zero numbers"),
            ((1, 0, 3), False, "One zero number"),
            (("yes", "true", "1"), True, "All truthy strings"),
            (("yes", "", "1"), False, "One empty string"),
            ((True,), True, "Single true value"),
            ((False,), False, "Single false value"),
        ]

        for i, (args, expected, description) in enumerate(test_cases, 1):
            try:
                result = AND(ctx, *args)
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

        print(f"Results for AND: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_and()

    # Test AND function with Polars Series
    def test_and_series():
        print(f"\n=== Testing AND with Polars Series ===")
        passed = 0
        failed = 0

        try:
            # Test basic Series AND
            series1 = pl.Series([True, True, False, True])
            series2 = pl.Series([True, False, True, True])
            series3 = pl.Series([True, True, True, False])
            result = AND(ctx, series1, series2, series3)
            expected = pl.Series([True, False, False, False])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 1: Basic Series AND")
                passed += 1
            else:
                print("✗ Test 1: Basic Series AND")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 1: Basic Series AND - Error: {e}")
            failed += 1

        try:
            # Test mixed Series and scalar
            series1 = pl.Series([True, False, True])
            scalar = True
            result = AND(ctx, series1, scalar)
            expected = pl.Series([True, False, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 2: Mixed Series and scalar AND")
                passed += 1
            else:
                print("✗ Test 2: Mixed Series and scalar AND")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 2: Mixed Series and scalar AND - Error: {e}")
            failed += 1

        try:
            # Test numeric Series
            series1 = pl.Series([1, 0, 5, -2])
            series2 = pl.Series([2, 3, 0, 1])
            result = AND(ctx, series1, series2)
            expected = pl.Series([True, False, False, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 3: Numeric Series AND")
                passed += 1
            else:
                print("✗ Test 3: Numeric Series AND")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 3: Numeric Series AND - Error: {e}")
            failed += 1

        try:
            # Test financial scenario with Series
            pe_ratios = pl.Series([15.2, 25.1, 18.5])
            debt_ratios = pl.Series([0.3, 0.6, 0.4])
            roe_values = pl.Series([0.18, 0.12, 0.20])
            result = AND(ctx, pe_ratios < 20, debt_ratios < 0.5, roe_values > 0.15)
            expected = pl.Series([True, False, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 4: Financial portfolio analysis")
                passed += 1
            else:
                print("✗ Test 4: Financial portfolio analysis")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 4: Financial portfolio analysis - Error: {e}")
            failed += 1

        print(f"Results for AND with Series: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_and_series()

    # Test OR function
    def test_or():
        print(f"\n=== Testing OR ===")
        passed = 0
        failed = 0

        test_cases = [
            ((False, False, True), True, "One true"),
            ((False, False, False), False, "All false"),
            ((True, True, True), True, "All true"),
            ((10 > 20, 5 > 3, 1 > 2), True, "One numeric comparison true"),
            ((10 > 20, 5 > 10, 1 > 2), False, "All numeric comparisons false"),
            ((0, 0, 1), True, "One non-zero number"),
            ((0, 0, 0), False, "All zero numbers"),
            (("", "", "hello"), True, "One non-empty string"),
            (("", "", ""), False, "All empty strings"),
            ((True,), True, "Single true value"),
            ((False,), False, "Single false value"),
        ]

        for i, (args, expected, description) in enumerate(test_cases, 1):
            try:
                result = OR(ctx, *args)
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

        print(f"Results for OR: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_or()

    # Test OR function with Polars Series
    def test_or_series():
        print(f"\n=== Testing OR with Polars Series ===")
        passed = 0
        failed = 0

        try:
            # Test basic Series OR
            series1 = pl.Series([False, False, True, False])
            series2 = pl.Series([False, True, False, False])
            series3 = pl.Series([False, False, False, True])
            result = OR(ctx, series1, series2, series3)
            expected = pl.Series([False, True, True, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 1: Basic Series OR")
                passed += 1
            else:
                print("✗ Test 1: Basic Series OR")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 1: Basic Series OR - Error: {e}")
            failed += 1

        try:
            # Test mixed Series and scalar
            series1 = pl.Series([False, False, False])
            scalar = True
            result = OR(ctx, series1, scalar)
            expected = pl.Series([True, True, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 2: Mixed Series and scalar OR")
                passed += 1
            else:
                print("✗ Test 2: Mixed Series and scalar OR")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 2: Mixed Series and scalar OR - Error: {e}")
            failed += 1

        try:
            # Test financial risk assessment with Series
            debt_ratios = pl.Series([0.85, 0.45, 0.90])
            liquidity_ratios = pl.Series([0.8, 1.2, 0.6])
            profit_margins = pl.Series([-0.05, 0.15, 0.02])
            result = OR(ctx, debt_ratios > 0.8, liquidity_ratios < 1.0, profit_margins < 0)
            expected = pl.Series([True, False, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 3: Financial risk assessment")
                passed += 1
            else:
                print("✗ Test 3: Financial risk assessment")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 3: Financial risk assessment - Error: {e}")
            failed += 1

        print(f"Results for OR with Series: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_or_series()

    # Test NOT function
    not_tests = [
        ({'logical': True}, False, "NOT True"),
        ({'logical': False}, True, "NOT False"),
        ({'logical': 1}, False, "NOT 1"),
        ({'logical': 0}, True, "NOT 0"),
        ({'logical': "hello"}, False, "NOT non-empty string"),
        ({'logical': ""}, True, "NOT empty string"),
        ({'logical': None}, True, "NOT None"),
        ({'logical': "true"}, False, "NOT 'true' string"),
        ({'logical': "false"}, True, "NOT 'false' string"),
        ({'logical': Decimal('5')}, False, "NOT non-zero Decimal"),
        ({'logical': Decimal('0')}, True, "NOT zero Decimal"),
    ]
    all_passed &= test_function("NOT", NOT, not_tests)

    # Test NOT function with Polars Series
    def test_not_series():
        print(f"\n=== Testing NOT with Polars Series ===")
        passed = 0
        failed = 0

        try:
            # Test basic Series NOT
            series1 = pl.Series([True, False, True, False])
            result = NOT(ctx, series1)
            expected = pl.Series([False, True, False, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 1: Basic Series NOT")
                passed += 1
            else:
                print("✗ Test 1: Basic Series NOT")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 1: Basic Series NOT - Error: {e}")
            failed += 1

        try:
            # Test numeric Series NOT
            series1 = pl.Series([1, 0, 5, -2])
            result = NOT(ctx, series1)
            expected = pl.Series([False, True, False, False])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 2: Numeric Series NOT")
                passed += 1
            else:
                print("✗ Test 2: Numeric Series NOT")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 2: Numeric Series NOT - Error: {e}")
            failed += 1

        try:
            # Test financial exclusion criteria with Series
            tobacco_companies = pl.Series([False, True, False])
            result = NOT(ctx, tobacco_companies)
            expected = pl.Series([True, False, True])

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 3: Financial exclusion criteria")
                passed += 1
            else:
                print("✗ Test 3: Financial exclusion criteria")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 3: Financial exclusion criteria - Error: {e}")
            failed += 1

        print(f"Results for NOT with Series: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_not_series()

    # Test XOR function
    def test_xor():
        print(f"\n=== Testing XOR ===")
        passed = 0
        failed = 0

        test_cases = [
            ((True, False, False), True, "One true (odd)"),
            ((True, True, False), False, "Two true (even)"),
            ((True, True, True), True, "Three true (odd)"),
            ((False, False, False), False, "All false"),
            ((True, False, True, False), False, "Two true (even)"),
            ((True, False, True, False, True), True, "Three true (odd)"),
            ((1, 0, 0), True, "One non-zero (odd)"),
            (("hello", "", ""), True, "One non-empty string (odd)"),
            ((True,), True, "Single true"),
            ((False,), False, "Single false"),
        ]

        for i, (args, expected, description) in enumerate(test_cases, 1):
            try:
                result = XOR(ctx, *args)
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

        print(f"Results for XOR: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_xor()

    # Test XOR function with Polars Series
    def test_xor_series():
        print(f"\n=== Testing XOR with Polars Series ===")
        passed = 0
        failed = 0

        try:
            # Test basic Series XOR
            series1 = pl.Series([True, False, True])
            series2 = pl.Series([False, True, False])
            series3 = pl.Series([False, False, True])
            result = XOR(ctx, series1, series2, series3)
            expected = pl.Series([True, True, False])  # [1,1,2] -> [odd,odd,even] -> [True,True,False]

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 1: Basic Series XOR")
                passed += 1
            else:
                print("✗ Test 1: Basic Series XOR")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 1: Basic Series XOR - Error: {e}")
            failed += 1

        try:
            # Test mixed Series and scalar
            series1 = pl.Series([True, False, True])
            scalar = True
            result = XOR(ctx, series1, scalar)
            expected = pl.Series([False, True, False])  # [2,1,2] -> [even,odd,even] -> [False,True,False]

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 2: Mixed Series and scalar XOR")
                passed += 1
            else:
                print("✗ Test 2: Mixed Series and scalar XOR")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 2: Mixed Series and scalar XOR - Error: {e}")
            failed += 1

        try:
            # Test financial exclusive conditions with Series
            stocks = pl.Series([True, False, True])
            bonds = pl.Series([False, True, False])
            real_estate = pl.Series([False, False, True])
            result = XOR(ctx, stocks, bonds, real_estate)
            expected = pl.Series([True, True, False])  # [1,1,2] -> [odd,odd,even] -> [True,True,False]

            if isinstance(result, pl.Series) and result.equals(expected):
                print("✓ Test 3: Financial exclusive investments")
                passed += 1
            else:
                print("✗ Test 3: Financial exclusive investments")
                print(f"  Expected: {expected.to_list()}, Got: {result.to_list() if isinstance(result, pl.Series) else result}")
                failed += 1
        except Exception as e:
            print(f"✗ Test 3: Financial exclusive investments - Error: {e}")
            failed += 1

        print(f"Results for XOR with Series: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_xor_series()

    # Test IFERROR function
    iferror_tests = [
        ({'value': 10, 'value_if_error': "Error"}, 10, "Valid value"),
        ({'value': "#DIV/0!", 'value_if_error': "Division Error"}, "Division Error", "Division by zero error"),
        ({'value': "#N/A", 'value_if_error': "Not Available"}, "Not Available", "N/A error"),
        ({'value': "#VALUE!", 'value_if_error': "Value Error"}, "Value Error", "Value error"),
        ({'value': "Normal Text", 'value_if_error': "Error"}, "Normal Text", "Normal text value"),
        ({'value': 0, 'value_if_error': "Error"}, 0, "Zero value"),
        ({'value': None, 'value_if_error': "Null Error"}, None, "None value"),
    ]
    all_passed &= test_function("IFERROR", IFERROR, iferror_tests)

    # Test IFERROR with callable
    print(f"\n=== Testing IFERROR with Callable ===")
    iferror_callable_passed = 0
    iferror_callable_failed = 0
    try:
        # Test with successful function
        def safe_divide():
            return 10 / 2
        result = IFERROR(ctx, safe_divide, "Error")
        if result == 5.0:
            print("✓ Test 1: Successful callable")
            iferror_callable_passed += 1
        else:
            print(f"✗ Test 1: Successful callable - Expected: 5.0, Got: {result}")
            iferror_callable_failed += 1
    except Exception as e:
        print(f"✗ Test 1: Successful callable - Error: {e}")
        iferror_callable_failed += 1

    try:
        # Test with failing function
        def failing_divide():
            return 10 / 0
        result = IFERROR(ctx, failing_divide, "Division Error")
        if result == "Division Error":
            print("✓ Test 2: Failing callable")
            iferror_callable_passed += 1
        else:
            print(f"✗ Test 2: Failing callable - Expected: 'Division Error', Got: {result}")
            iferror_callable_failed += 1
    except Exception as e:
        print(f"✗ Test 2: Failing callable - Error: {e}")
        iferror_callable_failed += 1

    print(f"Results for IFERROR with Callable: {iferror_callable_passed} passed, {iferror_callable_failed} failed")
    all_passed &= (iferror_callable_failed == 0)

    # Test IFNA function
    ifna_tests = [
        ({'value': "Valid Value", 'value_if_na': "N/A Replacement"}, "Valid Value", "Valid value"),
        ({'value': "#N/A", 'value_if_na': "Not Available"}, "Not Available", "N/A error"),
        ({'value': "N/A", 'value_if_na': "Missing Data"}, "Missing Data", "N/A text"),
        ({'value': "#NA", 'value_if_na': "Not Available"}, "Not Available", "NA error"),
        ({'value': "na", 'value_if_na': "Missing"}, "Missing", "Lowercase na"),
        ({'value': 42, 'value_if_na': "Missing"}, 42, "Numeric value"),
        ({'value': "", 'value_if_na': "Missing"}, "", "Empty string"),
        ({'value': None, 'value_if_na': "Missing"}, None, "None value"),
    ]
    all_passed &= test_function("IFNA", IFNA, ifna_tests)

    # Test ISERROR function
    iserror_tests = [
        ({'value': "#DIV/0!"}, True, "Division by zero error"),
        ({'value': "#N/A"}, True, "N/A error"),
        ({'value': "#NAME?"}, True, "Name error"),
        ({'value': "#NULL!"}, True, "Null error"),
        ({'value': "#NUM!"}, True, "Number error"),
        ({'value': "#REF!"}, True, "Reference error"),
        ({'value': "#VALUE!"}, True, "Value error"),
        ({'value': "#ERROR!"}, True, "Generic error"),
        ({'value': 42}, False, "Valid number"),
        ({'value': "Valid Text"}, False, "Valid text"),
        ({'value': None}, False, "None value"),
        ({'value': ""}, False, "Empty string"),
        ({'value': 0}, False, "Zero value"),
    ]
    all_passed &= test_function("ISERROR", ISERROR, iserror_tests)

    # Test ISBLANK function
    isblank_tests = [
        ({'value': None}, True, "None value"),
        ({'value': ""}, True, "Empty string"),
        ({'value': "   "}, True, "Whitespace string"),
        ({'value': "\t\n"}, True, "Tab and newline"),
        ({'value': 0}, False, "Zero number"),
        ({'value': "Text"}, False, "Non-empty text"),
        ({'value': " Text "}, False, "Text with spaces"),
        ({'value': False}, False, "Boolean false"),
        ({'value': []}, False, "Empty list"),
        ({'value': {}}, False, "Empty dict"),
    ]
    all_passed &= test_function("ISBLANK", ISBLANK, isblank_tests)

    # Test ISNUMBER function
    isnumber_tests = [
        ({'value': 42}, True, "Integer"),
        ({'value': 3.14}, True, "Float"),
        ({'value': Decimal('5.5')}, True, "Decimal"),
        ({'value': -10}, True, "Negative integer"),
        ({'value': 0}, True, "Zero"),
        ({'value': "123"}, False, "String number"),
        ({'value': "Text"}, False, "Text"),
        ({'value': None}, False, "None"),
        ({'value': True}, False, "Boolean true"),
        ({'value': False}, False, "Boolean false"),
        ({'value': []}, False, "List"),
        ({'value': {}}, False, "Dict"),
    ]
    all_passed &= test_function("ISNUMBER", ISNUMBER, isnumber_tests)

    # Test ISTEXT function
    istext_tests = [
        ({'value': "Hello"}, True, "Text string"),
        ({'value': "123"}, True, "Numeric string"),
        ({'value': ""}, True, "Empty string"),
        ({'value': " "}, True, "Space string"),
        ({'value': 123}, False, "Number"),
        ({'value': None}, False, "None"),
        ({'value': True}, False, "Boolean"),
        ({'value': []}, False, "List"),
        ({'value': {}}, False, "Dict"),
        ({'value': Decimal('5')}, False, "Decimal"),
    ]
    all_passed &= test_function("ISTEXT", ISTEXT, istext_tests)

    # Test SWITCH function
    def test_switch():
        print(f"\n=== Testing SWITCH ===")
        passed = 0
        failed = 0

        test_cases = [
            (("B", "A", 1, "B", 2, "C", 3), {}, 2, "Basic match"),
            (("D", "A", 1, "B", 2, "C", 3), {"default": "Not Found"}, "Not Found", "Default value"),
            ((2, 1, "One", 2, "Two", 3, "Three"), {}, "Two", "Numeric key"),
            (("Marketing", "Sales", 1.2, "Marketing", 1.1, "Operations", 1.0), {"default": 0.9}, 1.1, "Department lookup"),
            (("Gold", "Platinum", 0.08, "Gold", 0.06, "Silver", 0.04), {}, 0.06, "Performance tier"),
            (("Unknown", "A", 1, "B", 2), {"default": 99}, 99, "No match with default"),
        ]

        for i, (expression_and_pairs, kwargs, expected, description) in enumerate(test_cases, 1):
            try:
                expression = expression_and_pairs[0]
                pairs = expression_and_pairs[1:]
                result = SWITCH(ctx, expression, *pairs, **kwargs)

                # Handle numeric comparisons
                if isinstance(expected, (int, float)) and isinstance(result, (int, float, Decimal)):
                    if isinstance(result, Decimal):
                        result_val = float(result)
                    else:
                        result_val = result
                    if abs(result_val - expected) < 1e-10:
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected: {expected}, Got: {result}")
                        failed += 1
                elif result == expected:
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

        print(f"Results for SWITCH: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_switch()

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test ValidationError cases
    try:
        AND()  # No arguments
        print("✗ AND with no arguments should raise ValidationError")
        error_tests_failed += 1
    except ValidationError:
        print("✓ AND with no arguments raises ValidationError")
        error_tests_passed += 1
    except Exception as e:
        print(f"✗ AND with no arguments raised {type(e).__name__} instead of ValidationError")
        error_tests_failed += 1

    try:
        OR()  # No arguments
        print("✗ OR with no arguments should raise ValidationError")
        error_tests_failed += 1
    except ValidationError:
        print("✓ OR with no arguments raises ValidationError")
        error_tests_passed += 1
    except Exception as e:
        print(f"✗ OR with no arguments raised {type(e).__name__} instead of ValidationError")
        error_tests_failed += 1

    try:
        XOR()  # No arguments
        print("✗ XOR with no arguments should raise ValidationError")
        error_tests_failed += 1
    except ValidationError:
        print("✓ XOR with no arguments raises ValidationError")
        error_tests_passed += 1
    except Exception as e:
        print(f"✗ XOR with no arguments raised {type(e).__name__} instead of ValidationError")
        error_tests_failed += 1

    try:
        IFS(True)  # Odd number of arguments
        print("✗ IFS with odd arguments should raise ValidationError")
        error_tests_failed += 1
    except ValidationError:
        print("✓ IFS with odd arguments raises ValidationError")
        error_tests_passed += 1
    except Exception as e:
        print(f"✗ IFS with odd arguments raised {type(e).__name__} instead of ValidationError")
        error_tests_failed += 1

    try:
        IFS(False, "A", False, "B")  # No true conditions
        print("✗ IFS with no true conditions should raise CalculationError")
        error_tests_failed += 1
    except CalculationError:
        print("✓ IFS with no true conditions raises CalculationError")
        error_tests_passed += 1
    except Exception as e:
        print(f"✗ IFS with no true conditions raised {type(e).__name__} instead of CalculationError")
        error_tests_failed += 1

    try:
        SWITCH("X", "A", 1, "B")  # Odd number of pairs
        print("✗ SWITCH with odd pairs should raise ValidationError")
        error_tests_failed += 1
    except ValidationError:
        print("✓ SWITCH with odd pairs raises ValidationError")
        error_tests_passed += 1
    except Exception as e:
        print(f"✗ SWITCH with odd pairs raised {type(e).__name__} instead of ValidationError")
        error_tests_failed += 1

    try:
        SWITCH("X", "A", 1, "B", 2)  # No match and no default
        print("✗ SWITCH with no match should raise CalculationError")
        error_tests_failed += 1
    except CalculationError:
        print("✓ SWITCH with no match raises CalculationError")
        error_tests_passed += 1
    except Exception as e:
        print(f"✗ SWITCH with no match raised {type(e).__name__} instead of CalculationError")
        error_tests_failed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Test financial scenarios
    print("\n=== Testing Financial Scenarios ===")
    financial_tests_passed = 0
    financial_tests_failed = 0

    try:
        # Budget variance analysis
        actual = 105000
        budget = 100000
        variance_status = IF(ctx, actual > budget, "Over Budget", "Within Budget")
        if variance_status == "Over Budget":
            print("✓ Test 1: Budget variance analysis")
            financial_tests_passed += 1
        else:
            print(f"✗ Test 1: Budget variance analysis - Expected: 'Over Budget', Got: {variance_status}")
            financial_tests_failed += 1
    except Exception as e:
        print(f"✗ Test 1: Budget variance analysis - Error: {e}")
        financial_tests_failed += 1

    try:
        # Credit rating assignment
        credit_score = 780
        rating = IFS(ctx,
            credit_score >= 800, "AAA",
            credit_score >= 750, "AA",
            credit_score >= 700, "A",
            credit_score >= 650, "BBB",
            True, "Below Investment Grade"
        )
        if rating == "AA":
            print("✓ Test 2: Credit rating assignment")
            financial_tests_passed += 1
        else:
            print(f"✗ Test 2: Credit rating assignment - Expected: 'AA', Got: {rating}")
            financial_tests_failed += 1
    except Exception as e:
        print(f"✗ Test 2: Credit rating assignment - Error: {e}")
        financial_tests_failed += 1

    try:
        # Investment criteria validation
        pe_ratio = 15.2
        debt_to_equity = 0.3
        roe = 0.18
        meets_criteria = AND(ctx, pe_ratio < 20, debt_to_equity < 0.5, roe > 0.15)
        if meets_criteria == True:
            print("✓ Test 3: Investment criteria validation")
            financial_tests_passed += 1
        else:
            print(f"✗ Test 3: Investment criteria validation - Expected: True, Got: {meets_criteria}")
            financial_tests_failed += 1
    except Exception as e:
        print(f"✗ Test 3: Investment criteria validation - Error: {e}")
        financial_tests_failed += 1

    try:
        # Risk flag detection
        debt_ratio = 0.85
        liquidity_ratio = 0.8
        profit_margin = -0.05
        risk_flag = OR(ctx, debt_ratio > 0.8, liquidity_ratio < 1.0, profit_margin < 0)
        if risk_flag == True:
            print("✓ Test 4: Risk flag detection")
            financial_tests_passed += 1
        else:
            print(f"✗ Test 4: Risk flag detection - Expected: True, Got: {risk_flag}")
            financial_tests_failed += 1
    except Exception as e:
        print(f"✗ Test 4: Risk flag detection - Error: {e}")
        financial_tests_failed += 1

    try:
        # Safe division for financial ratios
        revenue = 1000000
        shares_outstanding = 0  # Could cause division by zero
        eps = IFERROR(ctx, revenue / shares_outstanding if shares_outstanding != 0 else "#DIV/0!", "N/A")
        if eps == "N/A":
            print("✓ Test 5: Safe division for financial ratios")
            financial_tests_passed += 1
        else:
            print(f"✗ Test 5: Safe division for financial ratios - Expected: 'N/A', Got: {eps}")
            financial_tests_failed += 1
    except Exception as e:
        print(f"✗ Test 5: Safe division for financial ratios - Error: {e}")
        financial_tests_failed += 1

    try:
        # Department budget allocation
        department = "Marketing"
        budget_multiplier = SWITCH(ctx,
            department,
            "Sales", 1.2,
            "Marketing", 1.1,
            "Operations", 1.0,
            "HR", 0.8,
            default=0.9
        )
        if abs(budget_multiplier - 1.1) < 1e-10:
            print("✓ Test 6: Department budget allocation")
            financial_tests_passed += 1
        else:
            print(f"✗ Test 6: Department budget allocation - Expected: 1.1, Got: {budget_multiplier}")
            financial_tests_failed += 1
    except Exception as e:
        print(f"✗ Test 6: Department budget allocation - Error: {e}")
        financial_tests_failed += 1

    try:
        # Data validation scenario
        quarterly_revenue = None
        data_missing = ISBLANK(ctx, quarterly_revenue)
        revenue_is_number = ISNUMBER(ctx, quarterly_revenue)
        department_name = "Finance"
        name_is_text = ISTEXT(ctx, department_name)

        validation_passed = AND(ctx, data_missing == True, revenue_is_number == False, name_is_text == True)
        if validation_passed:
            print("✓ Test 7: Data validation scenario")
            financial_tests_passed += 1
        else:
            print(f"✗ Test 7: Data validation scenario - Validation failed")
            financial_tests_failed += 1
    except Exception as e:
        print(f"✗ Test 7: Data validation scenario - Error: {e}")
        financial_tests_failed += 1

    print(f"Financial scenario tests: {financial_tests_passed} passed, {financial_tests_failed} failed")
    all_passed &= (financial_tests_failed == 0)

    # Summary
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The logical and error handling functions are working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the failed tests and fix any issues.")
        return False


if __name__ == "__main__":
    print("Logical and Error Handling Functions Test Suite")
    print("=" * 60)

    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)
