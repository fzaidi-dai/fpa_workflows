#!/usr/bin/env python3
"""
Test script for conditional aggregation and counting functions.
Tests all functions in the conditional_aggregation_and_counting.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.conditional_aggregation_and_counting.conditional_aggregation_and_counting import (
    COUNTIF, COUNTIFS, SUMIF, SUMIFS, AVERAGEIF, AVERAGEIFS, MAXIFS, MINIFS,
    SUMPRODUCT, COUNTBLANK, COUNTA, AGGREGATE, SUBTOTAL,
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
                args = {'ctx': ctx, **args}
            result = func(**args)

            # Handle comparison of Decimal values
            if isinstance(expected, Decimal):
                if abs(result - expected) < Decimal('1e-10'):
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            elif isinstance(expected, (int, float)):
                if isinstance(result, Decimal):
                    if abs(result - Decimal(str(expected))) < Decimal('1e-10'):
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected: {expected}, Got: {result}")
                        failed += 1
                else:
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

def run_all_tests():
    """Run all tests for the conditional aggregation and counting functions."""
    print("Starting comprehensive test of conditional aggregation and counting functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    all_passed = True

    # Test COUNTIF function
    countif_tests = [
        ({'range_to_evaluate': [100, 200, 150, 300, 50], 'criteria': ">150"}, 2, "Count values greater than 150"),
        ({'range_to_evaluate': ["Sales", "Marketing", "Sales", "IT"], 'criteria': "Sales"}, 2, "Count exact text matches"),
        ({'range_to_evaluate': [1, 2, 3, 4, 5], 'criteria': ">=3"}, 3, "Count values greater than or equal to 3"),
        ({'range_to_evaluate': ["A", "B", "A", "C", "A"], 'criteria': "A"}, 3, "Count text values"),
        ({'range_to_evaluate': pl.Series([10, 20, 30, 40]), 'criteria': "<25"}, 2, "Count with Polars Series"),
        ({'range_to_evaluate': np.array([5, 15, 25, 35]), 'criteria': "<=20"}, 2, "Count with NumPy array"),
    ]
    all_passed &= test_function("COUNTIF", COUNTIF, countif_tests, ctx=ctx)

    # Test COUNTIFS function
    countifs_tests = [
        ({'criteria_ranges': [[100, 200, 150, 300, 50], ["A", "B", "A", "A", "B"]], 'criteria_values': [">100", "A"]}, 2, "Count with multiple criteria"),
        ({'criteria_ranges': [[1000, 2500, 500, 7500, 1200], ["North", "South", "North", "West", "South"]], 'criteria_values': [">1000", "North"]}, 0, "Financial example: high-value North sales"),
        ({'criteria_ranges': [[50000, -10000, 75000, 25000, -5000], ["Q1", "Q2", "Q3", "Q4", "Q4"]], 'criteria_values': [">0", "Q4"]}, 1, "Financial example: profitable Q4 products"),
        ({'criteria_ranges': [pl.Series([100, 200, 150]), pl.Series(["X", "Y", "X"])], 'criteria_values': [">100", "X"]}, 1, "Count with Polars Series"),
    ]
    all_passed &= test_function("COUNTIFS", COUNTIFS, countifs_tests, ctx=ctx)

    # Test SUMIF function
    sumif_tests = [
        ({'range_to_evaluate': ["A", "B", "A", "C", "A"], 'criteria': "A", 'sum_range': [100, 200, 150, 300, 50]}, Decimal('300'), "Sum with separate sum range"),
        ({'range_to_evaluate': [5000, 12000, 8000, 15000, 6000, 20000], 'criteria': ">10000"}, Decimal('47000'), "Sum values above threshold"),
        ({'range_to_evaluate': ["North", "South", "East", "West", "North"], 'criteria': "North", 'sum_range': [100000, 75000, 120000, 90000, 110000]}, Decimal('210000'), "Financial example: North region revenue"),
        ({'range_to_evaluate': pl.Series(["Q1", "Q2", "Q3", "Q4", "Q4"]), 'criteria': "Q4", 'sum_range': pl.Series([250000, 280000, 300000, 320000, 310000])}, Decimal('630000'), "Sum with Polars Series"),
    ]
    all_passed &= test_function("SUMIF", SUMIF, sumif_tests, ctx=ctx)

    # Test SUMIFS function
    sumifs_tests = [
        ({'sum_range': [100, 200, 150, 300, 50], 'criteria_ranges': [["A", "B", "A", "A", "B"], ["North", "South", "North", "West", "South"]], 'criteria_values': ["A", "North"]}, Decimal('250'), "Sum with multiple criteria"),
        ({'sum_range': [100000, 75000, 120000, 90000, 110000], 'criteria_ranges': [["North", "South", "North", "West", "North"], ["Premium", "Standard", "Premium", "Premium", "Standard"]], 'criteria_values': ["North", "Premium"]}, Decimal('220000'), "Financial example: Premium North revenue"),
        ({'sum_range': [50000, 75000, 60000, 80000, 45000, 90000], 'criteria_ranges': [["Q3", "Q4", "Q4", "Q4", "Q3", "Q4"], ["Sales", "Marketing", "Sales", "IT", "Sales", "Marketing"], [50000, 75000, 60000, 80000, 45000, 90000]], 'criteria_values': ["Q4", "Marketing", ">70000"]}, Decimal('165000'), "Financial example: Q4 high Marketing expenses"),
    ]
    all_passed &= test_function("SUMIFS", SUMIFS, sumifs_tests, ctx=ctx)

    # Test AVERAGEIF function
    averageif_tests = [
        ({'range_to_evaluate': ["A", "B", "A", "C", "A"], 'criteria': "A", 'average_range': [100, 200, 150, 300, 50]}, Decimal('100'), "Average with separate average range"),
        ({'range_to_evaluate': ["SMB", "Enterprise", "SMB", "Enterprise", "Mid-market"], 'criteria': "Enterprise", 'average_range': [25000, 150000, 30000, 200000, 75000]}, Decimal('175000'), "Financial example: Enterprise average deal size"),
        ({'range_to_evaluate': [10, 20, 30, 40, 50], 'criteria': ">25"}, Decimal('40'), "Average values above threshold"),
        ({'range_to_evaluate': pl.Series(["A", "B", "A", "B"]), 'criteria': "A", 'average_range': pl.Series([100, 200, 300, 400])}, Decimal('200'), "Average with Polars Series"),
    ]
    all_passed &= test_function("AVERAGEIF", AVERAGEIF, averageif_tests, ctx=ctx)

    # Test AVERAGEIFS function
    averageifs_tests = [
        ({'average_range': [100, 200, 150, 300, 50], 'criteria_ranges': [["A", "B", "A", "A", "B"], ["North", "South", "North", "West", "South"]], 'criteria_values': ["A", "North"]}, Decimal('125'), "Average with multiple criteria"),
        ({'average_range': [3.5, 4.2, 2.8, 5.1, 3.9, 4.7], 'criteria_ranges': [["AAA", "AA", "BBB", "AAA", "AA", "AAA"], [5, 10, 3, 7, 15, 8]], 'criteria_values': ["AAA", "<=8"]}, Decimal('4.433333333333333333333333333'), "Financial example: AAA bonds ≤8 years"),
    ]
    all_passed &= test_function("AVERAGEIFS", AVERAGEIFS, averageifs_tests, ctx=ctx)

    # Test MAXIFS function
    maxifs_tests = [
        ({'max_range': [100, 200, 150, 300, 50], 'criteria_ranges': [["A", "B", "A", "A", "B"], ["North", "South", "North", "West", "South"]], 'criteria_values': ["A", "North"]}, Decimal('150'), "Max with multiple criteria"),
        ({'max_range': [100000, 250000, 150000, 300000, 200000], 'criteria_ranges': [["Standard", "Premium", "Premium", "Premium", "Standard"], ["Q3", "Q4", "Q4", "Q4", "Q3"]], 'criteria_values': ["Premium", "Q4"]}, Decimal('300000'), "Financial example: Highest Q4 premium revenue"),
    ]
    all_passed &= test_function("MAXIFS", MAXIFS, maxifs_tests, ctx=ctx)

    # Test MINIFS function
    minifs_tests = [
        ({'min_range': [100, 200, 150, 300, 50], 'criteria_ranges': [["A", "B", "A", "A", "B"], ["North", "South", "North", "West", "South"]], 'criteria_values': ["A", "North"]}, Decimal('100'), "Min with multiple criteria"),
        ({'min_range': [50000, 25000, 75000, 30000, 60000], 'criteria_ranges': [["Standard", "Premium", "Premium", "Premium", "Standard"], ["Q3", "Q4", "Q4", "Q4", "Q3"]], 'criteria_values': ["Premium", "Q4"]}, Decimal('25000'), "Financial example: Lowest Q4 premium cost"),
    ]
    all_passed &= test_function("MINIFS", MINIFS, minifs_tests, ctx=ctx)

    # Test SUMPRODUCT function
    def test_sumproduct():
        print(f"\n=== Testing SUMPRODUCT ===")
        passed = 0
        failed = 0

        test_cases = [
            ((ctx, [1, 2, 3], [4, 5, 6]), Decimal('32'), "Basic SUMPRODUCT: (1*4) + (2*5) + (3*6)"),
            ((ctx, [10, 20], [5, 3], [2, 1]), Decimal('160'), "Three ranges: (10*5*2) + (20*3*1)"),
            ((ctx, [100, 200, 150, 300], [50.25, 75.80, 120.50, 45.75]), Decimal('51985'), "Financial example: Portfolio value"),
            ((ctx, pl.Series([2, 3]), pl.Series([4, 5])), Decimal('23'), "SUMPRODUCT with Polars Series"),
        ]

        for i, (args, expected, description) in enumerate(test_cases, 1):
            try:
                result = SUMPRODUCT(*args)

                if abs(result - expected) < Decimal('1e-10'):
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

        print(f"Results for SUMPRODUCT: {passed} passed, {failed} failed")
        return failed == 0

    all_passed &= test_sumproduct()

    # Test COUNTBLANK function
    countblank_tests = [
        ({'range_to_evaluate': [1, None, 3, "", 5, None]}, 3, "Count nulls and empty strings"),
        ({'range_to_evaluate': ["A", "B", "", None, "C"]}, 2, "Count blanks in text data"),
        ({'range_to_evaluate': [100000, None, 120000, None, 110000, 130000]}, 2, "Financial example: Missing revenue data"),
        ({'range_to_evaluate': pl.Series([1, None, 3, None])}, 2, "Count blanks with Polars Series"),
    ]
    all_passed &= test_function("COUNTBLANK", COUNTBLANK, countblank_tests, ctx=ctx)

    # Test COUNTA function
    counta_tests = [
        ({'range_to_evaluate': [1, None, 3, "", 5, 0]}, 4, "Count non-empty values (0 is not blank)"),
        ({'range_to_evaluate': ["A", "B", "", None, "C"]}, 3, "Count non-empty text values"),
        ({'range_to_evaluate': [1000, None, 2500, 0, "", 3000, 1500]}, 5, "Financial example: Valid transaction records"),
        ({'range_to_evaluate': pl.Series([1, None, 3, 4])}, 3, "Count non-empty with Polars Series"),
    ]
    all_passed &= test_function("COUNTA", COUNTA, counta_tests, ctx=ctx)

    # Test AGGREGATE function
    aggregate_tests = [
        ({'function_num': 9, 'options': 2, 'array': [10, "Error", 20, 30]}, Decimal('60'), "SUM ignoring errors"),
        ({'function_num': 4, 'options': 0, 'array': [10, 20, 30, 40]}, Decimal('40'), "MAX function"),
        ({'function_num': 1, 'options': 2, 'array': [10, None, 20, 30]}, Decimal('20'), "AVERAGE ignoring errors"),
        ({'function_num': 14, 'options': 0, 'array': [10, 20, 30, 40], 'k': 2}, Decimal('30'), "LARGE function (2nd largest)"),
        ({'function_num': 15, 'options': 0, 'array': [10, 20, 30, 40], 'k': 2}, Decimal('20'), "SMALL function (2nd smallest)"),
        ({'function_num': 12, 'options': 0, 'array': [10, 20, 30, 40]}, Decimal('25'), "MEDIAN function"),
    ]
    all_passed &= test_function("AGGREGATE", AGGREGATE, aggregate_tests)

    # Test SUBTOTAL function
    subtotal_tests = [
        ({'function_num': 109, 'ref1': [10, 20, 30, 40]}, Decimal('100'), "SUM subtotal"),
        ({'function_num': 101, 'ref1': [10, 20, 30, 40]}, Decimal('25'), "AVERAGE subtotal"),
        ({'function_num': 104, 'ref1': [10, 20, 30, 40]}, Decimal('40'), "MAX subtotal"),
        ({'function_num': 105, 'ref1': [10, 20, 30, 40]}, Decimal('10'), "MIN subtotal"),
        ({'function_num': 102, 'ref1': [10, 20, 30, 40]}, Decimal('4'), "COUNT subtotal"),
        ({'function_num': 106, 'ref1': [2, 3, 4]}, Decimal('24'), "PRODUCT subtotal"),
    ]
    all_passed &= test_function("SUBTOTAL", SUBTOTAL, subtotal_tests)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test empty input validation for COUNTIF
    try:
        COUNTIF(ctx, [], criteria=">0")
        print("✗ Empty input validation failed for COUNTIF")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty input validation passed for COUNTIF")
        error_tests_passed += 1

    # Test mismatched lengths for COUNTIFS
    try:
        COUNTIFS(ctx, [[1, 2, 3], [1, 2]], criteria_values=[">1", "A"])
        print("✗ Mismatched length validation failed for COUNTIFS")
        error_tests_failed += 1
    except (ValidationError, CalculationError):
        print("✓ Mismatched length validation passed for COUNTIFS")
        error_tests_passed += 1

    # Test no matching criteria for AVERAGEIF
    try:
        AVERAGEIF(ctx, [1, 2, 3], criteria=">10")
        print("✗ No matching criteria validation failed for AVERAGEIF")
        error_tests_failed += 1
    except CalculationError:
        print("✓ No matching criteria validation passed for AVERAGEIF")
        error_tests_passed += 1

    # Test invalid function number for AGGREGATE
    try:
        AGGREGATE(99, options=0, array=[1, 2, 3])
        print("✗ Invalid function number validation failed for AGGREGATE")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Invalid function number validation passed for AGGREGATE")
        error_tests_passed += 1

    # Test insufficient ranges for SUMPRODUCT
    try:
        SUMPRODUCT([1, 2, 3])
        print("✗ Insufficient ranges validation failed for SUMPRODUCT")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Insufficient ranges validation passed for SUMPRODUCT")
        error_tests_passed += 1

    # Test criteria parsing edge cases
    print("\n=== Testing Criteria Parsing ===")
    criteria_tests_passed = 0
    criteria_tests_failed = 0

    # Test wildcard patterns
    try:
        result = COUNTIF(ctx, ["Apple", "Banana", "Apricot", "Orange"], criteria="A*")
        if result == 2:  # Should match "Apple" and "Apricot"
            print("✓ Wildcard pattern matching passed")
            criteria_tests_passed += 1
        else:
            print(f"✗ Wildcard pattern matching failed: expected 2, got {result}")
            criteria_tests_failed += 1
    except Exception as e:
        print(f"✗ Wildcard pattern matching error: {e}")
        criteria_tests_failed += 1

    # Test null criteria
    try:
        result = COUNTIF(ctx, [1, None, 3, None, 5], criteria=None)
        if result == 2:  # Should match the None values
            print("✓ Null criteria matching passed")
            criteria_tests_passed += 1
        else:
            print(f"✗ Null criteria matching failed: expected 2, got {result}")
            criteria_tests_failed += 1
    except Exception as e:
        print(f"✗ Null criteria matching error: {e}")
        criteria_tests_failed += 1

    # Test not equal operator
    try:
        result = COUNTIF(ctx, [1, 2, 3, 2, 4], criteria="<>2")
        if result == 3:  # Should match 1, 3, 4
            print("✓ Not equal operator passed")
            criteria_tests_passed += 1
        else:
            print(f"✗ Not equal operator failed: expected 3, got {result}")
            criteria_tests_failed += 1
    except Exception as e:
        print(f"✗ Not equal operator error: {e}")
        criteria_tests_failed += 1

    print(f"Criteria parsing tests: {criteria_tests_passed} passed, {criteria_tests_failed} failed")
    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")

    all_passed &= (error_tests_failed == 0 and criteria_tests_failed == 0)

    # Performance test with larger datasets
    print("\n=== Performance Testing ===")
    performance_passed = 0
    performance_failed = 0

    try:
        # Test with 10,000 records
        large_data = list(range(10000))
        large_categories = ["A" if i % 3 == 0 else "B" if i % 3 == 1 else "C" for i in range(10000)]

        # Test COUNTIFS performance
        start_time = __import__('time').time()
        result = COUNTIFS(ctx, [large_data, large_categories], criteria_values=[">5000", "A"])
        end_time = __import__('time').time()

        if end_time - start_time < 1.0:  # Should complete in under 1 second
            print(f"✓ COUNTIFS performance test passed: {result} matches in {end_time - start_time:.3f}s")
            performance_passed += 1
        else:
            print(f"✗ COUNTIFS performance test failed: took {end_time - start_time:.3f}s")
            performance_failed += 1

    except Exception as e:
        print(f"✗ Performance test error: {e}")
        performance_failed += 1

    print(f"Performance tests: {performance_passed} passed, {performance_failed} failed")
    all_passed &= (performance_failed == 0)

    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All conditional aggregation functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the failed tests above.")
    print("="*60)

    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
