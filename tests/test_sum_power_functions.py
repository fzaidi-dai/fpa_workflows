#!/usr/bin/env python3
"""
Focused test script for SUM and POWER functions with new functionality.
Tests file input/output and RunContext integration.
"""

import sys
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.basic_math_and_aggregation.basic_math_and_aggregation import (
    SUM, POWER, ValidationError, CalculationError, DataQualityError
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
            elif isinstance(expected, list):
                # Handle list comparisons (for functions that return lists)
                if isinstance(result, list):
                    if len(result) == len(expected):
                        if all(abs(r - e) < Decimal('1e-10') for r, e in zip(result, expected)):
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
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected list: {expected}, Got single value: {result}")
                    failed += 1
            else:
                # Handle other types (int, etc.)
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

def run_focused_tests():
    """Run focused tests for SUM and POWER functions."""
    print("Starting focused test of SUM and POWER functions with new functionality...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    all_passed = True

    # Test SUM function with new functionality
    sum_tests = [
        ({'values': [1, 2, 3, 4, 5]}, Decimal('15'), "Basic sum with integers"),
        ({'values': "test_data.csv"}, Decimal('15'), "Sum with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('15'), "Sum with Parquet file input"),
    ]
    all_passed &= test_function("SUM", SUM, sum_tests, ctx=ctx)

    # Test POWER function with new functionality
    power_tests = [
        ({'values': [2], 'power': 3}, [Decimal('8')], "Integer power"),
        ({'values': "test_data.csv", 'power': 2, 'output_filename': "power_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/power_results_csv.parquet"), "Power with CSV file input and output"),
        ({'values': "test_data.parquet", 'power': 2, 'output_filename': "power_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/power_results_parquet.parquet"), "Power with Parquet file input and output"),
    ]
    all_passed &= test_function("POWER", POWER, power_tests, ctx=ctx)

    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL FOCUSED TESTS PASSED! SUM and POWER functions are working correctly.")
    else:
        print("❌ SOME FOCUSED TESTS FAILED! Please review the failed tests above.")
    print("="*50)

    return all_passed

if __name__ == "__main__":
    success = run_focused_tests()
    sys.exit(0 if success else 1)
