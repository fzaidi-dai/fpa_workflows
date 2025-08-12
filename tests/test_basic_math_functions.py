#!/usr/bin/env python3
"""
Test script for basic math and aggregation functions.
Tests all functions in the basic_math_and_aggregation.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.basic_math_and_aggregation.basic_math_and_aggregation import (
    SUM, AVERAGE, MIN, MAX, PRODUCT, MEDIAN, MODE, PERCENTILE,
    POWER, SQRT, EXP, LN, LOG, ABS, SIGN, MOD, ROUND, ROUNDUP, ROUNDDOWN,
    WEIGHTED_AVERAGE, GEOMETRIC_MEAN, HARMONIC_MEAN, CUMSUM, CUMPROD, VARIANCE_WEIGHTED,
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

def run_all_tests():
    """Run all tests for the basic math functions."""
    print("Starting comprehensive test of basic math and aggregation functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    all_passed = True

    # Test SUM function
    sum_tests = [
        ({'values': [1, 2, 3, 4, 5]}, Decimal('15'), "Basic sum with integers"),
        ({'values': [1.5, 2.5, 3.5]}, Decimal('7.5'), "Sum with floats"),
        ({'values': [Decimal('1.1'), Decimal('2.2'), Decimal('3.3')]}, Decimal('6.6'), "Sum with Decimals"),
        ({'values': pl.Series([1, 2, 3, 4, 5])}, Decimal('15'), "Sum with Polars Series"),
        ({'values': np.array([1, 2, 3, 4, 5])}, Decimal('15'), "Sum with NumPy array"),
        ({'values': "test_data.csv"}, Decimal('15'), "Sum with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('15'), "Sum with Parquet file input"),
    ]
    all_passed &= test_function("SUM", SUM, sum_tests, ctx=ctx)

    # Test AVERAGE function
    average_tests = [
        ({'values': [10, 20, 30]}, Decimal('20'), "Basic average"),
        ({'values': [1.5, 2.5, 3.5]}, Decimal('2.5'), "Average with floats"),
        ({'values': pl.Series([10, 20, 30])}, Decimal('20'), "Average with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('3'), "Average with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('3'), "Average with Parquet file input"),
    ]
    all_passed &= test_function("AVERAGE", AVERAGE, average_tests, ctx=ctx)

    # Test MIN function
    min_tests = [
        ({'values': [10, 5, 20, 3]}, Decimal('3'), "Basic min"),
        ({'values': [-5, 0, 5]}, Decimal('-5'), "Min with negative numbers"),
        ({'values': pl.Series([10, 5, 20, 3])}, Decimal('3'), "Min with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('1'), "Min with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('1'), "Min with Parquet file input"),
    ]
    all_passed &= test_function("MIN", MIN, min_tests, ctx=ctx)

    # Test MAX function
    max_tests = [
        ({'values': [10, 5, 20, 3]}, Decimal('20'), "Basic max"),
        ({'values': [-5, 0, 5]}, Decimal('5'), "Max with negative numbers"),
        ({'values': pl.Series([10, 5, 20, 3])}, Decimal('20'), "Max with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('5'), "Max with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('5'), "Max with Parquet file input"),
    ]
    all_passed &= test_function("MAX", MAX, max_tests, ctx=ctx)

    # Test PRODUCT function
    product_tests = [
        ({'values': [2, 3, 4]}, Decimal('24'), "Basic product"),
        ({'values': [1.5, 2, 3]}, Decimal('9'), "Product with floats"),
        ({'values': [Decimal('2'), Decimal('3'), Decimal('4')]}, Decimal('24'), "Product with Decimals"),
        ({'values': "test_data.csv"}, Decimal('120'), "Product with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('120'), "Product with Parquet file input"),
    ]
    all_passed &= test_function("PRODUCT", PRODUCT, product_tests, ctx=ctx)

    # Test MEDIAN function
    median_tests = [
        ({'values': [1, 2, 3, 4, 5]}, Decimal('3'), "Median with odd count"),
        ({'values': [1, 2, 3, 4]}, Decimal('2.5'), "Median with even count"),
        ({'values': pl.Series([1, 2, 3, 4, 5])}, Decimal('3'), "Median with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('3'), "Median with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('3'), "Median with Parquet file input"),
    ]
    all_passed &= test_function("MEDIAN", MEDIAN, median_tests, ctx=ctx)

    # Test MODE function
    mode_tests = [
        ({'values': [1, 2, 2, 3, 3, 3]}, Decimal('3'), "Single mode"),
        ({'values': [1, 1, 2, 2, 3]}, [Decimal('1'), Decimal('2')], "Multiple modes"),
        ({'values': np.array([1, 2, 2, 3, 3, 3])}, Decimal('3'), "Mode with NumPy array"),
        ({'values': "test_data.csv"}, [Decimal('1'), Decimal('2'), Decimal('3'), Decimal('4'), Decimal('5')], "Mode with CSV file input (all unique, returns all)"),
        ({'values': "test_data.parquet"}, [Decimal('1'), Decimal('2'), Decimal('3'), Decimal('4'), Decimal('5')], "Mode with Parquet file input (all unique, returns all)"),
    ]
    all_passed &= test_function("MODE", MODE, mode_tests, ctx=ctx)

    # Test PERCENTILE function
    percentile_tests = [
        ({'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'percentile_value': 0.75}, Decimal('7.75'), "75th percentile"),
        ({'values': [1, 2, 3, 4, 5], 'percentile_value': 0.5}, Decimal('3'), "50th percentile (median)"),
        ({'values': pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 'percentile_value': 0.75}, Decimal('7.75'), "Percentile with Polars Series"),
        ({'values': "test_data.csv", 'percentile_value': 0.5}, Decimal('3'), "Percentile with CSV file input"),
        ({'values': "test_data.parquet", 'percentile_value': 0.5}, Decimal('3'), "Percentile with Parquet file input"),
    ]
    all_passed &= test_function("PERCENTILE", PERCENTILE, percentile_tests, ctx=ctx)

    # Test POWER function
    power_tests = [
        ({'values': [2], 'power': 3}, [Decimal('8')], "Integer power"),
        ({'values': [1.05], 'power': 10}, [Decimal('1.62889462677744140625')], "Decimal power"),
        ({'values': [Decimal('2')], 'power': Decimal('3')}, [Decimal('8')], "Decimal inputs"),
        ({'values': pl.Series([2]), 'power': 3}, [Decimal('8')], "Power with Polars Series"),
        ({'values': np.array([2]), 'power': 3}, [Decimal('8')], "Power with NumPy array"),
        ({'values': "test_data.csv", 'power': 2, 'output_filename': "power_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/power_results_csv.parquet"), "Power with CSV file input and output"),
        ({'values': "test_data.parquet", 'power': 2, 'output_filename': "power_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/power_results_parquet.parquet"), "Power with Parquet file input and output"),
    ]
    all_passed &= test_function("POWER", POWER, power_tests, ctx=ctx)

    # Test SQRT function
    sqrt_tests = [
        ({'values': [25]}, [Decimal('5')], "Perfect square"),
        ({'values': [2]}, [Decimal('1.4142135623730950488016887242097')], "Square root of 2"),
        ({'values': [Decimal('16')]}, [Decimal('4')], "Decimal input"),
        ({'values': pl.Series([25])}, [Decimal('5')], "Square root with Polars Series"),
        ({'values': np.array([25])}, [Decimal('5')], "Square root with NumPy array"),
        ({'values': "test_data.csv", 'output_filename': "sqrt_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/sqrt_results_csv.parquet"), "SQRT with CSV file input and output"),
        ({'values': "test_data.parquet", 'output_filename': "sqrt_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/sqrt_results_parquet.parquet"), "SQRT with Parquet file input and output"),
    ]
    all_passed &= test_function("SQRT", SQRT, sqrt_tests, ctx=ctx)

    # Test EXP function
    exp_tests = [
        ({'values': [1]}, [Decimal('2.7182818284590452353602874713527')], "e^1"),
        ({'values': [0]}, [Decimal('1')], "e^0"),
        ({'values': [Decimal('1')]}, [Decimal('2.7182818284590452353602874713527')], "Decimal input"),
        ({'values': pl.Series([1])}, [Decimal('2.7182818284590452353602874713527')], "EXP with Polars Series"),
        ({'values': np.array([1])}, [Decimal('2.7182818284590452353602874713527')], "EXP with NumPy array"),
        ({'values': "test_data.csv", 'output_filename': "exp_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/exp_results_csv.parquet"), "EXP with CSV file input and output"),
        ({'values': "test_data.parquet", 'output_filename': "exp_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/exp_results_parquet.parquet"), "EXP with Parquet file input and output"),
    ]
    all_passed &= test_function("EXP", EXP, exp_tests, ctx=ctx)

    # Test LN function
    ln_tests = [
        ({'values': [2.718281828459045]}, [Decimal('1.0')], "Natural log of e"),
        ({'values': [1]}, [Decimal('0')], "Natural log of 1"),
        ({'values': [Decimal('10')]}, [Decimal('2.3025850929940456840179914546844')], "Natural log of 10"),
        ({'values': pl.Series([2.718281828459045])}, [Decimal('1.0')], "LN with Polars Series"),
        ({'values': np.array([2.718281828459045])}, [Decimal('1.0')], "LN with NumPy array"),
        ({'values': "test_data.csv", 'output_filename': "ln_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/ln_results_csv.parquet"), "LN with CSV file input and output"),
        ({'values': "test_data.parquet", 'output_filename': "ln_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/ln_results_parquet.parquet"), "LN with Parquet file input and output"),
    ]
    all_passed &= test_function("LN", LN, ln_tests, ctx=ctx)

    # Test LOG function
    log_tests = [
        ({'values': [100], 'base': 10}, [Decimal('2')], "Log base 10 of 100"),
        ({'values': [8], 'base': 2}, [Decimal('3')], "Log base 2 of 8"),
        ({'values': [100]}, [Decimal('2')], "Default base 10"),
        ({'values': pl.Series([100]), 'base': 10}, [Decimal('2')], "LOG with Polars Series"),
        ({'values': np.array([100]), 'base': 10}, [Decimal('2')], "LOG with NumPy array"),
        ({'values': "test_data.csv", 'base': 10, 'output_filename': "log_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/log_results_csv.parquet"), "LOG with CSV file input and output"),
        ({'values': "test_data.parquet", 'base': 10, 'output_filename': "log_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/log_results_parquet.parquet"), "LOG with Parquet file input and output"),
    ]
    all_passed &= test_function("LOG", LOG, log_tests, ctx=ctx)

    # Test ABS function
    abs_tests = [
        ({'values': [-10]}, [Decimal('10')], "Absolute value of negative"),
        ({'values': [10]}, [Decimal('10')], "Absolute value of positive"),
        ({'values': [Decimal('-5.5')]}, [Decimal('5.5')], "Decimal input"),
        ({'values': pl.Series([-10])}, [Decimal('10')], "ABS with Polars Series"),
        ({'values': np.array([-10])}, [Decimal('10')], "ABS with NumPy array"),
        ({'values': "test_data.csv", 'output_filename': "abs_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/abs_results_csv.parquet"), "ABS with CSV file input and output"),
        ({'values': "test_data.parquet", 'output_filename': "abs_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/abs_results_parquet.parquet"), "ABS with Parquet file input and output"),
    ]
    all_passed &= test_function("ABS", ABS, abs_tests, ctx=ctx)

    # Test SIGN function
    sign_tests = [
        ({'values': [-15]}, [-1], "Sign of negative"),
        ({'values': [15]}, [1], "Sign of positive"),
        ({'values': [0]}, [0], "Sign of zero"),
        ({'values': pl.Series([-15])}, [-1], "SIGN with Polars Series"),
        ({'values': np.array([-15])}, [-1], "SIGN with NumPy array"),
        ({'values': "test_data.csv"}, [1, 1, 1, 1, 1], "SIGN with CSV file input"),
        ({'values': "test_data.parquet"}, [1, 1, 1, 1, 1], "SIGN with Parquet file input"),
    ]
    all_passed &= test_function("SIGN", SIGN, sign_tests, ctx=ctx)

    # Test MOD function
    mod_tests = [
        ({'dividends': [23], 'divisors': [5]}, [Decimal('3')], "Basic modulo"),
        ({'dividends': [10], 'divisors': [3]}, [Decimal('1')], "Modulo with remainder"),
        ({'dividends': [Decimal('23')], 'divisors': [Decimal('5')]}, [Decimal('3')], "Decimal inputs"),
        ({'dividends': pl.Series([23]), 'divisors': pl.Series([5])}, [Decimal('3')], "MOD with Polars Series"),
        ({'dividends': np.array([23]), 'divisors': np.array([5])}, [Decimal('3')], "MOD with NumPy array"),
        ({'dividends': "test_data.csv", 'divisors': [2]}, [Decimal('1'), Decimal('0'), Decimal('1'), Decimal('0'), Decimal('1')], "MOD with CSV file input"),
        ({'dividends': "test_data.parquet", 'divisors': [2]}, [Decimal('1'), Decimal('0'), Decimal('1'), Decimal('0'), Decimal('1')], "MOD with Parquet file input"),
    ]
    all_passed &= test_function("MOD", MOD, mod_tests, ctx=ctx)

    # Test ROUND function
    round_tests = [
        ({'values': [3.14159], 'num_digits': 2}, [Decimal('3.14')], "Round to 2 decimal places"),
        ({'values': [3.14159], 'num_digits': 0}, [Decimal('3')], "Round to integer"),
        ({'values': [1234.5678], 'num_digits': -1}, [Decimal('1230')], "Round to tens place"),
        ({'values': pl.Series([3.14159]), 'num_digits': 2}, [Decimal('3.14')], "ROUND with Polars Series"),
        ({'values': np.array([3.14159]), 'num_digits': 2}, [Decimal('3.14')], "ROUND with NumPy array"),
        ({'values': "test_data.csv", 'num_digits': 2}, [Decimal('1.00'), Decimal('2.00'), Decimal('3.00'), Decimal('4.00'), Decimal('5.00')], "ROUND with CSV file input"),
        ({'values': "test_data.parquet", 'num_digits': 2}, [Decimal('1.00'), Decimal('2.00'), Decimal('3.00'), Decimal('4.00'), Decimal('5.00')], "ROUND with Parquet file input"),
    ]
    all_passed &= test_function("ROUND", ROUND, round_tests, ctx=ctx)

    # Test ROUNDUP function
    roundup_tests = [
        ({'values': [3.14159], 'num_digits': 2}, [Decimal('3.15')], "Round up to 2 decimal places"),
        ({'values': [3.14159], 'num_digits': 0}, [Decimal('4')], "Round up to integer"),
        ({'values': [-3.14159], 'num_digits': 2}, [Decimal('-3.14')], "Round up negative number"),
        ({'values': pl.Series([3.14159]), 'num_digits': 2}, [Decimal('3.15')], "ROUNDUP with Polars Series"),
        ({'values': np.array([3.14159]), 'num_digits': 2}, [Decimal('3.15')], "ROUNDUP with NumPy array"),
        ({'values': "test_data.csv", 'num_digits': 2}, [Decimal('1.00'), Decimal('2.00'), Decimal('3.00'), Decimal('4.00'), Decimal('5.00')], "ROUNDUP with CSV file input"),
        ({'values': "test_data.parquet", 'num_digits': 2}, [Decimal('1.00'), Decimal('2.00'), Decimal('3.00'), Decimal('4.00'), Decimal('5.00')], "ROUNDUP with Parquet file input"),
    ]
    all_passed &= test_function("ROUNDUP", ROUNDUP, roundup_tests, ctx=ctx)

    # Test ROUNDDOWN function
    rounddown_tests = [
        ({'values': [3.14159], 'num_digits': 2}, [Decimal('3.14')], "Round down to 2 decimal places"),
        ({'values': [3.14159], 'num_digits': 0}, [Decimal('3')], "Round down to integer"),
        ({'values': [-3.14159], 'num_digits': 2}, [Decimal('-3.15')], "Round down negative number"),
        ({'values': pl.Series([3.14159]), 'num_digits': 2}, [Decimal('3.14')], "ROUNDDOWN with Polars Series"),
        ({'values': np.array([3.14159]), 'num_digits': 2}, [Decimal('3.14')], "ROUNDDOWN with NumPy array"),
        ({'values': "test_data.csv", 'num_digits': 2}, [Decimal('1.00'), Decimal('2.00'), Decimal('3.00'), Decimal('4.00'), Decimal('5.00')], "ROUNDDOWN with CSV file input"),
        ({'values': "test_data.parquet", 'num_digits': 2}, [Decimal('1.00'), Decimal('2.00'), Decimal('3.00'), Decimal('4.00'), Decimal('5.00')], "ROUNDDOWN with Parquet file input"),
    ]
    all_passed &= test_function("ROUNDDOWN", ROUNDDOWN, rounddown_tests, ctx=ctx)

    # Test WEIGHTED_AVERAGE function
    weighted_avg_tests = [
        ({'values': [100, 200, 300], 'weights': [0.2, 0.3, 0.5]}, Decimal('230'), "Basic weighted average"),
        ({'values': [10, 20, 30], 'weights': [0.5, 0.3, 0.2]}, Decimal('17'), "Different weights"),
        ({'values': [Decimal('100'), Decimal('200')], 'weights': [Decimal('0.4'), Decimal('0.6')]}, Decimal('160'), "Decimal inputs"),
        ({'values': "test_data.csv", 'weights': [0.2, 0.2, 0.2, 0.2, 0.2]}, Decimal('3'), "Weighted average with CSV file input"),
        ({'values': "test_data.parquet", 'weights': [0.2, 0.2, 0.2, 0.2, 0.2]}, Decimal('3'), "Weighted average with Parquet file input"),
    ]
    all_passed &= test_function("WEIGHTED_AVERAGE", WEIGHTED_AVERAGE, weighted_avg_tests, ctx=ctx)

    # Test GEOMETRIC_MEAN function
    geometric_mean_tests = [
        ({'values': [1.05, 1.08, 1.12, 1.03]}, Decimal('1.069'), "Geometric mean of growth rates"),
        ({'values': [2, 4, 8]}, Decimal('4'), "Geometric mean of powers of 2"),
        ({'values': pl.Series([1.05, 1.08, 1.12, 1.03])}, Decimal('1.069'), "Geometric mean with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('2.605'), "Geometric mean with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('2.605'), "Geometric mean with Parquet file input"),
    ]
    all_passed &= test_function("GEOMETRIC_MEAN", GEOMETRIC_MEAN, geometric_mean_tests, ctx=ctx)

    # Test HARMONIC_MEAN function
    harmonic_mean_tests = [
        ({'values': [2, 4, 8]}, Decimal('3.4285714285714285714285714285714'), "Harmonic mean"),
        ({'values': [1.5, 2, 3]}, Decimal('1.9999999999999999999999999999999'), "Harmonic mean with decimals"),
        ({'values': pl.Series([2, 4, 8])}, Decimal('3.4285714285714285714285714285714'), "Harmonic mean with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('2.189781021897810218978102190'), "Harmonic mean with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('2.189781021897810218978102190'), "Harmonic mean with Parquet file input"),
    ]
    all_passed &= test_function("HARMONIC_MEAN", HARMONIC_MEAN, harmonic_mean_tests, ctx=ctx)

    # Test CUMSUM function
    cumsum_tests = [
        ({'values': [10, 20, 30, 40]}, [Decimal('10'), Decimal('30'), Decimal('60'), Decimal('100')], "Basic cumulative sum"),
        ({'values': [1, 2, 3, 4, 5]}, [Decimal('1'), Decimal('3'), Decimal('6'), Decimal('10'), Decimal('15')], "Cumulative sum of integers"),
        ({'values': pl.Series([10, 20, 30, 40])}, [Decimal('10'), Decimal('30'), Decimal('60'), Decimal('100')], "Cumulative sum with Polars Series"),
        ({'values': "test_data.csv", 'output_filename': "cumsum_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/cumsum_results_csv.parquet"), "CUMSUM with CSV file input and output"),
        ({'values': "test_data.parquet", 'output_filename': "cumsum_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/cumsum_results_parquet.parquet"), "CUMSUM with Parquet file input and output"),
    ]
    all_passed &= test_function("CUMSUM", CUMSUM, cumsum_tests, ctx=ctx)

    # Test CUMPROD function
    cumprod_tests = [
        ({'values': [1.05, 1.08, 1.12]}, [Decimal('1.050'), Decimal('1.134'), Decimal('1.270')], "Basic cumulative product"),
        ({'values': [2, 3, 4, 5]}, [Decimal('2'), Decimal('6'), Decimal('24'), Decimal('120')], "Cumulative product of integers"),
        ({'values': pl.Series([1.05, 1.08, 1.12])}, [Decimal('1.050'), Decimal('1.134'), Decimal('1.270')], "Cumulative product with Polars Series"),
        ({'values': "test_data.csv", 'output_filename': "cumprod_results_csv.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/cumprod_results_csv.parquet"), "CUMPROD with CSV file input and output"),
        ({'values': "test_data.parquet", 'output_filename': "cumprod_results_parquet.parquet"}, Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/cumprod_results_parquet.parquet"), "CUMPROD with Parquet file input and output"),
    ]
    all_passed &= test_function("CUMPROD", CUMPROD, cumprod_tests, ctx=ctx)

    # Test VARIANCE_WEIGHTED function
    variance_weighted_tests = [
        ({'values': [100, 200, 300], 'weights': [0.2, 0.3, 0.5]}, Decimal('6100'), "Weighted variance"),
        ({'values': [10, 20, 30], 'weights': [0.5, 0.3, 0.2]}, Decimal('61'), "Weighted variance with different weights"),
        ({'values': [Decimal('100'), Decimal('200')], 'weights': [Decimal('0.4'), Decimal('0.6')]}, Decimal('2400'), "Decimal inputs"),
        ({'values': "test_data.csv", 'weights': [0.2, 0.2, 0.2, 0.2, 0.2]}, Decimal('2'), "Variance weighted with CSV file input"),
        ({'values': "test_data.parquet", 'weights': [0.2, 0.2, 0.2, 0.2, 0.2]}, Decimal('2'), "Variance weighted with Parquet file input"),
    ]
    all_passed &= test_function("VARIANCE_WEIGHTED", VARIANCE_WEIGHTED, variance_weighted_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test empty input validation
    try:
        SUM(ctx, [])
        print("✗ Empty input validation failed for SUM")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty input validation passed for SUM")
        error_tests_passed += 1

    # Test invalid percentile value
    try:
        PERCENTILE(ctx, [1, 2, 3], percentile_value=1.5)
        print("✗ Invalid percentile validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid percentile validation passed")
        error_tests_passed += 1

    # Test division by zero
    try:
        MOD(ctx, [10], divisors=[0])
        print("✗ Division by zero validation failed for MOD")
        error_tests_failed += 1
    except CalculationError:
        print("✓ Division by zero validation passed for MOD")
        error_tests_passed += 1

    # Test negative square root
    try:
        SQRT(ctx, [-1])
        print("✗ Negative square root validation failed")
        error_tests_failed += 1
    except CalculationError:
        print("✓ Negative square root validation passed")
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
