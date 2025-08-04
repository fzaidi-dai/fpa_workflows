#!/usr/bin/env python3
"""
Test script for statistical and trend analysis functions.
Tests all functions in the statistical_and_trend_analysis_functions.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.statistical_and_trend_analysis_functions.statistical_and_trend_analysis_functions import (
    STDEV_P, STDEV_S, VAR_P, VAR_S, MEDIAN, MODE, CORREL, COVARIANCE_P, COVARIANCE_S,
    TREND, FORECAST, FORECAST_LINEAR, GROWTH, SLOPE, INTERCEPT, RSQ, LINEST, LOGEST,
    RANK, PERCENTRANK,
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
            elif isinstance(expected, Path):
                # Handle Path comparisons (for functions that return file paths)
                if isinstance(result, Path):
                    if result.exists():
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected file to exist at: {result}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected Path, Got: {type(result)}")
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
    """Run all tests for the statistical and trend analysis functions."""
    print("Starting comprehensive test of statistical and trend analysis functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    all_passed = True

    # Test STDEV_P function
    stdev_p_tests = [
        ({'values': [2, 4, 4, 4, 5, 5, 7, 9]}, Decimal('2.0'), "Population standard deviation"),
        ({'values': [1, 2, 3, 4, 5]}, Decimal('1.4142135623730950488016887242097'), "Standard deviation of 1-5"),
        ({'values': pl.Series([2, 4, 4, 4, 5, 5, 7, 9])}, Decimal('2.0'), "STDEV_P with Polars Series"),
        ({'values': np.array([2, 4, 4, 4, 5, 5, 7, 9])}, Decimal('2.0'), "STDEV_P with NumPy array"),
        ({'values': "test_data.csv"}, Decimal('1.4142135623730951'), "STDEV_P with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('1.4142135623730951'), "STDEV_P with Parquet file input"),
    ]
    all_passed &= test_function("STDEV_P", STDEV_P, stdev_p_tests, ctx=ctx)

    # Test STDEV_S function
    stdev_s_tests = [
        ({'values': [2, 4, 4, 4, 5, 5, 7, 9]}, Decimal('2.1380899352993'), "Sample standard deviation"),
        ({'values': [1, 2, 3, 4, 5]}, Decimal('1.5811388300841898'), "Sample standard deviation of 1-5"),
        ({'values': pl.Series([2, 4, 4, 4, 5, 5, 7, 9])}, Decimal('2.1380899352993'), "STDEV_S with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('1.5811388300841898'), "STDEV_S with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('1.5811388300841898'), "STDEV_S with Parquet file input"),
    ]
    all_passed &= test_function("STDEV_S", STDEV_S, stdev_s_tests, ctx=ctx)

    # Test VAR_P function
    var_p_tests = [
        ({'values': [2, 4, 4, 4, 5, 5, 7, 9]}, Decimal('4.0'), "Population variance"),
        ({'values': [1, 2, 3, 4, 5]}, Decimal('2.0'), "Population variance of 1-5"),
        ({'values': pl.Series([2, 4, 4, 4, 5, 5, 7, 9])}, Decimal('4.0'), "VAR_P with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('2.0'), "VAR_P with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('2.0'), "VAR_P with Parquet file input"),
    ]
    all_passed &= test_function("VAR_P", VAR_P, var_p_tests, ctx=ctx)

    # Test VAR_S function
    var_s_tests = [
        ({'values': [2, 4, 4, 4, 5, 5, 7, 9]}, Decimal('4.571428571428571'), "Sample variance"),
        ({'values': [1, 2, 3, 4, 5]}, Decimal('2.5'), "Sample variance of 1-5"),
        ({'values': pl.Series([2, 4, 4, 4, 5, 5, 7, 9])}, Decimal('4.571428571428571'), "VAR_S with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('2.5'), "VAR_S with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('2.5'), "VAR_S with Parquet file input"),
    ]
    all_passed &= test_function("VAR_S", VAR_S, var_s_tests, ctx=ctx)

    # Test MEDIAN function
    median_tests = [
        ({'values': [1, 2, 3, 4, 5]}, Decimal('3'), "Median with odd count"),
        ({'values': [1, 2, 3, 4]}, Decimal('2.5'), "Median with even count"),
        ({'values': [5, 1, 3, 2, 4]}, Decimal('3'), "Median with unsorted data"),
        ({'values': pl.Series([1, 2, 3, 4, 5])}, Decimal('3'), "MEDIAN with Polars Series"),
        ({'values': "test_data.csv"}, Decimal('3'), "MEDIAN with CSV file input"),
        ({'values': "test_data.parquet"}, Decimal('3'), "MEDIAN with Parquet file input"),
    ]
    all_passed &= test_function("MEDIAN", MEDIAN, median_tests, ctx=ctx)

    # Test MODE function
    mode_tests = [
        ({'values': [1, 2, 2, 3, 3, 3]}, Decimal('3'), "Single mode"),
        ({'values': [1, 1, 2, 2, 3]}, [Decimal('1'), Decimal('2')], "Multiple modes"),
        ({'values': [5, 5, 5, 1, 2, 3]}, Decimal('5'), "Clear single mode"),
        ({'values': np.array([1, 2, 2, 3, 3, 3])}, Decimal('3'), "MODE with NumPy array"),
        ({'values': "test_data.csv"}, [Decimal('1'), Decimal('2'), Decimal('3'), Decimal('4'), Decimal('5')], "MODE with CSV file input (all unique)"),
        ({'values': "test_data.parquet"}, [Decimal('1'), Decimal('2'), Decimal('3'), Decimal('4'), Decimal('5')], "MODE with Parquet file input (all unique)"),
    ]
    all_passed &= test_function("MODE", MODE, mode_tests, ctx=ctx)

    # Test CORREL function
    correl_tests = [
        ({'range1': [1, 2, 3, 4, 5], 'range2': [2, 4, 6, 8, 10]}, Decimal('1.0'), "Perfect positive correlation"),
        ({'range1': [1, 2, 3, 4, 5], 'range2': [5, 4, 3, 2, 1]}, Decimal('-1.0'), "Perfect negative correlation"),
        ({'range1': [1, 2, 3, 4, 5], 'range2': [1, 2, 3, 4, 5]}, Decimal('1.0'), "Identical arrays"),
        ({'range1': pl.Series([1, 2, 3, 4, 5]), 'range2': pl.Series([2, 4, 6, 8, 10])}, Decimal('1.0'), "CORREL with Polars Series"),
        ({'range1': "test_data.csv", 'range2': [1, 2, 3, 4, 5]}, Decimal('1.0'), "CORREL with CSV file input"),
        ({'range1': "test_data.parquet", 'range2': [1, 2, 3, 4, 5]}, Decimal('1.0'), "CORREL with Parquet file input"),
    ]
    all_passed &= test_function("CORREL", CORREL, correl_tests, ctx=ctx)

    # Test COVARIANCE_P function
    covariance_p_tests = [
        ({'range1': [1, 2, 3, 4, 5], 'range2': [2, 4, 6, 8, 10]}, Decimal('4.0'), "Population covariance"),
        ({'range1': [1, 2, 3], 'range2': [4, 5, 6]}, Decimal('0.6666666666666666666666666667'), "Small dataset covariance"),
        ({'range1': pl.Series([1, 2, 3, 4, 5]), 'range2': pl.Series([2, 4, 6, 8, 10])}, Decimal('4.0'), "COVARIANCE_P with Polars Series"),
        ({'range1': "test_data.csv", 'range2': [1, 2, 3, 4, 5]}, Decimal('2'), "COVARIANCE_P with CSV file input"),
        ({'range1': "test_data.parquet", 'range2': [1, 2, 3, 4, 5]}, Decimal('2'), "COVARIANCE_P with Parquet file input"),
    ]
    all_passed &= test_function("COVARIANCE_P", COVARIANCE_P, covariance_p_tests, ctx=ctx)

    # Test COVARIANCE_S function
    covariance_s_tests = [
        ({'range1': [1, 2, 3, 4, 5], 'range2': [2, 4, 6, 8, 10]}, Decimal('5.0'), "Sample covariance"),
        ({'range1': [1, 2, 3], 'range2': [4, 5, 6]}, Decimal('1.0'), "Small dataset sample covariance"),
        ({'range1': pl.Series([1, 2, 3, 4, 5]), 'range2': pl.Series([2, 4, 6, 8, 10])}, Decimal('5.0'), "COVARIANCE_S with Polars Series"),
        ({'range1': "test_data.csv", 'range2': [1, 2, 3, 4, 5]}, Decimal('2.5'), "COVARIANCE_S with CSV file input"),
        ({'range1': "test_data.parquet", 'range2': [1, 2, 3, 4, 5]}, Decimal('2.5'), "COVARIANCE_S with Parquet file input"),
    ]
    all_passed &= test_function("COVARIANCE_S", COVARIANCE_S, covariance_s_tests, ctx=ctx)

    # Test FORECAST function
    forecast_tests = [
        ({'new_x': 6, 'known_y': [1, 2, 3, 4, 5], 'known_x': [1, 2, 3, 4, 5]}, Decimal('6.0'), "Linear forecast"),
        ({'new_x': 0, 'known_y': [1, 2, 3, 4, 5], 'known_x': [1, 2, 3, 4, 5]}, Decimal('0.0'), "Forecast at x=0"),
        ({'new_x': 3.5, 'known_y': [2, 4, 6, 8, 10], 'known_x': [1, 2, 3, 4, 5]}, Decimal('7.0'), "Forecast between points"),
        ({'new_x': 6, 'known_y': pl.Series([1, 2, 3, 4, 5]), 'known_x': pl.Series([1, 2, 3, 4, 5])}, Decimal('6.0'), "FORECAST with Polars Series"),
        ({'new_x': 6, 'known_y': "test_data.csv", 'known_x': [1, 2, 3, 4, 5]}, Decimal('6.0'), "FORECAST with CSV file input"),
        ({'new_x': 6, 'known_y': "test_data.parquet", 'known_x': [1, 2, 3, 4, 5]}, Decimal('6.0'), "FORECAST with Parquet file input"),
    ]
    all_passed &= test_function("FORECAST", FORECAST, forecast_tests, ctx=ctx)

    # Test FORECAST_LINEAR function (alias for FORECAST)
    forecast_linear_tests = [
        ({'new_x': 6, 'known_y': [1, 2, 3, 4, 5], 'known_x': [1, 2, 3, 4, 5]}, Decimal('6.0'), "Linear forecast (alias)"),
        ({'new_x': 3.5, 'known_y': [2, 4, 6, 8, 10], 'known_x': [1, 2, 3, 4, 5]}, Decimal('7.0'), "Forecast between points (alias)"),
    ]
    all_passed &= test_function("FORECAST_LINEAR", FORECAST_LINEAR, forecast_linear_tests, ctx=ctx)

    # Test SLOPE function
    slope_tests = [
        ({'known_ys': [1, 2, 3, 4, 5], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "Slope of 1"),
        ({'known_ys': [2, 4, 6, 8, 10], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('2.0'), "Slope of 2"),
        ({'known_ys': [5, 4, 3, 2, 1], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('-1.0'), "Negative slope"),
        ({'known_ys': pl.Series([1, 2, 3, 4, 5]), 'known_xs': pl.Series([1, 2, 3, 4, 5])}, Decimal('1.0'), "SLOPE with Polars Series"),
        ({'known_ys': "test_data.csv", 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "SLOPE with CSV file input"),
        ({'known_ys': "test_data.parquet", 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "SLOPE with Parquet file input"),
    ]
    all_passed &= test_function("SLOPE", SLOPE, slope_tests, ctx=ctx)

    # Test INTERCEPT function
    intercept_tests = [
        ({'known_ys': [2, 4, 6, 8, 10], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('0.0'), "Zero intercept"),
        ({'known_ys': [3, 5, 7, 9, 11], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "Intercept of 1"),
        ({'known_ys': [1, 2, 3, 4, 5], 'known_xs': [0, 1, 2, 3, 4]}, Decimal('1.0'), "Intercept with x starting at 0"),
        ({'known_ys': pl.Series([2, 4, 6, 8, 10]), 'known_xs': pl.Series([1, 2, 3, 4, 5])}, Decimal('0.0'), "INTERCEPT with Polars Series"),
        ({'known_ys': "test_data.csv", 'known_xs': [0, 1, 2, 3, 4]}, Decimal('1.0'), "INTERCEPT with CSV file input"),
        ({'known_ys': "test_data.parquet", 'known_xs': [0, 1, 2, 3, 4]}, Decimal('1.0'), "INTERCEPT with Parquet file input"),
    ]
    all_passed &= test_function("INTERCEPT", INTERCEPT, intercept_tests, ctx=ctx)

    # Test RSQ function
    rsq_tests = [
        ({'known_ys': [1, 2, 3, 4, 5], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "Perfect R-squared"),
        ({'known_ys': [2, 4, 6, 8, 10], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "Perfect linear relationship"),
        ({'known_ys': [1, 3, 2, 5, 4], 'known_xs': [1, 2, 3, 4, 5]}, Decimal('0.64'), "Moderate R-squared"),
        ({'known_ys': pl.Series([1, 2, 3, 4, 5]), 'known_xs': pl.Series([1, 2, 3, 4, 5])}, Decimal('1.0'), "RSQ with Polars Series"),
        ({'known_ys': "test_data.csv", 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "RSQ with CSV file input"),
        ({'known_ys': "test_data.parquet", 'known_xs': [1, 2, 3, 4, 5]}, Decimal('1.0'), "RSQ with Parquet file input"),
    ]
    all_passed &= test_function("RSQ", RSQ, rsq_tests, ctx=ctx)

    # Test RANK function
    rank_tests = [
        ({'number': 85, 'ref': [100, 85, 90, 75, 95], 'order': 0}, 4, "Rank in descending order"),
        ({'number': 85, 'ref': [100, 85, 90, 75, 95], 'order': 1}, 2, "Rank in ascending order"),
        ({'number': 100, 'ref': [100, 85, 90, 75, 95], 'order': 0}, 1, "Highest rank descending"),
        ({'number': 75, 'ref': [100, 85, 90, 75, 95], 'order': 1}, 1, "Lowest rank ascending"),
        ({'number': 3, 'ref': pl.Series([1, 2, 3, 4, 5]), 'order': 0}, 3, "RANK with Polars Series"),
        ({'number': 3, 'ref': "test_data.csv", 'order': 0}, 3, "RANK with CSV file input"),
        ({'number': 3, 'ref': "test_data.parquet", 'order': 0}, 3, "RANK with Parquet file input"),
    ]
    all_passed &= test_function("RANK", RANK, rank_tests, ctx=ctx)

    # Test PERCENTRANK function
    percentrank_tests = [
        ({'array': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'x': 7}, Decimal('0.650'), "Percentile rank of 7"),
        ({'array': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'x': 5}, Decimal('0.450'), "Percentile rank of 5"),
        ({'array': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'x': 10}, Decimal('0.950'), "Maximum percentile rank"),
        ({'array': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'x': 1}, Decimal('0.050'), "Minimum percentile rank"),
        ({'array': pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 'x': 7}, Decimal('0.650'), "PERCENTRANK with Polars Series"),
        ({'array': "test_data.csv", 'x': 3, 'significance': 3}, Decimal('0.500'), "PERCENTRANK with CSV file input"),
        ({'array': "test_data.parquet", 'x': 3, 'significance': 3}, Decimal('0.500'), "PERCENTRANK with Parquet file input"),
    ]
    all_passed &= test_function("PERCENTRANK", PERCENTRANK, percentrank_tests, ctx=ctx)

    # Test TREND function (returns file path)
    trend_tests = [
        ({'known_y': [1, 2, 3, 4, 5], 'known_x': [1, 2, 3, 4, 5], 'new_x': [6, 7, 8], 'output_filename': "trend_test1.parquet"}, Path(ctx.deps.analysis_dir / "trend_test1.parquet"), "Basic trend prediction"),
        ({'known_y': [2, 4, 6, 8, 10], 'new_x': [6, 7, 8], 'output_filename': "trend_test2.parquet"}, Path(ctx.deps.analysis_dir / "trend_test2.parquet"), "Trend with default x values"),
        ({'known_y': pl.Series([1, 2, 3, 4, 5]), 'known_x': pl.Series([1, 2, 3, 4, 5]), 'new_x': pl.Series([6, 7, 8]), 'output_filename': "trend_test3.parquet"}, Path(ctx.deps.analysis_dir / "trend_test3.parquet"), "TREND with Polars Series"),
        ({'known_y': "test_data.csv", 'known_x': [1, 2, 3, 4, 5], 'new_x': [6, 7, 8], 'output_filename': "trend_test4.parquet"}, Path(ctx.deps.analysis_dir / "trend_test4.parquet"), "TREND with CSV file input"),
        ({'known_y': "test_data.parquet", 'known_x': [1, 2, 3, 4, 5], 'new_x': [6, 7, 8], 'output_filename': "trend_test5.parquet"}, Path(ctx.deps.analysis_dir / "trend_test5.parquet"), "TREND with Parquet file input"),
    ]
    all_passed &= test_function("TREND", TREND, trend_tests, ctx=ctx)

    # Test GROWTH function (returns file path)
    growth_tests = [
        ({'known_y': [1, 2, 4, 8, 16], 'known_x': [1, 2, 3, 4, 5], 'new_x': [6, 7, 8], 'output_filename': "growth_test1.parquet"}, Path(ctx.deps.analysis_dir / "growth_test1.parquet"), "Exponential growth prediction"),
        ({'known_y': pl.Series([1, 2, 4, 8, 16]), 'known_x': pl.Series([1, 2, 3, 4, 5]), 'new_x': pl.Series([6, 7, 8]), 'output_filename': "growth_test3.parquet"}, Path(ctx.deps.analysis_dir / "growth_test3.parquet"), "GROWTH with Polars Series"),
    ]
    all_passed &= test_function("GROWTH", GROWTH, growth_tests, ctx=ctx)

    # Test LINEST function (returns file path)
    linest_tests = [
        ({'known_ys': [1, 2, 3, 4, 5], 'known_xs': [1, 2, 3, 4, 5], 'output_filename': "linest_test1.parquet"}, Path(ctx.deps.analysis_dir / "linest_test1.parquet"), "Basic linear regression stats"),
        ({'known_ys': [2, 4, 6, 8, 10], 'stats_flag': True, 'output_filename': "linest_test2.parquet"}, Path(ctx.deps.analysis_dir / "linest_test2.parquet"), "Linear regression with full stats"),
        ({'known_ys': pl.Series([1, 2, 3, 4, 5]), 'known_xs': pl.Series([1, 2, 3, 4, 5]), 'output_filename': "linest_test3.parquet"}, Path(ctx.deps.analysis_dir / "linest_test3.parquet"), "LINEST with Polars Series"),
        ({'known_ys': "test_data.csv", 'known_xs': [1, 2, 3, 4, 5], 'output_filename': "linest_test4.parquet"}, Path(ctx.deps.analysis_dir / "linest_test4.parquet"), "LINEST with CSV file input"),
        ({'known_ys': "test_data.parquet", 'known_xs': [1, 2, 3, 4, 5], 'output_filename': "linest_test5.parquet"}, Path(ctx.deps.analysis_dir / "linest_test5.parquet"), "LINEST with Parquet file input"),
    ]
    all_passed &= test_function("LINEST", LINEST, linest_tests, ctx=ctx)

    # Test LOGEST function (returns file path)
    logest_tests = [
        ({'known_ys': [1, 2, 4, 8, 16], 'known_xs': [1, 2, 3, 4, 5], 'output_filename': "logest_test1.parquet"}, Path(ctx.deps.analysis_dir / "logest_test1.parquet"), "Basic exponential regression stats"),
        ({'known_ys': pl.Series([1, 2, 4, 8, 16]), 'known_xs': pl.Series([1, 2, 3, 4, 5]), 'output_filename': "logest_test3.parquet"}, Path(ctx.deps.analysis_dir / "logest_test3.parquet"), "LOGEST with Polars Series"),
    ]
    all_passed &= test_function("LOGEST", LOGEST, logest_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    try:
        # Test empty input
        STDEV_P(ctx, [])
        print("✗ Error handling: Should have raised ValidationError for empty input")
        all_passed = False
    except ValidationError:
        print("✓ Error handling: Empty input correctly raises ValidationError")
    except Exception as e:
        print(f"✗ Error handling: Unexpected error type: {type(e).__name__}")
        all_passed = False

    try:
        # Test mismatched lengths for correlation
        CORREL(ctx, [1, 2, 3], range2=[1, 2])
        print("✗ Error handling: Should have raised ValidationError for mismatched lengths")
        all_passed = False
    except ValidationError:
        print("✓ Error handling: Mismatched lengths correctly raises ValidationError")
    except Exception as e:
        print(f"✗ Error handling: Unexpected error type: {type(e).__name__}")
        all_passed = False

    try:
        # Test negative values for GROWTH
        GROWTH(ctx, [-1, 2, 4], output_filename="error_test.parquet")
        print("✗ Error handling: Should have raised CalculationError for negative values in GROWTH")
        all_passed = False
    except CalculationError:
        print("✓ Error handling: Negative values in GROWTH correctly raises CalculationError")
    except Exception as e:
        print(f"✗ Error handling: Unexpected error type: {type(e).__name__}")
        all_passed = False

    # Summary
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Statistical and trend analysis functions are working correctly.")
        return True
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
        return False


def create_test_data():
    """Create test data files for testing file input functionality."""
    print("Creating test data files...")

    # Create test data
    test_data = pl.DataFrame({
        "values": [1, 2, 3, 4, 5]
    })

    # Save as CSV and Parquet
    test_data.write_csv("test_data.csv")
    test_data.write_parquet("test_data.parquet")

    print("Test data files created: test_data.csv, test_data.parquet")


if __name__ == "__main__":
    try:
        # Create test data files
        create_test_data()

        # Run all tests
        success = run_all_tests()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"Test execution failed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
