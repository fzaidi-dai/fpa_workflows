#!/usr/bin/env python3
"""
Test script for date and time functions.
Tests all functions in the date_and_time_functions.py module.
"""

import sys
import traceback
from decimal import Decimal
import datetime
from pathlib import Path
import polars as pl

# Import the functions to test
from tools.core_data_and_math_utils.date_and_time_functions.date_and_time_functions import (
    TODAY, NOW, DATE, YEAR, MONTH, DAY, EDATE, EOMONTH, DATEDIF, YEARFRAC,
    WORKDAY, NETWORKDAYS, DATE_RANGE, WEEKDAY, QUARTER, TIME, HOUR, MINUTE, SECOND,
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
            if isinstance(expected, Decimal):
                if isinstance(result, Decimal) and abs(result - expected) < Decimal('1e-10'):
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
                        if all(r == e for r, e in zip(result, expected)):
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
                # Handle Path comparisons for file outputs
                if isinstance(result, Path) and result.exists():
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected file path: {expected}, Got: {result}")
                    failed += 1
            else:
                # Handle other types (datetime, int, etc.)
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


def create_test_data(ctx):
    """Create test data files for date/time testing."""
    # Create test date data
    test_dates = ["2025-01-01", "2025-02-15", "2025-03-31", "2025-06-30", "2025-12-31"]
    date_df = pl.DataFrame({"dates": test_dates})

    # Save as CSV and Parquet
    date_df.write_csv(ctx.deps.data_dir / "test_dates.csv")
    date_df.write_parquet(ctx.deps.data_dir / "test_dates.parquet")

    # Create test time data
    test_times = ["09:30:00", "14:15:30", "18:45:15", "23:59:59", "00:00:00"]
    time_df = pl.DataFrame({"times": test_times})

    # Save as CSV and Parquet
    time_df.write_csv(ctx.deps.data_dir / "test_times.csv")
    time_df.write_parquet(ctx.deps.data_dir / "test_times.parquet")


def run_all_tests():
    """Run all tests for the date and time functions."""
    print("Starting comprehensive test of date and time functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    # Create test data
    create_test_data(ctx)

    all_passed = True

    # Test TODAY function
    today_tests = [
        ({}, datetime.date.today(), "Get current date"),
    ]
    all_passed &= test_function("TODAY", TODAY, today_tests, ctx=ctx)

    # Test NOW function
    now_tests = [
        ({}, datetime.datetime.now().replace(microsecond=0), "Get current datetime (ignoring microseconds)"),
    ]
    # Special handling for NOW since exact time comparison is difficult
    print(f"\n=== Testing NOW ===")
    try:
        result = NOW(ctx)
        if isinstance(result, datetime.datetime):
            print("✓ Test 1: Get current datetime")
            print(f"Results for NOW: 1 passed, 0 failed")
        else:
            print("✗ Test 1: Get current datetime")
            print(f"Results for NOW: 0 passed, 1 failed")
            all_passed = False
    except Exception as e:
        print("✗ Test 1: Get current datetime")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        print(f"Results for NOW: 0 passed, 1 failed")
        all_passed = False

    # Test DATE function
    date_tests = [
        ({'year': 2025, 'month': 4, 'day': 15}, datetime.date(2025, 4, 15), "Construct date from components"),
        ({'year': 2024, 'month': 2, 'day': 29}, datetime.date(2024, 2, 29), "Leap year date"),
        ({'year': 2025, 'month': 12, 'day': 31}, datetime.date(2025, 12, 31), "Year end date"),
    ]
    all_passed &= test_function("DATE", DATE, date_tests, ctx=ctx)

    # Test YEAR function
    year_tests = [
        ({'date': datetime.date(2025, 4, 15)}, 2025, "Extract year from date object"),
        ({'date': "2025-04-15"}, 2025, "Extract year from string"),
        ({'date': "test_dates.csv"}, 2025, "Extract year from CSV file"),
        ({'date': "test_dates.parquet"}, 2025, "Extract year from Parquet file"),
    ]
    all_passed &= test_function("YEAR", YEAR, year_tests, ctx=ctx)

    # Test MONTH function
    month_tests = [
        ({'date': datetime.date(2025, 4, 15)}, 4, "Extract month from date object"),
        ({'date': "2025-04-15"}, 4, "Extract month from string"),
        ({'date': "test_dates.csv"}, 1, "Extract month from CSV file"),
        ({'date': "test_dates.parquet"}, 1, "Extract month from Parquet file"),
    ]
    all_passed &= test_function("MONTH", MONTH, month_tests, ctx=ctx)

    # Test DAY function
    day_tests = [
        ({'date': datetime.date(2025, 4, 15)}, 15, "Extract day from date object"),
        ({'date': "2025-04-15"}, 15, "Extract day from string"),
        ({'date': "test_dates.csv"}, 1, "Extract day from CSV file"),
        ({'date': "test_dates.parquet"}, 1, "Extract day from Parquet file"),
    ]
    all_passed &= test_function("DAY", DAY, day_tests, ctx=ctx)

    # Test EDATE function
    edate_tests = [
        ({'start_date': datetime.date(2025, 1, 15), 'months': 3}, datetime.date(2025, 4, 15), "Add 3 months"),
        ({'start_date': "2025-01-15", 'months': -2}, datetime.date(2024, 11, 15), "Subtract 2 months"),
        ({'start_date': datetime.date(2025, 1, 31), 'months': 1}, datetime.date(2025, 2, 28), "Month end adjustment"),
        ({'start_date': "test_dates.csv", 'months': 6}, datetime.date(2025, 7, 1), "EDATE with CSV file"),
    ]
    all_passed &= test_function("EDATE", EDATE, edate_tests, ctx=ctx)

    # Test EOMONTH function
    eomonth_tests = [
        ({'start_date': datetime.date(2025, 1, 15), 'months': 0}, datetime.date(2025, 1, 31), "End of current month"),
        ({'start_date': "2025-01-15", 'months': 2}, datetime.date(2025, 3, 31), "End of month +2"),
        ({'start_date': datetime.date(2025, 1, 15), 'months': 1}, datetime.date(2025, 2, 28), "End of February"),
        ({'start_date': "test_dates.csv", 'months': 0}, datetime.date(2025, 1, 31), "EOMONTH with CSV file"),
    ]
    all_passed &= test_function("EOMONTH", EOMONTH, eomonth_tests, ctx=ctx)

    # Test DATEDIF function
    datedif_tests = [
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2025, 1, 1), 'unit': "Y"}, 1, "Year difference"),
        ({'start_date': "2024-01-01", 'end_date': "2024-04-01", 'unit': "M"}, 3, "Month difference"),
        ({'start_date': datetime.date(2025, 1, 1), 'end_date': datetime.date(2025, 1, 10), 'unit': "D"}, 9, "Day difference"),
        ({'start_date': datetime.date(2024, 6, 15), 'end_date': datetime.date(2025, 6, 15), 'unit': "Y"}, 1, "Exact year difference"),
    ]
    all_passed &= test_function("DATEDIF", DATEDIF, datedif_tests, ctx=ctx)

    # Test YEARFRAC function
    yearfrac_tests = [
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 7, 1), 'basis': 1}, Decimal('0.4972677595628415'), "Actual/Actual basis"),
        ({'start_date': "2024-01-01", 'end_date': "2024-12-31", 'basis': 0}, Decimal('1.0'), "30/360 US basis"),
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 7, 1), 'basis': 2}, Decimal('0.5055555555555555555555555556'), "Actual/360 basis"),
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 7, 1), 'basis': 3}, Decimal('0.4986301369863014'), "Actual/365 basis"),
    ]
    all_passed &= test_function("YEARFRAC", YEARFRAC, yearfrac_tests, ctx=ctx)

    # Test WORKDAY function
    workday_tests = [
        ({'start_date': datetime.date(2025, 1, 1), 'days': 5}, datetime.date(2025, 1, 8), "Add 5 workdays from Wednesday"),
        ({'start_date': "2025-01-03", 'days': 3}, datetime.date(2025, 1, 8), "Add 3 workdays from Friday"),
        ({'start_date': datetime.date(2025, 1, 10), 'days': -5}, datetime.date(2025, 1, 3), "Subtract 5 workdays"),
        ({'start_date': datetime.date(2025, 1, 1), 'days': 1, 'holidays': [datetime.date(2025, 1, 2)]}, datetime.date(2025, 1, 3), "Workday with holiday"),
    ]
    all_passed &= test_function("WORKDAY", WORKDAY, workday_tests, ctx=ctx)

    # Test NETWORKDAYS function
    networkdays_tests = [
        ({'start_date': datetime.date(2025, 1, 1), 'end_date': datetime.date(2025, 1, 10)}, 8, "Count workdays in range"),
        ({'start_date': "2025-01-06", 'end_date': "2025-01-10"}, 5, "Monday to Friday"),
        ({'start_date': datetime.date(2025, 1, 1), 'end_date': datetime.date(2025, 1, 10), 'holidays': [datetime.date(2025, 1, 2)]}, 7, "Workdays with holiday"),
        ({'start_date': datetime.date(2025, 1, 4), 'end_date': datetime.date(2025, 1, 5)}, 0, "Weekend only"),
    ]
    all_passed &= test_function("NETWORKDAYS", NETWORKDAYS, networkdays_tests, ctx=ctx)

    # Test DATE_RANGE function
    date_range_tests = [
        ({'start_date': "2025-01-01", 'end_date': "2025-01-05", 'frequency': "D"},
         [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2), datetime.date(2025, 1, 3), datetime.date(2025, 1, 4), datetime.date(2025, 1, 5)],
         "Daily frequency"),
        ({'start_date': datetime.date(2025, 1, 1), 'end_date': datetime.date(2025, 3, 31), 'frequency': "M"},
         [datetime.date(2025, 1, 31), datetime.date(2025, 2, 28), datetime.date(2025, 3, 31)],
         "Monthly frequency"),
        ({'start_date': "2025-01-01", 'end_date': "2025-12-31", 'frequency': "Q"},
         [datetime.date(2025, 3, 31), datetime.date(2025, 6, 30), datetime.date(2025, 9, 30), datetime.date(2025, 12, 31)],
         "Quarterly frequency"),
        ({'start_date': datetime.date(2025, 1, 1), 'end_date': datetime.date(2027, 12, 31), 'frequency': "Y"},
         [datetime.date(2025, 12, 31), datetime.date(2026, 12, 31), datetime.date(2027, 12, 31)],
         "Yearly frequency"),
        ({'start_date': "2025-01-01", 'end_date': "2025-01-05", 'frequency': "D", 'output_filename': "date_range_results.parquet"},
         Path("/Users/farhan/work/fpa_agents/scratch_pad/analysis/date_range_results.parquet"),
         "Date range with file output"),
    ]
    all_passed &= test_function("DATE_RANGE", DATE_RANGE, date_range_tests, ctx=ctx)

    # Test WEEKDAY function
    weekday_tests = [
        ({'serial_number': datetime.date(2025, 1, 8)}, 4, "Wednesday with default return type"),  # Wednesday
        ({'serial_number': "2025-01-05", 'return_type': 1}, 1, "Sunday with return type 1"),  # Sunday
        ({'serial_number': datetime.date(2025, 1, 6), 'return_type': 2}, 1, "Monday with return type 2"),  # Monday
        ({'serial_number': datetime.date(2025, 1, 6), 'return_type': 3}, 0, "Monday with return type 3"),  # Monday
        ({'serial_number': "test_dates.csv"}, 4, "Weekday from CSV file"),
    ]
    all_passed &= test_function("WEEKDAY", WEEKDAY, weekday_tests, ctx=ctx)

    # Test QUARTER function
    quarter_tests = [
        ({'date': datetime.date(2024, 7, 15)}, 3, "Q3 date"),
        ({'date': "2025-01-15"}, 1, "Q1 date"),
        ({'date': datetime.date(2025, 12, 31)}, 4, "Q4 date"),
        ({'date': "test_dates.csv"}, 1, "Quarter from CSV file"),
    ]
    all_passed &= test_function("QUARTER", QUARTER, quarter_tests, ctx=ctx)

    # Test TIME function
    time_tests = [
        ({'hour': 14, 'minute': 30, 'second': 0}, datetime.time(14, 30, 0), "Create afternoon time"),
        ({'hour': 0, 'minute': 0, 'second': 0}, datetime.time(0, 0, 0), "Create midnight time"),
        ({'hour': 23, 'minute': 59, 'second': 59}, datetime.time(23, 59, 59), "Create end of day time"),
    ]
    all_passed &= test_function("TIME", TIME, time_tests, ctx=ctx)

    # Test HOUR function
    hour_tests = [
        ({'serial_number': datetime.time(14, 30, 0)}, 14, "Extract hour from time object"),
        ({'serial_number': datetime.datetime(2025, 1, 1, 14, 30, 0)}, 14, "Extract hour from datetime object"),
        ({'serial_number': "14:30:00"}, 14, "Extract hour from string"),
        ({'serial_number': "test_times.csv"}, 9, "Extract hour from CSV file"),
    ]
    all_passed &= test_function("HOUR", HOUR, hour_tests, ctx=ctx)

    # Test MINUTE function
    minute_tests = [
        ({'serial_number': datetime.time(14, 30, 45)}, 30, "Extract minute from time object"),
        ({'serial_number': datetime.datetime(2025, 1, 1, 14, 30, 45)}, 30, "Extract minute from datetime object"),
        ({'serial_number': "14:30:45"}, 30, "Extract minute from string"),
        ({'serial_number': "test_times.csv"}, 30, "Extract minute from CSV file"),
    ]
    all_passed &= test_function("MINUTE", MINUTE, minute_tests, ctx=ctx)

    # Test SECOND function
    second_tests = [
        ({'serial_number': datetime.time(14, 30, 45)}, 45, "Extract second from time object"),
        ({'serial_number': datetime.datetime(2025, 1, 1, 14, 30, 45)}, 45, "Extract second from datetime object"),
        ({'serial_number': "14:30:45"}, 45, "Extract second from string"),
        ({'serial_number': "test_times.csv"}, 0, "Extract second from CSV file"),
    ]
    all_passed &= test_function("SECOND", SECOND, second_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test invalid date construction
    try:
        DATE(ctx, 2025, 2, 30)  # Invalid date
        print("✗ Invalid date validation failed for DATE")
        error_tests_failed += 1
    except CalculationError:
        print("✓ Invalid date validation passed for DATE")
        error_tests_passed += 1

    # Test invalid date string
    try:
        YEAR(ctx, "invalid-date")
        print("✗ Invalid date string validation failed for YEAR")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Invalid date string validation passed for YEAR")
        error_tests_passed += 1

    # Test invalid DATEDIF unit
    try:
        DATEDIF(ctx, datetime.date(2024, 1, 1), datetime.date(2025, 1, 1), "X")
        print("✗ Invalid unit validation failed for DATEDIF")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid unit validation passed for DATEDIF")
        error_tests_passed += 1

    # Test invalid YEARFRAC basis
    try:
        YEARFRAC(ctx, datetime.date(2024, 1, 1), datetime.date(2025, 1, 1), 5)
        print("✗ Invalid basis validation failed for YEARFRAC")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid basis validation passed for YEARFRAC")
        error_tests_passed += 1

    # Test invalid time construction
    try:
        TIME(ctx, 25, 30, 0)  # Invalid hour
        print("✗ Invalid time validation failed for TIME")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid time validation passed for TIME")
        error_tests_passed += 1

    # Test invalid frequency for DATE_RANGE
    try:
        DATE_RANGE(ctx, "2025-01-01", "2025-01-31", "X")
        print("✗ Invalid frequency validation failed for DATE_RANGE")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid frequency validation passed for DATE_RANGE")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All date and time functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the failed tests above.")
    print("="*50)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
