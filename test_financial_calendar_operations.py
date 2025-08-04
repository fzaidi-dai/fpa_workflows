#!/usr/bin/env python3
"""
Test script for financial calendar operations functions.
Tests all functions in the financial_calendar_operations.py module.
"""

import sys
import traceback
import datetime
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.financial_calendar_operations.financial_calendar_operations import (
    FISCAL_YEAR, FISCAL_QUARTER, BUSINESS_DAYS_BETWEEN, END_OF_PERIOD, PERIOD_OVERLAP,
    ValidationError, CalculationError, ConfigurationError
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

            # Handle comparison based on expected type
            if isinstance(expected, datetime.date):
                if result == expected:
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            elif isinstance(expected, (int, str)):
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
    """Run all tests for the financial calendar operations functions."""
    print("Starting comprehensive test of financial calendar operations functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    all_passed = True

    # Test FISCAL_YEAR function
    fiscal_year_tests = [
        # Standard fiscal year starting in April
        ({'date': datetime.date(2024, 3, 15), 'fiscal_year_start_month': 4}, 2023, "March 2024 in April fiscal year"),
        ({'date': datetime.date(2024, 5, 15), 'fiscal_year_start_month': 4}, 2024, "May 2024 in April fiscal year"),
        ({'date': datetime.date(2024, 4, 1), 'fiscal_year_start_month': 4}, 2024, "April 1 (fiscal year start)"),

        # US Federal fiscal year starting in October
        ({'date': datetime.date(2024, 9, 30), 'fiscal_year_start_month': 10}, 2023, "September 2024 in October fiscal year"),
        ({'date': datetime.date(2024, 10, 1), 'fiscal_year_start_month': 10}, 2024, "October 1 (fiscal year start)"),

        # Calendar year fiscal year
        ({'date': datetime.date(2024, 6, 15), 'fiscal_year_start_month': 1}, 2024, "June 2024 in calendar fiscal year"),
        ({'date': datetime.date(2024, 1, 1), 'fiscal_year_start_month': 1}, 2024, "January 1 (calendar year start)"),

        # String date inputs
        ({'date': '2024-03-15', 'fiscal_year_start_month': 4}, 2023, "String date input"),
        ({'date': '03/15/2024', 'fiscal_year_start_month': 4}, 2023, "US format string date"),

        # Edge cases
        ({'date': datetime.date(2024, 12, 31), 'fiscal_year_start_month': 1}, 2024, "Year end in calendar fiscal year"),
        ({'date': datetime.date(2024, 12, 31), 'fiscal_year_start_month': 4}, 2024, "Year end in April fiscal year"),
    ]
    all_passed &= test_function("FISCAL_YEAR", FISCAL_YEAR, fiscal_year_tests, ctx=ctx)

    # Test FISCAL_QUARTER function
    fiscal_quarter_tests = [
        # Fiscal year starting in April (Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar)
        ({'date': datetime.date(2024, 5, 15), 'fiscal_year_start_month': 4}, 'Q1', "May in April fiscal year"),
        ({'date': datetime.date(2024, 8, 15), 'fiscal_year_start_month': 4}, 'Q2', "August in April fiscal year"),
        ({'date': datetime.date(2024, 11, 15), 'fiscal_year_start_month': 4}, 'Q3', "November in April fiscal year"),
        ({'date': datetime.date(2024, 2, 15), 'fiscal_year_start_month': 4}, 'Q4', "February in April fiscal year"),

        # Calendar year fiscal year (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
        ({'date': datetime.date(2024, 3, 15), 'fiscal_year_start_month': 1}, 'Q1', "March in calendar fiscal year"),
        ({'date': datetime.date(2024, 6, 15), 'fiscal_year_start_month': 1}, 'Q2', "June in calendar fiscal year"),
        ({'date': datetime.date(2024, 9, 15), 'fiscal_year_start_month': 1}, 'Q3', "September in calendar fiscal year"),
        ({'date': datetime.date(2024, 12, 15), 'fiscal_year_start_month': 1}, 'Q4', "December in calendar fiscal year"),

        # US Federal fiscal year starting in October
        ({'date': datetime.date(2024, 10, 15), 'fiscal_year_start_month': 10}, 'Q1', "October in October fiscal year"),
        ({'date': datetime.date(2024, 1, 15), 'fiscal_year_start_month': 10}, 'Q2', "January in October fiscal year"),

        # String date inputs
        ({'date': '2024-03-15', 'fiscal_year_start_month': 4}, 'Q4', "String date input"),

        # Quarter boundaries
        ({'date': datetime.date(2024, 4, 1), 'fiscal_year_start_month': 4}, 'Q1', "First day of Q1"),
        ({'date': datetime.date(2024, 6, 30), 'fiscal_year_start_month': 4}, 'Q1', "Last day of Q1"),
        ({'date': datetime.date(2024, 7, 1), 'fiscal_year_start_month': 4}, 'Q2', "First day of Q2"),
    ]
    all_passed &= test_function("FISCAL_QUARTER", FISCAL_QUARTER, fiscal_quarter_tests, ctx=ctx)

    # Test BUSINESS_DAYS_BETWEEN function
    business_days_tests = [
        # Basic business days (no holidays)
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 1, 8)}, 5, "Week with weekend"),
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 1, 2)}, 1, "Single business day"),
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 1, 1)}, 0, "Same day (exclusive end)"),

        # With weekends
        ({'start_date': datetime.date(2024, 1, 5), 'end_date': datetime.date(2024, 1, 8)}, 1, "Friday to Monday"),
        ({'start_date': datetime.date(2024, 1, 6), 'end_date': datetime.date(2024, 1, 8)}, 0, "Saturday to Monday"),

        # With holidays
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 1, 8), 'holidays_list': [datetime.date(2024, 1, 1)]}, 4, "With New Year holiday"),
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 1, 8), 'holidays_list': ['2024-01-01', '2024-01-02']}, 3, "With multiple holidays"),

        # String date inputs
        ({'start_date': '2024-01-01', 'end_date': '2024-01-08'}, 5, "String date inputs"),
        ({'start_date': '01/01/2024', 'end_date': '01/08/2024'}, 5, "US format string dates"),

        # Longer periods
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 1, 31)}, 22, "Full month"),
        ({'start_date': datetime.date(2024, 1, 1), 'end_date': datetime.date(2024, 2, 1)}, 23, "Month boundary"),

        # Edge cases
        ({'start_date': datetime.date(2024, 2, 29), 'end_date': datetime.date(2024, 3, 1)}, 1, "Leap year boundary"),
    ]
    all_passed &= test_function("BUSINESS_DAYS_BETWEEN", BUSINESS_DAYS_BETWEEN, business_days_tests, ctx=ctx)

    # Test END_OF_PERIOD function
    end_of_period_tests = [
        # Month end
        ({'date': datetime.date(2024, 2, 15), 'period_type': 'month'}, datetime.date(2024, 2, 29), "February leap year month end"),
        ({'date': datetime.date(2023, 2, 15), 'period_type': 'month'}, datetime.date(2023, 2, 28), "February non-leap year month end"),
        ({'date': datetime.date(2024, 1, 15), 'period_type': 'month'}, datetime.date(2024, 1, 31), "January month end"),
        ({'date': datetime.date(2024, 4, 15), 'period_type': 'month'}, datetime.date(2024, 4, 30), "April month end"),
        ({'date': datetime.date(2024, 12, 31), 'period_type': 'month'}, datetime.date(2024, 12, 31), "Already at month end"),

        # Quarter end
        ({'date': datetime.date(2024, 2, 15), 'period_type': 'quarter'}, datetime.date(2024, 3, 31), "Q1 quarter end"),
        ({'date': datetime.date(2024, 5, 15), 'period_type': 'quarter'}, datetime.date(2024, 6, 30), "Q2 quarter end"),
        ({'date': datetime.date(2024, 8, 15), 'period_type': 'quarter'}, datetime.date(2024, 9, 30), "Q3 quarter end"),
        ({'date': datetime.date(2024, 11, 15), 'period_type': 'quarter'}, datetime.date(2024, 12, 31), "Q4 quarter end"),
        ({'date': datetime.date(2024, 3, 31), 'period_type': 'quarter'}, datetime.date(2024, 3, 31), "Already at quarter end"),

        # Year end
        ({'date': datetime.date(2024, 6, 15), 'period_type': 'year'}, datetime.date(2024, 12, 31), "Mid-year to year end"),
        ({'date': datetime.date(2024, 1, 1), 'period_type': 'year'}, datetime.date(2024, 12, 31), "Year start to year end"),
        ({'date': datetime.date(2024, 12, 31), 'period_type': 'year'}, datetime.date(2024, 12, 31), "Already at year end"),

        # String date inputs
        ({'date': '2024-03-15', 'period_type': 'quarter'}, datetime.date(2024, 3, 31), "String date input"),
        ({'date': '03/15/2024', 'period_type': 'month'}, datetime.date(2024, 3, 31), "US format string date"),

        # Edge cases
        ({'date': datetime.date(2024, 2, 29), 'period_type': 'month'}, datetime.date(2024, 2, 29), "Leap day month end"),
    ]
    all_passed &= test_function("END_OF_PERIOD", END_OF_PERIOD, end_of_period_tests, ctx=ctx)

    # Test PERIOD_OVERLAP function
    period_overlap_tests = [
        # Basic overlaps
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 6, 30), 'start2': datetime.date(2024, 4, 1), 'end2': datetime.date(2024, 9, 30)}, 91, "Partial overlap"),
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 3, 31), 'start2': datetime.date(2024, 2, 1), 'end2': datetime.date(2024, 4, 30)}, 60, "Revenue recognition overlap"),

        # No overlap
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 1, 31), 'start2': datetime.date(2024, 2, 1), 'end2': datetime.date(2024, 2, 29)}, 0, "No overlap"),
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 1, 30), 'start2': datetime.date(2024, 2, 1), 'end2': datetime.date(2024, 2, 29)}, 0, "Gap between periods"),

        # Adjacent periods (touching)
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 1, 31), 'start2': datetime.date(2024, 1, 31), 'end2': datetime.date(2024, 2, 29)}, 1, "Adjacent periods touching"),

        # Complete overlap
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 12, 31), 'start2': datetime.date(2024, 6, 1), 'end2': datetime.date(2024, 8, 31)}, 92, "Complete containment"),
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 3, 31), 'start2': datetime.date(2024, 1, 1), 'end2': datetime.date(2024, 3, 31)}, 91, "Identical periods"),

        # String date inputs
        ({'start1': '2024-01-01', 'end1': '2024-06-30', 'start2': '2024-04-01', 'end2': '2024-09-30'}, 91, "String date inputs"),
        ({'start1': '01/01/2024', 'end1': '06/30/2024', 'start2': '04/01/2024', 'end2': '09/30/2024'}, 91, "US format string dates"),

        # Edge cases
        ({'start1': datetime.date(2024, 2, 28), 'end1': datetime.date(2024, 3, 1), 'start2': datetime.date(2024, 2, 29), 'end2': datetime.date(2024, 3, 2)}, 2, "Leap year overlap"),
        ({'start1': datetime.date(2024, 1, 1), 'end1': datetime.date(2024, 1, 1), 'start2': datetime.date(2024, 1, 1), 'end2': datetime.date(2024, 1, 1)}, 1, "Single day overlap"),

        # Reverse order (period 2 starts before period 1)
        ({'start1': datetime.date(2024, 4, 1), 'end1': datetime.date(2024, 9, 30), 'start2': datetime.date(2024, 1, 1), 'end2': datetime.date(2024, 6, 30)}, 91, "Reverse order periods"),
    ]
    all_passed &= test_function("PERIOD_OVERLAP", PERIOD_OVERLAP, period_overlap_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test invalid fiscal year start month
    try:
        FISCAL_YEAR(ctx, datetime.date(2024, 1, 1), fiscal_year_start_month=13)
        print("✗ Invalid fiscal year start month validation failed")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Invalid fiscal year start month validation passed")
        error_tests_passed += 1

    # Test invalid date format
    try:
        FISCAL_YEAR(ctx, "invalid-date", fiscal_year_start_month=1)
        print("✗ Invalid date format validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid date format validation passed")
        error_tests_passed += 1

    # Test invalid period type
    try:
        END_OF_PERIOD(ctx, datetime.date(2024, 1, 1), period_type='invalid')
        print("✗ Invalid period type validation failed")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Invalid period type validation passed")
        error_tests_passed += 1

    # Test invalid date range for business days
    try:
        BUSINESS_DAYS_BETWEEN(ctx, datetime.date(2024, 1, 8), datetime.date(2024, 1, 1))
        print("✗ Invalid date range validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid date range validation passed")
        error_tests_passed += 1

    # Test invalid date range for period overlap
    try:
        PERIOD_OVERLAP(ctx, datetime.date(2024, 1, 8), datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), datetime.date(2024, 2, 28))
        print("✗ Invalid period range validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid period range validation passed")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All financial calendar functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the failed tests above.")
    print("="*50)

    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
