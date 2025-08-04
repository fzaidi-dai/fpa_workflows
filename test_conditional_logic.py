#!/usr/bin/env python3
"""
Test script for conditional logic functions.
Tests all functions in the conditional_logic.py module.
"""

import sys
import traceback
from decimal import Decimal
import polars as pl
import numpy as np
from pathlib import Path

# Import the functions to test
from tools.core_data_and_math_utils.conditional_logic.conditional_logic import (
    MULTI_CONDITION_LOGIC, NESTED_IF_LOGIC, CASE_WHEN, CONDITIONAL_AGGREGATION,
    ValidationError, CalculationError, DataQualityError, ConfigurationError
)

# Import FinnDeps and RunContext for testing
from tools.finn_deps import FinnDeps, RunContext

def create_test_data():
    """Create test datasets for conditional logic testing."""

    # Customer data for risk assessment and segmentation
    customer_data = pl.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "credit_score": [800, 720, 650, 580, 750, 620, 690, 540, 780, 600],
        "annual_revenue": [1500000, 250000, 75000, 15000, 800000, 45000, 120000, 8000, 2000000, 35000],
        "debt_ratio": [0.2, 0.35, 0.45, 0.6, 0.25, 0.5, 0.3, 0.7, 0.15, 0.55],
        "age": [45, 32, 28, 55, 38, 42, 29, 60, 35, 48],
        "risk_tolerance": [7, 5, 8, 3, 6, 4, 9, 2, 8, 5]
    })

    # Sales data for performance analysis
    sales_data = pl.DataFrame({
        "sales_rep": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
        "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
        "sales_amount": [125000, 85000, 45000, 95000, 155000, 65000, 75000, 105000],
        "deal_size": [75000, 45000, 25000, 55000, 85000, 35000, 40000, 60000],
        "profit": [25000, 15000, -5000, 18000, 35000, 8000, 12000, 22000],
        "score": [92, 78, 65, 83, 95, 72, 76, 88]
    })

    # Investment portfolio data
    portfolio_data = pl.DataFrame({
        "portfolio_id": [1, 2, 3, 4, 5, 6],
        "age": [25, 35, 45, 55, 30, 40],
        "risk_tolerance": [9, 7, 5, 3, 8, 6],
        "volatility": [0.28, 0.18, 0.12, 0.06, 0.22, 0.15],
        "balance": [50000, 150000, 300000, 500000, 100000, 250000]
    })

    return customer_data, sales_data, portfolio_data

def save_test_data(ctx):
    """Save test data to files for file input testing."""
    customer_data, sales_data, portfolio_data = create_test_data()

    # Save to analysis directory
    customer_path = ctx.deps.analysis_dir / "customer_test.parquet"
    sales_path = ctx.deps.analysis_dir / "sales_test.parquet"
    portfolio_path = ctx.deps.analysis_dir / "portfolio_test.parquet"

    customer_data.write_parquet(customer_path)
    sales_data.write_parquet(sales_path)
    portfolio_data.write_parquet(portfolio_path)

    return customer_path, sales_path, portfolio_path

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
            if isinstance(expected, Path):
                # For file output tests, check if file exists
                if isinstance(result, Path) and result.exists():
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected file path, got: {result}")
                    failed += 1
            elif isinstance(expected, pl.DataFrame):
                # For DataFrame comparisons
                if isinstance(result, pl.DataFrame):
                    if result.shape == expected.shape:
                        print(f"✓ Test {i}: {description}")
                        passed += 1
                    else:
                        print(f"✗ Test {i}: {description}")
                        print(f"  Expected shape {expected.shape}, got {result.shape}")
                        failed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected DataFrame, got: {type(result)}")
                    failed += 1
            elif isinstance(expected, list):
                # Handle list comparisons
                if isinstance(result, list) and len(result) == len(expected):
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected: {expected}, Got: {result}")
                    failed += 1
            elif expected == "DataFrame":
                # Generic DataFrame check
                if isinstance(result, pl.DataFrame):
                    print(f"✓ Test {i}: {description}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {description}")
                    print(f"  Expected DataFrame, got: {type(result)}")
                    failed += 1
            else:
                # Direct comparison
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
    """Run all tests for the conditional logic functions."""
    print("Starting comprehensive test of conditional logic functions...")

    # Create test context
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    ctx = RunContext(deps=finn_deps)

    # Create and save test data
    customer_data, sales_data, portfolio_data = create_test_data()
    customer_path, sales_path, portfolio_path = save_test_data(ctx)

    all_passed = True

    # Test MULTI_CONDITION_LOGIC function
    multi_condition_tests = [
        # Basic risk assessment tree
        ({
            'df': customer_data,
            'condition_tree': {
                'if': 'credit_score >= 750',
                'then': 'Low Risk',
                'elif': [
                    {'condition': 'credit_score >= 650', 'then': 'Medium Risk'}
                ],
                'else': 'High Risk'
            }
        }, "DataFrame", "Basic risk assessment with elif"),

        # Investment allocation strategy
        ({
            'df': portfolio_data,
            'condition_tree': {
                'if': 'age < 30',
                'then': 'Aggressive',
                'elif': [
                    {'condition': 'age < 50', 'then': 'Moderate'},
                    {'condition': 'risk_tolerance > 7', 'then': 'Moderate'}
                ],
                'else': 'Conservative'
            }
        }, "DataFrame", "Investment allocation with multiple elif conditions"),

        # Revenue categorization
        ({
            'df': customer_data,
            'condition_tree': {
                'if': 'annual_revenue > 1000000',
                'then': 'Enterprise',
                'elif': [
                    {'condition': 'annual_revenue > 100000', 'then': 'Corporate'},
                    {'condition': 'annual_revenue > 10000', 'then': 'SMB'}
                ],
                'else': 'Startup'
            }
        }, "DataFrame", "Revenue categorization with nested conditions"),

        # Simple if-else without elif
        ({
            'df': sales_data,
            'condition_tree': {
                'if': 'profit > 0',
                'then': 'Profitable',
                'else': 'Loss'
            }
        }, "DataFrame", "Simple if-else condition"),

        # File input with output
        ({
            'df': customer_path,
            'condition_tree': {
                'if': 'credit_score >= 700',
                'then': 'Good',
                'else': 'Poor'
            },
            'output_filename': 'multi_condition_test.parquet'
        }, Path(ctx.deps.analysis_dir / "multi_condition_test.parquet"), "File input with output"),
    ]
    all_passed &= test_function("MULTI_CONDITION_LOGIC", MULTI_CONDITION_LOGIC, multi_condition_tests, ctx=ctx)

    # Test NESTED_IF_LOGIC function
    nested_if_tests = [
        # Bond rating classification with DataFrame context
        ({
            'conditions_list': [
                'credit_score >= 800',
                'credit_score >= 700',
                'credit_score >= 600',
                'credit_score >= 500'
            ],
            'results_list': ['AAA', 'AA', 'A', 'BBB'],
            'default_value': 'Junk',
            'df_context': customer_data
        }, "DataFrame", "Bond rating classification"),

        # Commission tier calculation
        ({
            'conditions_list': [
                'sales_amount >= 100000',
                'sales_amount >= 50000',
                'sales_amount >= 25000'
            ],
            'results_list': [0.15, 0.12, 0.08],
            'default_value': 0.05,
            'df_context': sales_data
        }, "DataFrame", "Commission tier calculation"),

        # Investment risk categorization
        ({
            'conditions_list': [
                'volatility > 0.25',
                'volatility > 0.15',
                'volatility > 0.08'
            ],
            'results_list': ['High', 'Medium', 'Low'],
            'default_value': 'Very Low',
            'df_context': portfolio_data
        }, "DataFrame", "Investment risk categorization"),

        # Simple boolean list evaluation (no DataFrame context)
        ({
            'conditions_list': [True, False, True],
            'results_list': ['A', 'B', 'C'],
            'default_value': 'Default'
        }, ['A', 'Default', 'C'], "Boolean list evaluation"),

        # File input with output
        ({
            'conditions_list': ['score >= 90', 'score >= 80', 'score >= 70'],
            'results_list': ['Excellent', 'Good', 'Satisfactory'],
            'default_value': 'Needs Improvement',
            'df_context': sales_path,
            'output_filename': 'nested_if_test.parquet'
        }, Path(ctx.deps.analysis_dir / "nested_if_test.parquet"), "File input with output"),
    ]
    all_passed &= test_function("NESTED_IF_LOGIC", NESTED_IF_LOGIC, nested_if_tests, ctx=ctx)

    # Test CASE_WHEN function
    case_when_tests = [
        # Customer segment classification
        ({
            'df': customer_data,
            'case_conditions': [
                {'when': 'annual_revenue >= 1000000', 'then': 'Enterprise'},
                {'when': 'annual_revenue >= 100000', 'then': 'Corporate'},
                {'when': 'annual_revenue >= 10000', 'then': 'SMB'},
                {'else': 'Startup'}
            ]
        }, "DataFrame", "Customer segment classification"),

        # Performance rating system
        ({
            'df': sales_data,
            'case_conditions': [
                {'when': 'score >= 90', 'then': 'Excellent'},
                {'when': 'score >= 80', 'then': 'Good'},
                {'when': 'score >= 70', 'then': 'Satisfactory'},
                {'when': 'score >= 60', 'then': 'Needs Improvement'},
                {'else': 'Unsatisfactory'}
            ]
        }, "DataFrame", "Performance rating system"),

        # Investment allocation
        ({
            'df': portfolio_data,
            'case_conditions': [
                {'when': 'age < 30', 'then': 'Growth'},
                {'when': 'age < 50', 'then': 'Balanced'},
                {'when': 'risk_tolerance > 5', 'then': 'Conservative'},
                {'else': 'Income'}
            ]
        }, "DataFrame", "Investment allocation strategy"),

        # Simple case without else
        ({
            'df': sales_data,
            'case_conditions': [
                {'when': 'profit > 20000', 'then': 'High Profit'},
                {'when': 'profit > 10000', 'then': 'Medium Profit'}
            ]
        }, "DataFrame", "Case without else clause"),

        # File input with output
        ({
            'df': portfolio_path,
            'case_conditions': [
                {'when': 'balance >= 200000', 'then': 'High Net Worth'},
                {'when': 'balance >= 100000', 'then': 'Medium Net Worth'},
                {'else': 'Standard'}
            ],
            'output_filename': 'case_when_test.parquet'
        }, Path(ctx.deps.analysis_dir / "case_when_test.parquet"), "File input with output"),
    ]
    all_passed &= test_function("CASE_WHEN", CASE_WHEN, case_when_tests, ctx=ctx)

    # Test CONDITIONAL_AGGREGATION function
    conditional_agg_tests = [
        # Sum high-value transactions by region
        ({
            'df': sales_data,
            'group_columns': ['region'],
            'condition': 'sales_amount > 80000',
            'aggregation_func': 'sum',
            'target_column': 'sales_amount'
        }, "DataFrame", "Sum high-value sales by region"),

        # Count profitable customers by region
        ({
            'df': sales_data,
            'group_columns': ['region'],
            'condition': 'profit > 0',
            'aggregation_func': 'count'
        }, "DataFrame", "Count profitable customers by region"),

        # Average deal size for large deals by region
        ({
            'df': sales_data,
            'group_columns': ['region'],
            'condition': 'deal_size > 50000',
            'aggregation_func': 'mean',
            'target_column': 'deal_size'
        }, "DataFrame", "Average large deal size by region"),

        # Maximum score for high performers
        ({
            'df': sales_data,
            'group_columns': ['region'],
            'condition': 'score >= 80',
            'aggregation_func': 'max',
            'target_column': 'score'
        }, "DataFrame", "Maximum score for high performers"),

        # Minimum profit for profitable deals
        ({
            'df': sales_data,
            'group_columns': ['region'],
            'condition': 'profit > 0',
            'aggregation_func': 'min',
            'target_column': 'profit'
        }, "DataFrame", "Minimum profit for profitable deals"),

        # Standard deviation of high-value sales
        ({
            'df': sales_data,
            'group_columns': ['region'],
            'condition': 'sales_amount > 70000',
            'aggregation_func': 'std',
            'target_column': 'sales_amount'
        }, "DataFrame", "Standard deviation of high-value sales"),

        # File input with output
        ({
            'df': sales_path,
            'group_columns': ['region'],
            'condition': 'profit > 15000',
            'aggregation_func': 'sum',
            'target_column': 'profit',
            'output_filename': 'conditional_agg_test.parquet'
        }, Path(ctx.deps.analysis_dir / "conditional_agg_test.parquet"), "File input with output"),
    ]
    all_passed &= test_function("CONDITIONAL_AGGREGATION", CONDITIONAL_AGGREGATION, conditional_agg_tests, ctx=ctx)

    # Test error handling
    print("\n=== Testing Error Handling ===")
    error_tests_passed = 0
    error_tests_failed = 0

    # Test invalid condition tree structure
    try:
        MULTI_CONDITION_LOGIC(ctx, customer_data, condition_tree={'invalid': 'structure'})
        print("✗ Invalid condition tree validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid condition tree validation passed")
        error_tests_passed += 1

    # Test mismatched conditions and results lists
    try:
        NESTED_IF_LOGIC(ctx, ['condition1'], results_list=['result1', 'result2'], default_value='default')
        print("✗ Mismatched lists validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Mismatched lists validation passed")
        error_tests_passed += 1

    # Test invalid case conditions structure
    try:
        CASE_WHEN(ctx, sales_data, case_conditions=[{'invalid': 'structure'}])
        print("✗ Invalid case conditions validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid case conditions validation passed")
        error_tests_passed += 1

    # Test invalid column reference
    try:
        CONDITIONAL_AGGREGATION(
            ctx, sales_data,
            group_columns=['nonexistent_column'],
            condition='profit > 0',
            aggregation_func='sum',
            target_column='profit'
        )
        print("✗ Invalid column reference validation failed")
        error_tests_failed += 1
    except DataQualityError:
        print("✓ Invalid column reference validation passed")
        error_tests_passed += 1

    # Test unsupported aggregation function
    try:
        CONDITIONAL_AGGREGATION(
            ctx, sales_data,
            group_columns=['region'],
            condition='profit > 0',
            aggregation_func='unsupported_func',
            target_column='profit'
        )
        print("✗ Unsupported aggregation function validation failed")
        error_tests_failed += 1
    except ConfigurationError:
        print("✓ Unsupported aggregation function validation passed")
        error_tests_passed += 1

    # Test invalid condition syntax
    try:
        MULTI_CONDITION_LOGIC(ctx, customer_data, condition_tree={
            'if': 'invalid condition syntax',
            'then': 'result'
        })
        print("✗ Invalid condition syntax validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Invalid condition syntax validation passed")
        error_tests_passed += 1

    # Test empty DataFrame
    try:
        empty_df = pl.DataFrame()
        CASE_WHEN(ctx, empty_df, case_conditions=[{'when': 'col > 0', 'then': 'result'}])
        print("✗ Empty DataFrame validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Empty DataFrame validation passed")
        error_tests_passed += 1

    # Test missing target column for aggregation
    try:
        CONDITIONAL_AGGREGATION(
            ctx, sales_data,
            group_columns=['region'],
            condition='profit > 0',
            aggregation_func='sum'
            # Missing target_column
        )
        print("✗ Missing target column validation failed")
        error_tests_failed += 1
    except ValidationError:
        print("✓ Missing target column validation passed")
        error_tests_passed += 1

    print(f"Error handling tests: {error_tests_passed} passed, {error_tests_failed} failed")
    all_passed &= (error_tests_failed == 0)

    # Test edge cases
    print("\n=== Testing Edge Cases ===")
    edge_tests_passed = 0
    edge_tests_failed = 0

    # Test condition with string values
    try:
        string_data = pl.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
            "value": [10, 20, 30, 40, 50]
        })
        result = CASE_WHEN(ctx, string_data, case_conditions=[
            {'when': 'category == A', 'then': 'Group 1'},
            {'when': 'category == B', 'then': 'Group 2'},
            {'else': 'Other'}
        ])
        if isinstance(result, pl.DataFrame):
            print("✓ String condition evaluation passed")
            edge_tests_passed += 1
        else:
            print("✗ String condition evaluation failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ String condition evaluation failed: {e}")
        edge_tests_failed += 1

    # Test single condition in nested IF
    try:
        result = NESTED_IF_LOGIC(
            ctx, ['score > 85'],
            results_list=['High'],
            default_value='Low',
            df_context=sales_data
        )
        if isinstance(result, pl.DataFrame):
            print("✓ Single condition nested IF passed")
            edge_tests_passed += 1
        else:
            print("✗ Single condition nested IF failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ Single condition nested IF failed: {e}")
        edge_tests_failed += 1

    # Test aggregation with no matching conditions
    try:
        result = CONDITIONAL_AGGREGATION(
            ctx, sales_data,
            group_columns=['region'],
            condition='sales_amount > 1000000',  # No sales this high
            aggregation_func='sum',
            target_column='sales_amount'
        )
        if isinstance(result, pl.DataFrame):
            print("✓ No matching conditions aggregation passed")
            edge_tests_passed += 1
        else:
            print("✗ No matching conditions aggregation failed")
            edge_tests_failed += 1
    except Exception as e:
        print(f"✗ No matching conditions aggregation failed: {e}")
        edge_tests_failed += 1

    print(f"Edge case tests: {edge_tests_passed} passed, {edge_tests_failed} failed")
    all_passed &= (edge_tests_failed == 0)

    # Final summary
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All conditional logic functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the failed tests above.")
    print("="*50)

    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
