"""
Conditional Logic Functions

Advanced conditional operations for complex business rules.
All functions use Polars for optimal performance and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, List, Dict, Union
from pathlib import Path
import polars as pl
import re
from tools.tool_exceptions import (
    FPABaseException,
    RetryAfterCorrectionError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)
from tools.toolset_utils import load_df, save_df_to_analysis_dir

# Set decimal precision for financial calculations
getcontext().prec = 28


def _parse_condition_string(condition: str, df: pl.DataFrame) -> pl.Expr:
    """
    Parse a string condition into a Polars expression.

    Args:
        condition: String condition like 'revenue > 1000' or 'score >= 90'
        df: DataFrame to validate column names against

    Returns:
        pl.Expr: Polars expression

    Raises:
        ValidationError: If condition syntax is invalid
        DataQualityError: If column references are invalid
    """
    try:
        # Clean the condition string
        condition = condition.strip()

        # Pattern to match: column_name operator value
        # Supports: ==, !=, >=, <=, >, <
        pattern = r'(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+)'
        match = re.match(pattern, condition)

        if not match:
            raise ValidationError(f"Invalid condition syntax: {condition}")

        column_name, operator, value_str = match.groups()

        # Validate column exists
        if column_name not in df.columns:
            raise DataQualityError(
                f"Column '{column_name}' not found in DataFrame",
                f"Available columns: {df.columns}"
            )

        # Parse value - try numeric first, then string
        try:
            # Try to parse as number
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            # Treat as string, remove quotes if present
            value = value_str.strip('\'"')

        # Build Polars expression
        col_expr = pl.col(column_name)

        if operator == '==':
            return col_expr == value
        elif operator == '!=':
            return col_expr != value
        elif operator == '>':
            return col_expr > value
        elif operator == '<':
            return col_expr < value
        elif operator == '>=':
            return col_expr >= value
        elif operator == '<=':
            return col_expr <= value
        else:
            raise ValidationError(f"Unsupported operator: {operator}")

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise ValidationError(f"Failed to parse condition '{condition}': {str(e)}")


def MULTI_CONDITION_LOGIC(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    condition_tree: Dict[str, Any],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Apply complex multi-condition logic with nested if/elif/else structures.

    This function processes hierarchical conditional logic trees, enabling complex
    business rule implementations for financial analysis and decision-making.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to apply logic to, or file path to load data from
        condition_tree: Nested dictionary defining conditional logic structure
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with conditional results column, or path if output_filename provided

    Raises:
        ValidationError: If condition tree structure is invalid
        DataQualityError: If column references are invalid
        CalculationError: If conditional evaluation fails

    Financial Examples:
        # Credit risk assessment with multiple criteria
        >>> risk_tree = {
        ...     'if': 'credit_score >= 750',
        ...     'then': 'Low Risk',
        ...     'elif': [
        ...         {'condition': 'credit_score >= 650', 'then': 'Medium Risk'},
        ...         {'condition': 'debt_ratio <= 0.3', 'then': 'Medium Risk'}
        ...     ],
        ...     'else': 'High Risk'
        ... }
        >>> result = MULTI_CONDITION_LOGIC(ctx, customer_df, condition_tree=risk_tree)

        # Investment allocation strategy
        >>> allocation_tree = {
        ...     'if': 'age < 30',
        ...     'then': 'Aggressive',
        ...     'elif': [
        ...         {'condition': 'age < 50', 'then': 'Moderate'},
        ...         {'condition': 'risk_tolerance > 7', 'then': 'Moderate'}
        ...     ],
        ...     'else': 'Conservative'
        ... }
        >>> result = MULTI_CONDITION_LOGIC(ctx, portfolio_df, condition_tree=allocation_tree)

        # Revenue categorization for reporting
        >>> revenue_tree = {
        ...     'if': 'revenue > 1000000',
        ...     'then': 'Enterprise',
        ...     'elif': [
        ...         {'condition': 'revenue > 100000', 'then': 'Corporate'},
        ...         {'condition': 'revenue > 10000', 'then': 'SMB'}
        ...     ],
        ...     'else': 'Startup'
        ... }
        >>> result = MULTI_CONDITION_LOGIC(ctx, "revenue_data.parquet", condition_tree=revenue_tree)

    Condition Tree Structure:
        {
            'if': 'condition_string',           # Primary condition
            'then': 'result_value',             # Result if primary condition is true
            'elif': [                           # Optional list of additional conditions
                {'condition': 'condition_string', 'then': 'result_value'},
                {'condition': 'condition_string', 'then': 'result_value'}
            ],
            'else': 'default_value'             # Default result if no conditions match
        }
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Validate input DataFrame
    if not isinstance(df, pl.DataFrame):
        raise ValidationError("Input must be a Polars DataFrame or file path")

    if df.is_empty():
        raise ValidationError("Input DataFrame cannot be empty")

    # Validate condition tree structure
    if not isinstance(condition_tree, dict):
        raise ValidationError("Condition tree must be a dictionary")

    if 'if' not in condition_tree or 'then' not in condition_tree:
        raise ValidationError("Condition tree must contain 'if' and 'then' keys")

    try:
        # Start building the conditional expression
        primary_condition = _parse_condition_string(condition_tree['if'], df)
        primary_result = condition_tree['then']

        # Build the when/then chain
        expr = pl.when(primary_condition).then(pl.lit(primary_result))

        # Process elif conditions if present
        if 'elif' in condition_tree:
            elif_conditions = condition_tree['elif']
            if not isinstance(elif_conditions, list):
                raise ValidationError("'elif' must be a list of condition dictionaries")

            for elif_item in elif_conditions:
                if not isinstance(elif_item, dict) or 'condition' not in elif_item or 'then' not in elif_item:
                    raise ValidationError("Each elif item must have 'condition' and 'then' keys")

                elif_condition = _parse_condition_string(elif_item['condition'], df)
                elif_result = elif_item['then']
                expr = expr.when(elif_condition).then(pl.lit(elif_result))

        # Add else clause if present
        if 'else' in condition_tree:
            expr = expr.otherwise(pl.lit(condition_tree['else']))
        else:
            expr = expr.otherwise(pl.lit(None))

        # Apply the conditional logic to create result DataFrame
        result_df = df.with_columns(expr.alias("conditional_result"))

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"Multi-condition logic evaluation failed: {str(e)}")


def NESTED_IF_LOGIC(
    run_context: Any,
    conditions_list: List[Union[str, pl.Expr]],
    *,
    results_list: List[Any],
    default_value: Any,
    df_context: Union[pl.DataFrame, str, Path] | None = None,
    output_filename: str | None = None
) -> Union[List[Any], pl.DataFrame, Path]:
    """
    Handle nested conditional statements with cascading if-then-else logic.

    This function evaluates a series of conditions in order, returning the result
    corresponding to the first true condition. Essential for complex financial
    decision trees and multi-tier classification systems.

    Args:
        run_context: RunContext object for file operations
        conditions_list: List of condition strings or Polars expressions
        results_list: List of corresponding results for each condition
        default_value: Default value if no conditions are met
        df_context: Optional DataFrame context for string condition evaluation
        output_filename: Optional filename to save results as parquet file

    Returns:
        List, DataFrame, or Path: Conditional results or path if output_filename provided

    Raises:
        ValidationError: If input lists have mismatched lengths
        DataQualityError: If conditions reference invalid columns
        CalculationError: If conditional evaluation fails

    Financial Examples:
        # Bond rating classification
        >>> conditions = [
        ...     'credit_score >= 800',
        ...     'credit_score >= 700',
        ...     'credit_score >= 600',
        ...     'credit_score >= 500'
        ... ]
        >>> ratings = ['AAA', 'AA', 'A', 'BBB']
        >>> result = NESTED_IF_LOGIC(ctx, conditions, results_list=ratings,
        ...                         default_value='Junk', df_context=bond_df)

        # Commission tier calculation
        >>> sales_conditions = [
        ...     'sales_amount >= 100000',
        ...     'sales_amount >= 50000',
        ...     'sales_amount >= 25000'
        ... ]
        >>> commission_rates = [0.15, 0.12, 0.08]
        >>> result = NESTED_IF_LOGIC(ctx, sales_conditions, results_list=commission_rates,
        ...                         default_value=0.05, df_context="sales_data.parquet")

        # Investment risk categorization
        >>> risk_conditions = [
        ...     'volatility > 0.25',
        ...     'volatility > 0.15',
        ...     'volatility > 0.08'
        ... ]
        >>> risk_levels = ['High', 'Medium', 'Low']
        >>> result = NESTED_IF_LOGIC(ctx, risk_conditions, results_list=risk_levels,
        ...                         default_value='Very Low', df_context=portfolio_df)

    Use Cases:
        - Credit scoring and risk assessment
        - Performance bonus calculations
        - Investment allocation strategies
        - Customer segmentation
        - Pricing tier determination
    """
    # Validate input parameters
    if len(conditions_list) != len(results_list):
        raise ValidationError("Conditions list and results list must have the same length")

    if not conditions_list:
        raise ValidationError("Conditions list cannot be empty")

    # Handle DataFrame context for string conditions
    df = None
    if df_context is not None:
        if isinstance(df_context, (str, Path)):
            df = load_df(run_context, df_context)
        elif isinstance(df_context, pl.DataFrame):
            df = df_context
        else:
            raise ValidationError("df_context must be a DataFrame or file path")

    try:
        # If we have a DataFrame context, apply conditions to DataFrame
        if df is not None:
            # Build cascading when/then expression
            expr = None

            for i, (condition, result) in enumerate(zip(conditions_list, results_list)):
                # Parse condition if it's a string
                if isinstance(condition, str):
                    condition_expr = _parse_condition_string(condition, df)
                else:
                    condition_expr = condition

                if expr is None:
                    expr = pl.when(condition_expr).then(pl.lit(result))
                else:
                    expr = expr.when(condition_expr).then(pl.lit(result))

            # Add default value
            expr = expr.otherwise(pl.lit(default_value))

            # Apply to DataFrame
            result_df = df.with_columns(expr.alias("nested_if_result"))

            # Save results to file if output_filename is provided
            if output_filename is not None:
                return save_df_to_analysis_dir(run_context, result_df, output_filename)

            return result_df

        else:
            # Simple list-based evaluation (conditions must be boolean values)
            results = []
            for condition, result in zip(conditions_list, results_list):
                if isinstance(condition, bool) and condition:
                    results.append(result)
                elif hasattr(condition, '__bool__') and bool(condition):
                    results.append(result)
                else:
                    results.append(default_value)

            return results

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"Nested IF logic evaluation failed: {str(e)}")


def CASE_WHEN(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    case_conditions: List[Dict[str, Any]],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    SQL-style CASE WHEN logic for multiple conditional branches.

    This function implements SQL-like CASE WHEN statements, providing a clean
    and efficient way to handle multiple conditional branches in financial
    data processing and business rule implementation.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to apply case logic to, or file path to load data from
        case_conditions: List of dictionaries with 'when' and 'then' keys
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with case results column, or path if output_filename provided

    Raises:
        ValidationError: If case conditions structure is invalid
        DataQualityError: If column references are invalid
        CalculationError: If case evaluation fails

    Financial Examples:
        # Customer segment classification
        >>> segments = [
        ...     {'when': 'annual_revenue >= 1000000', 'then': 'Enterprise'},
        ...     {'when': 'annual_revenue >= 100000', 'then': 'Corporate'},
        ...     {'when': 'annual_revenue >= 10000', 'then': 'SMB'},
        ...     {'else': 'Startup'}
        ... ]
        >>> result = CASE_WHEN(ctx, customer_df, case_conditions=segments)

        # Performance rating system
        >>> performance_cases = [
        ...     {'when': 'score >= 90', 'then': 'Excellent'},
        ...     {'when': 'score >= 80', 'then': 'Good'},
        ...     {'when': 'score >= 70', 'then': 'Satisfactory'},
        ...     {'when': 'score >= 60', 'then': 'Needs Improvement'},
        ...     {'else': 'Unsatisfactory'}
        ... ]
        >>> result = CASE_WHEN(ctx, "performance_data.parquet", case_conditions=performance_cases)

        # Investment allocation based on age and risk tolerance
        >>> allocation_cases = [
        ...     {'when': 'age < 30', 'then': 'Growth'},
        ...     {'when': 'age < 50', 'then': 'Balanced'},
        ...     {'when': 'risk_tolerance > 5', 'then': 'Conservative'},
        ...     {'else': 'Income'}
        ... ]
        >>> result = CASE_WHEN(ctx, portfolio_df, case_conditions=allocation_cases)

    Case Conditions Structure:
        [
            {'when': 'condition_string', 'then': 'result_value'},
            {'when': 'condition_string', 'then': 'result_value'},
            {'else': 'default_value'}  # Optional else clause
        ]
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Validate input DataFrame
    if not isinstance(df, pl.DataFrame):
        raise ValidationError("Input must be a Polars DataFrame or file path")

    if df.is_empty():
        raise ValidationError("Input DataFrame cannot be empty")

    # Validate case conditions
    if not isinstance(case_conditions, list) or not case_conditions:
        raise ValidationError("Case conditions must be a non-empty list")

    try:
        expr = None
        else_value = None

        for i, case in enumerate(case_conditions):
            if not isinstance(case, dict):
                raise ValidationError(f"Case condition {i} must be a dictionary")

            # Handle else clause
            if 'else' in case:
                else_value = case['else']
                continue

            # Validate when/then structure
            if 'when' not in case or 'then' not in case:
                raise ValidationError(f"Case condition {i} must have 'when' and 'then' keys")

            # Parse condition
            condition_expr = _parse_condition_string(case['when'], df)
            result_value = case['then']

            # Build expression chain
            if expr is None:
                expr = pl.when(condition_expr).then(pl.lit(result_value))
            else:
                expr = expr.when(condition_expr).then(pl.lit(result_value))

        # Add else clause if provided
        if else_value is not None:
            expr = expr.otherwise(pl.lit(else_value))
        else:
            expr = expr.otherwise(pl.lit(None))

        # Apply case logic to DataFrame
        result_df = df.with_columns(expr.alias("case_result"))

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"CASE WHEN evaluation failed: {str(e)}")


def CONDITIONAL_AGGREGATION(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    group_columns: List[str],
    condition: str,
    aggregation_func: str,
    target_column: str | None = None,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Aggregate data based on conditions, similar to SQL HAVING clause.

    This function performs conditional aggregations, enabling sophisticated
    financial analysis such as conditional sums, counts, and averages based
    on business rules and filtering criteria.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to aggregate, or file path to load data from
        group_columns: List of columns to group by
        condition: Condition string to filter data before aggregation
        aggregation_func: Aggregation function ('sum', 'count', 'mean', 'max', 'min', 'std')
        target_column: Column to aggregate (required for most functions except 'count')
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with conditional aggregations, or path if output_filename provided

    Raises:
        ValidationError: If parameters are invalid
        DataQualityError: If column references are invalid
        CalculationError: If aggregation fails
        ConfigurationError: If aggregation function is not supported

    Financial Examples:
        # Sum high-value transactions by region
        >>> result = CONDITIONAL_AGGREGATION(
        ...     ctx, sales_df,
        ...     group_columns=['region'],
        ...     condition='amount > 1000',
        ...     aggregation_func='sum',
        ...     target_column='amount'
        ... )

        # Count profitable customers by segment
        >>> result = CONDITIONAL_AGGREGATION(
        ...     ctx, "customer_data.parquet",
        ...     group_columns=['segment'],
        ...     condition='profit > 0',
        ...     aggregation_func='count'
        ... )

        # Average revenue for enterprise clients by quarter
        >>> result = CONDITIONAL_AGGREGATION(
        ...     ctx, revenue_df,
        ...     group_columns=['quarter'],
        ...     condition='client_type == Enterprise',
        ...     aggregation_func='mean',
        ...     target_column='revenue'
        ... )

        # Maximum deal size by sales rep for deals over $50K
        >>> result = CONDITIONAL_AGGREGATION(
        ...     ctx, deals_df,
        ...     group_columns=['sales_rep'],
        ...     condition='deal_size > 50000',
        ...     aggregation_func='max',
        ...     target_column='deal_size'
        ... )

    Supported Aggregation Functions:
        - 'sum': Sum of values
        - 'count': Count of rows
        - 'mean'/'avg': Average of values
        - 'max': Maximum value
        - 'min': Minimum value
        - 'std': Standard deviation
        - 'var': Variance
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Validate input DataFrame
    if not isinstance(df, pl.DataFrame):
        raise ValidationError("Input must be a Polars DataFrame or file path")

    if df.is_empty():
        raise ValidationError("Input DataFrame cannot be empty")

    # Validate parameters
    if not isinstance(group_columns, list) or not group_columns:
        raise ValidationError("Group columns must be a non-empty list")

    if not isinstance(condition, str) or not condition.strip():
        raise ValidationError("Condition must be a non-empty string")

    if not isinstance(aggregation_func, str):
        raise ValidationError("Aggregation function must be a string")

    # Validate group columns exist
    for col in group_columns:
        if col not in df.columns:
            raise DataQualityError(
                f"Group column '{col}' not found in DataFrame",
                f"Available columns: {df.columns}"
            )

    # Validate target column for functions that need it
    agg_func_lower = aggregation_func.lower()
    if agg_func_lower not in ['count'] and target_column is None:
        raise ValidationError(f"Target column is required for aggregation function '{aggregation_func}'")

    if target_column is not None and target_column not in df.columns:
        raise DataQualityError(
            f"Target column '{target_column}' not found in DataFrame",
            f"Available columns: {df.columns}"
        )

    # Validate aggregation function
    supported_funcs = ['sum', 'count', 'mean', 'avg', 'max', 'min', 'std', 'var']
    if agg_func_lower not in supported_funcs:
        raise ConfigurationError(f"Unsupported aggregation function: {aggregation_func}. Supported: {supported_funcs}")

    try:
        # Parse condition
        condition_expr = _parse_condition_string(condition, df)

        # Apply condition filter and group by
        if agg_func_lower == 'count':
            # For count, we don't need a target column
            agg_expr = pl.col(group_columns[0]).filter(condition_expr).count().alias(f"conditional_{agg_func_lower}")
        else:
            # For other functions, use target column
            col_expr = pl.col(target_column).filter(condition_expr)

            if agg_func_lower == 'sum':
                agg_expr = col_expr.sum().alias(f"conditional_{agg_func_lower}")
            elif agg_func_lower in ['mean', 'avg']:
                agg_expr = col_expr.mean().alias(f"conditional_{agg_func_lower}")
            elif agg_func_lower == 'max':
                agg_expr = col_expr.max().alias(f"conditional_{agg_func_lower}")
            elif agg_func_lower == 'min':
                agg_expr = col_expr.min().alias(f"conditional_{agg_func_lower}")
            elif agg_func_lower == 'std':
                agg_expr = col_expr.std().alias(f"conditional_{agg_func_lower}")
            elif agg_func_lower == 'var':
                agg_expr = col_expr.var().alias(f"conditional_{agg_func_lower}")

        # Perform grouped aggregation
        result_df = df.group_by(group_columns).agg(agg_expr)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError, ConfigurationError)):
            raise
        raise CalculationError(f"Conditional aggregation failed: {str(e)}")
