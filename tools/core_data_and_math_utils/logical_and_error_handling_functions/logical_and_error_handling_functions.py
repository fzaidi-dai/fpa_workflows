"""
Logical & Error-Handling Functions

These functions help structure decision-making processes and manage errors gracefully.
All functions use Decimal precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, Union, Optional, Callable
import polars as pl
import numpy as np
import re
from functools import lru_cache
from pathlib import Path
from tools.tool_exceptions import (
    FPABaseException,
    RetryAfterCorrectionError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)
from tools.toolset_utils import load_df

# Set decimal precision for financial calculations
getcontext().prec = 28

# Performance optimization: Cache compiled regex patterns and validation results
_VALIDATION_CACHE = {}
_ERROR_PATTERN_CACHE = {}


def _validate_logical_input(value: Any, function_name: str) -> Any:
    """
    Enhanced input validation for logical operations with comprehensive type checking.

    Args:
        value: Input value to validate
        function_name: Name of calling function for error messages

    Returns:
        Any: Validated value

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid values
    """
    try:
        # Handle None values
        if value is None:
            return False

        # Handle boolean values
        if isinstance(value, bool):
            return value

        # Enhanced numeric validation with range checking
        if isinstance(value, (int, float, Decimal)):
            # Check for invalid numeric values using numpy for float validation
            if isinstance(value, float):
                if np.isnan(value):
                    raise DataQualityError(
                        f"NaN value detected in {function_name}",
                        "Replace NaN values with valid numbers or use IFNA() to handle missing data"
                    )
                if np.isinf(value):
                    raise DataQualityError(
                        f"Infinite value detected in {function_name}",
                        "Check calculations for division by zero or overflow conditions"
                    )

            # Check for extremely large values that might indicate calculation errors
            if isinstance(value, (int, float)) and abs(value) > 1e15:
                raise DataQualityError(
                    f"Extremely large value ({value}) detected in {function_name}",
                    "Verify calculation logic and consider using IFERROR() for safe calculations"
                )

            return value != 0

        # Enhanced string validation
        if isinstance(value, str):
            # Check for error strings first
            if _is_error_value(value):
                raise DataQualityError(
                    f"Error value '{value}' detected in {function_name}",
                    "Use IFERROR() or IFNA() to handle error conditions before logical evaluation"
                )

            lower_val = value.lower().strip()

            # Explicit boolean string mappings
            true_strings = ['true', '1', 'yes', 'y', 't', 'on', 'enabled', 'active']
            false_strings = ['false', '0', 'no', 'n', 'f', 'off', 'disabled', 'inactive', '']

            if lower_val in true_strings:
                return True
            elif lower_val in false_strings:
                return False
            else:
                # For other non-empty strings, return True
                return bool(value.strip())

        # Enhanced Polars Series validation using built-in methods
        if isinstance(value, pl.Series):
            # Check for empty Series using len() method
            if len(value) == 0:
                raise ValidationError(
                    f"Empty Series provided to {function_name}",
                    "Provide a Series with at least one element for logical evaluation"
                )

            # Check for problematic values using Polars built-in methods
            if value.dtype.is_numeric():
                # Use Polars built-in methods for validation
                if value.is_nan().any():
                    raise DataQualityError(
                        f"NaN values detected in Series for {function_name}",
                        "Use Series.fill_nan() or IFNA() to handle NaN values before logical evaluation"
                    )
                if value.is_infinite().any():
                    raise DataQualityError(
                        f"Infinite values detected in Series for {function_name}",
                        "Check calculations for division by zero or use IFERROR() for safe operations"
                    )

            try:
                return value.map_elements(lambda x: _validate_logical_input(x, function_name), return_dtype=pl.Boolean)
            except Exception as e:
                raise DataQualityError(
                    f"Series validation failed in {function_name}: {str(e)}",
                    "Ensure all Series elements are valid logical values (boolean, numeric, or string)"
                )

        # Enhanced list and array validation
        if isinstance(value, (list, np.ndarray)):
            # Check for empty containers
            if len(value) == 0:
                raise ValidationError(
                    f"Empty list/array provided to {function_name}",
                    "Provide a list/array with at least one element for logical evaluation"
                )

            try:
                return [_validate_logical_input(item, function_name) for item in value]
            except Exception as e:
                raise DataQualityError(
                    f"List/array validation failed in {function_name}: {str(e)}",
                    "Ensure all list/array elements are valid logical values"
                )

        # Handle complex numbers (should not be used in logical operations)
        if isinstance(value, complex):
            raise ValidationError(
                f"Complex number provided to {function_name}",
                "Complex numbers cannot be used in logical operations. Use real numbers instead."
            )

        # Handle datetime objects (common in financial data)
        if hasattr(value, 'year') and hasattr(value, 'month'):  # datetime-like object
            raise ValidationError(
                f"Date/datetime object provided to {function_name}",
                "Convert date/datetime to boolean using comparison operations (e.g., date > threshold_date)"
            )

        # Handle dictionary and other complex objects
        if isinstance(value, (dict, set)):
            raise ValidationError(
                f"Complex object ({type(value).__name__}) provided to {function_name}",
                "Extract specific values from complex objects before logical evaluation"
            )

        # For other types, use Python's truthiness with validation
        try:
            result = bool(value)
            return result
        except Exception:
            raise ValidationError(
                f"Cannot convert {type(value).__name__} to boolean in {function_name}",
                "Provide a value that can be evaluated as True/False (boolean, number, string, or Series)"
            )

    except (ValidationError, DataQualityError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        raise DataQualityError(
            f"Unexpected error during logical validation in {function_name}: {str(e)}",
            "Ensure input is a valid type for logical operations (boolean, number, string, or Series)"
        )


@lru_cache(maxsize=512)
def _convert_to_decimal_safe(value: Any) -> Decimal:
    """
    Safely convert value to Decimal with caching for performance.

    Args:
        value: Value to convert

    Returns:
        Decimal: Converted value

    Raises:
        DataQualityError: If conversion fails
    """
    try:
        if isinstance(value, Decimal):
            return value
        if value is None:
            return Decimal('0')
        return Decimal(str(value))
    except (ValueError, TypeError, OverflowError) as e:
        raise DataQualityError(
            f"Cannot convert value to Decimal: {str(e)}",
            "Ensure value is a valid numeric type"
        )


def _is_error_value(value: Any) -> bool:
    """
    Check if value represents an error condition.

    Args:
        value: Value to check

    Returns:
        bool: True if value represents an error
    """
    if isinstance(value, Exception):
        return True

    if isinstance(value, str):
        error_patterns = ['#DIV/0!', '#N/A', '#NAME?', '#NULL!', '#NUM!', '#REF!', '#VALUE!', '#ERROR!']
        return value.upper() in error_patterns

    return False


def _is_na_value(value: Any) -> bool:
    """
    Check if value represents a #N/A error condition.

    Args:
        value: Value to check

    Returns:
        bool: True if value represents #N/A error
    """
    if isinstance(value, str):
        return value.upper() in ['#N/A', '#NA', 'N/A', 'NA']

    return False


def IF(run_context: Any, logical_test: Any, value_if_true: Any, value_if_false: Any) -> Any:
    """
    Return different values depending on whether a condition is met.

    Essential for financial decision-making, budget validation, and conditional calculations
    in FP&A workflows where different actions are required based on specific criteria.

    Args:
        logical_test: Logical test to evaluate
        value_if_true: Value to return if test is True
        value_if_false: Value to return if test is False

    Returns:
        Any: Value based on condition result

    Raises:
        ValidationError: If logical test cannot be evaluated
        DataQualityError: If input contains invalid values

    Financial Examples:
        # Budget variance analysis
        >>> actual = 105000
        >>> budget = 100000
        >>> variance_status = IF(actual > budget, "Over Budget", "Within Budget")
        >>> print(f"Status: {variance_status}")
        Status: Over Budget

        # Credit approval logic
        >>> credit_score = 750
        >>> annual_income = 85000
        >>> approval = IF(credit_score >= 700 and annual_income >= 50000, "Approved", "Denied")
        >>> print(f"Credit application: {approval}")
        Credit application: Approved

        # Performance bonus calculation
        >>> performance_rating = 4.2
        >>> bonus_amount = IF(performance_rating >= 4.0, 5000, 2000)
        >>> print(f"Bonus: ${bonus_amount:,}")
        Bonus: $5,000

    Example:
        >>> IF(100 > 50, "High", "Low")
        'High'
        >>> IF(False, 1000, 500)
        500
    """
    try:
        # Validate and evaluate logical test
        test_result = _validate_logical_input(logical_test, "IF")

        # Handle vectorized operations for Polars Series
        if isinstance(test_result, pl.Series):
            # Create a temporary DataFrame to use pl.when expression
            temp_df = pl.DataFrame({"condition": test_result})
            result_series = temp_df.select(
                pl.when(pl.col("condition"))
                .then(pl.lit(value_if_true))
                .otherwise(pl.lit(value_if_false))
                .alias("result")
            )["result"]
            return result_series

        # Handle list operations
        if isinstance(test_result, list):
            return [value_if_true if result else value_if_false for result in test_result]

        # Single value operation
        return value_if_true if test_result else value_if_false

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"IF calculation failed: {str(e)}")


def IFS(run_context: Any, *conditions_and_values: Any) -> Any:
    """
    Test multiple conditions without nesting several IF statements.

    Critical for complex financial decision trees, multi-tier pricing models,
    and sophisticated business rule evaluation in FP&A systems.

    Args:
        conditions_and_values: Alternating logical tests and values (condition1, value1, condition2, value2, ...)

    Returns:
        Any: Value from first true condition

    Raises:
        ValidationError: If arguments are not in pairs or no conditions are true
        CalculationError: If condition evaluation fails

    Financial Examples:
        # Credit rating assignment
        >>> credit_score = 780
        >>> rating = IFS(
        ...     credit_score >= 800, "AAA",
        ...     credit_score >= 750, "AA",
        ...     credit_score >= 700, "A",
        ...     credit_score >= 650, "BBB",
        ...     True, "Below Investment Grade"
        ... )
        >>> print(f"Credit Rating: {rating}")
        Credit Rating: AA

        # Commission tier calculation
        >>> sales_amount = 125000
        >>> commission_rate = IFS(
        ...     sales_amount >= 200000, 0.08,
        ...     sales_amount >= 150000, 0.06,
        ...     sales_amount >= 100000, 0.04,
        ...     sales_amount >= 50000, 0.02,
        ...     True, 0.01
        ... )
        >>> print(f"Commission Rate: {commission_rate:.1%}")
        Commission Rate: 4.0%

        # Budget approval workflow
        >>> amount = 75000
        >>> department = "Marketing"
        >>> approval_level = IFS(
        ...     amount >= 100000, "Board Approval Required",
        ...     amount >= 50000 and department == "Marketing", "VP Approval Required",
        ...     amount >= 25000, "Director Approval Required",
        ...     True, "Manager Approval Sufficient"
        ... )
        >>> print(f"Approval Level: {approval_level}")
        Approval Level: VP Approval Required

    Example:
        >>> IFS(False, "A", True, "B", False, "C")
        'B'
        >>> IFS(10 > 100, "High", 10 > 5, "Medium", True, "Low")
        'Medium'
    """
    # Input validation
    if len(conditions_and_values) < 2:
        raise ValidationError("IFS requires at least one condition-value pair")

    if len(conditions_and_values) % 2 != 0:
        raise ValidationError("IFS requires an even number of arguments (condition-value pairs)")

    try:
        # Process condition-value pairs
        for i in range(0, len(conditions_and_values), 2):
            condition = conditions_and_values[i]
            value = conditions_and_values[i + 1]

            # Evaluate condition
            test_result = _validate_logical_input(condition, "IFS")

            if test_result:
                return value

        # If no conditions are true, raise error
        raise CalculationError("No conditions in IFS evaluated to True")

    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError)):
            raise
        raise CalculationError(f"IFS calculation failed: {str(e)}")


def AND(run_context: Any, *logical_tests: Any) -> Union[bool, pl.Series]:
    """
    Test if all conditions are true.

    Essential for financial compliance checks, multi-criteria validation,
    and comprehensive risk assessment where all conditions must be satisfied.

    Args:
        logical_tests: Multiple logical tests to evaluate

    Returns:
        bool or pl.Series: True if all conditions are true, False otherwise.
                          Returns Series if any input is a Series.

    Raises:
        ValidationError: If no tests provided
        DataQualityError: If logical tests cannot be evaluated

    Financial Examples:
        # Investment criteria validation
        >>> pe_ratio = 15.2
        >>> debt_to_equity = 0.3
        >>> roe = 0.18
        >>> meets_criteria = AND(pe_ratio < 20, debt_to_equity < 0.5, roe > 0.15)
        >>> print(f"Investment meets all criteria: {meets_criteria}")
        Investment meets all criteria: True

        # Loan approval requirements
        >>> credit_score = 720
        >>> debt_to_income = 0.35
        >>> employment_years = 3
        >>> down_payment_pct = 0.20
        >>> loan_approved = AND(
        ...     credit_score >= 700,
        ...     debt_to_income <= 0.40,
        ...     employment_years >= 2,
        ...     down_payment_pct >= 0.15
        ... )
        >>> print(f"Loan approved: {loan_approved}")
        Loan approved: True

        # Budget compliance check
        >>> actual_expense = 95000
        >>> budget_limit = 100000
        >>> within_variance = 0.03
        >>> compliance = AND(
        ...     actual_expense <= budget_limit,
        ...     abs(actual_expense - budget_limit) / budget_limit <= within_variance
        ... )
        >>> print(f"Budget compliant: {compliance}")
        Budget compliant: False

        # Vectorized portfolio analysis
        >>> pe_ratios = pl.Series([15.2, 25.1, 18.5])
        >>> debt_ratios = pl.Series([0.3, 0.6, 0.4])
        >>> roe_values = pl.Series([0.18, 0.12, 0.20])
        >>> meets_criteria = AND(pe_ratios < 20, debt_ratios < 0.5, roe_values > 0.15)
        >>> print(f"Portfolio criteria: {meets_criteria.to_list()}")
        Portfolio criteria: [True, False, True]

    Example:
        >>> AND(True, True, True)
        True
        >>> AND(True, False, True)
        False
        >>> AND(10 > 5, 20 > 15, 30 > 25)
        True
    """
    if not logical_tests:
        raise ValidationError(
            "AND function requires at least one logical test for financial validation",
            "Provide at least one logical condition to evaluate, e.g., AND(revenue > 100000, profit_margin > 0.1)"
        )

    try:
        # Check if any input is a Polars Series for vectorized operation
        series_inputs = [test for test in logical_tests if isinstance(test, pl.Series)]

        if series_inputs:
            # Vectorized operation for Series inputs
            # Get the length from the first Series
            series_length = len(series_inputs[0])

            # Validate all Series have the same length
            for i, series in enumerate(series_inputs[1:], 1):
                if len(series) != series_length:
                    raise DataQualityError(
                        f"AND function: Series length mismatch at position {i+1}. Expected {series_length}, got {len(series)}",
                        "Ensure all Series inputs have the same length for vectorized operations"
                    )

            # Convert all inputs to boolean Series
            validated_series = []
            for test in logical_tests:
                if isinstance(test, pl.Series):
                    # Use map_elements to convert to boolean
                    bool_series = test.map_elements(
                        lambda x: _validate_logical_input(x, "AND"),
                        return_dtype=pl.Boolean
                    )
                    validated_series.append(bool_series)
                else:
                    # Convert scalar to Series of same length
                    scalar_result = _validate_logical_input(test, "AND")
                    scalar_series = pl.Series([scalar_result] * series_length)
                    validated_series.append(scalar_series)

            # Combine all conditions with AND logic using bitwise &
            result = validated_series[0]
            for series in validated_series[1:]:
                result = result & series

            return result

        # Handle list operations
        list_inputs = [test for test in logical_tests if isinstance(test, list)]
        if list_inputs:
            # Convert all inputs to lists of same length
            max_length = max(len(test) if isinstance(test, list) else 1 for test in logical_tests)
            validated_lists = []

            for test in logical_tests:
                if isinstance(test, list):
                    validated_lists.append([_validate_logical_input(item, "AND") for item in test])
                else:
                    scalar_result = _validate_logical_input(test, "AND")
                    validated_lists.append([scalar_result] * max_length)

            # Combine with AND logic
            result = []
            for i in range(max_length):
                all_true = True
                for validated_list in validated_lists:
                    if not validated_list[i]:
                        all_true = False
                        break
                result.append(all_true)

            return result

        # Single value operations
        for test in logical_tests:
            test_result = _validate_logical_input(test, "AND")
            if not test_result:
                return False

        return True

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"AND function failed during financial validation: {str(e)}. Ensure all inputs are valid logical expressions for financial analysis.")


def OR(run_context: Any, *logical_tests: Any) -> Union[bool, pl.Series]:
    """
    Test if any condition is true.

    Critical for financial risk flagging, alternative criteria evaluation,
    and flexible business rule implementation where any condition can trigger an action.

    Args:
        logical_tests: Multiple logical tests to evaluate

    Returns:
        bool or pl.Series: True if any condition is true, False otherwise.
                          Returns Series if any input is a Series.

    Raises:
        ValidationError: If no tests provided
        DataQualityError: If logical tests cannot be evaluated

    Financial Examples:
        # Risk flag detection
        >>> debt_ratio = 0.85
        >>> liquidity_ratio = 0.8
        >>> profit_margin = -0.05
        >>> risk_flag = OR(debt_ratio > 0.8, liquidity_ratio < 1.0, profit_margin < 0)
        >>> print(f"Risk flag triggered: {risk_flag}")
        Risk flag triggered: True

        # Alternative payment methods
        >>> has_credit_card = True
        >>> has_bank_account = False
        >>> has_digital_wallet = False
        >>> can_pay = OR(has_credit_card, has_bank_account, has_digital_wallet)
        >>> print(f"Payment method available: {can_pay}")
        Payment method available: True

        # Audit trigger conditions
        >>> revenue_variance = 0.12
        >>> expense_variance = 0.08
        >>> margin_change = 0.15
        >>> audit_required = OR(
        ...     revenue_variance > 0.10,
        ...     expense_variance > 0.10,
        ...     margin_change > 0.10
        ... )
        >>> print(f"Audit required: {audit_required}")
        Audit required: True

        # Vectorized risk assessment
        >>> debt_ratios = pl.Series([0.85, 0.45, 0.90])
        >>> liquidity_ratios = pl.Series([0.8, 1.2, 0.6])
        >>> profit_margins = pl.Series([-0.05, 0.15, 0.02])
        >>> risk_flags = OR(debt_ratios > 0.8, liquidity_ratios < 1.0, profit_margins < 0)
        >>> print(f"Risk flags: {risk_flags.to_list()}")
        Risk flags: [True, False, True]

    Example:
        >>> OR(False, False, True)
        True
        >>> OR(False, False, False)
        False
        >>> OR(10 > 20, 5 > 3, 1 > 2)
        True
    """
    if not logical_tests:
        raise ValidationError(
            "OR function requires at least one logical test for financial validation",
            "Provide at least one logical condition to evaluate, e.g., OR(debt_ratio > 0.8, liquidity_ratio < 1.0)"
        )

    try:
        # Check if any input is a Polars Series for vectorized operation
        series_inputs = [test for test in logical_tests if isinstance(test, pl.Series)]

        if series_inputs:
            # Vectorized operation for Series inputs
            # Get the length from the first Series
            series_length = len(series_inputs[0])

            # Validate all Series have the same length
            for i, series in enumerate(series_inputs[1:], 1):
                if len(series) != series_length:
                    raise DataQualityError(
                        f"OR function: Series length mismatch at position {i+1}. Expected {series_length}, got {len(series)}",
                        "Ensure all Series inputs have the same length for vectorized operations"
                    )

            # Convert all inputs to boolean Series
            validated_series = []
            for test in logical_tests:
                if isinstance(test, pl.Series):
                    # Use map_elements to convert to boolean
                    bool_series = test.map_elements(
                        lambda x: _validate_logical_input(x, "OR"),
                        return_dtype=pl.Boolean
                    )
                    validated_series.append(bool_series)
                else:
                    # Convert scalar to Series of same length
                    scalar_result = _validate_logical_input(test, "OR")
                    scalar_series = pl.Series([scalar_result] * series_length)
                    validated_series.append(scalar_series)

            # Combine all conditions with OR logic using bitwise |
            result = validated_series[0]
            for series in validated_series[1:]:
                result = result | series

            return result

        # Handle list operations
        list_inputs = [test for test in logical_tests if isinstance(test, list)]
        if list_inputs:
            # Convert all inputs to lists of same length
            max_length = max(len(test) if isinstance(test, list) else 1 for test in logical_tests)
            validated_lists = []

            for test in logical_tests:
                if isinstance(test, list):
                    validated_lists.append([_validate_logical_input(item, "OR") for item in test])
                else:
                    scalar_result = _validate_logical_input(test, "OR")
                    validated_lists.append([scalar_result] * max_length)

            # Combine with OR logic
            result = []
            for i in range(max_length):
                any_true = False
                for validated_list in validated_lists:
                    if validated_list[i]:
                        any_true = True
                        break
                result.append(any_true)

            return result

        # Single value operations
        for test in logical_tests:
            test_result = _validate_logical_input(test, "OR")
            if test_result:
                return True

        return False

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"OR function failed during financial validation: {str(e)}. Ensure all inputs are valid logical expressions for financial analysis.")


def NOT(run_context: Any, logical: Any) -> Union[bool, pl.Series]:
    """
    Reverse the logical value of a condition.

    Essential for financial logic inversion, exception handling,
    and implementing negative criteria in business rules.

    Args:
        logical: Logical value to reverse

    Returns:
        bool or pl.Series: Opposite boolean value. Returns Series if input is a Series.

    Raises:
        DataQualityError: If logical value cannot be evaluated

    Financial Examples:
        # Investment exclusion criteria
        >>> is_tobacco_company = False
        >>> is_ethical_investment = NOT(is_tobacco_company)
        >>> print(f"Ethical investment: {is_ethical_investment}")
        Ethical investment: True

        # Non-compliance detection
        >>> meets_regulations = False
        >>> requires_action = NOT(meets_regulations)
        >>> print(f"Action required: {requires_action}")
        Action required: True

        # Opposite condition checking
        >>> is_profitable = True
        >>> needs_restructuring = NOT(is_profitable)
        >>> print(f"Needs restructuring: {needs_restructuring}")
        Needs restructuring: False

        # Vectorized exclusion criteria
        >>> tobacco_companies = pl.Series([False, True, False])
        >>> ethical_investments = NOT(tobacco_companies)
        >>> print(f"Ethical investments: {ethical_investments.to_list()}")
        Ethical investments: [True, False, True]

    Example:
        >>> NOT(True)
        False
        >>> NOT(False)
        True
        >>> NOT(10 > 5)
        False
    """
    try:
        test_result = _validate_logical_input(logical, "NOT")

        # Handle Polars Series
        if isinstance(test_result, pl.Series):
            return test_result.not_()

        # Handle lists
        if isinstance(test_result, list):
            return [not item for item in test_result]

        # Single value
        return not test_result

    except Exception as e:
        if isinstance(e, DataQualityError):
            raise
        raise CalculationError(f"NOT function failed during financial validation: {str(e)}. Ensure input is a valid logical expression for financial analysis.")


def XOR(run_context: Any, *logical_tests: Any) -> Union[bool, pl.Series]:
    """
    Exclusive OR - returns True if odd number of arguments are True.

    Useful for financial scenarios requiring mutually exclusive conditions,
    alternative investment strategies, or either-or business decisions.

    Args:
        logical_tests: Multiple logical tests to evaluate

    Returns:
        bool or pl.Series: True if odd number of conditions are true.
                          Returns Series if any input is a Series.

    Raises:
        ValidationError: If no tests provided
        DataQualityError: If logical tests cannot be evaluated

    Financial Examples:
        # Mutually exclusive investment options
        >>> invest_in_stocks = True
        >>> invest_in_bonds = False
        >>> invest_in_real_estate = False
        >>> single_investment = XOR(invest_in_stocks, invest_in_bonds, invest_in_real_estate)
        >>> print(f"Single investment strategy: {single_investment}")
        Single investment strategy: True

        # Alternative approval paths
        >>> ceo_approval = False
        >>> board_approval = True
        >>> committee_approval = False
        >>> has_approval = XOR(ceo_approval, board_approval, committee_approval)
        >>> print(f"Has single approval path: {has_approval}")
        Has single approval path: True

        # Exclusive market conditions
        >>> bull_market = True
        >>> bear_market = False
        >>> sideways_market = True
        >>> clear_trend = XOR(bull_market, bear_market, sideways_market)
        >>> print(f"Clear market trend: {clear_trend}")
        Clear market trend: False

        # Vectorized exclusive conditions
        >>> stocks = pl.Series([True, False, True])
        >>> bonds = pl.Series([False, True, False])
        >>> real_estate = pl.Series([False, False, True])
        >>> exclusive_investments = XOR(stocks, bonds, real_estate)
        >>> print(f"Exclusive investments: {exclusive_investments.to_list()}")
        Exclusive investments: [True, True, False]

    Example:
        >>> XOR(True, False, False)
        True
        >>> XOR(True, True, False)
        False
        >>> XOR(True, True, True)
        True
    """
    if not logical_tests:
        raise ValidationError(
            "XOR function requires at least one logical test for financial validation",
            "Provide at least one logical condition to evaluate, e.g., XOR(stocks_selected, bonds_selected, real_estate_selected)"
        )

    try:
        # Check if any input is a Polars Series for vectorized operation
        series_inputs = [test for test in logical_tests if isinstance(test, pl.Series)]

        if series_inputs:
            # Vectorized operation for Series inputs
            # Get the length from the first Series
            series_length = len(series_inputs[0])

            # Validate all Series have the same length
            for i, series in enumerate(series_inputs[1:], 1):
                if len(series) != series_length:
                    raise DataQualityError(
                        f"XOR function: Series length mismatch at position {i+1}. Expected {series_length}, got {len(series)}",
                        "Ensure all Series inputs have the same length for vectorized operations"
                    )

            # Convert all inputs to boolean Series
            validated_series = []
            for test in logical_tests:
                if isinstance(test, pl.Series):
                    # Use map_elements to convert to boolean
                    bool_series = test.map_elements(
                        lambda x: _validate_logical_input(x, "XOR"),
                        return_dtype=pl.Boolean
                    )
                    validated_series.append(bool_series)
                else:
                    # Convert scalar to Series of same length
                    scalar_result = _validate_logical_input(test, "XOR")
                    scalar_series = pl.Series([scalar_result] * series_length)
                    validated_series.append(scalar_series)

            # Count True values for each row and check if odd
            # Sum all boolean Series (True = 1, False = 0)
            sum_series = validated_series[0].cast(pl.Int32)
            for series in validated_series[1:]:
                sum_series = sum_series + series.cast(pl.Int32)

            # Return True where count is odd
            return (sum_series % 2) == 1

        # Handle list operations
        list_inputs = [test for test in logical_tests if isinstance(test, list)]
        if list_inputs:
            # Convert all inputs to lists of same length
            max_length = max(len(test) if isinstance(test, list) else 1 for test in logical_tests)
            validated_lists = []

            for test in logical_tests:
                if isinstance(test, list):
                    validated_lists.append([_validate_logical_input(item, "XOR") for item in test])
                else:
                    scalar_result = _validate_logical_input(test, "XOR")
                    validated_lists.append([scalar_result] * max_length)

            # Combine with XOR logic (count True values, return odd)
            result = []
            for i in range(max_length):
                true_count = 0
                for validated_list in validated_lists:
                    if validated_list[i]:
                        true_count += 1
                result.append(true_count % 2 == 1)

            return result

        # Single value operations
        true_count = 0
        for test in logical_tests:
            test_result = _validate_logical_input(test, "XOR")
            if test_result:
                true_count += 1

        # Return True if odd number of conditions are True
        return true_count % 2 == 1

    except Exception as e:
        if isinstance(e, (ValidationError, DataQualityError)):
            raise
        raise CalculationError(f"XOR function failed during financial validation: {str(e)}. Ensure all inputs are valid logical expressions for financial analysis.")


def IFERROR(run_context: Any, value: Any, value_if_error: Any) -> Any:
    """
    Return a specified value if a formula results in an error.

    Critical for robust financial calculations, ensuring graceful error handling
    in complex FP&A models where data quality issues or calculation errors can occur.

    Args:
        value: Value or calculation to test
        value_if_error: Value to return if error occurs

    Returns:
        Any: Original value or error replacement

    Raises:
        ValidationError: If inputs are invalid

    Financial Examples:
        # Safe division for financial ratios
        >>> revenue = 1000000
        >>> shares_outstanding = 0  # Could cause division by zero
        >>> eps = IFERROR(revenue / shares_outstanding, "N/A")
        >>> print(f"Earnings per share: {eps}")
        Earnings per share: N/A

        # Lookup with fallback for missing data
        >>> customer_id = "CUST999"
        >>> customer_data = {}  # Empty lookup table
        >>> customer_name = IFERROR(customer_data[customer_id], "Unknown Customer")
        >>> print(f"Customer: {customer_name}")
        Customer: Unknown Customer

        # Safe percentage calculation
        >>> actual = 105000
        >>> budget = 0  # Could cause division by zero
        >>> variance_pct = IFERROR((actual - budget) / budget * 100, 0)
        >>> print(f"Variance %: {variance_pct}")
        Variance %: 0

    Example:
        >>> IFERROR(10 / 2, "Error")
        5.0
        >>> IFERROR(10 / 0, "Division Error")
        'Division Error'
        >>> IFERROR("#DIV/0!", "Calculation Error")
        'Calculation Error'
    """
    try:
        # Check if value is already an error
        if _is_error_value(value):
            return value_if_error

        # If value is a callable (function), try to execute it
        if callable(value):
            try:
                result = value()
                if _is_error_value(result):
                    return value_if_error
                return result
            except Exception:
                return value_if_error

        # Return the value if no error
        return value

    except Exception:
        # If any error occurs during evaluation, return error value
        return value_if_error


def IFNA(run_context: Any, value: Any, value_if_na: Any) -> Any:
    """
    Return a specified value if a formula results in #N/A error.

    Essential for handling missing data scenarios in financial lookups,
    reference tables, and data integration where #N/A errors are common.

    Args:
        value: Value to test for #N/A error
        value_if_na: Value to return if #N/A error

    Returns:
        Any: Original value or #N/A replacement

    Raises:
        ValidationError: If inputs are invalid

    Financial Examples:
        # Product lookup with fallback
        >>> product_code = "PROD999"
        >>> product_price = "#N/A"  # Not found in lookup
        >>> price = IFNA(product_price, 0)
        >>> print(f"Product price: ${price}")
        Product price: $0

        # Customer credit rating lookup
        >>> customer_id = "NEW001"
        >>> credit_rating = "N/A"  # New customer, no rating yet
        >>> rating = IFNA(credit_rating, "Unrated")
        >>> print(f"Credit rating: {rating}")
        Credit rating: Unrated

        # Exchange rate lookup
        >>> currency_pair = "USD/XYZ"
        >>> exchange_rate = "#N/A"  # Currency not supported
        >>> rate = IFNA(exchange_rate, 1.0)
        >>> print(f"Exchange rate: {rate}")
        Exchange rate: 1.0

    Example:
        >>> IFNA("Valid Value", "N/A Replacement")
        'Valid Value'
        >>> IFNA("#N/A", "Not Available")
        'Not Available'
        >>> IFNA("N/A", "Missing Data")
        'Missing Data'
    """
    try:
        # Check if value is #N/A error
        if _is_na_value(value):
            return value_if_na

        # If value is a callable (function), try to execute it
        if callable(value):
            try:
                result = value()
                if _is_na_value(result):
                    return value_if_na
                return result
            except Exception:
                return value_if_na

        # Return the value if not #N/A
        return value

    except Exception:
        # If any error occurs during evaluation, return N/A value
        return value_if_na


def ISERROR(run_context: Any, value: Any) -> bool:
    """
    Test if value is an error.

    Essential for financial data validation, error detection in calculations,
    and implementing robust error handling workflows in FP&A systems.

    Args:
        value: Value to test for error condition

    Returns:
        bool: True if value is an error, False otherwise

    Financial Examples:
        # Validate calculation results
        >>> division_result = "#DIV/0!"
        >>> has_error = ISERROR(division_result)
        >>> print(f"Calculation has error: {has_error}")
        Calculation has error: True

        # Check lookup results
        >>> customer_data = "#N/A"
        >>> lookup_failed = ISERROR(customer_data)
        >>> print(f"Lookup failed: {lookup_failed}")
        Lookup failed: True

        # Validate financial ratios
        >>> pe_ratio = 15.5
        >>> ratio_error = ISERROR(pe_ratio)
        >>> print(f"P/E ratio is valid: {not ratio_error}")
        P/E ratio is valid: True

    Example:
        >>> ISERROR("#DIV/0!")
        True
        >>> ISERROR("#N/A")
        True
        >>> ISERROR(42)
        False
        >>> ISERROR("Valid Text")
        False
    """
    try:
        return _is_error_value(value)
    except Exception:
        return False


def ISBLANK(run_context: Any, value: Any) -> bool:
    """
    Test if cell is blank.

    Critical for financial data completeness validation, missing data detection,
    and ensuring data quality in FP&A reporting and analysis.

    Args:
        value: Value to test for blank condition

    Returns:
        bool: True if value is blank/null, False otherwise

    Financial Examples:
        # Check for missing financial data
        >>> quarterly_revenue = None
        >>> data_missing = ISBLANK(quarterly_revenue)
        >>> print(f"Revenue data missing: {data_missing}")
        Revenue data missing: True

        # Validate required fields
        >>> customer_name = ""
        >>> name_blank = ISBLANK(customer_name)
        >>> print(f"Customer name is blank: {name_blank}")
        Customer name is blank: True

        # Check budget allocations
        >>> department_budget = 150000
        >>> budget_assigned = not ISBLANK(department_budget)
        >>> print(f"Budget assigned: {budget_assigned}")
        Budget assigned: True

    Example:
        >>> ISBLANK(None)
        True
        >>> ISBLANK("")
        True
        >>> ISBLANK("   ")
        True
        >>> ISBLANK(0)
        False
        >>> ISBLANK("Text")
        False
    """
    try:
        if value is None:
            return True

        if isinstance(value, str):
            return value.strip() == ""

        # For other types, None is the only "blank" value
        return False

    except Exception:
        return False


def ISNUMBER(run_context: Any, value: Any) -> bool:
    """
    Test if value is a number.

    Essential for financial data validation, ensuring numeric calculations
    are performed on valid data types in FP&A systems.

    Args:
        value: Value to test for numeric type

    Returns:
        bool: True if value is numeric, False otherwise

    Financial Examples:
        # Validate financial inputs
        >>> revenue_input = "1000000"
        >>> is_valid_revenue = ISNUMBER(float(revenue_input)) if revenue_input.replace('.','').isdigit() else False
        >>> print(f"Revenue input is numeric: {is_valid_revenue}")
        Revenue input is numeric: True

        # Check calculation results
        >>> profit_margin = 0.15
        >>> is_valid_margin = ISNUMBER(profit_margin)
        >>> print(f"Profit margin is numeric: {is_valid_margin}")
        Profit margin is numeric: True

        # Validate user inputs
        >>> budget_amount = "Not a number"
        >>> is_valid_budget = ISNUMBER(budget_amount)
        >>> print(f"Budget amount is numeric: {is_valid_budget}")
        Budget amount is numeric: False

    Example:
        >>> ISNUMBER(42)
        True
        >>> ISNUMBER(3.14)
        True
        >>> ISNUMBER("123")
        False
        >>> ISNUMBER("Text")
        False
        >>> ISNUMBER(None)
        False
    """
    try:
        return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
    except Exception:
        return False


def ISTEXT(run_context: Any, value: Any) -> bool:
    """
    Test if value is text.

    Important for financial data categorization, text field validation,
    and ensuring proper data type handling in FP&A reporting systems.

    Args:
        value: Value to test for text type

    Returns:
        bool: True if value is text, False otherwise

    Financial Examples:
        # Validate text fields
        >>> department_name = "Finance"
        >>> is_text_field = ISTEXT(department_name)
        >>> print(f"Department name is text: {is_text_field}")
        Department name is text: True

        # Check account codes
        >>> account_code = "ACC-001"
        >>> is_text_code = ISTEXT(account_code)
        >>> print(f"Account code is text: {is_text_code}")
        Account code is text: True

        # Validate currency symbols
        >>> currency = "$"
        >>> is_text_symbol = ISTEXT(currency)
        >>> print(f"Currency symbol is text: {is_text_symbol}")
        Currency symbol is text: True

    Example:
        >>> ISTEXT("Hello")
        True
        >>> ISTEXT("123")
        True
        >>> ISTEXT(123)
        False
        >>> ISTEXT(None)
        False
        >>> ISTEXT(True)
        False
    """
    try:
        return isinstance(value, str)
    except Exception:
        return False


def SWITCH(run_context: Any, expression: Any, *values_and_results: Any, default: Optional[Any] = None) -> Any:
    """
    Compare expression against list of values and return corresponding result.

    Critical for financial categorization, multi-tier pricing models,
    and complex business rule implementation in FP&A systems.

    Args:
        expression: Expression to compare
        values_and_results: Alternating value and result pairs (value1, result1, value2, result2, ...)
        default: Default value if no matches found (optional)

    Returns:
        Any: Matched result or default value

    Raises:
        ValidationError: If arguments are not in pairs
        CalculationError: If no match found and no default provided

    Financial Examples:
        # Department budget allocation
        >>> department = "Marketing"
        >>> budget_multiplier = SWITCH(
        ...     department,
        ...     "Sales", 1.2,
        ...     "Marketing", 1.1,
        ...     "Operations", 1.0,
        ...     "HR", 0.8,
        ...     default=0.9
        ... )
        >>> print(f"Budget multiplier: {budget_multiplier}")
        Budget multiplier: 1.1

        # Credit rating to interest rate mapping
        >>> credit_rating = "AA"
        >>> interest_rate = SWITCH(
        ...     credit_rating,
        ...     "AAA", 0.025,
        ...     "AA", 0.030,
        ...     "A", 0.035,
        ...     "BBB", 0.045,
        ...     default=0.060
        ... )
        >>> print(f"Interest rate: {interest_rate:.1%}")
        Interest rate: 3.0%

        # Performance tier commission rates
        >>> performance_tier = "Gold"
        >>> commission_rate = SWITCH(
        ...     performance_tier,
        ...     "Platinum", 0.08,
        ...     "Gold", 0.06,
        ...     "Silver", 0.04,
        ...     "Bronze", 0.02,
        ...     default=0.01
        ... )
        >>> print(f"Commission rate: {commission_rate:.1%}")
        Commission rate: 6.0%

    Example:
        >>> SWITCH("B", "A", 1, "B", 2, "C", 3)
        2
        >>> SWITCH("D", "A", 1, "B", 2, "C", 3, default="Not Found")
        'Not Found'
        >>> SWITCH(2, 1, "One", 2, "Two", 3, "Three")
        'Two'
    """
    # Input validation
    if len(values_and_results) % 2 != 0:
        raise ValidationError("SWITCH requires an even number of value-result pairs")

    try:
        # Process value-result pairs
        for i in range(0, len(values_and_results), 2):
            value = values_and_results[i]
            result = values_and_results[i + 1]

            # Check for exact match
            if expression == value:
                return result

        # If no match found, return default or raise error
        if default is not None:
            return default
        else:
            raise CalculationError(f"No match found for expression '{expression}' in SWITCH")

    except Exception as e:
        if isinstance(e, (ValidationError, CalculationError)):
            raise
        raise CalculationError(f"SWITCH calculation failed: {str(e)}")
