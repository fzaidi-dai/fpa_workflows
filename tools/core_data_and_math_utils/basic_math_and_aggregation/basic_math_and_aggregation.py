"""
Basic Arithmetic & Aggregation Functions

These functions are the building blocks for financial summaries and aggregations.
"""

from typing import Any


def SUM(values: list[float] | Any) -> float:
    """
    Add up a range of numbers.

    Args:
        values: Array or range of numeric values

    Returns:
        Single numeric value (sum)

    Example:
        SUM([1, 2, 3, 4, 5])  # Returns 15
    """
    raise NotImplementedError("SUM function not yet implemented")


def AVERAGE(values: list[float] | Any) -> float:
    """
    Calculate the mean of a dataset.

    Args:
        values: Array or range of numeric values

    Returns:
        Single numeric value (mean)

    Example:
        AVERAGE([10, 20, 30])  # Returns 20.0
    """
    raise NotImplementedError("AVERAGE function not yet implemented")


def MIN(values: list[float] | Any) -> float:
    """
    Identify the smallest number in a dataset.

    Args:
        values: Array or range of numeric values

    Returns:
        Single numeric value (minimum)

    Example:
        MIN([10, 5, 20, 3])  # Returns 3
    """
    raise NotImplementedError("MIN function not yet implemented")


def MAX(values: list[float] | Any) -> float:
    """
    Identify the largest number in a dataset.

    Args:
        values: Array or range of numeric values

    Returns:
        Single numeric value (maximum)

    Example:
        MAX([10, 5, 20, 3])  # Returns 20
    """
    raise NotImplementedError("MAX function not yet implemented")


def PRODUCT(values: list[float] | Any) -> float:
    """
    Multiply values together.

    Args:
        values: Array or range of numeric values

    Returns:
        Single numeric value (product)

    Example:
        PRODUCT([2, 3, 4])  # Returns 24
    """
    raise NotImplementedError("PRODUCT function not yet implemented")


def MEDIAN(values: list[float] | Any) -> float:
    """
    Calculate the middle value of a dataset.

    Args:
        values: Series/array of numbers

    Returns:
        Float

    Example:
        MEDIAN([1, 2, 3, 4, 5])  # Returns 3.0
    """
    raise NotImplementedError("MEDIAN function not yet implemented")


def MODE(values: list[float] | Any) -> float | list[float]:
    """
    Find the most frequently occurring value.

    Args:
        values: Series/array of numbers

    Returns:
        Float or list of floats

    Example:
        MODE([1, 2, 2, 3, 3, 3])  # Returns 3.0
    """
    raise NotImplementedError("MODE function not yet implemented")


def PERCENTILE(values: list[float] | Any, percentile_value: float) -> float:
    """
    Calculate specific percentiles (e.g., 25th, 75th percentile).

    Args:
        values: Series/array of numbers
        percentile_value: Percentile value (0-1)

    Returns:
        Float

    Example:
        PERCENTILE([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.75)  # Returns 7.75
    """
    raise NotImplementedError("PERCENTILE function not yet implemented")


def POWER(number: float, power: float) -> float:
    """
    Raise number to a power.

    Args:
        number: Base number
        power: Exponent

    Returns:
        Float result of number^power

    Example:
        POWER(1.05, 10)  # Returns 1.05^10
    """
    raise NotImplementedError("POWER function not yet implemented")


def SQRT(number: float) -> float:
    """
    Calculate square root.

    Args:
        number: Number to calculate square root of

    Returns:
        Float

    Example:
        SQRT(25)  # Returns 5.0
    """
    raise NotImplementedError("SQRT function not yet implemented")


def EXP(number: float) -> float:
    """
    Calculate e^x.

    Args:
        number: Exponent

    Returns:
        Float

    Example:
        EXP(1)  # Returns e^1 ≈ 2.718
    """
    raise NotImplementedError("EXP function not yet implemented")


def LN(number: float) -> float:
    """
    Calculate natural logarithm.

    Args:
        number: Number to calculate natural log of

    Returns:
        Float

    Example:
        LN(2.718)  # Returns ≈ 1.0
    """
    raise NotImplementedError("LN function not yet implemented")


def LOG(number: float, base: float | None = None) -> float:
    """
    Calculate logarithm with specified base.

    Args:
        number: Number to calculate log of
        base: Base of logarithm (optional, defaults to 10)

    Returns:
        Float

    Example:
        LOG(100, 10)  # Returns 2.0
    """
    raise NotImplementedError("LOG function not yet implemented")


def ABS(number: float) -> float:
    """
    Calculate absolute value.

    Args:
        number: Number to calculate absolute value of

    Returns:
        Float

    Example:
        ABS(-10)  # Returns 10.0
    """
    raise NotImplementedError("ABS function not yet implemented")


def SIGN(number: float) -> int:
    """
    Return sign of number (-1, 0, or 1).

    Args:
        number: Number to get sign of

    Returns:
        Integer (-1, 0, or 1)

    Example:
        SIGN(-15)  # Returns -1
    """
    raise NotImplementedError("SIGN function not yet implemented")


def MOD(number: float, divisor: float) -> float:
    """
    Calculate remainder after division.

    Args:
        number: Dividend
        divisor: Divisor

    Returns:
        Float

    Example:
        MOD(23, 5)  # Returns 3.0
    """
    raise NotImplementedError("MOD function not yet implemented")


def ROUND(number: float, num_digits: int) -> float:
    """
    Round number to specified digits.

    Args:
        number: Number to round
        num_digits: Number of decimal places

    Returns:
        Float

    Example:
        ROUND(3.14159, 2)  # Returns 3.14
    """
    raise NotImplementedError("ROUND function not yet implemented")


def ROUNDUP(number: float, num_digits: int) -> float:
    """
    Round number up.

    Args:
        number: Number to round up
        num_digits: Number of decimal places

    Returns:
        Float

    Example:
        ROUNDUP(3.14159, 2)  # Returns 3.15
    """
    raise NotImplementedError("ROUNDUP function not yet implemented")


def ROUNDDOWN(number: float, num_digits: int) -> float:
    """
    Round number down.

    Args:
        number: Number to round down
        num_digits: Number of decimal places

    Returns:
        Float

    Example:
        ROUNDDOWN(3.14159, 2)  # Returns 3.14
    """
    raise NotImplementedError("ROUNDDOWN function not yet implemented")


def WEIGHTED_AVERAGE(values: list[float], weights: list[float]) -> float:
    """
    Calculate weighted average of values.

    Args:
        values: Array of values
        weights: Array of weights

    Returns:
        Float weighted average

    Example:
        WEIGHTED_AVERAGE([100, 200, 300], [0.2, 0.3, 0.5])  # Returns 230.0
    """
    raise NotImplementedError("WEIGHTED_AVERAGE function not yet implemented")


def GEOMETRIC_MEAN(values: list[float] | Any) -> float:
    """
    Calculate geometric mean (useful for growth rates).

    Args:
        values: Series/array of positive numbers

    Returns:
        Float

    Example:
        GEOMETRIC_MEAN([1.05, 1.08, 1.12, 1.03])  # Returns geometric mean
    """
    raise NotImplementedError("GEOMETRIC_MEAN function not yet implemented")


def HARMONIC_MEAN(values: list[float] | Any) -> float:
    """
    Calculate harmonic mean (useful for rates/ratios).

    Args:
        values: Series/array of positive numbers

    Returns:
        Float

    Example:
        HARMONIC_MEAN([2, 4, 8])  # Returns harmonic mean
    """
    raise NotImplementedError("HARMONIC_MEAN function not yet implemented")


def CUMSUM(values: list[float] | Any) -> list[float]:
    """
    Calculate cumulative sum.

    Args:
        values: Series/array of numbers

    Returns:
        Array of cumulative sums

    Example:
        CUMSUM([10, 20, 30, 40])  # Returns [10, 30, 60, 100]
    """
    raise NotImplementedError("CUMSUM function not yet implemented")


def CUMPROD(values: list[float] | Any) -> list[float]:
    """
    Calculate cumulative product.

    Args:
        values: Series/array of numbers

    Returns:
        Array of cumulative products

    Example:
        CUMPROD([1.05, 1.08, 1.12])  # Returns [1.05, 1.134, 1.269]
    """
    raise NotImplementedError("CUMPROD function not yet implemented")


def VARIANCE_WEIGHTED(values: list[float], weights: list[float]) -> float:
    """
    Calculate weighted variance.

    Args:
        values: Array of values
        weights: Array of weights

    Returns:
        Float

    Example:
        VARIANCE_WEIGHTED([100, 200, 300], [0.2, 0.3, 0.5])  # Returns weighted variance
    """
    raise NotImplementedError("VARIANCE_WEIGHTED function not yet implemented")
