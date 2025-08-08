"""
Standalone math functions for MCP server to avoid circular imports.
This file contains copies of the essential functions from basic_math_and_aggregation.py
without any dependencies on the tools package.
"""

import polars as pl
from pathlib import Path
from typing import Union, List, Optional, Any
from decimal import Decimal, ROUND_HALF_UP
import math
import statistics
from collections import Counter


def SUM(ctx, values: Union[List, str, Path], **kwargs) -> Union[Decimal, float]:
    """Calculate the sum of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    return sum(decimal_values)


def AVERAGE(ctx, values: Union[List, str, Path], **kwargs) -> Union[Decimal, float]:
    """Calculate the average of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to calculate average")

    return sum(decimal_values) / len(decimal_values)


def MIN(ctx, values: Union[List, str, Path], **kwargs) -> Union[Decimal, float]:
    """Find the minimum value."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to find minimum")

    return min(decimal_values)


def MAX(ctx, values: Union[List, str, Path], **kwargs) -> Union[Decimal, float]:
    """Find the maximum value."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to find maximum")

    return max(decimal_values)


def PRODUCT(ctx, values: Union[List, str, Path], **kwargs) -> Union[Decimal, float]:
    """Calculate the product of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to calculate product")

    result = Decimal('1')
    for v in decimal_values:
        result *= v
    return result


def MEDIAN(ctx, values: Union[List, str, Path], **kwargs) -> Union[Decimal, float]:
    """Calculate the median of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Convert to float for statistics module
    float_values = [float(v) for v in values if v is not None]
    if not float_values:
        raise ValueError("No valid values to calculate median")

    return Decimal(str(statistics.median(float_values)))


def MODE(ctx, values: Union[List, str, Path], **kwargs) -> Union[List, Any]:
    """Find the mode(s) of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Filter out None values
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        raise ValueError("No valid values to find mode")

    # Count frequencies
    counter = Counter(clean_values)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]

    return modes if len(modes) > 1 else modes[0]


def PERCENTILE(ctx, values: Union[List, str, Path], percentile_value: float, **kwargs) -> Union[Decimal, float]:
    """Calculate the percentile of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Convert to float for statistics module
    float_values = [float(v) for v in values if v is not None]
    if not float_values:
        raise ValueError("No valid values to calculate percentile")

    # Calculate percentile
    float_values.sort()
    k = (len(float_values) - 1) * (percentile_value / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        result = float_values[int(k)]
    else:
        result = float_values[int(f)] * (c - k) + float_values[int(c)] * (k - f)

    return Decimal(str(result))


def POWER(ctx, values: Union[List, str, Path], power: float, output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Raise values to a power."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate power
    result_values = [float(v) ** power for v in values if v is not None]

    if output_filename:
        # Save to file
        result_df = pl.DataFrame({"result": result_values})
        output_path = Path(ctx.deps.thread_dir) / "analysis" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        return output_path
    else:
        return result_values


def SQRT(ctx, values: Union[List, str, Path], output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Calculate square root of values."""
    return POWER(ctx, values, 0.5, output_filename, **kwargs)


def EXP(ctx, values: Union[List, str, Path], output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Calculate exponential of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate exponential
    result_values = [math.exp(float(v)) for v in values if v is not None]

    if output_filename:
        # Save to file
        result_df = pl.DataFrame({"result": result_values})
        output_path = Path(ctx.deps.thread_dir) / "analysis" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        return output_path
    else:
        return result_values


def LN(ctx, values: Union[List, str, Path], output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Calculate natural logarithm of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate natural logarithm
    result_values = [math.log(float(v)) for v in values if v is not None and float(v) > 0]

    if output_filename:
        # Save to file
        result_df = pl.DataFrame({"result": result_values})
        output_path = Path(ctx.deps.thread_dir) / "analysis" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        return output_path
    else:
        return result_values


def LOG(ctx, values: Union[List, str, Path], base: Optional[float] = None, output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Calculate logarithm of values with specified base."""
    if base is None:
        base = 10

    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate logarithm
    result_values = [math.log(float(v), base) for v in values if v is not None and float(v) > 0]

    if output_filename:
        # Save to file
        result_df = pl.DataFrame({"result": result_values})
        output_path = Path(ctx.deps.thread_dir) / "analysis" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        return output_path
    else:
        return result_values


def ABS(ctx, values: Union[List, str, Path], output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Calculate absolute value of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate absolute value
    result_values = [abs(float(v)) for v in values if v is not None]

    if output_filename:
        # Save to file
        result_df = pl.DataFrame({"result": result_values})
        output_path = Path(ctx.deps.thread_dir) / "analysis" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        return output_path
    else:
        return result_values


def SIGN(ctx, values: Union[List, str, Path], **kwargs) -> List[int]:
    """Calculate sign of values."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate sign
    result_values = []
    for v in values:
        if v is not None:
            if float(v) > 0:
                result_values.append(1)
            elif float(v) < 0:
                result_values.append(-1)
            else:
                result_values.append(0)

    return result_values


def MOD(ctx, dividends: Union[List, str, Path], divisors: Union[List, str, Path], **kwargs) -> List[float]:
    """Calculate modulo of dividends by divisors."""
    # Handle file inputs
    if isinstance(dividends, (str, Path)):
        if str(dividends).endswith('.parquet'):
            df = pl.read_parquet(dividends)
        else:
            df = pl.read_csv(dividends)

        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in dividends file")

        dividends = df[numeric_cols[0]].to_list()

    if isinstance(divisors, (str, Path)):
        if str(divisors).endswith('.parquet'):
            df = pl.read_parquet(divisors)
        else:
            df = pl.read_csv(divisors)

        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in divisors file")

        divisors = df[numeric_cols[0]].to_list()

    # Calculate modulo
    result_values = []
    for i, dividend in enumerate(dividends):
        if dividend is not None:
            divisor = divisors[i % len(divisors)] if isinstance(divisors, list) else divisors
            if divisor is not None and divisor != 0:
                result_values.append(float(dividend) % float(divisor))

    return result_values


def ROUND(ctx, values: Union[List, str, Path], num_digits: int, **kwargs) -> List[Decimal]:
    """Round values to specified number of digits."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Round values
    result_values = []
    for v in values:
        if v is not None:
            decimal_v = Decimal(str(v))
            rounded = decimal_v.quantize(Decimal('0.1') ** num_digits, rounding=ROUND_HALF_UP)
            result_values.append(rounded)

    return result_values


def ROUNDUP(ctx, values: Union[List, str, Path], num_digits: int, **kwargs) -> List[Decimal]:
    """Round values up to specified number of digits."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Round up values
    result_values = []
    for v in values:
        if v is not None:
            decimal_v = Decimal(str(v))
            # Round up by adding a small amount and then rounding
            factor = Decimal('0.1') ** num_digits
            rounded = (decimal_v + factor - Decimal('0.0000000001')).quantize(factor, rounding=ROUND_HALF_UP)
            result_values.append(rounded)

    return result_values


def ROUNDDOWN(ctx, values: Union[List, str, Path], num_digits: int, **kwargs) -> List[Decimal]:
    """Round values down to specified number of digits."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Round down values
    result_values = []
    for v in values:
        if v is not None:
            decimal_v = Decimal(str(v))
            # Round down by subtracting a small amount and then rounding
            factor = Decimal('0.1') ** num_digits
            rounded = (decimal_v - Decimal('0.0000000001')).quantize(factor, rounding=ROUND_HALF_UP)
            result_values.append(rounded)

    return result_values


def WEIGHTED_AVERAGE(ctx, values: Union[List, str, Path], weights: Union[List, str, Path], **kwargs) -> Decimal:
    """Calculate weighted average."""
    # Handle file inputs
    if isinstance(values, (str, Path)):
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in values file")

        values = df[numeric_cols[0]].to_list()

    if isinstance(weights, (str, Path)):
        if str(weights).endswith('.parquet'):
            df = pl.read_parquet(weights)
        else:
            df = pl.read_csv(weights)

        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in weights file")

        weights = df[numeric_cols[0]].to_list()

    # Calculate weighted average
    weighted_sum = Decimal('0')
    weight_sum = Decimal('0')

    for i, value in enumerate(values):
        if value is not None and i < len(weights) and weights[i] is not None:
            weight = Decimal(str(weights[i]))
            weighted_sum += Decimal(str(value)) * weight
            weight_sum += weight

    if weight_sum == 0:
        raise ValueError("Sum of weights is zero")

    return weighted_sum / weight_sum


def GEOMETRIC_MEAN(ctx, values: Union[List, str, Path], **kwargs) -> Decimal:
    """Calculate geometric mean."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate geometric mean
    clean_values = [float(v) for v in values if v is not None and float(v) > 0]
    if not clean_values:
        raise ValueError("No valid positive values for geometric mean")

    product = 1
    for v in clean_values:
        product *= v

    result = product ** (1.0 / len(clean_values))
    return Decimal(str(result))


def HARMONIC_MEAN(ctx, values: Union[List, str, Path], **kwargs) -> Decimal:
    """Calculate harmonic mean."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate harmonic mean
    clean_values = [float(v) for v in values if v is not None and float(v) != 0]
    if not clean_values:
        raise ValueError("No valid non-zero values for harmonic mean")

    reciprocal_sum = sum(1.0 / v for v in clean_values)
    result = len(clean_values) / reciprocal_sum
    return Decimal(str(result))


def CUMSUM(ctx, values: Union[List, str, Path], output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Calculate cumulative sum."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate cumulative sum
    cumsum_values = []
    running_sum = Decimal('0')

    for v in values:
        if v is not None:
            running_sum += Decimal(str(v))
            cumsum_values.append(float(running_sum))

    if output_filename:
        # Save to file
        result_df = pl.DataFrame({"cumsum": cumsum_values})
        output_path = Path(ctx.deps.thread_dir) / "analysis" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        return output_path
    else:
        return cumsum_values


def CUMPROD(ctx, values: Union[List, str, Path], output_filename: Optional[str] = None, **kwargs) -> Union[List, Path]:
    """Calculate cumulative product."""
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        # Get first numeric column
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in file")

        values = df[numeric_cols[0]].to_list()

    # Calculate cumulative product
    cumprod_values = []
    running_product = Decimal('1')

    for v in values:
        if v is not None:
            running_product *= Decimal(str(v))
            cumprod_values.append(float(running_product))

    if output_filename:
        # Save to file
        result_df = pl.DataFrame({"cumprod": cumprod_values})
        output_path = Path(ctx.deps.thread_dir) / "analysis" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        return output_path
    else:
        return cumprod_values


def VARIANCE_WEIGHTED(ctx, values: Union[List, str, Path], weights: Union[List, str, Path], **kwargs) -> Decimal:
    """Calculate weighted variance."""
    # Handle file inputs
    if isinstance(values, (str, Path)):
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)

        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in values file")

        values = df[numeric_cols[0]].to_list()

    if isinstance(weights, (str, Path)):
        if str(weights).endswith('.parquet'):
            df = pl.read_parquet(weights)
        else:
            df = pl.read_csv(weights)

        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if not numeric_cols:
            raise ValueError("No numeric columns found in weights file")

        weights = df[numeric_cols[0]].to_list()

    # Calculate weighted variance
    weighted_mean = WEIGHTED_AVERAGE(ctx, values, weights)

    weighted_sum_sq_diff = Decimal('0')
    weight_sum = Decimal('0')

    for i, value in enumerate(values):
        if value is not None and i < len(weights) and weights[i] is not None:
            weight = Decimal(str(weights[i]))
            diff = Decimal(str(value)) - weighted_mean
            weighted_sum_sq_diff += weight * (diff ** 2)
            weight_sum += weight

    if weight_sum == 0:
        raise ValueError("Sum of weights is zero")

    return weighted_sum_sq_diff / weight_sum
