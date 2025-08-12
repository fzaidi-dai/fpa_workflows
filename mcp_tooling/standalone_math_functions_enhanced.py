"""
Enhanced standalone math functions with range parameter support.
This file contains enhanced versions of the math functions that support
range specifications for more flexible data manipulation.
"""

import polars as pl
from pathlib import Path
from typing import Union, List, Optional, Any, Dict
from decimal import Decimal, ROUND_HALF_UP
import math
import statistics
from collections import Counter
import sys
import os

# Add the google_sheets/api directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'google_sheets', 'api'))
from polars_range_resolver import PolarsRangeResolver


def apply_range_to_dataframe(
    df: pl.DataFrame,
    range_spec: Optional[Union[str, Dict[str, Any]]] = None,
    column: Optional[str] = None
) -> Union[pl.DataFrame, List]:
    """
    Apply range specification to a DataFrame and optionally extract column values.
    
    Args:
        df: Input DataFrame
        range_spec: Range specification (A1 notation or dict)
        column: Optional column name to extract values from
        
    Returns:
        Sliced DataFrame or list of values from specified column
    """
    # Apply range if specified
    if range_spec:
        df = PolarsRangeResolver.resolve_range(df, range_spec)
    
    # Extract column values if specified
    if column:
        if column in df.columns:
            return df[column].to_list()
        else:
            # Try to get first numeric column
            numeric_cols = [col for col in df.columns 
                          if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if numeric_cols:
                return df[numeric_cols[0]].to_list()
            else:
                raise ValueError(f"Column '{column}' not found and no numeric columns available")
    
    return df


def SUM(ctx, values: Union[List, str, Path], 
        range_spec: Optional[Union[str, Dict[str, Any]]] = None,
        column: Optional[str] = None,
        **kwargs) -> Union[Decimal, float]:
    """
    Calculate the sum of values with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Sum of the values
    """
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, sum all numeric columns
            numeric_cols = [col for col in result.columns 
                          if result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if not numeric_cols:
                raise ValueError("No numeric columns found in specified range")
            values = []
            for col in numeric_cols:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    return sum(decimal_values)


def AVERAGE(ctx, values: Union[List, str, Path],
           range_spec: Optional[Union[str, Dict[str, Any]]] = None,
           column: Optional[str] = None,
           **kwargs) -> Union[Decimal, float]:
    """
    Calculate the average of values with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Average of the values
    """
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, average all numeric columns
            numeric_cols = [col for col in result.columns 
                          if result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if not numeric_cols:
                raise ValueError("No numeric columns found in specified range")
            values = []
            for col in numeric_cols:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to calculate average")
    
    return sum(decimal_values) / len(decimal_values)


def MIN(ctx, values: Union[List, str, Path],
       range_spec: Optional[Union[str, Dict[str, Any]]] = None,
       column: Optional[str] = None,
       **kwargs) -> Union[Decimal, float]:
    """
    Find the minimum value with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Minimum value
    """
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, find min across all numeric columns
            numeric_cols = [col for col in result.columns 
                          if result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if not numeric_cols:
                raise ValueError("No numeric columns found in specified range")
            values = []
            for col in numeric_cols:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to find minimum")
    
    return min(decimal_values)


def MAX(ctx, values: Union[List, str, Path],
       range_spec: Optional[Union[str, Dict[str, Any]]] = None,
       column: Optional[str] = None,
       **kwargs) -> Union[Decimal, float]:
    """
    Find the maximum value with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Maximum value
    """
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, find max across all numeric columns
            numeric_cols = [col for col in result.columns 
                          if result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if not numeric_cols:
                raise ValueError("No numeric columns found in specified range")
            values = []
            for col in numeric_cols:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to find maximum")
    
    return max(decimal_values)


def PRODUCT(ctx, values: Union[List, str, Path],
           range_spec: Optional[Union[str, Dict[str, Any]]] = None,
           column: Optional[str] = None,
           **kwargs) -> Union[Decimal, float]:
    """
    Calculate the product of values with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Product of the values
    """
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, multiply all numeric columns
            numeric_cols = [col for col in result.columns 
                          if result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if not numeric_cols:
                raise ValueError("No numeric columns found in specified range")
            values = []
            for col in numeric_cols:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Convert to Decimal for precision
    decimal_values = [Decimal(str(v)) for v in values if v is not None]
    if not decimal_values:
        raise ValueError("No valid values to calculate product")
    
    product = Decimal(1)
    for val in decimal_values:
        product *= val
    return product


def MEDIAN(ctx, values: Union[List, str, Path],
          range_spec: Optional[Union[str, Dict[str, Any]]] = None,
          column: Optional[str] = None,
          **kwargs) -> Union[Decimal, float]:
    """
    Calculate the median of values with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Median of the values
    """
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, get all numeric values
            numeric_cols = [col for col in result.columns 
                          if result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if not numeric_cols:
                raise ValueError("No numeric columns found in specified range")
            values = []
            for col in numeric_cols:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        raise ValueError("No valid values to calculate median")
    
    return Decimal(str(statistics.median(valid_values)))


def MODE(ctx, values: Union[List, str, Path],
        range_spec: Optional[Union[str, Dict[str, Any]]] = None,
        column: Optional[str] = None,
        **kwargs) -> Any:
    """
    Find the mode (most common value) with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Mode value(s)
    """
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, get all values
            values = []
            for col in result.columns:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        raise ValueError("No valid values to find mode")
    
    counter = Counter(valid_values)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]
    
    return modes[0] if len(modes) == 1 else modes


def PERCENTILE(ctx, values: Union[List, str, Path], 
              percentile: Union[float, int],
              range_spec: Optional[Union[str, Dict[str, Any]]] = None,
              column: Optional[str] = None,
              **kwargs) -> Union[Decimal, float]:
    """
    Calculate the percentile of values with optional range specification.
    
    Args:
        ctx: Context object
        values: Input values (list, file path, or DataFrame path)
        percentile: Percentile to calculate (0-100)
        range_spec: Optional range specification (e.g., "A1:C10", "B:B")
        column: Optional column name for aggregation
        **kwargs: Additional arguments
        
    Returns:
        Value at the specified percentile
    """
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    if isinstance(values, (str, Path)):
        # Handle file input
        if str(values).endswith('.parquet'):
            df = pl.read_parquet(values)
        else:
            df = pl.read_csv(values)
        
        # Apply range and extract values
        result = apply_range_to_dataframe(df, range_spec, column)
        if isinstance(result, pl.DataFrame):
            # No specific column, get all numeric values
            numeric_cols = [col for col in result.columns 
                          if result[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            if not numeric_cols:
                raise ValueError("No numeric columns found in specified range")
            values = []
            for col in numeric_cols:
                values.extend(result[col].to_list())
        else:
            values = result
    
    # Filter out None values
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        raise ValueError("No valid values to calculate percentile")
    
    sorted_values = sorted(valid_values)
    k = (len(sorted_values) - 1) * percentile / 100
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return Decimal(str(sorted_values[int(k)]))
    
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return Decimal(str(d0 + d1))


# Simple context classes
class SimpleFinnDeps:
    def __init__(self, thread_dir: Path, workspace_dir: Path):
        self.thread_dir = thread_dir
        self.workspace_dir = workspace_dir

class SimpleRunContext:
    def __init__(self, deps: SimpleFinnDeps):
        self.deps = deps

# Export remaining functions from original module
from standalone_math_functions import (
    POWER, SQRT, EXP, LN, LOG, ABS, SIGN, MOD, ROUND, ROUNDUP, ROUNDDOWN,
    WEIGHTED_AVERAGE, GEOMETRIC_MEAN, HARMONIC_MEAN, CUMSUM, CUMPROD, VARIANCE_WEIGHTED
)