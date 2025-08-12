"""
Google Sheets Compatible Functions for Polars

This module provides Polars implementations that exactly match Google Sheets functions,
enabling seamless translation between local computation and sheet formulas.
All functions support range specifications in A1 notation.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Any, Dict, Tuple
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date, timedelta
import math
import statistics
from collections import Counter
import sys
import os

# Add the google_sheets/api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'google_sheets', 'api'))
from polars_range_resolver import PolarsRangeResolver


class SheetsCompatibleFunctions:
    """Google Sheets compatible functions implemented with Polars"""
    
    def __init__(self):
        self.resolver = PolarsRangeResolver()
    
    # ================== MATH FUNCTIONS ==================
    
    def SUM(self, 
            data: Union[pl.DataFrame, str, Path],
            range_spec: Optional[str] = None) -> float:
        """
        SUM function matching Google Sheets.
        Formula: =SUM(range)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Sum all numeric values in the range
        total = 0
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                total += df[col].sum()
        return float(total) if total is not None else 0
    
    def AVERAGE(self,
                data: Union[pl.DataFrame, str, Path],
                range_spec: Optional[str] = None) -> float:
        """
        AVERAGE function matching Google Sheets.
        Formula: =AVERAGE(range)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Collect all numeric values
        values = []
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values.extend(df[col].drop_nulls().to_list())
        
        return sum(values) / len(values) if values else 0
    
    def COUNT(self,
              data: Union[pl.DataFrame, str, Path],
              range_spec: Optional[str] = None,
              column: Optional[str] = None) -> int:
        """
        COUNT function matching Google Sheets.
        Formula: =COUNT(range)
        Counts cells containing numeric values.
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return df[column].drop_nulls().len()
            else:
                return 0  # Non-numeric columns don't count
        
        count = 0
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                count += df[col].drop_nulls().len()
        return count
    
    def COUNTA(self,
               data: Union[pl.DataFrame, str, Path],
               range_spec: Optional[str] = None,
               column: Optional[str] = None) -> int:
        """
        COUNTA function matching Google Sheets.
        Formula: =COUNTA(range)
        Counts non-empty cells.
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            return df[column].drop_nulls().len()
        
        count = 0
        for col in df.columns:
            count += df[col].drop_nulls().len()
        return count
    
    def MIN(self,
            data: Union[pl.DataFrame, str, Path],
            range_spec: Optional[str] = None) -> float:
        """
        MIN function matching Google Sheets.
        Formula: =MIN(range)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        values = []
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                col_min = df[col].min()
                if col_min is not None:
                    values.append(col_min)
        
        return float(min(values)) if values else 0
    
    def MAX(self,
            data: Union[pl.DataFrame, str, Path],
            range_spec: Optional[str] = None) -> float:
        """
        MAX function matching Google Sheets.
        Formula: =MAX(range)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        values = []
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                col_max = df[col].max()
                if col_max is not None:
                    values.append(col_max)
        
        return float(max(values)) if values else 0
    
    def PRODUCT(self,
                data: Union[pl.DataFrame, str, Path],
                range_spec: Optional[str] = None,
                column: Optional[str] = None) -> float:
        """
        PRODUCT function matching Google Sheets.
        Formula: =PRODUCT(range)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        if not values:
            return 0
        
        product = 1
        for val in values:
            product *= float(val)
        return product
    
    def MEDIAN(self,
               data: Union[pl.DataFrame, str, Path],
               range_spec: Optional[str] = None,
               column: Optional[str] = None) -> float:
        """
        MEDIAN function matching Google Sheets.
        Formula: =MEDIAN(range)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        if not values:
            return 0
        
        import statistics
        return float(statistics.median(values))
    
    def MODE(self,
             data: Union[pl.DataFrame, str, Path],
             range_spec: Optional[str] = None,
             column: Optional[str] = None) -> Any:
        """
        MODE function matching Google Sheets.
        Returns the most frequently occurring value.
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            values = df[column].drop_nulls().to_list()
        else:
            # Collect all values
            values = []
            for col in df.columns:
                values.extend(df[col].drop_nulls().to_list())
        
        if not values:
            return None
        
        from collections import Counter
        counter = Counter(values)
        max_count = max(counter.values())
        modes = [k for k, v in counter.items() if v == max_count]
        
        return modes[0] if len(modes) == 1 else modes
    
    def STDEV(self,
              data: Union[pl.DataFrame, str, Path],
              range_spec: Optional[str] = None,
              column: Optional[str] = None) -> float:
        """
        STDEV function matching Google Sheets.
        Formula: =STDEV(range)
        Calculates the sample standard deviation.
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                # Use Polars native std() with ddof=1 for sample standard deviation
                result = df[column].drop_nulls().std(ddof=1)
                return float(result) if result is not None else 0.0
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
            
            if not values:
                return 0.0
            
            # Calculate standard deviation using Polars
            temp_df = pl.DataFrame({"values": values})
            result = temp_df["values"].std(ddof=1)
            return float(result) if result is not None else 0.0
    
    def VAR(self,
            data: Union[pl.DataFrame, str, Path],
            range_spec: Optional[str] = None,
            column: Optional[str] = None) -> float:
        """
        VAR function matching Google Sheets.
        Formula: =VAR(range)
        Calculates the sample variance.
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                # Use Polars native var() with ddof=1 for sample variance
                result = df[column].drop_nulls().var(ddof=1)
                return float(result) if result is not None else 0.0
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
            
            if not values:
                return 0.0
            
            # Calculate variance using Polars
            temp_df = pl.DataFrame({"values": values})
            result = temp_df["values"].var(ddof=1)
            return float(result) if result is not None else 0.0
    
    def PERCENTILE(self,
                   data: Union[pl.DataFrame, str, Path],
                   percentile: Union[float, int],
                   range_spec: Optional[str] = None,
                   column: Optional[str] = None) -> float:
        """
        PERCENTILE function matching Google Sheets.
        Formula: =PERCENTILE(range, percentile)
        """
        if not 0 <= percentile <= 1:
            # Convert from 0-100 scale to 0-1 scale if needed
            if 1 < percentile <= 100:
                percentile = percentile / 100
            else:
                raise ValueError("Percentile must be between 0 and 1 (or 0 and 100)")
        
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        if not values:
            return 0
        
        values.sort()
        k = (len(values) - 1) * percentile
        f = int(math.floor(k))
        c = int(math.ceil(k))
        
        if f == c:
            return float(values[f])
        else:
            d0 = values[f] * (c - k)
            d1 = values[c] * (k - f)
            return float(d0 + d1)
    
    # ================== ENHANCED MATH FUNCTIONS ==================
    
    def POWER(self, 
              data: Union[pl.DataFrame, str, Path],
              power: float,
              range_spec: Optional[str] = None,
              column: Optional[str] = None) -> List[float]:
        """
        POWER function matching Google Sheets.
        Formula: =POWER(number, power) or array version
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        return [float(val) ** power for val in values]
    
    def SQRT(self, 
             data: Union[pl.DataFrame, str, Path],
             range_spec: Optional[str] = None,
             column: Optional[str] = None) -> List[float]:
        """
        SQRT function matching Google Sheets.
        Formula: =SQRT(number)
        """
        return self.POWER(data, 0.5, range_spec, column)
    
    def EXP(self, 
            data: Union[pl.DataFrame, str, Path],
            range_spec: Optional[str] = None,
            column: Optional[str] = None) -> List[float]:
        """
        EXP function matching Google Sheets.
        Formula: =EXP(number)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        return [math.exp(float(val)) for val in values]
    
    def LN(self, 
           data: Union[pl.DataFrame, str, Path],
           range_spec: Optional[str] = None,
           column: Optional[str] = None) -> List[float]:
        """
        LN function matching Google Sheets.
        Formula: =LN(number)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        return [math.log(float(val)) for val in values if float(val) > 0]
    
    def LOG(self, 
            data: Union[pl.DataFrame, str, Path],
            base: float = 10,
            range_spec: Optional[str] = None,
            column: Optional[str] = None) -> List[float]:
        """
        LOG function matching Google Sheets.
        Formula: =LOG(number, [base])
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        return [math.log(float(val), base) for val in values if float(val) > 0]
    
    def ABS(self, 
            data: Union[pl.DataFrame, str, Path],
            range_spec: Optional[str] = None,
            column: Optional[str] = None) -> List[float]:
        """
        ABS function matching Google Sheets.
        Formula: =ABS(number)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        return [abs(float(val)) for val in values]
    
    def SIGN(self, 
             data: Union[pl.DataFrame, str, Path],
             range_spec: Optional[str] = None,
             column: Optional[str] = None) -> List[int]:
        """
        SIGN function matching Google Sheets.
        Formula: =SIGN(number)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        result = []
        for val in values:
            fval = float(val)
            if fval > 0:
                result.append(1)
            elif fval < 0:
                result.append(-1)
            else:
                result.append(0)
        return result
    
    def ROUND(self, 
              data: Union[pl.DataFrame, str, Path],
              digits: int = 0,
              range_spec: Optional[str] = None,
              column: Optional[str] = None) -> List[float]:
        """
        ROUND function matching Google Sheets.
        Formula: =ROUND(number, digits)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        return [round(float(val), digits) for val in values]
    
    def ROUNDUP(self, 
                data: Union[pl.DataFrame, str, Path],
                digits: int = 0,
                range_spec: Optional[str] = None,
                column: Optional[str] = None) -> List[float]:
        """
        ROUNDUP function matching Google Sheets.
        Formula: =ROUNDUP(number, digits)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        result = []
        factor = 10 ** digits
        for val in values:
            result.append(math.ceil(float(val) * factor) / factor)
        return result
    
    def ROUNDDOWN(self, 
                  data: Union[pl.DataFrame, str, Path],
                  digits: int = 0,
                  range_spec: Optional[str] = None,
                  column: Optional[str] = None) -> List[float]:
        """
        ROUNDDOWN function matching Google Sheets.
        Formula: =ROUNDDOWN(number, digits)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Apply column filter if specified
        if column and column in df.columns:
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                values = df[column].drop_nulls().to_list()
            else:
                raise ValueError(f"Column '{column}' is not numeric")
        else:
            # Collect all numeric values
            values = []
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    values.extend(df[col].drop_nulls().to_list())
        
        result = []
        factor = 10 ** digits
        for val in values:
            result.append(math.floor(float(val) * factor) / factor)
        return result
    
    # ================== LOOKUP FUNCTIONS ==================
    
    def VLOOKUP(self,
                lookup_value: Any,
                table_array: Union[pl.DataFrame, str, Path],
                col_index_num: int,
                range_lookup: bool = False) -> Any:
        """
        VLOOKUP function matching Google Sheets.
        Formula: =VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])
        """
        df = self._load_data(table_array)
        
        if col_index_num < 1 or col_index_num > len(df.columns):
            raise ValueError(f"Column index {col_index_num} out of range")
        
        first_col = df.columns[0]
        return_col = df.columns[col_index_num - 1]
        
        if range_lookup:
            # Approximate match - find largest value <= lookup_value
            filtered = df.filter(pl.col(first_col) <= lookup_value)
            if filtered.is_empty():
                return None
            # Get the row with the maximum value in first column
            max_val = filtered[first_col].max()
            result = filtered.filter(pl.col(first_col) == max_val)[return_col]
        else:
            # Exact match
            result = df.filter(pl.col(first_col) == lookup_value)[return_col]
        
        if result.is_empty():
            return None
        return result[0]
    
    def HLOOKUP(self,
                lookup_value: Any,
                table_array: Union[pl.DataFrame, str, Path],
                row_index_num: int,
                range_lookup: bool = False) -> Any:
        """
        HLOOKUP function matching Google Sheets.
        Formula: =HLOOKUP(lookup_value, table_array, row_index_num, [range_lookup])
        """
        df = self._load_data(table_array)
        
        # Transpose for horizontal lookup
        df_transposed = df.transpose(include_header=True)
        
        # Use VLOOKUP logic on transposed data
        return self.VLOOKUP(lookup_value, df_transposed, row_index_num, range_lookup)
    
    def INDEX(self,
              array: Union[pl.DataFrame, str, Path],
              row_num: int,
              col_num: Optional[int] = None) -> Any:
        """
        INDEX function matching Google Sheets.
        Formula: =INDEX(array, row_num, [column_num])
        """
        df = self._load_data(array)
        
        # Convert to 0-based indexing
        row_idx = row_num - 1
        
        if row_idx < 0 or row_idx >= len(df):
            raise ValueError(f"Row index {row_num} out of range")
        
        if col_num is None:
            # Return entire row
            return df.row(row_idx)
        else:
            col_idx = col_num - 1
            if col_idx < 0 or col_idx >= len(df.columns):
                raise ValueError(f"Column index {col_num} out of range")
            return df.row(row_idx)[col_idx]
    
    def MATCH(self,
              lookup_value: Any,
              lookup_array: Union[pl.DataFrame, pl.Series, List],
              match_type: int = 1) -> int:
        """
        MATCH function matching Google Sheets.
        Formula: =MATCH(lookup_value, lookup_array, [match_type])
        match_type: 1 (less than), 0 (exact), -1 (greater than)
        """
        if isinstance(lookup_array, pl.DataFrame):
            # Use first column if DataFrame
            series = lookup_array[lookup_array.columns[0]]
        elif isinstance(lookup_array, list):
            series = pl.Series(lookup_array)
        else:
            series = lookup_array
        
        if match_type == 0:
            # Exact match
            matches = series == lookup_value
            if matches.any():
                return matches.arg_max() + 1  # 1-based index
        elif match_type == 1:
            # Largest value <= lookup_value (array must be sorted ascending)
            valid = series <= lookup_value
            if valid.any():
                # Get the index of the last True value
                indices = pl.Series(range(len(series)))[valid]
                return int(indices.max() + 1)
        elif match_type == -1:
            # Smallest value >= lookup_value (array must be sorted descending)
            valid = series >= lookup_value
            if valid.any():
                # Get the index of the last True value
                indices = pl.Series(range(len(series)))[valid]
                return int(indices.max() + 1)
        
        return None  # No match found
    
    # ================== CONDITIONAL AGGREGATION ==================
    
    def SUMIF(self,
              range_data: Union[pl.DataFrame, str, Path],
              criteria: Union[str, float, int],
              sum_range: Optional[Union[pl.DataFrame, str, Path]] = None) -> float:
        """
        SUMIF function matching Google Sheets.
        Formula: =SUMIF(range, criteria, [sum_range])
        """
        df = self._load_data(range_data)
        
        # Parse criteria
        condition = self._parse_criteria(df.columns[0], criteria)
        
        # Apply filter
        filtered = df.filter(condition)
        
        if sum_range is not None:
            sum_df = self._load_data(sum_range)
            # Sum from the sum_range using filtered indices
            indices = filtered.with_row_count("idx")["idx"].to_list()
            if sum_df.columns[0] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return float(sum_df[indices, 0].sum())
        else:
            # Sum the filtered range itself
            col = df.columns[0]
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return float(filtered[col].sum())
        
        return 0
    
    def COUNTIF(self,
                range_data: Union[pl.DataFrame, str, Path],
                criteria: Union[str, float, int]) -> int:
        """
        COUNTIF function matching Google Sheets.
        Formula: =COUNTIF(range, criteria)
        """
        df = self._load_data(range_data)
        
        # Parse criteria
        condition = self._parse_criteria(df.columns[0], criteria)
        
        # Count matching rows
        return df.filter(condition).height
    
    def AVERAGEIF(self,
                  range_data: Union[pl.DataFrame, str, Path],
                  criteria: Union[str, float, int],
                  average_range: Optional[Union[pl.DataFrame, str, Path]] = None) -> float:
        """
        AVERAGEIF function matching Google Sheets.
        Formula: =AVERAGEIF(range, criteria, [average_range])
        """
        df = self._load_data(range_data)
        
        # Parse criteria
        condition = self._parse_criteria(df.columns[0], criteria)
        
        # Apply filter
        filtered = df.filter(condition)
        
        if filtered.is_empty():
            return 0
        
        if average_range is not None:
            avg_df = self._load_data(average_range)
            # Average from the average_range using filtered indices
            indices = filtered.with_row_count("idx")["idx"].to_list()
            col = avg_df.columns[0]
            if avg_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return float(avg_df[indices, 0].mean())
        else:
            # Average the filtered range itself
            col = df.columns[0]
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return float(filtered[col].mean())
        
        return 0
    
    def SUMIFS(self,
               sum_range: Union[pl.DataFrame, str, Path],
               *criteria_pairs) -> float:
        """
        SUMIFS function matching Google Sheets.
        Formula: =SUMIFS(sum_range, criteria_range1, criteria1, [criteria_range2, criteria2, ...])
        """
        sum_df = self._load_data(sum_range)
        
        # Build combined filter
        conditions = []
        for i in range(0, len(criteria_pairs), 2):
            if i + 1 < len(criteria_pairs):
                criteria_range = self._load_data(criteria_pairs[i])
                criteria = criteria_pairs[i + 1]
                col = criteria_range.columns[0]
                conditions.append(self._parse_criteria(col, criteria))
        
        # Apply all conditions
        if conditions:
            combined_condition = conditions[0]
            for cond in conditions[1:]:
                combined_condition = combined_condition & cond
            
            filtered = sum_df.filter(combined_condition)
            col = sum_df.columns[0]
            if sum_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return float(filtered[col].sum())
        
        return 0
    
    # ================== TEXT FUNCTIONS ==================
    
    def CONCATENATE(self, *args) -> str:
        """
        CONCATENATE function matching Google Sheets.
        Formula: =CONCATENATE(text1, [text2, ...])
        """
        return "".join(str(arg) for arg in args if arg is not None)
    
    def LEFT(self, text: str, num_chars: int) -> str:
        """
        LEFT function matching Google Sheets.
        Formula: =LEFT(text, num_chars)
        """
        return str(text)[:num_chars]
    
    def RIGHT(self, text: str, num_chars: int) -> str:
        """
        RIGHT function matching Google Sheets.
        Formula: =RIGHT(text, num_chars)
        """
        return str(text)[-num_chars:] if num_chars > 0 else ""
    
    def MID(self, text: str, start_num: int, num_chars: int) -> str:
        """
        MID function matching Google Sheets.
        Formula: =MID(text, start_num, num_chars)
        """
        # Convert to 0-based indexing
        start_idx = start_num - 1
        return str(text)[start_idx:start_idx + num_chars]
    
    def LEN(self, text: str) -> int:
        """
        LEN function matching Google Sheets.
        Formula: =LEN(text)
        """
        return len(str(text))
    
    def UPPER(self, text: str) -> str:
        """
        UPPER function matching Google Sheets.
        Formula: =UPPER(text)
        """
        return str(text).upper()
    
    def LOWER(self, text: str) -> str:
        """
        LOWER function matching Google Sheets.
        Formula: =LOWER(text)
        """
        return str(text).lower()
    
    def PROPER(self, text: str) -> str:
        """
        PROPER function matching Google Sheets.
        Formula: =PROPER(text)
        """
        return str(text).title()
    
    def TRIM(self, text: str) -> str:
        """
        TRIM function matching Google Sheets.
        Formula: =TRIM(text)
        """
        return " ".join(str(text).split())
    
    def SUBSTITUTE(self, text: str, old_text: str, new_text: str, instance_num: Optional[int] = None) -> str:
        """
        SUBSTITUTE function matching Google Sheets.
        Formula: =SUBSTITUTE(text, old_text, new_text, [instance_num])
        """
        if instance_num:
            # Replace specific instance
            parts = str(text).split(old_text)
            if len(parts) > instance_num:
                result = old_text.join(parts[:instance_num])
                result += new_text
                result += old_text.join(parts[instance_num:])
                return result
            return str(text)
        else:
            # Replace all instances
            return str(text).replace(old_text, new_text)
    
    # ================== DATE/TIME FUNCTIONS ==================
    
    def TODAY(self) -> date:
        """
        TODAY function matching Google Sheets.
        Formula: =TODAY()
        """
        return datetime.now().date()
    
    def NOW(self) -> datetime:
        """
        NOW function matching Google Sheets.
        Formula: =NOW()
        """
        return datetime.now()
    
    def DATE(self, year: int, month: int, day: int) -> date:
        """
        DATE function matching Google Sheets.
        Formula: =DATE(year, month, day)
        """
        return date(year, month, day)
    
    def YEAR(self, date_value: Union[date, datetime, str]) -> int:
        """
        YEAR function matching Google Sheets.
        Formula: =YEAR(date)
        """
        if isinstance(date_value, str):
            date_value = datetime.strptime(date_value, "%Y-%m-%d")
        return date_value.year
    
    def MONTH(self, date_value: Union[date, datetime, str]) -> int:
        """
        MONTH function matching Google Sheets.
        Formula: =MONTH(date)
        """
        if isinstance(date_value, str):
            date_value = datetime.strptime(date_value, "%Y-%m-%d")
        return date_value.month
    
    def DAY(self, date_value: Union[date, datetime, str]) -> int:
        """
        DAY function matching Google Sheets.
        Formula: =DAY(date)
        """
        if isinstance(date_value, str):
            date_value = datetime.strptime(date_value, "%Y-%m-%d")
        return date_value.day
    
    def WEEKDAY(self, date_value: Union[date, datetime, str], type_num: int = 1) -> int:
        """
        WEEKDAY function matching Google Sheets.
        Formula: =WEEKDAY(date, [type])
        """
        if isinstance(date_value, str):
            date_value = datetime.strptime(date_value, "%Y-%m-%d")
        
        weekday = date_value.weekday()  # Monday = 0, Sunday = 6
        
        if type_num == 1:
            # 1 = Sunday, 2 = Monday, ..., 7 = Saturday
            return (weekday + 2) % 7 or 7
        elif type_num == 2:
            # 1 = Monday, 2 = Tuesday, ..., 7 = Sunday
            return weekday + 1 if weekday < 6 else 7
        elif type_num == 3:
            # 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
            return weekday if weekday < 6 else 0
        else:
            raise ValueError(f"Invalid type_num: {type_num}")
    
    def EOMONTH(self, start_date: Union[date, datetime, str], months: int) -> date:
        """
        EOMONTH function matching Google Sheets.
        Formula: =EOMONTH(start_date, months)
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
        
        # Add months
        year = start_date.year
        month = start_date.month + months
        
        # Handle year overflow
        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1
        
        # Get last day of the month
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        
        return next_month - timedelta(days=1)
    
    def DATEDIF(self, start_date: Union[date, datetime, str], 
                end_date: Union[date, datetime, str], unit: str) -> int:
        """
        DATEDIF function matching Google Sheets.
        Formula: =DATEDIF(start_date, end_date, unit)
        Units: "Y" (years), "M" (months), "D" (days), "MD", "YM", "YD"
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
            
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
        
        if unit == "D":
            return (end_date - start_date).days
        elif unit == "M":
            return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        elif unit == "Y":
            return end_date.year - start_date.year
        elif unit == "MD":
            # Days ignoring months and years
            return end_date.day - start_date.day
        elif unit == "YM":
            # Months ignoring years
            return end_date.month - start_date.month
        elif unit == "YD":
            # Days ignoring years
            return (end_date.replace(year=start_date.year) - start_date).days
        else:
            raise ValueError(f"Invalid unit: {unit}")
    
    # ================== LOGICAL FUNCTIONS ==================
    
    def IF(self, logical_test: bool, value_if_true: Any, value_if_false: Any) -> Any:
        """
        IF function matching Google Sheets.
        Formula: =IF(logical_test, value_if_true, value_if_false)
        """
        return value_if_true if logical_test else value_if_false
    
    def AND(self, *logical_values) -> bool:
        """
        AND function matching Google Sheets.
        Formula: =AND(logical1, [logical2, ...])
        """
        return all(logical_values)
    
    def OR(self, *logical_values) -> bool:
        """
        OR function matching Google Sheets.
        Formula: =OR(logical1, [logical2, ...])
        """
        return any(logical_values)
    
    def NOT(self, logical_value: bool) -> bool:
        """
        NOT function matching Google Sheets.
        Formula: =NOT(logical)
        """
        return not logical_value
    
    def IFERROR(self, value: Any, value_if_error: Any) -> Any:
        """
        IFERROR function matching Google Sheets.
        Formula: =IFERROR(value, value_if_error)
        """
        try:
            # Check if value is an error or None
            if value is None or (isinstance(value, str) and value.startswith("#")):
                return value_if_error
            return value
        except:
            return value_if_error
    
    def ISBLANK(self, value: Any) -> bool:
        """
        ISBLANK function matching Google Sheets.
        Formula: =ISBLANK(value)
        """
        return value is None or value == ""
    
    def ISNUMBER(self, value: Any) -> bool:
        """
        ISNUMBER function matching Google Sheets.
        Formula: =ISNUMBER(value)
        """
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    
    def ISTEXT(self, value: Any) -> bool:
        """
        ISTEXT function matching Google Sheets.
        Formula: =ISTEXT(value)
        """
        return isinstance(value, str)
    
    # ================== ARRAY FORMULAS ==================
    
    def ARRAYFORMULA(self, 
                     expression: str,
                     data: Union[pl.DataFrame, str, Path],
                     range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        ARRAYFORMULA function matching Google Sheets.
        Formula: =ARRAYFORMULA(expression)
        Applies expression to entire range.
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        # Parse and apply expression to entire columns
        # This is a simplified version - full implementation would parse the expression
        return df
    
    def TRANSPOSE(self,
                  data: Union[pl.DataFrame, str, Path],
                  range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        TRANSPOSE function matching Google Sheets.
        Formula: =TRANSPOSE(range)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        return df.transpose(include_header=True)
    
    def UNIQUE(self,
               data: Union[pl.DataFrame, str, Path],
               range_spec: Optional[str] = None,
               by_column: bool = False,
               exactly_once: bool = False) -> pl.DataFrame:
        """
        UNIQUE function matching Google Sheets.
        Formula: =UNIQUE(range, [by_column], [exactly_once])
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if exactly_once:
            # Return only values that appear exactly once
            return df.filter(pl.all_horizontal(pl.col("*").n_unique() == 1))
        else:
            return df.unique()
    
    def SORT(self,
             data: Union[pl.DataFrame, str, Path],
             sort_column: int = 1,
             is_ascending: bool = True,
             range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        SORT function matching Google Sheets.
        Formula: =SORT(range, sort_column, is_ascending)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        col_idx = sort_column - 1  # Convert to 0-based
        if col_idx < 0 or col_idx >= len(df.columns):
            raise ValueError(f"Sort column {sort_column} out of range")
        
        return df.sort(df.columns[col_idx], descending=not is_ascending)
    
    def FILTER(self,
               data: Union[pl.DataFrame, str, Path],
               conditions: Union[pl.Expr, List[bool]],
               range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        FILTER function matching Google Sheets.
        Formula: =FILTER(range, condition1, [condition2, ...])
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if isinstance(conditions, list):
            # Boolean mask
            return df.filter(pl.Series(conditions))
        else:
            # Polars expression
            return df.filter(conditions)
    
    # ================== HELPER METHODS ==================
    
    def _load_data(self, data: Union[pl.DataFrame, str, Path]) -> pl.DataFrame:
        """Load data from various sources"""
        if isinstance(data, pl.DataFrame):
            return data
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == '.csv':
                return pl.read_csv(path)
            elif path.suffix == '.parquet':
                return pl.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _parse_criteria(self, column: str, criteria: Union[str, float, int]) -> pl.Expr:
        """Parse criteria string into Polars expression"""
        col_expr = pl.col(column)
        
        if isinstance(criteria, (int, float)):
            return col_expr == criteria
        
        criteria_str = str(criteria)
        
        # Check for comparison operators
        if criteria_str.startswith(">="):
            value = self._parse_value(criteria_str[2:])
            return col_expr >= value
        elif criteria_str.startswith("<="):
            value = self._parse_value(criteria_str[2:])
            return col_expr <= value
        elif criteria_str.startswith("<>") or criteria_str.startswith("!="):
            value = self._parse_value(criteria_str[2:])
            return col_expr != value
        elif criteria_str.startswith(">"):
            value = self._parse_value(criteria_str[1:])
            return col_expr > value
        elif criteria_str.startswith("<"):
            value = self._parse_value(criteria_str[1:])
            return col_expr < value
        elif criteria_str.startswith("="):
            value = self._parse_value(criteria_str[1:])
            return col_expr == value
        elif "*" in criteria_str or "?" in criteria_str:
            # Wildcard pattern
            pattern = criteria_str.replace("*", ".*").replace("?", ".")
            return col_expr.str.contains(f"^{pattern}$")
        else:
            # Exact match
            return col_expr == criteria
    
    def _parse_value(self, value_str: str) -> Union[float, int, str]:
        """Parse string value to appropriate type"""
        value_str = value_str.strip()
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # Return as string
            return value_str.strip('"').strip("'")
    
    # ================== FINANCIAL FUNCTIONS ==================
    
    def NPV(self,
            rate: float,
            data: Union[pl.DataFrame, str, Path],
            cash_flow_column: str,
            range_spec: Optional[str] = None) -> float:
        """
        NPV (Net Present Value) function matching Google Sheets.
        Formula: =NPV(rate, range)
        
        Args:
            rate: Discount rate per period
            data: DataFrame or file path containing cash flows
            cash_flow_column: Column containing cash flow values
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns:
            raise ValueError(f"Column '{cash_flow_column}' not found")
        
        # Get cash flows and drop nulls
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        
        # Calculate NPV: sum of cash_flow / (1 + rate)^period
        npv = sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
        return float(npv)
    
    def IRR(self,
            data: Union[pl.DataFrame, str, Path],
            cash_flow_column: str,
            range_spec: Optional[str] = None,
            guess: float = 0.1,
            max_iter: int = 100,
            tolerance: float = 1e-6) -> float:
        """
        IRR (Internal Rate of Return) function matching Google Sheets.
        Formula: =IRR(range, guess)
        
        Uses Newton-Raphson method to find the rate where NPV = 0
        """
        try:
            from scipy.optimize import fsolve
        except ImportError:
            # Fallback if scipy is not available
            pass
        
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns:
            raise ValueError(f"Column '{cash_flow_column}' not found")
        
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        
        # Check if we have both positive and negative cash flows
        has_positive = any(cf > 0 for cf in cash_flows)
        has_negative = any(cf < 0 for cf in cash_flows)
        if not (has_positive and has_negative):
            raise ValueError("IRR requires both positive and negative cash flows")
        
        def npv_func(rate):
            return sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
        
        # Try scipy first if available
        try:
            result = fsolve(npv_func, guess)[0]
            return float(result)
        except:
            # Fallback to manual Newton-Raphson iteration
            rate = guess
            for _ in range(max_iter):
                npv = npv_func(rate)
                if abs(npv) < tolerance:
                    return float(rate)
                
                # Derivative of NPV with respect to rate
                dnpv = sum(-cf * (i + 1) / ((1 + rate) ** (i + 2)) for i, cf in enumerate(cash_flows))
                if abs(dnpv) < 1e-12:
                    break
                
                rate = rate - npv / dnpv
            
            return float(rate)
    
    def PV(self,
           rate: float,
           nper: int,
           pmt: float,
           fv: float = 0,
           type_: int = 0) -> float:
        """
        PV (Present Value) function matching Google Sheets.
        Formula: =PV(rate, nper, pmt, fv, type)
        
        Args:
            rate: Interest rate per period
            nper: Number of payment periods
            pmt: Payment made each period (negative for outgoing)
            fv: Future value (default 0)
            type_: When payments are due (0=end of period, 1=beginning)
        """
        if rate == 0:
            return -(pmt * nper + fv)
        
        pv_annuity = pmt * (1 - (1 + rate) ** (-nper)) / rate
        pv_lump_sum = fv / ((1 + rate) ** nper)
        
        if type_ == 1:  # Beginning of period
            pv_annuity *= (1 + rate)
        
        return -(pv_annuity + pv_lump_sum)
    
    def FV(self,
           rate: float,
           nper: int,
           pmt: float,
           pv: float = 0,
           type_: int = 0) -> float:
        """
        FV (Future Value) function matching Google Sheets.
        Formula: =FV(rate, nper, pmt, pv, type)
        
        Args:
            rate: Interest rate per period
            nper: Number of payment periods  
            pmt: Payment made each period
            pv: Present value (default 0)
            type_: When payments are due (0=end of period, 1=beginning)
        """
        if rate == 0:
            return -(pv + pmt * nper)
        
        fv_annuity = pmt * (((1 + rate) ** nper - 1) / rate)
        fv_lump_sum = pv * ((1 + rate) ** nper)
        
        if type_ == 1:  # Beginning of period
            fv_annuity *= (1 + rate)
        
        return -(fv_lump_sum + fv_annuity)
    
    def PMT(self,
            rate: float,
            nper: int,
            pv: float,
            fv: float = 0,
            type_: int = 0) -> float:
        """
        PMT (Payment) function matching Google Sheets.
        Formula: =PMT(rate, nper, pv, fv, type)
        
        Args:
            rate: Interest rate per period
            nper: Number of payment periods
            pv: Present value
            fv: Future value (default 0)
            type_: When payments are due (0=end of period, 1=beginning)
        """
        if rate == 0:
            return -(pv + fv) / nper
        
        factor = (1 + rate) ** nper
        pmt = -(pv * factor + fv) * rate / (factor - 1)
        
        if type_ == 1:  # Beginning of period
            pmt /= (1 + rate)
        
        return pmt
    
    def XNPV(self,
             rate: float,
             data: Union[pl.DataFrame, str, Path],
             cash_flow_column: str,
             date_column: str,
             range_spec: Optional[str] = None) -> float:
        """
        XNPV (Extended Net Present Value) function for irregular periods.
        Formula: =XNPV(rate, values, dates)
        
        Args:
            rate: Discount rate per year
            data: DataFrame containing cash flows and dates
            cash_flow_column: Column containing cash flow values
            date_column: Column containing dates
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns or date_column not in df.columns:
            raise ValueError(f"Columns '{cash_flow_column}' or '{date_column}' not found")
        
        # Convert dates and get cash flows
        dates_series = df[date_column].str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        dates = dates_series.drop_nulls().to_list()
        
        if len(cash_flows) != len(dates):
            raise ValueError("Cash flows and dates must have same length")
        
        # Use first date as reference
        start_date = min(dates)
        
        xnpv = 0
        for cf, dt in zip(cash_flows, dates):
            # Calculate years between dates
            years = (dt - start_date).days / 365.25
            xnpv += cf / ((1 + rate) ** years)
        
        return float(xnpv)
    
    def XIRR(self,
             data: Union[pl.DataFrame, str, Path],
             cash_flow_column: str,
             date_column: str,
             range_spec: Optional[str] = None,
             guess: float = 0.1,
             max_iter: int = 100,
             tolerance: float = 1e-6) -> float:
        """
        XIRR (Extended Internal Rate of Return) for irregular periods.
        Formula: =XIRR(values, dates, guess)
        
        Uses Newton-Raphson to find rate where XNPV = 0
        """
        try:
            from scipy.optimize import fsolve
        except ImportError:
            pass
        
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns or date_column not in df.columns:
            raise ValueError(f"Columns '{cash_flow_column}' or '{date_column}' not found")
        
        # Convert dates and get cash flows
        dates_series = df[date_column].str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        dates = dates_series.drop_nulls().to_list()
        
        if len(cash_flows) != len(dates):
            raise ValueError("Cash flows and dates must have same length")
        
        # Check for both positive and negative cash flows
        has_positive = any(cf > 0 for cf in cash_flows)
        has_negative = any(cf < 0 for cf in cash_flows)
        if not (has_positive and has_negative):
            raise ValueError("XIRR requires both positive and negative cash flows")
        
        start_date = min(dates)
        
        def xnpv_func(rate):
            xnpv = 0
            for cf, dt in zip(cash_flows, dates):
                years = (dt - start_date).days / 365.25
                xnpv += cf / ((1 + rate) ** years)
            return xnpv
        
        # Try scipy first if available
        try:
            result = fsolve(xnpv_func, guess)[0]
            return float(result)
        except:
            # Fallback to manual iteration
            rate = guess
            for _ in range(max_iter):
                xnpv = xnpv_func(rate)
                if abs(xnpv) < tolerance:
                    return float(rate)
                
                # Derivative calculation
                dxnpv = 0
                for cf, dt in zip(cash_flows, dates):
                    years = (dt - start_date).days / 365.25
                    dxnpv += -cf * years / ((1 + rate) ** (years + 1))
                
                if abs(dxnpv) < 1e-12:
                    break
                
                rate = rate - xnpv / dxnpv
            
            return float(rate)
    
    def NPER(self,
             rate: float,
             pmt: float,
             pv: float,
             fv: float = 0,
             type_: int = 0) -> float:
        """
        NPER (Number of Periods) function matching Google Sheets.
        Formula: =NPER(rate, pmt, pv, fv, type)
        
        Args:
            rate: Interest rate per period
            pmt: Payment made each period
            pv: Present value
            fv: Future value (default 0)
            type_: When payments are due (0=end, 1=beginning)
        """
        if rate == 0:
            return -(pv + fv) / pmt
        
        if type_ == 1:
            pmt = pmt * (1 + rate)
        
        # NPER formula: log((-fv + pmt/rate)/(-pv + pmt/rate)) / log(1 + rate)
        # Alternative: log((pmt - fv*rate)/(pmt + pv*rate)) / log(1 + rate)
        
        import math
        
        # Handle the case where we need to take absolute values for log calculation
        numerator = abs(pmt - fv * rate)
        denominator = abs(pmt + pv * rate)
        
        if numerator <= 0 or denominator <= 0:
            # Try alternative formula
            if pmt == 0:
                raise ValueError("Payment cannot be zero")
            
            # Simplified approach for common loan scenarios
            if pv > 0 and pmt < 0:  # Loan scenario
                # Use standard loan formula
                nper = -math.log(1 - (pv * rate) / (-pmt)) / math.log(1 + rate)
            else:
                raise ValueError("Invalid parameters for NPER calculation")
        else:
            nper = math.log(numerator / denominator) / math.log(1 + rate)
        return float(nper)
    
    def RATE(self,
             nper: int,
             pmt: float,
             pv: float,
             fv: float = 0,
             type_: int = 0,
             guess: float = 0.1,
             max_iter: int = 100,
             tolerance: float = 1e-6) -> float:
        """
        RATE function matching Google Sheets.
        Formula: =RATE(nper, pmt, pv, fv, type, guess)
        
        Uses Newton-Raphson to solve for interest rate
        """
        if nper == 0:
            raise ValueError("Number of periods cannot be zero")
        
        rate = guess
        
        for _ in range(max_iter):
            if type_ == 1:
                adjusted_pmt = pmt * (1 + rate)
            else:
                adjusted_pmt = pmt
            
            if rate == 0:
                # Special case: rate = 0
                f = pv + adjusted_pmt * nper + fv
            else:
                # General case
                factor = (1 + rate) ** nper
                f = pv * factor + adjusted_pmt * (factor - 1) / rate + fv
            
            if abs(f) < tolerance:
                return float(rate)
            
            # Calculate derivative
            if rate == 0:
                df = adjusted_pmt * nper
                if type_ == 1:
                    df += pmt * nper
            else:
                factor = (1 + rate) ** nper
                df = (pv * nper * factor / (1 + rate) + 
                      adjusted_pmt * (nper * factor - (factor - 1) / rate) / (rate * (1 + rate)))
                if type_ == 1:
                    df += pmt * (factor - 1) / rate
            
            if abs(df) < 1e-12:
                break
            
            rate = rate - f / df
        
        return float(rate)
    
    # ================== RISK METRICS ==================
    
    def BETA(self,
             asset_data: Union[pl.DataFrame, str, Path],
             market_data: Union[pl.DataFrame, str, Path],
             asset_column: str,
             market_column: str,
             range_spec: Optional[str] = None) -> float:
        """
        BETA function for measuring asset's correlation with market.
        Formula: COVARIANCE(asset, market) / VARIANCE(market)
        
        Args:
            asset_data: DataFrame or file with asset returns
            market_data: DataFrame or file with market returns
            asset_column: Column name for asset returns
            market_column: Column name for market returns
            range_spec: A1 notation range (optional)
        """
        asset_df = self._load_data(asset_data)
        market_df = self._load_data(market_data)
        
        if range_spec:
            asset_df = self.resolver.resolve_range(asset_df, range_spec)
            market_df = self.resolver.resolve_range(market_df, range_spec)
        
        if asset_column not in asset_df.columns or market_column not in market_df.columns:
            raise ValueError(f"Required columns not found")
        
        # Get returns data
        asset_returns = asset_df[asset_column].drop_nulls().to_list()
        market_returns = market_df[market_column].drop_nulls().to_list()
        
        # Ensure same length
        min_len = min(len(asset_returns), len(market_returns))
        asset_returns = asset_returns[:min_len]
        market_returns = market_returns[:min_len]
        
        if len(asset_returns) < 2:
            raise ValueError("Insufficient data for BETA calculation")
        
        # Calculate covariance and market variance
        import numpy as np
        covariance = np.cov(asset_returns, market_returns, ddof=1)[0][1]
        market_variance = np.var(market_returns, ddof=1)
        
        if market_variance == 0:
            raise ValueError("Market variance is zero - cannot calculate BETA")
        
        beta = covariance / market_variance
        return float(beta)
    
    def SHARPE_RATIO(self,
                     returns_data: Union[pl.DataFrame, str, Path],
                     returns_column: str,
                     risk_free_rate: float = 0.0,
                     range_spec: Optional[str] = None) -> float:
        """
        SHARPE RATIO function for risk-adjusted return measurement.
        Formula: (Mean Return - Risk Free Rate) / Standard Deviation
        
        Args:
            returns_data: DataFrame or file with returns
            returns_column: Column name for returns
            risk_free_rate: Risk-free rate (annualized)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(returns_data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if returns_column not in df.columns:
            raise ValueError(f"Column '{returns_column}' not found")
        
        returns = df[returns_column].drop_nulls().to_list()
        
        if len(returns) < 2:
            raise ValueError("Insufficient data for Sharpe ratio calculation")
        
        import numpy as np
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            raise ValueError("Standard deviation is zero - cannot calculate Sharpe ratio")
        
        sharpe = (mean_return - risk_free_rate) / std_return
        return float(sharpe)
    
    def CAPM(self,
             beta: float,
             risk_free_rate: float,
             market_return: float) -> float:
        """
        CAPM (Capital Asset Pricing Model) function.
        Formula: Risk Free Rate + Beta * (Market Return - Risk Free Rate)
        
        Args:
            beta: Asset's beta coefficient
            risk_free_rate: Risk-free rate
            market_return: Expected market return
        """
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        return float(expected_return)
    
    def VAR_HISTORICAL(self,
                       returns_data: Union[pl.DataFrame, str, Path],
                       returns_column: str,
                       confidence_level: float = 0.95,
                       portfolio_value: float = 1.0,
                       range_spec: Optional[str] = None) -> float:
        """
        Historical Value at Risk calculation.
        Formula: Percentile of historical returns at confidence level
        
        Args:
            returns_data: DataFrame or file with returns
            returns_column: Column name for returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            portfolio_value: Portfolio value for VaR calculation
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(returns_data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if returns_column not in df.columns:
            raise ValueError(f"Column '{returns_column}' not found")
        
        returns = df[returns_column].drop_nulls().to_list()
        
        if len(returns) < 10:
            raise ValueError("Insufficient data for VaR calculation")
        
        import numpy as np
        # Sort returns and find percentile
        sorted_returns = sorted(returns)
        percentile_index = int((1 - confidence_level) * len(sorted_returns))
        var_return = sorted_returns[percentile_index]
        
        # Convert to VaR (positive number representing potential loss)
        var = -var_return * portfolio_value
        return float(var)
    
    def VAR_PARAMETRIC(self,
                       returns_data: Union[pl.DataFrame, str, Path],
                       returns_column: str,
                       confidence_level: float = 0.95,
                       portfolio_value: float = 1.0,
                       range_spec: Optional[str] = None) -> float:
        """
        Parametric (Normal) Value at Risk calculation.
        Formula: Portfolio Value * (Mean Return - Z-score * Std Deviation)
        
        Args:
            returns_data: DataFrame or file with returns
            returns_column: Column name for returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            portfolio_value: Portfolio value for VaR calculation
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(returns_data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if returns_column not in df.columns:
            raise ValueError(f"Column '{returns_column}' not found")
        
        returns = df[returns_column].drop_nulls().to_list()
        
        if len(returns) < 2:
            raise ValueError("Insufficient data for VaR calculation")
        
        import numpy as np
        try:
            from scipy.stats import norm
        except ImportError:
            # Fallback approximation for 95% confidence
            z_score = -1.645 if confidence_level == 0.95 else -2.33
        else:
            z_score = norm.ppf(1 - confidence_level)
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Calculate VaR
        var_return = mean_return + z_score * std_return
        var = -var_return * portfolio_value
        
        return float(var)
    
    def CAGR(self,
             beginning_value: float,
             ending_value: float,
             years: float) -> float:
        """
        CAGR (Compound Annual Growth Rate) function.
        Formula: (Ending Value / Beginning Value)^(1/years) - 1
        
        Args:
            beginning_value: Starting value
            ending_value: Ending value
            years: Number of years
        """
        if beginning_value <= 0 or ending_value <= 0 or years <= 0:
            raise ValueError("All values must be positive")
        
        cagr = (ending_value / beginning_value) ** (1 / years) - 1
        return float(cagr)
    
    def MIRR(self,
             data: Union[pl.DataFrame, str, Path],
             cash_flow_column: str,
             finance_rate: float,
             reinvest_rate: float,
             range_spec: Optional[str] = None) -> float:
        """
        MIRR (Modified Internal Rate of Return) function.
        Formula: (FV of positive flows / PV of negative flows)^(1/n) - 1
        
        Args:
            data: DataFrame containing cash flows
            cash_flow_column: Column with cash flows
            finance_rate: Interest rate for negative cash flows
            reinvest_rate: Interest rate for positive cash flows
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns:
            raise ValueError(f"Column '{cash_flow_column}' not found")
        
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        n = len(cash_flows)
        
        if n < 2:
            raise ValueError("Need at least 2 cash flows for MIRR")
        
        # Separate positive and negative cash flows
        fv_positive = 0  # Future value of positive cash flows
        pv_negative = 0  # Present value of negative cash flows
        
        for i, cf in enumerate(cash_flows):
            if cf > 0:
                # Compound positive cash flows to final period
                fv_positive += cf * ((1 + reinvest_rate) ** (n - 1 - i))
            elif cf < 0:
                # Discount negative cash flows to present
                pv_negative += cf / ((1 + finance_rate) ** i)
        
        if pv_negative >= 0 or fv_positive <= 0:
            raise ValueError("Invalid cash flow pattern for MIRR calculation")
        
        # Calculate MIRR
        mirr = (fv_positive / (-pv_negative)) ** (1 / (n - 1)) - 1
        return float(mirr)
    
    # ================== SPECIALIZED FINANCIAL FUNCTIONS ==================
    
    def BLACK_SCHOLES(self,
                      spot_price: float,
                      strike_price: float,
                      time_to_expiry: float,
                      risk_free_rate: float,
                      volatility: float,
                      option_type: str = "call") -> float:
        """
        Black-Scholes option pricing model.
        Formula: Complex formula for European option pricing
        
        Args:
            spot_price: Current price of underlying asset
            strike_price: Strike price of option
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility of underlying asset (annual)
            option_type: "call" or "put"
        """
        import math
        try:
            from scipy.stats import norm
        except ImportError:
            # Fallback approximation using math.erf
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        else:
            norm_cdf = norm.cdf
        
        if time_to_expiry <= 0:
            # At expiration
            if option_type.lower() == "call":
                return max(spot_price - strike_price, 0)
            else:
                return max(strike_price - spot_price, 0)
        
        # Calculate d1 and d2
        d1 = (math.log(spot_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        
        if option_type.lower() == "call":
            # Call option price
            price = (spot_price * norm_cdf(d1) - 
                    strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2))
        else:
            # Put option price
            price = (strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2) - 
                    spot_price * norm_cdf(-d1))
        
        return float(price)
    
    def PAYBACK_PERIOD(self,
                       data: Union[pl.DataFrame, str, Path],
                       cash_flow_column: str,
                       range_spec: Optional[str] = None) -> float:
        """
        Simple payback period calculation.
        Formula: Time required to recover initial investment
        
        Args:
            data: DataFrame containing cash flows
            cash_flow_column: Column with cash flows (negative for outflows)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns:
            raise ValueError(f"Column '{cash_flow_column}' not found")
        
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        
        if len(cash_flows) < 2:
            raise ValueError("Need at least 2 cash flows for payback calculation")
        
        if cash_flows[0] >= 0:
            raise ValueError("First cash flow should be negative (initial investment)")
        
        cumulative = cash_flows[0]  # Initial investment (negative)
        
        for i in range(1, len(cash_flows)):
            if cumulative + cash_flows[i] >= 0:
                # Payback occurs during this period
                fraction = -cumulative / cash_flows[i]
                return float(i - 1 + fraction)
            cumulative += cash_flows[i]
        
        # Payback period exceeds the data period
        return float('inf')
    
    def DISCOUNTED_PAYBACK_PERIOD(self,
                                  data: Union[pl.DataFrame, str, Path],
                                  cash_flow_column: str,
                                  discount_rate: float,
                                  range_spec: Optional[str] = None) -> float:
        """
        Discounted payback period calculation.
        Formula: Time required to recover initial investment with discounting
        
        Args:
            data: DataFrame containing cash flows
            cash_flow_column: Column with cash flows
            discount_rate: Discount rate for NPV calculation
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns:
            raise ValueError(f"Column '{cash_flow_column}' not found")
        
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        
        if len(cash_flows) < 2:
            raise ValueError("Need at least 2 cash flows for discounted payback calculation")
        
        if cash_flows[0] >= 0:
            raise ValueError("First cash flow should be negative (initial investment)")
        
        cumulative_pv = cash_flows[0]  # Initial investment (negative)
        
        for i in range(1, len(cash_flows)):
            # Discount the cash flow
            pv_cash_flow = cash_flows[i] / ((1 + discount_rate) ** i)
            
            if cumulative_pv + pv_cash_flow >= 0:
                # Payback occurs during this period
                remaining = -cumulative_pv
                next_pv = cash_flows[i] / ((1 + discount_rate) ** i)
                fraction = remaining / next_pv
                return float(i - 1 + fraction)
            
            cumulative_pv += pv_cash_flow
        
        # Discounted payback period exceeds the data period
        return float('inf')
    
    def PROFITABILITY_INDEX(self,
                            data: Union[pl.DataFrame, str, Path],
                            cash_flow_column: str,
                            discount_rate: float,
                            range_spec: Optional[str] = None) -> float:
        """
        Profitability Index calculation.
        Formula: (PV of future cash flows) / Initial Investment
        
        Args:
            data: DataFrame containing cash flows
            cash_flow_column: Column with cash flows
            discount_rate: Discount rate for PV calculation
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if cash_flow_column not in df.columns:
            raise ValueError(f"Column '{cash_flow_column}' not found")
        
        cash_flows = df[cash_flow_column].drop_nulls().to_list()
        
        if len(cash_flows) < 2:
            raise ValueError("Need at least 2 cash flows for profitability index calculation")
        
        initial_investment = abs(cash_flows[0])  # Make positive
        
        # Calculate PV of future cash flows (excluding initial investment)
        pv_future_flows = sum(cf / ((1 + discount_rate) ** i) 
                             for i, cf in enumerate(cash_flows[1:], 1))
        
        if initial_investment == 0:
            raise ValueError("Initial investment cannot be zero")
        
        pi = pv_future_flows / initial_investment
        return float(pi)
    
    # ================== ADVANCED ANALYTICS FUNCTIONS ==================
    
    def PIVOT_SUM(self,
                  data: Union[pl.DataFrame, str, Path],
                  index_column: str,
                  values_column: str,
                  aggfunc_column: Optional[str] = None,
                  range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        PIVOT_SUM function for pivot table with sum aggregation.
        Formula: Creates a pivot table summing values by categories
        
        Args:
            data: DataFrame or file path
            index_column: Column to use as row index
            values_column: Column to sum
            aggfunc_column: Optional column for additional grouping
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if index_column not in df.columns or values_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        if aggfunc_column and aggfunc_column not in df.columns:
            raise ValueError(f"Aggregation column '{aggfunc_column}' not found")
        
        # Group by index column and sum values
        if aggfunc_column:
            result = df.group_by([index_column, aggfunc_column]).agg(
                pl.col(values_column).sum().alias("sum_" + values_column)
            )
            # Pivot to create cross-tabulation
            result = result.pivot(
                index=index_column,
                columns=aggfunc_column,
                values="sum_" + values_column,
                aggregate_function="first"
            )
        else:
            result = df.group_by(index_column).agg(
                pl.col(values_column).sum().alias("sum_" + values_column)
            )
        
        return result
    
    def PIVOT_COUNT(self,
                    data: Union[pl.DataFrame, str, Path],
                    index_column: str,
                    values_column: str,
                    aggfunc_column: Optional[str] = None,
                    range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        PIVOT_COUNT function for pivot table with count aggregation.
        Formula: Creates a pivot table counting values by categories
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if index_column not in df.columns or values_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        if aggfunc_column and aggfunc_column not in df.columns:
            raise ValueError(f"Aggregation column '{aggfunc_column}' not found")
        
        # Group by index column and count values
        if aggfunc_column:
            result = df.group_by([index_column, aggfunc_column]).agg(
                pl.col(values_column).count().alias("count_" + values_column)
            )
            # Pivot to create cross-tabulation
            result = result.pivot(
                index=index_column,
                columns=aggfunc_column,
                values="count_" + values_column,
                aggregate_function="first"
            )
        else:
            result = df.group_by(index_column).agg(
                pl.col(values_column).count().alias("count_" + values_column)
            )
        
        return result
    
    def PIVOT_AVERAGE(self,
                      data: Union[pl.DataFrame, str, Path],
                      index_column: str,
                      values_column: str,
                      aggfunc_column: Optional[str] = None,
                      range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        PIVOT_AVERAGE function for pivot table with average aggregation.
        Formula: Creates a pivot table averaging values by categories
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if index_column not in df.columns or values_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        if aggfunc_column and aggfunc_column not in df.columns:
            raise ValueError(f"Aggregation column '{aggfunc_column}' not found")
        
        # Group by index column and average values
        if aggfunc_column:
            result = df.group_by([index_column, aggfunc_column]).agg(
                pl.col(values_column).mean().alias("avg_" + values_column)
            )
            # Pivot to create cross-tabulation
            result = result.pivot(
                index=index_column,
                columns=aggfunc_column,
                values="avg_" + values_column,
                aggregate_function="first"
            )
        else:
            result = df.group_by(index_column).agg(
                pl.col(values_column).mean().alias("avg_" + values_column)
            )
        
        return result
    
    def PIVOT_MAX(self,
                  data: Union[pl.DataFrame, str, Path],
                  index_column: str,
                  values_column: str,
                  aggfunc_column: Optional[str] = None,
                  range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        PIVOT_MAX function for pivot table with maximum aggregation.
        Formula: Creates a pivot table finding max values by categories
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if index_column not in df.columns or values_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        if aggfunc_column and aggfunc_column not in df.columns:
            raise ValueError(f"Aggregation column '{aggfunc_column}' not found")
        
        # Group by index column and find max values
        if aggfunc_column:
            result = df.group_by([index_column, aggfunc_column]).agg(
                pl.col(values_column).max().alias("max_" + values_column)
            )
            # Pivot to create cross-tabulation
            result = result.pivot(
                index=index_column,
                columns=aggfunc_column,
                values="max_" + values_column,
                aggregate_function="first"
            )
        else:
            result = df.group_by(index_column).agg(
                pl.col(values_column).max().alias("max_" + values_column)
            )
        
        return result
    
    def PIVOT_MIN(self,
                  data: Union[pl.DataFrame, str, Path],
                  index_column: str,
                  values_column: str,
                  aggfunc_column: Optional[str] = None,
                  range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        PIVOT_MIN function for pivot table with minimum aggregation.
        Formula: Creates a pivot table finding min values by categories
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if index_column not in df.columns or values_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        if aggfunc_column and aggfunc_column not in df.columns:
            raise ValueError(f"Aggregation column '{aggfunc_column}' not found")
        
        # Group by index column and find min values
        if aggfunc_column:
            result = df.group_by([index_column, aggfunc_column]).agg(
                pl.col(values_column).min().alias("min_" + values_column)
            )
            # Pivot to create cross-tabulation
            result = result.pivot(
                index=index_column,
                columns=aggfunc_column,
                values="min_" + values_column,
                aggregate_function="first"
            )
        else:
            result = df.group_by(index_column).agg(
                pl.col(values_column).min().alias("min_" + values_column)
            )
        
        return result
    
    def GROUP_BY_SUM(self,
                     data: Union[pl.DataFrame, str, Path],
                     group_column: str,
                     sum_column: str,
                     range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        GROUP_BY_SUM function for grouping and summing data.
        Formula: SQL-like GROUP BY with SUM aggregation
        
        Args:
            data: DataFrame or file path
            group_column: Column to group by
            sum_column: Column to sum
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if group_column not in df.columns or sum_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        result = df.group_by(group_column).agg(
            pl.col(sum_column).sum().alias("sum_" + sum_column)
        )
        
        return result.sort(group_column)
    
    def GROUP_BY_COUNT(self,
                       data: Union[pl.DataFrame, str, Path],
                       group_column: str,
                       count_column: str,
                       range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        GROUP_BY_COUNT function for grouping and counting data.
        Formula: SQL-like GROUP BY with COUNT aggregation
        
        Args:
            data: DataFrame or file path
            group_column: Column to group by
            count_column: Column to count
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if group_column not in df.columns or count_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        result = df.group_by(group_column).agg(
            pl.col(count_column).count().alias("count_" + count_column)
        )
        
        return result.sort(group_column)
    
    def CROSSTAB(self,
                 data: Union[pl.DataFrame, str, Path],
                 index_column: str,
                 column_column: str,
                 values_column: Optional[str] = None,
                 range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        CROSSTAB function for creating cross-tabulation tables.
        Formula: Creates contingency table showing frequency/values
        
        Args:
            data: DataFrame or file path
            index_column: Column for row labels
            column_column: Column for column labels  
            values_column: Optional column for values (defaults to count)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if index_column not in df.columns or column_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        if values_column and values_column not in df.columns:
            raise ValueError(f"Values column '{values_column}' not found")
        
        if values_column:
            # Sum values by index and column
            grouped = df.group_by([index_column, column_column]).agg(
                pl.col(values_column).sum().alias("values")
            )
            result = grouped.pivot(
                index=index_column,
                columns=column_column,
                values="values",
                aggregate_function="first"
            )
        else:
            # Count occurrences
            grouped = df.group_by([index_column, column_column]).agg(
                pl.len().alias("count")
            )
            result = grouped.pivot(
                index=index_column,
                columns=column_column,
                values="count",
                aggregate_function="first"
            )
        
        return result
    
    def SUBTOTAL(self,
                 function_num: int,
                 data: Union[pl.DataFrame, str, Path],
                 column: str,
                 range_spec: Optional[str] = None) -> float:
        """
        SUBTOTAL function matching Google Sheets behavior.
        Formula: =SUBTOTAL(function_num, range)
        
        Args:
            function_num: Function code (1=AVERAGE, 9=SUM, 4=MAX, 5=MIN, 6=PRODUCT, 7=STDEV, 2=COUNT, 3=COUNTA)
            data: DataFrame or file path
            column: Column to operate on
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        values = df[column].drop_nulls()
        
        if function_num == 1:  # AVERAGE
            return float(values.mean())
        elif function_num == 2:  # COUNT (numbers only)
            return float(values.count())
        elif function_num == 3:  # COUNTA (non-empty values)
            return float(df[column].count())  # includes nulls check
        elif function_num == 4:  # MAX
            return float(values.max())
        elif function_num == 5:  # MIN
            return float(values.min())
        elif function_num == 6:  # PRODUCT
            result = 1.0
            for val in values.to_list():
                result *= float(val)
            return result
        elif function_num == 7:  # STDEV
            return float(values.std(ddof=1))
        elif function_num == 9:  # SUM
            return float(values.sum())
        else:
            raise ValueError(f"Unsupported function number: {function_num}")
    
    # Convenience wrapper functions for SUBTOTAL
    def SUBTOTAL_SUM(self, data: Union[pl.DataFrame, str, Path], column: str, range_spec: Optional[str] = None) -> float:
        """SUBTOTAL with SUM function (function_num=9)"""
        return self.SUBTOTAL(9, data, column, range_spec)
    
    def SUBTOTAL_AVERAGE(self, data: Union[pl.DataFrame, str, Path], column: str, range_spec: Optional[str] = None) -> float:
        """SUBTOTAL with AVERAGE function (function_num=1)"""
        return self.SUBTOTAL(1, data, column, range_spec)
    
    def SUBTOTAL_COUNT(self, data: Union[pl.DataFrame, str, Path], column: str, range_spec: Optional[str] = None) -> float:
        """SUBTOTAL with COUNT function (function_num=2)"""
        return self.SUBTOTAL(2, data, column, range_spec)
    
    def RANK_PARTITION(self,
                       data: Union[pl.DataFrame, str, Path],
                       value_column: str,
                       partition_column: str,
                       order_desc: bool = True,
                       range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        RANK_PARTITION function for ranking within partitions.
        Formula: Window function ranking values within groups
        
        Args:
            data: DataFrame or file path
            value_column: Column containing values to rank
            partition_column: Column to partition/group by
            order_desc: True for descending order (rank 1 = highest value)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns or partition_column not in df.columns:
            raise ValueError(f"Required columns not found")
        
        # Add rank within partitions
        if order_desc:
            result = df.with_columns(
                pl.col(value_column).rank(method="ordinal", descending=True).over(partition_column).alias("rank")
            )
        else:
            result = df.with_columns(
                pl.col(value_column).rank(method="ordinal", descending=False).over(partition_column).alias("rank")
            )
        
        return result.sort([partition_column, "rank"])
    
    def DENSE_RANK(self,
                   data: Union[pl.DataFrame, str, Path],
                   value_column: str,
                   partition_column: Optional[str] = None,
                   order_desc: bool = True,
                   range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        DENSE_RANK function for dense ranking (no gaps in ranks).
        Formula: Dense ranking where tied values get same rank, next rank is consecutive
        
        Args:
            data: DataFrame or file path
            value_column: Column containing values to rank
            partition_column: Optional column to partition/group by
            order_desc: True for descending order
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if partition_column and partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found")
        
        # Add dense rank
        if partition_column:
            if order_desc:
                result = df.with_columns(
                    pl.col(value_column).rank(method="dense", descending=True).over(partition_column).alias("dense_rank")
                )
            else:
                result = df.with_columns(
                    pl.col(value_column).rank(method="dense", descending=False).over(partition_column).alias("dense_rank")
                )
            return result.sort([partition_column, "dense_rank"])
        else:
            if order_desc:
                result = df.with_columns(
                    pl.col(value_column).rank(method="dense", descending=True).alias("dense_rank")
                )
            else:
                result = df.with_columns(
                    pl.col(value_column).rank(method="dense", descending=False).alias("dense_rank")
                )
            return result.sort("dense_rank")
    
    def PERCENTILE_RANK(self,
                        data: Union[pl.DataFrame, str, Path],
                        value_column: str,
                        target_value: Optional[float] = None,
                        partition_column: Optional[str] = None,
                        range_spec: Optional[str] = None) -> Union[pl.DataFrame, float]:
        """
        PERCENTILE_RANK function for calculating percentile ranks.
        Formula: Returns percentile rank of values (0 to 1)
        
        Args:
            data: DataFrame or file path
            value_column: Column containing values
            target_value: Specific value to find percentile rank for (returns single float)
            partition_column: Optional column to partition by
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if partition_column and partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found")
        
        if target_value is not None:
            # Return percentile rank for specific value
            if partition_column:
                # For now, return overall percentile rank (could be enhanced per partition)
                values = df[value_column].drop_nulls().sort()
                count_below = values.filter(values < target_value).len()
                count_equal = values.filter(values == target_value).len()
                total_count = values.len()
                
                if total_count == 0:
                    return 0.0
                
                # Standard percentile rank formula
                percentile_rank = (count_below + 0.5 * count_equal) / total_count
                return float(percentile_rank)
            else:
                values = df[value_column].drop_nulls().sort()
                count_below = values.filter(values < target_value).len()
                count_equal = values.filter(values == target_value).len()
                total_count = values.len()
                
                if total_count == 0:
                    return 0.0
                
                percentile_rank = (count_below + 0.5 * count_equal) / total_count
                return float(percentile_rank)
        else:
            # Add percentile rank column to DataFrame
            if partition_column:
                result = df.with_columns(
                    pl.col(value_column).rank(method="average", descending=False).over(partition_column).alias("temp_rank")
                ).with_columns(
                    ((pl.col("temp_rank") - 1) / (pl.count().over(partition_column) - 1)).alias("percentile_rank")
                ).drop("temp_rank")
                
                return result.sort([partition_column, value_column])
            else:
                n = df.height
                result = df.with_columns(
                    pl.col(value_column).rank(method="average", descending=False).alias("temp_rank")
                ).with_columns(
                    ((pl.col("temp_rank") - 1) / (n - 1)).alias("percentile_rank")
                ).drop("temp_rank")
                
                return result.sort(value_column)
    
    def RANK(self,
             data: Union[pl.DataFrame, str, Path],
             value_column: str,
             order_desc: bool = True,
             range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        Simple RANK function matching Google Sheets RANK behavior.
        Formula: =RANK(value, array, order)
        
        Args:
            data: DataFrame or file path
            value_column: Column containing values to rank
            order_desc: True for descending order (1 = highest)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if order_desc:
            result = df.with_columns(
                pl.col(value_column).rank(method="ordinal", descending=True).alias("rank")
            )
        else:
            result = df.with_columns(
                pl.col(value_column).rank(method="ordinal", descending=False).alias("rank")
            )
        
        return result.sort("rank")
    
    def RUNNING_TOTAL(self,
                      data: Union[pl.DataFrame, str, Path],
                      value_column: str,
                      order_column: Optional[str] = None,
                      partition_column: Optional[str] = None,
                      range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        RUNNING_TOTAL function for cumulative sum calculations.
        Formula: Cumulative/running sum over ordered data
        
        Args:
            data: DataFrame or file path
            value_column: Column to calculate running total for
            order_column: Column to order by (optional, uses row order if None)
            partition_column: Column to partition by (optional)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if order_column and order_column not in df.columns:
            raise ValueError(f"Order column '{order_column}' not found")
        
        if partition_column and partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found")
        
        # Sort by order column if specified
        if order_column:
            if partition_column:
                df = df.sort([partition_column, order_column])
            else:
                df = df.sort(order_column)
        elif partition_column:
            df = df.sort(partition_column)
        
        # Calculate running total
        if partition_column:
            result = df.with_columns(
                pl.col(value_column).cum_sum().over(partition_column).alias("running_total")
            )
        else:
            result = df.with_columns(
                pl.col(value_column).cum_sum().alias("running_total")
            )
        
        return result
    
    def PERCENT_OF_TOTAL(self,
                         data: Union[pl.DataFrame, str, Path],
                         value_column: str,
                         partition_column: Optional[str] = None,
                         range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        PERCENT_OF_TOTAL function for calculating percentage of total.
        Formula: Each value as percentage of total (within partition if specified)
        
        Args:
            data: DataFrame or file path
            value_column: Column to calculate percentages for
            partition_column: Column to partition by (optional)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if partition_column and partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found")
        
        # Calculate percentage of total
        if partition_column:
            result = df.with_columns(
                (pl.col(value_column) / pl.col(value_column).sum().over(partition_column) * 100).alias("percent_of_total")
            )
        else:
            total = df[value_column].sum()
            result = df.with_columns(
                (pl.col(value_column) / total * 100).alias("percent_of_total")
            )
        
        return result
    
    def MOVING_SUM(self,
                   data: Union[pl.DataFrame, str, Path],
                   value_column: str,
                   window_size: int,
                   order_column: Optional[str] = None,
                   partition_column: Optional[str] = None,
                   range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        MOVING_SUM function for moving/rolling sum calculations.
        Formula: Rolling sum over specified window size
        
        Args:
            data: DataFrame or file path
            value_column: Column to calculate moving sum for
            window_size: Size of the moving window
            order_column: Column to order by (optional)
            partition_column: Column to partition by (optional)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if order_column and order_column not in df.columns:
            raise ValueError(f"Order column '{order_column}' not found")
        
        if partition_column and partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found")
        
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        
        # Sort by order column if specified
        if order_column:
            if partition_column:
                df = df.sort([partition_column, order_column])
            else:
                df = df.sort(order_column)
        elif partition_column:
            df = df.sort(partition_column)
        
        # Calculate moving sum
        if partition_column:
            result = df.with_columns(
                pl.col(value_column).rolling_sum(window_size=window_size).over(partition_column).alias("moving_sum")
            )
        else:
            result = df.with_columns(
                pl.col(value_column).rolling_sum(window_size=window_size).alias("moving_sum")
            )
        
        return result
    
    def MOVING_AVERAGE(self,
                       data: Union[pl.DataFrame, str, Path],
                       value_column: str,
                       window_size: int,
                       order_column: Optional[str] = None,
                       partition_column: Optional[str] = None,
                       range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        MOVING_AVERAGE function for moving/rolling average calculations.
        Formula: Rolling average over specified window size
        
        Args:
            data: DataFrame or file path
            value_column: Column to calculate moving average for
            window_size: Size of the moving window
            order_column: Column to order by (optional)
            partition_column: Column to partition by (optional)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if order_column and order_column not in df.columns:
            raise ValueError(f"Order column '{order_column}' not found")
        
        if partition_column and partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found")
        
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        
        # Sort by order column if specified
        if order_column:
            if partition_column:
                df = df.sort([partition_column, order_column])
            else:
                df = df.sort(order_column)
        elif partition_column:
            df = df.sort(partition_column)
        
        # Calculate moving average
        if partition_column:
            result = df.with_columns(
                pl.col(value_column).rolling_mean(window_size=window_size).over(partition_column).alias("moving_average")
            )
        else:
            result = df.with_columns(
                pl.col(value_column).rolling_mean(window_size=window_size).alias("moving_average")
            )
        
        return result
    
    def LAG(self,
            data: Union[pl.DataFrame, str, Path],
            value_column: str,
            offset: int = 1,
            default_value: Optional[Any] = None,
            order_column: Optional[str] = None,
            partition_column: Optional[str] = None,
            range_spec: Optional[str] = None) -> pl.DataFrame:
        """
        LAG function for accessing previous row values.
        Formula: Window function to get value from previous rows
        
        Args:
            data: DataFrame or file path
            value_column: Column to get lagged values from
            offset: Number of rows to look back (default 1)
            default_value: Value to use when no previous row exists
            order_column: Column to order by (optional)
            partition_column: Column to partition by (optional)
            range_spec: A1 notation range (optional)
        """
        df = self._load_data(data)
        if range_spec:
            df = self.resolver.resolve_range(df, range_spec)
        
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found")
        
        if order_column and order_column not in df.columns:
            raise ValueError(f"Order column '{order_column}' not found")
        
        if partition_column and partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found")
        
        # Sort by order column if specified
        if order_column:
            if partition_column:
                df = df.sort([partition_column, order_column])
            else:
                df = df.sort(order_column)
        elif partition_column:
            df = df.sort(partition_column)
        
        # Calculate lag
        if partition_column:
            result = df.with_columns(
                pl.col(value_column).shift(offset, fill_value=default_value).over(partition_column).alias("lag")
            )
        else:
            result = df.with_columns(
                pl.col(value_column).shift(offset, fill_value=default_value).alias("lag")
            )
        
        return result