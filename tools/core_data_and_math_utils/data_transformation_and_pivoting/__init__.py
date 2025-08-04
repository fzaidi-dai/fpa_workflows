"""
Data Transformation and Pivoting Functions

This module provides atomic functions for data transformation and pivoting operations
optimized for financial planning and analysis (FP&A) applications.

All functions are designed as high-performance tools for AI agents with:
- Single responsibility principle
- Comprehensive error handling
- Polars-first approach for optimal performance
- Financial data validation
- AI agent integration support

Available Functions:
- PIVOT_TABLE: Create pivot tables with aggregations
- UNPIVOT: Transform wide data to long format
- GROUP_BY: Group data and apply aggregation functions
- CROSS_TAB: Create cross-tabulation tables
- GROUP_BY_AGG: Group with multiple aggregation functions
- STACK: Stack multiple columns into single column
- UNSTACK: Unstack index level to columns
- MERGE: Merge/join two DataFrames
- CONCAT: Concatenate DataFrames
- FILL_FORWARD: Forward fill missing values
- INTERPOLATE: Interpolate missing values
"""

from .data_transformation_and_pivoting import (
    PIVOT_TABLE,
    UNPIVOT,
    GROUP_BY,
    CROSS_TAB,
    GROUP_BY_AGG,
    STACK,
    UNSTACK,
    MERGE,
    CONCAT,
    FILL_FORWARD,
    INTERPOLATE,
)

__all__ = [
    "PIVOT_TABLE",
    "UNPIVOT",
    "GROUP_BY",
    "CROSS_TAB",
    "GROUP_BY_AGG",
    "STACK",
    "UNSTACK",
    "MERGE",
    "CONCAT",
    "FILL_FORWARD",
    "INTERPOLATE",
]
