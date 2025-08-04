"""
Data Transformation & Pivoting Functions

These functions help in performing data transformations and enabling dynamic results.
All functions use Polars for optimal performance and are optimized for AI agent integration.
"""

from typing import Any, List, Dict, Union
from pathlib import Path
import polars as pl
from tools.tool_exceptions import (
    FPABaseException,
    RetryAfterCorrectionError,
    ValidationError,
    CalculationError,
    ConfigurationError,
    DataQualityError,
)
from tools.toolset_utils import load_df, save_df_to_analysis_dir


def _validate_dataframe_input(df: Any, function_name: str) -> pl.DataFrame:
    """
    Standardized input validation for DataFrame data.

    Args:
        df: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.DataFrame: Validated Polars DataFrame

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains issues
    """
    if not isinstance(df, pl.DataFrame):
        raise ValidationError(f"Input must be a Polars DataFrame for {function_name}")

    if df.is_empty():
        raise ValidationError(f"Input DataFrame cannot be empty for {function_name}")

    return df


def _validate_columns_exist(df: pl.DataFrame, columns: List[str], function_name: str) -> None:
    """
    Validate that specified columns exist in the DataFrame.

    Args:
        df: DataFrame to check
        columns: List of column names to validate
        function_name: Name of calling function for error messages

    Raises:
        ValidationError: If any column doesn't exist
    """
    df_columns = set(df.columns)
    missing_columns = [col for col in columns if col not in df_columns]

    if missing_columns:
        raise ValidationError(
            f"Columns {missing_columns} not found in DataFrame for {function_name}. "
            f"Available columns: {list(df_columns)}"
        )


def _get_aggregation_expr(agg_func: str) -> pl.Expr:
    """
    Map aggregation function string to Polars expression.

    Args:
        agg_func: Aggregation function name

    Returns:
        pl.Expr: Polars aggregation expression

    Raises:
        ConfigurationError: If aggregation function is not supported
    """
    if agg_func.lower() not in ['sum', 'mean', 'avg', 'average', 'count', 'min', 'max', 'median', 'std', 'var', 'first', 'last']:
        raise ConfigurationError(
            f"Unsupported aggregation function: {agg_func}. "
            f"Supported functions: ['sum', 'mean', 'avg', 'average', 'count', 'min', 'max', 'median', 'std', 'var', 'first', 'last']"
        )

    # Return the appropriate expression based on function name
    if agg_func.lower() == 'sum':
        return pl.sum("*")
    elif agg_func.lower() in ['mean', 'avg', 'average']:
        return pl.mean("*")
    elif agg_func.lower() == 'count':
        return pl.count("*")
    elif agg_func.lower() == 'min':
        return pl.min("*")
    elif agg_func.lower() == 'max':
        return pl.max("*")
    elif agg_func.lower() == 'median':
        return pl.median("*")
    elif agg_func.lower() == 'std':
        return pl.std("*")
    elif agg_func.lower() == 'var':
        return pl.var("*")
    elif agg_func.lower() == 'first':
        return pl.first("*")
    elif agg_func.lower() == 'last':
        return pl.last("*")


def PIVOT_TABLE(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    index_cols: List[str],
    value_cols: List[str],
    agg_func: str,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Create pivot tables with aggregations by groups.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to pivot or file path
        index_cols: Index columns for grouping
        value_cols: Value columns to aggregate
        agg_func: Aggregation function ('sum', 'mean', 'count', etc.)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Pivoted DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If pivot operation fails

    Example:
        >>> PIVOT_TABLE(ctx, sales_df, index_cols=['region'], value_cols=['revenue'], agg_func='sum')
        DataFrame with regions as rows and aggregated revenue
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "PIVOT_TABLE")

    if not index_cols:
        raise ValidationError("Index columns cannot be empty for PIVOT_TABLE")

    if not value_cols:
        raise ValidationError("Value columns cannot be empty for PIVOT_TABLE")

    # Validate columns exist
    _validate_columns_exist(df, index_cols + value_cols, "PIVOT_TABLE")

    try:
        # Get aggregation expression
        agg_expr = _get_aggregation_expr(agg_func)

        # For simple pivot table, we'll group by index columns and aggregate value columns
        agg_exprs = []
        for col in value_cols:
            if agg_func.lower() == 'sum':
                agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
            elif agg_func.lower() in ['mean', 'avg', 'average']:
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            elif agg_func.lower() == 'count':
                agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
            elif agg_func.lower() == 'min':
                agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            elif agg_func.lower() == 'max':
                agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            elif agg_func.lower() == 'median':
                agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
            elif agg_func.lower() == 'std':
                agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
            elif agg_func.lower() == 'var':
                agg_exprs.append(pl.col(col).var().alias(f"{col}_var"))
            elif agg_func.lower() == 'first':
                agg_exprs.append(pl.col(col).first().alias(f"{col}_first"))
            elif agg_func.lower() == 'last':
                agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))

        result = df.group_by(index_cols).agg(agg_exprs)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"PIVOT_TABLE operation failed: {str(e)}")


def UNPIVOT(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    identifier_cols: List[str],
    value_cols: List[str],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Transform wide data to long format using melt operation.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to unpivot or file path
        identifier_cols: Identifier columns to keep
        value_cols: Value columns to melt
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Unpivoted DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If unpivot operation fails

    Example:
        >>> UNPIVOT(ctx, df, identifier_cols=['customer_id'], value_cols=['Q1', 'Q2', 'Q3', 'Q4'])
        DataFrame with customer_id, variable (Q1-Q4), and value columns
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "UNPIVOT")

    if not identifier_cols:
        raise ValidationError("Identifier columns cannot be empty for UNPIVOT")

    if not value_cols:
        raise ValidationError("Value columns cannot be empty for UNPIVOT")

    # Validate columns exist
    _validate_columns_exist(df, identifier_cols + value_cols, "UNPIVOT")

    try:
        # Use Polars melt operation
        result = df.melt(
            id_vars=identifier_cols,
            value_vars=value_cols,
            variable_name="variable",
            value_name="value"
        )

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"UNPIVOT operation failed: {str(e)}")


def GROUP_BY(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    grouping_cols: List[str],
    agg_func: str,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Group data and apply aggregation functions.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to group or file path
        grouping_cols: Grouping columns
        agg_func: Aggregation function ('sum', 'mean', 'count', etc.)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Grouped DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If groupby operation fails

    Example:
        >>> GROUP_BY(ctx, sales_df, grouping_cols=['category'], agg_func='sum')
        DataFrame grouped by category with aggregated values
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "GROUP_BY")

    if not grouping_cols:
        raise ValidationError("Grouping columns cannot be empty for GROUP_BY")

    # Validate columns exist
    _validate_columns_exist(df, grouping_cols, "GROUP_BY")

    try:
        # Get aggregation expression
        agg_expr = _get_aggregation_expr(agg_func)

        # Get numeric columns for aggregation (exclude grouping columns)
        numeric_cols = [col for col in df.columns
                       if col not in grouping_cols and df[col].dtype.is_numeric()]

        if not numeric_cols:
            raise ValidationError("No numeric columns found for aggregation in GROUP_BY")

        # Apply groupby with aggregation
        agg_exprs = []
        for col in numeric_cols:
            if agg_func.lower() == 'sum':
                agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
            elif agg_func.lower() in ['mean', 'avg', 'average']:
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            elif agg_func.lower() == 'count':
                agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
            elif agg_func.lower() == 'min':
                agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            elif agg_func.lower() == 'max':
                agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            elif agg_func.lower() == 'median':
                agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
            elif agg_func.lower() == 'std':
                agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
            elif agg_func.lower() == 'var':
                agg_exprs.append(pl.col(col).var().alias(f"{col}_var"))
            elif agg_func.lower() == 'first':
                agg_exprs.append(pl.col(col).first().alias(f"{col}_first"))
            elif agg_func.lower() == 'last':
                agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))

        result = df.group_by(grouping_cols).agg(agg_exprs)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"GROUP_BY operation failed: {str(e)}")


def CROSS_TAB(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    row_vars: List[str],
    col_vars: List[str],
    values: List[str],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Create cross-tabulation tables.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to cross-tabulate or file path
        row_vars: Row variables
        col_vars: Column variables
        values: Values to aggregate
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Cross-tabulated DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If cross-tabulation fails

    Example:
        >>> CROSS_TAB(ctx, df, row_vars=['region'], col_vars=['product'], values=['sales'])
        Cross-tabulation matrix with regions as rows and products as columns
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "CROSS_TAB")

    if not row_vars:
        raise ValidationError("Row variables cannot be empty for CROSS_TAB")

    if not col_vars:
        raise ValidationError("Column variables cannot be empty for CROSS_TAB")

    if not values:
        raise ValidationError("Values cannot be empty for CROSS_TAB")

    # Validate columns exist
    _validate_columns_exist(df, row_vars + col_vars + values, "CROSS_TAB")

    try:
        # For cross-tabulation, we'll use pivot with sum aggregation
        # First, ensure we have a single column variable for pivot
        if len(col_vars) > 1:
            # Combine multiple column variables into one
            df = df.with_columns(
                pl.concat_str(col_vars, separator="_").alias("_combined_col_var")
            )
            pivot_col = "_combined_col_var"
        else:
            pivot_col = col_vars[0]

        # Use the first value column for pivot
        value_col = values[0]

        # Create pivot table
        result = df.pivot(
            values=value_col,
            index=row_vars,
            columns=pivot_col,
            aggregate_function="sum"
        )

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"CROSS_TAB operation failed: {str(e)}")


def GROUP_BY_AGG(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    group_by_cols: List[str],
    agg_dict: Dict[str, str],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Group a DataFrame by one or more columns and apply multiple aggregation functions.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to group or file path
        group_by_cols: List of columns to group by
        agg_dict: Dictionary of column-aggregation function pairs
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Grouped DataFrame with multiple aggregations or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If groupby aggregation fails

    Example:
        >>> GROUP_BY_AGG(ctx, df, group_by_cols=['region'], agg_dict={'revenue': 'sum', 'users': 'count'})
        DataFrame grouped by region with sum of revenue and count of users
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "GROUP_BY_AGG")

    if not group_by_cols:
        raise ValidationError("Group by columns cannot be empty for GROUP_BY_AGG")

    if not agg_dict:
        raise ValidationError("Aggregation dictionary cannot be empty for GROUP_BY_AGG")

    # Validate columns exist
    agg_columns = list(agg_dict.keys())
    _validate_columns_exist(df, group_by_cols + agg_columns, "GROUP_BY_AGG")

    try:
        # Build aggregation expressions directly
        agg_exprs = []
        for col, agg_func in agg_dict.items():
            if agg_func.lower() == 'sum':
                agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
            elif agg_func.lower() in ['mean', 'avg', 'average']:
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            elif agg_func.lower() == 'count':
                agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
            elif agg_func.lower() == 'min':
                agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            elif agg_func.lower() == 'max':
                agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            elif agg_func.lower() == 'median':
                agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
            elif agg_func.lower() == 'std':
                agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
            elif agg_func.lower() == 'var':
                agg_exprs.append(pl.col(col).var().alias(f"{col}_var"))
            elif agg_func.lower() == 'first':
                agg_exprs.append(pl.col(col).first().alias(f"{col}_first"))
            elif agg_func.lower() == 'last':
                agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))
            else:
                raise ConfigurationError(f"Unsupported aggregation function: {agg_func}")

        # Apply groupby with multiple aggregations
        result = df.group_by(group_by_cols).agg(agg_exprs)

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"GROUP_BY_AGG operation failed: {str(e)}")


def STACK(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    columns_to_stack: List[str],
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Stack multiple columns into single column.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to stack or file path
        columns_to_stack: Columns to stack
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Stacked DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If stack operation fails

    Example:
        >>> STACK(ctx, df, columns_to_stack=['Q1', 'Q2', 'Q3', 'Q4'])
        DataFrame with quarters stacked into single column
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "STACK")

    if not columns_to_stack:
        raise ValidationError("Columns to stack cannot be empty for STACK")

    # Validate columns exist
    _validate_columns_exist(df, columns_to_stack, "STACK")

    try:
        # Get identifier columns (all columns except those to stack)
        id_cols = [col for col in df.columns if col not in columns_to_stack]

        # Use melt operation to stack columns
        result = df.melt(
            id_vars=id_cols,
            value_vars=columns_to_stack,
            variable_name="stacked_variable",
            value_name="stacked_value"
        )

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"STACK operation failed: {str(e)}")


def UNSTACK(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    level_to_unstack: str,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Unstack index level to columns.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame to unstack or file path
        level_to_unstack: Level to unstack
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Unstacked DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If unstack operation fails

    Example:
        >>> UNSTACK(ctx, stacked_df, level_to_unstack='quarter')
        DataFrame with quarter values as columns
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "UNSTACK")

    if not level_to_unstack:
        raise ValidationError("Level to unstack cannot be empty for UNSTACK")

    # Validate column exists
    _validate_columns_exist(df, [level_to_unstack], "UNSTACK")

    try:
        # For unstacking, we need to identify the value column and index columns
        # Assume the last column is the value column and others are index columns
        value_col = df.columns[-1]
        index_cols = [col for col in df.columns if col not in [level_to_unstack, value_col]]

        if not index_cols:
            # If no index columns, create a row number
            df = df.with_row_count("row_id")
            index_cols = ["row_id"]

        # Use pivot operation to unstack
        result = df.pivot(
            values=value_col,
            index=index_cols,
            columns=level_to_unstack,
            aggregate_function="first"  # Use first value if duplicates
        )

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"UNSTACK operation failed: {str(e)}")


def MERGE(
    run_context: Any,
    left_df: Union[pl.DataFrame, str, Path],
    right_df: Union[pl.DataFrame, str, Path],
    *,
    join_keys: Union[str, List[str]],
    join_type: str,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Merge/join two DataFrames.

    Args:
        run_context: RunContext object for file operations
        left_df: Left DataFrame or file path
        right_df: Right DataFrame or file path
        join_keys: Join keys (single string or list of strings)
        join_type: Join type ('inner', 'left', 'right', 'full', 'cross', 'semi', 'anti')
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Merged DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If merge operation fails

    Example:
        >>> MERGE(ctx, sales_df, customer_df, join_keys='customer_id', join_type='left')
        DataFrame with sales data merged with customer data
    """
    # Handle file path inputs
    if isinstance(left_df, (str, Path)):
        left_df = load_df(run_context, left_df)
    if isinstance(right_df, (str, Path)):
        right_df = load_df(run_context, right_df)

    # Input validation
    left_df = _validate_dataframe_input(left_df, "MERGE")
    right_df = _validate_dataframe_input(right_df, "MERGE")

    # Normalize join_keys to list
    if isinstance(join_keys, str):
        join_keys = [join_keys]

    if not join_keys:
        raise ValidationError("Join keys cannot be empty for MERGE")

    # Validate join type
    valid_join_types = ['inner', 'left', 'right', 'full', 'cross', 'semi', 'anti']
    if join_type.lower() not in valid_join_types:
        raise ValidationError(
            f"Invalid join type: {join_type}. Valid types: {valid_join_types}"
        )

    # Validate join keys exist in both DataFrames
    _validate_columns_exist(left_df, join_keys, "MERGE (left DataFrame)")
    _validate_columns_exist(right_df, join_keys, "MERGE (right DataFrame)")

    try:
        # Perform the join
        result = left_df.join(
            right_df,
            on=join_keys,
            how=join_type.lower(),
            suffix="_right"
        )

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"MERGE operation failed: {str(e)}")


def CONCAT(
    run_context: Any,
    dataframes: List[Union[pl.DataFrame, str, Path]],
    *,
    axis: int,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Concatenate DataFrames.

    Args:
        run_context: RunContext object for file operations
        dataframes: List of DataFrames or file paths
        axis: Axis to concatenate on (0 for vertical, 1 for horizontal)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: Concatenated DataFrame or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If concatenation fails

    Example:
        >>> CONCAT(ctx, [df1, df2, df3], axis=0)
        DataFrame with all DataFrames stacked vertically
    """
    if not dataframes:
        raise ValidationError("DataFrames list cannot be empty for CONCAT")

    if axis not in [0, 1]:
        raise ValidationError("Axis must be 0 (vertical) or 1 (horizontal) for CONCAT")

    try:
        # Load DataFrames from file paths if needed
        loaded_dfs = []
        for df in dataframes:
            if isinstance(df, (str, Path)):
                loaded_dfs.append(load_df(run_context, df))
            else:
                loaded_dfs.append(_validate_dataframe_input(df, "CONCAT"))

        # Perform concatenation
        if axis == 0:
            # Vertical concatenation
            result = pl.concat(loaded_dfs, how="vertical")
        else:
            # Horizontal concatenation
            result = pl.concat(loaded_dfs, how="horizontal")

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"CONCAT operation failed: {str(e)}")


def FILL_FORWARD(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Forward fill missing values.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame or Series to fill or file path
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with filled values or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If fill operation fails

    Example:
        >>> FILL_FORWARD(ctx, revenue_series)
        DataFrame with null values filled using previous non-null values
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "FILL_FORWARD")

    try:
        # Apply forward fill to all columns
        result = df.fill_null(strategy="forward")

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"FILL_FORWARD operation failed: {str(e)}")


def INTERPOLATE(
    run_context: Any,
    df: Union[pl.DataFrame, str, Path],
    *,
    method: str,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Interpolate missing values.

    Args:
        run_context: RunContext object for file operations
        df: DataFrame or Series to interpolate or file path
        method: Interpolation method ('linear')
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with interpolated values or path to saved file

    Raises:
        ValidationError: If input parameters are invalid
        CalculationError: If interpolation fails

    Example:
        >>> INTERPOLATE(ctx, data_series, method='linear')
        DataFrame with null values interpolated using linear method
    """
    # Handle file path input
    if isinstance(df, (str, Path)):
        df = load_df(run_context, df)

    # Input validation
    df = _validate_dataframe_input(df, "INTERPOLATE")

    # Validate interpolation method
    valid_methods = ['linear']
    if method.lower() not in valid_methods:
        raise ValidationError(
            f"Invalid interpolation method: {method}. Valid methods: {valid_methods}"
        )

    try:
        # Apply interpolation to numeric columns only
        numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]

        if not numeric_cols:
            raise ValidationError("No numeric columns found for interpolation")

        # Apply interpolation
        result = df.with_columns([
            pl.col(col).interpolate() for col in numeric_cols
        ])

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result, output_filename)

        return result

    except Exception as e:
        raise CalculationError(f"INTERPOLATE operation failed: {str(e)}")
