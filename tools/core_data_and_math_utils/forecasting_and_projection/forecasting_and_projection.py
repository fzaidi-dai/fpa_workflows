"""
Forecasting & Projection Functions

These functions help in performing forecasting and projections for financial planning and analysis.
All functions use appropriate precision for financial accuracy and are optimized for AI agent integration.
"""

from decimal import Decimal, getcontext
from typing import Any, Union, Dict, List
from pathlib import Path
import polars as pl
import numpy as np
from scipy import signal, stats
from scipy.stats import linregress
import math
import warnings

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

# Suppress statsmodels warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


def _validate_numeric_input(values: Any, function_name: str) -> pl.Series:
    """
    Standardized input validation for numeric data.

    Args:
        values: Input data to validate
        function_name: Name of calling function for error messages

    Returns:
        pl.Series: Validated Polars Series

    Raises:
        ValidationError: If input is invalid
        DataQualityError: If data contains invalid numeric values
    """
    try:
        # Convert to Polars Series for optimal processing
        if isinstance(values, (list, np.ndarray)):
            series = pl.Series(values)
        elif isinstance(values, pl.Series):
            series = values
        else:
            raise ValidationError(f"Unsupported input type for {function_name}: {type(values)}")

        # Check if series is empty
        if series.is_empty():
            raise ValidationError(f"Input values cannot be empty for {function_name}")

        # Check for null values
        if series.null_count() > 0:
            raise DataQualityError(
                f"Input contains null values for {function_name}",
                "Remove or replace null values with appropriate numeric values"
            )

        return series

    except (ValueError, TypeError) as e:
        raise DataQualityError(
            f"Invalid numeric values in {function_name}: {str(e)}",
            "Ensure all values are numeric (int, float, Decimal)"
        )


def _convert_to_decimal(value: Any) -> Decimal:
    """
    Safely convert value to Decimal with proper error handling.

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
        return Decimal(str(value))
    except (ValueError, TypeError, OverflowError) as e:
        raise DataQualityError(
            f"Cannot convert value to Decimal: {str(e)}",
            "Ensure value is a valid numeric type"
        )


def LINEAR_FORECAST(
    run_context: Any,
    historical_data: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    forecast_periods: int,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Simple linear trend forecasting using least squares regression.

    Args:
        run_context: RunContext object for file operations
        historical_data: Historical data series (DataFrame, Series, list, or file path)
        forecast_periods: Number of periods to forecast
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with forecasted values or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If regression calculation fails

    Example:
        >>> LINEAR_FORECAST(ctx, [100, 105, 110, 115, 120], forecast_periods=3)
        DataFrame with historical and forecasted values
        >>> LINEAR_FORECAST(ctx, "sales_data.parquet", forecast_periods=12, output_filename="forecast.parquet")
        Path to saved forecast file
    """
    # Handle file path input
    if isinstance(historical_data, (str, Path)):
        df = load_df(run_context, historical_data)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(historical_data, pl.DataFrame):
            series = historical_data[historical_data.columns[0]]
        else:
            series = _validate_numeric_input(historical_data, "LINEAR_FORECAST")

    if forecast_periods <= 0:
        raise ValidationError("Forecast periods must be positive")

    try:
        # Convert to numpy for regression calculation
        values = series.to_numpy()
        n = len(values)

        if n < 2:
            raise CalculationError("Linear regression requires at least 2 data points")

        # Create time index (0, 1, 2, ..., n-1)
        x = np.arange(n)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, values)

        # Generate forecasts
        forecast_x = np.arange(n, n + forecast_periods)
        forecasts = slope * forecast_x + intercept

        # Create result DataFrame
        historical_x = np.arange(n)
        historical_fitted = slope * historical_x + intercept

        # Combine historical and forecast data
        all_periods = list(range(n + forecast_periods))
        all_values = list(values) + [None] * forecast_periods
        all_fitted = list(historical_fitted) + list(forecasts)
        all_forecasts = [None] * n + list(forecasts)

        result_df = pl.DataFrame({
            "period": all_periods,
            "actual": all_values,
            "fitted": all_fitted,
            "forecast": all_forecasts,
            "trend_slope": [slope] * (n + forecast_periods),
            "r_squared": [r_value**2] * (n + forecast_periods)
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Linear forecast calculation failed: {str(e)}")


def MOVING_AVERAGE(
    run_context: Any,
    data_series: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    window_size: int,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Calculate moving averages for smoothing and forecasting using Polars rolling operations.

    Args:
        run_context: RunContext object for file operations
        data_series: Data series (DataFrame, Series, list, or file path)
        window_size: Window size for moving average
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with moving averages or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If moving average calculation fails

    Example:
        >>> MOVING_AVERAGE(ctx, [10, 12, 14, 16, 18, 20], window_size=3)
        DataFrame with moving averages
        >>> MOVING_AVERAGE(ctx, "revenue_data.parquet", window_size=12, output_filename="ma_results.parquet")
        Path to saved moving average file
    """
    # Handle file path input
    if isinstance(data_series, (str, Path)):
        df = load_df(run_context, data_series)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(data_series, pl.DataFrame):
            series = data_series[data_series.columns[0]]
        else:
            series = _validate_numeric_input(data_series, "MOVING_AVERAGE")

    if window_size <= 0:
        raise ValidationError("Window size must be positive")

    if window_size > len(series):
        raise ValidationError("Window size cannot be larger than data series length")

    try:
        # Calculate moving average using Polars rolling operations
        moving_avg = series.rolling_mean(window_size=window_size)

        # Create result DataFrame
        result_df = pl.DataFrame({
            "period": list(range(len(series))),
            "value": series,
            "moving_average": moving_avg,
            "window_size": [window_size] * len(series)
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Moving average calculation failed: {str(e)}")


def EXPONENTIAL_SMOOTHING(
    run_context: Any,
    data_series: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    smoothing_alpha: float,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Exponentially weighted forecasting for trend analysis.

    Args:
        run_context: RunContext object for file operations
        data_series: Data series (DataFrame, Series, list, or file path)
        smoothing_alpha: Smoothing parameter (0 < alpha <= 1)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with smoothed values or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If smoothing calculation fails

    Example:
        >>> EXPONENTIAL_SMOOTHING(ctx, [100, 105, 102, 108, 110], smoothing_alpha=0.3)
        DataFrame with exponentially smoothed values
        >>> EXPONENTIAL_SMOOTHING(ctx, "sales_data.parquet", smoothing_alpha=0.2, output_filename="smoothed.parquet")
        Path to saved smoothed data file
    """
    # Handle file path input
    if isinstance(data_series, (str, Path)):
        df = load_df(run_context, data_series)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(data_series, pl.DataFrame):
            series = data_series[data_series.columns[0]]
        else:
            series = _validate_numeric_input(data_series, "EXPONENTIAL_SMOOTHING")

    if not (0 < smoothing_alpha <= 1):
        raise ValidationError("Smoothing alpha must be between 0 and 1")

    try:
        values = series.to_list()
        smoothed = []

        # Initialize with first value
        smoothed.append(values[0])

        # Apply exponential smoothing formula: S_t = α * X_t + (1-α) * S_{t-1}
        for i in range(1, len(values)):
            smoothed_value = smoothing_alpha * values[i] + (1 - smoothing_alpha) * smoothed[i-1]
            smoothed.append(smoothed_value)

        # Create result DataFrame with explicit schema
        result_df = pl.DataFrame({
            "period": pl.Series(list(range(len(values))), dtype=pl.Int64),
            "actual": pl.Series(values, dtype=pl.Float64),
            "smoothed": pl.Series(smoothed, dtype=pl.Float64),
            "alpha": pl.Series([smoothing_alpha] * len(values), dtype=pl.Float64)
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Exponential smoothing calculation failed: {str(e)}")


def SEASONAL_DECOMPOSE(
    run_context: Any,
    time_series_data: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    seasonal_periods: int,
    model: str = "additive",
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Decompose time series into trend, seasonal, residual components using moving averages.

    Args:
        run_context: RunContext object for file operations
        time_series_data: Time series data (DataFrame, Series, list, or file path)
        seasonal_periods: Number of periods in a season (e.g., 12 for monthly data)
        model: Decomposition model ("additive" or "multiplicative")
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with decomposed components or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If decomposition fails

    Example:
        >>> SEASONAL_DECOMPOSE(ctx, quarterly_sales, seasonal_periods=4)
        DataFrame with trend, seasonal, and residual components
        >>> SEASONAL_DECOMPOSE(ctx, "monthly_data.parquet", seasonal_periods=12, output_filename="decomposed.parquet")
        Path to saved decomposition file
    """
    # Handle file path input
    if isinstance(time_series_data, (str, Path)):
        df = load_df(run_context, time_series_data)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(time_series_data, pl.DataFrame):
            series = time_series_data[time_series_data.columns[0]]
        else:
            series = _validate_numeric_input(time_series_data, "SEASONAL_DECOMPOSE")

    if seasonal_periods <= 0:
        raise ValidationError("Seasonal periods must be positive")

    if len(series) < 2 * seasonal_periods:
        raise ValidationError("Time series must have at least 2 complete seasonal cycles")

    if model not in ["additive", "multiplicative"]:
        raise ValidationError("Model must be 'additive' or 'multiplicative'")

    try:
        values = series.to_numpy()
        n = len(values)

        # Calculate trend using centered moving average
        trend = np.full(n, np.nan)
        half_window = seasonal_periods // 2

        for i in range(half_window, n - half_window):
            if seasonal_periods % 2 == 0:
                # Even seasonal periods - use weighted average
                window_sum = np.sum(values[i-half_window:i+half_window+1])
                window_sum -= 0.5 * values[i-half_window]
                window_sum -= 0.5 * values[i+half_window]
                trend[i] = window_sum / seasonal_periods
            else:
                # Odd seasonal periods - simple average
                trend[i] = np.mean(values[i-half_window:i+half_window+1])

        # Calculate seasonal component
        if model == "additive":
            detrended = values - trend
        else:  # multiplicative
            detrended = values / trend

        # Calculate seasonal indices
        seasonal_indices = np.full(seasonal_periods, np.nan)
        for s in range(seasonal_periods):
            season_values = []
            for i in range(s, n, seasonal_periods):
                if not np.isnan(detrended[i]):
                    season_values.append(detrended[i])
            if season_values:
                seasonal_indices[s] = np.mean(season_values)

        # Normalize seasonal indices
        if model == "additive":
            seasonal_indices = seasonal_indices - np.mean(seasonal_indices)
        else:  # multiplicative
            seasonal_indices = seasonal_indices / np.mean(seasonal_indices)

        # Create full seasonal component
        seasonal = np.tile(seasonal_indices, n // seasonal_periods + 1)[:n]

        # Calculate residual
        if model == "additive":
            residual = values - trend - seasonal
        else:  # multiplicative
            residual = values / (trend * seasonal)

        # Create result DataFrame
        result_df = pl.DataFrame({
            "period": list(range(n)),
            "observed": values,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "model": [model] * n,
            "seasonal_periods": [seasonal_periods] * n
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Seasonal decomposition failed: {str(e)}")


def SEASONAL_ADJUST(
    run_context: Any,
    time_series: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    seasonal_periods: int,
    model: str = "additive",
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Remove seasonal patterns from time series.

    Args:
        run_context: RunContext object for file operations
        time_series: Time series data (DataFrame, Series, list, or file path)
        seasonal_periods: Number of seasonal periods
        model: Adjustment model ("additive" or "multiplicative")
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with seasonally adjusted series or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If seasonal adjustment fails

    Example:
        >>> SEASONAL_ADJUST(ctx, monthly_sales, seasonal_periods=12)
        DataFrame with seasonally adjusted values
        >>> SEASONAL_ADJUST(ctx, "monthly_data.parquet", seasonal_periods=12, output_filename="adjusted.parquet")
        Path to saved adjusted data file
    """
    try:
        # Use SEASONAL_DECOMPOSE to get components
        decomposed_df = SEASONAL_DECOMPOSE(
            run_context, time_series,
            seasonal_periods=seasonal_periods,
            model=model
        )

        # Extract components
        observed = decomposed_df["observed"].to_numpy()
        seasonal = decomposed_df["seasonal"].to_numpy()

        # Calculate seasonally adjusted series
        if model == "additive":
            adjusted = observed - seasonal
        else:  # multiplicative
            adjusted = observed / seasonal

        # Create result DataFrame
        result_df = pl.DataFrame({
            "period": list(range(len(observed))),
            "original": observed,
            "seasonal_component": seasonal,
            "seasonally_adjusted": adjusted,
            "model": [model] * len(observed),
            "seasonal_periods": [seasonal_periods] * len(observed)
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Seasonal adjustment failed: {str(e)}")


def TREND_COEFFICIENT(
    run_context: Any,
    time_series_data: Union[pl.DataFrame, pl.Series, list, str, Path]
) -> Decimal:
    """
    Calculate trend coefficient (slope per period) using linear regression.

    Args:
        run_context: RunContext object for file operations
        time_series_data: Time series data (DataFrame, Series, list, or file path)

    Returns:
        Decimal: Trend coefficient (slope per period)

    Raises:
        ValidationError: If input is invalid
        CalculationError: If trend calculation fails

    Example:
        >>> TREND_COEFFICIENT(ctx, [100, 105, 110, 115, 120])
        Decimal('5.0')
        >>> TREND_COEFFICIENT(ctx, "quarterly_revenue.parquet")
        Decimal('2.5')
    """
    # Handle file path input
    if isinstance(time_series_data, (str, Path)):
        df = load_df(run_context, time_series_data)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(time_series_data, pl.DataFrame):
            series = time_series_data[time_series_data.columns[0]]
        else:
            series = _validate_numeric_input(time_series_data, "TREND_COEFFICIENT")

    try:
        values = series.to_numpy()
        n = len(values)

        if n < 2:
            raise CalculationError("Trend coefficient calculation requires at least 2 data points")

        # Create time index (0, 1, 2, ..., n-1)
        x = np.arange(n)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, values)

        return _convert_to_decimal(slope)

    except Exception as e:
        raise CalculationError(f"Trend coefficient calculation failed: {str(e)}")


def CYCLICAL_PATTERN(
    run_context: Any,
    time_series: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    cycle_length: int,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Identify cyclical patterns in data using autocorrelation analysis.

    Args:
        run_context: RunContext object for file operations
        time_series: Time series data (DataFrame, Series, list, or file path)
        cycle_length: Expected cycle length to analyze
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with cyclical indicators or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If cyclical analysis fails

    Example:
        >>> CYCLICAL_PATTERN(ctx, economic_data, cycle_length=60)
        DataFrame with cyclical pattern analysis
        >>> CYCLICAL_PATTERN(ctx, "economic_data.parquet", cycle_length=60, output_filename="cycles.parquet")
        Path to saved cyclical analysis file
    """
    # Handle file path input
    if isinstance(time_series, (str, Path)):
        df = load_df(run_context, time_series)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(time_series, pl.DataFrame):
            series = time_series[time_series.columns[0]]
        else:
            series = _validate_numeric_input(time_series, "CYCLICAL_PATTERN")

    if cycle_length <= 0:
        raise ValidationError("Cycle length must be positive")

    if cycle_length >= len(series):
        raise ValidationError("Cycle length must be less than series length")

    try:
        values = series.to_numpy()
        n = len(values)

        # Detrend the data first
        x = np.arange(n)
        slope, intercept, _, _, _ = linregress(x, values)
        trend = slope * x + intercept
        detrended = values - trend

        # Calculate autocorrelation for the expected cycle length
        autocorr_values = []
        max_lag = min(cycle_length * 2, n // 2)

        for lag in range(1, max_lag + 1):
            if lag < n:
                correlation = np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1]
                if not np.isnan(correlation):
                    autocorr_values.append(correlation)
                else:
                    autocorr_values.append(0.0)
            else:
                autocorr_values.append(0.0)

        # Find peaks in autocorrelation that might indicate cycles
        cycle_strength = 0.0
        if len(autocorr_values) >= cycle_length:
            cycle_strength = autocorr_values[cycle_length - 1]

        # Create cyclical indicators using sine wave approximation
        cycle_phase = 2 * np.pi * np.arange(n) / cycle_length
        cycle_indicator = cycle_strength * np.sin(cycle_phase)

        # Create result DataFrame
        result_df = pl.DataFrame({
            "period": list(range(n)),
            "original": values,
            "detrended": detrended,
            "cycle_indicator": cycle_indicator,
            "cycle_strength": [cycle_strength] * n,
            "cycle_length": [cycle_length] * n
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Cyclical pattern analysis failed: {str(e)}")


def AUTO_CORRELATION(
    run_context: Any,
    time_series: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    lags: int,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Calculate autocorrelation of time series using SciPy correlation functions.

    Args:
        run_context: RunContext object for file operations
        time_series: Time series data (DataFrame, Series, list, or file path)
        lags: Number of lags to calculate
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with correlation coefficients or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If autocorrelation calculation fails

    Example:
        >>> AUTO_CORRELATION(ctx, monthly_data, lags=12)
        DataFrame with autocorrelation coefficients
        >>> AUTO_CORRELATION(ctx, "monthly_data.parquet", lags=12, output_filename="autocorr.parquet")
        Path to saved autocorrelation file
    """
    # Handle file path input
    if isinstance(time_series, (str, Path)):
        df = load_df(run_context, time_series)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(time_series, pl.DataFrame):
            series = time_series[time_series.columns[0]]
        else:
            series = _validate_numeric_input(time_series, "AUTO_CORRELATION")

    if lags <= 0:
        raise ValidationError("Number of lags must be positive")

    if lags >= len(series):
        raise ValidationError("Number of lags must be less than series length")

    try:
        values = series.to_numpy()
        n = len(values)

        # Standardize the series (subtract mean, divide by std)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)

        if std_val == 0:
            raise CalculationError("Cannot calculate autocorrelation for constant series")

        standardized = (values - mean_val) / std_val

        # Calculate autocorrelation using numpy correlation
        autocorr_coeffs = []

        # Lag 0 is always 1.0
        autocorr_coeffs.append(1.0)

        # Calculate for each lag
        for lag in range(1, lags + 1):
            if lag < n:
                correlation = np.corrcoef(standardized[:-lag], standardized[lag:])[0, 1]
                if not np.isnan(correlation):
                    autocorr_coeffs.append(correlation)
                else:
                    autocorr_coeffs.append(0.0)
            else:
                autocorr_coeffs.append(0.0)

        # Create result DataFrame
        result_df = pl.DataFrame({
            "lag": list(range(len(autocorr_coeffs))),
            "autocorrelation": autocorr_coeffs,
            "series_length": [n] * len(autocorr_coeffs),
            "series_mean": [mean_val] * len(autocorr_coeffs),
            "series_std": [std_val] * len(autocorr_coeffs)
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Autocorrelation calculation failed: {str(e)}")


def HOLT_WINTERS(
    run_context: Any,
    time_series: Union[pl.DataFrame, pl.Series, list, str, Path],
    *,
    seasonal_periods: int,
    trend_type: str = "add",
    seasonal_type: str = "add",
    forecast_periods: int = 0,
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
    output_filename: str | None = None
) -> Union[pl.DataFrame, Path]:
    """
    Holt-Winters exponential smoothing (Triple exponential smoothing).

    Args:
        run_context: RunContext object for file operations
        time_series: Time series data (DataFrame, Series, list, or file path)
        seasonal_periods: Number of seasonal periods
        trend_type: Trend type ("add" for additive, "mul" for multiplicative, None for no trend)
        seasonal_type: Seasonal type ("add" for additive, "mul" for multiplicative)
        forecast_periods: Number of periods to forecast (default: 0)
        alpha: Level smoothing parameter (auto-optimized if None)
        beta: Trend smoothing parameter (auto-optimized if None)
        gamma: Seasonal smoothing parameter (auto-optimized if None)
        output_filename: Optional filename to save results as parquet file

    Returns:
        pl.DataFrame or Path: DataFrame with forecast and components or path to saved file

    Raises:
        ValidationError: If input is invalid
        CalculationError: If Holt-Winters calculation fails

    Example:
        >>> HOLT_WINTERS(ctx, quarterly_sales, seasonal_periods=4, trend_type="add", seasonal_type="add")
        DataFrame with Holt-Winters components and fitted values
        >>> HOLT_WINTERS(ctx, "quarterly_data.parquet", seasonal_periods=4, forecast_periods=8, output_filename="hw.parquet")
        Path to saved Holt-Winters results file
    """
    # Handle file path input
    if isinstance(time_series, (str, Path)):
        df = load_df(run_context, time_series)
        # Assume first column contains the numeric data
        series = df[df.columns[0]]
    else:
        # Input validation for direct data
        if isinstance(time_series, pl.DataFrame):
            series = time_series[time_series.columns[0]]
        else:
            series = _validate_numeric_input(time_series, "HOLT_WINTERS")

    if seasonal_periods <= 0:
        raise ValidationError("Seasonal periods must be positive")

    if len(series) < 2 * seasonal_periods:
        raise ValidationError("Time series must have at least 2 complete seasonal cycles")

    if trend_type not in ["add", "mul", None]:
        raise ValidationError("Trend type must be 'add', 'mul', or None")

    if seasonal_type not in ["add", "mul"]:
        raise ValidationError("Seasonal type must be 'add' or 'mul'")

    # Set default smoothing parameters if not provided
    if alpha is None:
        alpha = 0.3
    if beta is None:
        beta = 0.1
    if gamma is None:
        gamma = 0.1

    # Validate smoothing parameters
    if not (0 < alpha <= 1):
        raise ValidationError("Alpha must be between 0 and 1")
    if not (0 < beta <= 1):
        raise ValidationError("Beta must be between 0 and 1")
    if not (0 < gamma <= 1):
        raise ValidationError("Gamma must be between 0 and 1")

    try:
        values = series.to_numpy()
        n = len(values)

        # Initialize components
        level = np.zeros(n + forecast_periods)
        trend = np.zeros(n + forecast_periods)
        seasonal = np.zeros(n + forecast_periods)
        fitted = np.zeros(n + forecast_periods)

        # Initialize level (average of first seasonal cycle)
        level[0] = np.mean(values[:seasonal_periods])

        # Initialize trend (simple linear trend from first two cycles)
        if trend_type is not None and n >= 2 * seasonal_periods:
            first_cycle = np.mean(values[:seasonal_periods])
            second_cycle = np.mean(values[seasonal_periods:2*seasonal_periods])
            trend[0] = (second_cycle - first_cycle) / seasonal_periods
        else:
            trend[0] = 0

        # Initialize seasonal components
        for i in range(seasonal_periods):
            seasonal_values = []
            for j in range(i, n, seasonal_periods):
                if j < n:
                    seasonal_values.append(values[j])

            if seasonal_values:
                if seasonal_type == "add":
                    seasonal[i] = np.mean(seasonal_values) - level[0]
                else:  # multiplicative
                    seasonal[i] = np.mean(seasonal_values) / level[0] if level[0] != 0 else 1

        # Extend initial seasonal pattern
        for i in range(seasonal_periods, n + forecast_periods):
            seasonal[i] = seasonal[i % seasonal_periods]

        # Apply Holt-Winters equations
        for t in range(n):
            # Calculate fitted value
            if trend_type == "add":
                if seasonal_type == "add":
                    fitted[t] = level[t] + trend[t] + seasonal[t]
                else:  # multiplicative
                    fitted[t] = (level[t] + trend[t]) * seasonal[t]
            elif trend_type == "mul":
                if seasonal_type == "add":
                    fitted[t] = level[t] * trend[t] + seasonal[t]
                else:  # multiplicative
                    fitted[t] = level[t] * trend[t] * seasonal[t]
            else:  # no trend
                if seasonal_type == "add":
                    fitted[t] = level[t] + seasonal[t]
                else:  # multiplicative
                    fitted[t] = level[t] * seasonal[t]

            # Update components for next period
            if t < n - 1:
                # Update level
                if seasonal_type == "add":
                    level[t+1] = alpha * (values[t] - seasonal[t]) + (1 - alpha) * (level[t] + trend[t] if trend_type == "add" else level[t] * trend[t] if trend_type == "mul" else level[t])
                else:  # multiplicative
                    level[t+1] = alpha * (values[t] / seasonal[t] if seasonal[t] != 0 else values[t]) + (1 - alpha) * (level[t] + trend[t] if trend_type == "add" else level[t] * trend[t] if trend_type == "mul" else level[t])

                # Update trend
                if trend_type == "add":
                    trend[t+1] = beta * (level[t+1] - level[t]) + (1 - beta) * trend[t]
                elif trend_type == "mul":
                    trend[t+1] = beta * (level[t+1] / level[t] if level[t] != 0 else 1) + (1 - beta) * trend[t]
                else:
                    trend[t+1] = trend[t]

                # Update seasonal (only if within bounds)
                if t + seasonal_periods < len(seasonal):
                    if seasonal_type == "add":
                        seasonal[t+seasonal_periods] = gamma * (values[t] - level[t] - trend[t] if trend_type == "add" else values[t] - level[t] * trend[t] if trend_type == "mul" else values[t] - level[t]) + (1 - gamma) * seasonal[t]
                    else:  # multiplicative
                        denominator = level[t] + trend[t] if trend_type == "add" else level[t] * trend[t] if trend_type == "mul" else level[t]
                        seasonal[t+seasonal_periods] = gamma * (values[t] / denominator if denominator != 0 else 1) + (1 - gamma) * seasonal[t]

        # Generate forecasts
        forecasts = []
        if forecast_periods > 0:
            for h in range(1, forecast_periods + 1):
                if trend_type == "add":
                    if seasonal_type == "add":
                        forecast = level[n-1] + h * trend[n-1] + seasonal[n-1 + h % seasonal_periods]
                    else:  # multiplicative
                        forecast = (level[n-1] + h * trend[n-1]) * seasonal[n-1 + h % seasonal_periods]
                elif trend_type == "mul":
                    if seasonal_type == "add":
                        forecast = level[n-1] * (trend[n-1] ** h) + seasonal[n-1 + h % seasonal_periods]
                    else:  # multiplicative
                        forecast = level[n-1] * (trend[n-1] ** h) * seasonal[n-1 + h % seasonal_periods]
                else:  # no trend
                    if seasonal_type == "add":
                        forecast = level[n-1] + seasonal[n-1 + h % seasonal_periods]
                    else:  # multiplicative
                        forecast = level[n-1] * seasonal[n-1 + h % seasonal_periods]

                forecasts.append(forecast)

        # Create result DataFrame
        all_periods = list(range(n + forecast_periods))
        all_actual = list(values) + [None] * forecast_periods
        all_fitted = list(fitted[:n]) + forecasts
        all_level = list(level[:n + forecast_periods])
        all_trend = list(trend[:n + forecast_periods])
        all_seasonal = list(seasonal[:n + forecast_periods])
        all_forecasts = [None] * n + forecasts

        result_df = pl.DataFrame({
            "period": all_periods,
            "actual": all_actual,
            "fitted": all_fitted,
            "forecast": all_forecasts,
            "level": all_level,
            "trend": all_trend,
            "seasonal": all_seasonal,
            "alpha": [alpha] * (n + forecast_periods),
            "beta": [beta] * (n + forecast_periods),
            "gamma": [gamma] * (n + forecast_periods),
            "trend_type": [trend_type] * (n + forecast_periods),
            "seasonal_type": [seasonal_type] * (n + forecast_periods),
            "seasonal_periods": [seasonal_periods] * (n + forecast_periods)
        })

        # Save results to file if output_filename is provided
        if output_filename is not None:
            return save_df_to_analysis_dir(run_context, result_df, output_filename)

        return result_df

    except Exception as e:
        raise CalculationError(f"Holt-Winters calculation failed: {str(e)}")
