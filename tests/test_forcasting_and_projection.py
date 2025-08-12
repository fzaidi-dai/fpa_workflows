"""
Test suite for Forecasting & Projection Functions

This test suite validates all forecasting and projection functions with comprehensive test cases
covering normal operations, edge cases, error conditions, and financial accuracy requirements.
"""

import pytest
import polars as pl
import numpy as np
from decimal import Decimal
from pathlib import Path
import tempfile
import os

# Import the functions to test
from tools.core_data_and_math_utils.forcasting_and_projection.forcasting_and_projection import (
    LINEAR_FORECAST,
    MOVING_AVERAGE,
    EXPONENTIAL_SMOOTHING,
    SEASONAL_DECOMPOSE,
    SEASONAL_ADJUST,
    TREND_COEFFICIENT,
    CYCLICAL_PATTERN,
    AUTO_CORRELATION,
    HOLT_WINTERS,
)

from tools.tool_exceptions import (
    ValidationError,
    CalculationError,
    DataQualityError,
)

from tools.finn_deps import FinnDeps, RunContext


@pytest.fixture
def mock_ctx():
    """Fixture providing mock run context"""
    thread_dir = Path("scratch_pad").resolve()
    workspace_dir = Path(".").resolve()
    finn_deps = FinnDeps(thread_dir=thread_dir, workspace_dir=workspace_dir)
    return RunContext(deps=finn_deps)


@pytest.fixture
def sample_trend_data():
    """Sample data with clear linear trend"""
    return [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]


@pytest.fixture
def sample_seasonal_data():
    """Sample data with seasonal pattern (quarterly)"""
    # 3 years of quarterly data with trend and seasonality
    base_trend = [100 + i * 2 for i in range(12)]
    seasonal_pattern = [10, -5, -10, 5] * 3  # Q1 high, Q2 low, Q3 lowest, Q4 medium
    return [base + seasonal for base, seasonal in zip(base_trend, seasonal_pattern)]


@pytest.fixture
def sample_noisy_data():
    """Sample data with noise for testing robustness"""
    np.random.seed(42)  # For reproducible tests
    trend = np.linspace(100, 200, 20)
    noise = np.random.normal(0, 5, 20)
    return (trend + noise).tolist()


class TestLinearForecast:
    """Test cases for LINEAR_FORECAST function"""

    def test_basic_linear_forecast(self, mock_ctx, sample_trend_data):
        """Test basic linear forecasting functionality"""
        result = LINEAR_FORECAST(
            mock_ctx,
            sample_trend_data,
            forecast_periods=3
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_trend_data) + 3
        assert "actual" in result.columns
        assert "fitted" in result.columns
        assert "forecast" in result.columns
        assert "trend_slope" in result.columns
        assert "r_squared" in result.columns

        # Check that forecasts are generated
        forecasts = result.filter(pl.col("forecast").is_not_null())["forecast"].to_list()
        assert len(forecasts) == 3

        # Check trend slope is positive (data has upward trend)
        slope = result["trend_slope"][0]
        assert slope > 0

    def test_linear_forecast_with_file_output(self, mock_ctx, sample_trend_data):
        """Test linear forecast with file output"""
        result = LINEAR_FORECAST(
            mock_ctx,
            sample_trend_data,
            forecast_periods=2,
            output_filename="linear_forecast_test.parquet"
        )

        assert isinstance(result, Path)
        assert result.name == "linear_forecast_test.parquet"

    def test_linear_forecast_validation_errors(self, mock_ctx):
        """Test validation error conditions"""
        # Empty data
        with pytest.raises(ValidationError):
            LINEAR_FORECAST(mock_ctx, [], forecast_periods=1)

        # Negative forecast periods
        with pytest.raises(ValidationError):
            LINEAR_FORECAST(mock_ctx, [1, 2, 3], forecast_periods=-1)

        # Zero forecast periods
        with pytest.raises(ValidationError):
            LINEAR_FORECAST(mock_ctx, [1, 2, 3], forecast_periods=0)

    def test_linear_forecast_single_point(self, mock_ctx):
        """Test with single data point"""
        with pytest.raises(CalculationError):
            LINEAR_FORECAST(mock_ctx, [100], forecast_periods=1)

    def test_linear_forecast_constant_data(self, mock_ctx):
        """Test with constant data (zero slope)"""
        constant_data = [100] * 10
        result = LINEAR_FORECAST(mock_ctx, constant_data, forecast_periods=2)

        # Slope should be approximately zero
        slope = result["trend_slope"][0]
        assert abs(slope) < 1e-10

        # Forecasts should be approximately equal to the constant value
        forecasts = result.filter(pl.col("forecast").is_not_null())["forecast"].to_list()
        for forecast in forecasts:
            assert abs(forecast - 100) < 1e-6


class TestMovingAverage:
    """Test cases for MOVING_AVERAGE function"""

    def test_basic_moving_average(self, mock_ctx, sample_trend_data):
        """Test basic moving average calculation"""
        result = MOVING_AVERAGE(
            mock_ctx,
            sample_trend_data,
            window_size=3
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_trend_data)
        assert "value" in result.columns
        assert "moving_average" in result.columns
        assert "window_size" in result.columns

        # Check that window size is correctly recorded
        assert all(ws == 3 for ws in result["window_size"].to_list())

        # First few values should be null (insufficient window)
        ma_values = result["moving_average"].to_list()
        assert ma_values[0] is None
        assert ma_values[1] is None
        assert ma_values[2] is not None

    def test_moving_average_window_validation(self, mock_ctx, sample_trend_data):
        """Test window size validation"""
        # Zero window size
        with pytest.raises(ValidationError):
            MOVING_AVERAGE(mock_ctx, sample_trend_data, window_size=0)

        # Negative window size
        with pytest.raises(ValidationError):
            MOVING_AVERAGE(mock_ctx, sample_trend_data, window_size=-1)

        # Window size larger than data
        with pytest.raises(ValidationError):
            MOVING_AVERAGE(mock_ctx, sample_trend_data, window_size=len(sample_trend_data) + 1)

    def test_moving_average_with_output_file(self, mock_ctx, sample_trend_data):
        """Test moving average with file output"""
        result = MOVING_AVERAGE(
            mock_ctx,
            sample_trend_data,
            window_size=3,
            output_filename="ma_test.parquet"
        )

        assert isinstance(result, Path)

    def test_moving_average_smoothing_effect(self, mock_ctx, sample_noisy_data):
        """Test that moving average smooths noisy data"""
        result = MOVING_AVERAGE(
            mock_ctx,
            sample_noisy_data,
            window_size=5
        )

        # Moving average should be smoother than original data
        original_values = result["value"].to_list()
        ma_values = result["moving_average"].drop_nulls().to_list()

        # Calculate variance of non-null moving average values
        original_var = np.var(original_values[4:])  # Skip first 4 to match MA
        ma_var = np.var(ma_values)

        # Moving average should have lower variance (smoother)
        assert ma_var < original_var


class TestExponentialSmoothing:
    """Test cases for EXPONENTIAL_SMOOTHING function"""

    def test_basic_exponential_smoothing(self, mock_ctx, sample_trend_data):
        """Test basic exponential smoothing"""
        result = EXPONENTIAL_SMOOTHING(
            mock_ctx,
            sample_trend_data,
            smoothing_alpha=0.3
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_trend_data)
        assert "actual" in result.columns
        assert "smoothed" in result.columns
        assert "alpha" in result.columns

        # Check alpha is correctly recorded
        assert all(alpha == 0.3 for alpha in result["alpha"].to_list())

        # First smoothed value should equal first actual value
        assert result["smoothed"][0] == result["actual"][0]

    def test_exponential_smoothing_alpha_validation(self, mock_ctx, sample_trend_data):
        """Test alpha parameter validation"""
        # Alpha too small
        with pytest.raises(ValidationError):
            EXPONENTIAL_SMOOTHING(mock_ctx, sample_trend_data, smoothing_alpha=0)

        # Alpha too large
        with pytest.raises(ValidationError):
            EXPONENTIAL_SMOOTHING(mock_ctx, sample_trend_data, smoothing_alpha=1.5)

        # Negative alpha
        with pytest.raises(ValidationError):
            EXPONENTIAL_SMOOTHING(mock_ctx, sample_trend_data, smoothing_alpha=-0.1)

    def test_exponential_smoothing_alpha_effects(self, mock_ctx, sample_noisy_data):
        """Test different alpha values produce different smoothing effects"""
        # High alpha (more responsive)
        result_high = EXPONENTIAL_SMOOTHING(
            mock_ctx,
            sample_noisy_data,
            smoothing_alpha=0.9
        )

        # Low alpha (more smoothed)
        result_low = EXPONENTIAL_SMOOTHING(
            mock_ctx,
            sample_noisy_data,
            smoothing_alpha=0.1
        )

        # High alpha should be closer to actual values
        high_diff = np.mean(np.abs(np.array(result_high["actual"].to_list()) -
                                  np.array(result_high["smoothed"].to_list())))
        low_diff = np.mean(np.abs(np.array(result_low["actual"].to_list()) -
                                 np.array(result_low["smoothed"].to_list())))

        assert high_diff < low_diff

    def test_exponential_smoothing_with_output(self, mock_ctx, sample_trend_data):
        """Test exponential smoothing with file output"""
        result = EXPONENTIAL_SMOOTHING(
            mock_ctx,
            sample_trend_data,
            smoothing_alpha=0.3,
            output_filename="exp_smooth_test.parquet"
        )

        assert isinstance(result, Path)


class TestSeasonalDecompose:
    """Test cases for SEASONAL_DECOMPOSE function"""

    def test_basic_seasonal_decompose(self, mock_ctx, sample_seasonal_data):
        """Test basic seasonal decomposition"""
        result = SEASONAL_DECOMPOSE(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_seasonal_data)
        assert "observed" in result.columns
        assert "trend" in result.columns
        assert "seasonal" in result.columns
        assert "residual" in result.columns
        assert "model" in result.columns
        assert "seasonal_periods" in result.columns

    def test_seasonal_decompose_additive_vs_multiplicative(self, mock_ctx, sample_seasonal_data):
        """Test additive vs multiplicative decomposition"""
        result_add = SEASONAL_DECOMPOSE(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4,
            model="additive"
        )

        result_mul = SEASONAL_DECOMPOSE(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4,
            model="multiplicative"
        )

        # Both should have same structure but different values
        assert len(result_add) == len(result_mul)
        assert result_add["model"][0] == "additive"
        assert result_mul["model"][0] == "multiplicative"

    def test_seasonal_decompose_validation(self, mock_ctx):
        """Test validation errors"""
        # Insufficient data
        short_data = [1, 2, 3, 4, 5]
        with pytest.raises(ValidationError):
            SEASONAL_DECOMPOSE(mock_ctx, short_data, seasonal_periods=4)

        # Invalid seasonal periods
        with pytest.raises(ValidationError):
            SEASONAL_DECOMPOSE(mock_ctx, sample_seasonal_data, seasonal_periods=0)

        # Invalid model
        with pytest.raises(ValidationError):
            SEASONAL_DECOMPOSE(mock_ctx, sample_seasonal_data, seasonal_periods=4, model="invalid")

    def test_seasonal_decompose_with_output(self, mock_ctx, sample_seasonal_data):
        """Test seasonal decomposition with file output"""
        result = SEASONAL_DECOMPOSE(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4,
            output_filename="decompose_test.parquet"
        )

        assert isinstance(result, Path)


class TestSeasonalAdjust:
    """Test cases for SEASONAL_ADJUST function"""

    def test_basic_seasonal_adjust(self, mock_ctx, sample_seasonal_data):
        """Test basic seasonal adjustment"""
        result = SEASONAL_ADJUST(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4
        )

        assert isinstance(result, pl.DataFrame)
        assert "original" in result.columns
        assert "seasonal_component" in result.columns
        assert "seasonally_adjusted" in result.columns
        assert "model" in result.columns

    def test_seasonal_adjust_removes_seasonality(self, mock_ctx, sample_seasonal_data):
        """Test that seasonal adjustment reduces seasonal variation"""
        result = SEASONAL_ADJUST(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4
        )

        original = result["original"].to_list()
        adjusted = result["seasonally_adjusted"].to_list()

        # Adjusted series should have lower variance than original
        original_var = np.var(original)
        adjusted_var = np.var(adjusted)

        assert adjusted_var < original_var

    def test_seasonal_adjust_with_output(self, mock_ctx, sample_seasonal_data):
        """Test seasonal adjustment with file output"""
        result = SEASONAL_ADJUST(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4,
            output_filename="seasonal_adjust_test.parquet"
        )

        assert isinstance(result, Path)


class TestTrendCoefficient:
    """Test cases for TREND_COEFFICIENT function"""

    def test_basic_trend_coefficient(self, mock_ctx, sample_trend_data):
        """Test basic trend coefficient calculation"""
        result = TREND_COEFFICIENT(mock_ctx, sample_trend_data)

        assert isinstance(result, Decimal)
        # Sample data has increment of 5 per period
        assert abs(float(result) - 5.0) < 0.1

    def test_trend_coefficient_negative_trend(self, mock_ctx):
        """Test trend coefficient with negative trend"""
        declining_data = [100, 95, 90, 85, 80, 75]
        result = TREND_COEFFICIENT(mock_ctx, declining_data)

        assert isinstance(result, Decimal)
        assert float(result) < 0  # Should be negative

    def test_trend_coefficient_no_trend(self, mock_ctx):
        """Test trend coefficient with no trend"""
        flat_data = [100] * 10
        result = TREND_COEFFICIENT(mock_ctx, flat_data)

        assert isinstance(result, Decimal)
        assert abs(float(result)) < 1e-10  # Should be approximately zero

    def test_trend_coefficient_validation(self, mock_ctx):
        """Test validation errors"""
        # Empty data
        with pytest.raises(ValidationError):
            TREND_COEFFICIENT(mock_ctx, [])

        # Single point
        with pytest.raises(CalculationError):
            TREND_COEFFICIENT(mock_ctx, [100])


class TestCyclicalPattern:
    """Test cases for CYCLICAL_PATTERN function"""

    def test_basic_cyclical_pattern(self, mock_ctx):
        """Test basic cyclical pattern detection"""
        # Create data with known cycle
        t = np.arange(50)
        cyclical_data = 100 + 10 * np.sin(2 * np.pi * t / 10) + 0.1 * t

        result = CYCLICAL_PATTERN(
            mock_ctx,
            cyclical_data.tolist(),
            cycle_length=10
        )

        assert isinstance(result, pl.DataFrame)
        assert "original" in result.columns
        assert "detrended" in result.columns
        assert "cycle_indicator" in result.columns
        assert "cycle_strength" in result.columns
        assert "cycle_length" in result.columns

    def test_cyclical_pattern_validation(self, mock_ctx, sample_trend_data):
        """Test validation errors"""
        # Invalid cycle length
        with pytest.raises(ValidationError):
            CYCLICAL_PATTERN(mock_ctx, sample_trend_data, cycle_length=0)

        # Cycle length too large
        with pytest.raises(ValidationError):
            CYCLICAL_PATTERN(mock_ctx, sample_trend_data, cycle_length=len(sample_trend_data))

    def test_cyclical_pattern_with_output(self, mock_ctx, sample_trend_data):
        """Test cyclical pattern with file output"""
        result = CYCLICAL_PATTERN(
            mock_ctx,
            sample_trend_data,
            cycle_length=3,
            output_filename="cyclical_test.parquet"
        )

        assert isinstance(result, Path)


class TestAutoCorrelation:
    """Test cases for AUTO_CORRELATION function"""

    def test_basic_autocorrelation(self, mock_ctx, sample_trend_data):
        """Test basic autocorrelation calculation"""
        result = AUTO_CORRELATION(
            mock_ctx,
            sample_trend_data,
            lags=5
        )

        assert isinstance(result, pl.DataFrame)
        assert "lag" in result.columns
        assert "autocorrelation" in result.columns
        assert "series_length" in result.columns
        assert "series_mean" in result.columns
        assert "series_std" in result.columns

        # Lag 0 should be 1.0
        lag_0_corr = result.filter(pl.col("lag") == 0)["autocorrelation"][0]
        assert abs(lag_0_corr - 1.0) < 1e-10

    def test_autocorrelation_validation(self, mock_ctx, sample_trend_data):
        """Test validation errors"""
        # Invalid lags
        with pytest.raises(ValidationError):
            AUTO_CORRELATION(mock_ctx, sample_trend_data, lags=0)

        # Too many lags
        with pytest.raises(ValidationError):
            AUTO_CORRELATION(mock_ctx, sample_trend_data, lags=len(sample_trend_data))

    def test_autocorrelation_constant_series(self, mock_ctx):
        """Test autocorrelation with constant series"""
        constant_data = [100] * 10
        with pytest.raises(CalculationError):
            AUTO_CORRELATION(mock_ctx, constant_data, lags=3)

    def test_autocorrelation_with_output(self, mock_ctx, sample_trend_data):
        """Test autocorrelation with file output"""
        result = AUTO_CORRELATION(
            mock_ctx,
            sample_trend_data,
            lags=3,
            output_filename="autocorr_test.parquet"
        )

        assert isinstance(result, Path)


class TestHoltWinters:
    """Test cases for HOLT_WINTERS function"""

    def test_basic_holt_winters(self, mock_ctx, sample_seasonal_data):
        """Test basic Holt-Winters smoothing"""
        result = HOLT_WINTERS(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4,
            trend_type="add",
            seasonal_type="add"
        )

        assert isinstance(result, pl.DataFrame)
        assert "actual" in result.columns
        assert "fitted" in result.columns
        assert "level" in result.columns
        assert "trend" in result.columns
        assert "seasonal" in result.columns
        assert "alpha" in result.columns
        assert "beta" in result.columns
        assert "gamma" in result.columns

    def test_holt_winters_with_forecast(self, mock_ctx, sample_seasonal_data):
        """Test Holt-Winters with forecasting"""
        result = HOLT_WINTERS(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4,
            forecast_periods=4
        )

        # Should have forecast values
        forecasts = result.filter(pl.col("forecast").is_not_null())["forecast"].to_list()
        assert len(forecasts) == 4

    def test_holt_winters_parameter_validation(self, mock_ctx, sample_seasonal_data):
        """Test parameter validation"""
        # Invalid seasonal periods
        with pytest.raises(ValidationError):
            HOLT_WINTERS(mock_ctx, sample_seasonal_data, seasonal_periods=0)

        # Invalid trend type
        with pytest.raises(ValidationError):
            HOLT_WINTERS(mock_ctx, sample_seasonal_data, seasonal_periods=4, trend_type="invalid")

        # Invalid seasonal type
        with pytest.raises(ValidationError):
            HOLT_WINTERS(mock_ctx, sample_seasonal_data, seasonal_periods=4, seasonal_type="invalid")

        # Invalid alpha
        with pytest.raises(ValidationError):
            HOLT_WINTERS(mock_ctx, sample_seasonal_data, seasonal_periods=4, alpha=0)

        with pytest.raises(ValidationError):
            HOLT_WINTERS(mock_ctx, sample_seasonal_data, seasonal_periods=4, alpha=1.5)

    def test_holt_winters_insufficient_data(self, mock_ctx):
        """Test with insufficient data"""
        short_data = [1, 2, 3, 4, 5]
        with pytest.raises(ValidationError):
            HOLT_WINTERS(mock_ctx, short_data, seasonal_periods=4)

    def test_holt_winters_with_output(self, mock_ctx, sample_seasonal_data):
        """Test Holt-Winters with file output"""
        result = HOLT_WINTERS(
            mock_ctx,
            sample_seasonal_data,
            seasonal_periods=4,
            output_filename="holt_winters_test.parquet"
        )

        assert isinstance(result, Path)


class TestDataFrameInputs:
    """Test cases for DataFrame inputs"""

    def test_dataframe_input_linear_forecast(self, mock_ctx, sample_trend_data):
        """Test LINEAR_FORECAST with DataFrame input"""
        df = pl.DataFrame({"values": sample_trend_data})
        result = LINEAR_FORECAST(mock_ctx, df, forecast_periods=2)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_trend_data) + 2

    def test_dataframe_input_moving_average(self, mock_ctx, sample_trend_data):
        """Test MOVING_AVERAGE with DataFrame input"""
        df = pl.DataFrame({"values": sample_trend_data})
        result = MOVING_AVERAGE(mock_ctx, df, window_size=3)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_trend_data)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_small_datasets(self, mock_ctx):
        """Test with very small datasets"""
        small_data = [100, 105]

        # Should work for functions that can handle 2 points
        result = TREND_COEFFICIENT(mock_ctx, small_data)
        assert isinstance(result, Decimal)

        # Should fail for functions requiring more data
        with pytest.raises(ValidationError):
            SEASONAL_DECOMPOSE(mock_ctx, small_data, seasonal_periods=4)

    def test_large_datasets(self, mock_ctx):
        """Test with larger datasets"""
        # Generate larger dataset
        large_data = list(range(1000))

        result = TREND_COEFFICIENT(mock_ctx, large_data)
        assert isinstance(result, Decimal)
        assert abs(float(result) - 1.0) < 0.01  # Should be approximately 1

    def test_extreme_values(self, mock_ctx):
        """Test with extreme values"""
        extreme_data = [1e-10, 1e10, -1e10, 1e-5, 1e5]

        # Should handle extreme values gracefully
        result = TREND_COEFFICIENT(mock_ctx, extreme_data)
        assert isinstance(result, Decimal)

    def test_null_handling(self, mock_ctx):
        """Test null value handling"""
        # Functions should reject data with nulls
        data_with_nulls = [1, 2, None, 4, 5]

        with pytest.raises(DataQualityError):
            LINEAR_FORECAST(mock_ctx, data_with_nulls, forecast_periods=1)


class TestPerformance:
    """Performance and efficiency tests"""

    def test_large_dataset_performance(self, mock_ctx):
        """Test performance with large datasets"""
        import time

        # Generate large dataset
        large_data = list(range(10000))

        start_time = time.time()
        result = MOVING_AVERAGE(mock_ctx, large_data, window_size=100)
        end_time = time.time()

        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
        assert isinstance(result, pl.DataFrame)

    def test_memory_efficiency(self, mock_ctx):
        """Test memory efficiency with large datasets"""
        # This test ensures functions don't create excessive intermediate copies
        large_data = list(range(50000))

        # Should not raise memory errors
        result = TREND_COEFFICIENT(mock_ctx, large_data)
        assert isinstance(result, Decimal)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
