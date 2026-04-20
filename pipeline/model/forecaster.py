"""Spread forecasting using TimesFM.

The forecaster wraps the TimesFMLoader singleton and produces a structured
ForecastResult that the signal generator can consume directly.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from pipeline.model.loader import (
    QUANT_MEAN,
    QUANT_Q10,
    QUANT_Q50,
    QUANT_Q90,
    TimesFMLoader,
)


class ForecastResult(BaseModel):
    horizon: int
    point_forecast: list[float]
    q10: list[float]
    q50: list[float]
    q90: list[float]
    mean_forecast: list[float]
    forecast_endpoint: float
    current_spread: float
    forecast_direction: str  # "MEAN_REVERTING" | "DIVERGING" | "FLAT"
    confidence_interval_width: float


class ForecastError(Exception):
    """Raised when TimesFM inference fails."""


class SpreadForecaster:
    """Wraps the TimesFM loader to produce spread forecasts."""

    # Direction thresholds: endpoint is this fraction of |current| to classify
    _MEAN_REV_THRESHOLD = 0.7   # |endpoint| < 0.7 * |current| → MEAN_REVERTING
    _DIVERG_THRESHOLD = 1.3     # |endpoint| > 1.3 * |current| → DIVERGING

    def __init__(self, loader: TimesFMLoader | None = None) -> None:
        self._loader = loader or TimesFMLoader.get_instance()

    def forecast(self, spread_values: np.ndarray, horizon: int = 30) -> ForecastResult:
        """Forecast the spread for `horizon` steps.

        Args:
            spread_values: 1-D numpy array of historical spread values (must be
                ≥ 32 observations).
            horizon: Number of periods to forecast.

        Returns:
            ForecastResult with point forecast, quantile bands, direction, and
            confidence interval width.

        Raises:
            ForecastError: On validation failure or model inference error.
        """
        if not self._loader.is_loaded():
            raise ForecastError(
                "TimesFM not loaded. Call TimesFMLoader.get_instance().load() first."
            )

        arr = np.asarray(spread_values, dtype=np.float64).ravel()

        if len(arr) < 32:
            raise ForecastError(
                f"Spread series too short: {len(arr)} < 32 (TimesFM minimum)"
            )

        # Truncate to max_context if longer (model handles this too, but be explicit)
        max_ctx = self._loader.max_context
        if len(arr) > max_ctx:
            arr = arr[-max_ctx:]

        # Replace NaNs with linear interpolation before calling the model
        arr = self._interpolate_nans(arr)

        try:
            point_fc, quant_fc = self._loader.model.forecast(
                inputs=[arr],
                freq=[0],        # 0 = default/high frequency (daily financial data)
                normalize=True,  # per-series z-score normalisation + de-norm
            )
        except Exception as exc:
            raise ForecastError(f"TimesFM inference failed: {exc}") from exc

        # Shapes: point_fc (1, MaxH),  quant_fc (1, MaxH, 10) [mean + 9 quantiles]
        # Slice to requested horizon
        point = point_fc[0, :horizon].tolist()
        quants = quant_fc[0, :horizon, :]

        current_spread = float(arr[-1])
        endpoint = float(point[-1])
        direction = self._direction(current_spread, endpoint)

        ci_width = float(
            np.mean(quants[:, QUANT_Q90] - quants[:, QUANT_Q10])
        )

        return ForecastResult(
            horizon=horizon,
            point_forecast=point,
            q10=quants[:, QUANT_Q10].tolist(),
            q50=quants[:, QUANT_Q50].tolist(),
            q90=quants[:, QUANT_Q90].tolist(),
            mean_forecast=quants[:, QUANT_MEAN].tolist(),
            forecast_endpoint=endpoint,
            current_spread=current_spread,
            forecast_direction=direction,
            confidence_interval_width=ci_width,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _direction(self, current: float, endpoint: float) -> str:
        """Classify whether the forecast is mean-reverting, diverging, or flat."""
        abs_current = abs(current)
        if abs_current < 1e-9:
            return "FLAT"
        ratio = abs(endpoint) / abs_current
        if ratio <= self._MEAN_REV_THRESHOLD:
            return "MEAN_REVERTING"
        if ratio >= self._DIVERG_THRESHOLD:
            return "DIVERGING"
        return "FLAT"

    @staticmethod
    def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
        """Replace NaNs with linear interpolation; strip leading NaNs."""
        arr = arr.copy()
        # Strip leading NaNs
        first_valid = np.argmax(~np.isnan(arr))
        arr = arr[first_valid:]
        if len(arr) == 0:
            raise ForecastError("Spread array is all-NaN after stripping")
        # Interpolate interior NaNs
        nans = np.isnan(arr)
        if nans.any():
            indices = np.arange(len(arr))
            arr[nans] = np.interp(indices[nans], indices[~nans], arr[~nans])
        return arr
