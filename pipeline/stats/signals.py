"""Trading signal generation combining z-score and TimesFM forecast direction."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from pydantic import BaseModel, field_validator


class TradingSignal(BaseModel):
    action: Literal["BUY_SPREAD", "SELL_SPREAD", "EXIT", "HOLD"]
    confidence: float
    z_score: float
    forecast_direction: Literal["MEAN_REVERTING", "DIVERGING", "FLAT"]
    rationale: str


ForecastInput = Union[float, list, np.ndarray]


class SignalGenerator:
    """Combines z-score statistics with TimesFM forecast to generate trading signals."""

    def generate(
        self,
        zscore_current: float,
        forecast_point: ForecastInput,
        forecast_q10: ForecastInput,
        forecast_q90: ForecastInput,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> TradingSignal:
        """Generate a trading signal.

        Args:
            zscore_current: Current z-score of the spread.
            forecast_point: Point forecast (scalar or array; last element used).
            forecast_q10: 10th-percentile quantile forecast.
            forecast_q90: 90th-percentile quantile forecast.
            entry_z: Z-score threshold for entry signals.
            exit_z: Z-score threshold for exit signals.
        """
        forecast_end = self._last_value(forecast_point)
        direction = self._forecast_direction(zscore_current, forecast_end)

        # Exit signal takes priority over entry
        if abs(zscore_current) < exit_z:
            return TradingSignal(
                action="EXIT",
                confidence=1.0,
                z_score=zscore_current,
                forecast_direction=direction,
                rationale=f"Z-score {zscore_current:.2f} within exit band ±{exit_z}",
            )

        # Determine statistical entry signal
        if zscore_current < -entry_z:
            stat_action: Literal["BUY_SPREAD", "SELL_SPREAD"] = "BUY_SPREAD"
            signal_desc = f"z={zscore_current:.2f} < -{entry_z} (spread undervalued)"
        elif zscore_current > entry_z:
            stat_action = "SELL_SPREAD"
            signal_desc = f"z={zscore_current:.2f} > +{entry_z} (spread overvalued)"
        else:
            return TradingSignal(
                action="HOLD",
                confidence=0.0,
                z_score=zscore_current,
                forecast_direction=direction,
                rationale=(
                    f"Z-score {zscore_current:.2f} between ±{exit_z} and ±{entry_z}; "
                    "no entry signal"
                ),
            )

        # Confidence from forecast agreement
        if direction == "MEAN_REVERTING":
            confidence = 1.0
            forecast_desc = "forecast confirms mean reversion"
        elif direction == "FLAT":
            confidence = 0.5
            forecast_desc = "forecast is flat (neutral)"
        else:  # DIVERGING
            confidence = 0.1
            forecast_desc = "forecast is diverging — signal suppressed"

        if confidence < 0.4:
            return TradingSignal(
                action="HOLD",
                confidence=confidence,
                z_score=zscore_current,
                forecast_direction=direction,
                rationale=f"{signal_desc}; {forecast_desc}",
            )

        return TradingSignal(
            action=stat_action,
            confidence=confidence,
            z_score=zscore_current,
            forecast_direction=direction,
            rationale=f"{signal_desc}; {forecast_desc}",
        )

    @staticmethod
    def _last_value(x: ForecastInput) -> float:
        """Extract the terminal forecast value from a scalar or array."""
        if isinstance(x, (list, np.ndarray)):
            return float(np.asarray(x).flat[-1])
        return float(x)

    @staticmethod
    def _forecast_direction(
        zscore_current: float,
        forecast_end: float,
    ) -> Literal["MEAN_REVERTING", "DIVERGING", "FLAT"]:
        """Classify forecast direction relative to current z-score.

        MEAN_REVERTING: forecast endpoint is meaningfully closer to 0.
        DIVERGING: forecast endpoint is meaningfully further from 0.
        FLAT: forecast endpoint is within 5% of current absolute value.
        """
        current_abs = abs(zscore_current)
        if current_abs < 0.05:
            return "FLAT"

        tol = current_abs * 0.05
        forecast_abs = abs(forecast_end)

        if forecast_abs <= current_abs - tol:
            return "MEAN_REVERTING"
        if forecast_abs >= current_abs + tol:
            return "DIVERGING"
        return "FLAT"
