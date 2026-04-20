"""Spread construction, rolling z-score, and signal input extraction.

This module handles the mathematical transformation of raw asset prices into 
mean-reverting spreads and their statistical normalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from pipeline.stats.kalman import KalmanHedgeRatio


class SpreadSeries(BaseModel):
    """Container for the computed spread and its rolling statistics."""
    spread: pd.Series
    zscore: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    hedge_ratio: pd.Series | float  # Static β (OLS) or dynamic β (Kalman)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SpreadCalculator:
    """Constructs the log-price spread and computes rolling z-scores."""

    def compute(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float,
        window: int = 63,
    ) -> SpreadSeries:
        """Compute spread using a fixed OLS hedge ratio.

        Uses log-price spread: spread = log(prices_a) - hedge_ratio * log(prices_b).
        Normalization is performed using a rolling window to avoid look-ahead bias.

        Args:
            prices_a: Raw price series for asset A.
            prices_b: Raw price series for asset B.
            hedge_ratio: Static OLS hedge ratio β.
            window: Rolling window length for mean and std (default 63 ≈ 3 months).
        """
        log_a = np.log(prices_a.values.astype(float))
        log_b = np.log(prices_b.values.astype(float))
        spread = pd.Series(
            log_a - hedge_ratio * log_b,
            index=prices_a.index,
            name="spread",
        )

        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std

        # Trim NaN rows from the start of the series (warm-up period)
        valid_idx = zscore.dropna().index
        return SpreadSeries(
            spread=spread.loc[valid_idx],
            zscore=zscore.loc[valid_idx],
            rolling_mean=rolling_mean.loc[valid_idx],
            rolling_std=rolling_std.loc[valid_idx],
            hedge_ratio=hedge_ratio,
        )

    def compute_kalman(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        window: int = 63,
    ) -> SpreadSeries:
        """Compute spread using a Kalman Filter for dynamic β estimation.

        The spread is defined as the innovation (residual) of the state-space model:
        innovation_t = log(prices_a) - (alpha_t + beta_t * log(prices_b))
        
        This adapts to changing correlations and volatility regimes in real-time.
        """
        kalman = KalmanHedgeRatio()
        params = kalman.estimate(prices_b, prices_a)
        
        log_a = np.log(prices_a.values.astype(float))
        log_b = np.log(prices_b.values.astype(float))
        
        spread = pd.Series(
            log_a - (params["alpha"] + params["beta"] * log_b),
            index=prices_a.index,
            name="spread",
        )

        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std

        valid_idx = zscore.dropna().index
        return SpreadSeries(
            spread=spread.loc[valid_idx],
            zscore=zscore.loc[valid_idx],
            rolling_mean=rolling_mean.loc[valid_idx],
            rolling_std=rolling_std.loc[valid_idx],
            hedge_ratio=params["beta"].loc[valid_idx],
        )

    @staticmethod
    def get_current_signal_inputs(
        spread_series: pd.Series,
        zscore_series: pd.Series,
        lookback: int = 252,
    ) -> dict[str, np.ndarray]:
        """Extract the most recent context window for model inference.

        Args:
            spread_series: Full history of spread values.
            zscore_series: Full history of z-scores.
            lookback: Number of historical points to include as context.
        """
        return {
            "spread": spread_series.iloc[-lookback:].to_numpy(dtype=float),
            "zscore": zscore_series.iloc[-lookback:].to_numpy(dtype=float),
        }
