"""Cointegration analysis: ADF, Engle-Granger, OLS hedge ratio, half-life."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint


class CointegrationResult(BaseModel):
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    adf_pvalue: float
    eg_pvalue: float
    half_life_days: float
    is_cointegrated: bool


class CointegrationAnalyzer:
    """Tests two price series for cointegration and computes spread parameters."""

    def analyze(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        ticker_a: str = "",
        ticker_b: str = "",
    ) -> CointegrationResult:
        """Analyze cointegration between two price series.
        
        Tests both directions (A on B and B on A) and returns the most significant.
        """
        res_ab = self._analyze_one_way(prices_a, prices_b, ticker_a, ticker_b)
        res_ba = self._analyze_one_way(prices_b, prices_a, ticker_b, ticker_a)
        
        # Return the one with lower p-value
        if res_ab.eg_pvalue <= res_ba.eg_pvalue:
            return res_ab
        return res_ba

    def _analyze_one_way(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        ticker_a: str = "",
        ticker_b: str = "",
    ) -> CointegrationResult:
        """Analyze cointegration for log_a = alpha + beta * log_b."""
        log_a = np.log(prices_a.values.astype(float))
        log_b = np.log(prices_b.values.astype(float))

        # OLS: log_a = α + β * log_b + ε  →  hedge ratio β
        X = add_constant(log_b)
        ols_result = OLS(log_a, X).fit()
        hedge_ratio = float(ols_result.params[1])

        # Spread in log-price space
        spread = pd.Series(log_a - hedge_ratio * log_b, index=prices_a.index)

        # ADF test on spread (AIC lag selection)
        adf_stat, adf_pvalue, *_ = adfuller(spread.dropna(), autolag="AIC")

        # Engle-Granger test
        eg_stat, eg_pvalue, _ = coint(log_a, log_b, trend="c")

        # Half-life of mean reversion via OLS: Δspread_t = γ * spread_{t-1} + c + ε
        half_life = self._compute_half_life(spread)

        # Thresholds: p < 0.05 and half-life >= 1.0 day
        is_cointegrated = bool(eg_pvalue < 0.05 and 1.0 <= half_life <= 252.0)

        return CointegrationResult(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            hedge_ratio=hedge_ratio,
            adf_pvalue=float(adf_pvalue),
            eg_pvalue=float(eg_pvalue),
            half_life_days=float(half_life),
            is_cointegrated=is_cointegrated,
        )

    @staticmethod
    def _compute_half_life(spread: pd.Series) -> float:
        """Estimate mean-reversion half-life from AR(1) regression on spread differences."""
        delta = spread.diff().dropna()
        lag = spread.shift(1).dropna()

        # Align to common index
        common = delta.index.intersection(lag.index)
        delta = delta.loc[common]
        lag = lag.loc[common]

        X = add_constant(lag.values)
        model = OLS(delta.values, X).fit()
        gamma = float(model.params[1])

        if gamma >= 0:
            # No mean reversion detected
            return float("inf")

        return -np.log(2) / gamma
