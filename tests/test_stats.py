"""Tests for statistical analysis engine: cointegration, spread, signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.stats.cointegration import CointegrationAnalyzer, CointegrationResult
from pipeline.stats.signals import SignalGenerator, TradingSignal
from pipeline.stats.spread import SpreadCalculator, SpreadSeries


# ---------------------------------------------------------------------------
# Helpers: synthetic price series
# ---------------------------------------------------------------------------

def make_cointegrated_pair(n: int = 1000, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Generate two cointegrated price series via an Ornstein-Uhlenbeck spread.

    Theoretical half-life = ln(2) / theta ≈ ln(2) / 0.08 ≈ 8.7 days.
    """
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.01, n))
    beta = 1.3

    theta = 0.08  # mean reversion speed → half-life ~8.7 days
    sigma = 0.015
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = ou[t - 1] * (1 - theta) + rng.normal(0, sigma)

    log_a = beta * log_b + ou
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return (
        pd.Series(np.exp(log_a), index=dates, name="A"),
        pd.Series(np.exp(log_b), index=dates, name="B"),
    )


def make_random_walks(n: int = 3000, seed: int = 99) -> tuple[pd.Series, pd.Series]:
    """Generate two independent random walks (non-cointegrated).

    Uses n=3000 because the Engle-Granger test has finite-sample size
    distortion (~8-10% false positive rate at n=1000). Larger n pushes
    the test toward its nominal 5% level.
    """
    rng = np.random.default_rng(seed)
    log_a = np.cumsum(rng.normal(0, 0.01, n))
    log_b = np.cumsum(rng.normal(0, 0.01, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return (
        pd.Series(np.exp(log_a), index=dates, name="A"),
        pd.Series(np.exp(log_b), index=dates, name="B"),
    )


# ---------------------------------------------------------------------------
# CointegrationAnalyzer tests
# ---------------------------------------------------------------------------

class TestCointegrationAnalyzer:
    def setup_method(self) -> None:
        self.analyzer = CointegrationAnalyzer()

    def test_cointegrated_pair_detected(self) -> None:
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b, "A", "B")

        assert isinstance(result, CointegrationResult)
        assert result.is_cointegrated is True

    def test_cointegrated_pvalue_below_threshold(self) -> None:
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b)

        assert result.eg_pvalue < 0.05

    def test_cointegrated_half_life_in_range(self) -> None:
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b)

        assert 5.0 <= result.half_life_days <= 252.0

    def test_hedge_ratio_is_positive(self) -> None:
        """Hedge ratio should be positive for two positively correlated assets."""
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b)

        assert result.hedge_ratio > 0

    def test_is_cointegrated_false_when_eg_pvalue_too_high(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_cointegrated=False when EG p-value >= 0.05.

        The EG test has finite-sample size distortion (~8-15% false positive rate
        at n=1000-3000), so we can't reliably test statistical outcomes on synthetic
        random walks. Instead we mock the EG function to return a controlled p-value
        and verify our logic correctly propagates it.
        """
        monkeypatch.setattr(
            "pipeline.stats.cointegration.coint",
            lambda *a, **kw: (None, 0.12, None),
        )
        monkeypatch.setattr(
            "pipeline.stats.cointegration.adfuller",
            lambda *a, **kw: (-2.1, 0.12, 1, 100, {}, None),
        )
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b, "A", "B")

        assert result.eg_pvalue == pytest.approx(0.12)
        assert result.is_cointegrated is False

    def test_is_cointegrated_false_when_half_life_out_of_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_cointegrated=False when half_life < 5 or > 252."""
        monkeypatch.setattr(
            "pipeline.stats.cointegration.coint",
            lambda *a, **kw: (None, 0.01, None),  # passes EG test
        )
        monkeypatch.setattr(
            "pipeline.stats.cointegration.adfuller",
            lambda *a, **kw: (-3.5, 0.01, 1, 100, {}, None),
        )
        monkeypatch.setattr(
            CointegrationAnalyzer, "_compute_half_life", staticmethod(lambda s: 300.0)
        )
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b)

        assert result.half_life_days == pytest.approx(300.0)
        assert result.is_cointegrated is False

    def test_ticker_labels_stored(self) -> None:
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b, "KO", "PEP")

        assert result.ticker_a == "KO"
        assert result.ticker_b == "PEP"

    def test_result_fields_are_floats(self) -> None:
        prices_a, prices_b = make_cointegrated_pair()
        result = self.analyzer.analyze(prices_a, prices_b)

        assert isinstance(result.hedge_ratio, float)
        assert isinstance(result.adf_pvalue, float)
        assert isinstance(result.eg_pvalue, float)
        assert isinstance(result.half_life_days, float)


# ---------------------------------------------------------------------------
# SpreadCalculator tests
# ---------------------------------------------------------------------------

class TestSpreadCalculator:
    def setup_method(self) -> None:
        self.calc = SpreadCalculator()
        self.prices_a, self.prices_b = make_cointegrated_pair()
        self.analyzer = CointegrationAnalyzer()

    def _get_hedge_ratio(self) -> float:
        result = self.analyzer.analyze(self.prices_a, self.prices_b)
        return result.hedge_ratio

    def test_returns_spread_series_model(self) -> None:
        hedge = self._get_hedge_ratio()
        series = self.calc.compute(self.prices_a, self.prices_b, hedge)

        assert isinstance(series, SpreadSeries)

    def test_zscore_mean_near_zero(self) -> None:
        """Over a long window, z-score should be approximately zero-mean."""
        hedge = self._get_hedge_ratio()
        series = self.calc.compute(self.prices_a, self.prices_b, hedge)

        assert abs(series.zscore.mean()) < 0.3

    def test_zscore_std_near_one(self) -> None:
        """Z-score by construction has rolling std ≈ 1."""
        hedge = self._get_hedge_ratio()
        series = self.calc.compute(self.prices_a, self.prices_b, hedge)

        # Global std of a rolling z-score is approximately 1
        assert 0.7 < series.zscore.std() < 1.3

    def test_no_nans_in_output(self) -> None:
        hedge = self._get_hedge_ratio()
        series = self.calc.compute(self.prices_a, self.prices_b, hedge)

        assert not series.spread.isna().any()
        assert not series.zscore.isna().any()
        assert not series.rolling_mean.isna().any()
        assert not series.rolling_std.isna().any()

    def test_output_length_shorter_than_input(self) -> None:
        """Window burn-in should reduce output length."""
        hedge = self._get_hedge_ratio()
        series = self.calc.compute(self.prices_a, self.prices_b, hedge, window=63)

        # rolling(63) needs 63 values → first valid at index 62 → drops 62 rows
        assert len(series.spread) == len(self.prices_a) - (63 - 1)

    def test_get_current_signal_inputs_length(self) -> None:
        hedge = self._get_hedge_ratio()
        series = self.calc.compute(self.prices_a, self.prices_b, hedge)
        inputs = SpreadCalculator.get_current_signal_inputs(
            series.spread, series.zscore, lookback=252
        )

        assert inputs["spread"].shape == (252,)
        assert inputs["zscore"].shape == (252,)

    def test_get_current_signal_inputs_dtype(self) -> None:
        hedge = self._get_hedge_ratio()
        series = self.calc.compute(self.prices_a, self.prices_b, hedge)
        inputs = SpreadCalculator.get_current_signal_inputs(series.spread, series.zscore)

        assert inputs["spread"].dtype == np.float64
        assert inputs["zscore"].dtype == np.float64


# ---------------------------------------------------------------------------
# SignalGenerator tests
# ---------------------------------------------------------------------------

class TestSignalGenerator:
    def setup_method(self) -> None:
        self.gen = SignalGenerator()

    def test_buy_spread_mean_reverting(self) -> None:
        """z=-2.5 with MEAN_REVERTING forecast → BUY_SPREAD confidence=1.0."""
        signal = self.gen.generate(
            zscore_current=-2.5,
            forecast_point=-0.8,   # abs(-0.8) < abs(-2.5) → MEAN_REVERTING
            forecast_q10=-1.2,
            forecast_q90=-0.3,
        )

        assert signal.action == "BUY_SPREAD"
        assert signal.confidence == pytest.approx(1.0)
        assert signal.forecast_direction == "MEAN_REVERTING"

    def test_sell_spread_mean_reverting(self) -> None:
        """z=+2.7 with MEAN_REVERTING forecast → SELL_SPREAD confidence=1.0."""
        signal = self.gen.generate(
            zscore_current=2.7,
            forecast_point=0.9,   # abs(0.9) < abs(2.7) → MEAN_REVERTING
            forecast_q10=0.4,
            forecast_q90=1.5,
        )

        assert signal.action == "SELL_SPREAD"
        assert signal.confidence == pytest.approx(1.0)

    def test_buy_spread_flat_forecast(self) -> None:
        """z=-2.5 with FLAT forecast → BUY_SPREAD confidence=0.5."""
        signal = self.gen.generate(
            zscore_current=-2.5,
            forecast_point=-2.5,  # same magnitude → FLAT
            forecast_q10=-3.0,
            forecast_q90=-2.0,
        )

        assert signal.action == "BUY_SPREAD"
        assert signal.confidence == pytest.approx(0.5)

    def test_diverging_forecast_suppresses_signal(self) -> None:
        """z=-2.5 with DIVERGING forecast → HOLD (confidence=0.1 < 0.4)."""
        signal = self.gen.generate(
            zscore_current=-2.5,
            forecast_point=-4.0,  # abs(-4.0) > abs(-2.5) → DIVERGING
            forecast_q10=-5.0,
            forecast_q90=-3.0,
        )

        assert signal.action == "HOLD"
        assert signal.confidence == pytest.approx(0.1)

    def test_exit_signal_when_zscore_near_zero(self) -> None:
        """|z| < exit_z → EXIT regardless of forecast."""
        signal = self.gen.generate(
            zscore_current=0.3,
            forecast_point=0.1,
            forecast_q10=-0.2,
            forecast_q90=0.5,
        )

        assert signal.action == "EXIT"
        assert signal.confidence == pytest.approx(1.0)

    def test_hold_when_zscore_between_bands(self) -> None:
        """exit_z < |z| < entry_z → HOLD."""
        signal = self.gen.generate(
            zscore_current=1.2,
            forecast_point=0.8,
            forecast_q10=0.3,
            forecast_q90=1.5,
        )

        assert signal.action == "HOLD"

    def test_signal_fields_populated(self) -> None:
        signal = self.gen.generate(
            zscore_current=-2.5,
            forecast_point=-0.8,
            forecast_q10=-1.2,
            forecast_q90=-0.3,
        )

        assert isinstance(signal, TradingSignal)
        assert signal.z_score == pytest.approx(-2.5)
        assert isinstance(signal.rationale, str)
        assert len(signal.rationale) > 0

    def test_forecast_array_uses_last_element(self) -> None:
        """When forecast_point is an array, the last element determines direction."""
        # Array ending at -0.8: abs(-0.8) < abs(-2.5) → MEAN_REVERTING
        signal = self.gen.generate(
            zscore_current=-2.5,
            forecast_point=np.array([-2.4, -2.0, -1.5, -0.8]),
            forecast_q10=np.array([-2.5, -2.1, -1.6, -1.2]),
            forecast_q90=np.array([-2.3, -1.9, -1.4, -0.3]),
        )

        assert signal.forecast_direction == "MEAN_REVERTING"
        assert signal.action == "BUY_SPREAD"

    def test_custom_entry_z(self) -> None:
        """Signal with custom entry_z=1.5."""
        signal = self.gen.generate(
            zscore_current=-1.6,  # would be HOLD with default 2.0
            forecast_point=-0.5,
            forecast_q10=-0.8,
            forecast_q90=-0.2,
            entry_z=1.5,
        )

        assert signal.action == "BUY_SPREAD"
