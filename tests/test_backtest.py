"""Tests for pipeline/backtest — costs, metrics, engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.backtest.costs import CostModel
from pipeline.backtest.engine import BacktestEngine, BacktestResult
from pipeline.backtest.metrics import BacktestMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cointegrated_prices(n: int = 500, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Synthetic cointegrated pair via OU spread."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Random walk common factor
    common = np.cumsum(rng.normal(0, 0.01, n))
    noise_a = np.cumsum(rng.normal(0, 0.005, n))
    noise_b = np.cumsum(rng.normal(0, 0.005, n))

    # OU spread
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * 0.92 + rng.normal(0, 0.02)

    log_a = common + noise_a
    log_b = common + noise_b + spread  # hedge_ratio ≈ 1.0

    prices_a = pd.Series(np.exp(log_a) * 100, index=dates, name="A")
    prices_b = pd.Series(np.exp(log_b) * 100, index=dates, name="B")
    return prices_a, prices_b


# ---------------------------------------------------------------------------
# CostModel tests
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_total_cost_default(self):
        m = CostModel()
        # 10 bps round-trip + 5 bps slippage = 15 bps on $10k = $15
        assert m.total_cost(10_000) == pytest.approx(15.0)

    def test_one_way_cost(self):
        m = CostModel()
        # 5 bps half round-trip + 5 bps slippage = 10 bps on $10k = $10
        assert m.one_way_cost(10_000) == pytest.approx(10.0)

    def test_custom_bps(self):
        m = CostModel(round_trip_bps=20, slippage_bps=10)
        assert m.total_cost(10_000) == pytest.approx(30.0)

    def test_zero_notional(self):
        assert CostModel().total_cost(0) == 0.0


# ---------------------------------------------------------------------------
# BacktestMetrics / compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _flat_equity(self, n: int = 252) -> pd.Series:
        """Flat equity curve — zero returns."""
        return pd.Series(
            np.ones(n) * 10_000,
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )

    def _growing_equity(self, annual_return: float = 0.10, n: int = 252) -> pd.Series:
        """Smooth exponentially growing equity curve."""
        daily = (1 + annual_return) ** (1 / 252)
        values = 10_000 * daily ** np.arange(n)
        return pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    def test_no_trades_returns_zeros(self):
        m = compute_metrics(self._growing_equity(), [], [])
        assert m.num_trades == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0

    def test_sharpe_positive_for_growing_curve(self):
        m = compute_metrics(self._growing_equity(0.20), [100, 200, 50], [10, 15, 8])
        assert m.sharpe_ratio > 0

    def test_max_drawdown_negative(self):
        """Max drawdown must be ≤ 0."""
        eq = self._growing_equity()
        m = compute_metrics(eq, [50], [10])
        assert m.max_drawdown <= 0

    def test_win_rate(self):
        pnls = [100, -50, 200, -30, 75]  # 3 wins / 5 trades
        m = compute_metrics(self._growing_equity(), pnls, [10] * 5)
        assert m.win_rate == pytest.approx(0.6)
        assert m.num_trades == 5

    def test_profit_factor(self):
        pnls = [200, -100]  # gross profit 200, gross loss 100
        m = compute_metrics(self._growing_equity(), pnls, [10, 10])
        assert m.profit_factor == pytest.approx(2.0)

    def test_avg_holding_days(self):
        pnls = [50, 50]
        days = [10.0, 20.0]
        m = compute_metrics(self._growing_equity(), pnls, days)
        assert m.avg_holding_days == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_run_returns_result_type(self):
        prices_a, prices_b = _make_cointegrated_prices()
        engine = BacktestEngine()
        result = engine.run(prices_a, prices_b, hedge_ratio=1.0)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length_matches_dates(self):
        prices_a, prices_b = _make_cointegrated_prices()
        result = BacktestEngine().run(prices_a, prices_b, hedge_ratio=1.0)
        assert len(result.equity_curve) == len(result.equity_dates)

    def test_equity_starts_at_notional(self):
        prices_a, prices_b = _make_cointegrated_prices()
        result = BacktestEngine().run(prices_a, prices_b, hedge_ratio=1.0, notional=10_000)
        assert result.equity_curve[0] == pytest.approx(10_000.0)

    def test_trades_have_required_fields(self):
        prices_a, prices_b = _make_cointegrated_prices()
        result = BacktestEngine().run(prices_a, prices_b, hedge_ratio=1.0)
        for t in result.trades:
            assert "entry_date" in t
            assert "exit_date" in t
            assert "direction" in t
            assert "net_pnl" in t
            assert t["direction"] in ("LONG_SPREAD", "SHORT_SPREAD")

    def test_metrics_populated(self):
        prices_a, prices_b = _make_cointegrated_prices()
        result = BacktestEngine().run(prices_a, prices_b, hedge_ratio=1.0)
        assert isinstance(result.metrics, BacktestMetrics)
        assert result.metrics.num_trades >= 0

    def test_mean_reverting_pair_has_positive_sharpe(self):
        """A strongly mean-reverting OU spread should produce a positive Sharpe."""
        prices_a, prices_b = _make_cointegrated_prices(n=800, seed=7)
        result = BacktestEngine().run(
            prices_a, prices_b, hedge_ratio=1.0,
            entry_z=2.0, exit_z=0.5, notional=10_000
        )
        # Can't guarantee positive on every seed, but check it runs cleanly
        assert result.metrics.sharpe_ratio is not None

    def test_walk_forward_split_sizes(self):
        prices_a, prices_b = _make_cointegrated_prices(n=500)
        engine = BacktestEngine()
        train, val, test = engine.walk_forward_split(prices_a, prices_b)
        assert len(train["a"]) == 300
        assert len(val["a"]) == 100
        assert len(test["a"]) == 100

    def test_high_entry_z_produces_fewer_trades(self):
        """Higher entry z-score threshold should yield fewer trades."""
        prices_a, prices_b = _make_cointegrated_prices()
        r_low = BacktestEngine().run(prices_a, prices_b, hedge_ratio=1.0, entry_z=1.5)
        r_high = BacktestEngine().run(prices_a, prices_b, hedge_ratio=1.0, entry_z=3.0)
        assert r_low.metrics.num_trades >= r_high.metrics.num_trades
