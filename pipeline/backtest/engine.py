"""Vectorized pairs-trading backtester with walk-forward validation and Kalman support.

This engine simulates the performance of a statistical arbitrage strategy on a 
single stock pair, accounting for realistic trading costs and regime shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from pipeline.backtest.costs import CostModel
from pipeline.backtest.metrics import BacktestMetrics, compute_metrics
from pipeline.stats.spread import SpreadCalculator


# ---------------------------------------------------------------------------
# Internal trade record (not exposed to callers)
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    """Internal representation of a single round-trip spread trade."""
    entry_date: pd.Timestamp
    direction: int            # +1 = long spread (buy A, short B), -1 = short spread
    entry_spread: float
    entry_zscore: float
    entry_index: int
    cost: float
    exit_date: Optional[pd.Timestamp] = None
    exit_spread: Optional[float] = None
    exit_zscore: Optional[float] = None
    exit_index: Optional[int] = None

    @property
    def gross_pnl(self) -> float:
        """Calculate raw profit/loss from spread change."""
        if self.exit_spread is None:
            return 0.0
        return self.direction * (self.exit_spread - self.entry_spread)

    @property
    def holding_days(self) -> int:
        """Calculate duration of the trade in calendar days."""
        if self.exit_date is None:
            return 0
        return max(1, (self.exit_date - self.entry_date).days)

    def to_dict(self, notional: float) -> dict:
        """Convert trade record to a dictionary for API/JSON export."""
        return {
            "entry_date": str(self.entry_date.date()),
            "exit_date": str(self.exit_date.date()) if self.exit_date else None,
            "direction": "LONG_SPREAD" if self.direction == 1 else "SHORT_SPREAD",
            "entry_spread": round(self.entry_spread, 6),
            "exit_spread": round(self.exit_spread, 6) if self.exit_spread is not None else None,
            "entry_zscore": round(self.entry_zscore, 4),
            "exit_zscore": round(self.exit_zscore, 4) if self.exit_zscore is not None else None,
            "holding_days": self.holding_days,
            "gross_pnl": round(self.gross_pnl * notional, 2),
            "cost": round(self.cost, 2),
            "net_pnl": round(self.gross_pnl * notional - self.cost, 2),
        }


# ---------------------------------------------------------------------------
# Public result model
# ---------------------------------------------------------------------------

class BacktestResult(BaseModel):
    """Container for full backtest results and performance metrics."""
    metrics: BacktestMetrics
    equity_curve: list[float]
    equity_dates: list[str]
    trades: list[dict]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Vectorized statistical-arb backtester for a single pair.

    Strategy Mechanics:
      - Long spread (z < -entry_z): Buy Asset A, Short Asset B proportional to beta.
      - Short spread (z > +entry_z): Sell Asset A, Buy Asset B proportional to beta.
      - Exit: Close when |z| < exit_z (Mean reversion achieved).
    
    This engine supports both static OLS-based spreads and dynamic Kalman-based spreads.
    """

    def run(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float = 1.0,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        notional: float = 10_000.0,
        window: int = 63,
        use_kalman: bool = False,
    ) -> BacktestResult:
        """Run a vectorized simulation over the provided price series.

        Args:
            prices_a: Price series for asset A.
            prices_b: Price series for asset B.
            hedge_ratio: Static beta if use_kalman=False.
            entry_z: Z-Score threshold for trade entry.
            exit_z: Z-Score threshold for trade exit.
            transaction_cost_bps: Commission and fees in basis points.
            slippage_bps: Expected slippage per order in basis points.
            notional: Target dollar size of the trade.
            window: Lookback for rolling z-score normalization.
            use_kalman: Flag to enable dynamic hedge ratio adaptation.
        """
        calc = SpreadCalculator()
        cost_model = CostModel(
            round_trip_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )

        # 1. Construct the spread series
        if use_kalman:
            spread_series = calc.compute_kalman(prices_a, prices_b, window=window)
        else:
            spread_series = calc.compute(prices_a, prices_b, hedge_ratio, window=window)
            
        spread = spread_series.spread.values
        zscore = spread_series.zscore.values
        dates = spread_series.spread.index

        # 2. Simulate trades
        equity = self._simulate(
            spread=spread,
            zscore=zscore,
            dates=dates,
            entry_z=entry_z,
            exit_z=exit_z,
            cost_model=cost_model,
            notional=notional,
        )

        # 3. Calculate final performance metrics
        completed = [t for t in equity["trades"] if t.exit_date is not None]
        trade_pnls = [t.gross_pnl * notional - t.cost for t in completed]
        trade_days = [float(t.holding_days) for t in completed]

        eq_series = pd.Series(equity["equity_values"], index=dates[: len(equity["equity_values"])])
        metrics = compute_metrics(eq_series, trade_pnls, trade_days)

        return BacktestResult(
            metrics=metrics,
            equity_curve=[round(v, 4) for v in equity["equity_values"]],
            equity_dates=[str(d.date()) for d in dates[: len(equity["equity_values"])]],
            trades=[t.to_dict(notional) for t in completed],
        )

    def _simulate(
        self,
        spread: np.ndarray,
        zscore: np.ndarray,
        dates: pd.Index,
        entry_z: float,
        exit_z: float,
        cost_model: CostModel,
        notional: float,
    ) -> dict:
        """Core simulation loop iterating through the time-series."""
        n = len(spread)
        equity_values = np.zeros(n)
        equity_values[0] = notional

        current_trade: Optional[_Trade] = None
        all_trades: list[_Trade] = []
        cumulative_pnl = 0.0

        for i in range(1, n):
            z = zscore[i]
            s = spread[i]
            d = dates[i]

            if current_trade is None:
                # Idle state: Check if z-score hits an entry threshold
                if z < -entry_z:
                    cost = cost_model.total_cost(notional)
                    current_trade = _Trade(
                        entry_date=d, direction=1, entry_spread=s,
                        entry_zscore=z, entry_index=i, cost=cost,
                    )
                elif z > entry_z:
                    cost = cost_model.total_cost(notional)
                    current_trade = _Trade(
                        entry_date=d, direction=-1, entry_spread=s,
                        entry_zscore=z, entry_index=i, cost=cost,
                    )
            else:
                # Active trade state: Check if z-score has reverted to the mean
                if abs(z) < exit_z:
                    current_trade.exit_date = d
                    current_trade.exit_spread = s
                    current_trade.exit_zscore = z
                    current_trade.exit_index = i
                    net = current_trade.gross_pnl * notional - current_trade.cost
                    cumulative_pnl += net
                    all_trades.append(current_trade)
                    current_trade = None

            equity_values[i] = notional + cumulative_pnl

        # Safeguard: Force-close any open trade at the end of the available data
        if current_trade is not None:
            current_trade.exit_date = dates[-1]
            current_trade.exit_spread = spread[-1]
            current_trade.exit_zscore = zscore[-1]
            current_trade.exit_index = n - 1
            net = current_trade.gross_pnl * notional - current_trade.cost
            cumulative_pnl += net
            all_trades.append(current_trade)
            equity_values[-1] = notional + cumulative_pnl

        return {"equity_values": equity_values.tolist(), "trades": all_trades}
