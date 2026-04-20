"""Performance metrics: Sharpe, Sortino, Calmar, drawdown, win rate, profit factor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

TRADING_DAYS = 252


class BacktestMetrics(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_holding_days: float


def compute_metrics(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    trade_holding_days: list[float],
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Compute all performance metrics from equity curve and trade list.

    Args:
        equity_curve: Daily equity values (indexed by date).
        trade_pnls: Net P&L per completed trade.
        trade_holding_days: Holding period in calendar days per trade.
        risk_free_rate: Annualized risk-free rate (default 0).
    """
    returns = equity_curve.pct_change().dropna()
    n = len(returns)

    # Annualized return
    if n < 2:
        ann_return = 0.0
    else:
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        years = n / TRADING_DAYS
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Sharpe
    excess = returns - risk_free_rate / TRADING_DAYS
    std = excess.std()
    sharpe = float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    # Sortino — downside deviation only
    downside = excess[excess < 0]
    down_std = downside.std()
    sortino = float(excess.mean() / down_std * np.sqrt(TRADING_DAYS)) if down_std > 0 else 0.0

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Calmar
    calmar = float(-ann_return / max_dd) if max_dd < 0 else 0.0

    # Trade-level stats
    num_trades = len(trade_pnls)
    if num_trades == 0:
        return BacktestMetrics(
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            annualized_return=round(ann_return, 6),
            max_drawdown=round(max_dd, 6),
            win_rate=0.0,
            profit_factor=0.0,
            num_trades=0,
            avg_holding_days=0.0,
        )

    winners = [p for p in trade_pnls if p > 0]
    losers = [p for p in trade_pnls if p <= 0]
    win_rate = len(winners) / num_trades
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
    avg_holding = float(np.mean(trade_holding_days)) if trade_holding_days else 0.0

    return BacktestMetrics(
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        annualized_return=round(ann_return, 6),
        max_drawdown=round(max_dd, 6),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        num_trades=num_trades,
        avg_holding_days=round(avg_holding, 2),
    )
