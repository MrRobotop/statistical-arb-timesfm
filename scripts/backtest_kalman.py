"""Run a 2-year backtest with Kalman Filter and report results."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("backtest_kalman")


def main() -> None:
    from pipeline.data.fetcher import StockDataFetcher
    from pipeline.backtest.engine import BacktestEngine

    # Configuration
    ticker_a, ticker_b = "V", "MA"
    notional = 10_000.0
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=2 * 365)).isoformat()

    log.info("Starting 2-year backtest for %s / %s with Kalman Filter...", ticker_a, ticker_b)
    log.info("Period: %s to %s", start_date, end_date)

    # 1. Fetch data
    fetcher = StockDataFetcher()
    try:
        df = fetcher.fetch([ticker_a, ticker_b], start_date=start_date, end_date=end_date)
        prices_a = df[ticker_a]["Close"]
        prices_b = df[ticker_b]["Close"]
    except Exception as exc:
        log.error("Failed to fetch data: %s", exc)
        return

    # 2. Run backtest
    engine = BacktestEngine()
    result = engine.run(
        prices_a=prices_a,
        prices_b=prices_b,
        use_kalman=True,
        notional=notional,
        entry_z=2.0,
        exit_z=0.5,
    )

    # 3. Report metrics
    m = result.metrics
    log.info("\n── Backtest Metrics ───────────────────────────")
    log.info("  Annualized Return : %.2f%%", m.annualized_return * 100)
    log.info("  Sharpe Ratio      : %.2f", m.sharpe_ratio)
    log.info("  Sortino Ratio     : %.2f", m.sortino_ratio)
    log.info("  Max Drawdown      : %.2f%%", m.max_drawdown * 100)
    log.info("  Win Rate          : %.2f%%", m.win_rate * 100)
    log.info("  Profit Factor     : %.2f", m.profit_factor)
    log.info("  Total Trades      : %d", m.num_trades)
    log.info("  Avg Hold Days     : %.1f", m.avg_holding_days)
    log.info("  Total Net PnL     : $%.2f", sum(t["net_pnl"] for t in result.trades))
    log.info("───────────────────────────────────────────────")

    # 4. Save results for reference/frontend
    output_path = Path("data/backtest_kalman_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2))
    log.info("Full results saved to %s", output_path)


if __name__ == "__main__":
    main()
