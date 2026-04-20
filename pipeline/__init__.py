"""PairsTrader pipeline — convenience entry point for the full signal pipeline."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from pipeline.data.fetcher import StockDataFetcher
from pipeline.model.forecaster import ForecastResult, SpreadForecaster
from pipeline.model.loader import TimesFMLoader
from pipeline.stats.cointegration import CointegrationAnalyzer, CointegrationResult
from pipeline.stats.signals import SignalGenerator, TradingSignal
from pipeline.stats.spread import SpreadCalculator, SpreadSeries

# Type alias for the full pipeline output
PipelineResult = dict[str, Any]


def run_full_pipeline(
    ticker_a: str,
    ticker_b: str,
    horizon: int = 30,
    lookback_years: int = 2,
) -> PipelineResult:
    """Fetch data, run cointegration, forecast spread, generate signal.

    Args:
        ticker_a: First ticker symbol.
        ticker_b: Second ticker symbol.
        horizon: Forecast horizon in trading days.
        lookback_years: Years of history to fetch for analysis.

    Returns:
        PipelineResult dict containing:
          - cointegration: CointegrationResult
          - spread_series: SpreadSeries
          - forecast: ForecastResult
          - signal: TradingSignal
    """
    end = date.today()
    start = end - timedelta(days=lookback_years * 365)

    fetcher = StockDataFetcher()
    prices = fetcher.fetch(
        tickers=[ticker_a, ticker_b],
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )

    # Extract Adj Close for each ticker
    prices_a = prices["Adj Close"][ticker_a].dropna()
    prices_b = prices["Adj Close"][ticker_b].dropna()

    # Align to common dates
    common_idx = prices_a.index.intersection(prices_b.index)
    prices_a = prices_a.loc[common_idx]
    prices_b = prices_b.loc[common_idx]

    # Cointegration
    analyzer = CointegrationAnalyzer()
    coint_result: CointegrationResult = analyzer.analyze(
        prices_a, prices_b, ticker_a, ticker_b
    )

    # Spread and z-score
    calc = SpreadCalculator()
    spread_series: SpreadSeries = calc.compute(prices_a, prices_b, coint_result.hedge_ratio)

    # TimesFM forecast
    loader = TimesFMLoader.get_instance()
    forecaster = SpreadForecaster(loader)
    signal_inputs = SpreadCalculator.get_current_signal_inputs(
        spread_series.spread, spread_series.zscore
    )
    forecast: ForecastResult = forecaster.forecast(
        spread_values=signal_inputs["spread"],
        horizon=horizon,
    )

    # Signal
    gen = SignalGenerator()
    signal: TradingSignal = gen.generate(
        zscore_current=float(spread_series.zscore.iloc[-1]),
        forecast_point=np.array(forecast.point_forecast),
        forecast_q10=np.array(forecast.q10),
        forecast_q90=np.array(forecast.q90),
    )

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "cointegration": coint_result,
        "spread_series": spread_series,
        "forecast": forecast,
        "signal": signal,
    }
