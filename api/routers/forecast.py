"""POST /forecast/spread — TimesFM spread forecast + signal."""

from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException
from cachetools import TTLCache

from api.dependencies import get_ttl_cache
from api.schemas import (
    ForecastBands,
    ForecastRequest,
    ForecastResponse,
    SignalDetail,
)
from pipeline.data.fetcher import StockDataFetcher
from pipeline.model.forecaster import ForecastError, SpreadForecaster
from pipeline.stats.signals import SignalGenerator
from pipeline.stats.spread import SpreadCalculator

router = APIRouter(prefix="/forecast")
_fetcher = StockDataFetcher()
_forecaster = SpreadForecaster()
_signal_gen = SignalGenerator()
_spread_calc = SpreadCalculator()


@router.post("/spread", response_model=ForecastResponse)
def forecast_spread(
    req: ForecastRequest,
    cache: TTLCache = Depends(get_ttl_cache),
) -> ForecastResponse:
    """Forecast spread and generate a trading signal using TimesFM."""
    cache_key = "fc:" + hashlib.md5(
        json.dumps(req.model_dump(), sort_keys=True).encode()
    ).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    # Fetch ~2 years of data for context
    try:
        df = _fetcher.fetch(
            [req.ticker_a, req.ticker_b],
            start_date="2022-01-01",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Data fetch failed: {exc}") from exc

    if req.ticker_a not in df.columns or req.ticker_b not in df.columns:
        raise HTTPException(status_code=422, detail="One or both tickers not found.")

    prices_a = df[req.ticker_a]["Close"]
    prices_b = df[req.ticker_b]["Close"]

    if req.use_kalman:
        spread_series = _spread_calc.compute_kalman(prices_a, prices_b)
    else:
        spread_series = _spread_calc.compute(prices_a, prices_b, req.hedge_ratio)
        
    signal_inputs = _spread_calc.get_current_signal_inputs(
        spread_series.spread,
        spread_series.zscore,
        lookback=req.context_days,
    )
    spread_arr = signal_inputs["spread"]
    zscore_arr = signal_inputs["zscore"]

    # TimesFM forecast (may raise RuntimeError if model not loaded / auth fails)
    try:
        fc = _forecaster.forecast(spread_arr, horizon=req.horizon_days)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ForecastError as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    # Signal
    z_now = float(zscore_arr[-1])
    signal = _signal_gen.generate(
        zscore_current=z_now,
        forecast_point=fc.point_forecast,
        forecast_q10=fc.q10,
        forecast_q90=fc.q90,
    )

    # Latest prices for entry_price field
    last_a = float(prices_a.iloc[-1])
    last_b = float(prices_b.iloc[-1])

    response = ForecastResponse(
        ticker_a=req.ticker_a,
        ticker_b=req.ticker_b,
        spread_history=[round(v, 6) for v in spread_series.spread.tolist()[-252:]],
        spread_dates=[str(d.date()) for d in spread_series.spread.index[-252:]],
        forecast=ForecastBands(
            point=[round(v, 6) for v in fc.point_forecast],
            q10=[round(v, 6) for v in fc.q10],
            q50=[round(v, 6) for v in fc.q50],
            q90=[round(v, 6) for v in fc.q90],
        ),
        signal=SignalDetail(
            action=signal.action,
            confidence=round(signal.confidence, 4),
            z_score=round(z_now, 4),
            forecast_direction=fc.forecast_direction,
            entry_price={req.ticker_a: round(last_a, 2), req.ticker_b: round(last_b, 2)},
        ),
    )
    cache[cache_key] = response
    return response
