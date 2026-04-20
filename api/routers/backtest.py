"""POST /backtest/run, GET /backtest/{id}."""

from __future__ import annotations

import hashlib
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from cachetools import TTLCache

from api.dependencies import get_ttl_cache
from api.schemas import BacktestMetricsSchema, BacktestRequest, BacktestResponse
from pipeline.backtest.engine import BacktestEngine
from pipeline.data.fetcher import StockDataFetcher

router = APIRouter(prefix="/backtest")
_fetcher = StockDataFetcher()
_engine = BacktestEngine()

# Persistent store keyed by backtest_id (lives as long as the process)
_results: dict[str, BacktestResponse] = {}


@router.post("/run", response_model=BacktestResponse)
def run_backtest(
    req: BacktestRequest,
    cache: TTLCache = Depends(get_ttl_cache),
) -> BacktestResponse:
    """Run a full vectorized backtest for a pair."""
    cache_key = "bt:" + hashlib.md5(
        json.dumps(req.model_dump(), sort_keys=True).encode()
    ).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    try:
        df = _fetcher.fetch(
            [req.ticker_a, req.ticker_b],
            start_date=req.start_date,
            end_date=req.end_date,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Data fetch failed: {exc}") from exc

    if req.ticker_a not in df.columns or req.ticker_b not in df.columns:
        raise HTTPException(status_code=422, detail="One or both tickers not found in data.")

    try:
        result = _engine.run(
            prices_a=df[req.ticker_a]["Close"],
            prices_b=df[req.ticker_b]["Close"],
            hedge_ratio=req.hedge_ratio,
            entry_z=req.entry_z,
            exit_z=req.exit_z,
            transaction_cost_bps=req.transaction_cost_bps,
            slippage_bps=req.slippage_bps,
            notional=req.notional,
            use_kalman=req.use_kalman,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc

    bt_id = "bt_" + uuid.uuid4().hex[:8]
    response = BacktestResponse(
        backtest_id=bt_id,
        ticker_a=req.ticker_a,
        ticker_b=req.ticker_b,
        metrics=BacktestMetricsSchema(**result.metrics.model_dump()),
        equity_curve=result.equity_curve,
        equity_dates=result.equity_dates,
        trades=result.trades,
    )
    _results[bt_id] = response
    cache[cache_key] = response
    return response


@router.get("/{backtest_id}", response_model=BacktestResponse)
def get_backtest(backtest_id: str) -> BacktestResponse:
    """Retrieve a previously run backtest by ID."""
    if backtest_id not in _results:
        raise HTTPException(status_code=404, detail=f"Backtest '{backtest_id}' not found.")
    return _results[backtest_id]
