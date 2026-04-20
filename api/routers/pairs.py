"""GET /pairs/universe, POST /pairs/discover."""

from __future__ import annotations

import hashlib
import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from cachetools import TTLCache

from api.dependencies import get_ttl_cache
from api.schemas import (
    PairsDiscoverRequest,
    PairsDiscoverResponse,
    PairResult,
    PairUniverseItem,
    PairsUniverseResponse,
)
from pipeline.data.fetcher import StockDataFetcher
from pipeline.data.universe import list_pairs
from pipeline.stats.cointegration import CointegrationAnalyzer

router = APIRouter(prefix="/pairs")
_fetcher = StockDataFetcher()
_analyzer = CointegrationAnalyzer()


@router.get("/universe", response_model=PairsUniverseResponse)
def get_universe() -> PairsUniverseResponse:
    """Return the curated pairs universe."""
    return PairsUniverseResponse(
        pairs=[
            PairUniverseItem(
                name=p.name,
                ticker_a=p.ticker_a,
                ticker_b=p.ticker_b,
                sector=p.sector,
                rationale=p.rationale,
            )
            for p in list_pairs()
        ]
    )


@router.post("/discover", response_model=PairsDiscoverResponse)
def discover_pairs(
    req: PairsDiscoverRequest,
    cache: TTLCache = Depends(get_ttl_cache),
) -> PairsDiscoverResponse:
    """Run cointegration tests on all pairs from the provided ticker list."""
    cache_key = hashlib.md5(
        json.dumps(req.model_dump(), sort_keys=True).encode()
    ).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    try:
        df = _fetcher.fetch(req.tickers, start_date=req.start_date, end_date=req.end_date)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Data fetch failed: {exc}") from exc

    results: list[PairResult] = []
    for i, ta in enumerate(req.tickers):
        for tb in req.tickers[i + 1 :]:
            if ta not in df.columns or tb not in df.columns:
                continue
            try:
                prices_a = df[ta]["Close"]
                prices_b = df[tb]["Close"]
                coint = _analyzer.analyze(prices_a, prices_b)
                results.append(
                    PairResult(
                        ticker_a=ta,
                        ticker_b=tb,
                        p_value=round(coint.eg_pvalue, 6),
                        hedge_ratio=round(coint.hedge_ratio, 6),
                        half_life_days=round(coint.half_life_days, 2),
                        cointegrated=coint.is_cointegrated,
                    )
                )
            except Exception:
                continue

    response = PairsDiscoverResponse(pairs=results)
    cache[cache_key] = response
    return response
