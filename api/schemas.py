"""Pydantic v2 request/response schemas for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------

class PairsDiscoverRequest(BaseModel):
    tickers: list[str] = Field(min_length=2)
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    significance: float = Field(default=0.05, gt=0, lt=1)


class PairResult(BaseModel):
    ticker_a: str
    ticker_b: str
    p_value: float
    hedge_ratio: float
    half_life_days: float
    cointegrated: bool


class PairsDiscoverResponse(BaseModel):
    pairs: list[PairResult]


class PairUniverseItem(BaseModel):
    name: str
    ticker_a: str
    ticker_b: str
    sector: str
    rationale: str


class PairsUniverseResponse(BaseModel):
    pairs: list[PairUniverseItem]


class NewsItem(BaseModel):
    uuid: str
    title: str
    publisher: str
    link: str
    providerPublishTime: int
    type: str
    ticker: str
    sentiment: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    sentiment_score: float


class NewsResponse(BaseModel):
    news: list[NewsItem]


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    context_days: int = Field(default=252, ge=32, le=16384)
    horizon_days: int = Field(default=30, ge=1, le=512)
    use_kalman: bool = False


class ForecastBands(BaseModel):
    point: list[float]
    q10: list[float]
    q50: list[float]
    q90: list[float]


class SignalDetail(BaseModel):
    action: str
    confidence: float
    z_score: float
    forecast_direction: str
    entry_price: dict[str, float]


class ForecastResponse(BaseModel):
    ticker_a: str
    ticker_b: str
    spread_history: list[float]
    spread_dates: list[str]
    forecast: ForecastBands
    signal: SignalDetail


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    entry_z: float = Field(default=2.0, gt=0)
    exit_z: float = Field(default=0.5, gt=0)
    transaction_cost_bps: float = Field(default=10.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)
    notional: float = Field(default=10_000.0, gt=0)
    use_kalman: bool = False


class BacktestMetricsSchema(BaseModel):
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_holding_days: float


class BacktestResponse(BaseModel):
    backtest_id: str
    ticker_a: str
    ticker_b: str
    metrics: BacktestMetricsSchema
    equity_curve: list[float]
    equity_dates: list[str]
    trades: list[dict]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: int
