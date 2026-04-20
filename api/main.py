"""FastAPI application factory."""

from __future__ import annotations

import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import backtest, forecast, health, pairs, news
from pipeline.model.loader import TimesFMLoader

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    loader = TimesFMLoader.get_instance()
    if not loader.is_loaded():
        try:
            loader.load()
        except Exception as e:
            print(f"CRITICAL: Failed to load TimesFM model: {e}")
    yield
    # Clean up on shutdown if needed

app = FastAPI(
    title="PairsTrader API",
    description="Statistical arbitrage spread predictor powered by TimesFM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS — origins from env, comma-separated
_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(pairs.router, tags=["pairs"])
app.include_router(forecast.router, tags=["forecast"])
app.include_router(backtest.router, tags=["backtest"])
app.include_router(news.router, tags=["news"])
