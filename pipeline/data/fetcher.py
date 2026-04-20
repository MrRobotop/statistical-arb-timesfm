"""yfinance data fetching with parquet caching and exponential backoff."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("YFINANCE_CACHE_DIR", "./data/cache"))
CACHE_TTL_HOURS = 24


class DataFetchError(Exception):
    """Raised when data cannot be fetched after all retries."""


class StockDataFetcher:
    """Fetches OHLCV data from yfinance with parquet caching."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start_date: str, end_date: str) -> Path:
        key = f"{ticker}_{start_date}_{end_date}".replace("-", "")
        return self.cache_dir / f"{key}.parquet"

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(hours=CACHE_TTL_HOURS)

    def _fetch_single(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch one ticker with exponential backoff retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                )
                if df.empty:
                    raise DataFetchError(f"No data returned for ticker '{ticker}'")
                return df
            except DataFetchError:
                raise
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise DataFetchError(
                        f"Failed to fetch '{ticker}' after {max_retries} attempts: {exc}"
                    ) from exc
                wait = 2 ** attempt
                logger.warning("Retry %d for %s in %ds: %s", attempt + 1, ticker, wait, exc)
                time.sleep(wait)
        raise DataFetchError(f"Exhausted retries for '{ticker}'")

    def fetch(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for multiple tickers, using parquet cache.

        Returns a MultiIndex DataFrame: columns = (field, ticker),
        rows = dates. Fields include Open, High, Low, Close, Volume, Adj Close.
        """
        # Defaults: last 5 years to today
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=365 * 5)
            start_date = start_dt.strftime("%Y-%m-%d")

        frames: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            cache_path = self._cache_path(ticker, start_date, end_date)

            if self._cache_valid(cache_path):
                logger.debug("Cache hit for %s", ticker)
                df = pd.read_parquet(cache_path)
            else:
                logger.info("Fetching %s from yfinance", ticker)
                df = self._fetch_single(ticker, start_date, end_date)
                # Flatten MultiIndex columns if yfinance returns them
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.to_parquet(cache_path)

            frames[ticker] = df

        # Build MultiIndex DataFrame: (Ticker, Field)
        combined = pd.concat(frames, axis=1)
        combined.columns.names = ["Ticker", "Field"]
        combined = combined.sort_index(axis=1)
        return combined

    def fetch_news(self, ticker: str) -> list[dict]:
        """Fetch latest news for a ticker from yfinance."""
        try:
            t = yf.Ticker(ticker)
            return t.news
        except Exception as exc:
            logger.error("Failed to fetch news for %s: %s", ticker, exc)
            return []
