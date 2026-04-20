"""Tests for StockDataFetcher cache logic and error handling."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.data.fetcher import DataFetchError, StockDataFetcher


def _make_ohlcv(ticker: str = "KO") -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame mimicking yfinance output."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "Open": [50.0] * 10,
            "High": [51.0] * 10,
            "Low": [49.0] * 10,
            "Close": [50.5] * 10,
            "Adj Close": [50.5] * 10,
            "Volume": [1_000_000] * 10,
        },
        index=dates,
    )


class TestStockDataFetcherCache:
    def test_cache_miss_calls_yfinance(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)
        mock_df = _make_ohlcv()

        with patch("pipeline.data.fetcher.yf.download", return_value=mock_df) as mock_dl:
            fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")

        mock_dl.assert_called_once()

    def test_cache_hit_skips_yfinance(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)
        mock_df = _make_ohlcv()

        with patch("pipeline.data.fetcher.yf.download", return_value=mock_df):
            fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")

        # Second call should not hit yfinance
        with patch("pipeline.data.fetcher.yf.download") as mock_dl:
            fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")
            mock_dl.assert_not_called()

    def test_stale_cache_refreshes(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)
        mock_df = _make_ohlcv()

        with patch("pipeline.data.fetcher.yf.download", return_value=mock_df):
            fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")

        # Backdate the cache file by 25 hours
        cache_file = list(tmp_path.glob("*.parquet"))[0]
        stale_mtime = cache_file.stat().st_mtime - (25 * 3600)
        import os
        os.utime(cache_file, (stale_mtime, stale_mtime))

        with patch("pipeline.data.fetcher.yf.download", return_value=mock_df) as mock_dl:
            fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")
            mock_dl.assert_called_once()

    def test_returns_multiindex_dataframe(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)

        with patch("pipeline.data.fetcher.yf.download", return_value=_make_ohlcv()):
            result = fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")

        assert isinstance(result.columns, pd.MultiIndex)

    def test_multiple_tickers_combined(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)

        with patch("pipeline.data.fetcher.yf.download", return_value=_make_ohlcv()):
            result = fetcher.fetch(["KO", "PEP"], "2023-01-01", "2023-01-31")

        tickers_in_result = result.columns.get_level_values("Ticker").unique().tolist()
        assert "KO" in tickers_in_result
        assert "PEP" in tickers_in_result


class TestDataFetchError:
    def test_empty_response_raises(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)

        with patch("pipeline.data.fetcher.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(DataFetchError, match="No data returned"):
                fetcher.fetch(["INVALID_TICKER_XYZ"], "2023-01-01", "2023-01-31")

    def test_persistent_network_error_raises(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)

        with patch(
            "pipeline.data.fetcher.yf.download",
            side_effect=ConnectionError("network down"),
        ), patch("pipeline.data.fetcher.time.sleep"):
            with pytest.raises(DataFetchError, match="Failed to fetch"):
                fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")

    def test_retries_before_raising(self, tmp_path: Path) -> None:
        fetcher = StockDataFetcher(cache_dir=tmp_path)
        call_count = 0

        def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("timeout")

        with patch("pipeline.data.fetcher.yf.download", side_effect=flaky), \
             patch("pipeline.data.fetcher.time.sleep"):
            with pytest.raises(DataFetchError):
                fetcher.fetch(["KO"], "2023-01-01", "2023-01-31")

        assert call_count == 3  # max_retries
