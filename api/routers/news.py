"""GET /news — Latest news and sentiment for tickers."""

from __future__ import annotations

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from cachetools import TTLCache

from api.dependencies import get_ttl_cache
from api.schemas import NewsItem, NewsResponse
from pipeline.data.fetcher import StockDataFetcher
from pipeline.stats.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news")
_fetcher = StockDataFetcher()
_sentiment_analyzer = SentimentAnalyzer.get_instance()

@router.get("", response_model=NewsResponse)
def get_news(
    tickers: list[str] = Query(default=[], description="Tickers to fetch news for"),
    cache: TTLCache = Depends(get_ttl_cache)
) -> NewsResponse:
    """Fetch latest news for tickers and perform sentiment analysis."""
    all_news = []
    
    # If no tickers, use some defaults for general market news
    target_tickers = tickers if tickers else ["SPY", "AAPL", "TSLA", "NVDA"]
    
    for ticker in target_tickers:
        raw_news = _fetcher.fetch_news(ticker)
        
        if not raw_news:
            continue
            
        # Parse nested yfinance structure
        valid_items = []
        for n in raw_news:
            content = n.get("content", {})
            title = content.get("title", "")
            if not title:
                continue
                
            pub_date_str = content.get("pubDate", "")
            ts = 0
            if pub_date_str:
                try:
                    # '2026-04-18T11:38:47Z'
                    dt = datetime.strptime(pub_date_str.replace('Z', ''), '%Y-%m-%dT%H:%M:%S')
                    ts = int(dt.timestamp())
                except:
                    pass
            
            valid_items.append({
                "uuid": n.get("id", ""),
                "title": title,
                "publisher": content.get("provider", {}).get("displayName", "Yahoo Finance"),
                "link": content.get("canonicalUrl", {}).get("url", ""),
                "providerPublishTime": ts,
                "type": content.get("contentType", "STORY"),
                "ticker": ticker
            })

        if valid_items:
            titles = [v["title"] for v in valid_items]
            sentiments = _sentiment_analyzer.analyze_batch(titles)
            
            for i, item in enumerate(valid_items):
                all_news.append(NewsItem(
                    **item,
                    sentiment=sentiments[i]["label"],
                    sentiment_score=sentiments[i]["score"]
                ))
    
    # Sort by time desc
    all_news.sort(key=lambda x: x.providerPublishTime, reverse=True)
    
    return NewsResponse(news=all_news[:20])
