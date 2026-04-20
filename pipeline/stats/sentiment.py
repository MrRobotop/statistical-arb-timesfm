"""Sentiment analysis for financial news."""

from __future__ import annotations

import logging
from textblob import TextBlob
import nltk

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Singleton sentiment analyzer using TextBlob."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except Exception as e:
                logger.warning(f"NLTK download failed: {e}")
        return cls._instance

    @classmethod
    def get_instance(cls) -> SentimentAnalyzer:
        return cls()

    def analyze(self, text: str) -> dict:
        """Analyze sentiment of a single string.
        
        Returns:
            dict: {"label": "POSITIVE"|"NEGATIVE"|"NEUTRAL", "score": float}
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                label = "POSITIVE"
            elif polarity < -0.1:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
                
            return {
                "label": label,
                "score": abs(polarity)
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "NEUTRAL", "score": 0.0}

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze sentiment for a list of strings."""
        return [self.analyze(t) for t in texts]
