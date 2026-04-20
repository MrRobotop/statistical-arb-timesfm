"""FastAPI dependency providers."""

from __future__ import annotations

from cachetools import TTLCache

from api.cache import get_cache
from pipeline.model.loader import TimesFMLoader


def get_model_loader() -> TimesFMLoader:
    """Return the TimesFM singleton loader (does not trigger model load)."""
    return TimesFMLoader.get_instance()


def get_ttl_cache() -> TTLCache:
    """Return the shared TTL cache."""
    return get_cache()
