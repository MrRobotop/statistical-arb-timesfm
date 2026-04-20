"""TTL-aware LRU cache for API responses."""

from __future__ import annotations

from cachetools import TTLCache

# 512 items max, 5-minute TTL
_cache: TTLCache = TTLCache(maxsize=512, ttl=300)


def get_cache() -> TTLCache:
    """FastAPI dependency — returns the shared cache instance."""
    return _cache
