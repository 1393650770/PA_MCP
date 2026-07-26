# [AI:BEGIN]
# PA_MCP - Data Layer: Cache
# Multi-level cache: L1 (lru_cache in-process) + L2 (optional Redis).
# [AI:END]

from __future__ import annotations

import functools
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import structlog

from pa_mcp.config import CacheSettings

logger = structlog.get_logger(__name__)


def _make_cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """Generate a deterministic cache key."""
    raw = f"{prefix}:{json.dumps(args, sort_keys=True, default=str)}:{json.dumps(kwargs, sort_keys=True, default=str)}"
    return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()[:16]}"


class MemoryCache:
    """In-process LRU cache with TTL support."""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        self._store: dict[str, tuple[Any, datetime]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired."""
        if key in self._store:
            value, expiry = self._store[key]
            if datetime.now() < expiry:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set value with TTL."""
        # Evict oldest if at capacity
        if len(self._store) >= self.max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]

        expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        self._store[key] = (value, expiry)

    def invalidate(self, prefix: str) -> int:
        """Remove all keys starting with prefix. Returns count removed."""
        to_remove = [k for k in self._store if k.startswith(prefix)]
        for k in to_remove:
            del self._store[k]
        return len(to_remove)

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        now = datetime.now()
        active = sum(1 for _, (_, expiry) in self._store.items() if now < expiry)
        return {"size": len(self._store), "active": active, "max_size": self.max_size}


class RedisCache:
    """Redis-backed cache (optional, for multi-instance deployments)."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import redis
            self._client = redis.from_url(self.redis_url, decode_responses=False)
        return self._client

    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(key)
            if data:
                return pd.read_parquet(pd.io.common.BytesIO(data))
            return None
        except Exception as e:
            logger.warning("Redis get failed", key=key, error=str(e))
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        try:
            if isinstance(value, pd.DataFrame):
                buf = pd.io.common.BytesIO()
                value.to_parquet(buf, index=False)
                data = buf.getvalue()
            else:
                data = json.dumps(value, default=str).encode()
            self.client.setex(key, ttl_seconds, data)
        except Exception as e:
            logger.warning("Redis set failed", key=key, error=str(e))


class CacheManager:
    """Multi-level cache manager.

    Priority: L1 (memory) > L2 (Redis, optional) > direct query

    Usage:
        cache = CacheManager(settings)
        df = cache.get_or_set("kline:000001:daily", fetch_func, ttl=21600)
    """

    def __init__(self, settings: Optional[CacheSettings] = None) -> None:
        if settings is None:
            from pa_mcp.config import get_settings
            settings = get_settings().cache

        self.settings = settings
        self._l1 = MemoryCache(max_size=1024)
        self._l2: Optional[RedisCache] = None

        if settings.backend == "redis":
            self._l2 = RedisCache(settings.redis_url)
            logger.info("Cache backend: memory + Redis")
        else:
            logger.info("Cache backend: memory only", backend=settings.backend)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks L1 then L2)."""
        # L1: memory
        value = self._l1.get(key)
        if value is not None:
            logger.debug("Cache hit: L1", key=key)
            return value

        # L2: Redis
        if self._l2 is not None:
            value = self._l2.get(key)
            if value is not None:
                # Backfill L1
                self._l1.set(key, value, self.settings.default_ttl)
                logger.debug("Cache hit: L2", key=key)
                return value

        logger.debug("Cache miss", key=key)
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in all cache layers."""
        if ttl_seconds is None:
            ttl_seconds = self.settings.default_ttl

        self._l1.set(key, value, ttl_seconds)
        if self._l2 is not None:
            self._l2.set(key, value, ttl_seconds)

    async def get_or_set(
        self, key: str, fetch_func, ttl_seconds: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Any:
        """Get from cache or fetch and cache.

        Args:
            key: Cache key
            fetch_func: Async function to call on cache miss
            ttl_seconds: Cache TTL in seconds
            force_refresh: If True, skip cache and re-fetch
        """
        if not force_refresh:
            cached = self.get(key)
            if cached is not None:
                return cached

        # Cache miss — fetch fresh data
        value = await fetch_func()
        self.set(key, value, ttl_seconds or self.settings.default_ttl)
        return value

    def invalidate(self, prefix: str) -> int:
        """Invalidate all keys with given prefix."""
        return self._l1.invalidate(prefix)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {"l1": self._l1.stats()}
