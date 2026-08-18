# -*- coding: utf-8 -*-
"""
Recommendation Cache Layer (Multi-Backend: Thread-Safe In-Memory TTL LRU & Redis)
Supports deterministic key generation, TTL expiration, Redis persistence, and statistics tracking.
"""

import time
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
from app.config import settings
from app.core.logging import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class BaseRecommendationCache:
    """Abstract interface for recommendation cache backends."""

    @staticmethod
    def generate_key(prefix: str, **kwargs) -> str:
        """
        Generates a deterministic hash key from prefix and sorted arguments.
        Floats are rounded to 4 decimal places (~11 meters) for spatial tolerance.
        """
        normalized = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, float):
                normalized[k] = round(v, 4)
            elif isinstance(v, (dict, list)):
                normalized[k] = v
            else:
                normalized[k] = str(v).strip().lower()

        serialized = json.dumps(normalized, sort_keys=True, default=str)
        hash_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}:{hash_digest}"

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class InMemoryRecommendationCache(BaseRecommendationCache):
    """
    Thread-safe in-memory TTL LRU Cache for recommendation responses.
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 300,
        max_size: int = 1000
    ):
        self.enabled = bool(enabled)
        self.ttl_seconds = int(ttl_seconds)
        self.max_size = int(max_size)
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieves value from cache if present and unexpired."""
        if not self.enabled:
            return None

        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            value, timestamp = self._cache[key]
            now = time.time()

            if now - timestamp > self.ttl_seconds:
                del self._cache[key]
                self.expirations += 1
                self.misses += 1
                return None

            # Move to end for LRU order
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Inserts or updates value in cache with timestamp and LRU eviction."""
        if not self.enabled:
            return

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())

            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        """Clears all entries from cache and resets counters."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache performance metrics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_ratio = (self.hits / total_requests) if total_requests > 0 else 0.0
            return {
                "backend": "memory",
                "enabled": self.enabled,
                "total_keys": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total_requests,
                "hit_ratio": round(hit_ratio, 4),
                "evictions": self.evictions,
                "expirations": self.expirations
            }


# Backward compatibility alias
RecommendationCache = InMemoryRecommendationCache


class RedisRecommendationCache(BaseRecommendationCache):
    """
    Distributed Redis Cache for recommendation responses across multi-node deployments.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        enabled: bool = True,
        ttl_seconds: int = 300,
        max_size: int = 1000
    ):
        self.enabled = bool(enabled)
        self.redis_url = redis_url
        self.ttl_seconds = int(ttl_seconds)
        self.max_size = int(max_size)
        self.hits = 0
        self.misses = 0
        self._client: Optional[Any] = None

        if not REDIS_AVAILABLE:
            logger.warning("redis-py library is not installed. Redis cache disabled.")
            self.enabled = False
            return

        try:
            self._client = redis.from_url(redis_url, decode_responses=True, socket_timeout=2.0)
            self._client.ping()
            logger.info(f"Connected to Redis cache at {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {redis_url}: {e}. Operating in pass-through mode.")
            self._client = None

    def get(self, key: str) -> Optional[Any]:
        """Retrieves and deserializes JSON recommendation payload from Redis."""
        if not self.enabled or self._client is None:
            return None

        try:
            val = self._client.get(key)
            if val is not None:
                self.hits += 1
                return json.loads(val)
            self.misses += 1
            return None
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Serializes and persists recommendation response in Redis with TTL expiration."""
        if not self.enabled or self._client is None:
            return

        try:
            serialized = json.dumps(value, default=str)
            expiry = ttl if ttl is not None else self.ttl_seconds
            self._client.set(key, serialized, ex=expiry)
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")

    def clear(self) -> None:
        """Flushes all recommendation keys from Redis."""
        if not self.enabled or self._client is None:
            return

        try:
            self._client.flushdb()
            self.hits = 0
            self.misses = 0
            logger.info("Redis recommendation cache flushed.")
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Returns Redis cache status and metrics."""
        total_keys = 0
        is_connected = False

        if self._client is not None:
            try:
                self._client.ping()
                is_connected = True
                total_keys = self._client.dbsize()
            except Exception:
                is_connected = False

        total_requests = self.hits + self.misses
        hit_ratio = (self.hits / total_requests) if total_requests > 0 else 0.0

        return {
            "backend": "redis",
            "enabled": self.enabled,
            "connected": is_connected,
            "total_keys": total_keys,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_ratio": round(hit_ratio, 4)
        }


_recommendation_cache: Optional[BaseRecommendationCache] = None


def get_recommendation_cache() -> BaseRecommendationCache:
    """Returns singleton instance of recommendation cache according to configured backend."""
    global _recommendation_cache
    if _recommendation_cache is None:
        backend = getattr(settings, "CACHE_BACKEND", "memory").lower()
        if backend == "redis" and REDIS_AVAILABLE:
            try:
                _recommendation_cache = RedisRecommendationCache(
                    redis_url=settings.REDIS_URL,
                    enabled=settings.RECOMMENDATION_CACHE_ENABLED,
                    ttl_seconds=settings.RECOMMENDATION_CACHE_TTL_SECONDS,
                    max_size=settings.RECOMMENDATION_CACHE_MAX_SIZE
                )
            except Exception as e:
                logger.warning(f"Failed initializing RedisRecommendationCache: {e}. Falling back to in-memory cache.")
                _recommendation_cache = InMemoryRecommendationCache(
                    enabled=settings.RECOMMENDATION_CACHE_ENABLED,
                    ttl_seconds=settings.RECOMMENDATION_CACHE_TTL_SECONDS,
                    max_size=settings.RECOMMENDATION_CACHE_MAX_SIZE
                )
        else:
            _recommendation_cache = InMemoryRecommendationCache(
                enabled=settings.RECOMMENDATION_CACHE_ENABLED,
                ttl_seconds=settings.RECOMMENDATION_CACHE_TTL_SECONDS,
                max_size=settings.RECOMMENDATION_CACHE_MAX_SIZE
            )
    return _recommendation_cache

