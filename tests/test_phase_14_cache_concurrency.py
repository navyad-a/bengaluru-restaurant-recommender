# -*- coding: utf-8 -*-
"""
Phase 14 Test Suite: Cache Lifecycle, Determinism, LRU Eviction & Concurrency Safety
"""

import pytest
import time
import concurrent.futures
from app.core.cache import RecommendationCache


@pytest.mark.unit
def test_cache_miss_and_hit_lifecycle():
    """Tests basic cache insertion, retrieval, and statistics."""
    cache = RecommendationCache(max_size=50, ttl_seconds=60)
    key = cache.generate_key("test", a=1, b="xyz")
    
    assert cache.get(key) is None
    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    
    cache.set(key, {"result": "data"})
    assert cache.get(key) == {"result": "data"}
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["total_keys"] == 1


@pytest.mark.unit
def test_cache_ttl_expiration():
    """Tests that cache entries expire after TTL elapsed."""
    cache = RecommendationCache(max_size=50, ttl_seconds=1)
    key = cache.generate_key("expiring", x=10)
    cache.set(key, "fresh_data")
    
    assert cache.get(key) == "fresh_data"
    time.sleep(1.1)
    # Expired
    assert cache.get(key) is None
    stats = cache.get_stats()
    assert stats["expirations"] >= 1


@pytest.mark.unit
def test_cache_lru_capacity_eviction():
    """Tests that exceeding max_size evicts the least recently accessed item."""
    cache = RecommendationCache(max_size=3, ttl_seconds=60)
    
    k1 = cache.generate_key("k", i=1)
    k2 = cache.generate_key("k", i=2)
    k3 = cache.generate_key("k", i=3)
    k4 = cache.generate_key("k", i=4)
    
    cache.set(k1, "v1")
    cache.set(k2, "v2")
    cache.set(k3, "v3")
    
    # Access k1 so k2 becomes the oldest unaccessed
    _ = cache.get(k1)
    
    # Adding k4 must evict k2
    cache.set(k4, "v4")
    
    assert cache.get(k1) == "v1"
    assert cache.get(k2) is None  # Evicted
    assert cache.get(k3) == "v3"
    assert cache.get(k4) == "v4"
    assert cache.get_stats()["total_keys"] == 3


@pytest.mark.unit
def test_cache_deterministic_key_ordering():
    """Verifies that dictionary key insertion order does not alter the generated cache key."""
    key_a = RecommendationCache.generate_key("hybrid", user_id=2, area="Indiranagar", top_k=10, mmr_lambda=0.75)
    key_b = RecommendationCache.generate_key("hybrid", mmr_lambda=0.75, top_k=10, user_id=2, area="Indiranagar")
    assert key_a == key_b


@pytest.mark.unit
def test_cache_different_parameters_produce_unique_keys():
    """Verifies that changing any recommendation parameter produces distinct keys."""
    k_base = RecommendationCache.generate_key("hybrid", user_id=2, top_k=10, mmr_lambda=0.75)
    k_diff_user = RecommendationCache.generate_key("hybrid", user_id=3, top_k=10, mmr_lambda=0.75)
    k_diff_k = RecommendationCache.generate_key("hybrid", user_id=2, top_k=20, mmr_lambda=0.75)
    k_diff_lambda = RecommendationCache.generate_key("hybrid", user_id=2, top_k=10, mmr_lambda=0.60)
    k_diff_cuisines = RecommendationCache.generate_key("hybrid", user_id=2, top_k=10, cuisines=["Biryani"])

    keys = {k_base, k_diff_user, k_diff_k, k_diff_lambda, k_diff_cuisines}
    assert len(keys) == 5


@pytest.mark.integration
def test_cache_concurrent_multithreaded_stress():
    """Stress tests the thread-safe cache across 16 parallel threads performing 300 operations."""
    cache = RecommendationCache(max_size=20, ttl_seconds=10)
    
    def worker(worker_id):
        for i in range(25):
            k = cache.generate_key("worker", id=worker_id, idx=i % 10)
            cache.set(k, f"val-{worker_id}-{i}")
            _ = cache.get(k)
            if i % 8 == 0:
                cache.clear()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(worker, w) for w in range(16)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    stats = cache.get_stats()
    assert stats["total_keys"] <= 20

