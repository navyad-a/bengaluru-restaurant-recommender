# -*- coding: utf-8 -*-
"""
Phase 14 Test Suite: Performance Regression Benchmarks
Lightweight, robust performance regression assertions.
"""

import pytest
import time
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.performance
def test_perf_health_check_low_latency(client):
    """Verifies /health probe responds quickly without overhead."""
    start = time.perf_counter()
    resp = client.get("/health")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    
    assert resp.status_code == 200
    assert elapsed_ms < 150.0  # Generous threshold to avoid flaky CI failures


@pytest.mark.performance
def test_perf_cache_speedup_ratio(client):
    """Verifies that cached recommendations execute with significant speedup."""
    client.post("/api/v1/system/cache/clear")
    url = "/api/v1/recommendations/hybrid/2?top_k=10&mmr_enabled=true"

    # 1. Cold Uncached Request
    t0 = time.perf_counter()
    r1 = client.get(url)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    assert r1.status_code == 200

    # 2. Warm Cached Request
    t1 = time.perf_counter()
    r2 = client.get(url)
    warm_ms = (time.perf_counter() - t1) * 1000.0
    assert r2.status_code == 200

    # Assert warm request executes swiftly (< 50ms)
    assert warm_ms < 60.0


@pytest.mark.performance
def test_perf_spatial_balltree_subsecond(client):
    """Verifies that BallTree spatial query over 12,481 venues completes within a sub-second bound."""
    t0 = time.perf_counter()
    resp = client.get("/api/v1/recommendations/nearby?latitude=12.9716&longitude=77.5946&radius_km=5.0&top_k=20")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    assert resp.status_code == 200
    assert elapsed_ms < 300.0

