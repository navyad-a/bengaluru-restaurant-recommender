# -*- coding: utf-8 -*-
"""
Phase 12 Test Suite: FastAPI Async Performance, Concurrency, Caching & Telemetry
"""

import pytest
import concurrent.futures
from fastapi.testclient import TestClient
from app.main import app
from app.core.cache import get_recommendation_cache, RecommendationCache
from app.config import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_health_liveness_endpoint(client):
    """Verifies lightweight /health liveness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data


def test_readiness_probe_endpoint(client):
    """Verifies /ready endpoint accurately checks all underlying ML engines."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["content_recommender"] is True
    assert data["checks"]["collaborative_recommender"] is True
    assert data["checks"]["spatial_search_engine"] is True
    assert data["checks"]["hybrid_recommender"] is True


def test_system_status_telemetry(client):
    """Verifies /api/v1/system/status provides complete runtime telemetry."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["uptime_seconds"] >= 0.0
    assert "models" in data
    assert "cache" in data
    assert "concurrency" in data


def test_request_id_and_timing_headers(client):
    """Verifies X-Request-ID and X-Process-Time-Ms headers on all API responses."""
    # Test with custom request id
    custom_id = "test-req-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers.get("X-Request-ID") == custom_id
    assert "X-Process-Time-Ms" in response.headers
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0

    # Test with auto-generated request id
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) >= 8


def test_cache_hit_and_miss_lifecycle(client):
    """Verifies recommendation response caching, hits, and cache purge."""
    cache = get_recommendation_cache()
    cache.clear()

    # 1. First request -> Cache Miss
    res1 = client.get("/api/v1/recommendations/popular?top_k=5")
    assert res1.status_code == 200
    stats1 = cache.get_stats()
    assert stats1["misses"] >= 1

    # 2. Second identical request -> Cache Hit
    res2 = client.get("/api/v1/recommendations/popular?top_k=5")
    assert res2.status_code == 200
    stats2 = cache.get_stats()
    assert stats2["hits"] >= 1
    assert res1.json() == res2.json()

    # 3. Clear cache
    clear_res = client.post("/api/v1/system/cache/clear")
    assert clear_res.status_code == 200
    assert cache.get_stats()["total_keys"] == 0


def test_cache_deterministic_spatial_key_rounding():
    """Verifies coordinates within ~11 meters produce identical cache keys."""
    key1 = RecommendationCache.generate_key("nearby", lat=12.97161, lon=77.59461, k=5)
    key2 = RecommendationCache.generate_key("nearby", lat=12.97159, lon=77.59463, k=5)
    # Rounded to 4 decimal places: 12.9716, 77.5946
    assert key1 == key2


def test_structured_error_response_404(client):
    """Verifies structured error response on 404."""
    response = client.get("/api/v1/recommendations/similar/99999999")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "RESOURCE_NOT_FOUND"
    assert "not found" in data["message"].lower()
    assert "request_id" in data


def test_structured_error_response_422(client):
    """Verifies structured error response on Pydantic validation failure."""
    response = client.get("/api/v1/recommendations/nearby?latitude=999.0&longitude=500.0")
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "request_id" in data


def test_concurrent_recommendation_requests_safety(client):
    """
    Spawns 12 concurrent worker threads executing hybrid recommendation queries
    to verify thread safety, determinism, and zero race conditions.
    """
    def send_request(user_id):
        return client.get(f"/api/v1/recommendations/hybrid/{user_id}?top_k=5")

    user_ids = [2, 4, 5, 6, 7, 8, 9, 10, 2, 4, 5, 6]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        responses = list(executor.map(send_request, user_ids))

    assert len(responses) == len(user_ids)
    for r in responses:
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert len(data["recommendations"]) == 5

    # Check identical users produced identical deterministic results
    user_2_runs = [r.json()["recommendations"] for i, r in enumerate(responses) if user_ids[i] == 2]
    assert user_2_runs[0] == user_2_runs[1]

