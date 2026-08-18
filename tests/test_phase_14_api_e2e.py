# -*- coding: utf-8 -*-
"""
Phase 14 Test Suite: Exhaustive API End-to-End Test Matrix
Tests all 8 recommendation endpoints + 4 system telemetry endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# 1. /similar & /content Endpoints
# =============================================================================

@pytest.mark.api
def test_e2e_similar_restaurant_endpoint(client):
    """Tests GET /api/v1/recommendations/similar/{restaurant_id}."""
    # Valid call
    resp = client.get("/api/v1/recommendations/similar/1?top_k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 5
    assert all("similarity_score" in r for r in data["recommendations"])

    # Invalid ID -> 404
    resp_404 = client.get("/api/v1/recommendations/similar/99999999")
    assert resp_404.status_code == 404
    assert resp_404.json()["error_code"] == "RESOURCE_NOT_FOUND"


@pytest.mark.api
def test_e2e_content_preferences_endpoint(client):
    """Tests POST /api/v1/recommendations/content."""
    payload = {
        "preferred_cuisines": ["South Indian", "Kerala"],
        "preferred_area": "Indiranagar",
        "max_cost_for_two": 800,
        "top_k": 5
    }
    resp = client.post("/api/v1/recommendations/content", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) <= 5
    for r in data["recommendations"]:
        assert r["cost_for_two_inr"] <= 800


# =============================================================================
# 2. /collaborative & /nearby Endpoints
# =============================================================================

@pytest.mark.api
def test_e2e_collaborative_endpoint(client):
    """Tests GET /api/v1/recommendations/collaborative/{user_id}."""
    # Known benchmark user
    resp = client.get("/api/v1/recommendations/collaborative/2?top_k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["user_id"] == 2
    assert "synthetic" in data["disclaimer"].lower()

    # Unknown user -> 404
    resp_404 = client.get("/api/v1/recommendations/collaborative/999999")
    assert resp_404.status_code == 404


@pytest.mark.api
def test_e2e_nearby_spatial_endpoint(client):
    """Tests GET /api/v1/recommendations/nearby."""
    # Point near Vidhana Soudha / MG Road
    resp = client.get("/api/v1/recommendations/nearby?latitude=12.9716&longitude=77.5946&radius_km=3.0&top_k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) <= 5
    for r in data["recommendations"]:
        assert r["distance_km"] <= 3.0
        assert "centroid" in r["location_source"].lower()


# =============================================================================
# 3. /hybrid (GET & POST) Endpoints
# =============================================================================

@pytest.mark.api
def test_e2e_hybrid_get_endpoint(client):
    """Tests GET /api/v1/recommendations/hybrid/{user_id}."""
    # Known user with MMR
    resp = client.get("/api/v1/recommendations/hybrid/2?top_k=5&mmr_enabled=true&mmr_lambda=0.75")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["user_id"] == 2
    assert data["is_cold_start"] is False
    assert data["diversification"]["enabled"] is True
    assert data["diversification"]["lambda_param"] == 0.75

    # Cold-start fallback for unknown user
    resp_cold = client.get("/api/v1/recommendations/hybrid/999999?top_k=5")
    assert resp_cold.status_code == 200
    data_cold = resp_cold.json()
    assert data_cold["is_cold_start"] is True


@pytest.mark.api
def test_e2e_hybrid_post_endpoint(client):
    """Tests POST /api/v1/recommendations/hybrid."""
    payload = {
        "user_id": 2,
        "preferred_cuisines": ["Biryani", "North Indian"],
        "preferred_area": "Koramangala 5th Block",
        "max_cost_for_two": 1200,
        "min_rating": 3.8,
        "custom_weights": {"collaborative": 0.3, "content": 0.4, "location": 0.1, "quality": 0.2},
        "mmr_enabled": True,
        "mmr_lambda": 0.70,
        "top_k": 8
    }
    resp = client.post("/api/v1/recommendations/hybrid", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) <= 8
    assert data["diversification"]["lambda_param"] == 0.70


# =============================================================================
# 4. /popular & /onboarding Endpoints
# =============================================================================

@pytest.mark.api
def test_e2e_popular_endpoint(client):
    """Tests GET /api/v1/recommendations/popular."""
    resp = client.get("/api/v1/recommendations/popular?area=Indiranagar&top_k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "Indiranagar" in data["scope"]
    assert len(data["recommendations"]) == 5


@pytest.mark.api
def test_e2e_onboarding_endpoint(client):
    """Tests POST /api/v1/recommendations/onboarding."""
    payload = {
        "favorite_cuisines": ["Mughlai", "Biryani"],
        "preferred_area": "Frazer Town",
        "price_tier": "Moderate",
        "max_budget_for_two": 1000,
        "top_k": 5
    }
    resp = client.post("/api/v1/recommendations/onboarding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["is_cold_start"] is True
    assert len(data["recommendations"]) <= 5


# =============================================================================
# 5. System Endpoints & Determinism
# =============================================================================

@pytest.mark.api
def test_e2e_system_endpoints(client):
    """Tests /health, /ready, /api/v1/system/status, and /api/v1/system/cache/clear."""
    h_resp = client.get("/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["status"] == "healthy"

    r_resp = client.get("/ready")
    assert r_resp.status_code == 200
    assert r_resp.json()["status"] == "ready"

    s_resp = client.get("/api/v1/system/status")
    assert s_resp.status_code == 200
    assert s_resp.json()["models"]["content_recommender"]["catalog_size"] == 12481

    c_resp = client.post("/api/v1/system/cache/clear")
    assert c_resp.status_code == 200
    assert c_resp.json()["status"] == "success"


@pytest.mark.api
def test_e2e_deterministic_recommendation_ordering(client):
    """Verifies that repeat identical recommendation calls produce identically ordered items."""
    url = "/api/v1/recommendations/hybrid/2?top_k=10&mmr_enabled=false"
    r1 = client.get(url).json()["recommendations"]
    r2 = client.get(url).json()["recommendations"]
    assert [x["restaurant_id"] for x in r1] == [x["restaurant_id"] for x in r2]

