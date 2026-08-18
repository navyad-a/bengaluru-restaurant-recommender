# -*- coding: utf-8 -*-
"""
Phase 14 Test Suite: Security, Input Validation & Robustness
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.unit
def test_security_injection_payload_resilience(client):
    """Tests resilience against SQL injection and XSS in text query fields."""
    malicious_payloads = [
        "'; DROP TABLE restaurants; --",
        "' OR 1=1 --",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "{{ 7 * 7 }}"
    ]
    
    for attack_str in malicious_payloads:
        resp = client.post("/api/v1/recommendations/hybrid", json={
            "preferred_area": attack_str,
            "preferred_cuisines": [attack_str],
            "top_k": 5
        })
        # Should gracefully return 200 with empty/filtered recs or handle without 500 error
        assert resp.status_code in [200, 422]
        if resp.status_code == 200:
            assert resp.json()["status"] == "success"


@pytest.mark.unit
def test_security_out_of_bounds_top_k(client):
    """Verifies that top_k < 1 or top_k > 50 triggers 422 validation error."""
    resp_zero = client.get("/api/v1/recommendations/similar/1?top_k=0")
    assert resp_zero.status_code == 422

    resp_oversize = client.get("/api/v1/recommendations/similar/1?top_k=100")
    assert resp_oversize.status_code == 422


@pytest.mark.unit
def test_security_out_of_bounds_coordinates(client):
    """Verifies that latitude not in [-90, 90] or longitude not in [-180, 180] triggers 422."""
    resp_lat = client.get("/api/v1/recommendations/nearby?latitude=105.0&longitude=77.5")
    assert resp_lat.status_code == 422

    resp_lon = client.get("/api/v1/recommendations/nearby?latitude=12.9&longitude=250.0")
    assert resp_lon.status_code == 422


@pytest.mark.unit
def test_security_out_of_bounds_lambda(client):
    """Verifies that mmr_lambda not in [0.0, 1.0] triggers 422."""
    resp_high = client.get("/api/v1/recommendations/hybrid/2?mmr_lambda=1.5")
    assert resp_high.status_code == 422

    resp_neg = client.get("/api/v1/recommendations/hybrid/2?mmr_lambda=-0.2")
    assert resp_neg.status_code == 422


@pytest.mark.unit
def test_security_negative_budget_and_rating(client):
    """Verifies that negative budget or out-of-range rating triggers 422."""
    resp_neg_cost = client.post("/api/v1/recommendations/content", json={
        "max_cost_for_two": -200,
        "top_k": 5
    })
    assert resp_neg_cost.status_code == 422

    resp_high_rating = client.post("/api/v1/recommendations/content", json={
        "min_rating": 6.5,
        "top_k": 5
    })
    assert resp_high_rating.status_code == 422

