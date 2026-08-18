# -*- coding: utf-8 -*-
"""
Phase 13 Test Suite: Streamlit Frontend Client, State Management & UI Component Logic
"""

import pytest
from unittest.mock import patch, MagicMock
from streamlit_app.api_client import RecommendationAPIClient
from streamlit_app.config import (
    API_BASE_URL,
    DEFAULT_TOP_K,
    DEFAULT_MMR_ENABLED,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_SEARCH_RADIUS_KM
)
from streamlit_app.state import init_session_state, reset_filters, clear_recommendation_results


# =============================================================================
# API Client Tests
# =============================================================================

def test_api_client_initialization():
    """Verifies API client base URL normalization and default timeout."""
    client = RecommendationAPIClient(base_url="http://localhost:8000/", timeout_seconds=5.0)
    assert client.base_url == "http://localhost:8000"
    assert client.timeout_seconds == 5.0


@patch("requests.request")
def test_api_client_health_success(mock_req):
    """Verifies successful /health endpoint parsing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "version": "1.0.0"}
    mock_req.return_value = mock_resp

    client = RecommendationAPIClient()
    data, err = client.get_health()
    assert err is None
    assert data["status"] == "healthy"


@patch("requests.request")
def test_api_client_connection_error(mock_req):
    """Verifies graceful handling when FastAPI backend is unreachable."""
    import requests
    mock_req.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = RecommendationAPIClient(base_url="http://invalid-host:9999")
    data, err = client.get_health()
    assert data is None
    assert err["status_code"] == 503
    assert err["error_code"] == "BACKEND_UNREACHABLE"
    assert "Could not connect to FastAPI backend" in err["message"]


@patch("requests.request")
def test_api_client_timeout_error(mock_req):
    """Verifies graceful handling when request times out."""
    import requests
    mock_req.side_effect = requests.exceptions.Timeout("Read timed out")

    client = RecommendationAPIClient(timeout_seconds=2.0)
    data, err = client.get_hybrid_recommendations({"top_k": 5})
    assert data is None
    assert err["status_code"] == 504
    assert err["error_code"] == "REQUEST_TIMEOUT"


@patch("requests.request")
def test_api_client_http_error_envelope_parsing(mock_req):
    """Verifies structured error response extraction from 404/422 responses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.headers = {"X-Request-ID": "req-xyz-789"}
    mock_resp.json.return_value = {
        "status": "error",
        "error_code": "RESOURCE_NOT_FOUND",
        "message": "Restaurant not found in catalog",
        "request_id": "req-xyz-789"
    }
    mock_req.return_value = mock_resp

    client = RecommendationAPIClient()
    data, err = client.get_similar_restaurants(restaurant_id=999999)
    assert data is None
    assert err["status_code"] == 404
    assert err["error_code"] == "RESOURCE_NOT_FOUND"
    assert err["request_id"] == "req-xyz-789"


@patch("requests.request")
def test_api_client_hybrid_recommendations_call(mock_req):
    """Verifies POST /api/v1/recommendations/hybrid request formatting."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "user_id": 2,
        "is_cold_start": False,
        "count": 5,
        "recommendations": [{"name": "Empire Restaurant", "rating": 4.2}]
    }
    mock_req.return_value = mock_resp

    client = RecommendationAPIClient()
    payload = {"user_id": 2, "preferred_area": "Indiranagar", "top_k": 5}
    data, err = client.get_hybrid_recommendations(payload)
    assert err is None
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 1
    mock_req.assert_called_once_with(
        "POST",
        f"{client.base_url}/api/v1/recommendations/hybrid",
        json=payload,
        timeout=client.timeout_seconds
    )


@patch("requests.request")
def test_api_client_onboarding_call(mock_req):
    """Verifies POST /api/v1/recommendations/onboarding request formatting."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "is_cold_start": True,
        "strategy": "onboarding_profile",
        "count": 5,
        "recommendations": []
    }
    mock_req.return_value = mock_resp

    client = RecommendationAPIClient()
    payload = {
        "favorite_cuisines": ["Biryani"],
        "price_tier": "Moderate",
        "top_k": 5
    }
    data, err = client.get_onboarding_recommendations(payload)
    assert err is None
    assert data["strategy"] == "onboarding_profile"


@patch("requests.request")
def test_api_client_nearby_request(mock_req):
    """Verifies GET /api/v1/recommendations/nearby coordinate query parameters."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "radius_km": 3.0,
        "count": 10,
        "recommendations": []
    }
    mock_req.return_value = mock_resp

    client = RecommendationAPIClient()
    data, err = client.get_nearby_restaurants(latitude=12.9716, longitude=77.5946, radius_km=3.0, top_k=10)
    assert err is None
    assert data["status"] == "success"
    mock_req.assert_called_once_with(
        "GET",
        f"{client.base_url}/api/v1/recommendations/nearby",
        params={"latitude": 12.9716, "longitude": 77.5946, "radius_km": 3.0, "top_k": 10},
        timeout=client.timeout_seconds
    )


@patch("requests.request")
def test_api_client_popular_request(mock_req):
    """Verifies GET /api/v1/recommendations/popular parameter filtering."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "success", "scope": "locality: Indiranagar", "count": 5, "recommendations": []}
    mock_req.return_value = mock_resp

    client = RecommendationAPIClient()
    data, err = client.get_popular_restaurants(area="Indiranagar", top_k=5)
    assert err is None
    assert data["scope"] == "locality: Indiranagar"


@patch("requests.request")
def test_api_client_cache_clear_request(mock_req):
    """Verifies POST /api/v1/system/cache/clear endpoint call."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "success", "message": "Recommendation cache cleared successfully."}
    mock_req.return_value = mock_resp

    client = RecommendationAPIClient()
    data, err = client.clear_cache()
    assert err is None
    assert data["status"] == "success"


# =============================================================================
# Session State Tests
# =============================================================================

def test_session_state_defaults():
    """Verifies init_session_state sets proper default values."""
    import streamlit as st
    init_session_state()
    assert st.session_state["top_k"] == DEFAULT_TOP_K
    assert st.session_state["mmr_enabled"] == DEFAULT_MMR_ENABLED
    assert st.session_state["mmr_lambda"] == DEFAULT_MMR_LAMBDA
    assert st.session_state["selected_area"] == "All Localities"


def test_session_state_reset_filters():
    """Verifies reset_filters restores initial empty filter parameters."""
    import streamlit as st
    st.session_state["selected_cuisines"] = ["Italian", "Continental"]
    st.session_state["selected_area"] = "Koramangala 5th Block"
    st.session_state["max_budget_for_two"] = 3000
    
    reset_filters()
    assert st.session_state["selected_cuisines"] == []
    assert st.session_state["selected_area"] == "All Localities"
    assert st.session_state["max_budget_for_two"] == 1000


def test_session_state_clear_recommendations():
    """Verifies clear_recommendation_results cleans response state."""
    import streamlit as st
    st.session_state["last_recommendation_response"] = {"recommendations": [{"name": "Test"}]}
    st.session_state["last_query_type"] = "hybrid"
    
    clear_recommendation_results()
    assert st.session_state["last_recommendation_response"] is None
    assert st.session_state["last_query_type"] is None

