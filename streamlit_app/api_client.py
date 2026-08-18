# -*- coding: utf-8 -*-
"""
FastAPI HTTP Client Layer for Streamlit
Delegates all recommendation scoring and telemetry to backend REST endpoints.
"""

import requests
from typing import Dict, Any, Optional, Tuple, List
from streamlit_app.config import API_BASE_URL


class RecommendationAPIClient:
    """
    Resilient REST client interacting with the FastAPI Recommendation Backend.
    """

    def __init__(self, base_url: str = API_BASE_URL, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _handle_response(self, response: requests.Response) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Parses response and extracts standardized JSON or structured error envelopes."""
        try:
            data = response.json()
        except Exception:
            data = {"message": response.text}

        if response.status_code >= 200 and response.status_code < 300:
            return data, None

        error_payload = {
            "status_code": response.status_code,
            "error_code": data.get("error_code", f"HTTP_{response.status_code}"),
            "message": data.get("message") or data.get("detail") or "An unexpected server error occurred.",
            "detail": data.get("detail"),
            "request_id": data.get("request_id", response.headers.get("X-Request-ID", "N/A"))
        }
        return None, error_payload

    def _safe_request(self, method: str, path: str, **kwargs) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Wrapper for requests with timeout and network exception capture."""
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout_seconds)
        try:
            res = requests.request(method, url, **kwargs)
            return self._handle_response(res)
        except requests.exceptions.ConnectionError:
            return None, {
                "status_code": 503,
                "error_code": "BACKEND_UNREACHABLE",
                "message": f"Could not connect to FastAPI backend at {self.base_url}. Ensure the backend service is running.",
                "request_id": "N/A"
            }
        except requests.exceptions.Timeout:
            return None, {
                "status_code": 504,
                "error_code": "REQUEST_TIMEOUT",
                "message": f"Recommendation request timed out after {self.timeout_seconds} seconds.",
                "request_id": "N/A"
            }
        except Exception as exc:
            return None, {
                "status_code": 500,
                "error_code": "CLIENT_ERROR",
                "message": f"Client request error: {str(exc)}",
                "request_id": "N/A"
            }

    # Health & System Status
    def get_health(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """GET /health - Liveness probe."""
        return self._safe_request("GET", "/health")

    def get_ready(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """GET /ready - Readiness probe checking all models."""
        return self._safe_request("GET", "/ready")

    def get_system_status(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """GET /api/v1/system/status - Runtime telemetry and cache metrics."""
        return self._safe_request("GET", "/api/v1/system/status")

    def clear_cache(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """POST /api/v1/system/cache/clear - Purge backend recommendation cache."""
        return self._safe_request("POST", "/api/v1/system/cache/clear")

    # Recommendations
    def get_hybrid_recommendations(self, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """POST /api/v1/recommendations/hybrid - Flexible hybrid recommendation."""
        return self._safe_request("POST", "/api/v1/recommendations/hybrid", json=payload)

    def get_onboarding_recommendations(self, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """POST /api/v1/recommendations/onboarding - Cold-start questionnaire bootstrapper."""
        return self._safe_request("POST", "/api/v1/recommendations/onboarding", json=payload)

    def get_nearby_restaurants(
        self,
        latitude: float,
        longitude: float,
        radius_km: Optional[float] = None,
        top_k: int = 10
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """GET /api/v1/recommendations/nearby - Spatial BallTree radius search."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "top_k": top_k
        }
        if radius_km is not None:
            params["radius_km"] = radius_km
        return self._safe_request("GET", "/api/v1/recommendations/nearby", params=params)

    def get_popular_restaurants(
        self,
        area: Optional[str] = None,
        cuisine: Optional[str] = None,
        max_cost_for_two: Optional[int] = None,
        min_rating: Optional[float] = None,
        online_order_only: bool = False,
        book_table_only: bool = False,
        top_k: int = 10
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """GET /api/v1/recommendations/popular - Bayesian popularity rankings."""
        params: Dict[str, Any] = {
            "top_k": top_k,
            "online_order_only": online_order_only,
            "book_table_only": book_table_only
        }
        if area and area != "All Localities":
            params["area"] = area
        if cuisine:
            params["cuisine"] = cuisine
        if max_cost_for_two is not None and max_cost_for_two > 0:
            params["max_cost_for_two"] = max_cost_for_two
        if min_rating is not None and min_rating > 1.0:
            params["min_rating"] = min_rating
        return self._safe_request("GET", "/api/v1/recommendations/popular", params=params)

    def get_similar_restaurants(
        self,
        restaurant_id: int,
        top_k: int = 10
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """GET /api/v1/recommendations/similar/{restaurant_id} - Content similarity."""
        return self._safe_request("GET", f"/api/v1/recommendations/similar/{restaurant_id}", params={"top_k": top_k})


# Global singleton client instance
api_client = RecommendationAPIClient()

