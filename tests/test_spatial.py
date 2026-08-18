# -*- coding: utf-8 -*-
"""
Phase 8 Test Suite — Location-Aware Proximity Scoring & Spatial Search Optimization
Tests coordinate validation, Haversine exact and vectorized distances,
bounding box filters, BallTree index queries, nearest and radius searches,
locality spatial analytics, and API endpoints.
"""

import math
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from ml.spatial.coordinates import Coordinate, validate_coordinates
from ml.spatial.distance import (
    EARTH_RADIUS_KM,
    haversine_distance,
    haversine_vectorized,
    exponential_decay_score
)
from ml.spatial.bounding_box import BoundingBox, compute_bounding_box, filter_by_bounding_box
from ml.spatial.spatial_index import SpatialBallTreeIndex
from ml.spatial.spatial_search import SpatialSearchEngine
from ml.spatial.cluster_analysis import LocalitySpatialAnalytics
from app.services.recommendation_service import get_spatial_search_engine, get_hybrid_recommender
from app.main import app


# =====================================================================
# 1. COORDINATE VALIDATION & INTEGRITY TESTS
# =====================================================================

def test_valid_coordinate_accepted():
    lat, lon = validate_coordinates(12.9352, 77.6245)
    assert lat == 12.9352
    assert lon == 77.6245

    coord = Coordinate(latitude=12.9784, longitude=77.6408)
    assert coord.latitude == 12.9784
    assert coord.location_source == "Bengaluru locality centroid"
    assert coord.location_precision == "locality-level"


def test_invalid_latitude_rejected():
    with pytest.raises(ValueError, match="Latitude"):
        validate_coordinates(95.0, 77.6245)

    with pytest.raises(ValueError, match="Latitude"):
        validate_coordinates(-90.1, 77.6245)


def test_invalid_longitude_rejected():
    with pytest.raises(ValueError, match="Longitude"):
        validate_coordinates(12.9352, 185.0)

    with pytest.raises(ValueError, match="Longitude"):
        validate_coordinates(12.9352, -180.5)


def test_nan_and_inf_rejected():
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_coordinates(float("nan"), 77.6245)

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_coordinates(12.9352, float("inf"))


def test_none_coordinates_handled_gracefully():
    # Both None -> returns None without crashing
    assert validate_coordinates(None, None) is None

    # Only one None -> raises ValueError
    with pytest.raises(ValueError, match="Both latitude and longitude"):
        validate_coordinates(12.9352, None)


# =====================================================================
# 2. HAVERSINE DISTANCE & SPATIAL DECAY TESTS
# =====================================================================

def test_zero_distance_calculation():
    dist = haversine_distance(12.9352, 77.6245, 12.9352, 77.6245)
    assert dist == 0.0


def test_known_haversine_distance():
    # Koramangala 5th Block (12.9352, 77.6245) to Indiranagar (12.9784, 77.6408)
    dist = haversine_distance(12.9352, 77.6245, 12.9784, 77.6408)
    assert 4.8 <= dist <= 5.4


def test_haversine_symmetry():
    d1 = haversine_distance(12.9352, 77.6245, 12.9784, 77.6408)
    d2 = haversine_distance(12.9784, 77.6408, 12.9352, 77.6245)
    assert np.isclose(d1, d2, atol=1e-4)


def test_haversine_vectorized_consistency():
    u_lat, u_lon = 12.9352, 77.6245
    r_lats = np.array([12.9352, 12.9784, 12.9121])
    r_lons = np.array([77.6245, 77.6408, 77.6446])

    vec_dists = haversine_vectorized(u_lat, u_lon, r_lats, r_lons)
    
    assert vec_dists[0] == 0.0
    assert np.isclose(vec_dists[1], haversine_distance(u_lat, u_lon, 12.9784, 77.6408), atol=1e-3)
    assert np.isclose(vec_dists[2], haversine_distance(u_lat, u_lon, 12.9121, 77.6446), atol=1e-3)


def test_location_score_bounds_and_decay():
    s0 = exponential_decay_score(0.0, tau_km=3.0)
    assert s0 == 1.0

    s3 = exponential_decay_score(3.0, tau_km=3.0)
    assert np.isclose(s3, math.exp(-1.0), atol=1e-3)

    s10 = exponential_decay_score(10.0, tau_km=3.0)
    assert 0.0 < s10 < s3 < s0 <= 1.0


def test_location_score_monotonicity():
    distances = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    scores = [exponential_decay_score(d, tau_km=3.0) for d in distances]
    
    # Strictly descending order
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
        assert 0.0 <= scores[i] <= 1.0


# =====================================================================
# 3. BOUNDING BOX OPTIMIZATION TESTS
# =====================================================================

def test_bounding_box_generation_and_containment():
    lat, lon = 12.9352, 77.6245
    radius_km = 5.0
    bbox = compute_bounding_box(lat, lon, radius_km)

    assert bbox.min_lat < lat < bbox.max_lat
    assert bbox.min_lon < lon < bbox.max_lon
    assert bbox.contains(lat, lon) is True

    # A point 2 km away must be inside
    assert bbox.contains(lat + 0.01, lon + 0.01) is True

    # A point 50 km away must be outside
    assert bbox.contains(lat + 0.5, lon + 0.5) is False


def test_bounding_box_filter_dataframe():
    df = pd.DataFrame([
        {"id": 1, "latitude": 12.9352, "longitude": 77.6245},  # In Koramangala
        {"id": 2, "latitude": 12.9784, "longitude": 77.6408},  # In Indiranagar (~5.1 km)
        {"id": 3, "latitude": 13.5000, "longitude": 77.6245}   # Far north outside
    ])
    bbox = compute_bounding_box(12.9352, 77.6245, radius_km=2.0)
    filtered = filter_by_bounding_box(df, bbox)
    assert len(filtered) == 1
    assert filtered.iloc[0]["id"] == 1


# =====================================================================
# 4. BALLTREE SPATIAL INDEX & SEARCH TESTS
# =====================================================================

def test_spatial_balltree_queries():
    df = pd.DataFrame([
        {"restaurant_id": 101, "latitude": 12.9352, "longitude": 77.6245},
        {"restaurant_id": 102, "latitude": 12.9400, "longitude": 77.6250},
        {"restaurant_id": 103, "latitude": 12.9784, "longitude": 77.6408}
    ])
    index = SpatialBallTreeIndex.from_dataframe(df)

    # Query nearest
    ids, dists = index.query_nearest(12.9352, 77.6245, k=2)
    assert ids[0] == 101
    assert dists[0] == 0.0
    assert ids[1] == 102

    # Query radius
    r_ids, r_dists = index.query_radius(12.9352, 77.6245, radius_km=1.0)
    assert set(r_ids) == {101, 102}


def test_spatial_search_engine_catalog():
    engine = get_spatial_search_engine()
    assert engine is not None
    assert len(engine.df_restaurants) == 12481

    # Search nearest in Koramangala
    nearest = engine.find_nearest(12.9352, 77.6245, top_k=5)
    assert len(nearest) == 5
    assert nearest[0]["distance_km"] <= nearest[1]["distance_km"]


def test_spatial_search_radius_filtering():
    engine = get_spatial_search_engine()
    # Search within 1.0 km radius in Koramangala
    within_1km = engine.search_within_radius(12.9352, 77.6245, radius_km=1.0, top_k=10)
    assert len(within_1km) > 0
    for r in within_1km:
        assert r["distance_km"] <= 1.0


def test_deterministic_spatial_tie_breaking():
    engine = get_spatial_search_engine()
    res1 = engine.find_nearest(12.9352, 77.6245, top_k=5)
    res2 = engine.find_nearest(12.9352, 77.6245, top_k=5)
    assert [r["restaurant_id"] for r in res1] == [r["restaurant_id"] for r in res2]


def test_empty_radius_result_handled():
    engine = get_spatial_search_engine()
    # Query in middle of ocean / far location with tiny radius
    empty_res = engine.search_within_radius(0.0, 0.0, radius_km=0.5)
    assert empty_res == []


# =====================================================================
# 5. LOCALITY SPATIAL ANALYTICS TESTS
# =====================================================================

def test_locality_spatial_analytics():
    engine = get_spatial_search_engine()
    analytics = LocalitySpatialAnalytics(engine.df_restaurants)
    
    summary = analytics.get_locality_summary(min_outlets=10)
    assert not summary.empty
    assert "outlet_count" in summary.columns
    assert "mean_rating" in summary.columns
    assert "centroid_latitude" in summary.columns

    dist_matrix = analytics.get_locality_distance_matrix(top_n_localities=5)
    assert dist_matrix.shape == (5, 5)
    assert np.all(np.diag(dist_matrix.values) == 0.0)

    density = analytics.get_locality_density_metrics()
    assert density["total_outlets"] == 12481
    assert density["precision"] == "locality-level"


# =====================================================================
# 6. HYBRID INTEGRATION & API ENDPOINT TESTS
# =====================================================================

def test_hybrid_integration_with_radius_filter():
    hybrid_engine = get_hybrid_recommender()
    # Search within 2 km radius in Koramangala
    res = hybrid_engine.recommend(
        user_coords=(12.9352, 77.6245),
        filters={"radius_km": 2.0},
        top_k=5
    )
    assert res["count"] > 0
    for r in res["recommendations"]:
        assert r["distance_km"] <= 2.0


def test_api_nearby_endpoint_success():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/nearby?latitude=12.9352&longitude=77.6245&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["location_precision"] == "locality-level"


def test_api_nearby_endpoint_radius_search():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/nearby?latitude=12.9352&longitude=77.6245&radius_km=3.0&top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["radius_km"] == 3.0
    for r in data["recommendations"]:
        assert r["distance_km"] <= 3.0


def test_api_nearby_invalid_coordinates():
    client = TestClient(app)
    # Invalid latitude 95
    res = client.get("/api/v1/recommendations/nearby?latitude=95.0&longitude=77.6245")
    assert res.status_code == 422


def test_api_hybrid_post_with_radius_km():
    client = TestClient(app)
    payload = {
        "latitude": 12.9784,
        "longitude": 77.6408,
        "radius_km": 3.0,
        "preferred_cuisines": ["Cafe", "Continental"],
        "top_k": 5
    }
    response = client.post("/api/v1/recommendations/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 5
    for r in data["recommendations"]:
        assert r["distance_km"] <= 3.0
