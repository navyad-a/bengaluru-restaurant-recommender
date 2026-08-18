# -*- coding: utf-8 -*-
"""
Phase 7 Test Suite — Hybrid Recommendation Engine
Tests multi-signal normalization, Bayesian quality shrinkage, spatial proximity,
dynamic weight redistribution, hard constraints, cold-start handling, deterministic ranking,
and API endpoints.
"""

import os
import math
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from ml.hybrid.quality import BayesianQualityScorer
from ml.hybrid.location import LocationScorer, haversine_distance
from ml.hybrid.scoring import (
    DEFAULT_HYBRID_WEIGHTS,
    normalize_content_score,
    normalize_collaborative_score,
    compute_effective_weights,
    compute_hybrid_score
)
from ml.hybrid.candidate_generator import CandidateGenerator
from ml.hybrid.hybrid_recommender import HybridRecommender
from app.services.recommendation_service import get_hybrid_recommender
from app.main import app


# =====================================================================
# 1. SCORING & NORMALIZATION UNIT TESTS
# =====================================================================

def test_content_score_normalization():
    assert normalize_content_score(0.85) == 0.85
    assert normalize_content_score(1.5) == 1.0
    assert normalize_content_score(-0.2) == 0.0


def test_collaborative_score_normalization():
    # Maps [1.0, 5.0] -> [0.0, 1.0]
    assert normalize_collaborative_score(5.0) == 1.0
    assert normalize_collaborative_score(1.0) == 0.0
    assert normalize_collaborative_score(3.0) == 0.5
    assert normalize_collaborative_score(5.5) == 1.0
    assert normalize_collaborative_score(0.5) == 0.0


def test_effective_weight_sum_normalization():
    # All 4 signals available
    weights = compute_effective_weights(DEFAULT_HYBRID_WEIGHTS, {"content", "collaborative", "location", "quality"})
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights["content"] == 0.40
    assert weights["collaborative"] == 0.20
    assert weights["location"] == 0.15
    assert weights["quality"] == 0.25


def test_dynamic_weight_redistribution_missing_signal():
    # Unknown user (no collaborative) and no location
    available = {"content", "quality"}
    weights = compute_effective_weights(DEFAULT_HYBRID_WEIGHTS, available)
    assert weights["collaborative"] == 0.0
    assert weights["location"] == 0.0
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights["content"] > DEFAULT_HYBRID_WEIGHTS["content"]
    assert weights["quality"] > DEFAULT_HYBRID_WEIGHTS["quality"]


def test_hybrid_score_calculation():
    scores = {"content": 0.8, "collaborative": 0.6, "location": 0.9, "quality": 0.7}
    weights = {"content": 0.4, "collaborative": 0.2, "location": 0.15, "quality": 0.25}
    expected = 0.4*0.8 + 0.2*0.6 + 0.15*0.9 + 0.25*0.7
    h_score = compute_hybrid_score(scores, weights)
    assert np.isclose(h_score, expected, atol=1e-4)


# =====================================================================
# 2. BAYESIAN QUALITY SHRINKAGE TESTS
# =====================================================================

def test_bayesian_quality_scorer_basic():
    scorer = BayesianQualityScorer(global_mean=3.6, min_votes_threshold=50.0)
    # High votes -> close to raw rating
    wr_high_votes = scorer.calculate_weighted_rating(rating=4.8, review_count=5000)
    assert np.isclose(wr_high_votes, 4.8, atol=0.05)


def test_bayesian_shrinkage_favors_reliable_ratings():
    scorer = BayesianQualityScorer(global_mean=3.6, min_votes_threshold=50.0)
    # Case A: 5.0 rating with only 2 reviews (unreliable)
    score_unreliable = scorer.score(rating=5.0, review_count=2)
    # Case B: 4.5 rating with 1000 reviews (proven quality)
    score_reliable = scorer.score(rating=4.5, review_count=1000)
    
    # Proven 4.5 should beat unreliable 5.0 with 2 reviews!
    assert score_reliable > score_unreliable


def test_bayesian_quality_missing_rating_fallback():
    scorer = BayesianQualityScorer(global_mean=3.6, min_votes_threshold=50.0)
    wr_nan = scorer.calculate_weighted_rating(rating=None, review_count=0)
    assert wr_nan == 3.6
    score_nan = scorer.score(rating=None, review_count=0)
    assert np.isclose(score_nan, (3.6 - 1.0) / 4.0)


# =====================================================================
# 3. LOCATION & HAVERSINE PROXIMITY TESTS
# =====================================================================

def test_haversine_distance_known_coordinates():
    # Koramangala (12.9352, 77.6245) to Indiranagar (12.9784, 77.6408) is ~5.1 km
    dist = haversine_distance(12.9352, 77.6245, 12.9784, 77.6408)
    assert 4.5 <= dist <= 5.5


def test_location_scorer_exponential_decay():
    scorer = LocationScorer(decay_tau_km=3.0)
    s_0km = scorer.score_distance(0.0)
    assert s_0km == 1.0

    s_3km = scorer.score_distance(3.0)
    assert np.isclose(s_3km, math.exp(-1.0), atol=1e-3)

    s_10km = scorer.score_distance(10.0)
    assert s_0km > s_3km > s_10km >= 0.0


# =====================================================================
# 4. HARD CONSTRAINTS & CANDIDATE PRUNING TESTS
# =====================================================================

@pytest.fixture
def mock_catalog():
    return pd.DataFrame([
        {"restaurant_id": 1, "name": "MTR", "area": "Basavanagudi", "cuisines": "South Indian", "rest_type": "Quick Bites", "price_tier": "Budget", "cost_for_two_inr": 250, "rating": 4.5, "review_count": 5000, "online_order": True, "book_table": False, "latitude": 12.9416, "longitude": 77.5753, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 2, "name": "Toit", "area": "Indiranagar", "cuisines": "Italian, American", "rest_type": "Microbrewery", "price_tier": "Premium", "cost_for_two_inr": 1500, "rating": 4.7, "review_count": 14000, "online_order": True, "book_table": True, "latitude": 12.9784, "longitude": 77.6408, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 3, "name": "Empire", "area": "Koramangala", "cuisines": "North Indian, Biryani", "rest_type": "Casual Dining", "price_tier": "Moderate", "cost_for_two_inr": 700, "rating": 4.1, "review_count": 6000, "online_order": True, "book_table": False, "latitude": 12.9352, "longitude": 77.6245, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 4, "name": "Budget Biryani", "area": "BTM", "cuisines": "Biryani", "rest_type": "Quick Bites", "price_tier": "Budget", "cost_for_two_inr": 300, "rating": 3.8, "review_count": 200, "online_order": False, "book_table": False, "latitude": 12.9166, "longitude": 77.6101, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 5, "name": "Luxury Dine", "area": "Lavelle Road", "cuisines": "Continental", "rest_type": "Fine Dining", "price_tier": "Luxury", "cost_for_two_inr": 3500, "rating": 4.8, "review_count": 1200, "online_order": False, "book_table": True, "latitude": 12.9719, "longitude": 77.5956, "location_source": "centroid", "location_precision": "locality-level"}
    ])


def test_hard_max_cost_filtering(mock_catalog):
    gen = CandidateGenerator(mock_catalog)
    filtered = gen.apply_hard_filters(mock_catalog, {"max_cost_for_two": 500})
    assert len(filtered) == 2
    assert set(filtered["restaurant_id"]) == {1, 4}


def test_hard_min_rating_filtering(mock_catalog):
    gen = CandidateGenerator(mock_catalog)
    filtered = gen.apply_hard_filters(mock_catalog, {"min_rating": 4.5})
    assert len(filtered) == 3
    assert set(filtered["restaurant_id"]) == {1, 2, 5}


def test_hard_area_filtering(mock_catalog):
    gen = CandidateGenerator(mock_catalog)
    filtered = gen.apply_hard_filters(mock_catalog, {"area": "Indiranagar"})
    assert len(filtered) == 1
    assert filtered.iloc[0]["name"] == "Toit"


def test_hard_online_order_filtering(mock_catalog):
    gen = CandidateGenerator(mock_catalog)
    filtered = gen.apply_hard_filters(mock_catalog, {"online_order_only": True})
    assert len(filtered) == 3
    assert all(filtered["online_order"] == True)


def test_hard_book_table_filtering(mock_catalog):
    gen = CandidateGenerator(mock_catalog)
    filtered = gen.apply_hard_filters(mock_catalog, {"book_table_only": True})
    assert len(filtered) == 2
    assert all(filtered["book_table"] == True)


# =====================================================================
# 5. HYBRID RECOMMENDER ORCHESTRATION & INTEGRATION TESTS
# =====================================================================

def test_hybrid_recommender_singleton_loaded():
    hybrid_engine = get_hybrid_recommender()
    assert hybrid_engine is not None
    assert hybrid_engine.content_recommender is not None
    assert hybrid_engine.location_scorer is not None
    assert hybrid_engine.quality_scorer is not None
    assert len(hybrid_engine.df_restaurants) == 12481


def test_known_user_hybrid_recommendation():
    hybrid_engine = get_hybrid_recommender()
    res = hybrid_engine.recommend(
        user_id=1,
        user_coords=(12.9352, 77.6245),
        top_k=5
    )
    assert res["status"] if "status" in res else True
    assert res["user_id"] == 1
    assert res["is_cold_start"] is False
    assert len(res["recommendations"]) == 5
    
    # Check score bounds and fields
    for r in res["recommendations"]:
        assert 0.0 <= r["hybrid_score"] <= 1.0
        assert 0.0 <= r["content_score"] <= 1.0
        assert 0.0 <= r["collaborative_score"] <= 1.0
        assert 0.0 <= r["location_score"] <= 1.0
        assert 0.0 <= r["quality_score"] <= 1.0
        assert r["distance_km"] is not None
        assert "explanation" in r
        assert len(r["explanation"]) > 5


def test_unknown_user_cold_start_recommendation():
    hybrid_engine = get_hybrid_recommender()
    res = hybrid_engine.recommend(
        user_id=999999,  # Unseen user
        preferences={"preferred_cuisines": ["South Indian"], "preferred_price_tier": "Budget"},
        top_k=5
    )
    assert res["is_cold_start"] is True
    assert res["effective_weights"]["collaborative"] == 0.0
    assert np.isclose(sum(res["effective_weights"].values()), 1.0)
    assert len(res["recommendations"]) == 5


def test_already_rated_restaurant_exclusion_for_known_user():
    hybrid_engine = get_hybrid_recommender()
    if hybrid_engine.collaborative_recommender:
        rated_items = hybrid_engine.collaborative_recommender.user_rated_items.get(1, set())
        res = hybrid_engine.recommend(user_id=1, top_k=10)
        rec_ids = [r["restaurant_id"] for r in res["recommendations"]]
        # No rated restaurant should be present
        assert set(rec_ids).isdisjoint(rated_items)


def test_deterministic_ranking_and_top_k():
    hybrid_engine = get_hybrid_recommender()
    
    # 1. Deterministic repeatability
    res1 = hybrid_engine.recommend(user_id=1, top_k=5)
    res2 = hybrid_engine.recommend(user_id=1, top_k=5)
    
    ids1 = [r["restaurant_id"] for r in res1["recommendations"]]
    ids2 = [r["restaurant_id"] for r in res2["recommendations"]]
    assert ids1 == ids2

    # 2. Raw hybrid scores must be strictly descending when mmr_enabled=False
    res_raw = hybrid_engine.recommend(user_id=1, top_k=5, mmr_enabled=False)
    raw_scores = [r["hybrid_score"] for r in res_raw["recommendations"]]
    assert raw_scores == sorted(raw_scores, reverse=True)

    # 3. MMR scores must be strictly descending when mmr_enabled=True
    mmr_scores = [r["mmr_score"] for r in res1["recommendations"]]
    assert mmr_scores == sorted(mmr_scores, reverse=True)


def test_no_duplicate_restaurants_in_output():
    hybrid_engine = get_hybrid_recommender()
    res = hybrid_engine.recommend(
        user_id=1,
        preferences={"preferred_cuisines": ["North Indian", "Biryani"]},
        user_coords=(12.9784, 77.6408),
        top_k=20
    )
    rec_ids = [r["restaurant_id"] for r in res["recommendations"]]
    assert len(rec_ids) == len(set(rec_ids)), "Duplicate restaurant IDs detected in recommendations!"


def test_missing_location_handling():
    hybrid_engine = get_hybrid_recommender()
    res = hybrid_engine.recommend(user_id=1, user_coords=None, top_k=5)
    assert res["effective_weights"]["location"] == 0.0
    assert np.isclose(sum(res["effective_weights"].values()), 1.0)
    for r in res["recommendations"]:
        assert r["distance_km"] is None
        assert r["location_score"] == 0.0


# =====================================================================
# 6. FASTAPI ENDPOINT TESTS
# =====================================================================

def test_api_hybrid_get_known_user():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/hybrid/1?top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["user_id"] == 1
    assert data["is_cold_start"] is False
    assert len(data["recommendations"]) == 5


def test_api_hybrid_get_cold_start_user_no_500():
    client = TestClient(app)
    # Cold-start unknown user should return 200 with fallback rather than HTTP 500
    response = client.get("/api/v1/recommendations/hybrid/999999?top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_cold_start"] is True
    assert len(data["recommendations"]) > 0


def test_api_hybrid_post_full_request():
    client = TestClient(app)
    payload = {
        "user_id": 1,
        "preferred_cuisines": ["Biryani", "Mughlai"],
        "preferred_price_tier": "Moderate",
        "latitude": 12.9352,
        "longitude": 77.6245,
        "max_cost_for_two": 800,
        "top_k": 5
    }
    response = client.post("/api/v1/recommendations/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 5
    for item in data["recommendations"]:
        assert item["cost_for_two_inr"] <= 800


def test_api_hybrid_validation_bounds():
    client = TestClient(app)
    # Invalid top_k > 50
    res_bad_top_k = client.get("/api/v1/recommendations/hybrid/1?top_k=100")
    assert res_bad_top_k.status_code == 422

    # Invalid latitude
    res_bad_lat = client.get("/api/v1/recommendations/hybrid/1?latitude=150.0&longitude=77.0")
    assert res_bad_lat.status_code == 422
