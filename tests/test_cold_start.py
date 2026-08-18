# -*- coding: utf-8 -*-
"""
Phase 9 Test Suite — Cold-Start Strategy & Fallback Intelligence
Tests strategy routing, Bayesian popularity engine, item cold-start imputation,
onboarding questionnaire handling, explainability generation, and API endpoints.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from ml.cold_start.routing import ColdStartRouter, ColdStartStrategy
from ml.cold_start.popularity import BayesianPopularityEngine
from ml.cold_start.onboarding import OnboardingQuestionnaire, OnboardingPreferenceHandler
from ml.cold_start.item_cold_start import ItemColdStartHandler
from ml.cold_start.explainer import ColdStartExplainer
from app.services.recommendation_service import get_hybrid_recommender
from app.main import app


# =====================================================================
# 1. COLD-START ROUTER & STRATEGY TAXONOMY TESTS
# =====================================================================

def test_router_warm_hybrid_classification():
    strat = ColdStartRouter.determine_strategy(
        user_id=1,
        user_interaction_count=15,
        has_preferences=True,
        has_location=True,
        is_known_collaborative_user=True
    )
    assert strat == ColdStartStrategy.WARM_HYBRID


def test_router_sparse_hybrid_classification():
    strat = ColdStartRouter.determine_strategy(
        user_id=1,
        user_interaction_count=2,  # < 5 ratings -> sparse
        has_preferences=False,
        has_location=False,
        is_known_collaborative_user=True
    )
    assert strat == ColdStartStrategy.SPARSE_HYBRID


def test_router_profile_content_classification():
    strat = ColdStartRouter.determine_strategy(
        user_id=999999,  # Unknown user
        user_interaction_count=0,
        has_preferences=True,
        has_location=True,
        is_known_collaborative_user=False
    )
    assert strat == ColdStartStrategy.PROFILE_CONTENT_QUALITY


def test_router_location_popularity_classification():
    strat = ColdStartRouter.determine_strategy(
        user_id=None,
        user_interaction_count=0,
        has_preferences=False,
        has_location=True,
        is_known_collaborative_user=False
    )
    assert strat == ColdStartStrategy.LOCATION_POPULARITY


def test_router_global_popularity_classification():
    strat = ColdStartRouter.determine_strategy(
        user_id=None,
        user_interaction_count=0,
        has_preferences=False,
        has_location=False,
        is_known_collaborative_user=False
    )
    assert strat == ColdStartStrategy.GLOBAL_POPULARITY


def test_router_strategy_weights_sum_to_one():
    for strat in ColdStartStrategy:
        for has_loc in [True, False]:
            weights = ColdStartRouter.get_strategy_weights(strat, has_location=has_loc)
            assert np.isclose(sum(weights.values()), 1.0, atol=1e-3)


# =====================================================================
# 2. BAYESIAN POPULARITY ENGINE TESTS
# =====================================================================

def test_popularity_engine_global_and_locality():
    hybrid_engine = get_hybrid_recommender()
    pop_engine = hybrid_engine.popularity_engine
    
    # Global popular
    global_recs = pop_engine.get_global_popular(top_k=5)
    assert len(global_recs) == 5
    assert global_recs[0]["popularity_score"] >= global_recs[1]["popularity_score"]
    
    # Locality popular
    loc_recs = pop_engine.get_locality_popular(area="Indiranagar", top_k=5)
    assert len(loc_recs) == 5
    assert all(r["area"].lower() == "indiranagar" for r in loc_recs)


def test_popularity_engine_cuisine_and_filters():
    hybrid_engine = get_hybrid_recommender()
    pop_engine = hybrid_engine.popularity_engine
    
    # Cuisine popular with max cost filter
    c_recs = pop_engine.get_cuisine_popular(
        cuisine="Biryani",
        top_k=5,
        filters={"max_cost_for_two": 600}
    )
    assert len(c_recs) == 5
    assert all("biryani" in r["cuisines"].lower() for r in c_recs)
    assert all(r["cost_for_two_inr"] <= 600 for r in c_recs)


# =====================================================================
# 3. ITEM COLD-START & UNRATED IMPUTATION TESTS
# =====================================================================

def test_item_cold_start_unrated_detection():
    df = pd.DataFrame([
        {"restaurant_id": 1, "name": "Famous Place", "area": "Indiranagar", "rating": 4.5, "review_count": 1000},
        {"restaurant_id": 2, "name": "New Outlet", "area": "Indiranagar", "rating": np.nan, "review_count": 0}
    ])
    handler = ItemColdStartHandler(df)
    
    assert handler.is_cold_start_restaurant(1) is False
    assert handler.is_cold_start_restaurant(2) is True


def test_item_cold_start_imputation_and_enrichment():
    df = pd.DataFrame([
        {"restaurant_id": 1, "name": "MTR", "area": "Basavanagudi", "rating": 4.6, "review_count": 5000},
        {"restaurant_id": 2, "name": "New Idli Shop", "area": "Basavanagudi", "rating": np.nan, "review_count": 0}
    ])
    handler = ItemColdStartHandler(df)
    
    enriched1 = handler.enrich_item_metadata(df.iloc[0].to_dict())
    assert enriched1["is_unrated"] is False
    assert enriched1["imputed_rating_prior"] is None

    enriched2 = handler.enrich_item_metadata(df.iloc[1].to_dict())
    assert enriched2["is_unrated"] is True
    assert enriched2["imputed_rating_prior"] == 4.6  # Inherits Basavanagudi locality prior!


# =====================================================================
# 4. ONBOARDING QUESTIONNAIRE & EXPLAINABILITY TESTS
# =====================================================================

def test_onboarding_questionnaire_payload():
    q = OnboardingQuestionnaire(
        favorite_cuisines=["South Indian", "Karnataka"],
        preferred_dining_types=["Quick Bites"],
        preferred_area="Jayanagar",
        price_tier="Budget",
        max_budget_for_two=300,
        is_pure_veg_preferred=True,
        online_ordering=True
    )
    prefs = OnboardingPreferenceHandler.build_preference_payload(q)
    assert "South Indian" in prefs["preferred_cuisines"]
    assert "Pure Vegetarian" in prefs["preferred_cuisines"]
    assert prefs["preferred_area"] == "Jayanagar"
    assert prefs["max_cost_for_two"] == 300
    assert prefs["online_order_only"] is True


def test_cold_start_explainer_strategies():
    item = {
        "name": "MTR",
        "cuisines": "South Indian, Karnataka",
        "rating": 4.5,
        "review_count": 5000,
        "area": "Basavanagudi",
        "distance_km": 1.2
    }
    
    # Warm hybrid explanation
    exp_warm = ColdStartExplainer.generate_explanation(ColdStartStrategy.WARM_HYBRID, item)
    assert "taste alignment" in exp_warm

    # Location popularity explanation
    exp_loc = ColdStartExplainer.generate_explanation(ColdStartStrategy.LOCATION_POPULARITY, item, area_requested="Basavanagudi")
    assert "popular choice in Basavanagudi" in exp_loc

    # Global popularity explanation
    exp_glob = ColdStartExplainer.generate_explanation(ColdStartStrategy.GLOBAL_POPULARITY, item)
    assert "popular dining institution in Bengaluru" in exp_glob


# =====================================================================
# 5. HYBRID RECOMMENDER ORCHESTRATION & API ENDPOINTS
# =====================================================================

def test_hybrid_recommender_strategy_reporting():
    hybrid_engine = get_hybrid_recommender()
    
    # 1. Warm user
    res_warm = hybrid_engine.recommend(user_id=1, top_k=3)
    assert res_warm["strategy"] == ColdStartStrategy.WARM_HYBRID.value
    assert res_warm["is_cold_start"] is False

    # 2. Unknown user with preferences
    res_prof = hybrid_engine.recommend(
        preferences={"preferred_cuisines": ["South Indian"]},
        top_k=3
    )
    assert res_prof["strategy"] == ColdStartStrategy.PROFILE_CONTENT_QUALITY.value
    assert res_prof["is_cold_start"] is True


def test_api_popular_endpoint():
    client = TestClient(app)
    
    # Global popular
    res_g = client.get("/api/v1/recommendations/popular?top_k=5")
    assert res_g.status_code == 200
    data_g = res_g.json()
    assert data_g["status"] == "success"
    assert len(data_g["recommendations"]) == 5

    # Locality popular
    res_loc = client.get("/api/v1/recommendations/popular?area=Koramangala&top_k=3")
    assert res_loc.status_code == 200
    data_loc = res_loc.json()
    assert len(data_loc["recommendations"]) == 3


def test_api_onboarding_endpoint():
    client = TestClient(app)
    payload = {
        "favorite_cuisines": ["Biryani", "North Indian"],
        "preferred_dining_types": ["Casual Dining"],
        "preferred_area": "Koramangala 5th Block",
        "price_tier": "Moderate",
        "max_budget_for_two": 800,
        "online_ordering": True,
        "top_k": 5
    }
    response = client.post("/api/v1/recommendations/onboarding", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["is_cold_start"] is True
    assert data["strategy"] == "profile_content"
    assert len(data["recommendations"]) == 5
    for r in data["recommendations"]:
        assert r["cost_for_two_inr"] <= 800
