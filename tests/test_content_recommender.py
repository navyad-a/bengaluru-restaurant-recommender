# -*- coding: utf-8 -*-
"""
Phase 5 Test Suite — Content-Based Recommendation Engine
Comprehensive unit and integration tests covering tokenization, TF-IDF vectorization,
cosine similarity, hard constraints, artifact serialization, and API endpoints.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from ml.content_based.content_features import (
    clean_token,
    get_rating_bucket,
    get_cost_bucket,
    build_restaurant_feature_document,
    build_preference_feature_document
)
from ml.content_based.tfidf_engine import TfidfEngine
from ml.content_based.similarity import compute_cosine_similarity_vector, apply_hard_filters
from ml.content_based.content_recommender import ContentRecommender
from app.main import app


@pytest.fixture
def sample_catalog_df():
    """Small controlled fixture for deterministic unit testing."""
    return pd.DataFrame([
        {
            "restaurant_id": 1,
            "name": "MTR Lalbagh",
            "city": "Bengaluru",
            "area": "Basavanagudi",
            "address": "Lalbagh Road",
            "cuisines": "South Indian, Karnataka",
            "rest_type": "Quick Bites",
            "cost_for_two_inr": 250,
            "price_tier": "Budget",
            "rating": 4.5,
            "review_count": 5000,
            "online_order": True,
            "book_table": False,
            "dish_liked": "Masala Dosa, Filter Coffee",
            "latitude": 12.9416,
            "longitude": 77.5753,
            "location_source": "Bengaluru locality centroid",
            "location_precision": "locality-level"
        },
        {
            "restaurant_id": 2,
            "name": "Vidyarthi Bhavan",
            "city": "Bengaluru",
            "area": "Basavanagudi",
            "address": "Gandhi Bazaar",
            "cuisines": "South Indian",
            "rest_type": "Quick Bites",
            "cost_for_two_inr": 150,
            "price_tier": "Budget",
            "rating": 4.4,
            "review_count": 4500,
            "online_order": False,
            "book_table": False,
            "dish_liked": "Crispy Dosa, Filter Coffee",
            "latitude": 12.9416,
            "longitude": 77.5753,
            "location_source": "Bengaluru locality centroid",
            "location_precision": "locality-level"
        },
        {
            "restaurant_id": 3,
            "name": "Toit Pub",
            "city": "Bengaluru",
            "area": "Indiranagar",
            "address": "100ft Road",
            "cuisines": "Italian, American, Pizza",
            "rest_type": "Microbrewery",
            "cost_for_two_inr": 1500,
            "price_tier": "Premium",
            "rating": 4.7,
            "review_count": 14000,
            "online_order": True,
            "book_table": True,
            "dish_liked": "Craft Beer, Pizza",
            "latitude": 12.9784,
            "longitude": 77.6408,
            "location_source": "Bengaluru locality centroid",
            "location_precision": "locality-level"
        },
        {
            "restaurant_id": 4,
            "name": "Windmills Craftworks",
            "city": "Bengaluru",
            "area": "Whitefield",
            "address": "EPIP Zone",
            "cuisines": "American, Continental, Italian",
            "rest_type": "Microbrewery",
            "cost_for_two_inr": 2500,
            "price_tier": "Luxury",
            "rating": 4.6,
            "review_count": 8000,
            "online_order": False,
            "book_table": True,
            "dish_liked": "Craft Beer, Steak",
            "latitude": 12.9698,
            "longitude": 77.7500,
            "location_source": "Bengaluru locality centroid",
            "location_precision": "locality-level"
        },
        {
            "restaurant_id": 5,
            "name": "Meghana Foods",
            "city": "Bengaluru",
            "area": "Koramangala 5th Block",
            "address": "Koramangala",
            "cuisines": "Biryani, Andhra, North Indian",
            "rest_type": "Casual Dining",
            "cost_for_two_inr": 600,
            "price_tier": "Moderate",
            "rating": 4.4,
            "review_count": 9000,
            "online_order": True,
            "book_table": False,
            "dish_liked": "Special Chicken Biryani, Paneer Biryani",
            "latitude": 12.9352,
            "longitude": 77.6180,
            "location_source": "Bengaluru locality centroid",
            "location_precision": "locality-level"
        }
    ])


# =====================================================================
# 1. FEATURE & TOKENIZATION TESTS
# =====================================================================

def test_clean_token():
    assert clean_token("South Indian") == "south_indian"
    assert clean_token("Cafe / Quick Bites") == "cafe_quick_bites"
    assert clean_token("  Indiranagar 5th Block  ") == "indiranagar_5th_block"
    assert clean_token("") == ""
    assert clean_token(None) == ""


def test_rating_and_cost_buckets():
    assert get_rating_bucket(4.5) == "rating_exceptional"
    assert get_rating_bucket(3.9) == "rating_high"
    assert get_rating_bucket(3.5) == "rating_medium"
    assert get_rating_bucket(2.8) == "rating_low"
    assert get_rating_bucket(None) == "rating_unrated"

    assert get_cost_bucket(200) == "cost_under_300"
    assert get_cost_bucket(500) == "cost_300_to_600"
    assert get_cost_bucket(800) == "cost_600_to_1000"
    assert get_cost_bucket(1500) == "cost_1000_to_1800"
    assert get_cost_bucket(2500) == "cost_above_1800"
    assert get_cost_bucket(None) == "cost_bracket_moderate"


def test_restaurant_feature_document_prefixes_and_weights():
    row = {
        "cuisines": "South Indian, Karnataka",
        "rest_type": "Quick Bites",
        "area": "Jayanagar",
        "price_tier": "Budget",
        "rating": 4.5,
        "cost_for_two_inr": 200,
        "online_order": True,
        "book_table": False,
        "dish_liked": "Dosa, Vada"
    }
    doc = build_restaurant_feature_document(row)
    
    # Check prefixes
    assert "cuisine_south_indian" in doc
    assert "cuisine_karnataka" in doc
    assert "type_quick_bites" in doc
    assert "area_jayanagar" in doc
    assert "price_budget" in doc
    assert "rating_exceptional" in doc
    assert "cost_under_300" in doc
    assert "online_order_yes" in doc
    assert "book_table_no" in doc
    assert "dish_dosa" in doc
    
    # Check cuisine replication (3x weight)
    tokens = doc.split()
    assert tokens.count("cuisine_south_indian") == 3
    assert tokens.count("type_quick_bites") == 2
    assert tokens.count("area_jayanagar") == 2


def test_preference_document_generation():
    prefs = {
        "preferred_cuisines": ["North Indian", "Mughlai"],
        "preferred_type": "Casual Dining",
        "preferred_area": "Koramangala 5th Block",
        "preferred_price_tier": "Moderate",
        "max_cost_for_two": 700,
        "online_order_only": True
    }
    doc = build_preference_feature_document(prefs)
    assert "cuisine_north_indian" in doc
    assert "cuisine_mughlai" in doc
    assert "type_casual_dining" in doc
    assert "area_koramangala_5th_block" in doc
    assert "price_moderate" in doc
    assert "cost_600_to_1000" in doc
    assert "online_order_yes" in doc


# =====================================================================
# 2. TF-IDF ENGINE & MATRIX TESTS
# =====================================================================

def test_tfidf_engine_fit_and_matrix_properties(sample_catalog_df):
    engine = TfidfEngine(min_df=1)  # min_df=1 for small fixture
    engine.fit(sample_catalog_df)
    
    assert engine.is_fitted is True
    assert engine.tfidf_matrix is not None
    assert engine.tfidf_matrix.shape[0] == 5
    assert len(engine.vectorizer.vocabulary_) > 10
    assert len(engine.restaurant_id_to_idx) == 5
    
    # Row vectors must have L2 norm == 1.0
    row_vec = engine.get_restaurant_vector(1).toarray().ravel()
    norm = np.linalg.norm(row_vec)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_invalid_restaurant_id_raises_error(sample_catalog_df):
    engine = TfidfEngine(min_df=1).fit(sample_catalog_df)
    with pytest.raises(KeyError):
        engine.get_restaurant_vector(9999)


# =====================================================================
# 3. COSINE SIMILARITY & RECOMMENDATION TESTS
# =====================================================================

def test_restaurant_to_restaurant_similarity(sample_catalog_df):
    recommender = ContentRecommender.from_dataframe(sample_catalog_df)
    # Query MTR Lalbagh (South Indian in Basavanagudi)
    recs = recommender.recommend_similar_restaurants(restaurant_id=1, top_k=2)
    
    assert len(recs) == 2
    # Source restaurant ID 1 must be excluded
    rec_ids = [r["restaurant_id"] for r in recs]
    assert 1 not in rec_ids
    
    # Vidyarthi Bhavan (ID 2: South Indian in Basavanagudi) must be #1 similar match
    assert recs[0]["restaurant_id"] == 2
    assert recs[0]["similarity_score"] > 0.3
    assert recs[0]["similarity_score"] <= 1.0


def test_microbrewery_similarity(sample_catalog_df):
    recommender = ContentRecommender.from_dataframe(sample_catalog_df)
    # Query Toit (Microbrewery, Italian/American)
    recs = recommender.recommend_similar_restaurants(restaurant_id=3, top_k=2)
    
    # Windmills Craftworks (ID 4: Microbrewery, Italian/American) should rank high
    assert recs[0]["restaurant_id"] == 4
    assert recs[0]["similarity_score"] > 0.3


def test_preference_matching_with_hard_filter(sample_catalog_df):
    recommender = ContentRecommender.from_dataframe(sample_catalog_df)
    prefs = {
        "preferred_cuisines": ["Italian", "American"],
        "max_cost_for_two": 1600  # Must include Toit (Rs 1500) but exclude Windmills (Rs 2500)
    }
    recs = recommender.recommend_for_preferences(preferences=prefs, top_k=5)
    
    assert len(recs) > 0
    # Toit should be top match
    assert recs[0]["restaurant_id"] == 3
    # All results must obey max_cost <= 1600
    for r in recs:
        assert r["cost_for_two_inr"] <= 1600


# =====================================================================
# 4. ARTIFACT PERSISTENCE (SAVE & LOAD) TESTS
# =====================================================================

def test_artifact_save_and_load(sample_catalog_df):
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = TfidfEngine(min_df=1).fit(sample_catalog_df)
        saved = engine.save_artifacts(tmp_dir)
        
        assert os.path.exists(saved["vectorizer"])
        assert os.path.exists(saved["matrix"])
        assert os.path.exists(saved["catalog"])
        assert os.path.exists(saved["mappings"])
        
        # Load from disk
        recommender_loaded = ContentRecommender.from_artifacts(tmp_dir)
        assert recommender_loaded.is_ready is True
        assert recommender_loaded.catalog_size == 5
        
        # Execute query on loaded model
        recs = recommender_loaded.recommend_similar_restaurants(restaurant_id=1, top_k=2)
        assert len(recs) == 2
        assert recs[0]["restaurant_id"] == 2


# =====================================================================
# 5. FASTAPI REST API INTEGRATION TESTS
# =====================================================================

def test_api_similar_restaurant_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/similar/1?top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 3
    assert len(data["recommendations"]) == 3
    assert data["recommendations"][0]["restaurant_id"] != 1


def test_api_similar_invalid_id():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/similar/99999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_api_preference_endpoint():
    client = TestClient(app)
    payload = {
        "preferred_cuisines": ["South Indian"],
        "preferred_price_tier": "Budget",
        "max_cost_for_two": 400,
        "top_k": 5
    }
    response = client.post("/api/v1/recommendations/content", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) > 0
    for item in data["recommendations"]:
        assert item["cost_for_two_inr"] <= 400


# =====================================================================
# 6. AUTHENTIC 12,481 REAL CATALOG INTEGRATION TEST
# =====================================================================

def test_full_12481_catalog_sanity():
    """Validates recommender against the full 12,481 authentic restaurant catalog."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_dir = os.path.join(base_dir, "saved_models", "content_model")
    
    assert os.path.exists(os.path.join(artifact_dir, "tfidf_matrix.joblib")), "Artifacts must exist."
    recommender = ContentRecommender.from_artifacts(artifact_dir)
    
    assert recommender.catalog_size == 12481
    
    # Test query on Vidyarthi Bhavan (ID 45)
    recs = recommender.recommend_similar_restaurants(restaurant_id=45, top_k=5)
    assert len(recs) == 5
    assert all(r["restaurant_id"] != 45 for r in recs)
    # Scores must be sorted descending
    scores = [r["similarity_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
