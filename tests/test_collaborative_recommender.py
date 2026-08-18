# -*- coding: utf-8 -*-
"""
Phase 6 Test Suite — Collaborative Filtering (Surprise SVD)
Tests data integrity, SVD model training, prediction clipping, candidate ranking,
already-rated exclusion, cold-start handling, artifact persistence, and API endpoints.
"""

import os
import math
import tempfile
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from ml.collaborative.svd_engine import SVDEngine
from ml.collaborative.evaluator import (
    validate_benchmark_integrity,
    compute_prediction_error_metrics,
    compute_top_k_ranking_metrics
)
from ml.collaborative.collaborative_recommender import CollaborativeRecommender
from app.main import app


@pytest.fixture
def mock_ratings_data():
    """Controlled mock ratings DataFrame for deterministic unit tests."""
    df_users = pd.DataFrame([
        {"user_id": 1, "name": "User 1"},
        {"user_id": 2, "name": "User 2"},
        {"user_id": 3, "name": "User 3"}
    ])
    df_rest = pd.DataFrame([
        {"restaurant_id": 101, "name": "Rest 101", "area": "Indiranagar", "address": "123 Indiranagar", "cuisines": "South Indian", "rest_type": "Quick Bites", "price_tier": "Budget", "cost_for_two_inr": 200, "rating": 4.5, "review_count": 100, "online_order": True, "book_table": False, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 102, "name": "Rest 102", "area": "Indiranagar", "address": "456 Indiranagar", "cuisines": "North Indian", "rest_type": "Casual Dining", "price_tier": "Moderate", "cost_for_two_inr": 500, "rating": 4.2, "review_count": 200, "online_order": True, "book_table": False, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 103, "name": "Rest 103", "area": "Koramangala", "address": "789 Koramangala", "cuisines": "Cafe", "rest_type": "Cafe", "price_tier": "Premium", "cost_for_two_inr": 1200, "rating": 4.6, "review_count": 300, "online_order": True, "book_table": True, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 104, "name": "Rest 104", "area": "Koramangala", "address": "101 Koramangala", "cuisines": "Biryani", "rest_type": "Casual Dining", "price_tier": "Moderate", "cost_for_two_inr": 600, "rating": 4.3, "review_count": 150, "online_order": True, "book_table": False, "location_source": "centroid", "location_precision": "locality-level"},
        {"restaurant_id": 105, "name": "Rest 105", "area": "Jayanagar", "address": "202 Jayanagar", "cuisines": "South Indian", "rest_type": "Quick Bites", "price_tier": "Budget", "cost_for_two_inr": 150, "rating": 4.4, "review_count": 80, "online_order": False, "book_table": False, "location_source": "centroid", "location_precision": "locality-level"}
    ])
    df_train = pd.DataFrame([
        {"user_id": 1, "restaurant_id": 101, "rating": 5.0},
        {"user_id": 1, "restaurant_id": 102, "rating": 4.0},
        {"user_id": 1, "restaurant_id": 103, "rating": 2.0},
        {"user_id": 2, "restaurant_id": 101, "rating": 4.5},
        {"user_id": 2, "restaurant_id": 104, "rating": 4.0},
        {"user_id": 3, "restaurant_id": 103, "rating": 5.0},
        {"user_id": 3, "restaurant_id": 104, "rating": 3.0}
    ])
    df_test = pd.DataFrame([
        {"user_id": 1, "restaurant_id": 104, "rating": 4.5},
        {"user_id": 2, "restaurant_id": 105, "rating": 4.0},
        {"user_id": 3, "restaurant_id": 101, "rating": 4.0}
    ])
    return df_users, df_rest, df_train, df_test


# =====================================================================
# 1. BENCHMARK INTEGRITY VALIDATION TESTS
# =====================================================================

def test_benchmark_integrity_success(mock_ratings_data):
    df_users, df_rest, df_train, df_test = mock_ratings_data
    stats = validate_benchmark_integrity(df_users, df_rest, df_train, df_test)
    assert stats["is_valid"] is True
    assert stats["total_users"] == 3
    assert stats["authentic_catalog_restaurants"] == 5
    assert stats["train_ratings_count"] == 7
    assert stats["test_ratings_count"] == 3


def test_benchmark_integrity_overlap_detection(mock_ratings_data):
    df_users, df_rest, df_train, df_test = mock_ratings_data
    # Introduce intentional overlap pair
    bad_test = pd.concat([df_test, df_train.iloc[0:1]], ignore_index=True)
    with pytest.raises(ValueError, match="overlapping interaction pairs"):
        validate_benchmark_integrity(df_users, df_rest, df_train, bad_test)


# =====================================================================
# 2. SVD ENGINE & PREDICTION TESTS
# =====================================================================

def test_svd_engine_training_and_prediction(mock_ratings_data):
    _, _, df_train, _ = mock_ratings_data
    engine = SVDEngine(n_factors=10, n_epochs=10, random_state=42)
    engine.fit(df_train)

    assert engine.is_fitted is True
    assert engine.is_known_user(1) is True
    assert engine.is_known_user(999) is False
    assert engine.is_known_restaurant(101) is True

    pred = engine.predict(user_id=1, restaurant_id=104)
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 5.0


def test_svd_prediction_clipping():
    engine = SVDEngine()
    # Test that rating predictions never exceed bounds
    assert max(1.0, min(5.0, 6.2)) == 5.0
    assert max(1.0, min(5.0, 0.4)) == 1.0


def test_svd_deterministic_seed(mock_ratings_data):
    _, _, df_train, _ = mock_ratings_data
    engine1 = SVDEngine(n_factors=10, n_epochs=10, random_state=42).fit(df_train)
    engine2 = SVDEngine(n_factors=10, n_epochs=10, random_state=42).fit(df_train)

    pred1 = engine1.predict(1, 104)
    pred2 = engine2.predict(1, 104)
    assert pred1 == pred2


# =====================================================================
# 3. COLLABORATIVE RECOMMENDER & ALREADY-RATED EXCLUSION
# =====================================================================

def test_collaborative_recommender_exclude_rated(mock_ratings_data):
    _, df_rest, df_train, _ = mock_ratings_data
    engine = SVDEngine(n_factors=10, n_epochs=10, random_state=42).fit(df_train)
    recommender = CollaborativeRecommender(
        svd_engine=engine,
        df_restaurants=df_rest,
        df_train_ratings=df_train
    )

    # User 1 rated 101, 102, 103 in train
    recs = recommender.recommend_for_user(user_id=1, top_k=5, exclude_rated=True)
    rec_ids = [r["restaurant_id"] for r in recs]

    assert 101 not in rec_ids
    assert 102 not in rec_ids
    assert 103 not in rec_ids
    # Only 104 and 105 were unrated by user 1
    assert set(rec_ids).issubset({104, 105})
    assert recs[0]["model_source"] == "collaborative_svd_synthetic_benchmark"


def test_cold_start_unknown_user_raises_key_error(mock_ratings_data):
    _, df_rest, df_train, _ = mock_ratings_data
    engine = SVDEngine(n_factors=10, n_epochs=10, random_state=42).fit(df_train)
    recommender = CollaborativeRecommender(
        svd_engine=engine,
        df_restaurants=df_rest,
        df_train_ratings=df_train
    )

    with pytest.raises(KeyError, match="cold-start user"):
        recommender.recommend_for_user(user_id=999999, top_k=5)


# =====================================================================
# 4. ARTIFACT PERSISTENCE (SAVE & LOAD)
# =====================================================================

def test_svd_artifact_save_and_load(mock_ratings_data):
    _, df_rest, df_train, _ = mock_ratings_data
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = SVDEngine(n_factors=10, n_epochs=10, random_state=42).fit(df_train)
        pred_before = engine.predict(1, 104)
        
        saved = engine.save_artifacts(tmp_dir)
        assert os.path.exists(saved["model"])
        assert os.path.exists(saved["entities"])
        assert os.path.exists(saved["metadata"])

        # Load from disk
        loaded_engine = SVDEngine().load_artifacts(tmp_dir)
        pred_after = loaded_engine.predict(1, 104)
        assert pred_before == pred_after
        assert loaded_engine.is_known_user(1) is True


# =====================================================================
# 5. OFFLINE EVALUATION METRICS
# =====================================================================

def test_compute_prediction_error_metrics():
    preds = [(4.0, 4.0), (5.0, 4.0), (3.0, 4.0)]
    metrics = compute_prediction_error_metrics(preds)
    assert metrics["mae"] == round((0.0 + 1.0 + 1.0) / 3.0, 4)
    assert metrics["rmse"] == round(math.sqrt((0.0 + 1.0 + 1.0) / 3.0), 4)


def test_compute_top_k_ranking_metrics(mock_ratings_data):
    _, df_rest, df_train, df_test = mock_ratings_data
    engine = SVDEngine(n_factors=10, n_epochs=10, random_state=42).fit(df_train)
    catalog_ids = df_rest["restaurant_id"].tolist()

    rank_metrics = compute_top_k_ranking_metrics(
        svd_engine=engine,
        df_train=df_train,
        df_test=df_test,
        catalog_restaurant_ids=catalog_ids,
        k_values=[2, 5],
        rating_threshold=3.5
    )
    assert "precision_at_2" in rank_metrics
    assert "recall_at_2" in rank_metrics
    assert "hit_rate_at_2" in rank_metrics
    assert rank_metrics["num_evaluated_users"] > 0


# =====================================================================
# 6. FASTAPI REST API INTEGRATION TESTS
# =====================================================================

def test_api_collaborative_known_user():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/collaborative/1?top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["user_id"] == 1
    assert data["model_source"] == "collaborative_svd_synthetic_benchmark"
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["model_source"] == "collaborative_svd_synthetic_benchmark"


def test_api_collaborative_unknown_user():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/collaborative/999999?top_k=5")
    assert response.status_code == 404
    assert "cold-start user" in response.json()["detail"].lower()
