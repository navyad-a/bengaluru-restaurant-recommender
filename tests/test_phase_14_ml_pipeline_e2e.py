# -*- coding: utf-8 -*-
"""
Phase 14 Test Suite: ML Pipeline Integrity, Cold-Start Routing, MMR & Explainability
"""

import pytest
import os
import pandas as pd
from app.services.recommendation_service import (
    get_hybrid_recommender,
    get_content_recommender,
    get_collaborative_recommender,
    get_spatial_search_engine
)


@pytest.mark.ml
def test_ml_catalog_exact_count_and_columns():
    """Verifies authentic catalog contains exactly 12,481 venues with required attributes."""
    csv_path = "data/processed/restaurants_clean.csv"
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    
    assert len(df) == 12481
    assert df["restaurant_id"].nunique() == 12481
    
    required_cols = [
        "restaurant_id", "name", "area", "cuisines",
        "cost_for_two_inr", "price_tier", "rating", "review_count",
        "latitude", "longitude"
    ]
    for col in required_cols:
        assert col in df.columns
        assert df[col].isna().sum() == 0 or col == "rating"


@pytest.mark.ml
def test_ml_synthetic_benchmark_integrity():
    """Verifies synthetic collaborative filtering benchmark properties."""
    users_path = "data/processed/synthetic_users.csv"
    train_path = "data/processed/synthetic_train_ratings.csv"
    test_path = "data/processed/synthetic_test_ratings.csv"
    
    assert os.path.exists(users_path)
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    
    df_users = pd.read_csv(users_path)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    assert len(df_users) == 600
    assert df_train["rating"].between(1.0, 5.0).all()
    assert df_test["rating"].between(1.0, 5.0).all()
    
    # Zero leakage check
    train_pairs = set(zip(df_train["user_id"], df_train["restaurant_id"]))
    test_pairs = set(zip(df_test["user_id"], df_test["restaurant_id"]))
    assert len(train_pairs.intersection(test_pairs)) == 0


@pytest.mark.ml
def test_ml_cold_start_routing_strategies():
    """Verifies hybrid recommender correctly activates 5-tier routing strategies."""
    hybrid = get_hybrid_recommender()
    
    # 1. Warm Known User
    res_warm = hybrid.recommend(user_id=2, top_k=5)
    assert res_warm["is_cold_start"] is False
    assert res_warm["strategy"] == "warm_hybrid"
    
    # 2. Unknown User with Preferences
    res_pref = hybrid.recommend(
        user_id=None,
        preferences={"preferred_cuisines": ["Biryani"]},
        top_k=5
    )
    assert res_pref["is_cold_start"] is True
    assert res_pref["strategy"] == "profile_content"
    
    # 3. Completely Unknown User -> Bayesian Popularity
    res_anon = hybrid.recommend(user_id=None, top_k=5)
    assert res_anon["is_cold_start"] is True
    assert res_anon["strategy"] == "global_popularity"


@pytest.mark.ml
def test_ml_mmr_lambda_tradeoff_gradient():
    """Verifies that decreasing lambda (0.90 -> 0.50) expands diversity."""
    hybrid = get_hybrid_recommender()
    
    res_high_lambda = hybrid.recommend(user_id=2, top_k=10, mmr_enabled=True, mmr_lambda=0.95)
    res_low_lambda = hybrid.recommend(user_id=2, top_k=10, mmr_enabled=True, mmr_lambda=0.50)
    
    ild_high = res_high_lambda["diversification"]["diversity_metrics"]["intra_list_diversity"]
    ild_low = res_low_lambda["diversification"]["diversity_metrics"]["intra_list_diversity"]
    
    assert ild_low >= ild_high - 1e-5


@pytest.mark.ml
def test_ml_explainability_factual_consistency():
    """Verifies that explanations are grounded and never claim unselected cuisines."""
    hybrid = get_hybrid_recommender()
    target_cuisines = ["Kerala", "South Indian"]
    
    res = hybrid.recommend(
        user_id=None,
        preferences={"preferred_cuisines": target_cuisines},
        top_k=5
    )
    
    for item in res["recommendations"]:
        assert "explanation" in item
        assert len(item["explanation"]) > 10
        meta = item.get("explanation_metadata")
        if meta and meta.get("matched_cuisines"):
            for c in meta["matched_cuisines"]:
                assert any(c.lower() in tc.lower() or tc.lower() in c.lower() for tc in target_cuisines)

