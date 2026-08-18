# -*- coding: utf-8 -*-
"""
Phase 11 Test Suite — Offline ML Benchmark Evaluation & Comparative Study
Tests metric implementations, leakage checks, candidate exclusion, and bootstrap CIs.
"""

import pytest
import numpy as np
import pandas as pd

from ml.evaluation.leakage_checker import LeakageChecker, DataLeakageError
from ml.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_mrr_at_k,
    compute_map_at_k,
    compute_catalog_coverage,
    compute_rmse,
    compute_mae,
    compute_bootstrap_ci
)
from ml.evaluation.evaluator import OfflineBenchmarkEvaluator
from app.services.recommendation_service import get_hybrid_recommender


# ----------------------------------------------------------------------
# 1. LEAKAGE CHECKER & INTEGRITY TESTS
# ----------------------------------------------------------------------

def test_leakage_checker_valid_benchmark_data():
    df_catalog = pd.DataFrame([
        {"restaurant_id": 1, "name": "A"},
        {"restaurant_id": 2, "name": "B"},
        {"restaurant_id": 3, "name": "C"}
    ])
    df_train = pd.DataFrame([
        {"user_id": 101, "restaurant_id": 1, "rating": 4.5},
        {"user_id": 101, "restaurant_id": 2, "rating": 4.0}
    ])
    df_test = pd.DataFrame([
        {"user_id": 101, "restaurant_id": 3, "rating": 5.0}
    ])

    report = LeakageChecker.verify_integrity(df_train, df_test, df_catalog)
    assert report["status"] == "passed"
    assert report["overlap_count"] == 0


def test_leakage_checker_raises_on_collision():
    df_catalog = pd.DataFrame([{"restaurant_id": 1}, {"restaurant_id": 2}])
    df_train = pd.DataFrame([{"user_id": 1, "restaurant_id": 1, "rating": 4.0}])
    # Introduce deliberate collision
    df_test = pd.DataFrame([{"user_id": 1, "restaurant_id": 1, "rating": 4.5}])

    with pytest.raises(DataLeakageError, match="Data Leakage Detected"):
        LeakageChecker.verify_integrity(df_train, df_test, df_catalog)


def test_leakage_checker_raises_on_invalid_ratings():
    df_catalog = pd.DataFrame([{"restaurant_id": 1}])
    df_train = pd.DataFrame([{"user_id": 1, "restaurant_id": 1, "rating": 6.0}])  # > 5.0
    df_test = pd.DataFrame([{"user_id": 2, "restaurant_id": 1, "rating": 4.0}])

    with pytest.raises(DataLeakageError, match="Invalid rating values found"):
        LeakageChecker.verify_integrity(df_train, df_test, df_catalog)


# ----------------------------------------------------------------------
# 2. METRICS UNIT TESTS
# ----------------------------------------------------------------------

def test_precision_recall_hit_rate_at_k():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4, 6}  # Hits at rank 2 and 4

    assert compute_precision_at_k(recs, rel, k=5) == 2 / 5
    assert compute_precision_at_k(recs, rel, k=2) == 1 / 2
    assert compute_recall_at_k(recs, rel, k=5) == 2 / 3
    assert compute_hit_rate_at_k(recs, rel, k=1) == 0.0
    assert compute_hit_rate_at_k(recs, rel, k=2) == 1.0


def test_ndcg_at_k():
    recs = [1, 2, 3, 4, 5]
    # Perfect ranking (hits at rank 1 and 2)
    rel_perfect = {1, 2}
    ndcg_perf = compute_ndcg_at_k(recs, rel_perfect, k=5)
    assert np.isclose(ndcg_perf, 1.0)

    # Sub-optimal ranking (hits at rank 4 and 5)
    rel_low = {4, 5}
    ndcg_low = compute_ndcg_at_k(recs, rel_low, k=5)
    assert 0.0 < ndcg_low < 1.0
    assert ndcg_low < ndcg_perf


def test_mrr_and_map_at_k():
    recs = [10, 20, 30, 40]
    rel = {30, 40}

    # First hit is at rank 3 -> MRR = 1/3
    assert np.isclose(compute_mrr_at_k(recs, rel, k=4), 1.0 / 3.0)
    # No hit in top 2 -> MRR = 0.0
    assert compute_mrr_at_k(recs, rel, k=2) == 0.0

    # MAP@4
    map_score = compute_map_at_k(recs, rel, k=4)
    assert 0.0 < map_score <= 1.0


def test_rmse_and_mae():
    y_true = [4.0, 3.0, 5.0]
    y_pred = [4.5, 3.0, 4.0]
    # Errors: [0.5, 0.0, -1.0] -> Sq: [0.25, 0, 1.0] -> mean = 1.25/3 -> sqrt = 0.6455
    assert np.isclose(compute_rmse(y_true, y_pred), np.sqrt((0.25 + 0.0 + 1.0) / 3))
    assert np.isclose(compute_mae(y_true, y_pred), (0.5 + 0.0 + 1.0) / 3)


def test_catalog_coverage_and_bootstrap_ci():
    all_recs = [[1, 2], [2, 3], [4, 5]]
    # Unique = {1, 2, 3, 4, 5} -> 5 / 10 = 0.50
    cov = compute_catalog_coverage(all_recs, total_catalog_size=10, k=2)
    assert cov == 0.50

    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean, low, high = compute_bootstrap_ci(vals, n_bootstrap=100)
    assert low <= mean <= high


# ----------------------------------------------------------------------
# 3. EVALUATOR INTEGRATION & CANDIDATE EXCLUSION
# ----------------------------------------------------------------------

def test_evaluator_candidate_exclusion_and_metrics():
    hybrid_engine = get_hybrid_recommender()
    df_catalog = hybrid_engine.df_restaurants
    
    # Create small mock train & test data
    df_train = pd.DataFrame([
        {"user_id": 1, "restaurant_id": 1, "rating": 5.0},
        {"user_id": 1, "restaurant_id": 2, "rating": 4.5}
    ])
    df_test = pd.DataFrame([
        {"user_id": 1, "restaurant_id": 3, "rating": 5.0},
        {"user_id": 1, "restaurant_id": 4, "rating": 4.0}
    ])

    evaluator = OfflineBenchmarkEvaluator(
        hybrid_recommender=hybrid_engine,
        df_train_ratings=df_train,
        df_test_ratings=df_test,
        df_catalog=df_catalog,
        positive_rating_threshold=4.0
    )

    metrics = evaluator.evaluate_model(model_name="Popularity", k_values=[5, 10])
    assert metrics["model"] == "Popularity"
    assert "precision@5" in metrics
    assert "ndcg@5" in metrics
    assert "intra_list_diversity@10" in metrics
