# -*- coding: utf-8 -*-
"""
Phase 10 Test Suite — Recommendation Diversification (MMR) & Explainability Engine
Tests sparse similarity, MMR mathematical formulation, duplicate suppression,
diversity metrics, explainability truthfulness, edge cases, and API integration.
"""

import json
import os
import pytest
import numpy as np
import scipy.sparse as sp
from fastapi.testclient import TestClient

from ml.diversification.similarity import SparseSimilarityEngine
from ml.diversification.redundancy import RedundancyChecker, normalize_restaurant_name
from ml.diversification.diversity_metrics import DiversityMetricsCalculator
from ml.diversification.mmr import MMRDiversifier
from ml.diversification.explainability import RecommendationExplainabilityEngine
from ml.cold_start.routing import ColdStartStrategy
from app.services.recommendation_service import get_hybrid_recommender
from app.main import app


# ----------------------------------------------------------------------
# 1. SPARSE SIMILARITY & REDUNDANCY UNIT TESTS
# ----------------------------------------------------------------------

def test_sparse_similarity_engine_normalization_and_pairwise():
    # Build 3 small sparse TF-IDF vectors
    mat = sp.csr_matrix([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],  # Identical to 0
        [0.0, 0.0, 1.0]   # Orthogonal to 0
    ])
    id_to_idx = {101: 0, 102: 1, 103: 2}
    idx_to_id = {0: 101, 1: 102, 2: 103}

    engine = SparseSimilarityEngine(mat, id_to_idx, idx_to_id)
    
    # Self similarity is 1.0
    assert np.isclose(engine.compute_pairwise_similarity(101, 101), 1.0)
    # Identical vector similarity is 1.0
    assert np.isclose(engine.compute_pairwise_similarity(101, 102), 1.0)
    # Orthogonal vector similarity is 0.0
    assert np.isclose(engine.compute_pairwise_similarity(101, 103), 0.0)


def test_sparse_similarity_compute_max_similarity_to_set():
    mat = sp.csr_matrix([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0]
    ])
    id_to_idx = {1: 0, 2: 1, 3: 2}
    idx_to_id = {0: 1, 1: 2, 2: 3}
    engine = SparseSimilarityEngine(mat, id_to_idx, idx_to_id)

    max_sim = engine.compute_max_similarity_to_set(candidate_id=3, selected_ids=[1, 2])
    assert max_sim > 0.0
    assert engine.compute_max_similarity_to_set(candidate_id=3, selected_ids=[]) == 0.0


def test_normalize_restaurant_name():
    assert normalize_restaurant_name("Meghana Foods - Koramangala") == "meghana foods"
    assert normalize_restaurant_name("Empire Restaurant, Indiranagar") == "empire restaurant"
    assert normalize_restaurant_name("Toit : 100ft Road") == "toit"
    assert normalize_restaurant_name("") == ""


def test_redundancy_checker_near_duplicate_and_chain_limits():
    checker = RedundancyChecker(max_same_chain_in_top_k=2, max_similarity_threshold=0.90)
    
    selected = [
        {"restaurant_id": 1, "name": "Meghana Foods - Koramangala", "area": "Koramangala"},
        {"restaurant_id": 2, "name": "Meghana Foods - Jayanagar", "area": "Jayanagar"}
    ]
    
    cand3 = {"restaurant_id": 3, "name": "Meghana Foods - Indiranagar", "area": "Indiranagar"}
    assert checker.violates_soft_chain_limit(cand3, selected) is True

    cand_other = {"restaurant_id": 4, "name": "Nagarjuna", "area": "Indiranagar"}
    assert checker.violates_soft_chain_limit(cand_other, selected) is False


# ----------------------------------------------------------------------
# 2. MMR MATHEMATICAL FORMULATION & SELECTION LOGIC
# ----------------------------------------------------------------------

def test_mmr_first_item_selection_and_greedy_step():
    mat = sp.csr_matrix([
        [1.0, 0.0],
        [1.0, 0.0],  # Highly similar to item 1
        [0.0, 1.0]   # Diverse from item 1
    ])
    id_to_idx = {1: 0, 2: 1, 3: 2}
    idx_to_id = {0: 1, 1: 2, 2: 3}
    engine = SparseSimilarityEngine(mat, id_to_idx, idx_to_id)
    diversifier = MMRDiversifier(similarity_engine=engine)

    candidates = [
        {"restaurant_id": 1, "name": "A", "hybrid_score": 0.90, "review_count": 100, "rating": 4.5, "cuisines": "South Indian", "area": "Area 1", "restaurant_type": "Dine"},
        {"restaurant_id": 2, "name": "B", "hybrid_score": 0.88, "review_count": 90, "rating": 4.4, "cuisines": "South Indian", "area": "Area 1", "restaurant_type": "Dine"},
        {"restaurant_id": 3, "name": "C", "hybrid_score": 0.82, "review_count": 80, "rating": 4.3, "cuisines": "North Indian", "area": "Area 2", "restaurant_type": "Cafe"}
    ]

    # With lambda = 0.50, item 3 should be selected second over item 2 due to diversity!
    # MMR(2) = 0.50*0.88 - 0.50*1.0 = 0.44 - 0.50 = -0.06
    # MMR(3) = 0.50*0.82 - 0.50*0.0 = 0.41
    selected, metrics = diversifier.diversify(candidates, top_k=2, lambda_param=0.50)
    assert len(selected) == 2
    assert selected[0]["restaurant_id"] == 1
    assert selected[1]["restaurant_id"] == 3


def test_mmr_lambda_1_equals_pure_relevance():
    mat = sp.csr_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    engine = SparseSimilarityEngine(mat, {1: 0, 2: 1, 3: 2}, {0: 1, 1: 2, 2: 3})
    diversifier = MMRDiversifier(similarity_engine=engine)

    candidates = [
        {"restaurant_id": 1, "name": "A", "hybrid_score": 0.90, "review_count": 100, "rating": 4.5, "cuisines": "C1", "area": "A1", "restaurant_type": "T1"},
        {"restaurant_id": 2, "name": "B", "hybrid_score": 0.88, "review_count": 90, "rating": 4.4, "cuisines": "C1", "area": "A1", "restaurant_type": "T1"},
        {"restaurant_id": 3, "name": "C", "hybrid_score": 0.82, "review_count": 80, "rating": 4.3, "cuisines": "C2", "area": "A2", "restaurant_type": "T2"}
    ]

    selected, _ = diversifier.diversify(candidates, top_k=2, lambda_param=1.0)
    assert [r["restaurant_id"] for r in selected] == [1, 2]


def test_mmr_lambda_0_maximizes_diversity():
    mat = sp.csr_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    engine = SparseSimilarityEngine(mat, {1: 0, 2: 1, 3: 2}, {0: 1, 1: 2, 2: 3})
    diversifier = MMRDiversifier(similarity_engine=engine)

    candidates = [
        {"restaurant_id": 1, "name": "A", "hybrid_score": 0.90, "review_count": 100, "rating": 4.5, "cuisines": "C1", "area": "A1", "restaurant_type": "T1"},
        {"restaurant_id": 2, "name": "B", "hybrid_score": 0.89, "review_count": 90, "rating": 4.4, "cuisines": "C1", "area": "A1", "restaurant_type": "T1"},
        {"restaurant_id": 3, "name": "C", "hybrid_score": 0.50, "review_count": 80, "rating": 4.3, "cuisines": "C2", "area": "A2", "restaurant_type": "T2"}
    ]

    # At lambda = 0.0, diversity penalty completely dominates
    selected, _ = diversifier.diversify(candidates, top_k=2, lambda_param=0.0)
    assert selected[0]["restaurant_id"] == 1
    assert selected[1]["restaurant_id"] == 3


def test_mmr_invalid_lambda_raises_value_error():
    mat = sp.csr_matrix([[1.0]])
    engine = SparseSimilarityEngine(mat, {1: 0}, {0: 1})
    diversifier = MMRDiversifier(similarity_engine=engine)

    with pytest.raises(ValueError, match="lambda_param must be in"):
        diversifier.diversify([{"restaurant_id": 1, "name": "A"}], lambda_param=-0.1)

    with pytest.raises(ValueError, match="lambda_param must be in"):
        diversifier.diversify([{"restaurant_id": 1, "name": "A"}], lambda_param=1.5)


# ----------------------------------------------------------------------
# 3. DIVERSITY METRICS CALCULATION
# ----------------------------------------------------------------------

def test_diversity_metrics_calculation():
    mat = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    engine = SparseSimilarityEngine(mat, {1: 0, 2: 1}, {0: 1, 1: 2})

    recs = [
        {"restaurant_id": 1, "name": "A", "cuisines": "South Indian, Kerala", "restaurant_type": "Dine", "area": "Indiranagar", "hybrid_score": 0.9},
        {"restaurant_id": 2, "name": "B", "cuisines": "North Indian", "restaurant_type": "Cafe", "area": "Koramangala", "hybrid_score": 0.8}
    ]

    metrics = DiversityMetricsCalculator.compute_list_metrics(recs, engine, pre_mmr_relevance=[0.9, 0.85])
    assert metrics["top_k"] == 2
    assert metrics["unique_restaurant_type_ratio"] == 1.0  # 2 types / 2 items
    assert metrics["unique_locality_ratio"] == 1.0         # 2 areas / 2 items
    assert metrics["avg_pairwise_similarity"] == 0.0       # Orthogonal
    assert metrics["intra_list_diversity"] == 1.0          # 1 - 0.0 = 1.0
    assert metrics["redundancy_rate"] == 0.0
    assert metrics["relevance_retention_pct"] > 0.0


# ----------------------------------------------------------------------
# 4. EXPLAINABILITY ENGINE TRUTHFULNESS & SIGNAL GROUNDING
# ----------------------------------------------------------------------

def test_explainability_no_false_collaborative_signal():
    item = {
        "restaurant_id": 1,
        "name": "MTR",
        "cuisines": "South Indian",
        "rating": 4.5,
        "review_count": 3000,
        "area": "Basavanagudi",
        "cost_for_two_inr": 200,
        "distance_km": None
    }
    # Cold start user with 0 collaborative weight
    weights = {"content": 0.60, "collaborative": 0.0, "location": 0.0, "quality": 0.40}
    scores = {"content": 0.85, "collaborative": 0.0, "location": 0.0, "quality": 0.90}

    meta = RecommendationExplainabilityEngine.generate_explanation_metadata(
        item=item,
        strategy=ColdStartStrategy.PROFILE_CONTENT_QUALITY,
        effective_weights=weights,
        scores=scores,
        user_preferences={"preferred_cuisines": ["South Indian"]}
    )

    assert "collaborative" not in meta["contributing_signals"]
    assert "taste profile" not in meta["explanation"].lower()
    assert "matches your criteria" in meta["explanation"].lower()


def test_explainability_diversified_reason():
    item = {
        "restaurant_id": 10,
        "name": "Toit",
        "cuisines": "Italian, Continental",
        "rating": 4.7,
        "review_count": 14000,
        "area": "Indiranagar",
        "cost_for_two_inr": 1500,
        "distance_km": 2.0
    }
    weights = {"content": 0.40, "collaborative": 0.20, "location": 0.15, "quality": 0.25}
    scores = {"content": 0.70, "collaborative": 0.80, "location": 0.60, "quality": 0.95}

    meta = RecommendationExplainabilityEngine.generate_explanation_metadata(
        item=item,
        strategy=ColdStartStrategy.WARM_HYBRID,
        effective_weights=weights,
        scores=scores,
        is_diversified=True,
        similarity_to_prior=0.45
    )

    assert meta["diversity_reason"] is not None
    assert "introduce menu and cuisine variety" in meta["diversity_reason"]


# ----------------------------------------------------------------------
# 5. FULL HYBRID ORCHESTRATION & EDGE CASES
# ----------------------------------------------------------------------

def test_hybrid_recommender_mmr_diversification_flow():
    hybrid_engine = get_hybrid_recommender()
    
    # Run recommendation with MMR enabled
    res = hybrid_engine.recommend(
        preferences={"preferred_cuisines": ["Biryani", "North Indian"]},
        filters={"area": "Koramangala 5th Block"},
        mmr_enabled=True,
        mmr_lambda=0.75,
        top_k=5
    )

    assert res["count"] == 5
    assert "diversification" in res
    assert res["diversification"]["enabled"] is True
    assert res["diversification"]["lambda_param"] == 0.75
    assert "diversity_metrics" in res["diversification"]
    assert "intra_list_diversity" in res["diversification"]["diversity_metrics"]

    # Each item must have explanation metadata
    for r in res["recommendations"]:
        assert "explanation_metadata" in r
        assert r["explanation_metadata"] is not None
        assert "primary_signal" in r["explanation_metadata"]


def test_hybrid_recommender_top_k_1_and_empty():
    hybrid_engine = get_hybrid_recommender()
    
    # Top-K = 1
    res_1 = hybrid_engine.recommend(top_k=1)
    assert res_1["count"] == 1
    assert len(res_1["recommendations"]) == 1

    # Impossible budget filter -> empty candidate pool
    res_empty = hybrid_engine.recommend(
        filters={"max_cost_for_two": 1, "min_rating": 5.0}
    )
    assert res_empty["count"] == 0
    assert len(res_empty["recommendations"]) == 0
    assert res_empty["diversification"]["diversity_metrics"]["top_k"] == 0


def test_saved_model_artifact_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models", "diversification_model", "mmr_config.json"
    )
    assert os.path.exists(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["phase"] == 10
    assert cfg["default_lambda"] == 0.75
    assert cfg["similarity_metric"] == "sparse_tfidf_cosine"


# ----------------------------------------------------------------------
# 6. REST API INTEGRATION TESTS
# ----------------------------------------------------------------------

def test_api_hybrid_post_with_mmr_parameters():
    client = TestClient(app)
    payload = {
        "preferred_cuisines": ["South Indian", "Karnataka"],
        "preferred_area": "Jayanagar",
        "top_k": 5,
        "mmr_enabled": True,
        "mmr_lambda": 0.70
    }
    response = client.post("/api/v1/recommendations/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "diversification" in data
    assert data["diversification"]["enabled"] is True
    assert data["diversification"]["lambda_param"] == 0.70
    assert len(data["recommendations"]) == 5
    for item in data["recommendations"]:
        assert "explanation" in item
        assert "explanation_metadata" in item


def test_api_hybrid_get_with_mmr_query_params():
    client = TestClient(app)
    response = client.get("/api/v1/recommendations/hybrid/1?top_k=4&mmr_enabled=true&mmr_lambda=0.80")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["diversification"]["lambda_param"] == 0.80
    assert len(data["recommendations"]) == 4
