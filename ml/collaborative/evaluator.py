# -*- coding: utf-8 -*-
"""
Collaborative Filtering Evaluation & Benchmark Integrity Module
===============================================================
Computes offline rating error metrics (RMSE, MAE) and ranking metrics (Precision@K,
Recall@K, Hit Rate@K) on the synthetic collaborative filtering benchmark.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Tuple, Optional


def validate_benchmark_integrity(
    df_users: pd.DataFrame,
    df_restaurants: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validates data integrity rules across the synthetic benchmark partitions:
    - User ID existence in users table
    - Restaurant ID existence in authentic catalog
    - Rating bounds in [1.0, 5.0]
    - No duplicate (user_id, restaurant_id) interactions
    - No train/test interaction overlap
    - User coverage across partitions
    """
    errors = []

    # 1. User existence
    valid_user_ids = set(df_users["user_id"].unique())
    train_users = set(df_train["user_id"].unique())
    test_users = set(df_test["user_id"].unique())

    if not train_users.issubset(valid_user_ids):
        errors.append(f"Found {len(train_users - valid_user_ids)} train user_ids not in synthetic_users.csv")
    if not test_users.issubset(valid_user_ids):
        errors.append(f"Found {len(test_users - valid_user_ids)} test user_ids not in synthetic_users.csv")

    # 2. Restaurant existence in authentic catalog
    valid_rest_ids = set(df_restaurants["restaurant_id"].unique())
    train_rests = set(df_train["restaurant_id"].unique())
    test_rests = set(df_test["restaurant_id"].unique())

    if not train_rests.issubset(valid_rest_ids):
        errors.append(f"Found {len(train_rests - valid_rest_ids)} train restaurant_ids not in restaurants_clean.csv")
    if not test_rests.issubset(valid_rest_ids):
        errors.append(f"Found {len(test_rests - valid_rest_ids)} test restaurant_ids not in restaurants_clean.csv")

    # 3. Rating bounds [1.0, 5.0]
    if (df_train["rating"] < 1.0).any() or (df_train["rating"] > 5.0).any():
        errors.append("Train ratings contain values outside [1.0, 5.0]")
    if (df_test["rating"] < 1.0).any() or (df_test["rating"] > 5.0).any():
        errors.append("Test ratings contain values outside [1.0, 5.0]")

    # 4. Duplicate checks
    train_dups = df_train.duplicated(subset=["user_id", "restaurant_id"]).sum()
    if train_dups > 0:
        errors.append(f"Found {train_dups} duplicate (user_id, restaurant_id) pairs in train set")

    test_dups = df_test.duplicated(subset=["user_id", "restaurant_id"]).sum()
    if test_dups > 0:
        errors.append(f"Found {test_dups} duplicate (user_id, restaurant_id) pairs in test set")

    # 5. Train/Test overlap check
    train_pairs = set(zip(df_train["user_id"], df_train["restaurant_id"]))
    test_pairs = set(zip(df_test["user_id"], df_test["restaurant_id"]))
    overlap = train_pairs.intersection(test_pairs)
    if len(overlap) > 0:
        errors.append(f"Found {len(overlap)} overlapping interaction pairs between train and test partitions")

    if errors:
        raise ValueError(f"Benchmark integrity validation failed: {'; '.join(errors)}")

    # Compute descriptive dataset statistics
    total_ratings = len(df_train) + len(df_test)
    num_users = len(valid_user_ids)
    num_catalog_rests = len(valid_rest_ids)
    represented_rests = len(train_rests.union(test_rests))
    
    ratings_per_user = df_train.groupby("user_id").size()
    ratings_per_rest = df_train.groupby("restaurant_id").size()
    
    total_possible_entries = num_users * num_catalog_rests
    density = (total_ratings / total_possible_entries) * 100
    sparsity = 100.0 - density

    stats = {
        "is_valid": True,
        "total_users": num_users,
        "authentic_catalog_restaurants": num_catalog_rests,
        "represented_restaurants": represented_rests,
        "train_ratings_count": len(df_train),
        "test_ratings_count": len(df_test),
        "total_ratings_count": total_ratings,
        "avg_train_ratings_per_user": round(float(ratings_per_user.mean()), 2),
        "min_train_ratings_per_user": int(ratings_per_user.min()),
        "max_train_ratings_per_user": int(ratings_per_user.max()),
        "avg_train_ratings_per_restaurant": round(float(ratings_per_rest.mean()), 2),
        "matrix_density_percent": f"{density:.4f}%",
        "matrix_sparsity_percent": f"{sparsity:.4f}%",
        "train_rating_mean": round(float(df_train["rating"].mean()), 3),
        "train_rating_std": round(float(df_train["rating"].std()), 3),
        "test_rating_mean": round(float(df_test["rating"].mean()), 3),
        "test_rating_std": round(float(df_test["rating"].std()), 3),
    }
    return stats


def compute_prediction_error_metrics(
    predictions: List[Tuple[float, float]]
) -> Dict[str, float]:
    """
    Computes RMSE and MAE from a list of (actual_rating, predicted_rating) pairs.
    """
    if not predictions:
        return {"rmse": 0.0, "mae": 0.0}

    actuals = np.array([p[0] for p in predictions], dtype=np.float64)
    preds = np.array([p[1] for p in predictions], dtype=np.float64)

    errors = actuals - preds
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "num_evaluations": len(predictions)
    }


def compute_top_k_ranking_metrics(
    svd_engine,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    catalog_restaurant_ids: List[int],
    k_values: List[int] = [5, 10],
    rating_threshold: float = 3.5
) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, and HitRate@K across test users.
    
    Protocol:
    - Known training items are excluded from recommendations for each user.
    - Ground truth positives: Held-out test items for user u where test_rating >= rating_threshold.
    - Candidates for ranking: All unrated catalog restaurants.
    - Top-K items with highest predicted rating are selected.
    """
    train_user_items = df_train.groupby("user_id")["restaurant_id"].apply(set).to_dict()
    test_user_ratings = df_test.groupby("user_id").apply(
        lambda g: dict(zip(g["restaurant_id"], g["rating"]))
    ).to_dict()

    precisions = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}
    hit_rates = {k: [] for k in k_values}

    catalog_set = set(catalog_restaurant_ids)

    for u_id, test_dict in test_user_ratings.items():
        ground_truth_positives = {r_id for r_id, r_val in test_dict.items() if r_val >= rating_threshold}
        if not ground_truth_positives:
            continue

        known_train = train_user_items.get(u_id, set())
        # Candidate pool: all catalog restaurants not seen in train
        candidate_ids = list(catalog_set - known_train)
        
        # Predict ratings for all candidates
        # Batch predict for speed
        scored_candidates = []
        for r_id in candidate_ids:
            pred_score = svd_engine.predict(u_id, r_id)
            scored_candidates.append((r_id, pred_score))

        # Sort candidates descending by predicted rating
        # Tie-breaking by restaurant ID ascending
        scored_candidates.sort(key=lambda x: (-x[1], x[0]))

        max_k = max(k_values)
        top_items_max = [x[0] for x in scored_candidates[:max_k]]

        for k in k_values:
            top_k_items = set(top_items_max[:k])
            hits = len(top_k_items.intersection(ground_truth_positives))
            
            p_k = hits / float(k)
            r_k = hits / float(len(ground_truth_positives))
            hr_k = 1.0 if hits > 0 else 0.0

            precisions[k].append(p_k)
            recalls[k].append(r_k)
            hit_rates[k].append(hr_k)

    results = {}
    for k in k_values:
        results[f"precision_at_{k}"] = round(float(np.mean(precisions[k])), 4) if precisions[k] else 0.0
        results[f"recall_at_{k}"] = round(float(np.mean(recalls[k])), 4) if recalls[k] else 0.0
        results[f"hit_rate_at_{k}"] = round(float(np.mean(hit_rates[k])), 4) if hit_rates[k] else 0.0

    results["num_evaluated_users"] = len(precisions[k_values[0]]) if k_values and precisions[k_values[0]] else 0
    return results
