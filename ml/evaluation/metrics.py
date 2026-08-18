# -*- coding: utf-8 -*-
"""
Evaluation Metrics Module
=========================
Computes standard offline ranking and rating prediction metrics:
Precision@K, Recall@K, HitRate@K, NDCG@K, MRR@K, MAP@K, Catalog Coverage, User Coverage,
RMSE, MAE, and bootstrap confidence intervals.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Optional, Tuple


def compute_precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Precision@K = |Rec[:K] ∩ Rel| / K"""
    if k <= 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = len(set(rec_k).intersection(relevant_ids))
    return float(hits / k)


def compute_recall_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Recall@K = |Rec[:K] ∩ Rel| / |Rel|"""
    if not relevant_ids or k <= 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = len(set(rec_k).intersection(relevant_ids))
    return float(hits / len(relevant_ids))


def compute_hit_rate_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """HitRate@K = 1.0 if any hit in Rec[:K] else 0.0"""
    if not relevant_ids or k <= 0:
        return 0.0
    rec_k = recommended_ids[:k]
    return 1.0 if len(set(rec_k).intersection(relevant_ids)) > 0 else 0.0


def compute_ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K with binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0

    rec_k = recommended_ids[:k]
    dcg = 0.0
    for idx, item_id in enumerate(rec_k):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(idx + 2)  # idx + 2 because idx is 0-indexed (rank 1 = log2(2) = 1)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_mrr_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """MRR@K = 1 / rank of first relevant item in Rec[:K], or 0.0"""
    if not relevant_ids or k <= 0:
        return 0.0
    for idx, item_id in enumerate(recommended_ids[:k]):
        if item_id in relevant_ids:
            return float(1.0 / (idx + 1))
    return 0.0


def compute_map_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Mean Average Precision (AP@K) for a single user."""
    if not relevant_ids or k <= 0:
        return 0.0

    score = 0.0
    num_hits = 0
    rec_k = recommended_ids[:k]

    for idx, item_id in enumerate(rec_k):
        if item_id in relevant_ids:
            num_hits += 1
            precision_at_i = num_hits / (idx + 1)
            score += precision_at_i

    return float(score / min(k, len(relevant_ids))) if len(relevant_ids) > 0 else 0.0


def compute_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """RMSE = sqrt(mean((y_true - y_pred)^2))"""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    errs = np.array(y_true, dtype=np.float64) - np.array(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(errs ** 2)))


def compute_mae(y_true: List[float], y_pred: List[float]) -> float:
    """MAE = mean(|y_true - y_pred|)"""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    errs = np.abs(np.array(y_true, dtype=np.float64) - np.array(y_pred, dtype=np.float64))
    return float(np.mean(errs))


def compute_catalog_coverage(all_recommended_ids: List[List[int]], total_catalog_size: int, k: int) -> float:
    """Catalog Coverage@K = |Union_u(Rec_u[:K])| / |Catalog|"""
    if total_catalog_size <= 0:
        return 0.0
    unique_recs = set()
    for rec_list in all_recommended_ids:
        unique_recs.update(rec_list[:k])
    return float(len(unique_recs) / total_catalog_size)


def compute_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Computes (mean, lower_ci, upper_ci) using non-parametric bootstrap resampling.
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(random_state)
    boot_means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_bootstrap)
    ]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return float(np.mean(arr)), low, high
