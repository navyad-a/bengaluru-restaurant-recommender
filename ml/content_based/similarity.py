# -*- coding: utf-8 -*-
"""
Cosine Similarity & Candidate Ranking Module
============================================
Performs sparse vector cosine similarity calculation, constraint filtering,
and deterministic ranking.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.sparse import csr_matrix


def compute_cosine_similarity_vector(
    tfidf_matrix: csr_matrix,
    query_vector: csr_matrix
) -> np.ndarray:
    """
    Computes cosine similarity between all catalog restaurants and a single query vector.
    Since both matrix rows and query vectors have L2 norm = 1.0, the cosine similarity
    is the standard sparse matrix dot product: S = X . q^T.
    
    Returns:
        1D numpy array of shape (N,) with similarity scores in [0.0, 1.0].
    """
    # Sparse dot product -> (N, 1) -> flatten to 1D array of length N
    scores = tfidf_matrix.dot(query_vector.T)
    if hasattr(scores, "toarray"):
        scores_array = scores.toarray().ravel()
    else:
        scores_array = np.asarray(scores).ravel()
    # Clip numerical precision artifacts to [0.0, 1.0]
    return np.clip(scores_array, 0.0, 1.0)


def apply_hard_filters(
    df_catalog: pd.DataFrame,
    filters: Dict[str, Any]
) -> pd.Series:
    """
    Evaluates hard constraints against the restaurant catalog.
    Returns a Boolean mask Series of length N.
    """
    mask = pd.Series(True, index=df_catalog.index)

    # 1. Maximum Cost for Two Filter
    max_cost = filters.get("max_cost_for_two")
    if max_cost is not None and max_cost > 0:
        mask &= (df_catalog["cost_for_two_inr"] <= max_cost)

    # 2. Minimum Rating Filter
    min_rating = filters.get("min_rating")
    if min_rating is not None and min_rating > 0.0:
        # Include restaurants with rating >= min_rating (or optionally include unrated if specified)
        include_unrated = filters.get("include_unrated", False)
        if include_unrated:
            mask &= (df_catalog["rating"].isna() | (df_catalog["rating"] >= min_rating))
        else:
            mask &= (df_catalog["rating"].notna() & (df_catalog["rating"] >= min_rating))

    # 3. Price Tier Filter
    price_tier = filters.get("price_tier") or filters.get("preferred_price_tier")
    if price_tier and isinstance(price_tier, str) and price_tier.strip():
        mask &= (df_catalog["price_tier"].str.lower() == price_tier.strip().lower())

    # 4. Area / Locality Filter
    area = filters.get("area") or filters.get("preferred_area") or filters.get("location")
    if area and isinstance(area, str) and area.strip():
        mask &= (df_catalog["area"].str.lower() == area.strip().lower())

    # 5. Online Order Only Filter
    if filters.get("online_order_only") is True or filters.get("online_order") is True:
        mask &= (df_catalog["online_order"] == True)

    # 6. Table Booking Only Filter
    if filters.get("book_table_only") is True or filters.get("book_table") is True:
        mask &= (df_catalog["book_table"] == True)

    # 7. Specific Required Cuisine Filter (Hard Filter)
    req_cuisine = filters.get("required_cuisine")
    if req_cuisine and isinstance(req_cuisine, str) and req_cuisine.strip():
        mask &= df_catalog["cuisines"].str.lower().str.contains(req_cuisine.strip().lower(), regex=False, na=False)

    return mask


def rank_candidates(
    df_catalog: pd.DataFrame,
    similarity_scores: np.ndarray,
    top_k: int = 10,
    exclude_restaurant_id: Optional[int] = None,
    filter_mask: Optional[pd.Series] = None
) -> List[Dict[str, Any]]:
    """
    Ranks restaurants by similarity score with deterministic tie-breaking.
    Tie-breaking hierarchy:
    1. Content similarity score (descending)
    2. Review count / votes (descending)
    3. Restaurant rating (descending)
    4. Restaurant ID (ascending)
    """
    df_ranked = df_catalog.copy()
    df_ranked["content_score"] = similarity_scores
    
    # Exclude source restaurant itself
    if exclude_restaurant_id is not None:
        df_ranked = df_ranked[df_ranked["restaurant_id"] != exclude_restaurant_id]
        
    # Apply hard filters mask if provided
    if filter_mask is not None:
        df_ranked = df_ranked[filter_mask]
        
    if df_ranked.empty:
        return []

    # Sort with deterministic tie-breaking
    # Fill NaN ratings with 0.0 temporarily for sort stability
    df_ranked["_sort_rating"] = df_ranked["rating"].fillna(0.0)
    
    df_sorted = df_ranked.sort_values(
        by=["content_score", "review_count", "_sort_rating", "restaurant_id"],
        ascending=[False, False, False, True]
    ).head(top_k)
    
    results: List[Dict[str, Any]] = []
    for _, row in df_sorted.iterrows():
        results.append({
            "restaurant_id": int(row["restaurant_id"]),
            "name": str(row["name"]),
            "content_score": round(float(row["content_score"]), 4),
            "similarity_score": round(float(row["content_score"]), 4),
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
            "review_count": int(row["review_count"]),
            "cuisines": str(row["cuisines"]),
            "restaurant_type": str(row["rest_type"]) if pd.notna(row["rest_type"]) else "Restaurant",
            "area": str(row["area"]),
            "address": str(row["address"]),
            "price_tier": str(row["price_tier"]),
            "cost_for_two_inr": int(row["cost_for_two_inr"]),
            "online_order": bool(row["online_order"]),
            "book_table": bool(row["book_table"]),
            "location_source": str(row["location_source"]),
            "location_precision": str(row["location_precision"])
        })
        
    return results
