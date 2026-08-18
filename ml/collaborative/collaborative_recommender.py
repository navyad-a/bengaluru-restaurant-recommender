# -*- coding: utf-8 -*-
"""
Collaborative Recommendation Service
====================================
Generates personalized restaurant recommendations for users using the trained
SVD Matrix Factorization model and authentic catalog metadata.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional, Set
from ml.collaborative.svd_engine import SVDEngine


class CollaborativeRecommender:
    """
    Production-grade Collaborative Filtering Recommender service.
    """

    def __init__(
        self,
        svd_engine: Optional[SVDEngine] = None,
        df_restaurants: Optional[pd.DataFrame] = None,
        df_train_ratings: Optional[pd.DataFrame] = None,
        artifact_dir: Optional[str] = None
    ):
        self.engine = svd_engine or SVDEngine()
        self.df_restaurants = df_restaurants
        self.user_rated_items: Dict[int, Set[int]] = {}

        if df_train_ratings is not None:
            self.user_rated_items = (
                df_train_ratings.groupby("user_id")["restaurant_id"]
                .apply(set)
                .to_dict()
            )

        if artifact_dir and os.path.exists(os.path.join(artifact_dir, "svd_model.joblib")):
            self.engine.load_artifacts(artifact_dir)

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str,
        catalog_csv_path: str,
        train_ratings_csv_path: Optional[str] = None
    ) -> "CollaborativeRecommender":
        """
        Initializes CollaborativeRecommender by loading trained SVD artifacts and catalog.
        """
        engine = SVDEngine().load_artifacts(artifact_dir)
        df_restaurants = pd.read_csv(catalog_csv_path)
        
        df_train = None
        if train_ratings_csv_path and os.path.exists(train_ratings_csv_path):
            df_train = pd.read_csv(train_ratings_csv_path)

        return cls(
            svd_engine=engine,
            df_restaurants=df_restaurants,
            df_train_ratings=df_train
        )

    @property
    def is_ready(self) -> bool:
        return self.engine.is_fitted and self.df_restaurants is not None

    def is_known_user(self, user_id: int) -> bool:
        return self.engine.is_known_user(user_id)

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generates top-K collaborative filtering recommendations for a known user.
        
        Raises:
            KeyError: If user_id is unseen / cold-start in the SVD training interaction matrix.
        """
        if not self.is_ready:
            raise RuntimeError("Collaborative Recommender is not initialized or fitted.")

        if not self.is_known_user(user_id):
            raise KeyError(
                f"User ID {user_id} is unknown to Collaborative SVD (cold-start user). "
                "Collaborative Filtering requires historical interaction ratings."
            )

        # 1. Identify candidate restaurant IDs
        rated_by_user = self.user_rated_items.get(user_id, set()) if exclude_rated else set()
        
        # Candidate pool: All authentic catalog restaurants not already rated by this user
        df_candidates = self.df_restaurants[
            ~self.df_restaurants["restaurant_id"].isin(rated_by_user)
        ].copy()

        if df_candidates.empty:
            return []

        # 2. Predict SVD ratings for all candidate restaurants
        candidate_ids = df_candidates["restaurant_id"].tolist()
        pred_tuples = self.engine.predict_batch(user_id, candidate_ids)
        
        # Map predictions back to candidates
        pred_dict = dict(pred_tuples)
        df_candidates["predicted_rating"] = df_candidates["restaurant_id"].map(pred_dict)

        # 3. Deterministic tie-breaking:
        # 1. predicted_rating descending
        # 2. review_count descending
        # 3. authentic rating descending
        # 4. restaurant_id ascending
        df_candidates["_sort_rating"] = df_candidates["rating"].fillna(0.0) if "rating" in df_candidates.columns else 0.0
        sort_cols = ["predicted_rating"]
        ascending_flags = [False]
        if "review_count" in df_candidates.columns:
            sort_cols.append("review_count")
            ascending_flags.append(False)
        sort_cols.extend(["_sort_rating", "restaurant_id"])
        ascending_flags.extend([False, True])

        df_sorted = df_candidates.sort_values(
            by=sort_cols,
            ascending=ascending_flags
        ).head(top_k)

        # 4. Format structured output with authentic metadata
        recommendations: List[Dict[str, Any]] = []
        for _, row in df_sorted.iterrows():
            row_dict = row.to_dict()
            recommendations.append({
                "restaurant_id": int(row_dict.get("restaurant_id", 0)),
                "name": str(row_dict.get("name", "Unknown")),
                "predicted_rating": round(float(row_dict.get("predicted_rating", 0.0)), 4),
                "rating": float(row_dict["rating"]) if pd.notna(row_dict.get("rating")) else None,
                "review_count": int(row_dict.get("review_count", 0)),
                "cuisines": str(row_dict.get("cuisines", "")),
                "restaurant_type": str(row_dict.get("rest_type", row_dict.get("restaurant_type", "Restaurant"))),
                "area": str(row_dict.get("area", "")),
                "address": str(row_dict.get("address", "")),
                "price_tier": str(row_dict.get("price_tier", "Moderate")),
                "cost_for_two_inr": int(row_dict.get("cost_for_two_inr", 400)),
                "online_order": bool(row_dict.get("online_order", False)),
                "book_table": bool(row_dict.get("book_table", False)),
                "location_source": str(row_dict.get("location_source", "Bengaluru locality centroid")),
                "location_precision": str(row_dict.get("location_precision", "locality-level")),
                "model_source": "collaborative_svd_synthetic_benchmark"
            })

        return recommendations
