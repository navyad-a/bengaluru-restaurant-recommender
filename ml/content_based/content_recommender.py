# -*- coding: utf-8 -*-
"""
Content-Based Recommendation Engine
====================================
Unified production recommender providing:
1. Mode A: Restaurant-to-Restaurant Similarity
2. Mode B: User Preference-to-Restaurant Similarity
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from ml.content_based.tfidf_engine import TfidfEngine
from ml.content_based.content_features import build_preference_feature_document
from ml.content_based.similarity import (
    compute_cosine_similarity_vector,
    apply_hard_filters,
    rank_candidates
)


class ContentRecommender:
    """
    Production-grade Content-Based Recommendation Service.
    """

    def __init__(self, artifact_dir: Optional[str] = None):
        self.engine = TfidfEngine()
        self.artifact_dir = artifact_dir
        if artifact_dir and os.path.exists(os.path.join(artifact_dir, "tfidf_matrix.joblib")):
            self.engine.load_artifacts(artifact_dir)

    @classmethod
    def from_dataframe(cls, df_restaurants: pd.DataFrame) -> "ContentRecommender":
        """
        Initializes and fits a ContentRecommender directly from a cleaned restaurant catalog DataFrame.
        """
        instance = cls()
        instance.engine.fit(df_restaurants)
        return instance

    @classmethod
    def from_artifacts(cls, artifact_dir: str) -> "ContentRecommender":
        """
        Initializes a ContentRecommender by loading precomputed model artifacts from disk.
        """
        instance = cls(artifact_dir=artifact_dir)
        return instance

    @property
    def is_ready(self) -> bool:
        return self.engine.is_fitted

    @property
    def catalog_size(self) -> int:
        return len(self.engine.restaurant_id_to_idx) if self.engine.is_fitted else 0

    def get_restaurant_by_id(self, restaurant_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves authentic restaurant record by its integer surrogate ID.
        """
        if not self.is_ready or self.engine.restaurant_catalog is None:
            return None
        matches = self.engine.restaurant_catalog[
            self.engine.restaurant_catalog["restaurant_id"] == restaurant_id
        ]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def recommend_similar_restaurants(
        self,
        restaurant_id: int,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Mode A: Restaurant-to-Restaurant Recommendation
        Finds restaurants with the highest TF-IDF metadata cosine similarity to the query restaurant.
        """
        if not self.is_ready:
            raise RuntimeError("Content recommender model is not fitted or loaded.")
            
        # 1. Fetch sparse query vector for target restaurant
        query_vector = self.engine.get_restaurant_vector(restaurant_id)
        
        # 2. Compute 1xN cosine similarity vector across the entire catalog
        similarity_scores = compute_cosine_similarity_vector(
            self.engine.tfidf_matrix,
            query_vector
        )
        
        # 3. Apply optional hard filters
        filter_mask = None
        if filters:
            filter_mask = apply_hard_filters(self.engine.restaurant_catalog, filters)
            
        # 4. Rank candidates and return top-K (excluding source restaurant)
        return rank_candidates(
            df_catalog=self.engine.restaurant_catalog,
            similarity_scores=similarity_scores,
            top_k=top_k,
            exclude_restaurant_id=restaurant_id,
            filter_mask=filter_mask
        )

    def recommend_for_preferences(
        self,
        preferences: Dict[str, Any],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Mode B: Preference-to-Restaurant Recommendation
        Constructs a query document from user preferences and scores restaurants using cosine similarity.
        """
        if not self.is_ready:
            raise RuntimeError("Content recommender model is not fitted or loaded.")
            
        # 1. Build structured preference query document
        pref_doc = build_preference_feature_document(preferences)
        
        # 2. Transform into sparse TF-IDF query vector
        query_vector = self.engine.transform_query(pref_doc)
        
        # 3. Compute cosine similarity against all catalog restaurants
        similarity_scores = compute_cosine_similarity_vector(
            self.engine.tfidf_matrix,
            query_vector
        )
        
        # 4. Combine explicit preferences and hard filters
        combined_filters = {}
        if filters:
            combined_filters.update(filters)
        # Inherit numeric constraints from preferences if not overridden
        if "max_cost_for_two" in preferences and "max_cost_for_two" not in combined_filters:
            combined_filters["max_cost_for_two"] = preferences["max_cost_for_two"]
        if "min_rating" in preferences and "min_rating" not in combined_filters:
            combined_filters["min_rating"] = preferences["min_rating"]
        if "online_order_only" in preferences and "online_order_only" not in combined_filters:
            combined_filters["online_order_only"] = preferences["online_order_only"]
        if "book_table_only" in preferences and "book_table_only" not in combined_filters:
            combined_filters["book_table_only"] = preferences["book_table_only"]
            
        filter_mask = None
        if combined_filters:
            filter_mask = apply_hard_filters(self.engine.restaurant_catalog, combined_filters)
            
        # 5. Rank and return top-K
        return rank_candidates(
            df_catalog=self.engine.restaurant_catalog,
            similarity_scores=similarity_scores,
            top_k=top_k,
            exclude_restaurant_id=None,
            filter_mask=filter_mask
        )
