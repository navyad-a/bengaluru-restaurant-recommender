# -*- coding: utf-8 -*-
"""
Recommendation Service Layer
Singleton managers for ContentRecommender, CollaborativeRecommender, HybridRecommender, and SpatialSearchEngine.
"""

import os
import pandas as pd
from typing import Optional
from ml.content_based.content_recommender import ContentRecommender
from ml.collaborative.collaborative_recommender import CollaborativeRecommender
from ml.hybrid.hybrid_recommender import HybridRecommender
from ml.spatial.spatial_search import SpatialSearchEngine

_content_recommender: Optional[ContentRecommender] = None
_collaborative_recommender: Optional[CollaborativeRecommender] = None
_hybrid_recommender: Optional[HybridRecommender] = None
_spatial_search_engine: Optional[SpatialSearchEngine] = None


def get_content_recommender() -> ContentRecommender:
    """
    Returns the singleton instance of ContentRecommender loaded from saved_models.
    """
    global _content_recommender
    if _content_recommender is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        artifact_dir = os.path.join(base_dir, "saved_models", "content_model")
        if os.path.exists(os.path.join(artifact_dir, "tfidf_matrix.joblib")):
            _content_recommender = ContentRecommender.from_artifacts(artifact_dir)
        else:
            csv_path = os.path.join(base_dir, "data", "processed", "restaurants_clean.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                _content_recommender = ContentRecommender.from_dataframe(df)
            else:
                raise FileNotFoundError(f"Neither model artifacts in {artifact_dir} nor catalog in {csv_path} exist.")
                
    return _content_recommender


def get_collaborative_recommender() -> CollaborativeRecommender:
    """
    Returns the singleton instance of CollaborativeRecommender loaded from saved_models.
    """
    global _collaborative_recommender
    if _collaborative_recommender is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        artifact_dir = os.path.join(base_dir, "saved_models", "collaborative_model")
        catalog_csv = os.path.join(base_dir, "data", "processed", "restaurants_clean.csv")
        train_csv = os.path.join(base_dir, "data", "processed", "synthetic_train_ratings.csv")

        if os.path.exists(os.path.join(artifact_dir, "svd_model.joblib")):
            _collaborative_recommender = CollaborativeRecommender.from_artifacts(
                artifact_dir=artifact_dir,
                catalog_csv_path=catalog_csv,
                train_ratings_csv_path=train_csv
            )
        else:
            raise FileNotFoundError(f"Collaborative SVD model artifacts not found in: {artifact_dir}. Run scripts/build_svd_model.py first.")

    return _collaborative_recommender


def get_hybrid_recommender() -> HybridRecommender:
    """
    Returns the singleton instance of HybridRecommender combining Content, Collaborative, Location, and Quality.
    """
    global _hybrid_recommender
    if _hybrid_recommender is None:
        content_rec = get_content_recommender()
        
        collab_rec = None
        try:
            collab_rec = get_collaborative_recommender()
        except Exception:
            collab_rec = None

        _hybrid_recommender = HybridRecommender(
            content_recommender=content_rec,
            collaborative_recommender=collab_rec,
            df_restaurants=content_rec.engine.restaurant_catalog
        )

    return _hybrid_recommender


def get_spatial_search_engine() -> SpatialSearchEngine:
    """
    Returns the singleton instance of SpatialSearchEngine with BallTree spatial index.
    """
    global _spatial_search_engine
    if _spatial_search_engine is None:
        content_rec = get_content_recommender()
        _spatial_search_engine = SpatialSearchEngine(content_rec.engine.restaurant_catalog)
        
    return _spatial_search_engine
