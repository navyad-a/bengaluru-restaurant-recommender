# -*- coding: utf-8 -*-
"""
Hybrid Recommendation Engine Orchestrator (with MMR Diversification & Explainability Engine)
=============================================================================================
Combines Content-Based Similarity, Collaborative SVD, Location Proximity,
and Bayesian Quality Shrinkage into a unified recommendation score, integrated with
the Cold-Start Strategy Router, Item Imputation, and Maximal Marginal Relevance (MMR) Diversification.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set

from ml.content_based.content_recommender import ContentRecommender
from ml.collaborative.collaborative_recommender import CollaborativeRecommender
from ml.hybrid.quality import BayesianQualityScorer
from ml.hybrid.location import LocationScorer
from ml.hybrid.scoring import (
    DEFAULT_HYBRID_WEIGHTS,
    normalize_content_score,
    normalize_collaborative_score,
    compute_effective_weights,
    compute_hybrid_score
)
from ml.hybrid.candidate_generator import CandidateGenerator
from ml.cold_start.routing import ColdStartRouter, ColdStartStrategy
from ml.cold_start.popularity import BayesianPopularityEngine
from ml.cold_start.item_cold_start import ItemColdStartHandler
from ml.diversification.similarity import SparseSimilarityEngine
from ml.diversification.mmr import MMRDiversifier
from ml.diversification.diversity_metrics import DiversityMetricsCalculator
from ml.diversification.explainability import RecommendationExplainabilityEngine


class HybridRecommender:
    """
    Unified production Hybrid Recommender service with MMR Diversification & Explainability.
    """

    def __init__(
        self,
        content_recommender: ContentRecommender,
        collaborative_recommender: Optional[CollaborativeRecommender] = None,
        location_scorer: Optional[LocationScorer] = None,
        quality_scorer: Optional[BayesianQualityScorer] = None,
        df_restaurants: Optional[pd.DataFrame] = None,
        base_weights: Optional[Dict[str, float]] = None,
        similarity_engine: Optional[SparseSimilarityEngine] = None,
        mmr_diversifier: Optional[MMRDiversifier] = None
    ):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.location_scorer = location_scorer or LocationScorer(decay_tau_km=3.0)
        self.df_restaurants = (
            df_restaurants
            if df_restaurants is not None
            else content_recommender.engine.restaurant_catalog
        )
        self.quality_scorer = quality_scorer or BayesianQualityScorer.from_dataframe(
            self.df_restaurants
        )
        self.candidate_generator = CandidateGenerator(self.df_restaurants)
        self.popularity_engine = BayesianPopularityEngine(self.df_restaurants, self.quality_scorer)
        self.item_cold_start_handler = ItemColdStartHandler(self.df_restaurants)
        self.base_weights = dict(base_weights or DEFAULT_HYBRID_WEIGHTS)

        # Initialize Sparse Similarity Engine & MMR Diversifier
        if similarity_engine is not None:
            self.similarity_engine = similarity_engine
        else:
            self.similarity_engine = SparseSimilarityEngine(
                tfidf_matrix=content_recommender.engine.tfidf_matrix,
                id_to_idx=content_recommender.engine.restaurant_id_to_idx,
                idx_to_id=content_recommender.engine.idx_to_restaurant_id
            )

        self.mmr_diversifier = mmr_diversifier or MMRDiversifier(
            similarity_engine=self.similarity_engine
        )

    @classmethod
    def from_artifacts(
        cls,
        content_artifact_dir: str,
        collaborative_artifact_dir: Optional[str] = None,
        catalog_csv_path: Optional[str] = None,
        train_ratings_csv_path: Optional[str] = None,
        base_weights: Optional[Dict[str, float]] = None
    ) -> "HybridRecommender":
        """
        Instantiates HybridRecommender by loading all component model artifacts.
        """
        content_rec = ContentRecommender.from_artifacts(content_artifact_dir)
        
        collab_rec = None
        if collaborative_artifact_dir and os.path.exists(
            os.path.join(collaborative_artifact_dir, "svd_model.joblib")
        ):
            collab_rec = CollaborativeRecommender.from_artifacts(
                artifact_dir=collaborative_artifact_dir,
                catalog_csv_path=catalog_csv_path or os.path.join(content_artifact_dir, "restaurant_catalog.joblib"),
                train_ratings_csv_path=train_ratings_csv_path
            )

        df_cat = content_rec.engine.restaurant_catalog
        return cls(
            content_recommender=content_rec,
            collaborative_recommender=collab_rec,
            df_restaurants=df_cat,
            base_weights=base_weights
        )

    def recommend(
        self,
        user_id: Optional[int] = None,
        preferences: Optional[Dict[str, Any]] = None,
        target_restaurant_id: Optional[int] = None,
        user_coords: Optional[Tuple[float, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        mmr_enabled: bool = True,
        mmr_lambda: float = 0.75,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Executes hybrid recommendation routed through Cold-Start, MMR Diversification, and Explainability.
        """
        # 1. Inspect user history & context to classify cold-start strategy
        is_known_user = False
        user_interaction_count = 0

        if user_id is not None and self.collaborative_recommender is not None:
            if self.collaborative_recommender.is_known_user(user_id):
                is_known_user = True
                user_interaction_count = len(
                    self.collaborative_recommender.user_rated_items.get(user_id, set())
                )

        has_preferences = bool(preferences) or (target_restaurant_id is not None)
        has_location = bool(user_coords is not None or (filters and filters.get("area")))

        strategy = ColdStartRouter.determine_strategy(
            user_id=user_id,
            user_interaction_count=user_interaction_count,
            has_preferences=has_preferences,
            has_location=has_location,
            is_known_collaborative_user=is_known_user
        )

        # 2. Compute effective weights
        if weights is not None:
            active_signals = {"content", "quality"}
            if is_known_user:
                active_signals.add("collaborative")
            if user_coords is not None:
                active_signals.add("location")
            effective_weights = compute_effective_weights(weights, active_signals)
        else:
            effective_weights = ColdStartRouter.get_strategy_weights(
                strategy=strategy,
                has_location=(user_coords is not None)
            )

        # 3. Generate candidate pool (at least 5x top_k to give MMR rich variety)
        pool_size = max(top_k * 6, 80)
        df_candidates = self.candidate_generator.generate_candidates(
            content_recommender=self.content_recommender,
            collaborative_recommender=self.collaborative_recommender,
            location_scorer=self.location_scorer,
            user_id=user_id if is_known_user else None,
            preferences=preferences,
            target_restaurant_id=target_restaurant_id,
            user_coords=user_coords,
            filters=filters,
            candidate_pool_size=pool_size
        )

        if df_candidates.empty:
            empty_metrics = DiversityMetricsCalculator.compute_list_metrics([], self.similarity_engine)
            return {
                "user_id": user_id,
                "is_cold_start": not is_known_user,
                "strategy": strategy.value,
                "model_source": "hybrid",
                "effective_weights": effective_weights,
                "count": 0,
                "diversification": {
                    "enabled": mmr_enabled,
                    "method": "MMR",
                    "lambda_param": mmr_lambda,
                    "diversity_metrics": empty_metrics
                },
                "recommendations": []
            }

        # 4. Compute Component Scores for Candidates
        # A. Content Score
        content_scores_dict: Dict[int, float] = {}
        if target_restaurant_id is not None:
            c_recs = self.content_recommender.recommend_similar_restaurants(
                restaurant_id=target_restaurant_id,
                top_k=max(len(df_candidates) * 2, 200)
            )
            content_scores_dict = {r["restaurant_id"]: r["content_score"] for r in c_recs}
        elif preferences:
            c_recs = self.content_recommender.recommend_for_preferences(
                preferences=preferences,
                top_k=max(len(df_candidates) * 2, 200)
            )
            content_scores_dict = {r["restaurant_id"]: r["content_score"] for r in c_recs}

        # B. Collaborative Score
        collab_scores_dict: Dict[int, float] = {}
        if is_known_user and self.collaborative_recommender is not None:
            cand_ids = df_candidates["restaurant_id"].tolist()
            preds = self.collaborative_recommender.engine.predict_batch(user_id, cand_ids)
            collab_scores_dict = {
                r_id: normalize_collaborative_score(score) for r_id, score in preds
            }

        # C. Location Score & Distance
        loc_scores_dict: Dict[int, float] = {}
        distances_dict: Dict[int, float] = {}
        if user_coords is not None:
            u_lat, u_lon = user_coords
            dists, loc_scores = self.location_scorer.score_vectorized(
                user_lat=u_lat,
                user_lon=u_lon,
                restaurant_lats=df_candidates["latitude"].to_numpy(),
                restaurant_lons=df_candidates["longitude"].to_numpy()
            )
            for idx, r_id in enumerate(df_candidates["restaurant_id"]):
                loc_scores_dict[r_id] = float(loc_scores[idx])
                distances_dict[r_id] = float(dists[idx])

        # D. Quality Score (Bayesian Quality)
        quality_scores_arr = self.quality_scorer.score_series(
            ratings=df_candidates["rating"],
            review_counts=df_candidates["review_count"]
        )
        quality_scores_dict = dict(zip(df_candidates["restaurant_id"], quality_scores_arr))

        # 5. Assemble Candidate Records
        scored_records = []
        for _, row in df_candidates.iterrows():
            r_id = int(row["restaurant_id"])
            s_content = normalize_content_score(content_scores_dict.get(r_id, 0.0))
            s_collab = collab_scores_dict.get(r_id, 0.0) if is_known_user else 0.0
            s_loc = loc_scores_dict.get(r_id, 0.0) if user_coords is not None else 0.0
            s_qual = float(quality_scores_dict.get(r_id, 0.5))

            scores = {
                "content": s_content,
                "collaborative": s_collab,
                "location": s_loc,
                "quality": s_qual
            }
            s_hybrid = compute_hybrid_score(scores, effective_weights)

            row_dict = row.to_dict()
            row_dict["distance_km"] = distances_dict.get(r_id)
            
            # Enrich with item cold-start metadata
            item_enriched = self.item_cold_start_handler.enrich_item_metadata(row_dict)

            scored_records.append({
                "restaurant_id": r_id,
                "name": str(row["name"]),
                "hybrid_score": s_hybrid,
                "content_score": s_content,
                "collaborative_score": s_collab,
                "location_score": s_loc,
                "quality_score": s_qual,
                "distance_km": distances_dict.get(r_id),
                "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
                "review_count": int(row["review_count"]),
                "cuisines": str(row["cuisines"]),
                "restaurant_type": str(row.get("rest_type", row.get("restaurant_type", "Restaurant"))),
                "area": str(row["area"]),
                "address": str(row["address"]),
                "price_tier": str(row["price_tier"]),
                "cost_for_two_inr": int(row["cost_for_two_inr"]),
                "online_order": bool(row["online_order"]),
                "book_table": bool(row["book_table"]),
                "location_source": str(row["location_source"]),
                "location_precision": str(row["location_precision"]),
                "model_source": "hybrid",
                "is_unrated": item_enriched.get("is_unrated", False),
                "imputed_rating_prior": item_enriched.get("imputed_rating_prior"),
                "component_scores": scores
            })

        # Pre-sort candidates deterministically before MMR
        scored_records.sort(
            key=lambda x: (
                -x["hybrid_score"],
                -x["content_score"],
                -x["quality_score"],
                -x["collaborative_score"],
                -x["review_count"],
                -(x["rating"] or 0.0),
                x["restaurant_id"]
            )
        )

        # 6. Apply MMR Diversification or Pure Relevance Ranking
        if mmr_enabled:
            selected_items, diversity_metrics = self.mmr_diversifier.diversify(
                candidates=scored_records,
                top_k=top_k,
                lambda_param=mmr_lambda,
                relevance_field="hybrid_score"
            )
        else:
            selected_items = scored_records[:top_k]
            diversity_metrics = DiversityMetricsCalculator.compute_list_metrics(
                recommendations=selected_items,
                similarity_engine=self.similarity_engine,
                pre_mmr_relevance=[r["hybrid_score"] for r in scored_records]
            )

        # 7. Generate Truthful Structured Explanations
        final_recommendations = []
        for idx, item in enumerate(selected_items):
            scores = item.pop("component_scores", {
                "content": item["content_score"],
                "collaborative": item["collaborative_score"],
                "location": item["location_score"],
                "quality": item["quality_score"]
            })

            expl_meta = RecommendationExplainabilityEngine.generate_explanation_metadata(
                item=item,
                strategy=strategy,
                effective_weights=effective_weights,
                scores=scores,
                user_preferences=preferences,
                is_diversified=item.get("is_diversified", False),
                similarity_to_prior=item.get("max_similarity_to_prior", 0.0)
            )

            item_final = dict(item)
            item_final["explanation"] = expl_meta["explanation"]
            item_final["explanation_metadata"] = expl_meta
            final_recommendations.append(item_final)

        return {
            "user_id": user_id,
            "is_cold_start": not is_known_user,
            "strategy": strategy.value,
            "model_source": "hybrid",
            "effective_weights": effective_weights,
            "count": len(final_recommendations),
            "diversification": {
                "enabled": mmr_enabled,
                "method": "MMR",
                "lambda_param": mmr_lambda,
                "diversity_metrics": diversity_metrics
            },
            "recommendations": final_recommendations
        }
