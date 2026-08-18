# -*- coding: utf-8 -*-
"""
Offline Benchmark Evaluator & Comparative Study Orchestrator
============================================================
Evaluates recommendation models on held-out test interactions using candidate exclusion,
relevance thresholds, ranking/diversity metrics, cold-start segmentation, and ablation sweeps.
"""

import os
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Optional, Tuple

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
from ml.evaluation.leakage_checker import LeakageChecker
from ml.hybrid.hybrid_recommender import HybridRecommender
from ml.diversification.diversity_metrics import DiversityMetricsCalculator


class OfflineBenchmarkEvaluator:
    """
    Orchestrates rigorous, leakage-free offline evaluation of all recommendation engines.
    """

    def __init__(
        self,
        hybrid_recommender: HybridRecommender,
        df_train_ratings: pd.DataFrame,
        df_test_ratings: pd.DataFrame,
        df_catalog: pd.DataFrame,
        df_users: Optional[pd.DataFrame] = None,
        positive_rating_threshold: float = 4.0,
        random_state: int = 42
    ):
        self.hybrid_recommender = hybrid_recommender
        self.df_train = df_train_ratings
        self.df_test = df_test_ratings
        self.df_catalog = df_catalog
        self.df_users = df_users
        self.positive_threshold = float(positive_rating_threshold)
        self.random_state = int(random_state)

        # Verify zero leakage
        self.integrity_report = LeakageChecker.verify_integrity(
            df_train=self.df_train,
            df_test=self.df_test,
            df_catalog=self.df_catalog,
            df_users=self.df_users
        )

        # Precompute user training sets & ground-truth relevant test sets
        self.train_user_items: Dict[int, Set[int]] = (
            self.df_train.groupby("user_id")["restaurant_id"].apply(set).to_dict()
        )
        self.train_user_ratings: Dict[int, Dict[int, float]] = (
            self.df_train.groupby("user_id")
            .apply(lambda g: dict(zip(g["restaurant_id"], g["rating"])))
            .to_dict()
        )

        # Filter test items to relevant ones (rating >= threshold)
        df_test_rel = self.df_test[self.df_test["rating"] >= self.positive_threshold]
        self.test_user_relevant: Dict[int, Set[int]] = (
            df_test_rel.groupby("user_id")["restaurant_id"].apply(set).to_dict()
        )
        self.test_users = sorted(list(self.test_user_relevant.keys()))

    def _build_user_content_preference_from_train(self, user_id: int) -> Dict[str, Any]:
        """
        Builds user preference dictionary from user profile and training ratings (no test leakage).
        """
        prefs = {}
        if self.df_users is not None:
            u_row = self.df_users[self.df_users["user_id"] == user_id]
            if not u_row.empty:
                cuis = u_row["preferred_cuisines"].iloc[0]
                area = u_row["preferred_area"].iloc[0] if "preferred_area" in u_row.columns else None
                if pd.notna(cuis):
                    prefs["preferred_cuisines"] = str(cuis)
                if pd.notna(area):
                    prefs["preferred_area"] = str(area)
                return prefs

        user_train = self.df_train[
            (self.df_train["user_id"] == user_id) & (self.df_train["rating"] >= self.positive_threshold)
        ]
        if user_train.empty:
            user_train = self.df_train[self.df_train["user_id"] == user_id]

        if user_train.empty:
            return {}

        top_rated_rests = user_train.sort_values("rating", ascending=False)["restaurant_id"].head(5)
        matched_cat = self.df_catalog[self.df_catalog["restaurant_id"].isin(top_rated_rests)]
        
        # Extract favorite cuisines
        cuisines_list = []
        for c in matched_cat["cuisines"].dropna():
            cuisines_list.extend([x.strip() for x in c.split(",")])

        top_cuisines = pd.Series(cuisines_list).value_counts().head(3).index.tolist()
        pref_cuisines_str = ", ".join(top_cuisines) if top_cuisines else None

        return {"preferred_cuisines": pref_cuisines_str} if pref_cuisines_str else {}

    def evaluate_model(
        self,
        model_name: str,
        k_values: List[int] = [5, 10, 20],
        user_subset: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a specified recommendation strategy across the test users.
        """
        eval_users = user_subset if user_subset is not None else self.test_users
        max_k = max(k_values)

        user_metrics: Dict[int, Dict[str, float]] = {u: {} for u in eval_users}
        all_recommendations: List[List[int]] = []
        all_rec_dicts: List[List[Dict[str, Any]]] = []

        for user_id in eval_users:
            relevant_items = self.test_user_relevant.get(user_id, set())
            train_items = self.train_user_items.get(user_id, set())

            # 1. Generate model recommendations
            if model_name == "Popularity":
                # Pure Bayesian Popularity baseline
                pop_recs = self.hybrid_recommender.popularity_engine.get_global_popular(
                    top_k=max_k + len(train_items)
                )
                filtered_recs = [r for r in pop_recs if int(r["restaurant_id"]) not in train_items][:max_k]
                rec_ids = [int(r["restaurant_id"]) for r in filtered_recs]
                rec_dicts = filtered_recs

            elif model_name == "Content-Based":
                prefs = self._build_user_content_preference_from_train(user_id)
                res = self.hybrid_recommender.content_recommender.recommend_for_preferences(
                    preferences=prefs,
                    top_k=max_k + len(train_items)
                )
                filtered_recs = [r for r in res if int(r["restaurant_id"]) not in train_items][:max_k]
                rec_ids = [int(r["restaurant_id"]) for r in filtered_recs]
                rec_dicts = filtered_recs

            elif model_name == "SVD (Collaborative)":
                if self.hybrid_recommender.collaborative_recommender is not None and self.hybrid_recommender.collaborative_recommender.is_known_user(user_id):
                    res = self.hybrid_recommender.collaborative_recommender.recommend_for_user(
                        user_id=user_id,
                        top_k=max_k,
                        exclude_rated=True
                    )
                    rec_ids = [int(r["restaurant_id"]) for r in res]
                    rec_dicts = res
                else:
                    rec_ids = []
                    rec_dicts = []

            elif model_name == "Hybrid":
                prefs = self._build_user_content_preference_from_train(user_id)
                res = self.hybrid_recommender.recommend(
                    user_id=user_id,
                    preferences=prefs,
                    mmr_enabled=False,
                    top_k=max_k
                )
                rec_ids = [int(r["restaurant_id"]) for r in res["recommendations"]]
                rec_dicts = res["recommendations"]

            elif model_name == "Hybrid + MMR (λ=0.75)":
                prefs = self._build_user_content_preference_from_train(user_id)
                res = self.hybrid_recommender.recommend(
                    user_id=user_id,
                    preferences=prefs,
                    mmr_enabled=True,
                    mmr_lambda=0.75,
                    top_k=max_k
                )
                rec_ids = [int(r["restaurant_id"]) for r in res["recommendations"]]
                rec_dicts = res["recommendations"]

            else:
                raise ValueError(f"Unknown model name: {model_name}")

            all_recommendations.append(rec_ids)
            all_rec_dicts.append(rec_dicts)

            # 2. Compute metrics for this user
            for k in k_values:
                user_metrics[user_id][f"precision@{k}"] = compute_precision_at_k(rec_ids, relevant_items, k)
                user_metrics[user_id][f"recall@{k}"] = compute_recall_at_k(rec_ids, relevant_items, k)
                user_metrics[user_id][f"hit_rate@{k}"] = compute_hit_rate_at_k(rec_ids, relevant_items, k)
                user_metrics[user_id][f"ndcg@{k}"] = compute_ndcg_at_k(rec_ids, relevant_items, k)
                user_metrics[user_id][f"mrr@{k}"] = compute_mrr_at_k(rec_ids, relevant_items, k)
                user_metrics[user_id][f"map@{k}"] = compute_map_at_k(rec_ids, relevant_items, k)

        # Aggregate across users
        agg_metrics: Dict[str, float] = {"model": model_name, "test_users_count": len(eval_users)}
        for k in k_values:
            for metric in ["precision", "recall", "hit_rate", "ndcg", "mrr", "map"]:
                vals = [user_metrics[u][f"{metric}@{k}"] for u in eval_users]
                mean_val, low_ci, high_ci = compute_bootstrap_ci(vals, random_state=self.random_state)
                agg_metrics[f"{metric}@{k}"] = round(mean_val, 4)
                agg_metrics[f"{metric}@{k}_ci_low"] = round(low_ci, 4)
                agg_metrics[f"{metric}@{k}_ci_high"] = round(high_ci, 4)

            agg_metrics[f"catalog_coverage@{k}"] = round(
                compute_catalog_coverage(all_recommendations, len(self.df_catalog), k), 4
            )
            agg_metrics[f"user_coverage@{k}"] = round(
                sum(1 for recs in all_recommendations if len(recs) >= k) / len(eval_users), 4
            )

        # Diversity analysis on top-10 lists
        ild_vals = []
        redundancy_vals = []
        cuisine_ratios = []
        similarity_engine = self.hybrid_recommender.similarity_engine

        for r_list in all_rec_dicts:
            if r_list:
                m_div = DiversityMetricsCalculator.compute_list_metrics(r_list[:10], similarity_engine)
                ild_vals.append(m_div["intra_list_diversity"])
                redundancy_vals.append(m_div["redundancy_rate"])
                cuisine_ratios.append(m_div["unique_cuisine_ratio"])

        agg_metrics["intra_list_diversity@10"] = round(float(np.mean(ild_vals)), 4) if ild_vals else 0.0
        agg_metrics["redundancy_rate@10"] = round(float(np.mean(redundancy_vals)), 4) if redundancy_vals else 0.0
        agg_metrics["unique_cuisine_ratio@10"] = round(float(np.mean(cuisine_ratios)), 4) if cuisine_ratios else 0.0

        return agg_metrics

    def run_full_benchmark(
        self,
        k_values: List[int] = [5, 10, 20],
        max_eval_users: int = 150
    ) -> pd.DataFrame:
        """
        Runs comprehensive benchmark across all 5 core recommendation strategies.
        """
        eval_users = self.test_users[:max_eval_users]
        models = [
            "Popularity",
            "Content-Based",
            "SVD (Collaborative)",
            "Hybrid",
            "Hybrid + MMR (λ=0.75)"
        ]

        results = []
        for model_name in models:
            metrics = self.evaluate_model(model_name=model_name, k_values=k_values, user_subset=eval_users)
            results.append(metrics)

        return pd.DataFrame(results)
