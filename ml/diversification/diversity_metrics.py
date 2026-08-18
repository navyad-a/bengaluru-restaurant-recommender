# -*- coding: utf-8 -*-
"""
Recommendation Set Diversity Metrics
====================================
Computes formal diversity, redundancy, and intra-list metrics for top-K recommendation lists.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from ml.diversification.similarity import SparseSimilarityEngine


class DiversityMetricsCalculator:
    """
    Evaluates top-K recommendation set diversity and relevance retention.
    """

    @staticmethod
    def compute_list_metrics(
        recommendations: List[Dict[str, Any]],
        similarity_engine: SparseSimilarityEngine,
        pre_mmr_relevance: Optional[List[float]] = None,
        redundancy_similarity_threshold: float = 0.80
    ) -> Dict[str, Any]:
        """
        Calculates all standard diversification metrics for a top-K recommendation list.
        """
        k = len(recommendations)
        if k == 0:
            return {
                "top_k": 0,
                "unique_cuisine_ratio": 0.0,
                "unique_restaurant_type_ratio": 0.0,
                "unique_locality_ratio": 0.0,
                "avg_pairwise_similarity": 0.0,
                "intra_list_diversity": 1.0,
                "redundancy_rate": 0.0,
                "mean_relevance": 0.0,
                "relevance_retention_pct": 100.0,
                "relevance_drop_pct": 0.0
            }

        # 1. Attribute diversity ratios
        all_cuisines = set()
        for r in recommendations:
            c_str = str(r.get("cuisines", ""))
            for c in c_str.split(","):
                c_clean = c.strip().lower()
                if c_clean:
                    all_cuisines.add(c_clean)

        unique_types = {str(r.get("restaurant_type", "")).strip().lower() for r in recommendations if r.get("restaurant_type")}
        unique_localities = {str(r.get("area", "")).strip().lower() for r in recommendations if r.get("area")}

        cuisine_ratio = round(len(all_cuisines) / k, 4)
        type_ratio = round(len(unique_types) / k, 4)
        locality_ratio = round(len(unique_localities) / k, 4)

        # 2. Pairwise Cosine Similarities via sparse submatrix
        item_ids = [int(r["restaurant_id"]) for r in recommendations]
        sim_matrix = similarity_engine.compute_similarity_matrix_for_ids(item_ids)

        if k > 1:
            # Extract upper triangle without diagonal
            upper_indices = np.triu_indices(k, k=1)
            pairwise_sims = sim_matrix[upper_indices]
            avg_pairwise_sim = float(np.mean(pairwise_sims))
            redundancy_count = int(np.sum(pairwise_sims >= redundancy_similarity_threshold))
            total_pairs = len(pairwise_sims)
            redundancy_rate = float(redundancy_count / total_pairs) if total_pairs > 0 else 0.0
        else:
            avg_pairwise_sim = 0.0
            redundancy_rate = 0.0

        ild = float(np.clip(1.0 - avg_pairwise_sim, 0.0, 1.0))

        # 3. Relevance retention
        post_relevance = [
            float(r.get("hybrid_score", r.get("content_score", r.get("popularity_score", 0.5))))
            for r in recommendations
        ]
        mean_post_rel = float(np.mean(post_relevance)) if post_relevance else 0.0

        if pre_mmr_relevance and len(pre_mmr_relevance) > 0:
            mean_pre_rel = float(np.mean(pre_mmr_relevance[:k]))
            if mean_pre_rel > 0:
                retention_pct = float(np.clip((mean_post_rel / mean_pre_rel) * 100.0, 0.0, 100.0))
            else:
                retention_pct = 100.0
            drop_pct = round(100.0 - retention_pct, 2)
        else:
            retention_pct = 100.0
            drop_pct = 0.0

        return {
            "top_k": k,
            "unique_cuisine_ratio": cuisine_ratio,
            "unique_restaurant_type_ratio": type_ratio,
            "unique_locality_ratio": locality_ratio,
            "avg_pairwise_similarity": round(avg_pairwise_sim, 4),
            "intra_list_diversity": round(ild, 4),
            "redundancy_rate": round(redundancy_rate, 4),
            "mean_relevance": round(mean_post_rel, 4),
            "relevance_retention_pct": round(retention_pct, 2),
            "relevance_drop_pct": drop_pct
        }
