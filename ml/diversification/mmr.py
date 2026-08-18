# -*- coding: utf-8 -*-
"""
Maximal Marginal Relevance (MMR) Diversification Engine
# Production default lambda parameter: 0.75 for Pareto-optimal relevance-diversity trade-off
========================================================
Implements greedy, deterministic MMR diversification over candidate recommendation pools
using sparse TF-IDF cosine similarity, duplicate suppression, and soft constraint controls.

Mathematical Formulation:
    MMR(i) = lambda * Relevance(i) - (1 - lambda) * max_{j in S} Similarity(i, j)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set

from ml.diversification.similarity import SparseSimilarityEngine
from ml.diversification.redundancy import RedundancyChecker
from ml.diversification.diversity_metrics import DiversityMetricsCalculator


class MMRDiversifier:
    """
    Greedy deterministic MMR diversification engine.
    """

    DEFAULT_LAMBDA: float = 0.75

    def __init__(
        self,
        similarity_engine: SparseSimilarityEngine,
        redundancy_checker: Optional[RedundancyChecker] = None
    ):
        self.similarity_engine = similarity_engine
        self.redundancy_checker = redundancy_checker or RedundancyChecker()

    def diversify(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        lambda_param: float = 0.75,
        relevance_field: str = "hybrid_score"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Selects top-k diverse items from candidates using greedy MMR.
        
        Returns:
            (diversified_recommendations, diversity_metrics_dict)
        """
        # Validate lambda parameter
        if not (0.0 <= lambda_param <= 1.0):
            raise ValueError(f"lambda_param must be in [0.0, 1.0], got {lambda_param}")

        if not candidates or top_k <= 0:
            metrics = DiversityMetricsCalculator.compute_list_metrics([], self.similarity_engine)
            return [], metrics

        # Keep track of original pre-MMR relevance scores for evaluation
        pre_mmr_relevance = [float(c.get(relevance_field, 0.5)) for c in candidates]

        # Candidate pool working list
        remaining = list(candidates)
        selected: List[Dict[str, Any]] = []
        selected_ids: List[int] = []

        while len(selected) < top_k and remaining:
            best_cand = None
            best_idx = -1
            best_mmr_score = -float("inf")
            best_max_sim = 0.0

            # Evaluate each remaining candidate
            for idx, cand in enumerate(remaining):
                cand_id = int(cand["restaurant_id"])
                relevance = float(cand.get(relevance_field, 0.5))

                if not selected_ids:
                    # First item has no redundancy penalty
                    mmr_score = lambda_param * relevance
                    max_sim = 0.0
                else:
                    max_sim = self.similarity_engine.compute_max_similarity_to_set(
                        candidate_id=cand_id,
                        selected_ids=selected_ids
                    )
                    mmr_score = (lambda_param * relevance) - ((1.0 - lambda_param) * max_sim)

                # Soft redundancy check (penalize if violating near-duplicate or chain limits)
                is_near_dup = self.redundancy_checker.is_near_duplicate(cand, selected, max_sim)
                violates_chain = self.redundancy_checker.violates_soft_chain_limit(cand, selected)

                if is_near_dup:
                    mmr_score -= 0.50  # Strong penalty for near duplicates
                elif violates_chain:
                    mmr_score -= 0.20  # Mild penalty for exceeding chain limit

                # Deterministic comparison tuple
                cand_key = (
                    mmr_score,
                    relevance,
                    float(cand.get("hybrid_score", 0.0)),
                    float(cand.get("quality_score", 0.0)),
                    int(cand.get("review_count", 0)),
                    float(cand.get("rating") or 0.0),
                    -cand_id  # Ascending ID
                )

                best_key = (
                    best_mmr_score,
                    float(best_cand.get(relevance_field, 0.0)) if best_cand else 0.0,
                    float(best_cand.get("hybrid_score", 0.0)) if best_cand else 0.0,
                    float(best_cand.get("quality_score", 0.0)) if best_cand else 0.0,
                    int(best_cand.get("review_count", 0)) if best_cand else 0,
                    float(best_cand.get("rating") or 0.0) if best_cand else 0.0,
                    -int(best_cand["restaurant_id"]) if best_cand else 0
                )

                if best_cand is None or cand_key > best_key:
                    best_cand = cand
                    best_idx = idx
                    best_mmr_score = mmr_score
                    best_max_sim = max_sim

            if best_cand is not None and best_idx >= 0:
                selected_item = dict(best_cand)
                selected_item["mmr_score"] = round(float(best_mmr_score), 4)
                selected_item["max_similarity_to_prior"] = round(float(best_max_sim), 4)
                selected_item["is_diversified"] = bool(len(selected) > 0 and best_max_sim < 0.85)

                selected.append(selected_item)
                selected_ids.append(int(best_cand["restaurant_id"]))
                remaining.pop(best_idx)
            else:
                break

        # Compute full diversity metrics
        metrics = DiversityMetricsCalculator.compute_list_metrics(
            recommendations=selected,
            similarity_engine=self.similarity_engine,
            pre_mmr_relevance=pre_mmr_relevance
        )

        return selected, metrics
