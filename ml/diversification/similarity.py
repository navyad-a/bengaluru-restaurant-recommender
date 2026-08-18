# -*- coding: utf-8 -*-
"""
Sparse Item Similarity Engine for MMR Diversification
=====================================================
Computes exact on-demand cosine similarity between candidate restaurants and already-selected
sets using memory-efficient sparse CSR matrix dot-products, strictly avoiding dense N x N allocation.
"""

import os
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from typing import Dict, List, Optional, Union


class SparseSimilarityEngine:
    """
    Computes pairwise and set-wise cosine similarities on-the-fly using sparse TF-IDF vectors.
    """

    def __init__(
        self,
        tfidf_matrix: sp.csr_matrix,
        id_to_idx: Optional[Dict[int, int]] = None,
        idx_to_id: Optional[Dict[int, int]] = None,
        restaurant_id_to_idx: Optional[Dict[int, int]] = None,
        idx_to_restaurant_id: Optional[Dict[int, int]] = None
    ):
        self.tfidf_matrix = sp.csr_matrix(tfidf_matrix)
        self.id_to_idx = id_to_idx or restaurant_id_to_idx or {}
        self.idx_to_id = idx_to_id or idx_to_restaurant_id or {}
        # Ensure row L2 normalization so sparse dot-product equals cosine similarity
        self.normed_tfidf = normalize(self.tfidf_matrix, norm="l2", axis=1, copy=True)

    @classmethod
    def from_content_artifacts(cls, artifact_dir: str) -> "SparseSimilarityEngine":
        """
        Loads precomputed sparse TF-IDF matrix and index mappings from content model artifacts.
        """
        matrix_path = os.path.join(artifact_dir, "tfidf_matrix.joblib")
        mappings_path = os.path.join(artifact_dir, "restaurant_mappings.joblib")

        if not os.path.exists(matrix_path) or not os.path.exists(mappings_path):
            raise FileNotFoundError(
                f"Missing content artifacts in {artifact_dir}. Ensure Phase 5 models are trained."
            )

        tfidf_matrix = joblib.load(matrix_path)
        mappings = joblib.load(mappings_path)
        
        return cls(
            tfidf_matrix=tfidf_matrix,
            id_to_idx=mappings.get("id_to_idx", mappings.get("restaurant_id_to_idx", {})),
            idx_to_id=mappings.get("idx_to_id", mappings.get("idx_to_restaurant_id", {}))
        )

    def compute_pairwise_similarity(self, id_a: int, id_b: int) -> float:
        """
        Computes cosine similarity between two restaurants.
        """
        if id_a == id_b:
            return 1.0
        if id_a not in self.id_to_idx or id_b not in self.id_to_idx:
            return 0.0

        idx_a = self.id_to_idx[id_a]
        idx_b = self.id_to_idx[id_b]

        vec_a = self.normed_tfidf[idx_a]
        vec_b = self.normed_tfidf[idx_b]

        sim = vec_a.dot(vec_b.T).toarray()[0, 0]
        return float(np.clip(sim, 0.0, 1.0))

    def compute_max_similarity_to_set(
        self,
        candidate_id: int,
        selected_ids: List[int]
    ) -> float:
        """
        Computes max_{j in S} Similarity(candidate, j) using a single sparse slice dot product.
        Time complexity: O(|S| * non_zeros) with zero dense N x N matrix allocation.
        """
        if not selected_ids:
            return 0.0
        if candidate_id not in self.id_to_idx:
            return 0.0

        cand_idx = self.id_to_idx[candidate_id]
        cand_vec = self.normed_tfidf[cand_idx]  # Shape: (1, V)

        selected_indices = [self.id_to_idx[s_id] for s_id in selected_ids if s_id in self.id_to_idx]
        if not selected_indices:
            return 0.0

        selected_matrix = self.normed_tfidf[selected_indices]  # Shape: (|S|, V)
        sims = cand_vec.dot(selected_matrix.T).toarray().ravel()

        return float(np.clip(np.max(sims), 0.0, 1.0))

    def compute_similarity_matrix_for_ids(
        self,
        item_ids: List[int]
    ) -> np.ndarray:
        """
        Computes pairwise similarity submatrix solely for the provided item IDs (e.g. K x K for top-K).
        """
        k = len(item_ids)
        if k == 0:
            return np.empty((0, 0), dtype=np.float64)

        valid_indices = []
        valid_pos = []
        for i, item_id in enumerate(item_ids):
            if item_id in self.id_to_idx:
                valid_indices.append(self.id_to_idx[item_id])
                valid_pos.append(i)

        sim_matrix = np.zeros((k, k), dtype=np.float64)
        np.fill_diagonal(sim_matrix, 1.0)

        if len(valid_indices) > 0:
            sub_tfidf = self.normed_tfidf[valid_indices]
            sub_sims = (sub_tfidf @ sub_tfidf.T).toarray()
            for r_idx, orig_r in enumerate(valid_pos):
                for c_idx, orig_c in enumerate(valid_pos):
                    sim_matrix[orig_r, orig_c] = float(np.clip(sub_sims[r_idx, c_idx], 0.0, 1.0))

        return sim_matrix
