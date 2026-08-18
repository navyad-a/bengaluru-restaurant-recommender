# -*- coding: utf-8 -*-
"""
Redundancy & Near-Duplicate Detection Module
============================================
Provides normalized string matching, chain outlet grouping, and soft redundancy
controls to suppress excessive duplication in top-K recommendations.
"""

import re
from typing import Dict, Any, List, Optional, Set


def normalize_restaurant_name(name: str) -> str:
    """
    Normalizes restaurant names by removing location suffixes, punctuation, and extra whitespace.
    Example: 'Meghana Foods - Koramangala 5th Block' -> 'meghana foods'
    """
    if not name:
        return ""
    # Strip common delimiter suffixes like '-', ':', '@', ','
    clean = re.split(r"[-:@,]", name)[0].strip().lower()
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


class RedundancyChecker:
    """
    Evaluates near-duplicates and enforces soft chain/locality limits in top-K selection.
    """

    def __init__(
        self,
        max_same_chain_in_top_k: int = 2,
        max_same_locality_in_top_k: Optional[int] = None,
        max_similarity_threshold: float = 0.95
    ):
        self.max_same_chain_in_top_k = int(max_same_chain_in_top_k)
        self.max_same_locality_in_top_k = (
            int(max_same_locality_in_top_k) if max_same_locality_in_top_k is not None else None
        )
        self.max_similarity_threshold = float(max_similarity_threshold)

    def is_near_duplicate(
        self,
        candidate: Dict[str, Any],
        selected_items: List[Dict[str, Any]],
        max_similarity_to_selected: float
    ) -> bool:
        """
        Detects if candidate is an extreme near-duplicate (> threshold TF-IDF similarity and identical base name).
        """
        if not selected_items:
            return False

        cand_name_norm = normalize_restaurant_name(candidate.get("name", ""))
        for item in selected_items:
            # Identical ID
            if candidate.get("restaurant_id") == item.get("restaurant_id"):
                return True
            # Identical normalized name AND extremely high similarity
            item_name_norm = normalize_restaurant_name(item.get("name", ""))
            if cand_name_norm == item_name_norm and max_similarity_to_selected >= self.max_similarity_threshold:
                return True

        return False

    def violates_soft_chain_limit(
        self,
        candidate: Dict[str, Any],
        selected_items: List[Dict[str, Any]]
    ) -> bool:
        """
        Checks if adding this outlet exceeds the allowed count for the same restaurant chain in top-K.
        """
        if self.max_same_chain_in_top_k <= 0 or not selected_items:
            return False

        cand_chain = normalize_restaurant_name(candidate.get("name", ""))
        count = sum(
            1 for item in selected_items
            if normalize_restaurant_name(item.get("name", "")) == cand_chain
        )
        return count >= self.max_same_chain_in_top_k

    def violates_soft_locality_limit(
        self,
        candidate: Dict[str, Any],
        selected_items: List[Dict[str, Any]]
    ) -> bool:
        """
        Checks if adding this outlet exceeds the allowed count for the same locality in top-K.
        """
        if self.max_same_locality_in_top_k is None or self.max_same_locality_in_top_k <= 0 or not selected_items:
            return False

        cand_area = str(candidate.get("area", "")).lower().strip()
        if not cand_area:
            return False

        count = sum(
            1 for item in selected_items
            if str(item.get("area", "")).lower().strip() == cand_area
        )
        return count >= self.max_same_locality_in_top_k
