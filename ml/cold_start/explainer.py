# -*- coding: utf-8 -*-
"""
Cold-Start Explainability & Reasoning Module
============================================
Generates transparent, contextual explanation strings according to the
active cold-start strategy and recommendation signals.
"""

from typing import Dict, Any, Optional
from ml.cold_start.routing import ColdStartStrategy


class ColdStartExplainer:
    """
    Produces honest, non-fabricated rationale strings for cold-start and hybrid recommendations.
    """

    @staticmethod
    def generate_explanation(
        strategy: ColdStartStrategy,
        item: Dict[str, Any],
        scores: Optional[Dict[str, float]] = None,
        area_requested: Optional[str] = None
    ) -> str:
        """
        Constructs an explainability string tailored to the routing strategy.
        """
        scores = scores or {}
        name = item.get("name", "Restaurant")
        cuisines = item.get("cuisines", "popular dishes")
        rating = item.get("rating")
        review_count = item.get("review_count", 0)
        area = item.get("area", "Bengaluru")
        dist = item.get("distance_km")

        if strategy == ColdStartStrategy.WARM_HYBRID:
            reasons = ["high taste alignment with your dining history", f"matching cuisine ({cuisines})"]
            if rating and review_count > 50:
                reasons.append(f"proven customer reviews ({rating}★ with {review_count} votes)")
            if dist is not None:
                reasons.append(f"close proximity ({dist:.1f} km away)")
            return "Recommended based on " + ", ".join(reasons) + "."

        elif strategy == ColdStartStrategy.SPARSE_HYBRID:
            reasons = ["early taste signals from your ratings", f"matching cuisine ({cuisines})"]
            if rating and review_count > 50:
                reasons.append(f"solid community rating ({rating}★)")
            return "Recommended based on " + ", ".join(reasons) + "."

        elif strategy == ColdStartStrategy.PROFILE_CONTENT_QUALITY:
            reasons = [f"matching your preferred cuisines ({cuisines})"]
            if rating and review_count > 50:
                reasons.append(f"strong community reviews ({rating}★ with {review_count} votes)")
            if dist is not None:
                reasons.append(f"nearby in {area} ({dist:.1f} km away)")
            return "Recommended for " + ", ".join(reasons) + "."

        elif strategy == ColdStartStrategy.LOCATION_POPULARITY:
            loc_name = area_requested or area
            if rating:
                return f"Top trending & popular choice in {loc_name} ({rating}★ with {review_count} community reviews)."
            return f"Popular local favorite in {loc_name}."

        elif strategy == ColdStartStrategy.GLOBAL_POPULARITY:
            if rating:
                return f"Highly rated and most popular dining institution in Bengaluru ({rating}★ with {review_count:,} reviews)."
            return "Widely popular dining choice in Bengaluru."

        return f"Recommended for {cuisines} in {area}."
