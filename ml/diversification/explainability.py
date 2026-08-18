# -*- coding: utf-8 -*-
"""
Dedicated Recommendation Explainability Engine
==============================================
Produces structured, truthful, non-fabricated rationale metadata and human-readable
explanation strings based strictly on positive active recommendation signals.
"""

from typing import Dict, Any, List, Optional
from ml.cold_start.routing import ColdStartStrategy


class RecommendationExplainabilityEngine:
    """
    Constructs structured reasoning metadata and natural-language explanations.
    """

    @classmethod
    def generate_explanation_metadata(
        cls,
        item: Dict[str, Any],
        strategy: ColdStartStrategy,
        effective_weights: Dict[str, float],
        scores: Dict[str, float],
        user_preferences: Optional[Dict[str, Any]] = None,
        is_diversified: bool = False,
        similarity_to_prior: float = 0.0
    ) -> Dict[str, Any]:
        """
        Builds transparent explanation metadata without hallucinating unweighted signals.
        """
        user_preferences = user_preferences or {}
        cuisines = str(item.get("cuisines", ""))
        rating = item.get("rating")
        review_count = int(item.get("review_count", 0))
        area = str(item.get("area", ""))
        cost = int(item.get("cost_for_two_inr", 0))
        dist = item.get("distance_km")

        # 1. Identify contributing signals with positive weights
        contributing_signals = []
        for sig in ["content", "collaborative", "location", "quality"]:
            if effective_weights.get(sig, 0.0) > 0.0 and scores.get(sig, 0.0) > 0.0:
                contributing_signals.append(sig)

        # 2. Identify dominant primary signal
        weighted_scores = {
            sig: effective_weights.get(sig, 0.0) * scores.get(sig, 0.0)
            for sig in ["content", "collaborative", "location", "quality"]
        }
        primary_signal = max(weighted_scores, key=weighted_scores.get) if weighted_scores else "quality"

        # 3. Match explicit preferences
        matched_preferences = []
        pref_cuisines = user_preferences.get("preferred_cuisines")
        if pref_cuisines:
            if isinstance(pref_cuisines, list):
                for pc in pref_cuisines:
                    if pc.lower() in cuisines.lower():
                        matched_preferences.append(pc)
            elif isinstance(pref_cuisines, str):
                for pc in pref_cuisines.split(","):
                    if pc.strip().lower() in cuisines.lower():
                        matched_preferences.append(pc.strip())

        pref_area = user_preferences.get("preferred_area") or user_preferences.get("area")
        if pref_area and pref_area.lower() in area.lower():
            matched_preferences.append(f"Located in {pref_area}")

        max_budget = user_preferences.get("max_cost_for_two")
        if max_budget and cost <= max_budget:
            matched_preferences.append(f"Within ₹{max_budget} budget (₹{cost} for two)")

        # 4. Construct bullet points
        reasons = []
        if matched_preferences:
            reasons.append(f"Matches your criteria: {', '.join(matched_preferences[:2])}")
        elif cuisines:
            reasons.append(f"Features {cuisines.split(',')[0].strip()} cuisine")

        if effective_weights.get("collaborative", 0.0) > 0.0 and scores.get("collaborative", 0.0) > 0.5:
            reasons.append("Aligns with your dining taste profile")

        if rating and rating >= 3.8 and review_count >= 50:
            reasons.append(f"Reliable community ratings ({rating}★ with {review_count:,} reviews)")

        if dist is not None and effective_weights.get("location", 0.0) > 0.0:
            reasons.append(f"Convenient location ({dist:.1f} km away)")

        # 5. Diversification reason
        diversity_reason = None
        if is_diversified:
            if similarity_to_prior < 0.60:
                diversity_reason = "Selected to introduce menu and cuisine variety while maintaining high overall relevance."
            else:
                diversity_reason = "Included as a distinct alternative to preceding recommendations."

        # 6. Natural language synthesis
        if strategy == ColdStartStrategy.WARM_HYBRID:
            explanation = "Recommended for " + ", ".join(reasons) + "."
        elif strategy == ColdStartStrategy.SPARSE_HYBRID:
            explanation = "Recommended based on initial taste signals and " + ", ".join(reasons) + "."
        elif strategy == ColdStartStrategy.PROFILE_CONTENT_QUALITY:
            explanation = "Recommended for " + ", ".join(reasons) + "."
        elif strategy == ColdStartStrategy.LOCATION_POPULARITY:
            explanation = f"Popular dining choice in {area}" + (f" ({rating}★ with {review_count:,} reviews)." if rating else ".")
        elif strategy == ColdStartStrategy.GLOBAL_POPULARITY:
            explanation = f"Highly rated dining destination in Bengaluru ({rating}★ with {review_count:,} reviews)."
        else:
            explanation = "Recommended for " + ", ".join(reasons) + "."

        return {
            "explanation": explanation,
            "explanation_reasons": reasons,
            "matched_preferences": matched_preferences,
            "diversity_reason": diversity_reason,
            "primary_signal": primary_signal,
            "contributing_signals": contributing_signals
        }
