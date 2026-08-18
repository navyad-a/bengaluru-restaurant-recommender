# -*- coding: utf-8 -*-
"""
Cold-Start Strategy & Decision Routing Module
=============================================
Deterministically routes recommendation queries to the optimal strategy based on
user interaction maturity, preference profile completeness, and geographic context.
"""

from enum import Enum
from typing import Dict, Any, Optional, Tuple, Set


class ColdStartStrategy(str, Enum):
    """
    Cold-start recommendation routing taxonomy.
    """
    WARM_HYBRID = "warm_hybrid"                      # Known user with >= 5 ratings
    SPARSE_HYBRID = "sparse_hybrid"                  # Known user with 1-4 ratings (sparse profile)
    PROFILE_CONTENT_QUALITY = "profile_content"      # Unknown user with explicit preferences
    LOCATION_POPULARITY = "location_popularity"      # Unknown user with location but no preferences
    GLOBAL_POPULARITY = "global_popularity"          # Complete cold start (no ID, no prefs, no location)


class ColdStartRouter:
    """
    Classifies incoming requests and determines optimal fallback weights.
    """

    SPARSE_THRESHOLD: int = 5  # Users with < 5 interactions are classified as sparse

    @classmethod
    def determine_strategy(
        cls,
        user_id: Optional[int] = None,
        user_interaction_count: int = 0,
        has_preferences: bool = False,
        has_location: bool = False,
        is_known_collaborative_user: bool = False
    ) -> ColdStartStrategy:
        """
        Determines the cold-start recommendation strategy.
        """
        # Case 1: Known user in collaborative training benchmark
        if user_id is not None and is_known_collaborative_user:
            if user_interaction_count >= cls.SPARSE_THRESHOLD:
                return ColdStartStrategy.WARM_HYBRID
            else:
                return ColdStartStrategy.SPARSE_HYBRID

        # Case 2: User has explicit dining preferences (cuisines, price tier, etc.)
        if has_preferences:
            return ColdStartStrategy.PROFILE_CONTENT_QUALITY

        # Case 3: User provided coordinates or area but no explicit preferences
        if has_location:
            return ColdStartStrategy.LOCATION_POPULARITY

        # Case 4: Complete cold-start
        return ColdStartStrategy.GLOBAL_POPULARITY

    @classmethod
    def get_strategy_weights(
        cls,
        strategy: ColdStartStrategy,
        has_location: bool = False
    ) -> Dict[str, float]:
        """
        Returns the appropriate feature weighting scheme for the selected cold-start strategy.
        """
        if strategy == ColdStartStrategy.WARM_HYBRID:
            if has_location:
                return {"content": 0.40, "collaborative": 0.20, "location": 0.15, "quality": 0.25}
            else:
                return {"content": 0.4706, "collaborative": 0.2353, "location": 0.0, "quality": 0.2941}

        elif strategy == ColdStartStrategy.SPARSE_HYBRID:
            # Dampen collaborative signal from 0.20 -> 0.10 to prevent overfitting sparse ratings
            if has_location:
                return {"content": 0.45, "collaborative": 0.10, "location": 0.15, "quality": 0.30}
            else:
                return {"content": 0.5294, "collaborative": 0.1176, "location": 0.0, "quality": 0.3530}

        elif strategy == ColdStartStrategy.PROFILE_CONTENT_QUALITY:
            if has_location:
                return {"content": 0.50, "collaborative": 0.0, "location": 0.1875, "quality": 0.3125}
            else:
                return {"content": 0.6154, "collaborative": 0.0, "location": 0.0, "quality": 0.3846}

        elif strategy == ColdStartStrategy.LOCATION_POPULARITY:
            return {"content": 0.20, "collaborative": 0.0, "location": 0.40, "quality": 0.40}

        elif strategy == ColdStartStrategy.GLOBAL_POPULARITY:
            return {"content": 0.0, "collaborative": 0.0, "location": 0.0, "quality": 1.0}

        return {"content": 0.40, "collaborative": 0.20, "location": 0.15, "quality": 0.25}
