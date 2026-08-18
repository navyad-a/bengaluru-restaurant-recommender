# -*- coding: utf-8 -*-
"""
Bayesian Popularity Engine
==========================
Computes robust, log-volume-weighted Bayesian popularity priors for global,
locality-specific, and cuisine-specific cold-start recommendations.

Mathematical Formulation:
    S_pop = alpha * S_quality + (1 - alpha) * (log(1 + v) / log(1 + v_max))
    
Where:
    - S_quality: Bayesian regularized quality score in [0.0, 1.0]
    - v: Restaurant review count (votes)
    - v_max: Maximum review count in the catalog (or locality partition)
    - alpha: Weight balancing quality and volume (default: 0.60 quality, 0.40 volume)
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class BayesianPopularityEngine:
    """
    Computes global, locality-level, and cuisine-level popularity rankings.
    """

    def __init__(
        self,
        df_restaurants: pd.DataFrame,
        quality_scorer: Optional[Any] = None,
        alpha: float = 0.60
    ):
        self.df_restaurants = df_restaurants
        if quality_scorer is None:
            from ml.hybrid.quality import BayesianQualityScorer
            self.quality_scorer = BayesianQualityScorer.from_dataframe(df_restaurants)
        else:
            self.quality_scorer = quality_scorer
            
        self.alpha = float(alpha)
        self._precompute_popularity()

    def _precompute_popularity(self):
        """
        Precomputes normalized popularity scores across the entire 12,481 catalog.
        """
        # 1. Quality scores
        quality_scores = self.quality_scorer.score_series(
            ratings=self.df_restaurants["rating"],
            review_counts=self.df_restaurants["review_count"]
        )

        # 2. Log-scaled review volumes
        votes = np.maximum(0.0, self.df_restaurants["review_count"].fillna(0).to_numpy(dtype=np.float64))
        max_votes = np.max(votes) if len(votes) > 0 and np.max(votes) > 0 else 1.0
        log_votes = np.log1p(votes) / np.log1p(max_votes)

        # 3. Combined popularity score
        pop_scores = self.alpha * quality_scores + (1.0 - self.alpha) * log_votes
        self.df_restaurants = self.df_restaurants.copy()
        self.df_restaurants["popularity_score"] = np.round(np.clip(pop_scores, 0.0, 1.0), 4)

    def _apply_hard_filters(
        self,
        df: pd.DataFrame,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Applies hard filters to candidates.
        """
        if filters is None or df.empty:
            return df

        mask = pd.Series(True, index=df.index)

        max_cost = filters.get("max_cost_for_two")
        if max_cost is not None and max_cost > 0:
            mask &= (df["cost_for_two_inr"] <= max_cost)

        min_rating = filters.get("min_rating")
        if min_rating is not None and min_rating > 0.0:
            mask &= (df["rating"].notna() & (df["rating"] >= min_rating))

        price_tier = filters.get("price_tier") or filters.get("preferred_price_tier")
        if price_tier and isinstance(price_tier, str) and price_tier.strip():
            mask &= (df["price_tier"].str.lower() == price_tier.strip().lower())

        area = filters.get("area") or filters.get("preferred_area") or filters.get("location")
        if area and isinstance(area, str) and area.strip():
            mask &= (df["area"].str.lower() == area.strip().lower())

        if filters.get("online_order_only") is True:
            mask &= (df["online_order"] == True)

        if filters.get("book_table_only") is True:
            mask &= (df["book_table"] == True)

        return df[mask].copy()

    def get_global_popular(
        self,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns globally most popular & reliable restaurants in Bengaluru.
        """
        df_filtered = self._apply_hard_filters(self.df_restaurants, filters)
        
        # Deterministic sorting
        df_sorted = df_filtered.sort_values(
            by=["popularity_score", "review_count", "rating", "restaurant_id"],
            ascending=[False, False, False, True]
        ).head(top_k)

        return df_sorted.to_dict(orient="records")

    def get_locality_popular(
        self,
        area: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns most popular restaurants in a specific locality (e.g. Koramangala, Indiranagar).
        """
        loc_mask = self.df_restaurants["area"].str.lower() == area.strip().lower()
        df_loc = self.df_restaurants[loc_mask].copy()

        if df_loc.empty:
            # Fallback to global popular with area filter
            return self.get_global_popular(top_k=top_k, filters=filters)

        df_filtered = self._apply_hard_filters(df_loc, filters)

        df_sorted = df_filtered.sort_values(
            by=["popularity_score", "review_count", "rating", "restaurant_id"],
            ascending=[False, False, False, True]
        ).head(top_k)

        return df_sorted.to_dict(orient="records")

    def get_cuisine_popular(
        self,
        cuisine: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns most popular restaurants serving a specific regional cuisine.
        """
        c_mask = self.df_restaurants["cuisines"].str.contains(cuisine.strip(), case=False, na=False)
        df_c = self.df_restaurants[c_mask].copy()

        if df_c.empty:
            return self.get_global_popular(top_k=top_k, filters=filters)

        df_filtered = self._apply_hard_filters(df_c, filters)

        df_sorted = df_filtered.sort_values(
            by=["popularity_score", "review_count", "rating", "restaurant_id"],
            ascending=[False, False, False, True]
        ).head(top_k)

        return df_sorted.to_dict(orient="records")
