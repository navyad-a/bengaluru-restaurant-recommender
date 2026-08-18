# -*- coding: utf-8 -*-
"""
Bayesian Quality Shrinkage Module
=================================
Calculates Bayesian weighted ratings to prevent low-sample rating anomalies
and penalize unproven extreme reviews.

Mathematical Formulation (IMDb/Bayesian Weighted Rating):
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    
Where:
    - R: Restaurant's raw average rating
    - v: Restaurant's review count / votes
    - C: Global catalog mean rating (prior expectation)
    - m: Minimum votes threshold (regularization strength parameter)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class BayesianQualityScorer:
    """
    Computes regularized quality scores using Bayesian shrinkage.
    """

    def __init__(
        self,
        global_mean: float = 3.626,
        min_votes_threshold: float = 50.0
    ):
        self.global_mean = float(global_mean)
        self.min_votes_threshold = float(min_votes_threshold)

    @classmethod
    def from_dataframe(
        cls,
        df_restaurants: pd.DataFrame,
        min_votes_percentile: float = 50.0
    ) -> "BayesianQualityScorer":
        """
        Derives global mean (C) and minimum votes threshold (m) directly from catalog.
        """
        rated = df_restaurants[df_restaurants["rating"].notna()]
        if rated.empty:
            c = 3.626
            m = 50.0
        else:
            c = float(rated["rating"].mean())
            m = float(np.percentile(rated["review_count"].fillna(0), min_votes_percentile))
            m = max(10.0, m)
            
        return cls(global_mean=c, min_votes_threshold=m)

    def calculate_weighted_rating(
        self,
        rating: Optional[float],
        review_count: Optional[int]
    ) -> float:
        """
        Calculates the Bayesian weighted rating in range [1.0, 5.0].
        If rating is missing (NaN), falls back gracefully to the catalog prior C.
        """
        if rating is None or pd.isna(rating):
            return self.global_mean

        try:
            r = float(rating)
            v = max(0.0, float(review_count or 0))
            m = self.min_votes_threshold
            c = self.global_mean

            if v + m <= 0:
                return c

            wr = (v / (v + m)) * r + (m / (v + m)) * c
            return max(1.0, min(5.0, wr))
        except (ValueError, TypeError):
            return self.global_mean

    def score(
        self,
        rating: Optional[float],
        review_count: Optional[int]
    ) -> float:
        """
        Calculates normalized Bayesian quality score in [0.0, 1.0].
        Maps [1.0, 5.0] -> [0.0, 1.0].
        """
        wr = self.calculate_weighted_rating(rating, review_count)
        norm_score = (wr - 1.0) / 4.0
        return max(0.0, min(1.0, round(float(norm_score), 4)))

    def score_series(
        self,
        ratings: pd.Series,
        review_counts: pd.Series
    ) -> np.ndarray:
        """
        Vectorized quality score computation for high performance.
        """
        r_arr = ratings.fillna(self.global_mean).to_numpy(dtype=np.float64, copy=True)
        v_arr = np.maximum(0.0, review_counts.fillna(0).to_numpy(dtype=np.float64, copy=True))

        m = self.min_votes_threshold
        c = self.global_mean

        wr_arr = (v_arr / (v_arr + m)) * r_arr + (m / (v_arr + m)) * c
        wr_arr = np.clip(wr_arr, 1.0, 5.0)
        
        norm_arr = (wr_arr - 1.0) / 4.0
        return np.clip(norm_arr, 0.0, 1.0)
