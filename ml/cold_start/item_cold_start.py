# -*- coding: utf-8 -*-
"""
Item Cold-Start & Unrated Restaurant Imputation Module
======================================================
Handles unrated, newly added, and low-review restaurants by imputing baseline
quality priors from locality/cuisine clusters, preventing cold-start starvation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class ItemColdStartHandler:
    """
    Imputes priors and manages exploratory exposure for newly listed and unrated restaurants.
    """

    def __init__(self, df_restaurants: pd.DataFrame, global_mean: float = 3.626):
        self.df_restaurants = df_restaurants
        self.global_mean = float(global_mean)
        self._precompute_locality_priors()

    def _precompute_locality_priors(self):
        """
        Computes mean ratings per locality to use as localized priors for unrated restaurants.
        """
        rated = self.df_restaurants[self.df_restaurants["rating"].notna()]
        self.locality_priors = rated.groupby("area")["rating"].mean().to_dict()

    def is_cold_start_restaurant(self, restaurant_id: int) -> bool:
        """
        Checks if a restaurant has 0 reviews or missing rating.
        """
        row = self.df_restaurants[self.df_restaurants["restaurant_id"] == restaurant_id]
        if row.empty:
            return True
        r = row.iloc[0]["rating"]
        v = row.iloc[0]["review_count"]
        return bool(pd.isna(r) or v == 0)

    def impute_restaurant_rating(
        self,
        area: Optional[str] = None,
        cuisines: Optional[str] = None
    ) -> float:
        """
        Imputes a conservative baseline rating prior for an unrated restaurant.
        Priority: Locality mean -> Global catalog mean.
        """
        if area and area in self.locality_priors:
            return round(float(self.locality_priors[area]), 2)
        return self.global_mean

    def enrich_item_metadata(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches restaurant dictionary with cold-start and imputation flags.
        """
        enriched = dict(item_dict)
        r = enriched.get("rating")
        v = enriched.get("review_count", 0)

        if r is None or pd.isna(r) or v == 0:
            enriched["is_unrated"] = True
            enriched["imputed_rating_prior"] = self.impute_restaurant_rating(
                area=enriched.get("area"),
                cuisines=enriched.get("cuisines")
            )
        else:
            enriched["is_unrated"] = False
            enriched["imputed_rating_prior"] = None

        return enriched
