# -*- coding: utf-8 -*-
"""
Locality Spatial Analytics Module
=================================
Computes locality-level spatial statistics, pricing distributions,
and inter-locality distance matrices across Bengaluru neighborhoods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ml.spatial.distance import haversine_distance


class LocalitySpatialAnalytics:
    """
    Analyzes geographic and dining distributions across Bengaluru locality clusters.
    """

    def __init__(self, df_restaurants: pd.DataFrame):
        self.df_restaurants = df_restaurants

    def get_locality_summary(self, min_outlets: int = 10) -> pd.DataFrame:
        """
        Generates aggregate metrics by Bengaluru locality.
        """
        agg_dict = {
            "restaurant_id": "count",
            "rating": ["mean", "median"],
            "cost_for_two_inr": ["mean", "median"],
            "review_count": "sum",
            "latitude": "first",
            "longitude": "first"
        }
        summary = self.df_restaurants.groupby("area").agg(agg_dict)
        summary.columns = [
            "outlet_count",
            "mean_rating",
            "median_rating",
            "mean_cost_inr",
            "median_cost_inr",
            "total_reviews",
            "centroid_latitude",
            "centroid_longitude"
        ]
        summary = summary[summary["outlet_count"] >= min_outlets]
        summary = summary.sort_values(by="outlet_count", ascending=False)
        return summary.round(2).reset_index()

    def get_locality_distance_matrix(self, top_n_localities: int = 10) -> pd.DataFrame:
        """
        Computes pairwise Haversine distance matrix (in km) between top Bengaluru dining clusters.
        """
        summary = self.get_locality_summary()
        top_locs = summary.head(top_n_localities)
        
        loc_names = top_locs["area"].tolist()
        lats = top_locs["centroid_latitude"].tolist()
        lons = top_locs["centroid_longitude"].tolist()

        n = len(loc_names)
        dist_mat = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_mat[i, j] = haversine_distance(lats[i], lons[i], lats[j], lons[j])

        df_dist = pd.DataFrame(dist_mat, index=loc_names, columns=loc_names)
        return df_dist.round(2)

    def get_locality_density_metrics(self) -> Dict[str, Any]:
        """
        Computes general locality-level density metrics across the city.
        """
        total_outlets = len(self.df_restaurants)
        unique_localities = self.df_restaurants["area"].nunique()
        summary = self.get_locality_summary(min_outlets=1)
        
        return {
            "total_outlets": total_outlets,
            "unique_localities": unique_localities,
            "avg_outlets_per_locality": round(float(summary["outlet_count"].mean()), 2),
            "max_locality": summary.iloc[0]["area"],
            "max_locality_count": int(summary.iloc[0]["outlet_count"]),
            "data_source": "Bengaluru locality centroid",
            "precision": "locality-level"
        }
