# -*- coding: utf-8 -*-
"""
Spatial Search Engine Module
============================
Executes high-performance radius queries and nearest-neighbor searches with
bounding-box pre-filtering, BallTree acceleration, and deterministic ranking.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ml.spatial.coordinates import validate_coordinates
from ml.spatial.distance import haversine_distance, exponential_decay_score
from ml.spatial.bounding_box import compute_bounding_box, filter_by_bounding_box
from ml.spatial.spatial_index import SpatialBallTreeIndex


class SpatialSearchEngine:
    """
    Production spatial search service over the Bengaluru restaurant catalog.
    """

    def __init__(
        self,
        df_restaurants: pd.DataFrame,
        index: Optional[SpatialBallTreeIndex] = None
    ):
        self.df_restaurants = df_restaurants
        self.restaurant_dict = {
            int(row["restaurant_id"]): row.to_dict()
            for _, row in df_restaurants.iterrows()
        }
        self.index = index or SpatialBallTreeIndex.from_dataframe(df_restaurants)

    def _apply_hard_filters(
        self,
        df: pd.DataFrame,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Prunes candidate restaurants violating business criteria.
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

    def find_nearest(
        self,
        latitude: float,
        longitude: float,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-K nearest restaurants from given coordinates.
        """
        lat, lon = validate_coordinates(latitude, longitude)
        
        # If filters are present, fetch more candidates from spatial index to ensure top_k after filtering
        query_k = max(top_k * 5, 50) if filters else top_k
        matched_ids, dists_km = self.index.query_nearest(lat, lon, k=query_k)

        results = []
        for r_id, dist in zip(matched_ids, dists_km):
            rest = self.restaurant_dict.get(int(r_id))
            if rest:
                item = dict(rest)
                item["distance_km"] = float(dist)
                item["location_score"] = exponential_decay_score(dist)
                results.append(item)

        df_results = pd.DataFrame(results)
        df_filtered = self._apply_hard_filters(df_results, filters)

        # Deterministic sorting: distance ASC, rating DESC, review_count DESC, id ASC
        sorted_records = df_filtered.sort_values(
            by=["distance_km", "rating", "review_count", "restaurant_id"],
            ascending=[True, False, False, True]
        ).head(top_k).to_dict(orient="records")

        return sorted_records

    def search_within_radius(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds all restaurants located within radius_km.
        """
        lat, lon = validate_coordinates(latitude, longitude)
        if radius_km <= 0:
            return []

        matched_ids, dists_km = self.index.query_radius(lat, lon, radius_km=radius_km)
        
        results = []
        for r_id, dist in zip(matched_ids, dists_km):
            rest = self.restaurant_dict.get(int(r_id))
            if rest:
                item = dict(rest)
                item["distance_km"] = float(dist)
                item["location_score"] = exponential_decay_score(dist)
                results.append(item)

        if not results:
            return []

        df_results = pd.DataFrame(results)
        df_filtered = self._apply_hard_filters(df_results, filters)

        df_sorted = df_filtered.sort_values(
            by=["distance_km", "rating", "review_count", "restaurant_id"],
            ascending=[True, False, False, True]
        )

        if top_k is not None:
            df_sorted = df_sorted.head(top_k)

        return df_sorted.to_dict(orient="records")

    def find_nearest_within_radius(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds top-K nearest restaurants within radius_km with deterministic tie-breaking.
        """
        return self.search_within_radius(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            filters=filters,
            top_k=top_k
        )
