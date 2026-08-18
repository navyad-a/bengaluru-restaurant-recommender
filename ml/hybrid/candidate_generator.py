# -*- coding: utf-8 -*-
"""
Multi-Source Candidate Generation Module
========================================
Retrieves candidate restaurants across Content-Based, Collaborative SVD,
and Location signals, applies hard constraints, and excludes already-rated outlets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Optional, Tuple
from ml.spatial.distance import haversine_vectorized


class CandidateGenerator:
    """
    Orchestrates multi-channel candidate generation and hard constraint pruning.
    """

    def __init__(self, df_restaurants: pd.DataFrame):
        self.df_restaurants = df_restaurants

    def apply_hard_filters(
        self,
        df: pd.DataFrame,
        filters: Optional[Dict[str, Any]] = None,
        user_coords: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Prunes candidate restaurants violating hard business rules.
        """
        if filters is None or df.empty:
            return df

        mask = pd.Series(True, index=df.index)

        # 1. Maximum Cost for Two
        max_cost = filters.get("max_cost_for_two")
        if max_cost is not None and max_cost > 0:
            mask &= (df["cost_for_two_inr"] <= max_cost)

        # 2. Minimum Rating
        min_rating = filters.get("min_rating")
        if min_rating is not None and min_rating > 0.0:
            include_unrated = filters.get("include_unrated", False)
            if include_unrated:
                mask &= (df["rating"].isna() | (df["rating"] >= min_rating))
            else:
                mask &= (df["rating"].notna() & (df["rating"] >= min_rating))

        # 3. Price Tier Exact Match
        price_tier = filters.get("price_tier") or filters.get("preferred_price_tier")
        if price_tier and isinstance(price_tier, str) and price_tier.strip():
            mask &= (df["price_tier"].str.lower() == price_tier.strip().lower())

        # 4. Area / Locality Exact Match
        area = filters.get("area") or filters.get("preferred_area") or filters.get("location")
        if area and isinstance(area, str) and area.strip():
            mask &= (df["area"].str.lower() == area.strip().lower())

        # 5. Online Order Only
        if filters.get("online_order_only") is True or filters.get("online_order") is True:
            mask &= (df["online_order"] == True)

        # 6. Table Booking Only
        if filters.get("book_table_only") is True or filters.get("book_table") is True:
            mask &= (df["book_table"] == True)

        # 7. Maximum Radius / Distance Filter
        radius_km = filters.get("radius_km") or filters.get("max_distance_km")
        if radius_km is not None and radius_km > 0 and user_coords is not None:
            u_lat, u_lon = user_coords
            dists = haversine_vectorized(
                user_lat=u_lat,
                user_lon=u_lon,
                restaurant_lats=df["latitude"].to_numpy(),
                restaurant_lons=df["longitude"].to_numpy()
            )
            mask &= (dists <= float(radius_km))

        return df[mask].copy()

    def generate_candidates(
        self,
        content_recommender,
        collaborative_recommender,
        location_scorer,
        user_id: Optional[int] = None,
        preferences: Optional[Dict[str, Any]] = None,
        target_restaurant_id: Optional[int] = None,
        user_coords: Optional[Tuple[float, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        candidate_pool_size: int = 150
    ) -> pd.DataFrame:
        """
        Generates candidate pool across all available recommendation channels.
        """
        candidate_ids: Set[int] = set()

        # Channel 1: Content-Based Candidates
        if target_restaurant_id is not None:
            c_recs = content_recommender.recommend_similar_restaurants(
                restaurant_id=target_restaurant_id,
                top_k=candidate_pool_size,
                filters=filters
            )
            candidate_ids.update(r["restaurant_id"] for r in c_recs)
        elif preferences:
            c_recs = content_recommender.recommend_for_preferences(
                preferences=preferences,
                top_k=candidate_pool_size,
                filters=filters
            )
            candidate_ids.update(r["restaurant_id"] for r in c_recs)

        # Channel 2: Collaborative SVD Candidates (if user is known)
        if user_id is not None and collaborative_recommender is not None:
            if collaborative_recommender.is_known_user(user_id):
                try:
                    cf_recs = collaborative_recommender.recommend_for_user(
                        user_id=user_id,
                        top_k=candidate_pool_size,
                        exclude_rated=True
                    )
                    candidate_ids.update(r["restaurant_id"] for r in cf_recs)
                except Exception:
                    pass

        # Channel 3: Location Proximity Candidates (if coordinates provided)
        if user_coords is not None and location_scorer is not None:
            u_lat, u_lon = user_coords
            dists, _ = location_scorer.score_vectorized(
                user_lat=u_lat,
                user_lon=u_lon,
                restaurant_lats=self.df_restaurants["latitude"].to_numpy(),
                restaurant_lons=self.df_restaurants["longitude"].to_numpy()
            )
            
            radius_km = filters.get("radius_km") if filters else None
            if radius_km is not None and radius_km > 0:
                within_r_indices = np.where(dists <= float(radius_km))[0]
                candidate_ids.update(self.df_restaurants.iloc[within_r_indices]["restaurant_id"])
            else:
                nearest_indices = dists.argsort()[:candidate_pool_size]
                candidate_ids.update(self.df_restaurants.iloc[nearest_indices]["restaurant_id"])

        # Fetch candidate records from catalog and apply hard filters
        df_candidate_pool = self.df_restaurants[
            self.df_restaurants["restaurant_id"].isin(candidate_ids)
        ].copy()
        df_filtered = self.apply_hard_filters(df_candidate_pool, filters, user_coords)

        # If candidate pool after hard filters is empty or smaller than desired,
        # pull matching restaurants directly from the hard-filtered catalog
        if len(df_filtered) < candidate_pool_size:
            df_filtered_catalog = self.apply_hard_filters(self.df_restaurants, filters, user_coords)
            if not df_filtered_catalog.empty:
                top_matches = df_filtered_catalog.sort_values(
                    by=["review_count", "rating"],
                    ascending=[False, False]
                ).head(candidate_pool_size)
                df_filtered = pd.concat([df_filtered, top_matches]).drop_duplicates(subset=["restaurant_id"]).head(candidate_pool_size)

        # Exclude target restaurant if doing restaurant-to-restaurant query
        if target_restaurant_id is not None and not df_filtered.empty:
            df_filtered = df_filtered[df_filtered["restaurant_id"] != target_restaurant_id]

        # Exclude already rated items for known collaborative user
        if user_id is not None and collaborative_recommender is not None and not df_filtered.empty:
            rated_set = collaborative_recommender.user_rated_items.get(user_id, set())
            df_filtered = df_filtered[~df_filtered["restaurant_id"].isin(rated_set)]

        return df_filtered
