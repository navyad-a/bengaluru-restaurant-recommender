# -*- coding: utf-8 -*-
"""
Spatial BallTree Index Module
=============================
Wraps scikit-learn's BallTree with the spherical Haversine metric for O(log N)
nearest-neighbor and radius search on the 12,481-outlet catalog.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.neighbors import BallTree
from ml.spatial.distance import EARTH_RADIUS_KM


class SpatialBallTreeIndex:
    """
    In-memory spatial index utilizing BallTree over spherical radians coordinates.
    """

    def __init__(
        self,
        restaurant_ids: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray
    ):
        self.restaurant_ids = np.asarray(restaurant_ids, dtype=np.int64)
        self.latitudes = np.asarray(latitudes, dtype=np.float64)
        self.longitudes = np.asarray(longitudes, dtype=np.float64)

        # Convert degrees to radians for Haversine metric
        coords_rad = np.radians(np.column_stack((self.latitudes, self.longitudes)))
        self.tree = BallTree(coords_rad, metric="haversine")

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        id_col: str = "restaurant_id",
        lat_col: str = "latitude",
        lon_col: str = "longitude"
    ) -> "SpatialBallTreeIndex":
        """
        Builds spatial index from restaurant catalog DataFrame.
        """
        valid_df = df[df[lat_col].notna() & df[lon_col].notna()]
        return cls(
            restaurant_ids=valid_df[id_col].to_numpy(),
            latitudes=valid_df[lat_col].to_numpy(),
            longitudes=valid_df[lon_col].to_numpy()
        )

    def query_nearest(
        self,
        latitude: float,
        longitude: float,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries top-K nearest restaurant indices and distances (in km).
        Returns: (restaurant_ids, distances_km)
        """
        k = min(k, len(self.restaurant_ids))
        point_rad = np.radians([[latitude, longitude]])
        
        # dists are in radians, indices are integer positions
        dists_rad, indices = self.tree.query(point_rad, k=k, return_distance=True)
        
        dists_km = dists_rad[0] * EARTH_RADIUS_KM
        matched_ids = self.restaurant_ids[indices[0]]
        return matched_ids, np.round(dists_km, 4)

    def query_radius(
        self,
        latitude: float,
        longitude: float,
        radius_km: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries all restaurants within radius_km.
        Returns: (restaurant_ids, distances_km)
        """
        radius_rad = radius_km / EARTH_RADIUS_KM
        point_rad = np.radians([[latitude, longitude]])
        
        indices, dists_rad = self.tree.query_radius(
            point_rad,
            r=radius_rad,
            return_distance=True,
            sort_results=True
        )
        
        dists_km = dists_rad[0] * EARTH_RADIUS_KM
        matched_ids = self.restaurant_ids[indices[0]]
        return matched_ids, np.round(dists_km, 4)
