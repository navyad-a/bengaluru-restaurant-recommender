# -*- coding: utf-8 -*-
"""
Location Proximity & Haversine Distance Module
==============================================
Calculates great-circle geographic distance and exponential spatial decay
proximity scores between user location and restaurant locality centroids.

Mathematical Formulation:
    Haversine Formula:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        c = 2 * atan2(√a, √(1-a))
        d = R * c   (where R = 6371.0 km)

    Spatial Decay Proximity Score:
        S_location = exp(-d / tau)
        
Where tau is the spatial decay distance constant (default: 3.0 km).
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Computes great-circle distance between two points in kilometers.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def haversine_vectorized(
    user_lat: float,
    user_lon: float,
    restaurant_lats: np.ndarray,
    restaurant_lons: np.ndarray
) -> np.ndarray:
    """
    Vectorized Haversine distance computation across array of restaurant coordinates.
    """
    u_lat_rad = np.radians(user_lat)
    u_lon_rad = np.radians(user_lon)
    r_lat_rad = np.radians(restaurant_lats)
    r_lon_rad = np.radians(restaurant_lons)

    dlat = r_lat_rad - u_lat_rad
    dlon = r_lon_rad - u_lon_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(u_lat_rad) * np.cos(r_lat_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


class LocationScorer:
    """
    Computes spatial proximity scores using exponential distance decay.
    """

    def __init__(self, decay_tau_km: float = 3.0):
        self.decay_tau_km = max(0.1, float(decay_tau_km))

    def calculate_distance(
        self,
        user_coords: Tuple[float, float],
        rest_coords: Tuple[float, float]
    ) -> float:
        """
        Calculates distance in km between user coordinates and restaurant coordinates.
        """
        lat1, lon1 = user_coords
        lat2, lon2 = rest_coords
        return round(haversine_distance(lat1, lon1, lat2, lon2), 2)

    def score_distance(self, distance_km: float) -> float:
        """
        Transforms distance (in km) into an exponential proximity score in [0.0, 1.0].
        - 0.0 km -> 1.0000
        - 3.0 km (tau) -> exp(-1) ≈ 0.3679
        - 10.0 km -> exp(-3.33) ≈ 0.0357
        """
        if distance_km < 0:
            return 0.0
        score = math.exp(-distance_km / self.decay_tau_km)
        return max(0.0, min(1.0, round(float(score), 4)))

    def score_vectorized(
        self,
        user_lat: float,
        user_lon: float,
        restaurant_lats: np.ndarray,
        restaurant_lons: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized distance (km) and proximity scores [0, 1].
        Returns (distances_km, proximity_scores).
        """
        distances = haversine_vectorized(user_lat, user_lon, restaurant_lats, restaurant_lons)
        scores = np.exp(-distances / self.decay_tau_km)
        return np.round(distances, 2), np.clip(np.round(scores, 4), 0.0, 1.0)
