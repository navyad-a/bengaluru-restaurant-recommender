# -*- coding: utf-8 -*-
"""
Haversine Distance & Spatial Decay Module
=========================================
Provides numerically stable great-circle distance calculation and
exponential proximity decay scoring functions.

Formulas:
    Haversine:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        c = 2 * atan2(√a, √(1-a))
        d = R * c   (where R = 6371.0088 km)

    Exponential Spatial Decay:
        S_location = exp(-d / tau)
"""

import math
import numpy as np
from typing import Tuple

# IUGG Mean Earth Radius in kilometers
EARTH_RADIUS_KM = 6371.0088


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Computes great-circle distance in kilometers between two points on Earth.
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    a = min(1.0, max(0.0, a))  # Clamp against numerical rounding anomalies
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(float(EARTH_RADIUS_KM * c), 4)


def haversine_vectorized(
    user_lat: float,
    user_lon: float,
    restaurant_lats: np.ndarray,
    restaurant_lons: np.ndarray
) -> np.ndarray:
    """
    Vectorized Haversine distance computation against an array of coordinates.
    """
    u_lat_rad = np.radians(user_lat)
    u_lon_rad = np.radians(user_lon)
    r_lats_rad = np.radians(restaurant_lats)
    r_lons_rad = np.radians(restaurant_lons)

    dlat = r_lats_rad - u_lat_rad
    dlon = r_lons_rad - u_lon_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(u_lat_rad) * np.cos(r_lats_rad) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return np.round(EARTH_RADIUS_KM * c, 4)


def exponential_decay_score(
    distance_km: float,
    tau_km: float = 3.0
) -> float:
    """
    Transforms distance (km) into an exponential proximity score in [0.0, 1.0].
    
    Monotonicity Guarantee:
    - distance = 0.0 -> score = 1.0
    - as distance increases -> score monotonically decreases towards 0.0
    """
    if distance_km < 0.0:
        return 0.0
    
    tau = max(0.01, float(tau_km))
    score = math.exp(-distance_km / tau)
    return max(0.0, min(1.0, round(float(score), 4)))
