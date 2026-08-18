# -*- coding: utf-8 -*-
"""
Spatial Bounding Box Module
===========================
Generates geographic bounding boxes around coordinate points for fast spatial
pre-filtering, reducing unnecessary trigonometric distance computations.
"""

import math
import pandas as pd
from typing import NamedTuple, Tuple
from ml.spatial.distance import EARTH_RADIUS_KM


class BoundingBox(NamedTuple):
    """
    Geographic bounding box representation in degrees.
    """
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        """
        Checks if a point falls within the bounding box.
        """
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)


def compute_bounding_box(
    latitude: float,
    longitude: float,
    radius_km: float
) -> BoundingBox:
    """
    Constructs a bounding box around (latitude, longitude) enclosing a circle of radius_km.
    
    Formula:
        Δlat = (radius_km / R) * (180 / π)
        Δlon = (radius_km / (R * cos(lat_rad))) * (180 / π)
    """
    if radius_km < 0:
        raise ValueError(f"Radius cannot be negative: {radius_km}")

    lat_rad = math.radians(latitude)
    
    # Angular radius in radians
    rad_dist = radius_km / EARTH_RADIUS_KM
    
    delta_lat = math.degrees(rad_dist)
    min_lat = max(-90.0, latitude - delta_lat)
    max_lat = min(90.0, latitude + delta_lat)

    # Longitude delta depends on latitude
    cos_lat = math.cos(lat_rad)
    if cos_lat > 1e-6:
        delta_lon = math.degrees(rad_dist / cos_lat)
        min_lon = longitude - delta_lon
        max_lon = longitude + delta_lon
        
        # Handle wrapping if needed
        if min_lon < -180.0:
            min_lon = -180.0
        if max_lon > 180.0:
            max_lon = 180.0
    else:
        # Near poles
        min_lon = -180.0
        max_lon = 180.0

    return BoundingBox(
        min_lat=round(min_lat, 6),
        max_lat=round(max_lat, 6),
        min_lon=round(min_lon, 6),
        max_lon=round(max_lon, 6)
    )


def filter_by_bounding_box(
    df: pd.DataFrame,
    bbox: BoundingBox,
    lat_col: str = "latitude",
    lon_col: str = "longitude"
) -> pd.DataFrame:
    """
    Filters DataFrame rows to only those whose coordinates fall inside the bounding box.
    """
    if df.empty:
        return df

    mask = (
        (df[lat_col] >= bbox.min_lat) &
        (df[lat_col] <= bbox.max_lat) &
        (df[lon_col] >= bbox.min_lon) &
        (df[lon_col] <= bbox.max_lon)
    )
    return df[mask].copy()
