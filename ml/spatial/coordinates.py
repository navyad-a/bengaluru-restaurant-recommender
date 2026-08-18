# -*- coding: utf-8 -*-
"""
Spatial Coordinate Validation & Precision Metadata Module
=========================================================
Validates geographic coordinates and tracks coordinate precision/source.
"""

import math
from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class Coordinate(BaseModel):
    """
    Validated geographic coordinate pair.
    """
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees [-90.0, 90.0]")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees [-180.0, 180.0]")
    location_source: str = Field(default="Bengaluru locality centroid", description="Data provenance of coordinates")
    location_precision: str = Field(default="locality-level", description="Precision level: exact, locality-level, or unknown")

    @field_validator("latitude", "longitude")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Coordinate value cannot be NaN or infinite.")
        return float(v)


def validate_coordinates(
    latitude: Optional[float],
    longitude: Optional[float]
) -> Optional[Tuple[float, float]]:
    """
    Validates a (latitude, longitude) pair.
    Returns (lat, lon) as floats if valid, or None if both are None.
    Raises ValueError if invalid, out of bounds, NaN, or infinite.
    """
    if latitude is None and longitude is None:
        return None

    if latitude is None or longitude is None:
        raise ValueError("Both latitude and longitude must be provided together.")

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid coordinate format: lat={latitude}, lon={longitude}")

    if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
        raise ValueError(f"Coordinates cannot be NaN or infinite: lat={lat}, lon={lon}")

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude {lat} is out of valid bounds [-90.0, 90.0]")

    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude {lon} is out of valid bounds [-180.0, 180.0]")

    return (lat, lon)
