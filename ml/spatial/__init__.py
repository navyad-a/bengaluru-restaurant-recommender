# -*- coding: utf-8 -*-
"""
Spatial Search & Proximity Subpackage Export
"""

from ml.spatial.coordinates import Coordinate, validate_coordinates
from ml.spatial.distance import (
    EARTH_RADIUS_KM,
    haversine_distance,
    haversine_vectorized,
    exponential_decay_score
)
from ml.spatial.bounding_box import BoundingBox, compute_bounding_box, filter_by_bounding_box
from ml.spatial.spatial_index import SpatialBallTreeIndex
from ml.spatial.spatial_search import SpatialSearchEngine
from ml.spatial.cluster_analysis import LocalitySpatialAnalytics

__all__ = [
    "Coordinate",
    "validate_coordinates",
    "EARTH_RADIUS_KM",
    "haversine_distance",
    "haversine_vectorized",
    "exponential_decay_score",
    "BoundingBox",
    "compute_bounding_box",
    "filter_by_bounding_box",
    "SpatialBallTreeIndex",
    "SpatialSearchEngine",
    "LocalitySpatialAnalytics"
]
