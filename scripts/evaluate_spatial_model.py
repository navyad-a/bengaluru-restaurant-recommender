# -*- coding: utf-8 -*-
"""
Phase 8 — Spatial Search Optimization & Proximity Scoring Evaluation
====================================================================
Runs spatial analytics, radius searches, nearest-neighbor queries, and
benchmarks Naive Haversine vs. Bounding Box vs. BallTree Index over 12,481 catalog outlets.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.spatial.distance import haversine_distance, haversine_vectorized, EARTH_RADIUS_KM
from ml.spatial.bounding_box import compute_bounding_box, filter_by_bounding_box
from ml.spatial.spatial_index import SpatialBallTreeIndex
from ml.spatial.spatial_search import SpatialSearchEngine
from ml.spatial.cluster_analysis import LocalitySpatialAnalytics


def run_spatial_evaluation():
    print("=" * 95)
    print("PHASE 8: SPATIAL SEARCH OPTIMIZATION & PROXIMITY SCORING EVALUATION")
    print("=" * 95)

    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "restaurants_clean.csv")
    df = pd.read_csv(catalog_path)
    print(f"[*] Loaded Authentic Catalog: {len(df):,} restaurants across Bengaluru.")

    # -------------------------------------------------------------
    # SECTION 1: COORDINATE COVERAGE & PROVENANCE METRICS
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("SECTION 1: COORDINATE COVERAGE & PROVENANCE METRICS")
    print("-" * 95)

    total_outlets = len(df)
    valid_coords = df["latitude"].notna().sum()
    coverage_pct = (valid_coords / total_outlets) * 100

    print(f"  Total Catalog Outlets            : {total_outlets:,}")
    print(f"  Valid Geographic Coordinates     : {valid_coords:,} ({coverage_pct:.2f}%)")
    print(f"  Latitude Range                   : [{df['latitude'].min():.4f}° N, {df['latitude'].max():.4f}° N]")
    print(f"  Longitude Range                  : [{df['longitude'].min():.4f}° E, {df['longitude'].max():.4f}° E]")
    print(f"  Coordinate Source Attribution    : {df['location_source'].iloc[0]}")
    print(f"  Coordinate Precision Level       : {df['location_precision'].iloc[0]}")

    # -------------------------------------------------------------
    # SECTION 2: LOCALITY SPATIAL ANALYTICS
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("SECTION 2: LOCALITY SPATIAL ANALYTICS (BENGALURU DINING HUBS)")
    print("-" * 95)

    analytics = LocalitySpatialAnalytics(df)
    loc_summary = analytics.get_locality_summary(min_outlets=50).head(10)
    print(f"{'Locality / Area':<25} | {'Outlets':<8} | {'Avg Rating':<10} | {'Median Cost (₹)':<16} | {'Centroid Coords':<22}")
    print("-" * 95)
    for _, row in loc_summary.iterrows():
        coords_str = f"({row['centroid_latitude']:.4f}, {row['centroid_longitude']:.4f})"
        print(f"{row['area']:<25} | {row['outlet_count']:<8} | {row['mean_rating']:<10.2f} | Rs. {row['median_cost_inr']:<12.0f} | {coords_str:<22}")

    print("\n[*] Inter-Locality Haversine Distance Matrix (km) between Major Hubs:")
    dist_matrix = analytics.get_locality_distance_matrix(top_n_localities=6)
    print(dist_matrix.to_string())

    # -------------------------------------------------------------
    # SECTION 3: RADIUS & NEAREST-NEIGHBOR QUERIES
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("SECTION 3: RADIUS & NEAREST-NEIGHBOR SEARCH VERIFICATION")
    print("-" * 95)

    engine = SpatialSearchEngine(df)

    test_locations = [
        ("Koramangala 5th Block", 12.9352, 77.6245),
        ("Indiranagar 100ft Road", 12.9784, 77.6408),
        ("Jayanagar 4th Block", 12.9250, 77.5838),
        ("Whitefield ITPL", 12.9698, 77.7499)
    ]

    for name, lat, lon in test_locations:
        print(f"\n[Hub: {name} ({lat}° N, {lon}° E)]")
        # Nearest 3
        nearest_3 = engine.find_nearest(lat, lon, top_k=3)
        for idx, r in enumerate(nearest_3, 1):
            print(f"  Nearest {idx}: {r['name']:<32} | Dist: {r['distance_km']:.2f} km | Area: {r['area']} | Cost: Rs. {r['cost_for_two_inr']}")

        # Counts within radii
        for r_km in [1.0, 3.0, 5.0]:
            matches = engine.search_within_radius(lat, lon, radius_km=r_km)
            print(f"  -> Outlets within {r_km:.1f} km radius: {len(matches):,} restaurants")

    # -------------------------------------------------------------
    # SECTION 4: SPATIAL BENCHMARKING (10, 100, 1,000 QUERIES)
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("SECTION 4: SPATIAL RETRIEVAL PERFORMANCE BENCHMARK (12,481 OUTLETS)")
    print("-" * 95)

    np.random.seed(42)
    sample_lats = np.random.uniform(12.85, 13.08, 1000)
    sample_lons = np.random.uniform(77.50, 77.72, 1000)
    radius_test_km = 3.0

    rest_lats = df["latitude"].to_numpy(dtype=np.float64)
    rest_lons = df["longitude"].to_numpy(dtype=np.float64)
    ball_tree = SpatialBallTreeIndex.from_dataframe(df)

    for n_queries in [10, 100, 1000]:
        print(f"\n>>> BENCHMARKING WITH N = {n_queries:,} QUERIES (Search Radius = {radius_test_km} km) <<<")

        # 1. Approach A: Naive Vectorized Haversine Scan over all 12,481 restaurants
        t_naive_list = []
        for i in range(n_queries):
            t0 = time.perf_counter()
            dists = haversine_vectorized(sample_lats[i], sample_lons[i], rest_lats, rest_lons)
            _ = np.where(dists <= radius_test_km)[0]
            t_naive_list.append((time.perf_counter() - t0) * 1000)

        # 2. Approach B: Bounding Box Pre-Filter + Vectorized Haversine
        t_bbox_list = []
        for i in range(n_queries):
            t0 = time.perf_counter()
            bbox = compute_bounding_box(sample_lats[i], sample_lons[i], radius_test_km)
            bbox_mask = (
                (rest_lats >= bbox.min_lat) &
                (rest_lats <= bbox.max_lat) &
                (rest_lons >= bbox.min_lon) &
                (rest_lons <= bbox.max_lon)
            )
            cand_lats = rest_lats[bbox_mask]
            cand_lons = rest_lons[bbox_mask]
            dists = haversine_vectorized(sample_lats[i], sample_lons[i], cand_lats, cand_lons)
            _ = np.where(dists <= radius_test_km)[0]
            t_bbox_list.append((time.perf_counter() - t0) * 1000)

        # 3. Approach C: BallTree Spherical Index Query
        t_tree_list = []
        for i in range(n_queries):
            t0 = time.perf_counter()
            _ = ball_tree.query_radius(sample_lats[i], sample_lons[i], radius_km=radius_test_km)
            t_tree_list.append((time.perf_counter() - t0) * 1000)

        # Print latency summary
        print(f"{'Method':<35} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'P95 (ms)':<10} | {'Max (ms)':<10} | {'Speedup':<8}")
        print("-" * 95)
        
        m_naive = np.mean(t_naive_list)
        m_bbox = np.mean(t_bbox_list)
        m_tree = np.mean(t_tree_list)

        print(f"{'A. Naive Haversine Scan (12,481)':<35} | {m_naive:<10.4f} | {np.median(t_naive_list):<12.4f} | {np.percentile(t_naive_list, 95):<10.4f} | {np.max(t_naive_list):<10.4f} | {'1.0x (baseline)'}")
        print(f"{'B. Bounding Box + Haversine':<35} | {m_bbox:<10.4f} | {np.median(t_bbox_list):<12.4f} | {np.percentile(t_bbox_list, 95):<10.4f} | {np.max(t_bbox_list):<10.4f} | {f'{m_naive/m_bbox:.2f}x':<8}")
        print(f"{'C. BallTree Spherical Index':<35} | {m_tree:<10.4f} | {np.median(t_tree_list):<12.4f} | {np.percentile(t_tree_list, 95):<10.4f} | {np.max(t_tree_list):<10.4f} | {f'{m_naive/m_tree:.2f}x':<8}")

    print("\n" + "=" * 95)
    print("PHASE 8 SPATIAL EVALUATION & BENCHMARK COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    run_spatial_evaluation()
