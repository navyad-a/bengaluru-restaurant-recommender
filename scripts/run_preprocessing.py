# -*- coding: utf-8 -*-
"""
Phase 3 Data Cleaning, Geocoding & Benchmark Generation Runner
Processes raw Zomato Bengaluru data into clean artifacts and builds the Synthetic CF benchmark.
"""

import os
import sys
import json
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessing.cleaner import clean_restaurant_data
from ml.data.synthetic_generator import generate_synthetic_benchmark


def main():
    print("=" * 80)
    print("PHASE 3: DATA CLEANING, GEOCODING & BENCHMARK GENERATION")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'zomato_bangalore_raw.csv')
    clean_path = os.path.join(base_dir, 'data', 'processed', 'restaurants_clean.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    if not os.path.exists(raw_path):
        print(f"[ERROR] Raw dataset not found at {raw_path}")
        return
        
    # Step 1: Clean and deduplicate restaurant catalog
    print("\n[*] Step 1: Executing Data Cleaning & Locality Geocoding...")
    df_clean, waterfall = clean_restaurant_data(raw_path, clean_path)
    
    print("\n  [+] Data Cleaning Waterfall:")
    print(f"      - Initial Raw Rows                  : {waterfall['raw_rows']:,}")
    print(f"      - Rows with Valid Identifiers       : {waterfall['valid_identifier_rows']:,} (Dropped {waterfall['invalid_dropped_rows']} invalid rows)")
    print(f"      - Multi-Zone Delivery Duplicates    : {waterfall['multi_zone_duplicates_removed']:,} rows consolidated")
    print(f"      - Final Clean Physical Restaurants   : {waterfall['deduplicated_physical_restaurants']:,}")
    print(f"      - Distinct Brand / Chain Names      : {waterfall['unique_restaurant_brands']:,}")
    print(f"      - Restaurants with Ratings          : {waterfall['rated_restaurants_count']:,} ({round(waterfall['rated_restaurants_count']/len(df_clean)*100, 1)}%)")
    print(f"      - Unrated / Cold-Start Restaurants  : {waterfall['unrated_cold_start_count']:,} ({round(waterfall['unrated_cold_start_count']/len(df_clean)*100, 1)}%)")
    print(f"      - Saved Clean Catalog to            : {clean_path}")
    
    # Step 2: Generate Synthetic Collaborative Filtering Benchmark
    print("\n[*] Step 2: Generating Synthetic Collaborative Filtering Benchmark...")
    meta = generate_synthetic_benchmark(clean_path, processed_dir, num_users=600, ratings_per_user_mean=20)
    
    print("\n  [+] Synthetic Benchmark Generated:")
    print(f"      - Total Simulated Users             : {meta['total_simulated_users']:,}")
    print(f"      - Total Synthetic Ratings           : {meta['total_simulated_ratings']:,}")
    print(f"      - Train Split (80% per-user)        : {meta['train_ratings_count']:,}")
    print(f"      - Test Split (20% per-user holdout) : {meta['test_ratings_count']:,}")
    print(f"      - Avg Ratings per User              : {meta['avg_ratings_per_user']}")
    print(f"      - Matrix Sparsity                   : {meta['sparsity_percentage']}")
    print(f"      - Deterministic Random Seed         : {meta['random_seed']}")
    print(f"      - Notice & Warning                  :\n        {meta['warning']}")
    
    # Step 3: Verify Output Integrity
    print("\n[*] Step 3: Verifying Processed File Integrity...")
    expected_files = [
        'restaurants_clean.csv',
        'synthetic_users.csv',
        'synthetic_ratings.csv',
        'synthetic_train_ratings.csv',
        'synthetic_test_ratings.csv',
        'benchmark_metadata.json'
    ]
    all_ok = True
    for ef in expected_files:
        p = os.path.join(processed_dir, ef)
        if os.path.exists(p):
            sz = round(os.path.getsize(p) / 1024, 1)
            print(f"  [OK] {ef:<30} ({sz} KB)")
        else:
            print(f"  [FAIL] Missing {ef}")
            all_ok = False
            
    print("\n" + "=" * 80)
    if all_ok:
        print("RESULT: PHASE 3 DATA CLEANING & PREPROCESSING COMPLETED SUCCESSFULLY!")
    else:
        print("RESULT: PHASE 3 VERIFICATION ENCOUNTERED ERRORS.")
    print("=" * 80)


if __name__ == "__main__":
    main()
