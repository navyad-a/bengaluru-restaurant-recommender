# -*- coding: utf-8 -*-
"""
Phase 2 Exploratory Data Analysis (EDA) & Feature Availability Report
Performs statistical profiling on the raw Zomato Bengaluru Restaurant Dataset.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Reconfigure stdout for Windows unicode safety
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.data.loader import DatasetLoader


def parse_rate(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val in ['NEW', '-', 'nan', '']:
        return np.nan
    if '/' in val:
        val = val.split('/')[0].strip()
    try:
        return float(val)
    except:
        return np.nan


def parse_cost(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace(',', '').strip()
    try:
        return float(val)
    except:
        return np.nan


def run_full_eda():
    print("=" * 80)
    print("PHASE 2: BENGALURU RESTAURANT DATASET -- EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    loader = DatasetLoader()
    if not loader.exists():
        print(f"[ERROR] Raw dataset not found at: {loader.raw_data_path}")
        return
        
    print(f"[*] Loading raw dataset from: {loader.raw_data_path}")
    df = loader.load_raw_data()
    print(f"[*] Raw Dataset Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # 1. Feature Availability Report
    print("\n" + "-" * 80)
    print("1. FEATURE AVAILABILITY & DATA QUALITY REPORT")
    print("-" * 80)
    rep_df = loader.get_feature_availability_report(df)
    print(rep_df.to_string(index=False))
    
    # 2. Collaborative Filtering Feasibility Check
    print("\n" + "-" * 80)
    print("2. COLLABORATIVE FILTERING FEASIBILITY CHECK")
    print("-" * 80)
    cf_check = loader.check_collaborative_feasibility(df)
    print(f"  * Direct User ID Column Present? : {'YES' if cf_check['has_direct_user_id'] else 'NO'}")
    print(f"  * User IDs inside Review Tuples? : {'YES' if cf_check['has_user_in_reviews_tuple'] else 'NO'}")
    print(f"  * Unique Restaurant Names       : {cf_check['unique_restaurant_names']:,}")
    print(f"  * Unique Physical Addresses      : {cf_check['unique_restaurant_addresses']:,}")
    print(f"  * CF Direct Training Feasible?   : {'YES' if cf_check['cf_direct_training_feasible'] else 'NO'}")
    print(f"  * Critical Finding & Verdict     :\n    {cf_check['reason']}")
    
    # 3. Bengaluru Geographic Scope Validation
    print("\n" + "-" * 80)
    print("3. BENGALURU GEOGRAPHIC SCOPE VALIDATION")
    print("-" * 80)
    geo_check = loader.validate_bengaluru_coverage(df)
    print(f"  * Geographic Target Area         : {geo_check['city_scope']}")
    print(f"  * Total Unique Localities Found  : {geo_check['unique_localities_count']}")
    print(f"  * Records with Missing Location  : {geo_check['missing_locations_count']}")
    print("  * Top 10 Localities in Dataset   :")
    for loc, count in list(geo_check['top_20_localities'].items())[:10]:
        print(f"      - {loc:<25}: {count:>5,} entries ({round(count/len(df)*100, 2)}%)")
        
    # 4. Statistical Distributions (Ratings & Cost for Two)
    print("\n" + "-" * 80)
    print("4. STATISTICAL DISTRIBUTIONS & METRICS")
    print("-" * 80)
    df['clean_rate'] = df['rate'].apply(parse_rate)
    df['clean_cost'] = df['approx_cost(for two people)'].apply(parse_cost)
    
    rate_desc = df['clean_rate'].describe()
    print("  [*] Aggregate Rating Distribution (1.0 to 5.0 scale):")
    print(f"      - Evaluated Restaurants : {int(rate_desc['count']):,} ({round(rate_desc['count']/len(df)*100, 1)}% of dataset)")
    print(f"      - Unrated / 'NEW' / '-' : {len(df) - int(rate_desc['count']):,} ({round((len(df)-rate_desc['count'])/len(df)*100, 1)}%)")
    print(f"      - Mean Rating           : {rate_desc['mean']:.2f} / 5.0")
    print(f"      - Standard Deviation    : {rate_desc['std']:.2f}")
    print(f"      - Min / Median / Max    : {rate_desc['min']:.1f} / {rate_desc['50%']:.1f} / {rate_desc['max']:.1f} (out of 5.0)")
    
    cost_desc = df['clean_cost'].describe()
    print("\n  [*] Approx Cost for Two People (Indian Rupees - INR):")
    print(f"      - Available Records     : {int(cost_desc['count']):,} ({round(cost_desc['count']/len(df)*100, 1)}%)")
    print(f"      - Mean Cost for Two     : Rs. {cost_desc['mean']:.0f}")
    print(f"      - Median Cost for Two   : Rs. {cost_desc['50%']:.0f}")
    print(f"      - Interquartile Range   : Rs. {cost_desc['25%']:.0f} (25th %) to Rs. {cost_desc['75%']:.0f} (75th %)")
    print(f"      - Min / Max Range       : Rs. {cost_desc['min']:.0f} to Rs. {cost_desc['max']:.0f}")
    
    # 5. Cuisine Distribution
    print("\n" + "-" * 80)
    print("5. TOP REGIONAL & INDIAN CUISINES")
    print("-" * 80)
    all_cuisines = []
    for c in df['cuisines'].dropna():
        for item in c.split(','):
            all_cuisines.append(item.strip())
    cuis_series = pd.Series(all_cuisines)
    print(f"  * Total Unique Cuisine Tags Detected: {cuis_series.nunique()}")
    print("  * Top 12 Most Prevalent Cuisines:")
    for cuis, count in cuis_series.value_counts().head(12).items():
        print(f"      - {cuis:<20}: {count:>6,} listings")
        
    # 6. Service & Dining Flags
    print("\n" + "-" * 80)
    print("6. SERVICE AVAILABILITY & RESTAURANT TYPES")
    print("-" * 80)
    online_order_pct = df['online_order'].value_counts(normalize=True).to_dict()
    book_table_pct = df['book_table'].value_counts(normalize=True).to_dict()
    print(f"  * Online Ordering Available  : {round(online_order_pct.get('Yes', 0)*100, 1)}% Yes | {round(online_order_pct.get('No', 0)*100, 1)}% No")
    print(f"  * Table Booking Available    : {round(book_table_pct.get('Yes', 0)*100, 1)}% Yes | {round(book_table_pct.get('No', 0)*100, 1)}% No")
    
    all_types = []
    for t in df['rest_type'].dropna():
        for item in t.split(','):
            all_types.append(item.strip())
    print("  * Top Dining Formats:")
    for rtype, count in pd.Series(all_types).value_counts().head(8).items():
        print(f"      - {rtype:<20}: {count:>6,} listings")
        
    # 7. Duplication & Multiple Listings Analysis
    print("\n" + "-" * 80)
    print("7. RECORD DUPLICATION & MULTI-ZONE LISTING ANALYSIS")
    print("-" * 80)
    exact_duplicates = df.duplicated().sum()
    subset_duplicates = df.duplicated(subset=['name', 'address']).sum()
    print(f"  * Exact 100% Identical Rows  : {exact_duplicates:,}")
    print(f"  * Same Restaurant & Address  : {subset_duplicates:,} (Listed across multiple Zomato delivery zones)")
    print(f"  * Unique Physical Eateries   : {len(df) - subset_duplicates:,}")
    
    print("\n" + "=" * 80)
    print("PHASE 2 EDA & INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_full_eda()
