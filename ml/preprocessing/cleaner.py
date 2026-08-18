# -*- coding: utf-8 -*-
"""
Data Cleaning & Feature Engineering Module
Transforms raw Zomato Bengaluru records into an authoritative, clean restaurant catalog.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from ml.preprocessing.geocoder import get_locality_coordinates


def parse_rating_value(val) -> float:
    """
    Parses 'rate' field (e.g. '4.1/5', 'NEW', '-') into float or NaN.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str in ['NEW', '-', 'nan', '']:
        return np.nan
    if '/' in val_str:
        val_str = val_str.split('/')[0].strip()
    try:
        r = float(val_str)
        return round(r, 1) if 1.0 <= r <= 5.0 else np.nan
    except:
        return np.nan


def parse_cost_value(val) -> Optional[int]:
    """
    Parses 'approx_cost(for two people)' into integer INR (₹) or NaN.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace(',', '').strip()
    try:
        cost = int(float(val_str))
        return cost if cost > 0 else np.nan
    except:
        return np.nan


def get_price_tier(cost: float) -> str:
    """
    Assigns Indian dining price tier based on Cost for Two in INR.
    - Budget: < ₹400
    - Moderate: ₹400 - ₹799
    - Premium: ₹800 - ₹1799
    - Luxury: >= ₹1800
    """
    if pd.isna(cost):
        return "Moderate"
    if cost < 400:
        return "Budget"
    elif cost < 800:
        return "Moderate"
    elif cost < 1800:
        return "Premium"
    else:
        return "Luxury"


def clean_restaurant_data(
    raw_csv_path: str,
    output_clean_csv_path: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the comprehensive cleaning and deduplication pipeline.
    """
    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    initial_rows = len(df_raw)
    
    # 1. Clean and normalize string fields
    df_raw['name'] = df_raw['name'].astype(str).str.strip()
    df_raw['address'] = df_raw['address'].astype(str).str.strip()
    df_raw['location'] = df_raw['location'].astype(str).str.strip()
    df_raw['cuisines'] = df_raw['cuisines'].astype(str).str.strip()
    df_raw['rest_type'] = df_raw['rest_type'].astype(str).str.strip()
    
    # 2. Filter records with invalid / missing core identifiers
    valid_mask = (
        df_raw['name'].notna() & (df_raw['name'] != 'nan') & (df_raw['name'] != '') &
        df_raw['location'].notna() & (df_raw['location'] != 'nan') & (df_raw['location'] != '') &
        df_raw['cuisines'].notna() & (df_raw['cuisines'] != 'nan') & (df_raw['cuisines'] != '')
    )
    df_valid = df_raw[valid_mask].copy()
    valid_rows = len(df_valid)
    
    # 3. Parse ratings and costs
    df_valid['rating'] = df_valid['rate'].apply(parse_rating_value)
    df_valid['cost_for_two_inr'] = df_valid['approx_cost(for two people)'].apply(parse_cost_value)
    
    # Impute missing cost for two with median cost of that restaurant type or global median (₹400)
    median_cost = df_valid['cost_for_two_inr'].median()
    df_valid['cost_for_two_inr'] = df_valid['cost_for_two_inr'].fillna(median_cost).astype(int)
    
    df_valid['price_tier'] = df_valid['cost_for_two_inr'].apply(get_price_tier)
    
    # 4. Standardize Boolean service flags
    df_valid['online_order'] = df_valid['online_order'].astype(str).str.strip().str.lower() == 'yes'
    df_valid['book_table'] = df_valid['book_table'].astype(str).str.strip().str.lower() == 'yes'
    
    # 5. Deduplicate physical branches: group by (name, address)
    # Sort by 'votes' descending to keep the listing with the most complete review history
    df_sorted = df_valid.sort_values(by='votes', ascending=False)
    df_dedup = df_sorted.drop_duplicates(subset=['name', 'address']).copy()
    
    # 6. Assign persistent integer restaurant_id
    df_dedup.reset_index(drop=True, inplace=True)
    df_dedup['restaurant_id'] = df_dedup.index + 1
    
    # 7. Map locality centroid coordinates
    coords = df_dedup['location'].apply(get_locality_coordinates)
    df_dedup['latitude'] = [c[0] for c in coords]
    df_dedup['longitude'] = [c[1] for c in coords]
    df_dedup['location_source'] = "Bengaluru locality centroid"
    df_dedup['location_precision'] = "locality-level"
    df_dedup['city'] = "Bengaluru"
    
    # 8. Select and rename final clean schema columns
    final_columns = [
        'restaurant_id', 'name', 'city', 'location', 'address',
        'latitude', 'longitude', 'location_source', 'location_precision',
        'cuisines', 'rest_type', 'cost_for_two_inr', 'price_tier',
        'rating', 'votes', 'online_order', 'book_table', 'dish_liked'
    ]
    df_clean = df_dedup[final_columns].copy()
    df_clean.rename(columns={'location': 'area', 'votes': 'review_count'}, inplace=True)
    
    # Ensure directory exists and write CSV
    os.makedirs(os.path.dirname(output_clean_csv_path), exist_ok=True)
    df_clean.to_csv(output_clean_csv_path, index=False, encoding='utf-8')
    
    waterfall_stats = {
        "raw_rows": initial_rows,
        "valid_identifier_rows": valid_rows,
        "invalid_dropped_rows": initial_rows - valid_rows,
        "deduplicated_physical_restaurants": len(df_clean),
        "multi_zone_duplicates_removed": valid_rows - len(df_clean),
        "unique_restaurant_brands": df_clean['name'].nunique(),
        "rated_restaurants_count": int(df_clean['rating'].notna().sum()),
        "unrated_cold_start_count": int(df_clean['rating'].isna().sum()),
        "cost_available_count": int(df_clean['cost_for_two_inr'].notna().sum())
    }
    
    return df_clean, waterfall_stats
