# -*- coding: utf-8 -*-
"""
Content Features & Token Normalization Module
=============================================
Transforms raw restaurant metadata and user preference profiles into structured,
prefix-isolated token documents to prevent semantic token collisions during TF-IDF vectorization.
"""

import re
import pandas as pd
from typing import Dict, Any, List, Optional


def clean_token(text: str) -> str:
    """
    Normalizes a text string into a clean lowercase underscore-separated token.
    Example: 'North Indian' -> 'north_indian', 'Cafe / Quick Bites' -> 'cafe_quick_bites'
    """
    if not text or pd.isna(text):
        return ""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return clean


def get_rating_bucket(rating: Optional[float]) -> str:
    """
    Maps a continuous rating [1.0, 5.0] into a categorical rating bucket token.
    """
    if rating is None or pd.isna(rating):
        return "rating_unrated"
    try:
        r = float(rating)
        if r >= 4.2:
            return "rating_exceptional"
        elif r >= 3.8:
            return "rating_high"
        elif r >= 3.3:
            return "rating_medium"
        else:
            return "rating_low"
    except (ValueError, TypeError):
        return "rating_unrated"


def get_cost_bucket(cost: Optional[int]) -> str:
    """
    Maps cost for two in INR into a categorical cost bracket token.
    """
    if cost is None or pd.isna(cost):
        return "cost_bracket_moderate"
    try:
        c = int(cost)
        if c < 300:
            return "cost_under_300"
        elif c < 600:
            return "cost_300_to_600"
        elif c < 1000:
            return "cost_600_to_1000"
        elif c < 1800:
            return "cost_1000_to_1800"
        else:
            return "cost_above_1800"
    except (ValueError, TypeError):
        return "cost_bracket_moderate"


def build_restaurant_feature_document(row: Dict[str, Any]) -> str:
    """
    Constructs a prefix-isolated feature document from a restaurant record.
    
    Feature Weights Applied via Token Replication:
    - Cuisines: 3x (Primary culinary taste driver)
    - Restaurant Types: 2x (Dining format driver)
    - Area / Locality: 2x (Neighborhood affinity)
    - Price Tier: 1x (Budget profile)
    - Rating Bucket: 1x (Quality prior)
    - Cost Bracket: 1x (Granular INR tier)
    - Service Flags: 1x (Online order / Table booking)
    - Dishes Liked: 1x (Signature menu items)
    """
    tokens: List[str] = []

    # 1. Cuisines (3x weight)
    cuisines_raw = str(row.get("cuisines", "") or "")
    if cuisines_raw and cuisines_raw != "nan":
        for item in cuisines_raw.split(","):
            c_tok = clean_token(item)
            if c_tok:
                token_str = f"cuisine_{c_tok}"
                tokens.extend([token_str] * 3)

    # 2. Restaurant Types (2x weight)
    rest_type_raw = str(row.get("rest_type", "") or "")
    if rest_type_raw and rest_type_raw != "nan":
        for item in rest_type_raw.split(","):
            t_tok = clean_token(item)
            if t_tok:
                token_str = f"type_{t_tok}"
                tokens.extend([token_str] * 2)

    # 3. Area / Locality (2x weight)
    area_raw = str(row.get("area", "") or "")
    if area_raw and area_raw != "nan":
        a_tok = clean_token(area_raw)
        if a_tok:
            token_str = f"area_{a_tok}"
            tokens.extend([token_str] * 2)

    # 4. Price Tier (1x weight)
    price_tier_raw = str(row.get("price_tier", "") or "")
    if price_tier_raw and price_tier_raw != "nan":
        p_tok = clean_token(price_tier_raw)
        if p_tok:
            tokens.append(f"price_{p_tok}")

    # 5. Rating Bucket (1x weight)
    rating_val = row.get("rating")
    tokens.append(get_rating_bucket(rating_val))

    # 6. Cost Bracket (1x weight)
    cost_val = row.get("cost_for_two_inr")
    tokens.append(get_cost_bucket(cost_val))

    # 7. Service Flags (1x weight)
    online_order = bool(row.get("online_order", False))
    tokens.append("online_order_yes" if online_order else "online_order_no")

    book_table = bool(row.get("book_table", False))
    tokens.append("book_table_yes" if book_table else "book_table_no")

    # 8. Dishes Liked (1x weight)
    dish_raw = str(row.get("dish_liked", "") or "")
    if dish_raw and dish_raw != "nan":
        for item in dish_raw.split(","):
            d_tok = clean_token(item)
            if d_tok:
                tokens.append(f"dish_{d_tok}")

    return " ".join(tokens)


def build_preference_feature_document(preferences: Dict[str, Any]) -> str:
    """
    Constructs a query document from user preference parameters using the EXACT same
    prefixed vocabulary format as the restaurant documents.
    """
    tokens: List[str] = []

    # 1. Preferred Cuisines (3x weight to match catalog representation)
    pref_cuisines = preferences.get("preferred_cuisines")
    if pref_cuisines:
        if isinstance(pref_cuisines, str):
            cuis_list = [c.strip() for c in pref_cuisines.split(",")]
        elif isinstance(pref_cuisines, list):
            cuis_list = pref_cuisines
        else:
            cuis_list = []
            
        for c in cuis_list:
            c_tok = clean_token(c)
            if c_tok:
                tokens.extend([f"cuisine_{c_tok}"] * 3)

    # 2. Preferred Restaurant Type (2x weight)
    pref_type = preferences.get("preferred_type") or preferences.get("restaurant_type")
    if pref_type:
        if isinstance(pref_type, str):
            types_list = [t.strip() for t in pref_type.split(",")]
        elif isinstance(pref_type, list):
            types_list = pref_type
        else:
            types_list = []
            
        for t in types_list:
            t_tok = clean_token(t)
            if t_tok:
                tokens.extend([f"type_{t_tok}"] * 2)

    # 3. Preferred Area / Locality (2x weight)
    pref_area = preferences.get("preferred_area") or preferences.get("area") or preferences.get("location")
    if pref_area:
        a_tok = clean_token(pref_area)
        if a_tok:
            tokens.extend([f"area_{a_tok}"] * 2)

    # 4. Preferred Price Tier (1x weight)
    pref_tier = preferences.get("preferred_price_tier") or preferences.get("price_tier")
    if pref_tier:
        p_tok = clean_token(pref_tier)
        if p_tok:
            tokens.append(f"price_{p_tok}")

    # 5. Service Preferences (1x weight)
    if preferences.get("online_order_only") is True:
        tokens.append("online_order_yes")
    if preferences.get("book_table_only") is True:
        tokens.append("book_table_yes")

    # 6. Max Cost Bracket
    max_cost = preferences.get("max_cost_for_two")
    if max_cost is not None:
        tokens.append(get_cost_bucket(max_cost))

    # Fallback to general token if completely empty
    if not tokens:
        tokens.append("rating_high")

    return " ".join(tokens)
