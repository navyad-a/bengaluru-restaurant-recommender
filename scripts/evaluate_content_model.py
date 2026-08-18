# -*- coding: utf-8 -*-
"""
Phase 5 — Content-Based Recommendation Sanity Checks & Evaluation
Runs real-data qualitative checks across diverse Indian dining categories on the 12,481 catalog.
"""

import os
import sys
import time
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.content_based.content_recommender import ContentRecommender


def run_sanity_checks():
    print("=" * 85)
    print("PHASE 5: CONTENT-BASED RECOMMENDATION SANITY CHECKS (12,481 CATALOG)")
    print("=" * 85)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_dir = os.path.join(base_dir, "saved_models", "content_model")
    
    if not os.path.exists(os.path.join(artifact_dir, "tfidf_matrix.joblib")):
        print(f"[ERROR] Content model artifacts not found in: {artifact_dir}")
        print("Run python scripts/build_content_model.py first.")
        return
        
    print("[*] Loading Content Recommender from artifacts...")
    load_start = time.time()
    recommender = ContentRecommender.from_artifacts(artifact_dir)
    load_time = (time.time() - load_start) * 1000
    print(f"  [OK] Recommender loaded in {load_time:.2f} ms with {recommender.catalog_size:,} restaurants.\n")
    
    df_cat = recommender.engine.restaurant_catalog
    
    # -------------------------------------------------------------
    # SECTION 1: RESTAURANT-TO-RESTAURANT SIMILARITY CHECKS (MODE A)
    # -------------------------------------------------------------
    print("-" * 85)
    print("SECTION 1: RESTAURANT-TO-RESTAURANT SIMILARITY (MODE A)")
    print("-" * 85)
    
    # Sample queries across diverse categories
    test_queries = [
        ("South Indian & Tiffin", "Vidyarthi Bhavan"),
        ("Biryani Specialist", "Meghana Foods"),
        ("Microbrewery & Pub", "Toit"),
        ("Indo-Chinese & Asian", "Mainland China"),
        ("Cafe & Desserts", "Glen's Bakehouse"),
    ]
    
    for category, name_query in test_queries:
        matches = df_cat[df_cat["name"].str.contains(name_query, case=False, na=False)]
        if matches.empty:
            continue
        query_rest = matches.iloc[0]
        q_id = int(query_rest["restaurant_id"])
        
        t0 = time.time()
        recs = recommender.recommend_similar_restaurants(restaurant_id=q_id, top_k=5)
        latency_ms = (time.time() - t0) * 1000
        
        print(f"\n[Category: {category}]")
        print(f"  Query Restaurant : ID {q_id} | {query_rest['name']} ({query_rest['area']})")
        print(f"  Cuisines         : {query_rest['cuisines']}")
        print(f"  Dining Type      : {query_rest['rest_type']} | Price: {query_rest['price_tier']} (Rs. {query_rest['cost_for_two_inr']}) | Rating: {query_rest['rating']}★")
        print(f"  Inference Latency: {latency_ms:.2f} ms")
        print("  Top 5 Similar Outlets:")
        
        for idx, r in enumerate(recs, 1):
            print(f"    {idx}. {r['name']:<35} | Sim: {r['similarity_score']:.4f} | {r['area']:<18} | {r['cuisines']:<30} | Rs.{r['cost_for_two_inr']}")

    # -------------------------------------------------------------
    # SECTION 2: PREFERENCE-BASED MATCHING & FILTERING CHECKS (MODE B)
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("SECTION 2: USER PREFERENCE MATCHING & HARD FILTERING (MODE B)")
    print("-" * 85)
    
    pref_cases = [
        {
            "title": "Budget South Indian in Jayanagar",
            "prefs": {
                "preferred_cuisines": ["South Indian", "Karnataka"],
                "preferred_price_tier": "Budget",
                "max_cost_for_two": 350,
                "preferred_area": "Jayanagar"
            }
        },
        {
            "title": "Biryani & Kebabs under Rs. 800 in Koramangala",
            "prefs": {
                "preferred_cuisines": ["Biryani", "Mughlai"],
                "preferred_price_tier": "Moderate",
                "max_cost_for_two": 800,
                "preferred_area": "Koramangala 5th Block",
                "min_rating": 3.8
            }
        },
        {
            "title": "Cafe & Continental with Online Ordering in Indiranagar",
            "prefs": {
                "preferred_cuisines": ["Cafe", "Continental", "Desserts"],
                "preferred_type": "Cafe",
                "preferred_area": "Indiranagar",
                "online_order_only": True
            }
        }
    ]
    
    for case in pref_cases:
        p = case["prefs"]
        t0 = time.time()
        recs = recommender.recommend_for_preferences(preferences=p, top_k=5)
        latency_ms = (time.time() - t0) * 1000
        
        print(f"\n[Scenario: {case['title']}]")
        print(f"  Preferences      : {p}")
        print(f"  Inference Latency: {latency_ms:.2f} ms")
        print(f"  Top {len(recs)} Recommendations:")
        for idx, r in enumerate(recs, 1):
            print(f"    {idx}. {r['name']:<35} | Score: {r['content_score']:.4f} | {r['area']:<18} | {r['cuisines']:<30} | Rs.{r['cost_for_two_inr']} | {r['rating']}★")

    print("\n" + "=" * 85)
    print("PHASE 5 SANITY CHECKS & EVALUATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_sanity_checks()
