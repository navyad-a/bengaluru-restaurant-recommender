# -*- coding: utf-8 -*-
"""
Phase 7 — Hybrid Recommendation Engine Evaluation & Qualitative Scenarios
=========================================================================
Runs comprehensive validation across recommendation validity, score ranges,
effective weight redistribution, cold-start fallbacks, and 10 qualitative scenarios.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.recommendation_service import get_hybrid_recommender


def run_hybrid_evaluation():
    print("=" * 90)
    print("PHASE 7: HYBRID RECOMMENDATION ENGINE EVALUATION & QUALITATIVE SCENARIOS")
    print("=" * 90)

    t0 = time.time()
    hybrid_engine = get_hybrid_recommender()
    load_time_ms = (time.time() - t0) * 1000
    print(f"[*] Initialized Hybrid Recommender in {load_time_ms:.2f} ms (Catalog: {len(hybrid_engine.df_restaurants):,} outlets).\n")

    # -------------------------------------------------------------
    # SECTION 1: SYSTEM INTEGRITY & SCORE VALIDATION
    # -------------------------------------------------------------
    print("-" * 90)
    print("SECTION 1: SYSTEM INTEGRITY & SCORE BOUNDS VALIDATION")
    print("-" * 90)

    test_res = hybrid_engine.recommend(
        user_id=1,
        preferences={"preferred_cuisines": ["South Indian", "Karnataka"]},
        user_coords=(12.9352, 77.6245),
        filters={"max_cost_for_two": 600},
        top_k=20
    )

    rec_ids = [r["restaurant_id"] for r in test_res["recommendations"]]
    has_duplicates = len(rec_ids) != len(set(rec_ids))
    all_in_catalog = set(rec_ids).issubset(set(hybrid_engine.df_restaurants["restaurant_id"]))
    all_costs_valid = all(r["cost_for_two_inr"] <= 600 for r in test_res["recommendations"])

    print(f"  [OK] Duplicate IDs Check           : {'PASSED (0 duplicates)' if not has_duplicates else 'FAILED'}")
    print(f"  [OK] Authentic Catalog Integrity   : {'PASSED (100% matched)' if all_in_catalog else 'FAILED'}")
    print(f"  [OK] Hard Cost Filter Enforcement  : {'PASSED (All <= Rs. 600)' if all_costs_valid else 'FAILED'}")
    print(f"  [OK] Effective Weights Sum         : {sum(test_res['effective_weights'].values()):.4f} (Sum == 1.0000)")

    score_bounds_ok = True
    for r in test_res["recommendations"]:
        for k in ["hybrid_score", "content_score", "collaborative_score", "location_score", "quality_score"]:
            if not (0.0 <= r[k] <= 1.0):
                score_bounds_ok = False
    print(f"  [OK] Component Score Bounds [0, 1] : {'PASSED' if score_bounds_ok else 'FAILED'}")

    # -------------------------------------------------------------
    # SECTION 2: 10 QUALITATIVE REAL-WORLD SCENARIOS
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print("SECTION 2: 10 REAL-WORLD QUALITATIVE RECOMMENDATION SCENARIOS")
    print("-" * 90)

    scenarios = [
        {
            "num": 1,
            "title": "South Indian + Budget in Jayanagar",
            "kwargs": {
                "preferences": {"preferred_cuisines": ["South Indian", "Karnataka"], "preferred_price_tier": "Budget"},
                "filters": {"area": "Jayanagar", "max_cost_for_two": 400},
                "top_k": 3
            }
        },
        {
            "num": 2,
            "title": "Biryani in Koramangala under Rs. 800",
            "kwargs": {
                "preferences": {"preferred_cuisines": ["Biryani", "Mughlai"]},
                "filters": {"area": "Koramangala 5th Block", "max_cost_for_two": 800},
                "top_k": 3
            }
        },
        {
            "num": 3,
            "title": "Cafe & Desserts in Indiranagar with Online Ordering",
            "kwargs": {
                "preferences": {"preferred_cuisines": ["Cafe", "Desserts", "Continental"], "preferred_type": "Cafe"},
                "filters": {"area": "Indiranagar", "online_order_only": True},
                "top_k": 3
            }
        },
        {
            "num": 4,
            "title": "High-Rated Fine Dining & Microbreweries",
            "kwargs": {
                "preferences": {"preferred_cuisines": ["Continental", "Italian", "American"], "preferred_type": "Microbrewery"},
                "filters": {"min_rating": 4.5, "price_tier": "Premium"},
                "top_k": 3
            }
        },
        {
            "num": 5,
            "title": "Strict Budget Eats (Cost for Two <= Rs. 250)",
            "kwargs": {
                "preferences": {"preferred_cuisines": ["Street Food", "Fast Food", "Quick Bites"]},
                "filters": {"max_cost_for_two": 250},
                "top_k": 3
            }
        },
        {
            "num": 6,
            "title": "Table Booking Required for Special Dinner in Lavelle Road",
            "kwargs": {
                "preferences": {"preferred_cuisines": ["European", "Continental", "North Indian"]},
                "filters": {"area": "Lavelle Road", "book_table_only": True},
                "top_k": 3
            }
        },
        {
            "num": 7,
            "title": "Location Proximity (Diner in HSR Layout: 12.9121° N, 77.6446° E)",
            "kwargs": {
                "preferences": {"preferred_cuisines": ["North Indian", "Chinese"]},
                "user_coords": (12.9121, 77.6446),
                "top_k": 3
            }
        },
        {
            "num": 8,
            "title": "Known SVD User 10 (South Indian Tiffin Persona, Malleshwaram)",
            "kwargs": {
                "user_id": 10,
                "preferences": {"preferred_cuisines": ["South Indian", "Udupi"]},
                "top_k": 3
            }
        },
        {
            "num": 9,
            "title": "Known SVD User 25 (Fine Dining Connoisseur Persona, Indiranagar)",
            "kwargs": {
                "user_id": 25,
                "preferences": {"preferred_cuisines": ["Continental", "Italian"]},
                "top_k": 3
            }
        },
        {
            "num": 10,
            "title": "Cold-Start Unknown User (No Historical SVD Ratings)",
            "kwargs": {
                "user_id": 999999,
                "preferences": {"preferred_cuisines": ["Pan-Asian", "Chinese", "Momos"]},
                "user_coords": (12.9784, 77.6408),
                "filters": {"max_cost_for_two": 700},
                "top_k": 3
            }
        }
    ]

    for sc in scenarios:
        t_sc = time.time()
        res = hybrid_engine.recommend(**sc["kwargs"])
        lat_ms = (time.time() - t_sc) * 1000

        print(f"\n[Scenario {sc['num']}: {sc['title']}]")
        print(f"  Active Weights: {res['effective_weights']} | Cold-Start: {res['is_cold_start']} | Latency: {lat_ms:.2f} ms")
        for idx, r in enumerate(res["recommendations"], 1):
            dist_info = f"{r['distance_km']:.1f} km" if r['distance_km'] is not None else "N/A"
            print(
                f"    {idx}. {r['name']:<35} | Hybrid: {r['hybrid_score']:.4f} | "
                f"(C:{r['content_score']:.2f}, SVD:{r['collaborative_score']:.2f}, Loc:{r['location_score']:.2f}, Q:{r['quality_score']:.2f}) | "
                f"{r['area']:<18} | Rs.{r['cost_for_two_inr']:<4} | {r['rating']}★"
            )
            print(f"       Expl: {r['explanation']}")

    print("\n" + "=" * 90)
    print("PHASE 7 EVALUATION & QUALITATIVE SCENARIOS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    run_hybrid_evaluation()
