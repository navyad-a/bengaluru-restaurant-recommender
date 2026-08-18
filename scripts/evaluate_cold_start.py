# -*- coding: utf-8 -*-
"""
Phase 9 — Cold-Start Strategy & Fallback Intelligence Evaluation
================================================================
Evaluates user routing hierarchy, Bayesian popularity priors, onboarding
questionnaire bootstrapping, unrated item imputation, and explainability.
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

from ml.cold_start.routing import ColdStartRouter, ColdStartStrategy
from ml.cold_start.popularity import BayesianPopularityEngine
from ml.cold_start.onboarding import OnboardingQuestionnaire, OnboardingPreferenceHandler
from ml.cold_start.item_cold_start import ItemColdStartHandler
from ml.cold_start.explainer import ColdStartExplainer
from app.services.recommendation_service import get_hybrid_recommender


def run_cold_start_evaluation():
    print("=" * 95)
    print("PHASE 9: COLD-START STRATEGY & FALLBACK INTELLIGENCE EVALUATION")
    print("=" * 95)

    hybrid_engine = get_hybrid_recommender()
    print(f"[*] Loaded Hybrid Engine with Bayesian Popularity & Cold-Start Router.\n")

    # -------------------------------------------------------------
    # SECTION 1: COLD-START STRATEGY ROUTING VERIFICATION
    # -------------------------------------------------------------
    print("-" * 95)
    print("SECTION 1: DECISION ROUTING TAXONOMY VERIFICATION")
    print("-" * 95)

    test_cases = [
        ("Known Warm User (15 ratings, with Coords)", {"user_id": 1, "user_interaction_count": 15, "has_location": True, "is_known_collaborative_user": True}),
        ("Known Sparse User (2 ratings, No Coords)", {"user_id": 2, "user_interaction_count": 2, "has_location": False, "is_known_collaborative_user": True}),
        ("Unknown User with Cuisines & Budget", {"user_id": None, "has_preferences": True, "has_location": False}),
        ("Unknown User with Location Only (Indiranagar)", {"user_id": None, "has_preferences": False, "has_location": True}),
        ("Complete Cold-Start (No Context)", {"user_id": None, "has_preferences": False, "has_location": False})
    ]

    for label, kwargs in test_cases:
        strat = ColdStartRouter.determine_strategy(**kwargs)
        weights = ColdStartRouter.get_strategy_weights(strat, has_location=kwargs.get("has_location", False))
        print(f"  [>] {label:<45} -> Strategy: {strat.value:<25} | Weights: {weights}")

    # -------------------------------------------------------------
    # SECTION 2: BAYESIAN POPULARITY ENGINE (GLOBAL & LOCALITY)
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("SECTION 2: BAYESIAN POPULARITY PRIORS (GLOBAL & LOCALITY CLUSTERS)")
    print("-" * 95)

    pop_engine = hybrid_engine.popularity_engine

    # 1. Global Bengaluru Top 3
    print("\n[A. Global Bengaluru Top Popular Outlets]")
    global_top = pop_engine.get_global_popular(top_k=3)
    for idx, r in enumerate(global_top, 1):
        print(f"  {idx}. {r['name']:<35} | Pop Score: {r['popularity_score']:.4f} | {r['rating']}★ ({r['review_count']:,} votes) | {r['area']} | Rs. {r['cost_for_two_inr']}")

    # 2. Locality Top 3 (Koramangala)
    print("\n[B. Locality Popular: Koramangala]")
    kora_top = pop_engine.get_locality_popular(area="Koramangala", top_k=3)
    for idx, r in enumerate(kora_top, 1):
        print(f"  {idx}. {r['name']:<35} | Pop Score: {r['popularity_score']:.4f} | {r['rating']}★ ({r['review_count']:,} votes) | {r['area']} | Rs. {r['cost_for_two_inr']}")

    # 3. Cuisine Top 3 (South Indian)
    print("\n[C. Cuisine Popular: South Indian / Karnataka]")
    south_top = pop_engine.get_cuisine_popular(cuisine="South Indian", top_k=3)
    for idx, r in enumerate(south_top, 1):
        print(f"  {idx}. {r['name']:<35} | Pop Score: {r['popularity_score']:.4f} | {r['rating']}★ ({r['review_count']:,} votes) | {r['area']} | Rs. {r['cost_for_two_inr']}")

    # -------------------------------------------------------------
    # SECTION 3: ONBOARDING QUESTIONNAIRE BOOTSTRAPPING
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("SECTION 3: ONBOARDING QUESTIONNAIRE ZERO-HISTORY BOOTSTRAPPING")
    print("-" * 95)

    q = OnboardingQuestionnaire(
        favorite_cuisines=["Biryani", "Andhra", "Mughlai"],
        preferred_dining_types=["Casual Dining"],
        preferred_area="Koramangala 5th Block",
        price_tier="Moderate",
        max_budget_for_two=700,
        online_ordering=True
    )
    prefs = OnboardingPreferenceHandler.build_preference_payload(q)
    print(f"  Constructed Preferences from Questionnaire: {prefs}\n")

    t0 = time.time()
    onboard_res = hybrid_engine.recommend(
        user_id=None,
        preferences=prefs,
        filters={"max_cost_for_two": 700, "online_order_only": True, "area": "Koramangala 5th Block"},
        top_k=3
    )
    lat_ms = (time.time() - t0) * 1000

    print(f"  Strategy: {onboard_res['strategy']} | Latency: {lat_ms:.2f} ms")
    for idx, r in enumerate(onboard_res["recommendations"], 1):
        print(f"    {idx}. {r['name']:<32} | Hybrid: {r['hybrid_score']:.4f} | Rs. {r['cost_for_two_inr']} | {r['cuisines']}")
        print(f"       Expl: {r['explanation']}")

    # -------------------------------------------------------------
    # SECTION 4: ITEM COLD-START (UNRATED RESTAURANT IMPUTATION)
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("SECTION 4: ITEM COLD-START (UNRATED RESTAURANT PRIOR IMPUTATION)")
    print("-" * 95)

    handler = ItemColdStartHandler(hybrid_engine.df_restaurants)
    unrated_sample = hybrid_engine.df_restaurants[hybrid_engine.df_restaurants["rating"].isna()].head(3)
    
    for _, row in unrated_sample.iterrows():
        enriched = handler.enrich_item_metadata(row.to_dict())
        print(f"  Outlet: {enriched['name']:<32} | Area: {enriched['area']:<15} | Is Unrated: {enriched['is_unrated']} | Imputed Prior: {enriched['imputed_rating_prior']}★")

    print("\n" + "=" * 95)
    print("PHASE 9 COLD-START EVALUATION COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    run_cold_start_evaluation()
