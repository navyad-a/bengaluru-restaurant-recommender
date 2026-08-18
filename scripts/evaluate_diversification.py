# -*- coding: utf-8 -*-
"""
Phase 10 — Recommendation Diversification (MMR) & Explainability Evaluation
=============================================================================
Runs lambda hyperparameter trade-off analysis, 10 qualitative scenario evaluations
(Before vs After MMR), and performance latency benchmarks across Top-K = 5, 10, 20.
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
from ml.diversification.diversity_metrics import DiversityMetricsCalculator


def run_diversification_evaluation():
    print("=" * 105, flush=True)
    print("PHASE 10: RECOMMENDATION DIVERSIFICATION (MMR) & EXPLAINABILITY EVALUATION", flush=True)
    print("=" * 105, flush=True)

    hybrid_engine = get_hybrid_recommender()
    print("[*] Loaded Hybrid Recommender with Sparse Similarity & MMR Diversification Engine.\n", flush=True)

    # -------------------------------------------------------------
    # SECTION 1: LAMBDA HYPERPARAMETER EXPERIMENT
    # -------------------------------------------------------------
    print("-" * 105, flush=True)
    print("SECTION 1: LAMBDA (λ) HYPERPARAMETER TRADE-OFF EXPERIMENT (Top-K = 10)", flush=True)
    print("-" * 105, flush=True)

    lambdas = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
    sample_queries = [
        {"preferred_cuisines": ["Biryani", "North Indian"], "area": "Koramangala 5th Block"},
        {"preferred_cuisines": ["South Indian", "Karnataka"], "area": "Jayanagar"},
        {"preferred_cuisines": ["Cafe", "Desserts", "Continental"], "area": "Indiranagar"},
        {"preferred_cuisines": ["Chinese", "Indo-Chinese"], "area": "Whitefield"},
        {"user_id": 1}
    ]

    results_table = []

    for lam in lambdas:
        rel_list = []
        top1_rel_list = []
        ild_list = []
        avg_sim_list = []
        redundancy_list = []
        cuisine_ratio_list = []
        locality_ratio_list = []
        type_ratio_list = []
        latency_list = []

        for q in sample_queries:
            t0 = time.perf_counter()
            res = hybrid_engine.recommend(
                user_id=q.get("user_id"),
                preferences={"preferred_cuisines": q.get("preferred_cuisines")} if "preferred_cuisines" in q else None,
                filters={"area": q.get("area")} if "area" in q else None,
                mmr_enabled=(lam < 1.0),
                mmr_lambda=lam,
                top_k=10
            )
            lat_ms = (time.perf_counter() - t0) * 1000

            metrics = res["diversification"]["diversity_metrics"]
            recs = res["recommendations"]

            if recs:
                top1_rel_list.append(recs[0]["hybrid_score"])
                rel_list.append(metrics["mean_relevance"])
                ild_list.append(metrics["intra_list_diversity"])
                avg_sim_list.append(metrics["avg_pairwise_similarity"])
                redundancy_list.append(metrics["redundancy_rate"])
                cuisine_ratio_list.append(metrics["unique_cuisine_ratio"])
                locality_ratio_list.append(metrics["unique_locality_ratio"])
                type_ratio_list.append(metrics["unique_restaurant_type_ratio"])
                latency_list.append(lat_ms)

        results_table.append({
            "Lambda (λ)": lam,
            "Mean Rel": np.mean(rel_list),
            "Top-1 Rel": np.mean(top1_rel_list),
            "ILD": np.mean(ild_list),
            "Avg Sim": np.mean(avg_sim_list),
            "Redundancy": np.mean(redundancy_list),
            "Cuisine Div": np.mean(cuisine_ratio_list),
            "Locality Div": np.mean(locality_ratio_list),
            "Type Div": np.mean(type_ratio_list),
            "MMR Latency": np.mean(latency_list)
        })

    df_lam = pd.DataFrame(results_table)
    print(f"{'Lambda (λ)':<10} | {'Mean Rel':<9} | {'Top-1 Rel':<10} | {'ILD':<8} | {'Avg Sim':<8} | {'Redundancy':<10} | {'Cuisine Div':<11} | {'Latency (ms)':<12}", flush=True)
    print("-" * 105, flush=True)
    for _, r in df_lam.iterrows():
        print(f"{r['Lambda (λ)']:<10.2f} | {r['Mean Rel']:<9.4f} | {r['Top-1 Rel']:<10.4f} | {r['ILD']:<8.4f} | {r['Avg Sim']:<8.4f} | {r['Redundancy']:<10.2%} | {r['Cuisine Div']:<11.2f} | {r['MMR Latency']:<12.2f}", flush=True)

    # -------------------------------------------------------------
    # SECTION 2: 10 QUALITATIVE SCENARIOS (BEFORE VS AFTER MMR)
    # -------------------------------------------------------------
    print("\n" + "-" * 105, flush=True)
    print("SECTION 2: 10 QUALITATIVE EVALUATION SCENARIOS (BEFORE VS AFTER MMR)", flush=True)
    print("-" * 105, flush=True)

    scenarios = [
        ("1. South Indian Budget", {"preferred_cuisines": ["South Indian", "Karnataka"], "max_cost_for_two": 300, "area": "Basavanagudi"}),
        ("2. Biryani Enthusiast", {"preferred_cuisines": ["Biryani", "Andhra", "Mughlai"], "area": "Koramangala 5th Block"}),
        ("3. Cafe + Desserts", {"preferred_cuisines": ["Cafe", "Desserts", "Bakery"], "area": "Indiranagar"}),
        ("4. Fine Dining", {"preferred_cuisines": ["North Indian", "Mughlai"], "preferred_price_tier": "Luxury", "max_cost_for_two": 2500}),
        ("5. Microbrewery & Pub", {"preferred_cuisines": ["Finger Food", "Continental"], "preferred_type": "Microbrewery", "area": "Indiranagar"}),
        ("6. Strict Budget (< ₹300)", {"max_cost_for_two": 300, "preferred_area": "BTM"}),
        ("7. Table Booking Outlets", {"book_table_only": True, "preferred_cuisines": ["North Indian", "Chinese"], "area": "Whitefield"}),
        ("8. Location-Constrained User", {"preferred_cuisines": ["Biryani"], "latitude": 12.9352, "longitude": 77.6245, "radius_km": 3.0}),
        ("9. Known Collaborative SVD User", {"user_id": 1}),
        ("10. Unknown Cold-Start User", {"preferred_cuisines": ["Chettinad", "South Indian"], "area": "Indiranagar"})
    ]

    for title, q_params in scenarios:
        print(f"\n>>> SCENARIO: {title} <<<", flush=True)
        
        # A. Pre-MMR (Pure Relevance)
        res_pre = hybrid_engine.recommend(
            user_id=q_params.get("user_id"),
            preferences={k: v for k, v in q_params.items() if k.startswith("preferred_")},
            user_coords=(q_params["latitude"], q_params["longitude"]) if "latitude" in q_params else None,
            filters={k: v for k, v in q_params.items() if not k.startswith("preferred_") and k not in ["user_id", "latitude", "longitude"]},
            mmr_enabled=False,
            top_k=3
        )
        
        # B. Post-MMR (Diversified, lambda = 0.75)
        res_post = hybrid_engine.recommend(
            user_id=q_params.get("user_id"),
            preferences={k: v for k, v in q_params.items() if k.startswith("preferred_")},
            user_coords=(q_params["latitude"], q_params["longitude"]) if "latitude" in q_params else None,
            filters={k: v for k, v in q_params.items() if not k.startswith("preferred_") and k not in ["user_id", "latitude", "longitude"]},
            mmr_enabled=True,
            mmr_lambda=0.75,
            top_k=3
        )

        print("  [Before MMR - Pure Relevance Top-3]:", flush=True)
        for idx, r in enumerate(res_pre["recommendations"], 1):
            print(f"    {idx}. {r['name']:<32} | Score: {r['hybrid_score']:.4f} | {r['cuisines'][:40]} | {r['area']}", flush=True)
        m_pre = res_pre["diversification"]["diversity_metrics"]
        print(f"    -> ILD: {m_pre['intra_list_diversity']:.4f} | Avg Sim: {m_pre['avg_pairwise_similarity']:.4f} | Cuisines: {m_pre['unique_cuisine_ratio']:.2f}", flush=True)

        print("  [After MMR (λ=0.75) - Diversified Top-3]:", flush=True)
        for idx, r in enumerate(res_post["recommendations"], 1):
            print(f"    {idx}. {r['name']:<32} | Score: {r['hybrid_score']:.4f} (MMR: {r.get('mmr_score', 0):.4f}) | {r['cuisines'][:40]} | {r['area']}", flush=True)
            print(f"       Expl: {r['explanation']}", flush=True)
        m_post = res_post["diversification"]["diversity_metrics"]
        print(f"    -> ILD: {m_post['intra_list_diversity']:.4f} | Avg Sim: {m_post['avg_pairwise_similarity']:.4f} | Cuisines: {m_post['unique_cuisine_ratio']:.2f} | Rel Retention: {m_post['relevance_retention_pct']:.1f}%", flush=True)

    # -------------------------------------------------------------
    # SECTION 3: PERFORMANCE BENCHMARK (Top-K = 5, 10, 20)
    # -------------------------------------------------------------
    print("\n" + "-" * 105, flush=True)
    print("SECTION 3: PERFORMANCE LATENCY BENCHMARK (PRE-MMR VS POST-MMR)", flush=True)
    print("-" * 105, flush=True)

    n_runs = 10
    print(f"{'Top-K':<8} | {'Pre-MMR Latency (ms)':<22} | {'MMR Latency (ms)':<18} | {'Total Latency (ms)':<20} | {'MMR Overhead':<12}", flush=True)
    print("-" * 105, flush=True)

    for k_val in [5, 10, 20]:
        t_pre_list = []
        t_post_list = []

        for _ in range(n_runs):
            # Pre-MMR
            t0 = time.perf_counter()
            _ = hybrid_engine.recommend(
                preferences={"preferred_cuisines": ["North Indian", "Biryani"]},
                filters={"area": "Koramangala 5th Block"},
                mmr_enabled=False,
                top_k=k_val
            )
            t_pre_list.append((time.perf_counter() - t0) * 1000)

            # Post-MMR
            t0 = time.perf_counter()
            _ = hybrid_engine.recommend(
                preferences={"preferred_cuisines": ["North Indian", "Biryani"]},
                filters={"area": "Koramangala 5th Block"},
                mmr_enabled=True,
                mmr_lambda=0.75,
                top_k=k_val
            )
            t_post_list.append((time.perf_counter() - t0) * 1000)

        mean_pre = np.mean(t_pre_list)
        mean_post = np.mean(t_post_list)
        mmr_delta = mean_post - mean_pre
        overhead_pct = (mmr_delta / mean_pre) * 100

        print(f"{k_val:<8} | {mean_pre:<22.2f} | {mmr_delta:<18.2f} | {mean_post:<20.2f} | +{overhead_pct:<11.1f}%", flush=True)

    print("\n" + "=" * 105, flush=True)
    print("PHASE 10 DIVERSIFICATION EVALUATION COMPLETE", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    run_diversification_evaluation()
