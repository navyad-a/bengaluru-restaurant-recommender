# -*- coding: utf-8 -*-
"""
Phase 6 — Collaborative Filtering SVD Sanity Checks & Qualitative Evaluation
Runs personalized SVD recommendations across diverse Indian dining personas and tests cold-start handling.
"""

import os
import sys
import time
import json
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.collaborative.collaborative_recommender import CollaborativeRecommender


def run_cf_evaluation():
    print("=" * 85)
    print("PHASE 6: COLLABORATIVE FILTERING SVD EVALUATION (SYNTHETIC BENCHMARK)")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_dir = os.path.join(base_dir, "saved_models", "collaborative_model")
    catalog_csv = os.path.join(base_dir, "data", "processed", "restaurants_clean.csv")
    users_csv = os.path.join(base_dir, "data", "processed", "synthetic_users.csv")
    train_csv = os.path.join(base_dir, "data", "processed", "synthetic_train_ratings.csv")
    meta_json = os.path.join(artifact_dir, "model_metadata.json")

    if not os.path.exists(meta_json):
        print(f"[ERROR] SVD model artifacts not found in: {artifact_dir}")
        print("Run python scripts/build_svd_model.py first.")
        return

    # 1. Print Model Metadata & Offline Metrics
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("\n[*] Model Architecture & Offline Test Evaluation:")
    print(f"  - Model Type           : {meta.get('model_type')}")
    print(f"  - Benchmark Notice     : {meta.get('dataset_notice')}")
    print(f"  - Hyperparameters      : {meta.get('hyperparameters')}")
    print(f"  - Known Users in Matrix: {meta.get('num_known_users'):,}")
    print(f"  - Known Outlets in SVD : {meta.get('num_known_restaurants'):,}")
    
    test_err = meta.get("test_error_metrics", {})
    test_rank = meta.get("test_ranking_metrics", {})
    print(f"\n  [Offline Test Holdout Metrics]:")
    print(f"  - Test RMSE            : {test_err.get('rmse')}")
    print(f"  - Test MAE             : {test_err.get('mae')}")
    print(f"  - Precision@5          : {test_rank.get('precision_at_5')}")
    print(f"  - Recall@5             : {test_rank.get('recall_at_5')}")
    print(f"  - Hit Rate@5           : {test_rank.get('hit_rate_at_5')}")
    print(f"  - Precision@10         : {test_rank.get('precision_at_10')}")
    print(f"  - Recall@10            : {test_rank.get('recall_at_10')}")
    print(f"  - Hit Rate@10          : {test_rank.get('hit_rate_at_10')}")

    # 2. Load Recommender Service
    print("\n[*] Initializing Collaborative Recommender Service...")
    t0 = time.time()
    recommender = CollaborativeRecommender.from_artifacts(
        artifact_dir=artifact_dir,
        catalog_csv_path=catalog_csv,
        train_ratings_csv_path=train_csv
    )
    load_time = (time.time() - t0) * 1000
    print(f"  [OK] Service loaded in {load_time:.2f} ms.")

    df_users = pd.read_csv(users_csv)
    df_train = pd.read_csv(train_csv)

    # 3. Qualitative Persona Recommendations
    sample_user_ids = [1, 10, 25, 50, 100]
    print("\n" + "-" * 85)
    print("SECTION 1: PERSONALIZED SVD RECOMMENDATIONS ACROSS SIMULATED PERSONAS")
    print("-" * 85)

    for u_id in sample_user_ids:
        user_row = df_users[df_users["user_id"] == u_id].iloc[0]
        user_train_count = len(df_train[df_train["user_id"] == u_id])

        t_rec = time.time()
        recs = recommender.recommend_for_user(user_id=u_id, top_k=5, exclude_rated=True)
        rec_time_ms = (time.time() - t_rec) * 1000

        print(f"\n[User ID {u_id}: {user_row['name']} ({user_row['persona_description']})]")
        print(f"  Home Area     : {user_row['home_locality']} | Preferred Cuisines: {user_row['preferred_cuisines']}")
        print(f"  Budget Tier   : {user_row['preferred_budget_tier']} (Max: Rs. {user_row['max_budget_inr']}) | Training Ratings: {user_train_count}")
        print(f"  Inference Time: {rec_time_ms:.2f} ms")
        print("  Top 5 SVD Recommendations (Unseen Outlets):")

        for idx, r in enumerate(recs, 1):
            print(
                f"    {idx}. {r['name']:<35} | SVD Pred: {r['predicted_rating']:.2f}★ | "
                f"{r['area']:<18} | {r['cuisines']:<28} | Rs.{r['cost_for_two_inr']}"
            )

    # 4. Cold-Start Handling Check
    print("\n" + "-" * 85)
    print("SECTION 2: COLD-START UNKNOWN USER HANDLING")
    print("-" * 85)
    unknown_id = 999999
    try:
        recommender.recommend_for_user(user_id=unknown_id, top_k=5)
        print(f"  [FAIL] Did not detect unknown user {unknown_id}")
    except KeyError as e:
        print(f"  [OK] Successfully caught cold-start user ID {unknown_id}:")
        print(f"       {e}")

    print("\n" + "=" * 85)
    print("PHASE 6 SANITY CHECKS & EVALUATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_cf_evaluation()
