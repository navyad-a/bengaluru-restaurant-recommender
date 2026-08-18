# -*- coding: utf-8 -*-
"""
Phase 6 — Surprise SVD Model Build & Benchmark Evaluation Script
===============================================================
Performs data integrity verification, K-fold cross-validation hyperparameter search,
final model training, test holdout evaluation, and artifact serialization.
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
from ml.collaborative.svd_engine import SVDEngine
from ml.collaborative.evaluator import (
    validate_benchmark_integrity,
    compute_prediction_error_metrics,
    compute_top_k_ranking_metrics
)


def main():
    print("=" * 85)
    print("PHASE 6: SURPRISE SVD MATRIX FACTORIZATION — BUILD & BENCHMARK EVALUATION")
    print("=" * 85)
    
    total_start = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    artifact_dir = os.path.join(base_dir, "saved_models", "collaborative_model")

    users_csv = os.path.join(data_dir, "synthetic_users.csv")
    rest_csv = os.path.join(data_dir, "restaurants_clean.csv")
    train_csv = os.path.join(data_dir, "synthetic_train_ratings.csv")
    test_csv = os.path.join(data_dir, "synthetic_test_ratings.csv")

    # 1. Load datasets
    print("[*] Step 1: Loading synthetic benchmark & authentic restaurant catalog...")
    df_users = pd.read_csv(users_csv)
    df_rest = pd.read_csv(rest_csv)
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # 2. Validate Benchmark Integrity
    print("\n[*] Step 2: Running Benchmark Data Integrity Validation...")
    stats = validate_benchmark_integrity(df_users, df_rest, df_train, df_test)
    print(f"  [OK] Simulated Users Count         : {stats['total_users']:,}")
    print(f"  [OK] Authentic Catalog Outlets     : {stats['authentic_catalog_restaurants']:,}")
    print(f"  [OK] Represented Outlets in SVD    : {stats['represented_restaurants']:,}")
    print(f"  [OK] Training Ratings (80% holdout): {stats['train_ratings_count']:,}")
    print(f"  [OK] Test Ratings (20% holdout)    : {stats['test_ratings_count']:,}")
    print(f"  [OK] Avg Ratings per User (Train)  : {stats['avg_train_ratings_per_user']}")
    print(f"  [OK] Matrix Sparsity               : {stats['matrix_sparsity_percent']}")
    print(f"  [OK] Training / Testing Overlap    : None (0 collisions)")

    # 3. Hyperparameter Experimentation on Training Data
    print("\n[*] Step 3: Running 3-Fold Cross-Validation Hyperparameter Search (Training Set Only)...")
    cv_start = time.time()
    cv_results = SVDEngine.cross_validate_hyperparameters(
        df_train=df_train,
        param_grid={
            "n_factors": [50, 100, 150],
            "n_epochs": [10, 20, 30],
            "reg_all": [0.02, 0.05, 0.10]
        },
        n_splits=3,
        random_state=42
    )
    cv_time = time.time() - cv_start
    best_params = cv_results["best_params"]
    
    print(f"  [+] Completed {len(cv_results['all_experiments'])} configurations in {cv_time:.2f} seconds.")
    print(f"  [+] Best Configuration: n_factors={best_params['n_factors']}, n_epochs={best_params['n_epochs']}, reg_all={best_params['reg_all']}")
    print(f"  [+] Best Validation RMSE: {cv_results['best_val_rmse']:.4f}")

    # 4. Final SVD Model Training
    print("\n[*] Step 4: Training Final SVD Model on Full Training Set...")
    train_start = time.time()
    engine = SVDEngine(
        n_factors=best_params["n_factors"],
        n_epochs=best_params["n_epochs"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
        random_state=best_params["random_state"]
    )
    engine.fit(df_train)
    train_time = time.time() - train_start
    print(f"  [OK] SVD Training Complete in {train_time:.3f} seconds (Global Rating Mean: {engine.global_mean:.3f})")

    # 5. Offline Test Evaluation
    print("\n[*] Step 5: Evaluating Final SVD Model on Test Holdout Partition...")
    # A. Rating Prediction Errors
    test_preds = []
    for _, row in df_test.iterrows():
        p = engine.predict(int(row["user_id"]), int(row["restaurant_id"]))
        test_preds.append((float(row["rating"]), p))
        
    error_metrics = compute_prediction_error_metrics(test_preds)
    print(f"  [+] Test RMSE : {error_metrics['rmse']:.4f}")
    print(f"  [+] Test MAE  : {error_metrics['mae']:.4f}")

    # B. Top-K Ranking Evaluation
    print("\n[*] Step 6: Computing Top-K Recommendation Metrics against Test Holdout...")
    ranking_start = time.time()
    catalog_ids = df_rest["restaurant_id"].tolist()
    ranking_metrics = compute_top_k_ranking_metrics(
        svd_engine=engine,
        df_train=df_train,
        df_test=df_test,
        catalog_restaurant_ids=catalog_ids,
        k_values=[5, 10],
        rating_threshold=3.5
    )
    ranking_time = time.time() - ranking_start

    print(f"  [+] Precision@5  : {ranking_metrics['precision_at_5']:.4f}")
    print(f"  [+] Recall@5     : {ranking_metrics['recall_at_5']:.4f}")
    print(f"  [+] Hit Rate@5   : {ranking_metrics['hit_rate_at_5']:.4f}")
    print(f"  [+] Precision@10 : {ranking_metrics['precision_at_10']:.4f}")
    print(f"  [+] Recall@10    : {ranking_metrics['recall_at_10']:.4f}")
    print(f"  [+] Hit Rate@10  : {ranking_metrics['hit_rate_at_10']:.4f}")
    print(f"  [+] Ranking Eval Time: {ranking_time:.2f} seconds ({ranking_metrics['num_evaluated_users']} users evaluated)")

    # 6. Save Model Artifacts
    print(f"\n[*] Step 7: Serializing SVD artifacts to: {artifact_dir}")
    eval_summary = {
        "dataset_statistics": stats,
        "best_hyperparameters": best_params,
        "validation_rmse": cv_results["best_val_rmse"],
        "test_error_metrics": error_metrics,
        "test_ranking_metrics": ranking_metrics,
        "training_time_seconds": round(train_time, 4),
        "cv_search_time_seconds": round(cv_time, 4)
    }
    saved_paths = engine.save_artifacts(artifact_dir, metadata=eval_summary)
    for key, path in saved_paths.items():
        sz_kb = round(os.path.getsize(path) / 1024, 1)
        print(f"  [OK] Saved {key:<10}: {os.path.basename(path):<25} ({sz_kb} KB)")

    total_time = time.time() - total_start
    print("\n" + "=" * 85)
    print(f"PHASE 6 BUILD & EVALUATION COMPLETE (Total Elapsed: {total_time:.2f} seconds)")
    print("=" * 85)


if __name__ == "__main__":
    main()
