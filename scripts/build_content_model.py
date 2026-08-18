# -*- coding: utf-8 -*-
"""
Phase 5 — Content Model Build & Serialization Script
Fits TF-IDF on the 12,481 Bengaluru restaurant catalog and serializes artifacts.
"""

import os
import sys
import time
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.content_based.tfidf_engine import TfidfEngine
from ml.content_based.content_recommender import ContentRecommender


def main():
    print("=" * 80)
    print("PHASE 5 — CONTENT MODEL BUILD & ARTIFACT SERIALIZATION")
    print("=" * 80)
    
    start_time = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(base_dir, "data", "processed", "restaurants_clean.csv")
    artifact_dir = os.path.join(base_dir, "saved_models", "content_model")
    
    # 1. Load authoritative restaurant catalog
    if not os.path.exists(catalog_path):
        print(f"[ERROR] Clean restaurant catalog not found at: {catalog_path}")
        sys.exit(1)
        
    print(f"[*] Loading restaurant catalog from: {catalog_path}")
    df_restaurants = pd.read_csv(catalog_path)
    num_loaded = len(df_restaurants)
    
    # Validate expected catalog count
    EXPECTED_CATALOG_COUNT = 12481
    if num_loaded != EXPECTED_CATALOG_COUNT:
        print(f"[ERROR] Catalog row count mismatch! Expected {EXPECTED_CATALOG_COUNT:,}, found {num_loaded:,}.")
        sys.exit(1)
    print(f"  [OK] Verified authoritative catalog count: {num_loaded:,} physical restaurant outlets.")
    
    # 2. Initialize and fit TF-IDF Engine
    print("\n[*] Constructing prefix-isolated feature documents and fitting TF-IDF...")
    fit_start = time.time()
    engine = TfidfEngine(min_df=2, sublinear_tf=True)
    engine.fit(df_restaurants)
    fit_time = time.time() - fit_start
    
    vocab_size = len(engine.vectorizer.vocabulary_)
    matrix_shape = engine.tfidf_matrix.shape
    
    print(f"  [OK] Feature documents created : {num_loaded:,}")
    print(f"  [OK] Vocabulary size           : {vocab_size:,} unique tokens")
    print(f"  [OK] Sparse matrix shape       : {matrix_shape[0]:,} rows x {matrix_shape[1]:,} columns")
    print(f"  [OK] Sparse matrix format      : {type(engine.tfidf_matrix).__name__} ({engine.tfidf_matrix.nnz:,} non-zero entries)")
    print(f"  [OK] TF-IDF Fit duration       : {fit_time:.3f} seconds")
    
    # 3. Serialize artifacts to disk
    print(f"\n[*] Saving model artifacts to: {artifact_dir}")
    save_start = time.time()
    saved_paths = engine.save_artifacts(artifact_dir)
    save_time = time.time() - save_start
    
    for key, path in saved_paths.items():
        size_kb = round(os.path.getsize(path) / 1024, 1)
        print(f"  [OK] Saved {key:<12} : {os.path.basename(path):<30} ({size_kb} KB)")
        
    # 4. Model Artifact Validation (Load & Test Inference)
    print("\n[*] Validating artifact deserialization and production inference...")
    val_start = time.time()
    recommender = ContentRecommender.from_artifacts(artifact_dir)
    assert recommender.is_ready, "Loaded recommender is not in ready state!"
    assert recommender.catalog_size == EXPECTED_CATALOG_COUNT, "Loaded catalog size mismatch!"
    
    # Test sample inference on Byg Brewski Brewing Company (ID 1)
    test_recs = recommender.recommend_similar_restaurants(restaurant_id=1, top_k=3)
    val_time = time.time() - val_start
    assert len(test_recs) == 3, f"Expected 3 recommendations, got {len(test_recs)}"
    print(f"  [OK] Artifacts successfully validated and test query executed in {val_time*1000:.2f} ms.")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PHASE 5 BUILD COMPLETE (Total Time: {total_time:.2f} seconds)")
    print("=" * 80)


if __name__ == "__main__":
    main()
