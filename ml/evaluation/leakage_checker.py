# -*- coding: utf-8 -*-
"""
Leakage Checker & Benchmark Integrity Verification Module
==========================================================
Enforces strict, leakage-free isolation between training and held-out test splits.
Verifies zero train/test interaction overlap, rating bounds, and catalog ID alignment.
"""

import pandas as pd
from typing import Dict, Any, Tuple, Set, Optional


class DataLeakageError(Exception):
    """Raised when data leakage or benchmark integrity violation is detected."""
    pass


class LeakageChecker:
    """
    Validates isolation and integrity between training and test benchmark datasets.
    """

    @staticmethod
    def verify_integrity(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_catalog: pd.DataFrame,
        df_users: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive data leakage and integrity checks.
        
        Raises:
            DataLeakageError: If any leakage or invalid records are detected.
        """
        # 1. Zero interaction collision check
        train_pairs = set(zip(df_train["user_id"], df_train["restaurant_id"]))
        test_pairs = set(zip(df_test["user_id"], df_test["restaurant_id"]))
        overlap = train_pairs.intersection(test_pairs)

        if len(overlap) > 0:
            raise DataLeakageError(
                f"Data Leakage Detected: {len(overlap)} user-restaurant interaction(s) "
                f"present in both train and test splits! Example overlap: {list(overlap)[:3]}"
            )

        # 2. Rating bounds validation [1.0, 5.0]
        train_invalid_ratings = df_train[~df_train["rating"].between(1.0, 5.0)]
        test_invalid_ratings = df_test[~df_test["rating"].between(1.0, 5.0)]

        if not train_invalid_ratings.empty or not test_invalid_ratings.empty:
            raise DataLeakageError(
                f"Invalid rating values found: {len(train_invalid_ratings)} in train, "
                f"{len(test_invalid_ratings)} in test. Ratings must be in [1.0, 5.0]."
            )

        # 3. Catalog ID alignment
        catalog_ids = set(df_catalog["restaurant_id"])
        train_unknown_rests = set(df_train["restaurant_id"]) - catalog_ids
        test_unknown_rests = set(df_test["restaurant_id"]) - catalog_ids

        if train_unknown_rests or test_unknown_rests:
            raise DataLeakageError(
                f"Unknown restaurant IDs detected: {len(train_unknown_rests)} in train, "
                f"{len(test_unknown_rests)} in test not present in authentic 12,481 catalog."
            )

        # 4. User ID alignment
        if df_users is not None:
            user_ids = set(df_users["user_id"])
            train_unknown_users = set(df_train["user_id"]) - user_ids
            test_unknown_users = set(df_test["user_id"]) - user_ids
            if train_unknown_users or test_unknown_users:
                raise DataLeakageError(
                    f"Unknown user IDs detected: {len(train_unknown_users)} in train, "
                    f"{len(test_unknown_users)} in test."
                )

        return {
            "status": "passed",
            "train_interactions": len(df_train),
            "test_interactions": len(df_test),
            "total_interactions": len(df_train) + len(df_test),
            "overlap_count": 0,
            "train_users": int(df_train["user_id"].nunique()),
            "test_users": int(df_test["user_id"].nunique()),
            "catalog_size": len(df_catalog),
            "rating_range": [float(df_train["rating"].min()), float(df_train["rating"].max())]
        }
