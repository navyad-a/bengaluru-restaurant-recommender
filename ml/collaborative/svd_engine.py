# -*- coding: utf-8 -*-
r"""
Surprise SVD Matrix Factorization Engine
========================================
Implements SVD matrix factorization for Collaborative Filtering on the
Synthetic Benchmark dataset using Surprise.

Mathematical Formulation:
    \hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u
    
Where:
    - \mu: Global baseline rating mean
    - b_u: User bias parameter
    - b_i: Restaurant item bias parameter
    - p_u \in \mathbb{R}^k: User latent factor vector
    - q_i \in \mathbb{R}^k: Restaurant latent factor vector
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from surprise import Dataset, Reader, SVD
from surprise.model_selection import KFold


class SVDEngine:
    """
    Manages Surprise SVD model training, cross-validation, prediction, and persistence.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 30,
        lr_all: float = 0.005,
        reg_all: float = 0.1,
        random_state: int = 42
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        
        self.reader = Reader(rating_scale=(1.0, 5.0))
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )
        self.is_fitted = False
        self.trainset = None
        self.known_users = set()
        self.known_restaurants = set()
        self.global_mean = 3.5

    def _prepare_surprise_data(self, df_ratings: pd.DataFrame) -> Dataset:
        """
        Converts a Pandas DataFrame (user_id, restaurant_id, rating) into a Surprise Dataset.
        """
        required_cols = ["user_id", "restaurant_id", "rating"]
        df_subset = df_ratings[required_cols].copy()
        return Dataset.load_from_df(df_subset, self.reader)

    @classmethod
    def cross_validate_hyperparameters(
        cls,
        df_train: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_splits: int = 3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Performs controlled K-fold cross-validation strictly on the training partition
        to select the optimal SVD hyperparameter configuration.
        """
        if param_grid is None:
            param_grid = {
                "n_factors": [50, 100, 150],
                "n_epochs": [10, 20, 30],
                "reg_all": [0.02, 0.05, 0.10]
            }

        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(df_train[["user_id", "restaurant_id", "rating"]], reader)
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

        experiments = []
        best_rmse = float("inf")
        best_params = None

        for n_f in param_grid["n_factors"]:
            for n_e in param_grid["n_epochs"]:
                for reg in param_grid["reg_all"]:
                    fold_rmses = []
                    fold_maes = []

                    for trainset, testset in kf.split(data):
                        model = SVD(
                            n_factors=n_f,
                            n_epochs=n_e,
                            lr_all=0.005,
                            reg_all=reg,
                            random_state=random_state
                        )
                        model.fit(trainset)
                        predictions = model.test(testset)
                        
                        # Calculate fold RMSE and MAE
                        errors = [pred.r_ui - pred.est for pred in predictions]
                        rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
                        mae = sum(abs(e) for e in errors) / len(errors)
                        fold_rmses.append(rmse)
                        fold_maes.append(mae)

                    avg_rmse = sum(fold_rmses) / len(fold_rmses)
                    avg_mae = sum(fold_maes) / len(fold_maes)

                    exp_record = {
                        "n_factors": n_f,
                        "n_epochs": n_e,
                        "reg_all": reg,
                        "val_rmse": round(avg_rmse, 4),
                        "val_mae": round(avg_mae, 4)
                    }
                    experiments.append(exp_record)

                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_params = {
                            "n_factors": n_f,
                            "n_epochs": n_e,
                            "reg_all": reg,
                            "lr_all": 0.005,
                            "random_state": random_state
                        }

        return {
            "best_params": best_params,
            "best_val_rmse": round(best_rmse, 4),
            "all_experiments": experiments
        }

    def fit(self, df_train: pd.DataFrame) -> "SVDEngine":
        """
        Fits the SVD model on the full training ratings partition.
        """
        data = self._prepare_surprise_data(df_train)
        self.trainset = data.build_full_trainset()
        
        # Instantiate model with instance hyperparameters
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )
        self.model.fit(self.trainset)
        
        self.known_users = set(df_train["user_id"].unique())
        self.known_restaurants = set(df_train["restaurant_id"].unique())
        self.global_mean = float(self.trainset.global_mean)
        self.is_fitted = True
        return self

    def predict(self, user_id: int, restaurant_id: int) -> float:
        r"""
        Predicts rating \hat{r}_{u,i} for a given user and restaurant pair.
        Returns clipped rating in [1.0, 5.0].
        """
        if not self.is_fitted:
            raise RuntimeError("SVD model must be fitted before predicting ratings.")
            
        pred = self.model.predict(uid=user_id, iid=restaurant_id)
        # Clip to valid rating range
        return max(1.0, min(5.0, float(pred.est)))

    def predict_batch(self, user_id: int, restaurant_ids: List[int]) -> List[Tuple[int, float]]:
        """
        Predicts ratings for a list of candidate restaurants for a given user.
        """
        results = []
        for r_id in restaurant_ids:
            score = self.predict(user_id, r_id)
            results.append((r_id, score))
        return results

    def is_known_user(self, user_id: int) -> bool:
        """Checks whether the user existed in the SVD training interaction matrix."""
        return user_id in self.known_users

    def is_known_restaurant(self, restaurant_id: int) -> bool:
        """Checks whether the restaurant existed in the SVD training interaction matrix."""
        return restaurant_id in self.known_restaurants

    def save_artifacts(self, artifact_dir: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Serializes the trained SVD model, training metadata, and provenance to disk.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted SVD model artifacts.")

        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, "svd_model.joblib")
        known_entities_path = os.path.join(artifact_dir, "known_entities.joblib")
        meta_path = os.path.join(artifact_dir, "model_metadata.json")

        joblib.dump(self.model, model_path)
        joblib.dump({
            "known_users": list(self.known_users),
            "known_restaurants": list(self.known_restaurants),
            "global_mean": self.global_mean,
            "hyperparameters": {
                "n_factors": self.n_factors,
                "n_epochs": self.n_epochs,
                "lr_all": self.lr_all,
                "reg_all": self.reg_all,
                "random_state": self.random_state
            }
        }, known_entities_path)

        full_meta = {
            "model_type": "Surprise SVD Matrix Factorization",
            "benchmark_type": "Synthetic Collaborative Filtering Benchmark",
            "dataset_notice": (
                "The training ratings are explicitly SYNTHETIC. Model evaluation metrics on this "
                "benchmark demonstrate algorithm mechanics and must NOT be cited as real customer behavior."
            ),
            "hyperparameters": {
                "n_factors": self.n_factors,
                "n_epochs": self.n_epochs,
                "lr_all": self.lr_all,
                "reg_all": self.reg_all,
                "random_state": self.random_state
            },
            "num_known_users": len(self.known_users),
            "num_known_restaurants": len(self.known_restaurants),
            "global_mean_rating": round(self.global_mean, 4),
        }
        if metadata:
            full_meta.update(metadata)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(full_meta, f, indent=2)

        return {
            "model": model_path,
            "entities": known_entities_path,
            "metadata": meta_path
        }

    def load_artifacts(self, artifact_dir: str) -> "SVDEngine":
        """
        Loads pre-trained SVD model and entity mappings from disk.
        """
        model_path = os.path.join(artifact_dir, "svd_model.joblib")
        known_entities_path = os.path.join(artifact_dir, "known_entities.joblib")

        if not (os.path.exists(model_path) and os.path.exists(known_entities_path)):
            raise FileNotFoundError(f"SVD model artifacts not found in: {artifact_dir}")

        self.model = joblib.load(model_path)
        entities = joblib.load(known_entities_path)
        
        self.known_users = set(entities["known_users"])
        self.known_restaurants = set(entities["known_restaurants"])
        self.global_mean = float(entities["global_mean"])
        
        hp = entities.get("hyperparameters", {})
        self.n_factors = hp.get("n_factors", 50)
        self.n_epochs = hp.get("n_epochs", 30)
        self.lr_all = hp.get("lr_all", 0.005)
        self.reg_all = hp.get("reg_all", 0.1)
        self.random_state = hp.get("random_state", 42)
        
        self.is_fitted = True
        return self
