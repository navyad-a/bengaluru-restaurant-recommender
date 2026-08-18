# -*- coding: utf-8 -*-
"""
TF-IDF Vectorization Engine
===========================
Builds, serializes, and manages the sparse TF-IDF feature matrix for the
12,481-restaurant Bengaluru catalog.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from ml.content_based.content_features import build_restaurant_feature_document


class TfidfEngine:
    """
    Manages the TF-IDF vectorizer and sparse feature matrix for restaurant metadata.
    """

    def __init__(
        self,
        min_df: int = 2,
        sublinear_tf: bool = True,
        ngram_range: Tuple[int, int] = (1, 1)
    ):
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.ngram_range = ngram_range
        
        # Vectorizer configuration:
        # - token_pattern r'(?u)\b\w+\b' matches underscored tokens (e.g. cuisine_north_indian)
        # - norm='l2' guarantees ||v||_2 = 1.0, enabling direct dot product for cosine similarity
        # - sublinear_tf=True applies 1 + log(tf) to prevent long dish strings from overpowering cuisines
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            ngram_range=self.ngram_range,
            norm="l2",
            lowercase=False  # Tokens are already normalized and clean
        )
        self.tfidf_matrix: Optional[csr_matrix] = None
        self.restaurant_id_to_idx: Dict[int, int] = {}
        self.idx_to_restaurant_id: Dict[int, int] = {}
        self.restaurant_catalog: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def build_feature_documents(self, df: pd.DataFrame) -> List[str]:
        """
        Transforms all restaurant records in the dataframe into normalized token documents.
        """
        records = df.to_dict("records")
        return [build_restaurant_feature_document(r) for r in records]

    def fit(self, df_restaurants: pd.DataFrame) -> "TfidfEngine":
        """
        Fits the TF-IDF vectorizer and transforms the entire restaurant catalog into a sparse matrix.
        """
        self.restaurant_catalog = df_restaurants.copy()
        
        # Build index mappings
        self.restaurant_id_to_idx = {
            int(r_id): idx for idx, r_id in enumerate(df_restaurants["restaurant_id"])
        }
        self.idx_to_restaurant_id = {
            idx: int(r_id) for idx, r_id in enumerate(df_restaurants["restaurant_id"])
        }

        # Build feature documents and fit vectorizer
        documents = self.build_feature_documents(df_restaurants)
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.is_fitted = True
        return self

    def transform_query(self, query_document: str) -> csr_matrix:
        """
        Transforms a single query document (e.g. user preference profile) into a sparse TF-IDF vector.
        """
        if not self.is_fitted:
            raise RuntimeError("TF-IDF engine must be fitted before transforming queries.")
        return self.vectorizer.transform([query_document])

    def get_restaurant_vector(self, restaurant_id: int) -> csr_matrix:
        """
        Retrieves the sparse TF-IDF row vector for a specific restaurant by ID.
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF engine is not fitted.")
        if restaurant_id not in self.restaurant_id_to_idx:
            raise KeyError(f"Restaurant ID {restaurant_id} not found in catalog mapping.")
        idx = self.restaurant_id_to_idx[restaurant_id]
        return self.tfidf_matrix.getrow(idx)

    def save_artifacts(self, artifact_dir: str) -> Dict[str, str]:
        """
        Serializes the fitted vectorizer, sparse matrix, and mappings to disk.
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Cannot save unfitted TF-IDF model artifacts.")
            
        os.makedirs(artifact_dir, exist_ok=True)
        
        vec_path = os.path.join(artifact_dir, "tfidf_vectorizer.joblib")
        matrix_path = os.path.join(artifact_dir, "tfidf_matrix.joblib")
        catalog_path = os.path.join(artifact_dir, "restaurant_catalog.joblib")
        mappings_path = os.path.join(artifact_dir, "restaurant_mappings.joblib")
        metadata_path = os.path.join(artifact_dir, "feature_metadata.json")
        
        joblib.dump(self.vectorizer, vec_path)
        joblib.dump(self.tfidf_matrix, matrix_path)
        joblib.dump(self.restaurant_catalog, catalog_path)
        joblib.dump({
            "id_to_idx": self.restaurant_id_to_idx,
            "idx_to_id": self.idx_to_restaurant_id
        }, mappings_path)
        
        metadata = {
            "model_type": "TF-IDF Metadata Content Model",
            "num_restaurants": len(self.restaurant_id_to_idx),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "matrix_shape": list(self.tfidf_matrix.shape),
            "matrix_format": "scipy.sparse.csr_matrix",
            "min_df": self.min_df,
            "sublinear_tf": self.sublinear_tf,
            "ngram_range": list(self.ngram_range)
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "vectorizer": vec_path,
            "matrix": matrix_path,
            "catalog": catalog_path,
            "mappings": mappings_path,
            "metadata": metadata_path
        }

    def load_artifacts(self, artifact_dir: str) -> "TfidfEngine":
        """
        Loads pre-computed TF-IDF artifacts from disk for production inference.
        """
        vec_path = os.path.join(artifact_dir, "tfidf_vectorizer.joblib")
        matrix_path = os.path.join(artifact_dir, "tfidf_matrix.joblib")
        catalog_path = os.path.join(artifact_dir, "restaurant_catalog.joblib")
        mappings_path = os.path.join(artifact_dir, "restaurant_mappings.joblib")
        
        if not (os.path.exists(vec_path) and os.path.exists(matrix_path)):
            raise FileNotFoundError(f"Content model artifacts not found in: {artifact_dir}")
            
        self.vectorizer = joblib.load(vec_path)
        self.tfidf_matrix = joblib.load(matrix_path)
        self.restaurant_catalog = joblib.load(catalog_path)
        mappings = joblib.load(mappings_path)
        self.restaurant_id_to_idx = mappings["id_to_idx"]
        self.idx_to_restaurant_id = mappings["idx_to_id"]
        self.is_fitted = True
        return self
