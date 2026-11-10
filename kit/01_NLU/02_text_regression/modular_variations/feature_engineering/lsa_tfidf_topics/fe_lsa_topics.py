"""
Feature Engineering: Latent Semantic Analysis (TF-IDF + TruncatedSVD Topics).
Projects high-dimensional text n-grams into compact dense topic vectors.
Fitted strictly on train data to prevent validation leakage.
All features generated with 'feat_' prefix.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class LSATopicExtractor:
    """Extracts dense LSA semantic topic features leak-free."""

    def __init__(self, n_components: int = 10, max_features: int = 5000, random_state: int = 42):
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self.is_fitted = False

    def fit_transform(self, train_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Fits TF-IDF and SVD on train texts, returns train DataFrame with feat_lsa_* columns."""
        texts = train_df[text_col].astype(str).tolist()
        tfidf_mat = self.vectorizer.fit_transform(texts)
        # Adjust n_components if samples < n_components
        actual_comp = min(self.n_components, tfidf_mat.shape[0] - 1, tfidf_mat.shape[1] - 1)
        actual_comp = max(1, actual_comp)
        if actual_comp != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=actual_comp, random_state=self.random_state)

        lsa_mat = self.svd.fit_transform(tfidf_mat)
        self.is_fitted = True

        df_out = train_df.copy()
        for idx in range(lsa_mat.shape[1]):
            df_out[f"feat_lsa_topic_{idx}"] = np.round(lsa_mat[:, idx], 4)
        return df_out

    def transform(self, test_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Transforms test texts using the pre-fitted TF-IDF and SVD models."""
        if not self.is_fitted:
            print("[Warning] LSATopicExtractor is not fitted! Call fit_transform first.")
            return test_df.copy()

        texts = test_df[text_col].astype(str).tolist()
        tfidf_mat = self.vectorizer.transform(texts)
        lsa_mat = self.svd.transform(tfidf_mat)

        df_out = test_df.copy()
        for idx in range(lsa_mat.shape[1]):
            df_out[f"feat_lsa_topic_{idx}"] = np.round(lsa_mat[:, idx], 4)
        return df_out

