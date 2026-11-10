"""
LSA (Latent Semantic Analysis) Topic Feature Extractor.
Projects sparse TF-IDF space into dense semantic topics via TruncatedSVD.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

class LSATopicExtractor:
    def __init__(self, n_components: int = 16, random_state: int = 42):
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit_transform(self, train_texts: pd.Series) -> np.ndarray:
        X_sparse = self.tfidf.fit_transform(train_texts.astype(str))
        return self.svd.fit_transform(X_sparse)

    def transform(self, test_texts: pd.Series) -> np.ndarray:
        X_sparse = self.tfidf.transform(test_texts.astype(str))
        return self.svd.transform(X_sparse)
