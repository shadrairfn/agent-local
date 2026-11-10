"""
Dual Word & Character N-Gram TF-IDF Feature Extractor.
Generates robust lexical representation for GBDT classifiers.
"""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

class DualTfidfExtractor:
    def __init__(self, max_word_features: int = 5000, max_char_features: int = 5000):
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_word_features,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        self.char_vec = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer='char_wb',
            max_features=max_char_features,
            sublinear_tf=True
        )

    def fit_transform(self, train_texts: pd.Series) -> csr_matrix:
        X_word = self.word_vec.fit_transform(train_texts.astype(str))
        X_char = self.char_vec.fit_transform(train_texts.astype(str))
        return hstack([X_word, X_char]).tocsr()

    def transform(self, test_texts: pd.Series) -> csr_matrix:
        X_word = self.word_vec.transform(test_texts.astype(str))
        X_char = self.char_vec.transform(test_texts.astype(str))
        return hstack([X_word, X_char]).tocsr()
