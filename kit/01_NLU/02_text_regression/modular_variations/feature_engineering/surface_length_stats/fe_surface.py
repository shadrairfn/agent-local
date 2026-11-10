"""
Feature Engineering: Surface & Length Statistics.
Extracts 10+ surface descriptors (character count, word count, sentence count,
average lengths, lexical diversity TTR, uppercase ratio, digit density).
All features generated with 'feat_' prefix.
"""

import re
import numpy as np
import pandas as pd


def extract_surface_length_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Extracts surface and lexical statistics from a single text column."""
    df_out = df.copy()
    texts = df_out[text_col].astype(str)

    char_counts = texts.str.len().to_numpy()
    words_list = [t.split() for t in texts]
    word_counts = np.array([max(1, len(w)) for w in words_list])

    # Sentence count
    sentence_counts = np.array([max(1, len(re.split(r"[.!?]+", t))) for t in texts])

    # Lexical Diversity: Type-Token Ratio (TTR)
    unique_word_counts = np.array([len(set(w.lower() for w in words)) for words in words_list])
    ttr = unique_word_counts / word_counts
    root_ttr = unique_word_counts / np.sqrt(word_counts)

    # Densities
    uppercase_counts = np.array([sum(1 for c in t if c.isupper()) for t in texts])
    digit_counts = np.array([sum(1 for c in t if c.isdigit()) for t in texts])

    df_out["feat_char_count"] = char_counts
    df_out["feat_word_count"] = word_counts
    df_out["feat_sentence_count"] = sentence_counts
    df_out["feat_avg_word_length"] = np.round(char_counts / word_counts, 3)
    df_out["feat_avg_sentence_length"] = np.round(word_counts / sentence_counts, 3)
    df_out["feat_type_token_ratio"] = np.round(ttr, 4)
    df_out["feat_root_ttr"] = np.round(root_ttr, 4)
    df_out["feat_uppercase_ratio"] = np.round(uppercase_counts / np.maximum(char_counts, 1), 4)
    df_out["feat_digit_density"] = np.round(digit_counts / np.maximum(char_counts, 1), 4)

    return df_out

