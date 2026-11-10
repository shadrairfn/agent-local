"""
Feature Engineering: Pairwise Text Similarity & Interactions.
Essential for sentence-pair regression (US Patent Phrase Matching, Prompt-Conditioned Summaries).
Extracts:
- Jaccard Word Similarity
- Word Overlap Count & Ratios
- Length Difference & Length Ratio
All features generated with 'feat_' prefix.
"""

import numpy as np
import pandas as pd


def extract_pairwise_similarity_features(df: pd.DataFrame, text_col_a: str, text_col_b: str) -> pd.DataFrame:
    """Calculates overlap, similarity, and delta features between two text columns."""
    df_out = df.copy()
    texts_a = df_out[text_col_a].astype(str)
    texts_b = df_out[text_col_b].astype(str)

    jaccard_scores = []
    overlap_counts = []
    overlap_ratios = []
    length_diffs = []
    length_ratios = []

    for ta, tb in zip(texts_a, texts_b):
        words_a = set(ta.lower().split())
        words_b = set(tb.lower().split())

        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)

        jaccard = len(intersection) / max(1, len(union))
        overlap_cnt = len(intersection)
        overlap_rat = overlap_cnt / max(1, min(len(words_a), len(words_b)))

        len_a = len(ta)
        len_b = len(tb)
        diff = abs(len_a - len_b)
        ratio = len_a / max(1, len_b)

        jaccard_scores.append(np.round(jaccard, 4))
        overlap_counts.append(overlap_cnt)
        overlap_ratios.append(np.round(overlap_rat, 4))
        length_diffs.append(diff)
        length_ratios.append(np.round(ratio, 4))

    df_out["feat_pair_jaccard_similarity"] = jaccard_scores
    df_out["feat_pair_overlap_count"] = overlap_counts
    df_out["feat_pair_overlap_ratio"] = overlap_ratios
    df_out["feat_pair_length_diff"] = length_diffs
    df_out["feat_pair_length_ratio"] = length_ratios

    return df_out

