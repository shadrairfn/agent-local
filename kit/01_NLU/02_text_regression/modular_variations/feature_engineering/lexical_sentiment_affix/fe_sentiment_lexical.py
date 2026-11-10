"""
Feature Engineering: Lexical Density, Sentiment Ratios & Academic Affixes.
Calculates stopword ratios, lexical density, sentiment polarity ratios,
and academic morphological affixes (-tion, -ment, -ity, -ism, -able).
All features generated with 'feat_' prefix.
"""

import re
import numpy as np
import pandas as pd

BUILTIN_POSITIVE = {
    "good", "great", "excellent", "amazing", "positive", "effective", "beneficial",
    "valuable", "superior", "helpful", "strong", "clear", "impressive", "promising",
    "bagus", "baik", "positif", "hebat", "mantap", "bermanfaat", "unggul", "jelas"
}

BUILTIN_NEGATIVE = {
    "bad", "poor", "terrible", "negative", "ineffective", "harmful", "flawed",
    "inferior", "weak", "confusing", "difficult", "fail", "failure", "worst",
    "buruk", "jelek", "negatif", "parah", "gagal", "lemah", "sulit", "kurang"
}

BUILTIN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "yang", "untuk", "pada", "ke", "dan", "ini", "itu", "dari", "di"
}


def extract_sentiment_lexical_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Extracts lexical density, stopword ratios, sentiment ratios, and academic affix densities."""
    df_out = df.copy()
    texts = df_out[text_col].astype(str)

    academic_suffix_pattern = re.compile(r"(?:tion|ment|ity|ism|able|ible|ence|ance)\b", re.IGNORECASE)

    pos_ratios = []
    neg_ratios = []
    polarity_scores = []
    stopword_ratios = []
    lexical_densities = []
    academic_affix_ratios = []

    for text in texts:
        words = [w.lower().strip() for w in text.split() if w.strip()]
        total_words = max(1, len(words))

        pos_count = sum(1 for w in words if w in BUILTIN_POSITIVE)
        neg_count = sum(1 for w in words if w in BUILTIN_NEGATIVE)
        stop_count = sum(1 for w in words if w in BUILTIN_STOPWORDS)
        content_words = total_words - stop_count

        academic_count = len(academic_suffix_pattern.findall(text))

        pos_ratio = pos_count / total_words
        neg_ratio = neg_count / total_words
        polarity = (pos_count - neg_count) / max(1, pos_count + neg_count)
        stop_ratio = stop_count / total_words
        lex_density = content_words / total_words
        academic_ratio = academic_count / total_words

        pos_ratios.append(np.round(pos_ratio, 4))
        neg_ratios.append(np.round(neg_ratio, 4))
        polarity_scores.append(np.round(polarity, 4))
        stopword_ratios.append(np.round(stop_ratio, 4))
        lexical_densities.append(np.round(lex_density, 4))
        academic_affix_ratios.append(np.round(academic_ratio, 4))

    df_out["feat_positive_word_ratio"] = pos_ratios
    df_out["feat_negative_word_ratio"] = neg_ratios
    df_out["feat_sentiment_polarity"] = polarity_scores
    df_out["feat_stopword_ratio"] = stopword_ratios
    df_out["feat_lexical_density"] = lexical_densities
    df_out["feat_academic_affix_ratio"] = academic_affix_ratios

    return df_out

