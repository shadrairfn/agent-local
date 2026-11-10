"""
Aggressive Preprocessing for Classical Machine Learning (TF-IDF + LightGBM / Ridge).
Strips all punctuation, numbers, and stopwords to maximize sparsity signal-to-noise ratio.
"""

import re
import string
import unicodedata
import pandas as pd

BUILTIN_STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "this", "that",
    # Indonesian
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia",
    "dua", "ia", "seperti", "jika", "sehingga", "kembali", "dan", "ini", "karena",
    "oleh", "saat", "olehnya", "adalah", "itu", "atau", "dengan", "dari", "di", "dalam"
}


def clean_aggressive_text(text: str) -> str:
    """Lowercases, removes punctuation, digits, URLs, and common stopwords."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    # Split, filter stopwords, and re-join
    tokens = [w for w in text.split() if len(w) > 2 and w not in BUILTIN_STOPWORDS]
    return " ".join(tokens)


def apply_aggressive_preprocessing(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Applies clean_aggressive_text to specified text columns."""
    df_out = df.copy()
    for col in text_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).apply(clean_aggressive_text)
    return df_out

