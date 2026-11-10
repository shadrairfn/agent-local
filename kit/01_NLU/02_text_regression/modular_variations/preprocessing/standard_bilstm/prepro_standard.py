"""
Standard Preprocessing for Recurrent & Word-Embedding Models (BiLSTM, GRU, Word2Vec).
Applies lowercase, URL removal, English contraction expansion, and spacing cleanup.
"""

import re
import unicodedata
import pandas as pd

CONTRACTIONS_DICT = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "they'd": "they would",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "they'll": "they will", "let's": "let us", "that's": "that is"
}


def clean_standard_text(text: str) -> str:
    """Lowercases, expands contractions, strips URLs and normalizes whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).lower()
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    # Expand contractions
    for cont, exp in CONTRACTIONS_DICT.items():
        text = text.replace(cont, exp)
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_standard_preprocessing(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Applies clean_standard_text to specified text columns."""
    df_out = df.copy()
    for col in text_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).apply(clean_standard_text)
    return df_out

