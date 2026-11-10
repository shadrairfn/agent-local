"""
Minimal Preprocessing for Pretrained Transformers (DeBERTa, RoBERTa, IndoBERT).
Preserves casing, punctuation, and natural sentence structures essential for self-attention.
"""

import re
import unicodedata
import pandas as pd


def clean_minimal_text(text: str) -> str:
    """Normalizes unicode and collapses irregular whitespace without altering casing or punctuation."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode to NFKD
    text = unicodedata.normalize("NFKD", text)
    # Replace non-breaking spaces and tabs with standard space
    text = re.sub(r"[\r\t\f\v]+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    # Normalize excessive newlines to maximum two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def apply_minimal_preprocessing(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Applies clean_minimal_text to specified text columns on a copied DataFrame."""
    df_out = df.copy()
    for col in text_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).apply(clean_minimal_text)
    return df_out

