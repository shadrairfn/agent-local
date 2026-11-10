"""
Lead-3 Sentence Extractor Module.
Extracts the first 3 sentences of a document as a high-speed baseline or prefix truncation filter.
"""
import re
from typing import List
import pandas as pd

def extract_lead_n_sentences(text: str, n: int = 3) -> str:
    """Splits text into sentences using standard punctuation heuristics and returns the first n."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Split on sentence boundaries: periods, exclamation marks, or question marks
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = sentences[:n]
    return " ".join(selected)

def apply_lead3(df: pd.DataFrame, source_col: str, output_col: str = "lead3_summary") -> pd.DataFrame:
    df_out = df.copy()
    df_out[output_col] = df_out[source_col].astype(str).apply(lambda x: extract_lead_n_sentences(x, n=3))
    return df_out
