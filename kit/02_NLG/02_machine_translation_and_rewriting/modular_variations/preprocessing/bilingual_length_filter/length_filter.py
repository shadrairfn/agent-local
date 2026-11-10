"""
Bilingual Sentence Pair Length Filter.
Prunes noisy parallel training pairs with extreme word fertility ratios.
"""
import pandas as pd

def filter_bilingual_pairs(df: pd.DataFrame, src_col: str, tgt_col: str, min_ratio: float = 0.3, max_ratio: float = 3.0) -> pd.DataFrame:
    df_clean = df.copy()
    src_len = df_clean[src_col].astype(str).apply(lambda x: len(x.split()))
    tgt_len = df_clean[tgt_col].astype(str).apply(lambda x: len(x.split()))
    ratio = tgt_len / (src_len + 1e-5)
    valid_mask = (ratio >= min_ratio) & (ratio <= max_ratio) & (src_len > 0) & (tgt_len > 0)
    return df_clean[valid_mask].reset_index(drop=True)
