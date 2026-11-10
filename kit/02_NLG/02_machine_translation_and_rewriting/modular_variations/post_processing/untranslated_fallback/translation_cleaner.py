"""
Untranslated Fallback & Terminal Punctuation Post-Processor.
Fixes empty generations and ensures consistent terminal punctuation.
"""
import pandas as pd

def polish_translations(df: pd.DataFrame, src_col: str, pred_col: str) -> pd.DataFrame:
    df_out = df.copy()
    polished = []
    for src, pred in zip(df_out[src_col], df_out[pred_col]):
        pred_str = str(pred).strip() if pd.notna(pred) else ""
        if not pred_str or len(pred_str) < 3:
            pred_str = str(src).strip()  # Fallback to source
        if pred_str and pred_str[-1] not in {'.', '!', '?'}:
            pred_str += '.'
        polished.append(pred_str)
    df_out[pred_col] = polished
    return df_out
