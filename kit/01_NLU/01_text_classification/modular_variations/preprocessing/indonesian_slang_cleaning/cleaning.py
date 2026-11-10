"""
Indonesian Slang Normalization & Text Cleaning Module.
Designed for competition text classification (sentiment, hate speech, topic).
"""
import re
from typing import Dict, List, Optional
import pandas as pd

SLANG_DICTIONARY: Dict[str, str] = {
    "yg": "yang", "dgn": "dengan", "sdh": "sudah", "udah": "sudah",
    "bgt": "banget", "bgtt": "banget", "bgtz": "banget", "bener": "benar",
    "tdk": "tidak", "gak": "tidak", "ga": "tidak", "g": "tidak", "ngga": "tidak", "nggak": "tidak",
    "tp": "tapi", "bgt": "banget", "kpd": "kepada", "utk": "untuk", "krn": "karena",
    "krna": "karena", "dpt": "dapat", "dr": "dari", "dri": "dari", "sy": "saya",
    "gw": "saya", "gue": "saya", "lo": "kamu", "lu": "kamu", "km": "kamu",
    "klo": "kalau", "kl": "kalau", "kalo": "kalau", "bisaa": "bisa",
    "bgs": "bagus", "bgus": "bagus", "jelek": "buruk", "mantap": "bagus",
    "mantul": "bagus", "oke": "baik", "ok": "baik", "baper": "bawa perasaan"
}

def clean_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """Reduces characters repeated more than max_repeat times (e.g. 'baguuuus' -> 'baguus')."""
    pattern = re.compile(r'(.){' + str(max_repeat) + r',}')
    return pattern.sub(r'' * max_repeat, text)

def normalize_indonesian_slang(text: str, custom_dict: Optional[Dict[str, str]] = None) -> str:
    """Normalizes informal Indonesian tokens to standard formal or standard informal equivalents."""
    if not isinstance(text, str):
        return ""
    
    dictionary = SLANG_DICTIONARY.copy()
    if custom_dict:
        dictionary.update(custom_dict)
        
    text = clean_repeated_chars(text)
    tokens = text.split()
    normalized = [dictionary.get(token.lower(), token) for token in tokens]
    return " ".join(normalized)

def apply_slang_cleaning(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    """Applies slang cleaning across specified columns in a dataframe non-destructively."""
    df_out = df.copy()
    for col in text_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).apply(normalize_indonesian_slang)
    return df_out

if __name__ == "__main__":
    sample = "Pelayanan di toko ini bgt jelek bgt, gak bgt recommended bgttt!"
    print("Original  :", sample)
    print("Normalized:", normalize_indonesian_slang(sample))
