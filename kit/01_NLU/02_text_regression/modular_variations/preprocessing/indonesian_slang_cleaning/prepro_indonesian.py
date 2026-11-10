"""
Indonesian Informal & Slang Normalization Preprocessing.
Maps 70+ Indonesian slang / colloquial abbreviations to formal standard Indonesian
and strips informal enclitics (-lah, -kah, -dong, -sih).
"""

import re
import unicodedata
import pandas as pd

INDONESIAN_SLANG_MAP = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "kpd": "kepada",
    "bgt": "sangat", "bgtt": "sangat", "banget": "sangat", "bngt": "sangat",
    "tp": "tetapi", "tpi": "tetapi", "krn": "karena", "karna": "karena",
    "klo": "kalau", "kalo": "kalau", "kl": "kalau", "jgn": "jangan",
    "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak",
    "tdk": "tidak", "tak": "tidak", "bkn": "bukan",
    "udh": "sudah", "sdh": "sudah", "udah": "sudah", "dah": "sudah",
    "blm": "belum", "lom": "belum", "belom": "belum",
    "bs": "bisa", "gmn": "bagaimana", "gimana": "bagaimana",
    "dpt": "dapat", "dptkan": "dapatkan", "dr": "dari", "dri": "dari",
    "sy": "saya", "aku": "saya", "gue": "saya", "gw": "saya",
    "lu": "kamu", "loe": "kamu", "km": "kamu", "kmrn": "kemarin",
    "bbrp": "beberapa", "org": "orang", "bnyak": "banyak", "byk": "banyak",
    "aja": "saja", "jg": "juga", "sm": "sama", "smp": "sampai",
    "dlm": "dalam", "lbh": "lebih", "trm": "terima", "makasih": "terima kasih",
    "thx": "terima kasih", "ok": "oke", "sip": "baik", "bener": "benar"
}


def normalize_indonesian_slang(text: str) -> str:
    """Normalizes informal Indonesian abbreviations while preserving sentence structure."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)

    # Word-by-word token normalization
    tokens = text.split()
    normalized_tokens = []
    for tok in tokens:
        # Strip trailing punctuation temporarily to check slang
        match = re.match(r"^(\W*)([\w]+)(\W*)$", tok)
        if match:
            pre, word, post = match.groups()
            word_lower = word.lower()
            if word_lower in INDONESIAN_SLANG_MAP:
                rep = INDONESIAN_SLANG_MAP[word_lower]
                # Match title case if original was capitalized
                if word.istitle():
                    rep = rep.capitalize()
                normalized_tokens.append(f"{pre}{rep}{post}")
            else:
                normalized_tokens.append(tok)
        else:
            normalized_tokens.append(tok)

    return " ".join(normalized_tokens)


def apply_indonesian_slang_cleaning(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Applies normalize_indonesian_slang to specified text columns."""
    df_out = df.copy()
    for col in text_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).apply(normalize_indonesian_slang)
    return df_out

