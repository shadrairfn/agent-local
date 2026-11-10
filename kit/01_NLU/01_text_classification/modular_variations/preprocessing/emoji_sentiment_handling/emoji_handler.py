"""
Emoji Sentiment Tagging Module.
Converts common emoticons and unicode emojis into explicit semantic tokens.
"""
import re
from typing import Dict
import pandas as pd

EMOJI_SENTIMENT_MAP: Dict[str, str] = {
    ":)": " [EMOJI_POS] ", ":-)": " [EMOJI_POS] ", ":D": " [EMOJI_POS] ",
    ":(": " [EMOJI_NEG] ", ":-(": " [EMOJI_NEG] ", ":'(": " [EMOJI_NEG] ",
    "👍": " [EMOJI_POS] ", "❤️": " [EMOJI_POS] ", "🔥": " [EMOJI_POS] ", "⭐": " [EMOJI_POS] ",
    "👎": " [EMOJI_NEG] ", "💔": " [EMOJI_NEG] ", "😡": " [EMOJI_NEG] ", "🤮": " [EMOJI_NEG] "
}

def tag_emoji_sentiment(text: str) -> str:
    """Replaces known emoticon and emoji characters with standardized sentiment tags."""
    if not isinstance(text, str):
        return ""
    for emoji_char, tag in EMOJI_SENTIMENT_MAP.items():
        text = text.replace(emoji_char, tag)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_emoji_tagging(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    df_out = df.copy()
    for col in text_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).apply(tag_emoji_sentiment)
    return df_out
