"""
Feature Engineering: Readability & Complexity Formulas.
Extracts classic linguistic readability scores:
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- Automated Readability Index (ARI)
- Coleman-Liau Index
All features generated with 'feat_' prefix.
"""

import re
import numpy as np
import pandas as pd


def count_syllables(word: str) -> int:
    """Estimates the number of syllables in an English/Latin word phonetically."""
    w = word.lower().strip()
    if len(w) <= 3:
        return 1
    # Strip non-pronounced endings
    w = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", w)
    w = re.sub(r"^y", "", w)
    syllables = re.findall(r"[aeiouy]{1,2}", w)
    return max(1, len(syllables))


def extract_readability_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Calculates classic mathematical readability formulas."""
    df_out = df.copy()
    texts = df_out[text_col].astype(str)

    flesch_reading = []
    flesch_kincaid = []
    gunning_fog = []
    ari_index = []
    coleman_liau = []
    complex_word_ratio = []

    for text in texts:
        words = text.split()
        num_words = max(1, len(words))
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        num_sentences = max(1, len(sentences))
        num_chars = sum(len(w) for w in words)

        syllables = [count_syllables(w) for w in words]
        total_syllables = sum(syllables)
        num_complex_words = sum(1 for syl in syllables if syl >= 3)

        asl = num_words / num_sentences             # Average Sentence Length
        asw = total_syllables / num_words           # Average Syllables per Word
        pcw = (num_complex_words / num_words) * 100 # Percent Complex Words

        # 1. Flesch Reading Ease
        fre = 206.835 - (1.015 * asl) - (84.6 * asw)
        # 2. Flesch-Kincaid Grade Level
        fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
        # 3. Gunning Fog Index
        fog = 0.4 * (asl + pcw)
        # 4. Automated Readability Index (ARI)
        ari = 4.71 * (num_chars / num_words) + 0.5 * asl - 21.43
        # 5. Coleman-Liau Index
        l = (num_chars / num_words) * 100
        s = (num_sentences / num_words) * 100
        cli = 0.0588 * l - 0.296 * s - 15.8

        flesch_reading.append(np.round(fre, 3))
        flesch_kincaid.append(np.round(fkgl, 3))
        gunning_fog.append(np.round(fog, 3))
        ari_index.append(np.round(ari, 3))
        coleman_liau.append(np.round(cli, 3))
        complex_word_ratio.append(np.round(num_complex_words / num_words, 4))

    df_out["feat_flesch_reading_ease"] = flesch_reading
    df_out["feat_flesch_kincaid_grade"] = flesch_kincaid
    df_out["feat_gunning_fog_index"] = gunning_fog
    df_out["feat_automated_readability_index"] = ari_index
    df_out["feat_coleman_liau_index"] = coleman_liau
    df_out["feat_complex_word_ratio"] = complex_word_ratio

    return df_out

