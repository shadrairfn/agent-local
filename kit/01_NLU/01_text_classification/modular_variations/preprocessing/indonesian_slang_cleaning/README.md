# 🇮🇩 Indonesian Slang & Abbreviation Cleaning

## Overview
Indonesian texts, especially from social media (Twitter/X, Instagram, Tokopedia, Shopee reviews), contain high proportions of informal abbreviations and character elongation (e.g., *"bgtttt"* $\to$ *"banget"*).

## Rationale
- **Vocabulary Compaction:** Standardizing 10 variants of "tidak" (`gak`, `ga`, `ngga`, `nggak`, `tdk`) into one unified token collapses sparse vocabulary matrices for TF-IDF / GBDT.
- **OOV Reduction:** Reduces out-of-vocabulary tokens for pretrained subword tokenizers (IndoBERT/DeBERTa).

## Usage
```python
from modular_variations.preprocessing.indonesian_slang_cleaning.cleaning import apply_slang_cleaning

train_df = apply_slang_cleaning(train_df, text_cols=['text'])
test_df  = apply_slang_cleaning(test_df, text_cols=['text'])
```
