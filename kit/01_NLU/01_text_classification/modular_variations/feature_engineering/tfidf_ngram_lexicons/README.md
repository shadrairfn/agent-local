# 🔤 Dual Word & Char N-Gram TF-IDF Extractor

## Rationale
- **Word (1, 2)-grams:** Captures key unigrams and negative collocations (e.g., "tidak bagus").
- **Char (3, 5)-grams (char_wb):** Captures Indonesian morphological affixes (*meng-kan*, *di-i*) and handles common typos without requiring a heavy lemmatizer.
