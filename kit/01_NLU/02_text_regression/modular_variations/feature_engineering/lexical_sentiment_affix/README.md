# 💬 Feature Engineering: Lexical Density & Sentiment Affixes

## Deskripsi & Rationale
Mengukur densitas kata-kata bermakna (*content words*) dibandingkan kata-kata sambung (*stop words*), rasio kata bersentimen positif/negatif, serta tingkat kecanggihan afiks akademik bahasa Inggris/Latin (seperti akhiran `-tion`, `-ment`, `-ity`, `-ism`).

## Fitur yang Dihasilkan:
- `feat_positive_word_ratio`: Proporsi kata bernada positif
- `feat_negative_word_ratio`: Proporsi kata bernada negatif
- `feat_sentiment_polarity`: Polarisasi sentimen $(-1.0 \text{ s.d. } +1.0)$
- `feat_stopword_ratio`: Rasio kata sambung umum
- `feat_lexical_density`: Densitas kata isi non-stopword
- `feat_academic_affix_ratio`: Frekuensi penggunaan morfem akademik formal

## Cara Menggunakan
```python
from modular_variations.feature_engineering.lexical_sentiment_affix.fe_sentiment_lexical import extract_sentiment_lexical_features

train_features_df = extract_sentiment_lexical_features(train_df, text_col='full_text')
test_features_df  = extract_sentiment_lexical_features(test_df, text_col='full_text')
```

