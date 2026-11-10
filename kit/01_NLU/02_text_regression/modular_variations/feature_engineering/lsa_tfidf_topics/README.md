# 🌀 Feature Engineering: Latent Semantic Analysis (TF-IDF + TruncatedSVD)

## Deskripsi & Rationale
Matriks TF-IDF n-gram (1-2 gram) memiliki ribuan dimensi sparse yang tidak dapat diproses secara langsung oleh model GBDT tanpa konsumsi memori masif. Latent Semantic Analysis (LSA) menggunakan dekomposisi nilai singular (*TruncatedSVD*) untuk mereduksi ruang n-gram menjadi 10–30 dimensi topik padat (*dense topic vectors*).

## Perlindungan Anti-Leakage
Sesuai **Rule 8**, objek `TfidfVectorizer` dan `TruncatedSVD` **hanya di-fit pada `train_df`**, lalu digunakan untuk mentransformasikan `test_df`.

## Cara Menggunakan
```python
from modular_variations.feature_engineering.lsa_tfidf_topics.fe_lsa_topics import LSATopicExtractor

extractor = LSATopicExtractor(n_components=10)
train_features_df = extractor.fit_transform(train_df, text_col='full_text')
test_features_df  = extractor.transform(test_df, text_col='full_text')
```

