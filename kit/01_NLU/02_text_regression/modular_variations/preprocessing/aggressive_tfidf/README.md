# 🪓 Aggressive Preprocessing (TF-IDF & GBDT Priority)

## Filosofi & Rationale
Model Machine Learning tabular (seperti LightGBM, CatBoost, atau Ridge Regression) yang dilatih di atas representasi n-gram TF-IDF sangat rentan terhadap *curse of dimensionality*. Tanda baca, angka acak, dan kata penghubung (*stopwords*) yang berulang dapat mendominasi bobot matriks frekuensi tanpa memberikan kontribusi semantik yang berarti.

Modul ini **menghapus seluruh tanda baca, angka, dan stopwords** untuk menghasilkan matriks TF-IDF yang ringkas, padat sinyal, dan bebas noise.

## Kapan Menggunakan
- Gunakan saat melatih **Pipeline 01: Machine Learning (TF-IDF + LightGBM / Ridge)**.
- **Peringatan Keras:** JANGAN gunakan modul ini untuk melatih model Transformer, karena menghapus tanda baca akan merusak attention map.

## Cara Menggunakan
```python
from modular_variations.preprocessing.aggressive_tfidf.prepro_aggressive import apply_aggressive_preprocessing

train_df = apply_aggressive_preprocessing(train_df, text_cols=['full_text'])
test_df  = apply_aggressive_preprocessing(test_df, text_cols=['full_text'])
```

