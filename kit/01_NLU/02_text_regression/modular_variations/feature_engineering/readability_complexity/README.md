# 📖 Feature Engineering: Readability & Complexity Formulas

## Deskripsi & Rationale
Formula keterbacaan (*readability scores*) adalah indeks matematis baku yang digunakan para ahli bahasa dan pendidik untuk mengukur tingkat kesulitan membaca teks kutipan, esai, atau dokumen ilmiah. Fitur-fitur ini menjadi dasar penilaian pada Kaggle CommonLit Readability Prize dan Automated Essay Scoring.

## Fitur yang Dihasilkan:
- `feat_flesch_reading_ease`: Skor kemudahan membaca (skala 0–100, makin tinggi makin mudah)
- `feat_flesch_kincaid_grade`: Tingkat kelas pendidikan (grade level US schooling)
- `feat_gunning_fog_index`: Indeks kesulitan kabut (bobot kata kompleks $\ge 3$ suku kata)
- `feat_automated_readability_index` (ARI): Indeks keterbacaan berbasis rasio karakter dan kalimat
- `feat_coleman_liau_index`: Indeks keterbacaan berbasis frekuensi huruf per 100 kata
- `feat_complex_word_ratio`: Proporsi kata kompleks bersuku kata 3 atau lebih

## Cara Menggunakan
```python
from modular_variations.feature_engineering.readability_complexity.fe_readability import extract_readability_features

train_features_df = extract_readability_features(train_df, text_col='full_text')
test_features_df  = extract_readability_features(test_df, text_col='full_text')
```

