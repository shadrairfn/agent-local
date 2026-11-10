# 📏 Feature Engineering: Surface & Length Statistics

## Deskripsi & Rationale
Panjang dokumen, variasi panjang kalimat, dan keragaman kosakata (*Type-Token Ratio*) adalah prediktor nomor satu dalam hampir seluruh kasus penilaian esai dan keterbacaan teks (CommonLit, AES 2.0). Penulis yang mahir cenderung memiliki kalimat yang lebih panjang dan kosakata yang tidak repetitif.

## Daftar Fitur yang Dihasilkan (Prefix `feat_`):
- `feat_char_count`: Jumlah karakter total
- `feat_word_count`: Jumlah kata total
- `feat_sentence_count`: Estimasi jumlah kalimat
- `feat_avg_word_length`: Rata-rata panjang karakter per kata
- `feat_avg_sentence_length`: Rata-rata kata per kalimat
- `feat_type_token_ratio`: Rasio kata unik terhadap total kata (TTR)
- `feat_root_ttr`: Root TTR ($V / \sqrt{N}$, kebal terhadap bias panjang teks)
- `feat_uppercase_ratio`: Rasio huruf kapital
- `feat_digit_density`: Densitas angka

## Cara Menggunakan
```python
from modular_variations.feature_engineering.surface_length_stats.fe_surface import extract_surface_length_features

train_features_df = extract_surface_length_features(train_df, text_col='full_text')
test_features_df  = extract_surface_length_features(test_df, text_col='full_text')
```

