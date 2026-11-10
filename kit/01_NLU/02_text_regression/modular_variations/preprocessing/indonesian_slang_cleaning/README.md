# 🇮🇩 Indonesian Informal & Slang Normalization

## Filosofi & Rationale
Dataset bahasa Indonesia di GEMASTIK atau kompetisi nasional sering kali berasal dari media sosial (Twitter/X, ulasan e-commerce, percakapan CS). Teks tersebut sarat dengan variasi singkatan gaul informal seperti `yg` (yang), `dgn` (dengan), `bgt` (sangat), `klo` (kalau), dan `ga` (tidak).

Modul ini **memetakan singkatan informal menjadi kata baku bahasa Indonesia** tanpa menghilangkan tanda baca aslinya.

## Kapan Menggunakan
- Sangat direkomendasikan untuk **seluruh kompetisi NLP berbahasa Indonesia (GEMASTIK)**, baik sebelum memasukkan teks ke TF-IDF maupun ke IndoBERT.

## Cara Menggunakan
```python
from modular_variations.preprocessing.indonesian_slang_cleaning.prepro_indonesian import apply_indonesian_slang_cleaning

train_df = apply_indonesian_slang_cleaning(train_df, text_cols=['full_text'])
test_df  = apply_indonesian_slang_cleaning(test_df, text_cols=['full_text'])
```

