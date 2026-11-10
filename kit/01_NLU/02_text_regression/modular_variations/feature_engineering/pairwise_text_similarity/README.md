# 🔗 Feature Engineering: Pairwise Text Similarity & Interactions

## Deskripsi & Rationale
Wajib digunakan pada dataset teks berpasangan (*sentence pair*), seperti:
1. Kasus 03: CommonLit Evaluate Student Summaries (`prompt_text` vs `student_summary`)
2. Kasus 05: U.S. Patent Phrase to Phrase Matching (`anchor` vs `target`)
3. Kasus 12: Pairwise Preference Margin (`response_a` vs `response_b`)

## Fitur yang Dihasilkan:
- `feat_pair_jaccard_similarity`: Kemiripan himpunan kata Jaccard
- `feat_pair_overlap_count`: Jumlah kata yang muncul di kedua teks
- `feat_pair_overlap_ratio`: Rasio overlap terhadap teks yang lebih pendek
- `feat_pair_length_diff`: Selisih absolut panjang karakter
- `feat_pair_length_ratio`: Rasio perbandingan panjang kedua teks

## Cara Menggunakan
```python
from modular_variations.feature_engineering.pairwise_text_similarity.fe_pairwise import extract_pairwise_similarity_features

train_features_df = extract_pairwise_similarity_features(train_df, text_col_a='full_text', text_col_b='prompt_text')
test_features_df  = extract_pairwise_similarity_features(test_df, text_col_a='full_text', text_col_b='prompt_text')
```

