# 📊 Feature Engineering: Multimodal Tabular Metadata Fusion

## Deskripsi & Rationale
Digunakan saat dataset NLP tidak hanya berisi teks, melainkan juga menyertakan kolom tabular metadata (misalnya kategori produk, lokasi, jumlah viewing, brand, atau skor tambahan). Modul ini menstandarisasi fitur numerik dengan imputasi median dan menyandikan kolom kategorikal dengan *Frequency Encoding* berbasis data latihan secara anti-kebocoran.

## Cara Menggunakan
```python
from modular_variations.feature_engineering.multimodal_tabular_fusion.fe_tabular_fusion import prepare_tabular_metadata_features

train_df, test_df = prepare_tabular_metadata_features(
    train_df, test_df,
    meta_num_cols=['word_count'],
    meta_cat_cols=['category']
)
```

