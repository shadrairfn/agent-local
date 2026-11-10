# ⚡ Minimal Preprocessing (Transformer Priority)

## Filosofi & Rationale
Model transformer modern (seperti DeBERTa, RoBERTa, dan IndoBERT) dilatih dengan tujuan *Masked Language Modeling (MLM)* pada korpus teks alami utuh. Huruf kapital (penanda entitas/awal kalimat), tanda baca (`?`, `!`, `,`), dan struktur paragraf mengandung bobot perhatian (*attention weights*) yang krusial. 

Modul ini **hanya menormalkan unicode (NFKD) dan merapikan spasi berlebih**, tanpa menyentuh huruf kapital maupun tanda baca.

## Kapan Menggunakan
- Wajib digunakan saat model utama Anda adalah **Pretrained Transformer (DeBERTa-v3, IndoBERT, RoBERTa, DistilBERT)**.

## Cara Menggunakan
```python
from modular_variations.preprocessing.minimal_transformer.prepro_minimal import apply_minimal_preprocessing

train_df = apply_minimal_preprocessing(train_df, text_cols=['full_text'])
test_df  = apply_minimal_preprocessing(test_df, text_cols=['full_text'])
```

