# 🔤 Standard Preprocessing (BiLSTM / Word2Vec / FastText)

## Filosofi & Rationale
Model berbasis recurrent neural network (BiLSTM/GRU) maupun embedding kata statis (Word2Vec, GloVe) memiliki ukuran kamus (*embedding table*) yang terbatas. Menyamakan huruf kapital menjadi huruf kecil (*lowercasing*) dan menguraikan singkatan bahasa Inggris (*contractions*) membantu menyusutkan ukuran embedding dan mencegah token kata langka menjadi *out-of-vocabulary (OOV)*.

## Kapan Menggunakan
- Digunakan saat membangun model baseline Deep Learning ringan seperti **BiLSTM, BiGRU, TextCNN, atau FastText**.

## Cara Menggunakan
```python
from modular_variations.preprocessing.standard_bilstm.prepro_standard import apply_standard_preprocessing

train_df = apply_standard_preprocessing(train_df, text_cols=['full_text'])
test_df  = apply_standard_preprocessing(test_df, text_cols=['full_text'])
```

