# ⚡ Post-Processing: Rank Averaging & Power Transformation

## Deskripsi & Rationale
Saat menggabungkan model yang memiliki skala atau kalibrasi probabilitas berbeda (misalnya GBDT berbasis TF-IDF vs Transformer DeBERTa):
- **Rank Averaging:** Mengonversi prediksi numerik ke persentil peringkat $[0.0, 1.0]$ sebelum merata-ratakannya. Sangat superior untuk metrik berbasis peringkat dan korelasi seperti **ROC-AUC, Spearman $\rho$, dan Pearson $r$**.
- **Power Averaging:** Menerapkan eksponen pangkat $p$ pada prediksi sebelum dirata-ratakan. Sangat efektif untuk metrik loss kuadratik (RMSE) karena memberikan bobot ekstra pada prediksi yang berkeyakinan tinggi (*high-confidence predictions*).

## Cara Menggunakan
```python
from modular_variations.post_processing.rank_power_averaging.postpro_ensembling import rank_average_predictions

ensemble_preds = rank_average_predictions([gbdt_preds, deberta_preds, roberta_preds])
```

