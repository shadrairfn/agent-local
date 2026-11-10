# 🔍 Error Analysis: Residual & Outlier Drill-Down

## Deskripsi & Rationale
Setelah model selesai dilatih pada 5-fold cross-validation, langkah terpenting untuk meningkatkan skor leaderboard adalah **menganalisis di mana model gagal**. Modul ini menghitung residual ($e = y - \hat{y}$), mendeteksi bias sistematis (apakah model cenderung memprediksi terlalu rendah atau terlalu tinggi), dan mencetak cuplikan teks dari sampel-sampel dengan error terbesar.

## Cara Menggunakan
```python
from modular_variations.error_analysis.residual_outlier_drilldown.error_residuals import analyze_residuals_and_outliers

results = analyze_residuals_and_outliers(
    df=oof_df,
    id_col='id',
    text_col='full_text',
    target_col='score',
    pred_col='oof_pred',
    top_k=5
)
```

