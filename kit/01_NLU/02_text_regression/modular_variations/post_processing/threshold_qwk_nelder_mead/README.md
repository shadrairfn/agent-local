# 🎯 Post-Processing: Nelder-Mead Threshold Optimization (QWK)

## Deskripsi & Rationale
Saat menyelesaikan kasus penilaian esai (Kaggle AES 2.0 atau scoring bertingkat 1–6), metrik evaluasi resminya adalah **Quadratic Weighted Kappa (QWK)**. Pembulatan standar (`round()`) mengasumsikan jarak antar kelas tersebar seragam pada $x.5$, padahal distribusi kelas esai biasanya memiliki sebaran skewed di mana batas optimal antar nilai tidak berada tepat di tengah.

Modul ini menggunakan optimasi **Nelder-Mead** pada prediksi Out-Of-Fold (OOF) untuk mencari ambang batas pemotongan (*cut-points*) yang secara langsung memaksimalkan metrik QWK.

## Aturan Kompetisi
Threshold **HANYA boleh dicari pada prediksi OOF**, dan TIDAK PERNAH pada test set.

## Cara Menggunakan
```python
from modular_variations.post_processing.threshold_qwk_nelder_mead.postpro_qwk import QWKThresholdOptimizer

optimizer = QWKThresholdOptimizer(num_classes=6, min_grade=1).fit(oof_true, oof_pred)
postprocessed_test_preds = optimizer.transform(raw_test_preds)
```

