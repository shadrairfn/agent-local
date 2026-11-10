# ✂️ Post-Processing: Bounded Clipping & Quantile Calibration

## Deskripsi & Rationale
Prediksi model regresi (terutama linear head atau neural network tak terikat) dapat memproyeksikan nilai di luar rentang valid dunia nyata (misalnya probabilitas toksisitas bernilai negatif $-0.04$ atau skor rating bernilai $> 5.0$). 

Modul ini menyediakan:
1. **`apply_boundary_clipping`**: Memotong nilai prediksi pada batas legal minimum dan maksimum domain data.
2. **`apply_quantile_scaling`**: Menyelaraskan kembali deviasi standar prediksi terhadap deviasi standar target asli di OOF untuk melawan penyusutan variansi (*variance collapse*).

## Cara Menggunakan
```python
from modular_variations.post_processing.bounded_clipping_calibration.postpro_clipping import apply_boundary_clipping

final_preds = apply_boundary_clipping(test_preds, min_val=0.0, max_val=1.0)
```

