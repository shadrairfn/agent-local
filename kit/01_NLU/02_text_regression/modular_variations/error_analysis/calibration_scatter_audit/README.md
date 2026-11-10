# 🎯 Error Analysis: Calibration & Quantile Distribution Audit

## Deskripsi & Rationale
Pada regresi, model sering mengalami *prediction compression* (variansi prediksi jauh lebih kecil daripada variansi target asli). Model cenderung bermain aman dengan memprediksi nilai di sekitar mean. Modul ini mengaudit persentil prediksi (P01, P25, P50, P75, P99) dan mendeteksi apakah model gagal menjangkau nilai ekstrem atas dan bawah.

## Cara Menggunakan
```python
from modular_variations.error_analysis.calibration_scatter_audit.error_calibration import audit_prediction_calibration

calib_df = audit_prediction_calibration(y_true=oof_true, y_pred=oof_pred)
```

