# 🍕 Error Analysis: Slice-Based Performance Auditing

## Deskripsi & Rationale
Sering kali skor evaluasi global terlihat bagus, namun saat dibedah per subset data (*slices*), model ternyata gagal total pada dokumen pendek (*short texts*) atau mengalami error ganda pada topik tertentu. Modul ini membedah metrik RMSE dan MAE berdasarkan kuantil panjang dokumen dan grup kategori untuk mengisolasi kelemahan spesifik model.

## Cara Menggunakan
```python
from modular_variations.error_analysis.slice_based_eval.error_slices import evaluate_slices_by_length_and_group

summary = evaluate_slices_by_length_and_group(
    df=oof_df,
    text_col='full_text',
    target_col='score',
    pred_col='oof_pred',
    group_col='prompt_id'
)
```

