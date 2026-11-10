# 🔍 Specific EDA: Text Classification Decision Matrix

Folder ini berisi notebook diagnostik khusus kasus klasifikasi teks (`eda_classification.ipynb`).

### Indikator Diagnostik & Solusi Modul:

| Temuan Diagnostik EDA | Masalah yang Terjadi | Solusi Modul Lego yang Dipasang |
| :--- | :--- | :--- |
| **Imbalance Ratio $> 3:1$** | Model bias ke kelas mayoritas; akurasi tinggi semu tapi Macro F1 anjlok | Gunakan `modular_variations/model_configs/focal_loss_imbalance/` |
| **Threshold 0.5 Kurang Optimal** | Precision/Recall tidak seimbang pada kelas minoritas | Gunakan `modular_variations/post_processing/probability_threshold_tuning/` |
| **Banyak Singkatan Gaul / Typo** | Kosakata OOV tinggi pada data teks media sosial | Pasang `modular_variations/preprocessing/indonesian_slang_cleaning/` |
| **Model Overfitting Cepat** | Akurasi training 99% tetapi CV fold stagnan di 80% | Pasang `modular_variations/model_configs/multi_sample_dropout/` |
