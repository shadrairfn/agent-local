# 🧩 Modul Modular: Error Analysis Variations

Folder ini menyediakan modul diagnostik pasca-pelatihan (*post-training error audit*) untuk membongkar kelemahan model secara presisi:

| Subfolder Variasi | Kegunaan Utama |
| :--- | :--- |
| **[`residual_outlier_drilldown/`](residual_outlier_drilldown/)** | Mengidentifikasi Top-K error terbesar, residual skewness, dan mencetak cuplikan teks outlier. |
| **[`slice_based_eval/`](slice_based_eval/)** | Mengaudit performa model per kuantil panjang dokumen (Short, Medium, Long) dan per grup entitas. |
| **[`calibration_scatter_audit/`](calibration_scatter_audit/)** | Mengaudit perbandingan persentil target vs prediksi untuk mendeteksi *variance compression*. |

