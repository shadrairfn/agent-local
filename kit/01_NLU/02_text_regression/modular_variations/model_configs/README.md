# 🧩 Modul Modular: Model Architectures & Advanced Heads

Folder ini menyediakan modul konfigurasi arsitektur neural network dan model GBDT:

| Subfolder Variasi | Rincian Komponen | Rationale Kompetisi |
| :--- | :--- | :--- |
| **[`pooling_heads/`](pooling_heads/)** | Mean, CLS, Max, Attention, dan GeM Pooling. | Mengonversi sekuens token transformer menjadi vektor representasi teks tunggal. |
| **[`multi_sample_dropout/`](multi_sample_dropout/)** | 5 jalur dropout paralel ($p=0.1$ s.d. $0.5$) dirata-ratakan. | Mempercepat konvergensi dan mencegah overfitting gradient update. |
| **[`llrd_optimization/`](llrd_optimization/)** | Layerwise Learning Rate Decay (faktor peluruhan $\approx 0.90$). | Mencegah *catastrophic forgetting* pada layer representasi bahasa dasar transformer. |
| **[`gbdt_presets/`](gbdt_presets/)** | Hyperparameter presets untuk LightGBM dan CatBoost. | Konfigurasi optimal untuk baseline cepat maupun model ensembling. |

