# 🧩 Modul Modular: Post-Processing Variations

Folder ini menyediakan modul kalibrasi dan pasca-pemrosesan hasil prediksi yang terbukti menaikkan peringkat leaderboard di detik-detik akhir:

| Subfolder Variasi | Kegunaan Utama |
| :--- | :--- |
| **[`threshold_qwk_nelder_mead/`](threshold_qwk_nelder_mead/)** | Optimasi ambang batas diskrit Nelder-Mead pada OOF untuk memaksimalkan metrik QWK. |
| **[`bounded_clipping_calibration/`](bounded_clipping_calibration/)** | Pemotongan nilai pada rentang batas legal target $[0, 1]$ atau rentang data latihan. |
| **[`rank_power_averaging/`](rank_power_averaging/)** | Rank averaging dan power transformation untuk ensembling berbagai jenis model. |

