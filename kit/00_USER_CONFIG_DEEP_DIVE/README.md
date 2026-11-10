# 📚 00_USER_CONFIG_DEEP_DIVE: Panduan Konsep Mendalam Master User Config
## Dokumentasi Rujukan untuk Mekanisme Non-Trivial di `NLP-FULL KIT`

Folder ini berada di luar `01_NLU/` dan `02_NLG/` sebagai **pusat pemahaman konsep teknis mendalam** di balik parameter-parameter pada **Master User Configuration** yang digunakan di seluruh 17 notebook.

---

## 🗂️ Daftar Modul Dokumentasi di Folder Ini

0. **[`00_PANIC_BUTTON_SAFE_DEFAULTS.md`](./00_PANIC_BUTTON_SAFE_DEFAULTS.md)** *(🚨 BACA INI DULU SAAT ONSITE!)*
   - **Panduan Tombol Anti-Panik:** Setelan "Sapu Jagat" (*Golden Safe Defaults*) kalau Anda sedang pusing, bingung, atau dikejar waktu saat final 5 jam. Cukup ubah 3 baris path/kolom dan biarkan sisanya pada setelan aman bawaan!

1. **[`01_LOSS_VS_METRIC_VS_CHECKPOINT_MONITOR.md`](./01_LOSS_VS_METRIC_VS_CHECKPOINT_MONITOR.md)**
   - Membedakan secara tuntas antara **`LOSS_FUNCTION`** (pelatih gradien penghasil `train_loss` & `val_loss`), **`EVAL_METRIC`** (rumus juri Leaderboard penghasil `val_f1` / `val_qwk`), dan **`CHECKPOINT_MONITOR`** (penentu penyimpanan bobot `.pth`: `'val_loss'` vs `'val_f1'` vs `'hybrid'`).
   - Dilengkapi tabel simulasi perjalanan Epoch 1 s/d Epoch 5.

2. **[`02_ADVANCED_HYPERPARAMETERS_AND_VRAM_MATH.md`](./02_ADVANCED_HYPERPARAMETERS_AND_VRAM_MATH.md)**
   - Mengapa pemotongan teks `TRUNCATION_SIDE = 'head_tail'` (25% awal + 75% akhir) jauh lebih unggul daripada `'right'` untuk artikel dan esai.
   - Matematika memori kuadratik $\mathcal{O}(L^2)$ pada `MAX_LENGTH` dan cara kerja `GRAD_ACCUM_STEPS` agar GPU RTX 3050 (4GB VRAM) tidak pernah *Out of Memory*.
   - Konsep parameter `FOCAL_GAMMA`, `LABEL_SMOOTHING`, dan `FBETA_VAL = 5.0` (`f5_pii`).

3. **[`03_POSTPROCESSING_AND_NELDER_MEAD_MECHANICS.md`](./03_POSTPROCESSING_AND_NELDER_MEAD_MECHANICS.md)**
   - Bagaimana satu toggle `OPTIMIZE_THRESHOLDS = True` secara otomatis mencabangkan algoritma optimasi pasca-pelatihan (*Nelder-Mead Class Multipliers*, *Temperature Scaling*, dan *QWK Ordinal Threshold Cutter*) serta mengatur format output submission (`LABEL` vs `PROBA`).

