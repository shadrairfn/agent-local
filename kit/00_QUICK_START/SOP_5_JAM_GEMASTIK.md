# ⏱️ SOP 5 JAM BABAK FINAL GEMASTIK XIX 2026
## Panduan Operasional Menit-demi-Menit untuk Tim Natural Language Processing

---

### [00:00 - 00:15] TAHAP 1: TRIAGE KILAT & PEMILIHAN KIT
1. Unduh dataset lomba (`train.csv`, `test.csv`, `sample_submission.csv`) ke laptop Anda.
2. Buka terminal di folder `NLP-FULL KIT` dan jalankan:
   ```bash
   python 00_QUICK_START/competition_triage.py --train path/to/train.csv --test path/to/test.csv
   ```
3. Lihat rekomendasi triage:
   * Jika **Klasifikasi Teks** $\rightarrow$ Buka `01_NLU/01_text_classification/`
   * Jika **Regresi / Penilaian** $\rightarrow$ Buka `01_NLU/02_text_regression/`
   * Jika **Ekstraksi Token / NER** $\rightarrow$ Buka `01_NLU/03_token_classification_ner/`
   * Jika **Peringkasan / Generatif** $\rightarrow$ Buka `02_NLG/01_text_summarization/`

---

### [00:15 - 00:45] TAHAP 2: ANCHOR BENCHMARK SUBMISSION (DALAM < 30 MENIT)
1. Buka notebook pertama: `01_ml_gbdt_*.ipynb`.
2. Salin blok kode **Master User Config (Sel 1)** yang dicetak oleh `competition_triage.py`.
3. Set `SMOKE_TEST = True` $\rightarrow$ Run All (verifikasi 10 detik lulus format submission).
4. Set `SMOKE_TEST = False` $\rightarrow$ Run All (latih fold 0 atau 5-fold GBDT, memakan waktu 2–4 menit).
5. Segera kirim `submission_*_postprocessed.csv` ke portal lomba!
   * *Hasil:* Tim Anda sudah memiliki skor patokan awal di leaderboard, rasa panik hilang!

---

### [00:45 - 01:15] TAHAP 3: DIAGNOSTIK DATA (EDA SPESIFIK)
1. Buka folder `eda/` di dalam kit kasus Anda.
2. Jalankan notebook `eda_*.ipynb` untuk mendiagnosis:
   * Apakah ada kelas timpang ekstrim?
   * Apakah target regresi memiliki skewness tinggi?
   * Apakah ada token duplikat atau kolom grup (`prompt_id`) yang bocor?
3. Baca `README.md` di folder `eda/` untuk memilih modul penjinak yang sesuai dari `modular_variations/`.

---

### [01:15 - 03:30] TAHAP 4: FULL DEEP LEARNING (TRANSFORMER DEBERTA / INDOBERT)
1. Buka notebook `02_transformer_*.ipynb`.
2. Pastikan `MIXED_PRECISION = True` aktif agar memanfaatkan Tensor Cores GPU RTX 3050 (hemat VRAM & 3x lebih cepat).
3. Latih 5-Fold Cross Validation.
4. Simpan Out-Of-Fold (OOF) predictions (`.npy`) dan test predictions (`.npy`).
5. Kirim submission kedua dari model Transformer ini ke portal!

---

### [03:30 - 04:30] TAHAP 5: ENSEMBLE BLENDING & POST-PROCESSING CALIBRATION
1. Buka notebook `03_ensemble_*.ipynb`.
2. Muat prediksi OOF dari model GBDT dan Transformer.
3. Jalankan optimasi bobot OOF ($w_A \cdot \text{GBDT} + w_B \cdot \text{Transformer}$).
4. Terapkan kalibrasi post-processing:
   * Klasifikasi: Threshold probability tuning untuk memaksimalkan Macro F1.
   * Regresi: Nelder-Mead threshold cutter (untuk QWK esai) atau clipping interval.
5. Ekspor `submission_ensemble_postprocessed.csv`!

---

### [04:30 - 05:00] TAHAP 6: FINAL INTEGRITY CHECK & KUNCI FILE
1. Lakukan audit integritas terakhir:
   * `len(submission) == len(test_df)`?
   * Bebas dari `NaN`, `null`, atau `inf`?
   * Kolom ID dan Target identik dengan `sample_submission.csv`?
2. Pilih file submission terbaik di portal lomba.
3. Siapkan slide presentasi teknis menggunakan insight dan grafik dari notebook EDA!

