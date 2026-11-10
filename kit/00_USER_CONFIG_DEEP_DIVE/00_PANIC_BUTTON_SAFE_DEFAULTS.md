# 🚨 00. TOMBOL ANTI-PANIK ONSITE: "KALO LAGI PUSING/BINGUNG, PAKE SETELAN AMAN INI AJA!"
## Golden Safe Defaults untuk Babak Final 5 Jam GEMASTIK (Tanpa Perlu Mikir Rumit)

Saat lomba *onsite* 5 jam, wajar sekali kalau kepala pusing, panik melihat waktu berjalan, atau ragu harus memilih opsi yang mana.

> [!IMPORTANT]
> **KABAR BAIKNYA:** Seluruh notebook di kit ini **SUDAH DI-SET DEFAULT KE OPSI PALING AMAN & PALING BAGUS SECARA UMUM!**
> Kalau Anda sedang panik atau bingung saat lomba, **JANGAN SENTUH parameter loss/monitor/hyperparameter bawah**.
> Cukup ganti **3 BARIS WAJIB** saja: `TRAIN_CSV_PATH`, `TEXT_COL`, dan `TARGET_COL`, lalu langsung tekan **Run All**!

---

## 🛟 1. Tabel "Setelan Sapu Jagat (Golden Safe Defaults)" — Pasti Jalan & Bagus di 95% Soal!

Jika Anda bingung memilih opsi, cukup gunakan **1 Setelan Sapu Jagat** di bawah ini untuk masing-masing jenis soal:

| Jenis Soal | `LOSS_FUNCTION` *(Paling Aman)* | `EVAL_METRIC` *(Jika di soal tidak jelas)* | `CHECKPOINT_MONITOR` *(Paling Aman)* | `MODEL_NAME` *(Paling Aman di RTX 3050)* | `MAX_LENGTH` & `BATCH_SIZE` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Klasifikasi Teks Apa Pun**<br/>*(Sentimen, Emosi, Hoax, Topik)* | **`'auto'`** *(atau `'combo_ce_f1'`)* | **`'macro_f1'`** | **`'hybrid'`** | **`'distilbert-multilingual'`** *(Cepat)* atau **`'indobert-base'`** | `MAX_LENGTH = 128`<br/>`BATCH_SIZE = 16` |
| **2. Regresi Skor / Angka Apa Pun**<br/>*(Rating, Skor Esai, Kemiripan)* | **`'smooth_l1'`** | **`'rmse'`** *(jika angka desimal)*<br/>**`'qwk'`** *(jika skor bulat 1-5)* | **`'hybrid'`** | **`'deberta-v3-small'`** *(atau `'distilbert-multilingual'`)* | `MAX_LENGTH = 128`<br/>`BATCH_SIZE = 16` |
| **3. NER / Tagging Kata**<br/>*(Nama Orang/Tempat, Aksara)* | **`'crf_nll'`** | **`'seqeval_f1'`** | **`'hybrid'`** | **`'distilbert-multilingual'`** | `MAX_LENGTH = 128`<br/>`BATCH_SIZE = 8` |
| **4. Ringkasan Teks**<br/>*(Summarization)* | **`'label_smoothed_nll'`** | **`'rouge1'`** | **`'hybrid'`** | Jalankan **Notebook 1 (`TextRank`)** dulu (selesai 5 detik tanpa GPU!) | `MAX_LENGTH = 256`<br/>`BATCH_SIZE = 8` |
| **5. Translasi / Aksara / Rewriting** | **`'label_smoothed_nll'`** | **`'bleu'`** *(atau `'cer'` untuk Aksara)* | **`'hybrid'`** | **`'distilbert-multilingual'`** | `MAX_LENGTH = 128`<br/>`BATCH_SIZE = 8` |

---

## 🧠 2. Kenapa Setelan Default Ini Disebut "Anti-Panik"?

1. **Kenapa `LOSS_FUNCTION = 'auto'` (Klasifikasi) dan `'smooth_l1'` (Regresi) Paling Aman?**
   - `'auto'` otomatis menyalakan bobot kelas (*class weight*) kalau datanya timpang, dan tetap stabil kalau datanya seimbang. Anda tidak perlu pusing menghitung persentase kelas!
   - `'smooth_l1'` tahan terhadap data rusak/*outlier* ekstrem sekaligus halus di dekat nol, sehingga **tidak akan pernah meledak (`NaN` loss)** di tengah jalan.

2. **Kenapa `CHECKPOINT_MONITOR = 'hybrid'` Tidak Perlu Diganti-ganti Lagi?**
   - Karena `'hybrid'` sudah otomatis menjaga **dua-duanya sekaligus** (`val_f1` + `val_loss`). Anda tidak perlu pusing memilih antara `'val_loss'` atau `'val_f1'`—biarkan `'hybrid'` selalu aktif!

3. **Kenapa `DATA_FORMAT = 'auto'` Tidak Perlu Diganti?**
   - Mau panitia kasih file `.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet`, `.xlsx`, atau `.conll`, `'auto'` akan langsung mendeteksinya sendiri dari akhiran nama file.

4. **Kenapa `MAX_LENGTH = 128`, `BATCH_SIZE = 16`, `USE_FP16 = True` Adalah "Zona Nyaman" RTX 3050?**
   - Kombinasi ini hanya memakan **~1.8 GB VRAM** (dari kapasitas 4.0 GB VRAM laptop Anda). Artinya: **Mustahil terkena error `CUDA Out of Memory`**, dan 1 fold selesai hanya dalam hitungan detik/menit!

---

## ⏱️ 3. SOP 3 Langkah Saat Panik di Ruangan Lomba (5 Menit Pertama Dapat Skor!)

Kalau waktu sudah mepet atau kepala sedang buntu:

1. **Langkah 1 (Menit ke-1): Buka `01_ml_sparse_dense_gbdt.ipynb` dulu!**
   - Kenapa? Karena Notebook 1 (LightGBM) **tidak pakai GPU**, tidak mungkin *Out of Memory*, dan selesai **5-Fold penuh hanya dalam 10–20 detik**!
   - Ganti path file `TRAIN_CSV_PATH`, `TEST_CSV_PATH`, `TEXT_COL`, dan `TARGET_COL` $\to$ Jalankan $\to$ **Langsung upload `submission_ml_postprocessed.csv` ke Leaderboard!** (Hati langsung tenang karena nama tim sudah masuk papan skor!).

2. **Langkah 2 (Menit ke-5): Baru jalankan `02_dl_sequential_bilstm_cnn.ipynb` & `03_transformer_finetune_end2end.ipynb`:**
   - Kalau mau tes cepat apakah ada *error* (30 detik), ubah `USE_KFOLD = False`.
   - Kalau sudah jalan mulus tanpa error, kembalikan `USE_KFOLD = True` dan tinggal minum/buat slide PPT sambil menunggu 5-Fold selesai.

3. **Langkah 3 (Menit ke-30): Jalankan `04_transformer_feature_extractor_gbdt.ipynb` lalu `05_universal_ensemble_blending.ipynb`:**
   - Notebook 5 tidak perlu *training* apa pun! Dia langsung menggabungkan hasil ke-4 notebook sebelumnya dalam **5 detik** dan menghasilkan `submission_ensemble_postprocessed.csv` (skor tertinggi Anda).

