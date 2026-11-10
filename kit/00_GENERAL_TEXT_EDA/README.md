# 🌐 General Text EDA: Universal Diagnostics Matrix

Folder ini berisi **4 Notebook Diagnostik Universal** yang wajib dijalankan pertama kali untuk masalah teks apa pun (baik NLU maupun NLG) sebelum memulai modeling:

| Notebook Diagnostik | Masalah yang Dianalisis | Threshold Alarm | Tindakan Rekayasa Pipeline |
| :--- | :--- | :--- | :--- |
| **`01_token_length_and_truncation.ipynb`** | Distribusi panjang token & kuantil 512 | > 10% teks > 512 token | Terapkan *Head + Tail Truncation* (misal 128 awal + 384 akhir). |
| **`02_vocabulary_and_oov_overlap.ipynb`** | Out-Of-Vocabulary rate test vs train | OOV Rate > 20% | Wajib gunakan Subword Tokenizer BPE (DeBERTa); hindari TF-IDF murni. |
| **`03_text_noise_and_cleaning.ipynb`** | Tag HTML, entities, unicode rusak | Ditemukan > 0.1% baris | Jalankan `html.unescape` dan pembersihan tag regex sebelum tokenisasi. |
| **`04_group_leakage_and_duplicates.ipynb`**| Duplikasi teks & kolom prompt/author | Kolom grup ada di train & test | **WAJIB gunakan `GroupKFold`** untuk mencegah skor CV palsu. |
