# 🚀 Persiapan Soal ke-2 GEMASTIK, Network Graph EDA & Kombinasi Regex Bahasa Indonesia

Folder khusus ini berada di luar `01_NLU` dan `02_NLG` sebagai **Senjata Taktis Soal ke-2 GEMASTIK Data Mining**, mencakup:
1. **Konteks & Pipeline Prediksi Soal ke-2 (Sambungan Aksara & Chained Context Prediction)**
2. **Network Graph Representation EDA & Graph Feature Engineering**
3. **Kombinasi Regex & Normalisasi Bahasa Indonesia Terpadu (`indonesian_regex_combinator.py`)**

---

## 📂 Daftar File & Notebook

| File / Notebook | Fungsi Utama & Kapan Digunakan | Output Model & Submission |
| :--- | :--- | :--- |
| [`notebooks/01_soal2_sambungan_aksara_dan_makna.ipynb`](notebooks/01_soal2_sambungan_aksara_dan_makna.ipynb) | **Solver Soal 2 Tipe A (Sambungan Kata Aksara -> Pemenggalan & Makna):** Memisahkan teks aksara/OCR yang menyambung tanpa spasi (*scriptio continua*) menggunakan **Hybrid Viterbi DP + PyTorch Character-BiLSTM Boundary Detector**, lalu memprediksi makna Bahasa Indonesia & kategorinya. | `models/char_bilstm_segmenter.pth`, `models/aksara_meaning_lgb_fold*.txt`, `submission_soal2_aksara_postprocessed.csv` |
| [`notebooks/02_soal2_chained_context_predictor.ipynb`](notebooks/02_soal2_chained_context_predictor.ipynb) | **Solver Soal 2 Tipe B (Prediksi Berantai Soal 1 $\to$ Soal 2):** Memanfaatkan label & probabilitas prediksi Soal 1 (`[KONTEKS_SOAL_1: ...]` + probability stacking) untuk mengekstrak frasa aspek (`aspect_rationale_span`) dan memprediksi tindakan lanjutan Soal 2. | `models/soal2_chained_lgb_fold*.txt`, `submission_soal2_chained_postprocessed.csv` |
| [`notebooks/03_network_graph_representation_eda.ipynb`](notebooks/03_network_graph_representation_eda.ipynb) | **Network Graph Representation EDA & Graph Features:** Membangun **Word Co-occurrence Network (PageRank & Degree Centrality)** serta **Document $k$-NN Homophily Graph** untuk visualisasi babak final dan ekstraksi fitur `graph_oof_neighbor_prob` untuk LightGBM. | `models/graph_enhanced_lgb_fold*.txt`, `models/03_graph_features_config_results.json` |
| [`notebooks/04_indonesian_regex_combinator_pipeline.ipynb`](notebooks/04_indonesian_regex_combinator_pipeline.ipynb) | **7 Lego Bricks Regex Bahasa Indonesia:** Mengekstrak **12 fitur sinyal pra-pembersihan** (`rx_caps_ratio`, `rx_angka2_redup_count`, `rx_negation_word_count`, dll.) dan menormalisasi kata ulang angka (`error2` $\to$ `error-error`), elongasi, 180+ slang, serta negasi secara aman. | `models/04_indo_regex_config_results.json` |
| [`indonesian_regex_combinator.py`](indonesian_regex_combinator.py) | Modul Python mandiri berisi class `IndonesianRegexCombinator` yang bisa langsung di-import di seluruh notebook NLU maupun NLG. | Modul helper siap pakai |

---

## 🧭 Panduan Taktis Saat Hari-H (Kapan Pakai yang Mana?)

1. **Jika Soal 2 berupa kelanjutan dari hasil OCR / Aksara Nusantara yang hurufnya menyambung tanpa spasi:**
   - Buka [`01_soal2_sambungan_aksara_dan_makna.ipynb`](notebooks/01_soal2_sambungan_aksara_dan_makna.ipynb).
   - Isi `UNSEGMENTED_COL` dengan kolom aksara sambung dan `MEANING_TARGET_COL` dengan kolom arti/terjemahan.
2. **Jika Soal 2 meminta prediksi lanjutan berdasarkan hasil klasifikasi Soal 1 (misal: setelah tahu sentimen negatif di Soal 1, prediksi aspek keluhan & prioritas tindakan di Soal 2):**
   - Buka [`02_soal2_chained_context_predictor.ipynb`](notebooks/02_soal2_chained_context_predictor.ipynb).
   - Masukkan kolom prediksi Soal 1 ke `STAGE1_LABEL_COL` dan probabilitas OOF/test Soal 1 ke `STAGE1_PROB_COLS`.
3. **Jika ingin meningkatkan skor model Bahasa Indonesia (Sosmed/Ulasan/Berita) & memperkaya slide presentasi Babak Final:**
   - Jalankan [`03_network_graph_representation_eda.ipynb`](notebooks/03_network_graph_representation_eda.ipynb) untuk mendapatkan grafik jejaring kata & fitur PageRank/Homophily.
   - Gunakan `IndonesianRegexCombinator` dari [`04_indonesian_regex_combinator_pipeline.ipynb`](notebooks/04_indonesian_regex_combinator_pipeline.ipynb) agar kata negasi (`ga`, `gk`, `tdk`, `bukan`, `belum`) tidak pernah terhapus dan terunifikasi menjadi `tidak`.
