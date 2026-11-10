# 🧩 Modul Modular: Preprocessing Variations

Folder ini menyediakan modul pembersihan teks (*data preparation*) yang dirancang modular agar dapat dicopot-pasang secara selektif sesuai dengan jenis model yang digunakan:

| Subfolder Variasi | Model Target | Tindakan Pembersihan Utama |
| :--- | :--- | :--- |
| **[`minimal_transformer/`](minimal_transformer/)** | DeBERTa / IndoBERT / RoBERTa | Menormalkan unicode (NFKD) dan spasi; mempertahankan kapital dan tanda baca. |
| **[`standard_bilstm/`](standard_bilstm/)** | BiLSTM / Word2Vec / FastText | Lowercase, ekspansi kontraksi bahasa Inggris, pembersihan URL. |
| **[`aggressive_tfidf/`](aggressive_tfidf/)** | TF-IDF + LightGBM / CatBoost | Menghapus seluruh tanda baca, angka, dan stopwords untuk menyusutkan dimensi sparse. |
| **[`indonesian_slang_cleaning/`](indonesian_slang_cleaning/)** | Seluruh Model (Kasus Indonesia) | Normalisasi 70+ singkatan gaul Indonesia (`yg` $\to$ `yang`, `dgn` $\to$ `dengan`, `bgt` $\to$ `sangat`). |

