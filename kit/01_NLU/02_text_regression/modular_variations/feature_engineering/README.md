# 🧩 Modul Modular: Feature Engineering Variations

Folder ini menyediakan modul ekstraksi fitur teks dan tabular yang seragam menghasilkan kolom berawalan prefix `feat_`, sehingga otomatis terseleksi oleh filter dinamis GBDT:

| Subfolder Variasi | Rincian Fitur Utama | Kapan Digunakan |
| :--- | :--- | :--- |
| **[`surface_length_stats/`](surface_length_stats/)** | Panjang karakter, kata, kalimat, rasio TTR, densitas angka/kapital. | **Wajib di seluruh kasus regresi teks.** |
| **[`readability_complexity/`](readability_complexity/)** | Formula Flesch-Kincaid, Gunning-Fog, ARI, Coleman-Liau, rasio kata kompleks. | Keterbacaan teks (CommonLit, esai analitis). |
| **[`lexical_sentiment_affix/`](lexical_sentiment_affix/)** | Rasio sentimen positif/negatif, stopword ratio, densitas afiks akademik. | Analisis sentimen, teks esai ilmiah, CSAT. |
| **[`pairwise_text_similarity/`](pairwise_text_similarity/)** | Jaccard similarity, word overlap, selisih dan rasio panjang antar teks. | Dataset pasangan teks (Patent, Prompt-Summary, Margin). |
| **[`lsa_tfidf_topics/`](lsa_tfidf_topics/)** | Proyeksi topik padat TruncatedSVD dari n-gram TF-IDF (fitted on train only). | Representasi semantik padat untuk model GBDT. |
| **[`multimodal_tabular_fusion/`](multimodal_tabular_fusion/)** | Imputasi & standarisasi numerik, frequency encoding kategorikal metadata. | Dataset gabungan teks + tabular (Mercari, PetFinder). |

