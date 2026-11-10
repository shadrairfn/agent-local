# 📂 Demo Multi-Task Dataset (Level A NLP)

Dataset sintetis ini dirancang untuk memverifikasi pipeline NLP, uji coba smoke test (10 detik), dan pengujian switchboard multi-skenario secara offline tanpa koneksi internet.

### Struktur Kolom:
- **`id`**: Identifier unik baris.
- **`full_text`**: Kolom teks utama (esai analitis, dwibahasa Indonesia & Inggris).
- **`prompt_text`**: Kolom teks konteks/pasangan (prompt penulisan).
- **`prompt_id`**: Kolom grup entitas (`P101` s.d. `P105`) untuk pengujian validasi anti-bocor `GroupKFold`.
- **`category`**: Metadata kategorikal (topik esai).
- **`word_count`**: Metadata numerik tabular.

### Target Multi-Skenario:
- `score`: Target kontinu (rentang 1.0 s.d. 5.0) untuk `TASK_TYPE = 'single_regression'`.
- `grade`: Target ordinal bertingkat (integer 1 s.d. 6) untuk `TASK_TYPE = 'ordinal_grading'`.
- `toxicity`: Target fraksi batas proporsi [0.0, 1.0] untuk `TASK_TYPE = 'bounded_regression'`.
- `likes_count`: Target hitungan non-negatif untuk `TASK_TYPE = 'counting_volume'`.
- `cohesion`, `syntax`, `vocabulary`: 3 target berkorelasi untuk `TASK_TYPE = 'multi_regression'`.

