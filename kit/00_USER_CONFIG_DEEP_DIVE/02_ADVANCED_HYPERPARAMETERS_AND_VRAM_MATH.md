# ⚙️ 02. Bedah Parameter Lanjutan: `TRUNCATION_SIDE`, `MAX_LENGTH`, `GRAD_ACCUM`, & Loss Khusus
## Memahami Konsep Teknis di Balik Konfigurasi Non-Trivial pada `NLP-FULL KIT`

Dokumen ini menjelaskan parameter-parameter di dalam **Master User Configuration** yang memiliki dampak besar terhadap akurasi dan ketahanan memori GPU RTX 3050 (4GB VRAM).

---

## 1. Mengapa `TRUNCATION_SIDE = 'head_tail'` Sangat Krusial untuk Artikel & Esai?

Model Transformer memiliki batas panjang token (`MAX_LENGTH`, misalnya `128` atau `256` kata/subword). Bagaimana jika sebuah esai atau berita memiliki panjang **600 kata**?

- **`'right'` (Standar Default Library Biasa — Potong Ekor):**
  Hanya mengambil 128 kata pertama dan **membuang 472 kata sisanya**.
  *Bahayanya:* Dalam penulisan berita, esai mahasiswa, atau ulasan produk berbahasa Indonesia, **kalimat kesimpulan/opini paling tegas justru berada di paragraf paling akhir!** Jika dipotong `'right'`, kalimat kesimpulan hilang total.
- **`'left'` (Potong Kepala):**
  Hanya mengambil 128 kata terakhir dan membuang judul serta paragraf pembuka.
- **`'head_tail'` (Strategi Juara Kaggle — 25% Awal + 75% Akhir):**
  Mengambil **25% kata pertama** (judul + kalimat topik pembuka) lalu disambung langsung dengan **75% kata terakhir** (argumen inti + kalimat kesimpulan penutup), dan hanya membuang bagian tengah yang biasanya berisi penjabaran bertele-tele.
  > **Kapan tidak boleh memakai `'head_tail'`?**
  > Hanya pada **`03_token_classification_ner` (NER)**, di mana urutan indeks kata per kata harus sejajar dengan daftar label BIO (`TRUNCATION_SIDE = 'right'`).

---

## 2. Mengapa `MAX_LENGTH` Memakan VRAM Secara Kuadratik $\mathcal{O}(L^2)$?

Pada arsitektur *Self-Attention* Transformer, setiap kata dibandingkan dengan seluruh kata lain dalam kalimat, menghasilkan matriks perhatian berukuran $L \times L$:
- Menaikkan `MAX_LENGTH` dari `128` ke `256` ($2\times$ lipat) akan melipatgandakan beban memori matriks perhatian menjadi **$4\times$ lipat**!
- Menaikkan `MAX_LENGTH` dari `128` ke `512` ($4\times$ lipat) akan melipatgandakan beban memori menjadi **$16\times$ lipat**!

### Cara Mengakali `MAX_LENGTH = 256` atau `512` di GPU 4GB VRAM (`GRAD_ACCUM_STEPS`):
Jika Anda harus memproses teks panjang (`MAX_LENGTH = 256` atau `512`) atau memakai model besar (`xlm-roberta-base`), jangan paksakan `BATCH_SIZE = 16` karena akan terjadi `CUDA Out of Memory`.
Gunakan rumus **Effective Batch Size**:
$$\text{Effective Batch Size} = \text{BATCH\_SIZE} \times \text{GRAD\_ACCUM\_STEPS}$$

| Panjang Teks (`MAX_LENGTH`) | Setelan `BATCH_SIZE` (Fisik di VRAM) | Setelan `GRAD_ACCUM_STEPS` (Akumulasi) | Batch Efektif (Untuk Kestabilan Gradien) | Status di RTX 3050 4GB (FP16=True) |
| :---: | :---: | :---: | :---: | :---: |
| **`64` – `128`** | `16` | `1` | $16 \times 1 = 16$ | ✅ Sangat Aman (~1.8 GB VRAM) |
| **`256`** | `8` | `2` | $8 \times 2 = 16$ | ✅ Aman (~2.7 GB VRAM) |
| **`512`** | `4` | `4` | $4 \times 4 = 16$ | ✅ Aman (~3.4 GB VRAM) |

*Cara kerja `GRAD_ACCUM_STEPS = 4`:* Model memproses 4 kalimat sekaligus (hemat VRAM), menyimpan gradiennya di kantong sementara sebanyak 4 kali berturut-turut, baru melakukan `optimizer.step()` setelah terkumpul 16 kalimat! Hasil matematikanya **identik dengan `BATCH_SIZE = 16`**, tetapi hemat memori hingga 75%!

---

## 3. Bedah Parameter Loss Tingkat Lanjut (`FOCAL_GAMMA`, `LABEL_SMOOTHING`, `FBETA_VAL`)

### A. `FOCAL_GAMMA = 2.0` (Pada `LOSS_FUNCTION = 'focal_loss'`)
Rumus Focal Loss adalah:
$$\mathcal{L}_{\text{Focal}} = -(1 - p_t)^\gamma \log(p_t)$$
- Jika $\gamma = 0$: Kembali menjadi Cross-Entropy biasa.
- Jika $\gamma = 2.0$ (Standar Emas):
  - Kalimat mudah yang sudah ditebak benar oleh model dengan keyakinan $p_t = 0.90$ hanya diberi bobot $(1 - 0.90)^2 = \mathbf{0.01}$ ($1\%$).
  - Kalimat sulit / kelas minoritas yang masih salah ditebak ($p_t = 0.10$) diberi bobot $(1 - 0.10)^2 = \mathbf{0.81}$ ($81\times$ lebih besar!).
- Jika data Anda memiliki kelas super langka ($< 2\%$), Anda bisa menaikkan `FOCAL_GAMMA = 3.0`.

### B. `LABEL_SMOOTHING = 0.05` vs `0.0`
- Saat `LABEL_SMOOTHING = 0.05`, label target one-hot `[0, 1, 0]` diubah menjadi `[0.0167, 0.9667, 0.0167]`.
- *Mengapa penting?* Tanpa *label smoothing*, jaringan saraf berusaha mendorong *logit* menuju $+\infty$ agar probabilitas mencapai $0.999999$. Hal ini menyebabkan model mudah *overfitting* terhadap salah ketik label (*human annotation noise*).
- *Kapan wajib di-set `0.0`?* Ketika metrik evaluasi lomba adalah **`'log_loss'`** murni dan data latih sangat bersih, karena *label smoothing* secara sengaja menahan probabilitas puncak di sekitar $0.95$.

### C. `FBETA_VAL = 5.0` (Pada `EVAL_METRIC = 'f5_pii'`)
Rumus umum $F_\beta$ adalah:
$$F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$
- Jika $\beta = 1.0$ ($F_1$): Precision dan Recall sama penting ($1 : 1$).
- Jika $\beta = 2.0$ ($F_2$): Recall $4\times$ lebih penting dari Precision.
- Jika $\beta = 5.0$ ($F_5$ — Standar Kaggle PII Detection): **Recall $25\times$ lebih penting dari Precision!** Artinya, membiarkan 1 data rahasia/entitas lolos (*False Negative*) dihukum 25 kali lebih berat daripada salah menandai kata biasa sebagai entitas (*False Positive*).

---

## 🛟 4. TIPS AMAN SAAT ONSITE (SETELAN ANTI-OOM & ANTI-PANIK RTX 3050)

Kalau saat lomba *onsite* Anda tidak sempat menganalisis panjang teks atau takut terkena *CUDA Out of Memory*, **pegang teguh 4 aturan aman ini**:
1. **Biarkan `TRUNCATION_SIDE = 'head_tail'`** (kecuali soal NER yang sudah otomatis `'right'`). Anda tidak perlu pusing memikirkan apakah informasi penting ada di awal atau di akhir paragraf—keduanya pasti terbawa!
2. **Biarkan `MAX_LENGTH = 128`, `BATCH_SIZE = 16`, dan `USE_FP16 = True`**. Ini adalah setelan paling aman di dunia untuk laptop RTX 3050 4GB (hanya memakan ~1.8 GB VRAM dan berjalan sangat cepat).
3. **Untuk `MODEL_NAME`**: Jika bingung pilih model mana di folder `03_OFFLINE_WEIGHTS/weights/`:
   - Kalau teksnya **Bahasa Indonesia / Campuran / Daerah** $\to$ pakai **`'distilbert-multilingual'`** (paling cepat & ringan) untuk percobaan pertama, lalu **`'indobert-base'`** untuk model kedua.
   - Kalau teksnya **Bahasa Inggris** $\to$ pakai **`'deberta-v3-small'`** atau **`'distilbert-uncased'`** (super kilat).
4. **Jangan sentuh `FOCAL_GAMMA` atau `LABEL_SMOOTHING`**—nilai bawaannya (`2.0` dan `0.05`) sudah merupakan standar emas di semua kompetisi Kaggle.


