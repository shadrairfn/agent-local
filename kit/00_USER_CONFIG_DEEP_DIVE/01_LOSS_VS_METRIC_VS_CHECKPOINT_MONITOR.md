# 🧠 01. Bedah Tuntas: `LOSS_FUNCTION` vs `EVAL_METRIC` vs `CHECKPOINT_MONITOR`
## Mengapa Ketiganya Berbeda dan Bagaimana Mereka Bekerja Sama di Setiap Epoch?

Dokumentasi ini membahas mekanisme internal yang sering membingungkan saat melatih model Deep Learning / Transformer: **apa bedanya rumus belajar (`LOSS_FUNCTION`), rumus rapor lomba (`EVAL_METRIC`), dan penjaga tombol simpan (`CHECKPOINT_MONITOR`)?**

---

## 1. Analogi Paling Sederhana: "Belajar Memanah untuk Olimpiade"

Bayangkan model AI Anda adalah seorang **atlet panahan** yang sedang berlatih sebanyak **5 Babak (5 Epoch)** untuk mengikuti Olimpiade:

| Variabel di User Config | Peran dalam Analogi Memanah | Kapan Bekerja? |
| :--- | :--- | :--- |
| **`1. LOSS_FUNCTION`**<br/>*(contoh: `'cross_entropy'`, `'focal_loss'`, `'smooth_l1'`)* | **Pelatih Teknik & Jarak Meleset (dalam milimeter).**<br/>Setiap kali anak panah menancap, pelatih mengukur berapa milimeter jaraknya dari titik tengah (`train_loss`), lalu memperbaiki posisi siku dan tarikan busur atlet (`loss.backward()` / *update bobot*).<br/>Saat diuji pada papan sasaran latihan, jarak rata-rata melesetnya disebut **`val_loss` (`eval_loss`)**. | Bekerja **di setiap tembakan (setiap batch)** untuk mengubah bobot saraf model. |
| **`2. EVAL_METRIC`**<br/>*(contoh: `'macro_f1'`, `'qwk'`, `'roc_auc'`)* | **Papan Skor Juri Olimpiade (0 atau 10 Poin).**<br/>Juri Olimpiade tidak peduli anak panah meleset 1 milimeter atau 4 milimeter—selama masih masuk lingkaran kuning nilainya tetap **10 poin**! | Bekerja **di akhir setiap Epoch** untuk menghitung berapa skor Olimpiade (`val_f1` / `val_qwk`) jika model diuji sekarang. |
| **`3. CHECKPOINT_MONITOR`**<br/>*(pilihan: `'val_loss'`, `'val_f1'`, `'hybrid'`)* | **Juru Kamera Penyimpan Rekaman Terbaik (`model.pth`).**<br/>Dari 5 babak latihan (Epoch 1 s/d Epoch 5), kita hanya boleh membawa pulang **1 setelan busur terbaik (`bilstm_fold0.pth`)** ke pertandingan final.<br/>`CHECKPOINT_MONITOR` adalah aturan yang dipakai Si Juru Kamera untuk memutuskan: *"Babak mana yang harus saya simpan ke file `.pth`, dan babak mana yang saya buang?"* | Bekerja **di detik terakhir setiap Epoch** untuk memutuskan apakah file `.pth` di hard disk ditimpa dengan bobot baru atau tidak. |

---

## 2. Simulasi Nyata 5 Epoch: Lihat Mengapa `CHECKPOINT_MONITOR` Sangat Penting!

Misalkan kita melatih model selama **5 Epoch** (`NUM_EPOCHS = 5`) dengan `EVAL_METRIC = 'macro_f1'`.
Di setiap epoch, `LOSS_FUNCTION` mengubah bobot model, lalu kita mendapatkan dua angka validasi: **`val_loss` (`eval_loss`)** dan **`val_f1` (`eval_macro_f1`)**.

Perhatikan tabel perjalanan training dari Epoch 1 sampai Epoch 5 berikut ini:

| Epoch | `train_loss` *(Makin kecil makin hafal)* | `val_loss` / `eval_loss` *(Makin kecil makin bagus $\downarrow$)* | `val_f1` / `eval_macro_f1` *(Makin besar makin bagus $\uparrow$)* | Apa yang Sebenarnya Terjadi pada Model di Epoch Ini? |
| :---: | :---: | :---: | :---: | :--- |
| **Epoch 1** | 0.850 | 0.620 | 0.7100 | Model baru mulai belajar mengenali kata dasar. |
| **Epoch 2** | 0.480 | 0.390 | **0.8400** | Model makin pintar! `val_loss` turun dan `val_f1` naik ke `0.8400`. |
| **Epoch 3** | 0.310 | **0.340 (Terendah!)** | **0.8400 (Seri!)** | Tebakan kelas bulatnya kebetulan masih sama (`val_f1 = 0.8400`), **TAPI** probabilitas model jauh lebih yakin dan terkalibrasi (`val_loss` turun dari `0.390` ke `0.340`)! |
| **Epoch 4** | 0.150 | 0.410 *(Mulai naik)* | **0.8550 (Tertinggi!)** | Model mulai agak *over-confident* pada kelas mayoritas (`val_loss` naik sedikit ke `0.410`), **TAPI** model akhirnya berhasil menebak kelas minoritas yang langka sehingga `val_f1` melonjak ke `0.8550`! |
| **Epoch 5** | 0.040 | 0.680 *(Rusak/Overfit)* | 0.8100 *(Turun)* | Model sudah menghafal mati data train (*overfitting* parah). |

### Sekarang Lihat Bedanya File `.pth` yang Disimpan oleh Ketiga Mode `CHECKPOINT_MONITOR`:

1. **Jika `CHECKPOINT_MONITOR = 'val_loss'` (atau `'eval_loss'`):**
   - Si Juru Kamera **hanya melihat kolom `val_loss`**.
   - Ia akan menyimpan bobot dari **Epoch 3** (`val_loss = 0.340`).
   - *Kelebihan:* Probabilitasnya paling jujur/terkalibrasi (tidak *over-confident*), sangat bagus untuk digabungkan di **Notebook 5 (Ensemble Blending)**.
   - *Kekurangan:* Kehilangan lonjakan skor `val_f1 = 0.8550` yang terjadi di Epoch 4.

2. **Jika `CHECKPOINT_MONITOR = 'val_f1'` (atau `'eval_metric'`):**
   - Si Juru Kamera **hanya melihat kolom `val_f1`**.
   - Dari Epoch 2 (`0.8400`) ke Epoch 3 (`0.8400`), karena angkanya sama, **Epoch 3 diabaikan**! Baru di **Epoch 4** (`0.8550`) ia menyimpan ulang.
   - *Pertanyaan:* Bagaimana jika pelatihan hanya berhenti di Epoch 3 (atau di Epoch 4 tidak ada kenaikan)? Maka `'val_f1'` murni akan memakai bobot Epoch 2 yang masih mentah (`val_loss = 0.390`), bukan Epoch 3 yang lebih matang (`val_loss = 0.340`)!

3. **Jika `CHECKPOINT_MONITOR = 'hybrid'` (Mode Rekomendasi Default Kit):**
   - Si Juru Kamera melihat **`val_f1` sebagai syarat utama**, dan memakai **`val_loss` sebagai pemutus seri (*tie-breaker*)**:
     - Di **Epoch 2**: Simpan (`val_f1` naik ke `0.8400`).
     - Di **Epoch 3**: Walaupun `val_f1` seri (`0.8400`), karena `val_loss` turun (`0.390 -> 0.340`), **tetap simpan Epoch 3!**
     - Di **Epoch 4**: Karena `val_f1` pecah rekor baru (`0.8550`), **simpan Epoch 4!**
     - Di **Epoch 5**: Karena keduanya memburuk, **buang Epoch 5!**

---

## 3. Ringkasan 1 Kalimat agar Tidak Pernah Tertukar Lagi

- **`LOSS_FUNCTION`**: Mengubah bobot saraf model di setiap batch menggunakan kalkulus turunan/gradien (menghasilkan `train_loss` & `val_loss`).
- **`EVAL_METRIC`**: Menghitung skor papan pertandingan sesuai rumus juri di akhir setiap epoch (menghasilkan `val_f1` / `val_qwk` / `val_auc`).
- **`CHECKPOINT_MONITOR`**: Memilih *epoch* mana yang bobotnya disimpan ke dalam file `models/*.pth` (apakah patokannya `val_loss`, `val_f1`, atau `hybrid`).

---

## 🛟 4. TIPS AMAN SAAT ONSITE (KALAU LAGI PUSING / BINGUNG / PANIK)

Kalau saat di ruangan lomba Anda merasa pusing atau ragu harus pilih opsi yang mana, **JANGAN UBAH APA-APA alias pakai setelan *Golden Safe Default* ini saja** (karena sudah bagus untuk 95% soal):

1. **Untuk Klasifikasi Teks Apa Pun:**
   - Biarkan `LOSS_FUNCTION = 'auto'` (atau `'combo_ce_f1'`)
   - Biarkan `CHECKPOINT_MONITOR = 'hybrid'`
   - Cukup pastikan `EVAL_METRIC` sama dengan kata yang tertulis di soal (kalau di soal tidak ditulis jelas, biarkan `'macro_f1'`).

2. **Untuk Regresi Skor / Esai Apa Pun:**
   - Biarkan `LOSS_FUNCTION = 'smooth_l1'` (paling tahan data rusak/*outlier*, tidak akan pernah meledak jadi `NaN`)
   - Biarkan `CHECKPOINT_MONITOR = 'hybrid'`
   - Untuk `EVAL_METRIC`: kalau targetnya angka bulat ($1, 2, 3, 4, 5$) pakai `'qwk'`; kalau targetnya angka desimal ($1.25, 3.40$) pakai `'rmse'`.


