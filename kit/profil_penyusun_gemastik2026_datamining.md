# Profil Penyusun GEMASTIK 2026 dan Implikasinya untuk Final Penambangan Data

Disusun 16 Oktober 2026. Hanya memuat informasi profesional yang dipublikasikan sendiri oleh institusi atau penulis (halaman dosen, SINTA, Google Scholar, dokumen resmi GEMASTIK).

---

## 1. Tujuan

1. Mengetahui siapa yang menyusun ketentuan Divisi III Penambangan Data GEMASTIK 2026 dan bagaimana profil riset mereka.
2. Menarik implikasi yang masuk akal untuk jenis kasus final, tanpa mengklaim lebih dari yang bisa dibuktikan.
3. Memperbarui rencana persiapan berdasarkan Panduan GEMASTIK 2026 versi final yang berhasil saya baca langsung (109 halaman).

---

## 2. Temuan

### 2.1 Batas bukti (baca ini dulu)

| Pertanyaan | Status |
|---|---|
| Siapa penyusun **pedoman** GEMASTIK 2026? | Terkonfirmasi, 14 nama (daftar di 2.2) |
| Siapa penyusun **soal final** Penambangan Data? | **Tidak disebut di dokumen mana pun yang saya temukan** |
| Siapa **dewan juri** Penambangan Data 2026? | **Tidak dipublikasikan** di panduan maupun situs resmi sampai hari ini |

"Tim Penyusun" di sampul panduan adalah penyusun dokumen aturan lintas 11 divisi, bukan otomatis pembuat dataset final. Seluruh bagian 2.4 sampai 2.6 adalah inferensi dari pola, bukan fakta tentang soal.

### 2.2 Tim Penyusun Pedoman 2026 (sampul resmi)

Heroe Wijanto, Wahyono, Muchammad Husni, Masayu Leylia Khodra, Dana Sulistiyo Kusumo, Basnendar Herry Prilosadoso, Asep Sholahuddin, Romi Satria Wahono, Dyah Anggraini, Lukito Edi Nugroho, Andry Chowanda, Bob Hardian, Sukino, Yulita Priyoningsih.

Perubahan dibanding 2025:
- Masuk: Dyah Anggraini, Bob Hardian, Yulita Priyoningsih.
- Keluar: Daniel Oranova Siahaan, Karyana Hutomo, Charoline Dewi Virasari.

Tiga nama paling stabil: **Masayu Leylia Khodra, Andry Chowanda, Dana Sulistiyo Kusumo**. Ketiganya tercantum sebagai tim penyusun pada pedoman 2022 dan 2023, dan tetap ada di 2024, 2025, 2026.

### 2.3 Profil nama yang relevan untuk divisi data mining

| Nama | Institusi | Fokus riset yang terverifikasi | Relevansi ke final data mining |
|---|---|---|---|
| **Masayu Leylia Khodra** | ITB (STEI), dosen sejak 2008, S1-S3 ITB | Klasifikasi teks, peringkasan teks, analisis sentimen berbasis aspek, NLP Bahasa Indonesia. Proyek: AMR Bahasa Indonesia, graph-to-text summarization dengan GNN (PPMI STEI ITB 2026), ekstraksi informasi dokumen regulasi untuk asesmen SPBE. Ko-penulis IndoSBERT (2023), deteksi kematangan buah dengan VGG16, representasi video self-supervised (2025) | Teks Indonesia kuat, ada jejak citra dan video |
| **Andry Chowanda** | BINUS (School of Computer Science) | Emotion recognition teks Indonesia (IndoBERT + BiLSTM, 7.629 tweet), hate speech (IndoBERTweet + BiLSTM), intent classification, ko-penulis PRDECT-ID (5.400 ulasan Tokopedia berlabel emosi), ko-penulis dataset authorship attribution teks pendek Indonesia (Data in Brief, Mei 2026) | **Pembuat dataset teks Indonesia publik**, dan BINUS adalah tuan rumah final 2026 |
| **Wahyono** | UGM (Ilmu Komputer dan Elektronika) | Computer vision, image dan video processing, intelligent surveillance. Pengalaman proyek di Korea dan inspeksi las visual dengan Toyota. Kepala lab Algorithm and Computation (menurut materinya) | Satu-satunya nama dengan rekam jejak **video** yang jelas |
| **Romi Satria Wahono** | Universitas Dian Nuswantoro, doktor UTeM 2015 | Software defect prediction, **class imbalance dan atribut noisy**, seleksi fitur metaheuristik + bagging | Data tabular tidak seimbang |
| **Dana Sulistiyo Kusumo** | Telkom University | Afiliasi terkonfirmasi. Fokus riset spesifiknya **tidak berhasil saya verifikasi** | Tidak bisa disimpulkan |

Nama lain (Heroe Wijanto, Muchammad Husni, Lukito Edi Nugroho, Asep Sholahuddin, Basnendar Herry Prilosadoso, Bob Hardian, Dyah Anggraini, Sukino, Yulita Priyoningsih) tidak saya profilkan karena tidak ada bukti keterkaitan dengan divisi data mining. Basnendar berasal dari institut seni, jadi hampir pasti lebih terkait divisi desain dan animasi.

### 2.4 Perubahan ketentuan yang paling penting (Panduan 2026 vs 2024)

| Aspek | 2024 | 2026 |
|---|---|---|
| Jenis data yang disebut | transaksi, gambar, teks/bahasa | **data terstruktur, gambar, video, dan teks** |
| Tema technical report | Pelayanan publik | Solusi yang berdampak, inklusif, berkelanjutan |
| Generative AI | tidak disebut | Boleh dipakai saat penyisihan, tetapi technical report tidak boleh hasil Generative AI |
| Alat bantu di final | tools, library, framework | tools, library, framework (**Generative AI tidak disebut untuk final**) |

Jadi kata **video** adalah tambahan baru. Ini sinyal lemah, bukan bukti bahwa soal final berupa video. Pedoman memang menulis cakupan umum, dan final 2023 yang terkonfirmasi berupa tabular log jaringan.

### 2.5 Jadwal dan format final (resmi)

| Waktu | Kegiatan |
|---|---|
| 26-30 Okt 2026 | Daftar ulang final + unggah dokumen (makalah, similaritas, video profil 60 detik sesuai persyaratan umum) |
| 10 Nov, 12.00-18.00 | Set-up workstation di perangkat panitia (6 jam) |
| 11 Nov, 13.00-18.00 | Penyelesaian kasus (5 jam) |
| 12 Nov, 08.00-17.00 | Presentasi solusi, 20 tim |
| 13 Nov malam | Penutupan dan pengumuman juara |

Bobot final: nilai penyisihan 20%, skor leaderboard 30%, ketepatan metode 25%, kejelasan dan kelengkapan presentasi 25%. Deliverables berupa zip: dokumentasi model, source code, PPT.

### 2.6 Pola historis kasus final (bukti terbatas)

| Tahun | Kasus | Sumber |
|---|---|---|
| 2021 | Korpus teks Bahasa Indonesia, bekerja sama dengan INACL | Pedoman 2021 |
| 2023 | Klasifikasi jenis serangan dari log jaringan, kelas minoritas langka | Berita ITB |
| 2022, 2024, 2025 | Tidak terpublikasi di sumber yang saya temukan | - |

Dua data point dari enam tahun. Jangan menarik pola kuat darinya.

### 2.7 Kesimpulan temuan

1. Penyusun pedoman didominasi akademisi NLP Indonesia (Khodra, Chowanda), dengan satu nama computer vision/video (Wahyono) dan satu nama imbalanced tabular (Wahono).
2. Probabilitas teks Indonesia naik karena dua alasan terpisah: komposisi penyusun dan BINUS sebagai tuan rumah. Tetapi keduanya tidak menjamin apa pun tentang soal.
3. Modalitas **video** layak mendapat persiapan minimum karena baru muncul di 2026, meski bobot kemungkinannya saya taksir rendah.
4. Final 5 jam di perangkat panitia membuat pipeline video penuh (misal 3D CNN) tidak realistis. Yang realistis hanya frame sampling + embedding pretrained + model ringan.

---

## 3. Rekomendasi

### 3.1 Tiga opsi penyesuaian persiapan

| Opsi | Isi | Kelebihan | Kekurangan |
|---|---|---|---|
| **A (disarankan)** | Pertahankan bobot latihan 40% NLP, 40% tabular dan time series, 20% CV, lalu tambahkan **modul video ringan** di dalam kit CV (sampling frame, embedding timm, klasifier ringan) | Biaya kecil (sekitar 1 hari), menutup celah modalitas baru | Video tidak dilatih mendalam |
| B | Geser ke 35% NLP, 35% tabular, 30% CV/video | Lebih siap kalau video benar-benar keluar | Mengorbankan latihan tabular yang preseden 2023-nya paling kuat |
| C | Tidak mengubah apa pun, hanya bertanya ke panitia saat technical meeting | Nol biaya | Kalau video keluar, tim mulai dari nol di mesin panitia |

Alasan memilih A: bukti untuk video sangat tipis, jadi menggeser bobot besar (opsi B) tidak dibenarkan, tetapi biaya asuransinya murah.

### 3.2 Hal yang perlu dipastikan lewat jalur resmi panitia (bukan ke penyusun atau juri)

1. Apakah perangkat panitia punya GPU dan berapa spesifikasinya?
2. Apakah ada akses internet di perangkat panitia, atau hanya instalasi dari media yang kita bawa?
3. Apakah boleh membawa bundle wheel dan bobot model lewat flashdisk?
4. Apakah Generative AI dan asisten kode boleh dipakai saat final? Panduan 2026 hanya menyebut "tools, library, atau framework" untuk final.
5. Berapa batas submit ke leaderboard per hari dan apa metriknya?

Jangan menghubungi nama-nama di 2.2 untuk menanyakan soal. Itu berisiko dianggap upaya mendapat informasi tidak sah, dan tidak diperlukan karena pertanyaan di atas cukup lewat panitia.

### 3.3 Yang saya anggap salah kaprah

1. Menganggap "penyusun pedoman" sama dengan "pembuat soal". Tidak ada bukti.
2. Menaikkan peluang satu modalitas jauh di atas yang lain hanya karena riset satu dosen.
3. Menganggap Generative AI otomatis boleh di final karena boleh di penyisihan.

---

## 4. Langkah Eksekusi

| Tanggal | Aksi |
|---|---|
| 16-18 Okt | Tambahkan modul video-frame-embedding ke `kit_cv_timm` dan uji di data video kecil |
| 16-18 Okt | Siapkan 5 pertanyaan di 3.2, kirim lewat kanal resmi (email ketua tim atau kontak panitia di laman GEMASTIK) |
| 19-25 Okt | Simulasi 1 dengan dataset teks Indonesia (misal PRDECT-ID, emosi Twitter) dalam 5 jam dan wajib selesai dengan deck |
| 26-30 Okt | Daftar ulang dan unggah dokumen final (batas 30 Okt) |
| 31 Okt-5 Nov | Simulasi 2 dengan modalitas berbeda (tabular imbalanced atau video ringan) |
| 6-9 Nov | Cek kit di laptop bersih tanpa internet, siapkan template deck, latihan tanya jawab metode |
| 10 Nov | Set-up workstation 12.00-18.00: pasang bundle, jalankan `verify_env`, jalankan satu notebook demo |

---

## 5. Acceptance Gate

1. Lima pertanyaan di 3.2 sudah terkirim sebelum 20 Oktober dan jawaban dicatat.
2. Modul video menghasilkan submission valid dari data video uji dalam kurang dari 30 menit.
3. Pada simulasi 1 dan 2, submission pertama keluar kurang dari 60 menit dan deck selesai sebelum waktu habis.
4. Seluruh anggota bisa menjelaskan pilihan split, metrik, dan alasan metode tanpa catatan.
5. Dokumen daftar ulang terunggah sebelum 30 Oktober 2026.

---

## 6. Output yang Diharapkan

1. Modul video ringan di kit CV (dinamai sesuai metode: `kit_cv_timm_video_frames.ipynb`).
2. Catatan jawaban panitia atas lima pertanyaan teknis.
3. Log dua simulasi: waktu ke baseline, skor, gap CV vs holdout.
4. Template PPT final yang sudah dipakai dua kali.

---

## Sumber

- Panduan GEMASTIK 2026 (Final), tautan resmi dari lldikti16 dan Dikti: https://lldikti16.kemdiktisaintek.go.id/2026/07/penawaran-gemastik-2026/
- Jadwal Gemastik 19: https://gemastik.kemdiktisaintek.go.id/landing/jadwal
- Panduan GEMASTIK 2025 (daftar penyusun): https://kemdiktisaintek.go.id/wp-content/uploads/2025/07/Panduan-GEMASTIK-2025.pdf
- Pedoman 2024: https://dit-mawa.upi.edu/wp-content/uploads/2024/01/Pedoman-GEMASTIK-2024.pdf
- Pedoman 2023 (penyusun): https://pusatprestasinasional.kemdikbud.go.id/uploads/lampiran/Pedoman%20GEMASTIK%202023%20v.pdf
- Pedoman 2022 (penyusun): https://tk.ilkom.unsri.ac.id/wp-content/uploads/2022/09/Pedoman-GEMASTIK-2022.pdf
- Pedoman 2021 (data teks Indonesia, INACL): https://docplayer.info/209075642-Pedoman-kegiatan-pagelaran-mahasiswa-nasional-bidang-tik-tahun-2021-tema-tik-untuk-indonesia-bangkit-menuju-kehidupan-bangsa-yang-lebih-baik.html
- Kasus final 2023: https://www.itb.ac.id/berita/tiga-mahasiswa-teknik-informatika-itb-raih-juara-1-gemastik-2023-divisi-data-mining/60055
- Masayu Leylia Khodra: https://itb.ac.id/staf/profil/masayu-leylia-khodra, https://sinta.kemdiktisaintek.go.id/authors/profile/6009714/?view=researches, https://scholar.google.com/citations?user=EkyJKakAAAAJ&hl=en
- Andry Chowanda: https://mti.binus.ac.id/?p=3514, https://data.mendeley.com/datasets/574v66hf2v/1, https://scik.org/index.php/cmbn/article/view/7858, https://pmc.ncbi.nlm.nih.gov/articles/PMC13226845/
- Wahyono: https://sistemcerdas.mipa.ugm.ac.id/wp-content/uploads/sites/1297/2020/07/Materi-Wahyono-S.Kom_.-Ph.D.-Vision-Based-Intelligent-System.pdf
- Romi Satria Wahono: https://dinus.academia.edu/RomiSatriaWahono, https://eprints.utem.edu.my/view/creators/Wahono=3ARomi_Satria=3A=3A.html
- Daftar penyusun dan afiliasi (slide sosialisasi 2024): https://www.slideshare.net/slideshow/materi-sosialisasi-ketentuan-umum-gemastik-2024-605157-pptx/272598498
