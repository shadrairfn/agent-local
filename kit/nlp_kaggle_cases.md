# Katalog Kasus NLP di Kaggle

Dokumen ini merekonstruksi daftar kompetisi/dataset Kaggle untuk NLP ke dalam taksonomi yang lebih terstruktur: **NLU**, **NLG**, dan **NLP lainnya**.

**Legenda**
- ✅ = sudah ada di daftar awal
- ➕ = tambahan baru untuk mengisi gap
- 📦 = dataset (bukan kompetisi), cocok untuk latihan/baseline

> Catatan: slug URL Kaggle bisa berubah atau kompetisi ditutup. Verifikasi tautan sebelum dipakai, terutama yang bertanda ➕ dan 📦.

---

## 0. Peta Taksonomi

```
NLP
├── NLU (Natural Language Understanding)
│   ├── A. Text Classification
│   │   ├── A1. Binary
│   │   ├── A2. Multiclass
│   │   ├── A3. Multi-label
│   │   └── A4. Sentence-pair / NLI
│   ├── B. Text Regression / Ranking
│   ├── C. Token Classification (NER, span extraction, segmentasi)
│   ├── D. Question Answering & Retrieval
│   └── E. Coreference / Relation (terbatas di Kaggle)
├── NLG (Natural Language Generation)
│   ├── F. Summarization
│   ├── G. Machine Translation
│   ├── H. Text Rewriting / Prompt Recovery
│   └── I. Open-ended generation & reasoning
└── J. Dialogue, Preference & Evaluasi LLM
```

---

# BAGIAN I — NLU

## A. Text Classification

### A1. Binary Classification

| # | Kasus | Target | Metrik | Status | Link |
|---|-------|--------|--------|--------|------|
| 1 | NLP with Disaster Tweets (pemanasan) | Tweet bencana / bukan | F1 | ✅ | https://www.kaggle.com/competitions/nlp-getting-started |
| 2 | Jigsaw Unintended Bias in Toxicity Classification | Toksik / tidak + bias subgrup | Bias-weighted AUC | ✅ | https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification |
| 3 | Jigsaw Multilingual Toxic Comment Classification | Toksik / tidak, multibahasa | ROC-AUC | ✅ | https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification |
| 4 | LLM - Detect AI Generated Text | Ditulis manusia / AI | ROC-AUC | ✅ | https://www.kaggle.com/competitions/llm-detect-ai-generated-text |
| 5 | Quora Insincere Questions Classification | Pertanyaan tulus / tidak | F1 | ➕ | https://www.kaggle.com/competitions/quora-insincere-questions-classification |
| 6 | 📦 IMDB 50K Movie Reviews | Sentimen positif / negatif | Accuracy / F1 | 📦 | https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews |
| 7 | 📦 Sentiment140 (1.6 juta tweet) | Sentimen tweet | Accuracy / F1 | 📦 | https://www.kaggle.com/datasets/kazanova/sentiment140 |

### A2. Multiclass Classification

| # | Kasus | Target | Metrik | Status | Link |
|---|-------|--------|--------|--------|------|
| 1 | Feedback Prize - Predicting Effective Arguments | 3 kelas (Ineffective / Adequate / Effective) | Log loss | ✅ | https://www.kaggle.com/competitions/feedback-prize-effective-arguments |
| 2 | MAP - Charting Student Math Misunderstandings | 65 kelas | MAP@3 | ✅ | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings |
| 3 | Spooky Author Identification | 3 penulis | Multiclass log loss | ➕ | https://www.kaggle.com/competitions/spooky-author-identification |
| 4 | Sentiment Analysis on Movie Reviews | 5 level sentimen | Accuracy | ➕ | https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews |
| 5 | 📦 News Category Dataset (HuffPost) | ±40 kategori berita | Accuracy / macro F1 | 📦 | https://www.kaggle.com/datasets/rmisra/news-category-dataset |

### A3. Multi-label Classification

| # | Kasus | Target | Metrik | Status | Link |
|---|-------|--------|--------|--------|------|
| 1 | Jigsaw Toxic Comment Classification | 6 label toksisitas | Mean column-wise ROC-AUC | ✅ | https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge |

### A4. Sentence-Pair / Natural Language Inference (NLI)

| # | Kasus | Target | Metrik | Status | Link |
|---|-------|--------|--------|--------|------|
| 1 | Contradictory, My Dear Watson | Entailment / neutral / contradiction (multibahasa) | Accuracy | ✅ | https://www.kaggle.com/competitions/contradictory-my-dear-watson |
| 2 | Quora Question Pairs | Duplikat / bukan | Log loss | ➕ | https://www.kaggle.com/competitions/quora-question-pairs |

---

## B. Text Regression / Ranking

| # | Kasus | Target | Metrik | Status | Link |
|---|-------|--------|--------|--------|------|
| 1 | CommonLit Readability Prize | 1 skor keterbacaan | RMSE | ✅ | https://www.kaggle.com/competitions/commonlitreadabilityprize |
| 2 | Feedback Prize - English Language Learning | 6 skor kemampuan menulis | MCRMSE | ✅ | https://www.kaggle.com/competitions/feedback-prize-english-language-learning |
| 3 | CommonLit - Evaluate Student Summaries | 2 skor (content, wording) | MCRMSE | ✅ | https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries |
| 4 | Google QUEST Q&A Labeling | 30 target kualitas Q&A | Spearman | ✅ | https://www.kaggle.com/competitions/google-quest-challenge |
| 5 | U.S. Patent Phrase to Phrase Matching | Skor kemiripan semantik | Pearson | ✅ | https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching |
| 6 | Learning Agency Lab - Automated Essay Scoring 2.0 | Skor esai 1–6 | Quadratic Weighted Kappa | ✅ | https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2 |
| 7 | Jigsaw Rate Severity of Toxic Comments | Ranking tingkat toksisitas | Agreement dengan pairwise ranking | ➕ | https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating |
| 8 | Mercari Price Suggestion Challenge | Harga dari teks + fitur | RMSLE | ➕ | https://www.kaggle.com/competitions/mercari-price-suggestion-challenge |

---

## C. Token Classification (NER, Span Extraction, Segmentasi)

| # | Kasus | Subtugas | Metrik | Status | Link |
|---|-------|----------|--------|--------|------|
| 1 | The Learning Agency Lab - PII Data Detection | NER (deteksi PII) | Micro F-beta (β=5) | ✅ | https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data |
| 2 | NBME - Score Clinical Patient Notes | Span extraction | Micro F1 | ✅ | https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes |
| 3 | Tweet Sentiment Extraction | Span extraction | Jaccard | ✅ | https://www.kaggle.com/competitions/tweet-sentiment-extraction |
| 4 | Feedback Prize - Evaluating Student Writing | Segmentasi + 7 tipe diskursus | Macro F1 berbasis overlap | ✅ | https://www.kaggle.com/competitions/feedback-prize-2021 |
| 5 | Coleridge Initiative - Show US the Data | Deteksi dataset mention | Jaccard-based F-score | ✅ | https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data |
| 6 | Make Data Count - Finding Data References | Sitasi dataset | F1 | ✅ | https://www.kaggle.com/competitions/make-data-count-finding-data-references |
| 7 | Text Normalization Challenge (English) | Token-level normalisasi (mendekati seq2seq) | Accuracy | ➕ | https://www.kaggle.com/competitions/text-normalization-challenge-english-language |

---

## D. Question Answering & Retrieval

| # | Kasus | Subtugas | Metrik | Status | Link |
|---|-------|----------|--------|--------|------|
| 1 | TensorFlow 2.0 Question Answering | Extractive QA (long + short answer) | Micro F1 | ➕ | https://www.kaggle.com/competitions/tensorflow2-question-answering |
| 2 | chaii - Hindi and Tamil Question Answering | Extractive QA multibahasa | Jaccard | ➕ | https://www.kaggle.com/competitions/chaii-hindi-and-tamil-question-answering |
| 3 | Kaggle - LLM Science Exam | Multiple-choice QA (dengan retrieval) | MAP@3 | ➕ | https://www.kaggle.com/competitions/kaggle-llm-science-exam |
| 4 | Eedi - Mining Misconceptions in Mathematics | Retrieval / ranking miskonsepsi | MAP@25 | ➕ | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics |
| 5 | 📦 Stanford Question Answering Dataset (SQuAD) | Extractive QA | EM / F1 | 📦 | https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset |

## E. Coreference & Relation Extraction (terbatas di Kaggle)

| # | Kasus | Subtugas | Metrik | Status | Link |
|---|-------|----------|--------|--------|------|
| 1 | Gendered Pronoun Resolution | Coreference (resolusi pronomina) | Multiclass log loss | ➕ | https://www.kaggle.com/competitions/gendered-pronoun-resolution |
| 2 | Relation extraction, SRL, parsing | Tidak ada kompetisi Kaggle utama | – | ❌ | Gunakan dataset eksternal (TACRED, CoNLL, Universal Dependencies) |

---

# BAGIAN II — NLG

> Kaggle relatif sedikit memiliki kompetisi NLG murni karena evaluasi teks generatif sulit diotomatisasi. Kebanyakan kasus di bawah memakai metrik berbasis similarity atau evaluasi LLM.

## F. Summarization

| # | Kasus | Subtugas | Metrik umum | Status | Link |
|---|-------|----------|-------------|--------|------|
| 1 | 📦 CNN / DailyMail Summarization | Abstractive / extractive summarization | ROUGE | 📦 | https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail |
| 2 | CommonLit - Evaluate Student Summaries | Evaluasi ringkasan (bukan generasi, tapi relevan) | MCRMSE | ✅ | https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries |

## G. Machine Translation

| # | Kasus | Subtugas | Metrik umum | Status | Link |
|---|-------|----------|-------------|--------|------|
| 1 | Google - Unlock Global Communication with Gemma | Terjemahan / tugas bahasa dengan Gemma | Berbasis task | ➕ | https://www.kaggle.com/competitions/gemma-language-tasks |
| 2 | Dataset terjemahan | Cari di Kaggle Datasets dengan kata kunci `translation`, `parallel corpus`, `en-id` | BLEU / chrF / COMET | 📦 | https://www.kaggle.com/datasets |

## H. Text Rewriting / Prompt Recovery

| # | Kasus | Subtugas | Metrik | Status | Link |
|---|-------|----------|--------|--------|------|
| 1 | LLM Prompt Recovery | Tebak prompt yang mengubah teks asli menjadi teks hasil rewrite | Sharpened cosine similarity | ➕ | https://www.kaggle.com/competitions/llm-prompt-recovery |

## I. Open-ended Generation & Reasoning

| # | Kasus | Subtugas | Metrik | Status | Link |
|---|-------|----------|--------|--------|------|
| 1 | AI Mathematical Olympiad - Progress Prize | Penalaran matematis, jawaban berbentuk teks/angka | Akurasi jawaban | ➕ | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize |
| 2 | LLM - Detect AI Generated Text | Sisi deteksi dari NLG (analisis keluaran generatif) | ROC-AUC | ✅ | https://www.kaggle.com/competitions/llm-detect-ai-generated-text |

---

# BAGIAN III — Dialogue, Preference & Evaluasi LLM

| # | Kasus | Subtugas | Metrik | Status | Link |
|---|-------|----------|--------|--------|------|
| 1 | LMSYS - Chatbot Arena Human Preference Predictions | Memprediksi respons LLM mana yang dipilih manusia | Log loss | ➕ | https://www.kaggle.com/competitions/lmsys-chatbot-arena |
| 2 | WSDM Cup - Multilingual Chatbot Arena | Preferensi manusia, multibahasa | Accuracy | ➕ | https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena |
| 3 | LLM 20 Questions | Dialog tanya-jawab antar agen LLM | Berbasis game | ➕ | https://www.kaggle.com/competitions/llm-20-questions |

---

# Ringkasan Cakupan

| Kategori | Jumlah kasus (✅ awal) | Jumlah (➕ tambahan) | Catatan |
|----------|-----------------------|---------------------|---------|
| Binary classification | 4 | 1 (+2 📦) | Kuat |
| Multiclass classification | 2 | 2 (+1 📦) | Ditambah authorship dan sentimen 5 kelas |
| Multi-label | 1 | 0 | Satu kasus klasik cukup |
| NLI / sentence-pair | 1 | 1 | |
| Text regression / ranking | 6 | 2 | |
| Token classification | 6 | 1 | |
| Question answering & retrieval | 0 | 4 (+1 📦) | Gap utama terisi |
| Coreference / relation | 0 | 1 | Relation/SRL/parsing tetap di luar Kaggle |
| Summarization | 0 (1 terkait) | 0 (+1 📦) | Hanya dataset |
| Machine translation | 0 | 1 (+📦) | Terbatas |
| Rewriting / generasi | 0 | 2 | |
| Dialogue / preferensi LLM | 0 | 3 | |

## Gap yang Tetap Ada

Bagian berikut sulit ditemukan sebagai kompetisi Kaggle dan biasanya perlu dataset eksternal:

- **Dependency / constituency parsing** → Universal Dependencies
- **Semantic Role Labeling** → CoNLL-2005 / OntoNotes
- **Relation extraction** → TACRED, DocRED
- **Language modeling / perplexity** → WikiText, OpenWebText
- **Data-to-text, question generation** → WebNLG, E2E NLG, SQuAD (dibalik jadi generasi)
- **Bahasa Indonesia** → cari di Kaggle Datasets dengan kata kunci `indonesian`, `bahasa indonesia`, `IndoNLU`, atau gunakan HuggingFace (IndoNLU, NusaX)

---

# Rekomendasi Jalur Belajar

1. **Pemanasan**: Disaster Tweets → IMDB 50K → Spooky Author
2. **Classification lanjutan**: Jigsaw Toxic (multi-label) → Feedback Prize Effective Arguments → Watson NLI
3. **Regression**: CommonLit Readability → Feedback Prize ELL → AES 2.0
4. **Token-level**: Tweet Sentiment Extraction → NBME → PII Detection
5. **QA & retrieval**: TF2.0 QA → LLM Science Exam → Eedi
6. **NLG & LLM**: LLM Prompt Recovery → LMSYS Chatbot Arena → AI Mathematical Olympiad
