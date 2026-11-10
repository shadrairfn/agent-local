# 📝 01_text_summarization: Universal Extractive & Abstractive Summarization Kit

Welcome to the **Text Summarization Kit**, engineered for both instant CPU extractive summarization (TextRank + Lead-K) and neural Seq2Seq abstractive generation.

---

## 📂 Directory Architecture (Clean Output Standard)

```
01_text_summarization/
├── demo_data/                                  # 📊 Document-Summary pairs
├── eda/                                        # 🎯 Task-Specific EDA (eda_summarization.ipynb + README.md)
├── notebooks/                                  # 🚀 Clean-Code Summarization Pipelines (Pre-executed with stored outputs)
│   ├── 01_extractive_textrank_summarization.ipynb # Tier 1: Instant CPU Graph TextRank + Lead-K + TF-IDF Centrality
│   ├── 02_seq2seq_abstractive_summarization.ipynb # Tier 2: Neural Encoder-Decoder Seq2Seq Summarizer (.pth export)
│   ├── 📁 models/                              # 💾 Trained Checkpoints (.pth) & JSON Results
│   │   ├── seq2seq_summarizer.pth
│   │   ├── 01_textrank_results.json
│   │   └── 02_seq2seq_results.json
│   └── submission_*.csv                        # 4 verified submission files (0 NaNs)
└── modular_variations/                         # 🧩 Lead-3 extractor, Beam search tuner, Repetition cleaner
```
