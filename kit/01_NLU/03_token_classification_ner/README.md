# 🏷️ 03_token_classification_ner: Universal Sequence Labeling & NER Kit

Welcome to the **Named Entity Recognition (NER) & Token Classification Kit**, designed for sequence labeling, PII detection, Aspect-Based Sentiment Analysis (ABSA), and span extraction.

---

## 📂 Directory Architecture (Clean Output Standard)

```
03_token_classification_ner/
├── demo_data/                                  # 📊 Token sequence dataset with BIO tags
├── eda/                                        # 🎯 Task-Specific EDA (eda_ner_tokens.ipynb + README.md)
├── notebooks/                                  # 🚀 3-Tier Clean-Code Universal Pipelines (Pre-executed with stored outputs)
│   ├── 01_dl_bilstm_crf_ner.ipynb              # Tier 1: Classical BiLSTM-CRF (Viterbi transition matrix, zero-OOM)
│   ├── 02_transformer_token_classification.ipynb # Tier 2: Contextual Transformer Token Classifier + Subword Alignment
│   ├── 03_ensemble_ner_voting.ipynb            # Tier 3: Token probability blending + Legal BIO sequence repair
│   ├── 📁 models/                              # 💾 Trained Checkpoints (.pth) & JSON Results
│   │   ├── bilstm_crf_fold0..2.pth
│   │   ├── transformer_ner_fold0..2.pth
│   │   ├── 01_bilstm_crf_results.json
│   │   ├── 02_transformer_ner_results.json
│   │   └── 03_ensemble_ner_results.json
│   └── submission_*.csv                        # 6 verified token-level submission files (0 NaNs)
└── modular_variations/                         # 🧩 Subword BIO alignment, Masked token loss, BIO Sequence repair
```
