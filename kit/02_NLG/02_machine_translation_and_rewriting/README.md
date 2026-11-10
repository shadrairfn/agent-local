# 🌐 02_machine_translation_and_rewriting: Universal Translation, OCR Restoration & Prompt Recovery Kit

Welcome to the **Machine Translation, Text Normalization & Prompt Recovery Kit**, engineered for bilingual/regional language translation, OCR/noisy text restoration, and Kaggle LLM Prompt Recovery.

---

## 📂 Directory Architecture (Clean Output Standard)

```
02_machine_translation_and_rewriting/
├── demo_data/                                  # 📊 Parallel source-target text pairs
├── eda/                                        # 🎯 Task-Specific EDA (eda_translation_rewriting.ipynb + README.md)
├── notebooks/                                  # 🚀 Clean-Code Translation & Recovery Pipelines (Pre-executed with stored outputs)
│   ├── 01_seq2seq_translation_rewriting.ipynb  # Tier 1: Seq2Seq Translator + Bilingual Lexicon/Phrase Memory (.pth export)
│   ├── 02_prompt_recovery_cosine_eval.ipynb    # Tier 2: Semantic Retrieval & Sharpened Cosine Similarity (max(0, cos)^3)
│   ├── 📁 models/                              # 💾 Trained Checkpoints (.pth) & JSON Results
│   │   ├── seq2seq_translator.pth
│   │   ├── 01_translation_results.json
│   │   └── 02_cosine_recovery_results.json
│   └── submission_*.csv                        # 4 verified submission files (0 NaNs)
└── modular_variations/                         # 🧩 Bilingual length filter, Label smoothing, Fallback cleaner
```
