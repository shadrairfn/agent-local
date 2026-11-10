# 🎯 01_text_classification: Universal NLP Classification Kit

Welcome to the **Text Classification Kit**, engineered specifically for high-speed, medal-contending performance in data science competitions (Kaggle, GEMASTIK XIX 2026, SATRIA DATA).

---

## 📂 Directory Architecture

```
01_text_classification/
├── demo_data/                                  # 📊 Verified Demo Dataset (train.csv, test.csv, sample_submission.csv)
├── eda/                                        # 🎯 Task-Specific EDA
│   ├── eda_classification.ipynb               # Executed notebook: class balance, imbalance ratio, text length per class
│   └── README.md                               # Decision Matrix for classification modeling
├── notebooks/                                  # 🚀 5-Tier Universal Clean-Code Pipelines (Pre-executed with stored outputs)
│   ├── 01_ml_sparse_dense_gbdt.ipynb           # Tier 1: Dual TF-IDF + FastText Dense + LSA + Surface stats -> LightGBM
│   ├── 02_dl_sequential_bilstm_cnn.ipynb       # Tier 2: Subword/Word Embeddings + BiLSTM + 1D-TextCNN + Attention Head
│   ├── 03_transformer_finetune_end2end.ipynb   # Tier 3: End-to-End Contextual Transformer (IndoBERT/DeBERTa) + MSD + FP16
│   ├── 04_transformer_feature_extractor_gbdt.ipynb # Tier 4: Frozen Transformer Embeddings (Mean/CLS) + Surface Stats -> GBDT
│   ├── 05_universal_ensemble_blending.ipynb    # Tier 5: 4-Way Probability Blending (Nelder-Mead) + Macro F1 Threshold Tuning
│   └── *.csv / *.npy                            # Pre-computed OOF, test probabilities & 10 verified submission files
├── modular_variations/                         # 🧩 Sharpened Lego Bricks (Each with code .py + explanatory README.md)
│   ├── preprocessing/
│   │   ├── indonesian_slang_cleaning/          # Slang dictionary, repetition reduction, token normalization
│   │   └── emoji_sentiment_handling/           # Converts emoticons & unicode emojis into semantic tokens
│   ├── feature_engineering/
│   │   ├── tfidf_ngram_lexicons/               # Dual word (1,2) & char (3,5) TF-IDF extraction
│   │   └── lsa_topic_features/                 # TruncatedSVD dense semantic topics for GBDTs
│   ├── model_configs/
│   │   ├── focal_loss_imbalance/               # PyTorch Focal Loss with gamma=2.0 and alpha class weights
│   │   └── multi_sample_dropout/               # Multi-Sample Dropout head (5 dropout paths)
│   ├── error_analysis/
│   │   └── confusion_matrix_drilldown/         # Extracts high-confidence false positives/negatives
│   └── post_processing/
│       ├── probability_threshold_tuning/       # Per-class probability scaling for Macro F1 maximization
│       └── temperature_scaling/                # Logit calibration (T > 0) before ensembling
└── smart_loader.py                             # 🔄 Universal column mapper and data loader
```

---

## ⚡ 5-Tier Representational Spectrum (Why This Wins Competitions)

Different model families learn fundamentally orthogonal text representations. Blending them minimizes individual model bias and produces leaderboard-topping ensemble stability:

| Tier | Paradigm | Representation | Primary Architecture | Fast-Path Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Sparse + Dense Classical ML** | Word/Char N-Grams + FastText Embeddings + LSA Topics + Surface Lengths | LightGBM / CatBoost | ~15 seconds |
| **Tier 2** | **Sequential Deep Learning** | Word/Subword Sequences $\to$ Dense Vectors | BiLSTM + 1D-TextCNN + Multi-Head Attention | ~30 seconds |
| **Tier 3** | **End-to-End Transformers** | Contextual Attention $\to$ Multi-Sample Dropout Classification Head | IndoBERT / ModernBERT / DeBERTa (FP16 AMP) | ~45 seconds |
| **Tier 4** | **Hybrid Transformer Feature Extractor** | Frozen Transformer Hidden States + Tabular Surface Features | LightGBM / Ridge / MLP | ~20 seconds |
| **Tier 5** | **Universal Ensemble & Post-Processing** | 4-Way Calibrated Probability Fusion | Nelder-Mead Optimization + Macro F1 Threshold Search | ~10 seconds |

---

## 🛠️ Step-by-Step Competition Speedrun

1. **Step 1: Run Tier 1 (`01_ml_sparse_dense_gbdt.ipynb`)**
   - Point `TRAIN_CSV_PATH` and `TEST_CSV_PATH` to your data in the single-cell **Master User Config**.
   - Execute to generate your immediate Day-1 valid submission (`submission_ml_raw.csv` and `submission_ml_postprocessed.csv`) and baseline OOF predictions.
2. **Step 2: Run Tier 2 (`02_dl_sequential_bilstm_cnn.ipynb`)**
   - Train the sequential BiLSTM + TextCNN pipeline to capture n-gram phrases and long-range sequential token dynamics.
   - Saves `dl_bilstm_oof_predictions.npy` and `submission_bilstm_postprocessed.csv`.
3. **Step 3: Run Tier 3 (`03_transformer_finetune_end2end.ipynb`)**
   - Fine-tune deep contextual transformer with FP16 mixed precision and Multi-Sample Dropout (MSD).
   - Saves `transformer_oof_predictions.npy` and `submission_transformer_postprocessed.csv`.
4. **Step 4: Run Tier 4 (`04_transformer_feature_extractor_gbdt.ipynb`)**
   - Extract frozen contextual embeddings (mean pooling across layers) concatenated with surface statistics and train LightGBM.
   - Saves `trans_gbdt_oof_predictions.npy` and `submission_trans_gbdt_postprocessed.csv`.
5. **Step 5: Run Tier 5 (`05_universal_ensemble_blending.ipynb`)**
   - Loads all available OOF predictions from Tiers 1–4, performs Nelder-Mead optimization to find optimal ensemble weights, tunes per-class decision thresholds, and writes the final winning submissions: `submission_ensemble_raw.csv` and `submission_ensemble_postprocessed.csv`.
