# 📈 02_text_regression: Universal NLP Regression & Essay Scoring Kit

Welcome to the **Text Regression & Ranking Kit**, engineered for high-speed, medal-contending performance in continuous scoring, readability prediction, and ordinal essay grading (Kaggle Automated Essay Scoring 2.0, CommonLit Readability, GEMASTIK XIX 2026).

---

## 📂 Directory Architecture (Clean Output Standard)

```
02_text_regression/
├── demo_data/                                  # 📊 Verified Demo Dataset (train.csv, test.csv, sample_submission.csv)
├── eda/                                        # 🎯 Task-Specific EDA
│   ├── eda_regression.ipynb                    # Executed notebook: target skewness, kurtosis, QWK grade distribution
│   └── README.md                               # Decision Matrix for regression modeling
├── notebooks/                                  # 🚀 5-Tier Clean-Code Universal Pipelines (Pre-executed with stored outputs)
│   ├── 01_ml_sparse_dense_gbdt.ipynb           # Tier 1: Dual TF-IDF + FastText Dense + Readability (ARI/LIX/TTR) -> LightGBM
│   ├── 02_dl_sequential_bilstm_cnn.ipynb       # Tier 2: BiLSTM + 1D-TextCNN + Attention Pooling (Smooth L1 Huber Loss)
│   ├── 03_transformer_finetune_end2end.ipynb   # Tier 3: Contextual Transformer Registry + Multi-Sample Dropout Head
│   ├── 04_transformer_feature_extractor_gbdt.ipynb # Tier 4: Kaggle AES 2.0 1st Place Formula (Frozen Embeddings + Readability -> LGBM + Ridge)
│   ├── 05_universal_ensemble_blending.ipynb    # Tier 5: 4-Way Blending + Nelder-Mead QWK Ordinal Threshold Cutter & Clipping
│   ├── 📁 models/                              # 💾 All Trained Checkpoints (.pth / .txt) & Structured JSON Results
│   │   ├── lgb_reg_fold0..4.txt
│   │   ├── bilstm_reg_fold0..4.pth
│   │   ├── transformer_reg_fold0..4.pth
│   │   ├── trans_gbdt_reg_fold0..4.txt
│   │   ├── 01_ml_config_results.json
│   │   ├── 02_dl_config_results.json
│   │   ├── 03_transformer_config_results.json
│   │   ├── 04_trans_gbdt_config_results.json
│   │   └── 05_ensemble_config_results.json
│   └── submission_*.csv                        # 10 verified submission files (raw & postprocessed, 0 NaNs)
├── modular_variations/                         # 🧩 Sharpened Lego Bricks (Readability, Pairwise Similarity, Huber Loss, QWK Cutter)
└── switchboard/                                # 🎛️ Dynamic Task Switchboard & Metric Evaluators
```

---

## ⚡ 5-Tier Regression Spectrum

| Tier | Paradigm | Representation | Primary Architecture | Output Checkpoints |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Sparse + Dense + Readability ML** | Word/Char TF-IDF + FastText + ARI/LIX/TTR + LSA | LightGBM Regressor | `models/lgb_reg_fold*.txt` |
| **Tier 2** | **Sequential Deep Learning** | Token Sequences $\to$ Dense Vectors | BiLSTM + 1D-CNN + Attention | `models/bilstm_reg_fold*.pth` |
| **Tier 3** | **End-to-End Transformers** | Contextual Backbone Registry $\to$ MSD Head | DeBERTa / IndoBERT / DistilBERT | `models/transformer_reg_fold*.pth` |
| **Tier 4** | **AES 2.0 Hybrid Feature Extractor** | Frozen Contextual Embeddings + Linguistic Stats | LightGBM (65%) + Ridge (35%) | `models/trans_gbdt_reg_fold*.txt` |
| **Tier 5** | **Ensemble & QWK Optimizer** | 4-Way Prediction Fusion | Nelder-Mead Weights + QWK Threshold Cutter | `models/05_ensemble_config_results.json` |
