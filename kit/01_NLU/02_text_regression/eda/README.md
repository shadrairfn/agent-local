# 🎯 Task-Specific EDA: Text Regression & Ordinal Scoring

## Overview
This diagnostic notebook analyzes continuous and ordinal numeric targets in text regression challenges (Kaggle AES 2.0, Feedback Prize, CommonLit Readability, Jigsaw Toxicity).

## Key Diagnostic Outputs
1. **Target Distribution & Skewness:** Determines whether target transformation (`log1p`, `Box-Cox`) is required.
2. **Discrete / Ordinal Check:** Identifies whether the competition evaluates continuous metrics (RMSE, MAE) or step-based penalties (Quadratic Weighted Kappa - QWK).
3. **Length vs Target Correlation:** Quantifies how much score is explainable purely by response length (a known phenomenon in student essay grading).

## Executed Notebook
- Open [eda_regression.ipynb](file:///c:/Users/yasse/OneDrive/Desktop/Mine%20All%20Ds/kitGemastikk/NLP-FULL%20KIT/01_NLU/02_text_regression/eda/eda_regression.ipynb) to inspect pre-rendered charts and statistical reports.
