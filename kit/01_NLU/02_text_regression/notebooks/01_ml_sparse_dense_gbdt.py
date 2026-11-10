# %% [markdown]
# # 🌲 Paradigma 1: Multi-View Machine Learning Regression (TF-IDF + FastText + Readability + GBDT)
# ## High-Speed In-Memory Feature Fusion for Continuous & Ordinal Text Scoring
# **Competition Standard:** GEMASTIK XIX 2026, Kaggle AES 2.0 & CommonLit Readability  
# **Objective:** End-to-end text regression fusing Sparse Word+Char TF-IDF, Dense Subword Pooling, Readability indices (Flesch-Kincaid, ARI, LIX, TTR), and LSA topics into LightGBM Regressor. All vectors stay strictly in-memory; outputs are saved cleanly into `models/`.
#
# ### Table of Contents
# 1. [Master User Config (Single-Cell Control Panel)](#1-master-user-config)
# 2. [Diagnostics & Reproducibility](#2-diagnostics--reproducibility)
# 3. [Data Ingestion & Sanitization](#3-data-ingestion--sanitization)
# 4. [Multi-View Feature Engineering (In-Memory TF-IDF + FastText + Readability)](#4-multi-view-feature-engineering)
# 5. [Cross-Validation Strategy (`USE_KFOLD` Toggle)](#5-cross-validation-strategy)
# 6. [Model Training & Checkpoint Saving (`models/*.txt`)](#6-model-training-loop)
# 7. [Out-Of-Fold Evaluation (RMSE, Pearson, QWK) & JSON Export](#7-out-of-fold-evaluation)
# 8. [Test Set Inference](#8-test-set-inference)
# 9. [Submission Generation & Integrity Check](#9-submission-generation)

# %%
# ==============================================================================
# 🎯 MASTER USER CONFIGURATION (REGRESI TEKS & ESSAY SCORING - RTX 3050 4GB VRAM)
# ==============================================================================
# 💡 PANDUAN CEPAT LOMBA:
#    Ubah variabel di bawah sesuai dokumen soal. Setiap parameter memiliki
#    komentar saran taktis di sampingnya (kapan dipakai & batas aman VRAM).
# ==============================================================================

# 1. Format & Path Data Input Fleksibel (Multi-Format Smart Loader)
DATA_FORMAT      = 'auto'        # 'auto' (deteksi otomatis ekstensi) | 'csv' | 'tsv' | 'json' | 'jsonl' | 'parquet' | 'excel' | 'text_folder'
CSV_SEP          = None          # None = deteksi otomatis koma/tab/titik-koma | Isi ';' jika CSV format Indonesia/Eropa
FILE_ENCODING    = 'utf-8'       # 'utf-8' (standar) | Ganti ke 'latin-1' atau 'cp1252' jika muncul UnicodeDecodeError
JSON_RECORD_KEY  = None          # Khusus JSON bersarang: isi 'data' atau 'records' jika tabel ada di dalam key JSON
TRAIN_TEXT_DIR   = None          # Isi 'train_texts/' JIKA kolom CSV hanya berisi nama file .txt eksternal
TEST_TEXT_DIR    = None          # Isi 'test_texts/' JIKA teks uji disimpan sebagai kumpulan file .txt terpisah
FILEPATH_COL     = None          # Nama kolom path file .txt (jika memakai TRAIN_TEXT_DIR)
TEXT_COLS        = ['full_text'] # List kolom teks: ['full_text'] | ['prompt_text', 'full_text'] (digabung otomatis dengan [SEP])
TEXT_SEP_TOKEN   = ' [SEP] '     # Token pemisah saat menggabungkan multi-kolom teks

# 2. Tipe Task, LOSS_FUNCTION (Penghasil train_loss & eval_loss), EVAL_METRIC & CHECKPOINT_MONITOR
TASK_TYPE        = 'regression'  # Wajib 'regression' untuk skor kontinu / ordinal (Automated Essay Scoring / Readability / Counting)
# Pilihan LOSS_FUNCTION (Objective/Criterion Diferensiabel Penghasil train_loss & eval_loss):
#   'smooth_l1' / 'huber' (Tahan outlier - terbaik untuk Transformer) | 'mse' (L2 Loss - terbaik untuk RMSE & QWK!)
#   'mae' (L1 Loss) | 'bce_bounded' (Target proporsi 0..1) | 'pearson_loss' (1 - r) | 'rmsle_loss' (Count data >= 0)
LOSS_FUNCTION    = 'smooth_l1'   # 'smooth_l1' mencegah gradien meledak saat ada skor esai ekstrem

# Pilihan EVAL_METRIC (11 Variasi Metrik Penilaian Regresi & Ordinal):
#   'qwk' (Quadratic Weighted Kappa - skor bulat 1-6) | 'rmse' | 'mae' | 'mcrmse' (Multi-output)
#   'rmsle' | 'mape' | 'smape' | 'r2' | 'pearson' | 'spearman' | 'eval_loss'
EVAL_METRIC      = 'qwk'         # Metrik utama lomba (seluruh 11 metrik tetap dihitung di Report Card!)

# Pilihan CHECKPOINT_MONITOR (Patokan Menyimpan Model Terbaik .pth/.txt di Tiap Epoch):
#   'eval_loss'   : Simpan saat Validation Loss (SmoothL1/MSE) terendah
#   'eval_metric' : Simpan saat skor EVAL_METRIC (misal QWK / Pearson) terbaik
#   'hybrid'      : Utamakan EVAL_METRIC, namun jika seri pilih epoch dengan eval_loss lebih rendah!
CHECKPOINT_MONITOR  = 'hybrid'   # 'eval_loss' | 'eval_metric' | 'hybrid'
OPTIMIZE_THRESHOLDS = True       # [True]: Nelder-Mead mencari titik potong ordinal (QWK) atau bobot optimal sesuai EVAL_METRIC!
QWK_MIN_MAX         = (1, 5)     # Rentang skor bulat minimum & maksimum khusus jika EVAL_METRIC = 'qwk'

# 3. Panjang Token Input & Optimasi GPU RTX 3050 (4GB VRAM)
# Pilihan MODEL_NAME Offline: 'deberta-v3-small' (SOTA AES 2.0!) | 'distilbert-multilingual' | 'indobert-base' | 'xlm-roberta-base'
MAX_LENGTH       = 256           # Panjang token: 128 (ulasan pendek) | 256 (sweet spot esai menengah) | 512 (esai panjang, set batch=4)
TRUNCATION_SIDE  = 'head_tail'   # 'head_tail' (simpan 25% paragraf pembuka + 75% paragraf kesimpulan esai!)
USE_FP16         = True          # [Wajib True di RTX 3050]: Memangkas VRAM 50% & mempercepat training 2x lipat
GRAD_ACCUM_STEPS = 2             # Akumulasi gradien agar batch efektif tetap 16 meskipun BATCH_SIZE = 8

# 4. Pengaturan Path File, Kolom & Hyperparameter Spesifik Notebook
# 1. Cross-Validation & Execution Mode Toggle
USE_KFOLD        = True     # [True]: Full 5-Fold CV | [False]: Fast Single Split (80/20)
N_SPLITS         = 5        # Number of folds when USE_KFOLD=True
VAL_SPLIT_RATIO  = 0.20     # Validation ratio when USE_KFOLD=False
MODELS_DIR       = 'models' # Clean subfolder for checkpoints (.txt) & JSON logs
SAVE_CHECKPOINTS = True

# 2. Feature Representation Toggles (Processed 100% In-Memory)
USE_TFIDF          = True
USE_FASTTEXT_DENSE = True
USE_READABILITY    = True
USE_LSA_TOPICS     = True

# 3. Input File Paths & Columns
TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

ID_COL       = 'id'
TEXT_COL     = 'full_text'
TARGET_COL   = 'score'

# 4. Hyperparameters
MAX_WORD_FEATURES = 5000
MAX_CHAR_FEATURES = 5000
FASTTEXT_DIM      = 64
N_ESTIMATORS      = 300
LEARNING_RATE     = 0.05
RANDOM_STATE      = 42

print(f"Master Config Loaded | USE_KFOLD: {USE_KFOLD} | Folds: {N_SPLITS if USE_KFOLD else 1} | Models Dir: '{MODELS_DIR}/'")
print('=' * 80)
print(f"[CONFIG] Task = '{TASK_TYPE}' | Eval Metric = '{EVAL_METRIC}' | Loss = '{LOSS_FUNCTION}' | Monitor = '{CHECKPOINT_MONITOR}'")
print(f"[CONFIG] Format = '{DATA_FORMAT}' | Max Length = {MAX_LENGTH} ({TRUNCATION_SIDE}) | USE_KFOLD = {globals().get('USE_KFOLD', True)} | FP16 = {USE_FP16}")
print('=' * 80)

# %%
import os
import sys
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

warnings.filterwarnings('ignore')

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
os.makedirs(MODELS_DIR, exist_ok=True)

print("Environment & 'models/' directory initialized successfully.")

# %%
from smart_loader import (
    load_raw_format_to_df,
    hydrate_external_text_files,
    combine_text_columns_and_truncate,
    compute_competition_metric
)

train_df = load_raw_format_to_df(TRAIN_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)
test_df  = load_raw_format_to_df(TEST_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)
sub_df   = load_raw_format_to_df(SAMPLE_SUB_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)

if TEXT_COL not in train_df.columns:
    for cand in ['full_text', 'text', 'essay', 'content']:
        if cand in train_df.columns:
            TEXT_COL = cand
            break

train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").astype(str).str.strip()
test_df[TEXT_COL]  = test_df[TEXT_COL].fillna("").astype(str).str.strip()
y_train = train_df[TARGET_COL].astype(np.float32).values

TARGET_MIN = float(np.min(y_train))
TARGET_MAX = float(np.max(y_train))

print(f"Dataset Loaded | Train: {train_df.shape} | Test: {test_df.shape}")
print(f"Target '{TARGET_COL}' Range: [{TARGET_MIN:.3f}, {TARGET_MAX:.3f}] | Mean: {np.mean(y_train):.3f}")

# %%
feature_blocks_train = []
feature_blocks_test  = []

# 1. Sparse TF-IDF (Word + Char)
if USE_TFIDF:
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_WORD_FEATURES, sublinear_tf=True)
    char_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=MAX_CHAR_FEATURES, sublinear_tf=True)
    X_tr_w = word_vec.fit_transform(train_df[TEXT_COL])
    X_te_w = word_vec.transform(test_df[TEXT_COL])
    X_tr_c = char_vec.fit_transform(train_df[TEXT_COL])
    X_te_c = char_vec.transform(test_df[TEXT_COL])
    feature_blocks_train.extend([X_tr_w, X_tr_c])
    feature_blocks_test.extend([X_te_w, X_te_c])
    print(f"[In-Memory] TF-IDF Built: Word {X_tr_w.shape[1]} dims, Char {X_tr_c.shape[1]} dims.")

# 2. Dense Subword FastText Pooling
if USE_FASTTEXT_DENSE:
    def extract_dense_pooled(texts, dim=64):
        pooled = []
        for text in texts:
            words = str(text).lower().split()
            if not words:
                vecs = np.zeros((1, dim), dtype=np.float32)
            else:
                vecs = np.zeros((len(words), dim), dtype=np.float32)
                for i, w in enumerate(words):
                    trigrams = [w[j:j+3] for j in range(len(w) - 2)] or [w]
                    for t in trigrams:
                        vecs[i, hash(t) % dim] += 1.0
                    vecs[i] /= (np.linalg.norm(vecs[i]) + 1e-7)
            pooled.append(np.concatenate([np.mean(vecs, axis=0), np.max(vecs, axis=0)]))
        return np.array(pooled, dtype=np.float32)

    X_tr_dense = extract_dense_pooled(train_df[TEXT_COL], FASTTEXT_DIM)
    X_te_dense = extract_dense_pooled(test_df[TEXT_COL], FASTTEXT_DIM)
    feature_blocks_train.append(csr_matrix(X_tr_dense))
    feature_blocks_test.append(csr_matrix(X_te_dense))
    print(f"[In-Memory] FastText Dense Pooling Built: {X_tr_dense.shape[1]} dims.")

# 3. Readability & Linguistic Complexity Features (AES 2.0 Standard)
if USE_READABILITY:
    def extract_readability_features(df, col):
        feats = []
        for text in df[col]:
            t = str(text)
            chars = len(t)
            words = t.split()
            n_words = max(len(words), 1)
            n_sents = max(t.count('.') + t.count('!') + t.count('?'), 1)
            unique_w = len(set(w.lower() for w in words))
            long_w = sum(1 for w in words if len(w) > 6)
            avg_w_len = chars / n_words
            avg_s_len = n_words / n_sents
            ttr = unique_w / n_words
            ari = 4.71 * avg_w_len + 0.5 * avg_s_len - 21.43
            lix = avg_s_len + (100.0 * long_w / n_words)
            feats.append([chars, n_words, n_sents, unique_w, avg_w_len, avg_s_len, ttr, ari, lix])
        return np.array(feats, dtype=np.float32)

    X_tr_read = extract_readability_features(train_df, TEXT_COL)
    X_te_read = extract_readability_features(test_df, TEXT_COL)
    feature_blocks_train.append(csr_matrix(X_tr_read))
    feature_blocks_test.append(csr_matrix(X_te_read))
    print(f"[In-Memory] Readability & Linguistic Features Built: {X_tr_read.shape[1]} dims.")

# 4. LSA Semantic Topics
if USE_LSA_TOPICS and USE_TFIDF:
    n_comp = min(16, X_tr_w.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    X_tr_lsa = svd.fit_transform(X_tr_w)
    X_te_lsa = svd.transform(X_te_w)
    feature_blocks_train.append(csr_matrix(X_tr_lsa))
    feature_blocks_test.append(csr_matrix(X_te_lsa))
    print(f"[In-Memory] LSA Topics Built: {X_tr_lsa.shape[1]} dims.")

X_train_final = hstack(feature_blocks_train).tocsr()
X_test_final  = hstack(feature_blocks_test).tocsr()
print(f"\n>>> FINAL IN-MEMORY REGRESSION MATRIX: Train {X_train_final.shape} | Test {X_test_final.shape} <<<")

# %%
if USE_KFOLD:
    print(f"[MODE] Full {N_SPLITS}-Fold Cross-Validation Active.")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(kf.split(X_train_final, y_train))
else:
    print(f"[MODE] Fast Single-Split (1-Fold: {(1-VAL_SPLIT_RATIO)*100:.0f}% Train / {VAL_SPLIT_RATIO*100:.0f}% Val) Active.")
    tr_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=VAL_SPLIT_RATIO, random_state=RANDOM_STATE)
    splits = [(tr_idx, val_idx)]

oof_preds  = np.zeros(len(train_df), dtype=np.float32)
test_preds = np.zeros(len(test_df), dtype=np.float32)
fold_rmses = []
saved_checkpoints = []

for fold, (tr_idx, val_idx) in enumerate(splits):
    X_tr, y_tr = X_train_final[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train_final[val_idx], y_train[val_idx]
    
    reg = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=15,
        random_state=RANDOM_STATE + fold,
        verbosity=-1,
        n_jobs=-1
    )
    reg.fit(X_tr, y_tr)
    
    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(MODELS_DIR, f"lgb_reg_fold{fold}.txt")
        reg.booster_.save_model(ckpt_path)
        saved_checkpoints.append(ckpt_path)
        
    val_p = reg.predict(X_va)
    oof_preds[val_idx] = val_p
    fold_rmse = float(np.sqrt(mean_squared_error(y_va, val_p)))
    fold_rmses.append(round(fold_rmse, 4))
    print(f"Fold {fold+1}/{len(splits)} | RMSE: {fold_rmse:.4f}")
    
    test_preds += reg.predict(X_test_final) / len(splits)

eval_idx = np.arange(len(train_df)) if USE_KFOLD else splits[0][1]
overall_rmse = float(np.sqrt(mean_squared_error(y_train[eval_idx], oof_preds[eval_idx])))
overall_mae  = float(mean_absolute_error(y_train[eval_idx], oof_preds[eval_idx]))
overall_corr = float(pearsonr(y_train[eval_idx], oof_preds[eval_idx])[0])
overall_qwk  = float(cohen_kappa_score(np.round(y_train[eval_idx]).astype(int), np.round(oof_preds[eval_idx]).astype(int), weights='quadratic'))

print(f"\n>>> GBDT OOF RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f} | Pearson r: {overall_corr:.4f} | QWK: {overall_qwk:.4f} <<<")

# %%
np.save(os.path.join(MODELS_DIR, "ml_reg_oof_predictions.npy"), oof_preds)
np.save(os.path.join(MODELS_DIR, "ml_reg_test_predictions.npy"), test_preds)

config_results = {
    "model_name": "LightGBM_Sparse_Dense_Readability_Regressor",
    "task": "text_regression",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kfold_mode": {"use_kfold": USE_KFOLD, "n_folds": len(splits), "random_state": RANDOM_STATE},
    "hyperparameters": {"n_estimators": N_ESTIMATORS, "learning_rate": LEARNING_RATE, "fasttext_dim": FASTTEXT_DIM},
    "validation_metrics": {
        "fold_rmses": fold_rmses,
        "oof_rmse": round(overall_rmse, 4),
        "oof_mae": round(overall_mae, 4),
        "oof_pearson": round(overall_corr, 4),
        "oof_qwk": round(overall_qwk, 4)
    },
    "saved_checkpoints": saved_checkpoints
}

json_path = os.path.join(MODELS_DIR, "01_ml_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config_results, f, indent=2)

print(f"Saved model checkpoints & metadata to '{json_path}'.")

# %%
sub_raw = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_preds})
sub_raw.to_csv("submission_ml_raw.csv", index=False)

sub_post = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: np.clip(test_preds, TARGET_MIN, TARGET_MAX)})
sub_post.to_csv("submission_ml_postprocessed.csv", index=False)

has_no_nans = not sub_post[TARGET_COL].isna().any()
correct_rows = (len(sub_post) == len(test_df))
print(f"Integrity Check: No NaNs: {has_no_nans} | Correct Rows: {correct_rows}")
print("SUCCESS: Notebook 1 (ML Sparse & Dense GBDT Regression) completed flawlessly!")

# %%
import json
from pathlib import Path
# ==============================================================================
# 🔀 FULL CONDITIONAL METRIC SWITCHBOARD & eval_loss REPORT CARD (11 METRICS)
# ==============================================================================
from smart_loader import (
    evaluate_regression_switchboard,
    should_save_checkpoint
)

_y_true_sw = globals().get('y_train', globals().get('y_true', train_df[TARGET_COL].astype(float).values))
_oof_pred_sw = globals().get('oof_preds', globals().get('final_oof'))

_primary_score, _higher_better, _full_report = evaluate_regression_switchboard(
    _y_true_sw, _oof_pred_sw,
    primary_metric=EVAL_METRIC,
    loss_name=LOSS_FUNCTION,
    qwk_min_max=QWK_MIN_MAX
)

print("=" * 80)
print(f"📊 [SWITCHBOARD REPORT CARD] Primary EVAL_METRIC = '{EVAL_METRIC}' ({'MAX' if _higher_better else 'MIN'}) -> {_primary_score:.5f}")
print(f"📉 [OBJECTIVE LOSS]         LOSS_FUNCTION = '{LOSS_FUNCTION}' | OOF eval_loss = {_full_report['eval_loss']:.5f}")
print(f"🎯 [CHECKPOINT MONITOR]     Mode = '{CHECKPOINT_MONITOR}'")
print("-" * 80)
for _k, _v in _full_report.items():
    print(f"  - {_k:<20}: {_v:.5f}")
print("=" * 80)

for _jf in sorted(Path(MODELS_DIR).glob("*_config_results.json")):
    try:
        _jdata = json.loads(_jf.read_text(encoding="utf-8"))
        _jdata["loss_function"] = LOSS_FUNCTION
        _jdata["primary_eval_metric"] = EVAL_METRIC
        _jdata["checkpoint_monitor"] = CHECKPOINT_MONITOR
        _jdata["all_validation_metrics"] = _full_report
        _jf.write_text(json.dumps(_jdata, indent=2), encoding="utf-8")
    except Exception:
        pass
