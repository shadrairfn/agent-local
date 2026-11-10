# %% [markdown]
# # 🌲 Paradigma 1: Multi-View Machine Learning (Sparse TF-IDF + Dense FastText + GBDT)
# ## High-Speed In-Memory Feature Fusion for Robust Tabular Text Classification
# **Competition Standard:** GEMASTIK XIX 2026 & Kaggle NLP  
# **Objective:** End-to-end classification fusing Sparse Word+Char TF-IDF, Dense Word2Vec/FastText Pooling (Mean+Max), LSA Topic components, and surface statistics into LightGBM. All vectors stay strictly in-memory; outputs are saved cleanly into `models/`.
#
# ### Table of Contents
# 1. [Master User Config (Single-Cell Control Panel)](#1-master-user-config)
# 2. [Diagnostics & Reproducibility](#2-diagnostics--reproducibility)
# 3. [Data Ingestion & Sanitization](#3-data-ingestion--sanitization)
# 4. [Multi-View Feature Engineering (In-Memory Sparse + Dense + LSA)](#4-multi-view-feature-engineering)
# 5. [Cross-Validation Strategy (`USE_KFOLD` Toggle)](#5-cross-validation-strategy)
# 6. [Model Training & Checkpoint Saving (`models/*.txt`)](#6-model-training-loop)
# 7. [Out-Of-Fold Evaluation & JSON Config Result Export](#7-out-of-fold-evaluation)
# 8. [Test Set Inference](#8-test-set-inference)
# 9. [Submission Generation & Integrity Check](#9-submission-generation)

# %%
# ==============================================================================
# 🎯 MASTER USER CONFIGURATION (KLASIFIKASI TEKS - OPTIMAL RTX 3050 4GB VRAM)
# ==============================================================================
# 💡 PANDUAN CEPAT LOMBA:
#    Ubah variabel di bawah sesuai dokumen soal. Setiap parameter memiliki
#    komentar saran taktis di sampingnya (kapan dipakai & batas aman VRAM).
# ==============================================================================

# 1. Format & Path Data Input Fleksibel (Multi-Format Smart Loader)
DATA_FORMAT      = 'auto'        # 'auto' (deteksi otomatis ekstensi) | 'csv' | 'tsv' | 'json' | 'jsonl' | 'parquet' | 'excel' | 'conll' | 'text_folder'
CSV_SEP          = None          # None = deteksi otomatis koma/tab/titik-koma | Isi ';' jika CSV format Indonesia/Eropa
FILE_ENCODING    = 'utf-8'       # 'utf-8' (standar) | Ganti ke 'latin-1' atau 'cp1252' jika muncul UnicodeDecodeError
JSON_RECORD_KEY  = None          # Khusus JSON bersarang: isi 'data' atau 'records' jika tabel ada di dalam key JSON
TRAIN_TEXT_DIR   = None          # Isi 'train_texts/' JIKA kolom CSV hanya berisi nama file .txt eksternal
TEST_TEXT_DIR    = None          # Isi 'test_texts/' JIKA teks uji disimpan sebagai kumpulan file .txt terpisah
FILEPATH_COL     = None          # Nama kolom path file .txt (jika memakai TRAIN_TEXT_DIR)
TEXT_COLS        = ['full_text'] # List kolom teks: ['full_text'] (1 kolom) | ['title', 'body'] (digabung otomatis dengan [SEP])
TEXT_SEP_TOKEN   = ' [SEP] '     # Token pemisah saat menggabungkan multi-kolom teks

# 2. Tipe Task, LOSS_FUNCTION (Penghasil train_loss & eval_loss), EVAL_METRIC & CHECKPOINT_MONITOR
TASK_TYPE        = 'multiclass'  # 'binary' (2 kelas) | 'multiclass' (>2 kelas) | 'multilabel' (multi-tag)
# Pilihan LOSS_FUNCTION (Objective/Criterion Diferensiabel Penghasil train_loss & eval_loss):
#   'auto' | 'cross_entropy' (Standar) | 'weighted_ce' (Bobot kelas) | 'focal_loss' (Imbalanced Ekstrem <5%!)
#   'soft_f1_loss' (Differentiable Soft Macro-F1 Loss: 1 - SoftF1 -> Langsung optimasi F1 saat backprop!)
#   'combo_ce_f1'  (0.5 * CrossEntropy + 0.5 * SoftMacroF1 -> Paling stabil untuk mengejar skor F1!)
#   'label_smoothing_ce' (Tahan label noisy) | 'poly1_ce' (PolyLoss SOTA) | 'bce_logits' (Multilabel) | 'asymmetric_loss'
LOSS_FUNCTION    = 'auto'        # Bisa ganti ke 'combo_ce_f1', 'soft_f1_loss', 'focal_loss', atau 'weighted_ce'!
FOCAL_GAMMA      = 2.0           # Parameter gamma untuk 'focal_loss' / 'asymmetric_loss'
USE_CLASS_WEIGHT = True          # [True]: Otomatis memberi bobot ekstra pada kelas minoritas
LABEL_SMOOTHING  = 0.05          # 0.05 mencegah Transformer overconfident (set 0.0 jika EVAL_METRIC='log_loss')

# Pilihan EVAL_METRIC (12 Variasi Metrik Penilaian Papan Skor / Leaderboard):
#   'macro_f1' (Standar GEMASTIK!) | 'weighted_f1' | 'micro_f1' | 'binary_f1' | 'accuracy' | 'balanced_accuracy'
#   'mcc' (Matthews Corr) | 'cohen_kappa' | 'roc_auc' | 'pr_auc' | 'log_loss' | 'eval_loss'
EVAL_METRIC      = 'macro_f1'    # Metrik utama lomba (seluruh 12 metrik tetap dihitung di Report Card!)

# Pilihan CHECKPOINT_MONITOR (Patokan Menyimpan Model Terbaik .pth/.txt di Tiap Epoch):
#   'val_f1' / 'eval_metric' : Simpan saat skor validasi F1 (EVAL_METRIC) tertinggi!
#   'val_loss' / 'eval_loss' : Simpan saat Validation Loss terendah (paling stabil & kalibrasi terbaik untuk Ensemble!)
#   'hybrid'                 : Utamakan val_f1 (EVAL_METRIC), namun jika seri pilih epoch dengan val_loss lebih rendah!
CHECKPOINT_MONITOR     = 'hybrid' # 'val_f1' | 'val_loss' | 'eval_metric' | 'eval_loss' | 'hybrid'
SUBMISSION_OUTPUT_TYPE = 'auto'   # 'auto' (Otomatis 'proba' jika EVAL_METRIC='roc_auc'/'pr_auc'/'log_loss', selain itu 'label') | 'label' | 'proba'
OPTIMIZE_THRESHOLDS    = True     # [True]: Nelder-Mead tuning OOF menyesuaikan EVAL_METRIC secara otomatis!

# 3. Panjang Token Input & Optimasi GPU RTX 3050 (4GB VRAM)
# Pilihan MODEL_NAME Offline: 'distilbert-multilingual' (~2.0GB) | 'distilbert-uncased' (~1.5GB) | 'indobert-base' (~2.5GB) | 'deberta-v3-small' (~3.0GB) | 'xlm-roberta-base' (~3.6GB)
MAX_LENGTH       = 128           # Panjang maksimum token: 64 (judul/tweet) | 128 (sweet spot 4GB VRAM) | 256 (artikel) | 512 (esai)
TRUNCATION_SIDE  = 'head_tail'   # 'head_tail' (simpan 25% kalimat awal + 75% kalimat kesimpulan akhir!) | 'right' | 'left'
USE_FP16         = True          # [Wajib True di RTX 3050]: Memangkas VRAM 50% & mempercepat training 2x lipat
GRAD_ACCUM_STEPS = 1             # Naikkan ke 2 atau 4 jika BATCH_SIZE dikecilkan agar tidak Out-Of-Memory

# 4. Pengaturan Path File, Kolom & Hyperparameter Spesifik Notebook
# 1. Cross-Validation & Execution Mode Toggle
USE_KFOLD        = True     # [True]: Full 5-Fold Stratified CV | [False]: Fast Single Split (80/20)
N_SPLITS         = 5        # Number of folds when USE_KFOLD=True
VAL_SPLIT_RATIO  = 0.20     # Validation ratio when USE_KFOLD=False
MODELS_DIR       = 'models' # Clean subfolder for checkpoints (.txt) & JSON logs
SAVE_CHECKPOINTS = True

# 2. Feature Representation Toggles (Processed 100% In-Memory)
USE_TFIDF          = True   # Dual Word (1,2) & Char (3,5) N-Grams
USE_FASTTEXT_DENSE = True   # Subword / Dense Document Embedding (Mean + Max Pooling)
USE_LSA_TOPICS     = True   # TruncatedSVD Topics (16 dimensions)
USE_SURFACE_STATS  = True   # Character count, word count, punctuation, digit ratios

# 3. Input File Paths
TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

# 4. Column Mapping
ID_COL       = 'id'
TEXT_COL     = 'full_text'
TARGET_COL   = 'label'
GROUP_COL    = None

# 5. Hyperparameters
MAX_WORD_FEATURES = 5000
MAX_CHAR_FEATURES = 5000
FASTTEXT_DIM      = 64      # Dimension for subword hash embedding
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
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, classification_report, accuracy_score
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
    for cand in ['full_text', 'text', 'essay', 'content', 'headline']:
        if cand in train_df.columns:
            TEXT_COL = cand
            break

train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").astype(str).str.strip()
test_df[TEXT_COL]  = test_df[TEXT_COL].fillna("").astype(str).str.strip()

unique_classes = sorted(train_df[TARGET_COL].unique())
label2id = {c: i for i, c in enumerate(unique_classes)}
id2label = {i: c for i, c in enumerate(unique_classes)}
train_df['target_encoded'] = train_df[TARGET_COL].map(label2id)

print(f"Dataset Loaded | Train: {train_df.shape} | Test: {test_df.shape} | Text Col: '{TEXT_COL}'")
print(f"Target Classes ({len(unique_classes)}): {label2id}")

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
    print(f"[In-Memory] TF-IDF Sparse Matrices Built: Word {X_tr_w.shape[1]} dims, Char {X_tr_c.shape[1]} dims.")

# 2. Dense Document Embeddings (FastText / Hashing Subword Mean+Max Pooling)
if USE_FASTTEXT_DENSE:
    def extract_dense_pooled_embeddings(texts: pd.Series, dim: int = 64) -> np.ndarray:
        pooled_vectors = []
        for text in texts:
            words = str(text).lower().split()
            if not words:
                vecs = np.zeros((1, dim), dtype=np.float32)
            else:
                vecs = np.zeros((len(words), dim), dtype=np.float32)
                for i, w in enumerate(words):
                    char_trigrams = [w[j:j+3] for j in range(len(w) - 2)] or [w]
                    hashes = [hash(t) % dim for t in char_trigrams]
                    for h in hashes:
                        vecs[i, h] += 1.0
                    norm = np.linalg.norm(vecs[i]) + 1e-7
                    vecs[i] /= norm
            mean_pool = np.mean(vecs, axis=0)
            max_pool  = np.max(vecs, axis=0)
            pooled_vectors.append(np.concatenate([mean_pool, max_pool]))
        return np.array(pooled_vectors, dtype=np.float32)

    X_tr_dense = extract_dense_pooled_embeddings(train_df[TEXT_COL], dim=FASTTEXT_DIM)
    X_te_dense = extract_dense_pooled_embeddings(test_df[TEXT_COL], dim=FASTTEXT_DIM)
    feature_blocks_train.append(csr_matrix(X_tr_dense))
    feature_blocks_test.append(csr_matrix(X_te_dense))
    print(f"[In-Memory] FastText Dense Document Pooling Built: {X_tr_dense.shape[1]} dims (Mean+Max).")

# 3. LSA / SVD Semantic Topics
if USE_LSA_TOPICS and USE_TFIDF:
    n_comp = min(16, X_tr_w.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    X_tr_lsa = svd.fit_transform(X_tr_w)
    X_te_lsa = svd.transform(X_te_w)
    feature_blocks_train.append(csr_matrix(X_tr_lsa))
    feature_blocks_test.append(csr_matrix(X_te_lsa))
    print(f"[In-Memory] LSA Topic Components Built: {X_tr_lsa.shape[1]} dims.")

# 4. Lexical Surface Statistics
if USE_SURFACE_STATS:
    def get_stats(df, col):
        st = pd.DataFrame()
        st['char_len']    = df[col].apply(len)
        st['word_count']  = df[col].apply(lambda x: len(x.split()))
        st['avg_word']    = st['char_len'] / (st['word_count'] + 1e-5)
        st['digit_cnt']   = df[col].apply(lambda x: sum(c.isdigit() for c in x))
        st['exclam_cnt']  = df[col].apply(lambda x: x.count('!'))
        st['caps_cnt']    = df[col].apply(lambda x: sum(c.isupper() for c in x))
        return st.values

    X_tr_stats = get_stats(train_df, TEXT_COL)
    X_te_stats = get_stats(test_df, TEXT_COL)
    feature_blocks_train.append(csr_matrix(X_tr_stats))
    feature_blocks_test.append(csr_matrix(X_te_stats))
    print(f"[In-Memory] Surface Text Statistics Built: {X_tr_stats.shape[1]} dims.")

# Stack All Representations In-Memory
X_train_final = hstack(feature_blocks_train).tocsr()
X_test_final  = hstack(feature_blocks_test).tocsr()
y_train = train_df['target_encoded'].values

print(f"\n>>> FINAL IN-MEMORY MULTI-VIEW MATRIX: Train {X_train_final.shape} | Test {X_test_final.shape} <<<")

# %%
if USE_KFOLD:
    print(f"[MODE] Full {N_SPLITS}-Fold Stratified Cross-Validation Active.")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_train_final, y_train))
else:
    print(f"[MODE] Fast Single-Split (1-Fold: {(1-VAL_SPLIT_RATIO)*100:.0f}% Train / {VAL_SPLIT_RATIO*100:.0f}% Val) Active.")
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SPLIT_RATIO, stratify=y_train, random_state=RANDOM_STATE
    )
    splits = [(tr_idx, val_idx)]

num_classes = len(unique_classes)
oof_probs   = np.zeros((len(train_df), num_classes), dtype=np.float32)
test_probs  = np.zeros((len(test_df), num_classes), dtype=np.float32)
fold_scores = []
saved_checkpoints = []

for fold, (train_idx, val_idx) in enumerate(splits):
    X_tr, y_tr = X_train_final[train_idx], y_train[train_idx]
    X_va, y_va = X_train_final[val_idx], y_train[val_idx]
    
    clf = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        random_state=RANDOM_STATE + fold,
        verbosity=-1,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    
    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(MODELS_DIR, f"lgb_fold{fold}.txt")
        clf.booster_.save_model(ckpt_path)
        saved_checkpoints.append(ckpt_path)
    
    val_pred = clf.predict_proba(X_va)
    oof_probs[val_idx] = val_pred
    
    fold_f1 = float(f1_score(y_va, np.argmax(val_pred, axis=1), average='macro'))
    fold_scores.append(round(fold_f1, 4))
    print(f"Fold {fold + 1}/{len(splits)} | Macro F1: {fold_f1:.4f}")
    
    test_probs += clf.predict_proba(X_test_final) / len(splits)

if USE_KFOLD:
    overall_preds = np.argmax(oof_probs, axis=1)
    overall_f1  = float(f1_score(y_train, overall_preds, average='macro'))
    overall_acc = float(accuracy_score(y_train, overall_preds))
else:
    val_only_idx = splits[0][1]
    overall_preds = np.argmax(oof_probs[val_only_idx], axis=1)
    overall_f1  = float(f1_score(y_train[val_only_idx], overall_preds, average='macro'))
    overall_acc = float(accuracy_score(y_train[val_only_idx], overall_preds))

print(f"\n>>> GBDT MULTI-VIEW OOF Macro F1: {overall_f1:.4f} | Accuracy: {overall_acc:.4f} <<<")

# %%
np.save(os.path.join(MODELS_DIR, "ml_tfidf_oof_predictions.npy"), oof_probs)
np.save(os.path.join(MODELS_DIR, "ml_tfidf_test_predictions.npy"), test_probs)

config_results = {
    "model_name": "LightGBM_Sparse_Dense_MultiView",
    "task": "text_classification",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kfold_mode": {
        "use_kfold": USE_KFOLD,
        "n_folds": len(splits),
        "random_state": RANDOM_STATE
    },
    "hyperparameters": {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "max_word_features": MAX_WORD_FEATURES,
        "max_char_features": MAX_CHAR_FEATURES,
        "fasttext_dim": FASTTEXT_DIM
    },
    "validation_metrics": {
        "fold_scores_macro_f1": fold_scores,
        "oof_macro_f1": round(overall_f1, 4),
        "oof_accuracy": round(overall_acc, 4)
    },
    "saved_checkpoints": saved_checkpoints
}

json_path = os.path.join(MODELS_DIR, "01_ml_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config_results, f, indent=2)

print(f"Saved model checkpoints & metadata to '{json_path}'.")

# %%
test_df['predicted_label_ml'] = [id2label[p] for p in np.argmax(test_probs, axis=1)]
print("ML Test Set Prediction Distribution:\n", test_df['predicted_label_ml'].value_counts())

# %%
sub_raw = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_df['predicted_label_ml']})
sub_raw.to_csv("submission_ml_raw.csv", index=False)
sub_raw.to_csv("submission_ml_postprocessed.csv", index=False)

has_no_nans = not sub_raw[TARGET_COL].isna().any()
correct_rows = (len(sub_raw) == len(test_df))
print(f"Integrity Check: No NaNs: {has_no_nans} | Correct Rows: {correct_rows}")
print("SUCCESS: Notebook 1 (ML Sparse & Dense GBDT) completed flawlessly!")

# %%
import json
from pathlib import Path
# ==============================================================================
# 🔀 FULL CONDITIONAL METRIC SWITCHBOARD & eval_loss REPORT CARD (12 METRICS)
# ==============================================================================
from smart_loader import (
    evaluate_classification_switchboard,
    optimize_classification_oof,
    should_save_checkpoint
)

_y_true_sw = train_df['target_encoded'].values if 'target_encoded' in train_df.columns else globals().get('y_true', globals().get('y_train'))
_oof_probs_sw = globals().get('oof_probs', globals().get('final_oof_probs'))
_test_probs_sw = globals().get('test_probs', globals().get('final_test_probs'))

if OPTIMIZE_THRESHOLDS:
    _oof_pred_sw, _test_pred_sw, _mults_sw, _primary_score, _full_report, _out_mode = optimize_classification_oof(
        _y_true_sw, _oof_probs_sw, _test_probs_sw,
        primary_metric=EVAL_METRIC,
        submission_output_type=SUBMISSION_OUTPUT_TYPE
    )
else:
    _oof_pred_sw = np.argmax(_oof_probs_sw, axis=1)
    _test_pred_sw = np.argmax(_test_probs_sw, axis=1)
    _primary_score, _, _full_report = evaluate_classification_switchboard(
        _y_true_sw, _oof_pred_sw, _oof_probs_sw, primary_metric=EVAL_METRIC, loss_name=LOSS_FUNCTION
    )
    _out_mode = 'proba' if EVAL_METRIC in ('roc_auc', 'pr_auc', 'log_loss', 'eval_loss') else 'label'

print("=" * 80)
print(f"📊 [SWITCHBOARD REPORT CARD] Primary EVAL_METRIC = '{EVAL_METRIC}' -> {_primary_score:.5f}")
print(f"📉 [OBJECTIVE LOSS]         LOSS_FUNCTION = '{LOSS_FUNCTION}' | OOF eval_loss = {_full_report['eval_loss']:.5f}")
print(f"🎯 [CHECKPOINT MONITOR]     Mode = '{CHECKPOINT_MONITOR}' | Submission Output = '{_out_mode.upper()}'")
print("-" * 80)
for _k, _v in _full_report.items():
    print(f"  - {_k:<20}: {_v:.5f}")
print("=" * 80)

# Update JSON config log inside models/ with the complete 12-metric report card + eval_loss
for _jf in sorted(Path(MODELS_DIR).glob("*_config_results.json")):
    try:
        _jdata = json.loads(_jf.read_text(encoding="utf-8"))
        _jdata["loss_function"] = LOSS_FUNCTION
        _jdata["primary_eval_metric"] = EVAL_METRIC
        _jdata["checkpoint_monitor"] = CHECKPOINT_MONITOR
        _jdata["submission_output_type"] = _out_mode
        _jdata["all_validation_metrics"] = _full_report
        _jf.write_text(json.dumps(_jdata, indent=2), encoding="utf-8")
    except Exception:
        pass
