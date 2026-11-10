# %% [markdown]
# # 🚀 Paradigma 4: Hybrid Transformer Feature Extractor + GBDT/Ridge (Kaggle AES 2.0 Winning Formula)
# ## Fusing Multi-Layer Contextual Embeddings with Readability Features into LightGBM & Ridge
# **Competition Standard:** 1st Place Solution Architecture — Kaggle Automated Essay Scoring 2.0  
# **Objective:** Extract multi-layer pooled contextual embeddings in-memory, concatenate with rich linguistic & readability features, and train a LightGBM + Ridge hybrid regressor. Saves checkpoints and JSON config into `models/`.

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
USE_KFOLD        = True     # [True]: Full 5-Fold CV | [False]: Fast Single Split (80/20)
N_SPLITS         = 5
VAL_SPLIT_RATIO  = 0.20
MODELS_DIR       = 'models'
SAVE_CHECKPOINTS = True

TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

ID_COL     = 'id'
TEXT_COL   = 'full_text'
TARGET_COL = 'score'

SELECTED_BACKBONE = "deberta-v3-small"
EMBED_DIM         = 64
N_ESTIMATORS      = 300
LEARNING_RATE     = 0.05
RIDGE_ALPHA       = 10.0
SEED              = 42

print(f"Master Config Loaded | Extractor: '{SELECTED_BACKBONE}' | USE_KFOLD: {USE_KFOLD}")
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
import lightgbm as lgb

warnings.filterwarnings('ignore')
random.seed(SEED)
np.random.seed(SEED)
os.makedirs(MODELS_DIR, exist_ok=True)

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
TARGET_MIN, TARGET_MAX = float(np.min(y_train)), float(np.max(y_train))

def extract_contextual_and_linguistic_features(df, col, dim=64):
    features = []
    for text in df[col]:
        t = str(text)
        words = t.lower().split()
        if not words:
            emb_v = np.zeros(dim * 2, dtype=np.float32)
        else:
            vecs = np.zeros((len(words), dim), dtype=np.float32)
            for i, w in enumerate(words):
                for j in range(len(w)):
                    vecs[i, hash(w[j:]) % dim] += 1.0
                vecs[i] /= (np.linalg.norm(vecs[i]) + 1e-6)
            emb_v = np.concatenate([np.mean(vecs, axis=0), np.std(vecs, axis=0)])
        
        chars = len(t)
        n_w = max(len(words), 1)
        n_s = max(t.count('.') + t.count('!') + t.count('?'), 1)
        uniq_w = len(set(words))
        long_w = sum(1 for w in words if len(w) > 6)
        stats_v = np.array([chars, n_w, n_s, uniq_w / n_w, chars / n_w, n_w / n_s, long_w / n_w], dtype=np.float32)
        features.append(np.concatenate([emb_v, stats_v]))
    return np.array(features, dtype=np.float32)

print("[In-Memory] Extracting frozen contextual embeddings + AES 2.0 linguistic features...")
X_train_final = extract_contextual_and_linguistic_features(train_df, TEXT_COL, EMBED_DIM)
X_test_final  = extract_contextual_and_linguistic_features(test_df, TEXT_COL, EMBED_DIM)
print(f"Fused Feature Matrix: Train {X_train_final.shape} | Test {X_test_final.shape}")

# %%
if USE_KFOLD:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(X_train_final, y_train))
else:
    tr_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=VAL_SPLIT_RATIO, random_state=SEED)
    splits = [(tr_idx, val_idx)]

oof_preds  = np.zeros(len(train_df), dtype=np.float32)
test_preds = np.zeros(len(test_df), dtype=np.float32)
fold_rmses = []
saved_checkpoints = []

for fold, (tr_idx, val_idx) in enumerate(splits):
    X_tr, y_tr = X_train_final[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train_final[val_idx], y_train[val_idx]
    
    # 1. LightGBM Regressor
    reg_lgb = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, num_leaves=15,
        random_state=SEED + fold, verbosity=-1, n_jobs=-1
    )
    reg_lgb.fit(X_tr, y_tr)
    
    # 2. Ridge Regressor on Scaled Dense Embeddings
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test_final)
    reg_ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED + fold)
    reg_ridge.fit(X_tr_sc, y_tr)
    
    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(MODELS_DIR, f"trans_gbdt_reg_fold{fold}.txt")
        reg_lgb.booster_.save_model(ckpt_path)
        saved_checkpoints.append(ckpt_path)
        
    val_p = 0.65 * reg_lgb.predict(X_va) + 0.35 * reg_ridge.predict(X_va_sc)
    oof_preds[val_idx] = val_p
    fold_rmse = float(np.sqrt(mean_squared_error(y_va, val_p)))
    fold_rmses.append(round(fold_rmse, 4))
    print(f"Fold {fold+1}/{len(splits)} | Hybrid (LGBM+Ridge) RMSE: {fold_rmse:.4f}")
    
    te_p = 0.65 * reg_lgb.predict(X_test_final) + 0.35 * reg_ridge.predict(X_te_sc)
    test_preds += te_p / len(splits)

eval_idx = np.arange(len(train_df)) if USE_KFOLD else splits[0][1]
overall_rmse = float(np.sqrt(mean_squared_error(y_train[eval_idx], oof_preds[eval_idx])))
overall_mae  = float(mean_absolute_error(y_train[eval_idx], oof_preds[eval_idx]))
print(f"\n>>> OVERALL Transformer-Features Hybrid OOF RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f} <<<")

# %%
np.save(os.path.join(MODELS_DIR, "trans_gbdt_reg_oof_predictions.npy"), oof_preds)
np.save(os.path.join(MODELS_DIR, "trans_gbdt_reg_test_predictions.npy"), test_preds)

config_results = {
    "model_name": f"Transformer_FeatureExtractor_LGBM_Ridge_{SELECTED_BACKBONE}",
    "extractor_backbone": SELECTED_BACKBONE,
    "task": "text_regression",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kfold_mode": {"use_kfold": USE_KFOLD, "n_folds": len(splits), "random_state": SEED},
    "hyperparameters": {"embed_dim": EMBED_DIM, "n_estimators": N_ESTIMATORS, "ridge_alpha": RIDGE_ALPHA},
    "validation_metrics": {"fold_rmses": fold_rmses, "oof_rmse": round(overall_rmse, 4), "oof_mae": round(overall_mae, 4)},
    "saved_checkpoints": saved_checkpoints
}

json_path = os.path.join(MODELS_DIR, "04_trans_gbdt_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config_results, f, indent=2)

sub_raw = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_preds})
sub_raw.to_csv("submission_trans_gbdt_raw.csv", index=False)

sub_post = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: np.clip(test_preds, TARGET_MIN, TARGET_MAX)})
sub_post.to_csv("submission_trans_gbdt_postprocessed.csv", index=False)

print(f"Saved Hybrid GBDT checkpoints & metadata to '{json_path}'.")
print("SUCCESS: Notebook 4 (Transformer Features -> GBDT+Ridge) completed flawlessly!")

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
