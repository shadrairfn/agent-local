# %% [markdown]
# # 🤝 Paradigma 5: Universal 4-Way Regression Blending & QWK Threshold Cutter
# ## Blending Continuous Predictions + Nelder-Mead Ordinal QWK Optimization
# **Competition Standard:** GEMASTIK XIX 2026 & Kaggle AES 2.0  
# **Objective:** Blend OOF predictions from `models/` across all 4 regression tiers using Nelder-Mead RMSE/Pearson optimization, apply target range clipping and optional Nelder-Mead QWK ordinal threshold cutting, and export `models/05_ensemble_config_results.json`.

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
USE_KFOLD = True  # [True]: Full 5-Fold CV | [False]: Single Split
TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

ID_COL     = 'id'
TARGET_COL = 'score'
MODELS_DIR = 'models'

# Post-Processing Mode: 'continuous_clip' (for RMSE/Pearson) or 'ordinal_qwk' (for integer grades 1..6)
POSTPROCESS_MODE = 'continuous_clip'

MODEL_OOF_PATHS = {
    'GBDT_MultiView': f'{MODELS_DIR}/ml_reg_oof_predictions.npy',
    'BiLSTM_CNN':     f'{MODELS_DIR}/dl_bilstm_reg_oof_predictions.npy',
    'Transformer_FT': f'{MODELS_DIR}/transformer_reg_oof_predictions.npy',
    'Trans_GBDT':     f'{MODELS_DIR}/trans_gbdt_reg_oof_predictions.npy'
}

MODEL_TEST_PATHS = {
    'GBDT_MultiView': f'{MODELS_DIR}/ml_reg_test_predictions.npy',
    'BiLSTM_CNN':     f'{MODELS_DIR}/dl_bilstm_reg_test_predictions.npy',
    'Transformer_FT': f'{MODELS_DIR}/transformer_reg_test_predictions.npy',
    'Trans_GBDT':     f'{MODELS_DIR}/trans_gbdt_reg_test_predictions.npy'
}

SEED = 42
print(f"Master Config Loaded | Postprocess Mode: '{POSTPROCESS_MODE}'")
print('=' * 80)
print(f"[CONFIG] Task = '{TASK_TYPE}' | Eval Metric = '{EVAL_METRIC}' | Loss = '{LOSS_FUNCTION}' | Monitor = '{CHECKPOINT_MONITOR}'")
print(f"[CONFIG] Format = '{DATA_FORMAT}' | Max Length = {MAX_LENGTH} ({TRUNCATION_SIDE}) | USE_KFOLD = {globals().get('USE_KFOLD', True)} | FP16 = {USE_FP16}")
print('=' * 80)

# %%
from smart_loader import (
    load_raw_format_to_df,
    hydrate_external_text_files,
    combine_text_columns_and_truncate,
    compute_competition_metric
)

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
from scipy.optimize import minimize

os.makedirs(MODELS_DIR, exist_ok=True)
train_df = load_raw_format_to_df(TRAIN_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)
test_df  = load_raw_format_to_df(TEST_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)

y_true = train_df[TARGET_COL].astype(np.float32).values
TARGET_MIN, TARGET_MAX = float(np.min(y_true)), float(np.max(y_true))

oof_dict, test_dict, individual_rmses = {}, {}, {}

print("--- INDIVIDUAL REGRESSION MODEL BENCHMARK ---")
for name, path in MODEL_OOF_PATHS.items():
    if os.path.exists(path) and os.path.exists(MODEL_TEST_PATHS[name]):
        oof_dict[name]  = np.load(path)
        test_dict[name] = np.load(MODEL_TEST_PATHS[name])
        rmse = float(np.sqrt(mean_squared_error(y_true, oof_dict[name])))
        individual_rmses[name] = round(rmse, 4)
        print(f"  * {name:<16} OOF RMSE: {rmse:.4f}")
    else:
        print(f"  ! {name:<16} Predictions not found at {path}, skipping.")

active_models = list(oof_dict.keys())

# %%
def rmse_objective(weights):
    weights = np.maximum(weights, 0.0)
    norm_w = weights / (np.sum(weights) + 1e-9)
    blend = np.zeros_like(y_true)
    for w, name in zip(norm_w, active_models):
        blend += w * oof_dict[name]
    return np.sqrt(mean_squared_error(y_true, blend))

init_w = np.ones(len(active_models)) / len(active_models)
res = minimize(rmse_objective, init_w, method='Nelder-Mead', options={'maxiter': 500})
best_weights = np.maximum(res.x, 0.0)
best_weights /= np.sum(best_weights)

weights_dict = {}
print("\n--- OPTIMAL REGRESSION BLENDING WEIGHTS ---")
for name, w in zip(active_models, best_weights):
    weights_dict[name] = round(float(w), 4)
    print(f"  - {name:<16}: {w * 100:.2f}%")

final_oof  = np.zeros_like(y_true)
final_test = np.zeros(len(test_df), dtype=np.float32)
for w, name in zip(best_weights, active_models):
    final_oof  += w * oof_dict[name]
    final_test += w * test_dict[name]

blended_rmse    = float(np.sqrt(mean_squared_error(y_true, final_oof)))
blended_pearson = float(pearsonr(y_true, final_oof)[0])

# Compute QWK on rounded grades (AES 2.0 metric check)
y_int = np.round(y_true).astype(int)
initial_cuts = np.arange(y_int.min() + 0.5, y_int.max() + 0.5, 1.0)

def apply_cuts(preds, cuts):
    return np.digitize(preds, np.sort(cuts)) + y_int.min()

def qwk_loss(cuts):
    return -cohen_kappa_score(y_int, apply_cuts(final_oof, cuts), weights='quadratic')

res_qwk = minimize(qwk_loss, initial_cuts, method='Nelder-Mead', options={'maxiter': 300})
best_cuts = np.sort(res_qwk.x)
tuned_qwk = float(-qwk_loss(best_cuts))

print(f"\n>>> 4-WAY BLENDED OOF RMSE: {blended_rmse:.4f} | Pearson r: {blended_pearson:.4f} | Tuned QWK: {tuned_qwk:.4f} <<<")

# %%
np.save(os.path.join(MODELS_DIR, "ensemble_raw_test_predictions.npy"), final_test)

if POSTPROCESS_MODE == 'ordinal_qwk':
    post_test_preds = apply_cuts(final_test, best_cuts)
else:
    post_test_preds = np.clip(final_test, TARGET_MIN, TARGET_MAX)

ensemble_results = {
    "model_name": "Universal_4Way_Regression_Ensemble_QWK",
    "task": "text_regression",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "postprocess_mode": POSTPROCESS_MODE,
    "active_models": active_models,
    "individual_oof_rmse": individual_rmses,
    "optimal_weights": weights_dict,
    "optimal_qwk_thresholds": [round(float(c), 4) for c in best_cuts],
    "validation_metrics": {
        "blended_oof_rmse": round(blended_rmse, 4),
        "blended_oof_pearson": round(blended_pearson, 4),
        "tuned_oof_qwk": round(tuned_qwk, 4)
    }
}

json_path = os.path.join(MODELS_DIR, "05_ensemble_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(ensemble_results, f, indent=2)

sub_raw = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: final_test})
sub_raw.to_csv("submission_ensemble_raw.csv", index=False)

sub_post = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: post_test_preds})
sub_post.to_csv("submission_ensemble_postprocessed.csv", index=False)

has_no_nans = not sub_post[TARGET_COL].isna().any()
correct_rows = (len(sub_post) == len(test_df))
print(f"Saved Ensemble Metadata to '{json_path}'.")
print(f"Integrity Check: No NaNs: {has_no_nans} | Correct Rows: {correct_rows}")
print("\n>>> ALL 4-WAY REGRESSION ENSEMBLE SUBMISSIONS GENERATED SUCCESSFULLY! <<<")

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
