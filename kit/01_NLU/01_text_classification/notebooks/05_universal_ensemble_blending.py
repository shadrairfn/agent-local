# %% [markdown]
# # 🤝 Paradigma 5: Universal 4-Way Meta-Ensemble Blending
# ## Blending Predictions from 4 Orthogonal Representational Paradigms
# **Competition Standard:** GEMASTIK XIX 2026 & Kaggle Standards  
# **Objective:** Load OOF probabilities from `models/` (GBDT Sparse+Dense, BiLSTM+TextCNN, Transformer Fine-tuned, and Transformer-to-GBDT). Optimize blending weights using Nelder-Mead, tune per-class decision thresholds to maximize Macro F1, and export `models/05_ensemble_config_results.json`.
#
# ### Table of Contents
# 1. [Master User Config](#1-master-user-config)
# 2. [Load Multi-Model OOF & Test Probabilities (`models/`)](#2-load-probabilities)
# 3. [Individual Model Benchmark](#3-individual-model-benchmark)
# 4. [Nelder-Mead 4-Way Weight Optimization](#4-weight-optimization)
# 5. [Probability Threshold Tuning (Macro F1)](#5-threshold-tuning)
# 6. [Ensemble Test Prediction & JSON Config Export](#6-ensemble-prediction)
# 7. [Non-destructive Submission Verification](#7-submission-verification)

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
USE_KFOLD = True  # [True]: Full 5-Fold CV | [False]: Single Split
TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

ID_COL     = 'id'
TARGET_COL = 'label'
MODELS_DIR = 'models'

# Artifact Paths inside models/
MODEL_OOF_PATHS = {
    'GBDT_TFIDF':     f'{MODELS_DIR}/ml_tfidf_oof_predictions.npy',
    'BiLSTM_CNN':     f'{MODELS_DIR}/dl_bilstm_oof_predictions.npy',
    'Transformer_FT': f'{MODELS_DIR}/transformer_oof_predictions.npy',
    'Trans_GBDT':     f'{MODELS_DIR}/trans_gbdt_oof_predictions.npy'
}

MODEL_TEST_PATHS = {
    'GBDT_TFIDF':     f'{MODELS_DIR}/ml_tfidf_test_predictions.npy',
    'BiLSTM_CNN':     f'{MODELS_DIR}/dl_bilstm_test_predictions.npy',
    'Transformer_FT': f'{MODELS_DIR}/transformer_test_predictions.npy',
    'Trans_GBDT':     f'{MODELS_DIR}/trans_gbdt_test_predictions.npy'
}

ENABLE_THRESHOLD_TUNING = True
SEED = 42

print("Master Config Loaded for 4-Way Ensemble Blending.")
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
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.optimize import minimize

os.makedirs(MODELS_DIR, exist_ok=True)
train_df = load_raw_format_to_df(TRAIN_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)
test_df  = load_raw_format_to_df(TEST_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)

unique_classes = sorted(train_df[TARGET_COL].unique())
label2id = {c: i for i, c in enumerate(unique_classes)}
id2label = {i: c for i, c in enumerate(unique_classes)}
y_true = train_df[TARGET_COL].map(label2id).values

oof_dict  = {}
test_dict = {}
individual_scores = {}

print("--- INDIVIDUAL MODEL PERFORMANCE BENCHMARK ---")
for name, path in MODEL_OOF_PATHS.items():
    if os.path.exists(path) and os.path.exists(MODEL_TEST_PATHS[name]):
        oof_dict[name]  = np.load(path)
        test_dict[name] = np.load(MODEL_TEST_PATHS[name])
        score = float(f1_score(y_true, np.argmax(oof_dict[name], axis=1), average='macro'))
        individual_scores[name] = round(score, 4)
        print(f"  * {name:<16} OOF Macro F1: {score:.4f}")
    else:
        print(f"  ! {name:<16} Predictions not found at {path}, skipping.")

active_models = list(oof_dict.keys())
print(f"\nActive Models for Blending ({len(active_models)}): {active_models}")

# %%
def objective_loss(weights):
    weights = np.maximum(weights, 0.0)
    w_sum = np.sum(weights) + 1e-9
    norm_w = weights / w_sum
    
    blended = np.zeros_like(list(oof_dict.values())[0])
    for w, name in zip(norm_w, active_models):
        blended += w * oof_dict[name]
        
    preds = np.argmax(blended, axis=1)
    return -f1_score(y_true, preds, average='macro')

init_weights = np.ones(len(active_models)) / len(active_models)
res = minimize(objective_loss, init_weights, method='Nelder-Mead', options={'maxiter': 500})
best_weights = np.maximum(res.x, 0.0)
best_weights /= np.sum(best_weights)

print("\n--- OPTIMAL BLENDING WEIGHTS ---")
weights_dict = {}
for name, w in zip(active_models, best_weights):
    weights_dict[name] = round(float(w), 4)
    print(f"  - {name:<16}: {w * 100:.2f}%")

final_oof_probs  = np.zeros_like(list(oof_dict.values())[0])
final_test_probs = np.zeros_like(list(test_dict.values())[0])

for w, name in zip(best_weights, active_models):
    final_oof_probs  += w * oof_dict[name]
    final_test_probs += w * test_dict[name]

blended_f1 = float(f1_score(y_true, np.argmax(final_oof_probs, axis=1), average='macro'))
print(f"\n>>> 4-WAY BLENDED OOF Macro F1: {blended_f1:.4f} <<<")

# %%
if ENABLE_THRESHOLD_TUNING:
    def thresh_loss(multipliers):
        scaled = final_oof_probs * multipliers
        preds = np.argmax(scaled, axis=1)
        return -f1_score(y_true, preds, average='macro')

    init_m = np.ones(len(unique_classes))
    res_m = minimize(thresh_loss, init_m, method='Nelder-Mead', options={'maxiter': 500})
    best_multipliers = res_m.x
    
    tuned_oof_preds = np.argmax(final_oof_probs * best_multipliers, axis=1)
    tuned_f1 = float(f1_score(y_true, tuned_oof_preds, average='macro'))
    print(f"\n>>> POST-PROCESSED OOF Macro F1 (Threshold Tuned): {tuned_f1:.4f} (Boost: {tuned_f1 - blended_f1:+.4f}) <<<")
    print("Class Multipliers:", np.round(best_multipliers, 3))
    
    final_test_preds = np.argmax(final_test_probs * best_multipliers, axis=1)
else:
    best_multipliers = np.ones(len(unique_classes))
    tuned_f1 = blended_f1
    final_test_preds = np.argmax(final_test_probs, axis=1)

# %%
np.save(os.path.join(MODELS_DIR, "ensemble_raw_test_predictions.npy"), final_test_probs)

ensemble_results = {
    "model_name": "Universal_4Way_NelderMead_Ensemble",
    "task": "text_classification",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "active_models": active_models,
    "individual_oof_macro_f1": individual_scores,
    "optimal_weights": weights_dict,
    "threshold_multipliers": [round(float(x), 4) for x in best_multipliers],
    "validation_metrics": {
        "blended_oof_macro_f1": round(blended_f1, 4),
        "postprocessed_oof_macro_f1": round(tuned_f1, 4)
    }
}

json_path = os.path.join(MODELS_DIR, "05_ensemble_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(ensemble_results, f, indent=2)

test_df['predicted_label_ensemble'] = [id2label[p] for p in final_test_preds]

sub_raw = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: [id2label[p] for p in np.argmax(final_test_probs, axis=1)]})
sub_raw.to_csv("submission_ensemble_raw.csv", index=False)

sub_post = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_df['predicted_label_ensemble']})
sub_post.to_csv("submission_ensemble_postprocessed.csv", index=False)

has_no_nans = not sub_post[TARGET_COL].isna().any()
correct_rows = (len(sub_post) == len(test_df))
print(f"Saved Ensemble Metadata to '{json_path}'.")
print(f"Integrity Check: No NaNs: {has_no_nans} | Correct Rows: {correct_rows}")
print("\n>>> ALL 4-WAY ENSEMBLE SUBMISSIONS GENERATED SUCCESSFULLY! <<<")

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
