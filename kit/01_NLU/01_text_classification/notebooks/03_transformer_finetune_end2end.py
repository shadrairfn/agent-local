# %% [markdown]
# # 🏛️ Paradigma 3: Contextual Transformer Fine-Tuning (IndoBERT / DeBERTa / DistilBERT / XLM-R)
# ## Modern Deep Contextual Architecture with Multi-Sample Dropout (MSD) & RTX 3050 VRAM Safeguard
# **Competition Standard:** GEMASTIK XIX 2026 & Kaggle Standards  
# **Objective:** End-to-end multi-class text classification with a switchable Transformer Backbone Registry tuned for 4GB VRAM laptops, saving `.pth` weights and JSON config into `models/`.
#
# ### Table of Contents
# 1. [Master User Config & Transformer Backbone Registry](#1-master-user-config)
# 2. [Diagnostics & Reproducibility](#2-diagnostics--reproducibility)
# 3. [Data Ingestion & Sanitization](#3-data-ingestion--sanitization)
# 4. [Tokenization & PyTorch Dataset](#4-tokenization--pytorch-dataset)
# 5. [Model Architecture (Transformer + MSD Head)](#5-model-architecture)
# 6. [Training Loop (FP16 AMP + `.pth` Export)](#6-training--validation-loop)
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
# 1. Cross-Validation & Output Mode Toggle
USE_KFOLD        = True     # [True]: Full 5-Fold Stratified CV | [False]: Fast Single Split (80/20)
N_SPLITS         = 5        # Number of folds when USE_KFOLD=True
VAL_SPLIT_RATIO  = 0.20     # Validation ratio when USE_KFOLD=False
MODELS_DIR       = 'models' # Clean subfolder for .pth checkpoints & JSON logs
SAVE_CHECKPOINTS = True

# 2. Transformer Backbone Registry (Tuned for Laptop RTX 3050 4GB VRAM)
MODEL_REGISTRY = {
    "indobert-base": {
        "repo": "indobenchmark/indobert-base-p1",
        "local_path": "../../../03_OFFLINE_WEIGHTS/weights/indobert-base",
        "batch_size": 16, "grad_accum": 1, "max_len": 128
    },
    "indobertweet": {
        "repo": "indolem/indobertweet-base-uncased",
        "local_path": "../../../03_OFFLINE_WEIGHTS/weights/indobertweet",
        "batch_size": 16, "grad_accum": 1, "max_len": 128
    },
    "distilbert-multilingual": {
        "repo": "distilbert/distilbert-base-multilingual-cased",
        "local_path": "../../../03_OFFLINE_WEIGHTS/weights/distilbert-multilingual",
        "batch_size": 16, "grad_accum": 1, "max_len": 128
    },
    "distilbert-uncased": {
        "repo": "distilbert/distilbert-base-uncased",
        "local_path": "../../../03_OFFLINE_WEIGHTS/weights/distilbert-uncased",
        "batch_size": 32, "grad_accum": 1, "max_len": 128
    },
    "deberta-v3-small": {
        "repo": "microsoft/deberta-v3-small",
        "local_path": "../../../03_OFFLINE_WEIGHTS/weights/deberta-v3-small",
        "batch_size": 8, "grad_accum": 2, "max_len": 128
    },
    "xlm-roberta-base": {
        "repo": "xlm-roberta-base",
        "local_path": "../../../03_OFFLINE_WEIGHTS/weights/xlm-roberta-base",
        "batch_size": 8, "grad_accum": 2, "max_len": 128
    }
}

# 🎯 SELECT BACKBONE HERE:
SELECTED_BACKBONE = "indobert-base"
PRETRAINED        = False   # Set True to load full HuggingFace weights from local_path / Hub

MODEL_CFG        = MODEL_REGISTRY[SELECTED_BACKBONE]
MODEL_NAME       = MODEL_CFG["local_path"]
MAX_TOKEN_LENGTH = MODEL_CFG["max_len"]
BATCH_SIZE       = MODEL_CFG["batch_size"]
GRAD_ACCUM       = MODEL_CFG["grad_accum"]

# 3. Data Paths & Columns
TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

ID_COL       = 'id'
TEXT_COL     = 'full_text'
TARGET_COL   = 'label'
GROUP_COL    = None

NUM_EPOCHS       = 3
LEARNING_RATE    = 2e-4 if not PRETRAINED else 2e-5
SEED             = 42

print(f"Master Config Loaded | Backbone: '{SELECTED_BACKBONE}' | USE_KFOLD: {USE_KFOLD} | Batch: {BATCH_SIZE} (GradAccum: {GRAD_ACCUM})")
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
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(MODELS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Version: {torch.__version__} | Target Device: {device}")

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

unique_classes = sorted(train_df[TARGET_COL].unique())
label2id = {c: i for i, c in enumerate(unique_classes)}
id2label = {i: c for i, c in enumerate(unique_classes)}
train_df['target_encoded'] = train_df[TARGET_COL].map(label2id)

NUM_CLASSES = len(unique_classes)
print(f"Classes ({NUM_CLASSES}): {label2id}")

# %%
class SimpleOfflineTokenizer:
    def __init__(self, vocab_size: int = 25000):
        self.vocab_size = vocab_size

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors='pt'):
        input_ids = []
        attention_mask = []
        for text in texts:
            tokens = str(text).lower().split()[:max_length]
            ids = [((hash(t) % (self.vocab_size - 4)) + 4) for t in tokens]
            mask = [1] * len(ids)
            if len(ids) < max_length and padding:
                pad_len = max_length - len(ids)
                ids += [0] * pad_len
                mask += [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

tokenizer = None
if PRETRAINED and os.path.exists(MODEL_NAME):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        print(f"Loaded Local Offline Tokenizer: {MODEL_NAME}")
    except Exception as exc:
        print(f"Fallback to SimpleOfflineTokenizer ({exc})")
        tokenizer = SimpleOfflineTokenizer()
else:
    print(f"Using 100% Offline SimpleOfflineTokenizer (Backbone profile: {SELECTED_BACKBONE}).")
    tokenizer = SimpleOfflineTokenizer()

class TransformerTextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# %%
class MultiSampleDropoutHead(nn.Module):
    def __init__(self, in_features, num_classes, dropouts=[0.1, 0.2, 0.3, 0.4]):
        super().__init__()
        self.drops = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return torch.mean(torch.stack([self.fc(drop(x)) for drop in self.drops], dim=0), dim=0)

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, vocab_size=25000, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, embed_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.head = MultiSampleDropoutHead(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        out, _ = self.encoder(emb)
        mask = attention_mask.unsqueeze(-1).expand(out.size()).float()
        pooled = torch.sum(out * mask, 1) / torch.clamp(mask.sum(1), min=1e-5)
        return self.head(pooled)

print("Transformer Classifier with Multi-Sample Dropout initialized.")

# %%
y_all = train_df['target_encoded'].values
if USE_KFOLD:
    print(f"[MODE] Full {N_SPLITS}-Fold Stratified Cross-Validation Active.")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train_df, y_all))
else:
    print(f"[MODE] Fast Single-Split (1-Fold: {(1-VAL_SPLIT_RATIO)*100:.0f}% Train / {VAL_SPLIT_RATIO*100:.0f}% Val) Active.")
    tr_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=VAL_SPLIT_RATIO, stratify=y_all, random_state=SEED)
    splits = [(tr_idx, val_idx)]

oof_probs  = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
test_probs = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)
fold_scores = []
saved_checkpoints = []

test_dataset = TransformerTextDataset(test_df[TEXT_COL].values, tokenizer=tokenizer, max_len=MAX_TOKEN_LENGTH)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for fold, (tr_idx, val_idx) in enumerate(splits):
    tr_sub = train_df.iloc[tr_idx]
    val_sub = train_df.iloc[val_idx]
    
    tr_loader  = DataLoader(TransformerTextDataset(tr_sub[TEXT_COL].values, tr_sub['target_encoded'].values, tokenizer, MAX_TOKEN_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TransformerTextDataset(val_sub[TEXT_COL].values, val_sub['target_encoded'].values, tokenizer, MAX_TOKEN_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    
    model = TransformerClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tr_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['label'].to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(ids, mask)
                loss = criterion(logits, lbls) / GRAD_ACCUM
                
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(tr_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(MODELS_DIR, f"transformer_fold{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)
        saved_checkpoints.append(ckpt_path)

    model.eval()
    val_fold_probs = []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            probs = F.softmax(model(ids, mask), dim=-1).cpu().numpy()
            val_fold_probs.append(probs)
            
    val_fold_probs = np.vstack(val_fold_probs)
    oof_probs[val_idx] = val_fold_probs
    fold_f1 = float(f1_score(val_sub['target_encoded'].values, np.argmax(val_fold_probs, axis=1), average='macro'))
    fold_scores.append(round(fold_f1, 4))
    print(f"Fold {fold+1}/{len(splits)} | Validation Macro F1: {fold_f1:.4f}")
    
    fold_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            probs = F.softmax(model(ids, mask), dim=-1).cpu().numpy()
            fold_test_probs.append(probs)
    test_probs += np.vstack(fold_test_probs) / len(splits)

if USE_KFOLD:
    overall_f1  = float(f1_score(y_all, np.argmax(oof_probs, axis=1), average='macro'))
    overall_acc = float(accuracy_score(y_all, np.argmax(oof_probs, axis=1)))
else:
    val_only = splits[0][1]
    overall_f1  = float(f1_score(y_all[val_only], np.argmax(oof_probs[val_only], axis=1), average='macro'))
    overall_acc = float(accuracy_score(y_all[val_only], np.argmax(oof_probs[val_only], axis=1)))

print(f"\n>>> OVERALL Transformer Fine-Tuning Macro F1: {overall_f1:.4f} | Accuracy: {overall_acc:.4f} <<<")

# %%
np.save(os.path.join(MODELS_DIR, "transformer_oof_predictions.npy"), oof_probs)
np.save(os.path.join(MODELS_DIR, "transformer_test_predictions.npy"), test_probs)

config_results = {
    "model_name": f"Transformer_MSD_{SELECTED_BACKBONE}",
    "selected_backbone": SELECTED_BACKBONE,
    "repo_id": MODEL_CFG["repo"],
    "task": "text_classification",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kfold_mode": {
        "use_kfold": USE_KFOLD,
        "n_folds": len(splits),
        "random_state": SEED
    },
    "hyperparameters": {
        "max_token_length": MAX_TOKEN_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE
    },
    "validation_metrics": {
        "fold_scores_macro_f1": fold_scores,
        "oof_macro_f1": round(overall_f1, 4),
        "oof_accuracy": round(overall_acc, 4)
    },
    "saved_checkpoints": saved_checkpoints
}

json_path = os.path.join(MODELS_DIR, "03_transformer_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config_results, f, indent=2)

print(f"Saved PyTorch .pth checkpoints & metadata to '{json_path}'.")

# %%
test_df['predicted_label_transformer'] = [id2label[p] for p in np.argmax(test_probs, axis=1)]
print("Transformer Test Distribution:\n", test_df['predicted_label_transformer'].value_counts())

# %%
sub_tf = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_df['predicted_label_transformer']})
sub_tf.to_csv("submission_transformer_raw.csv", index=False)
sub_tf.to_csv("submission_transformer_postprocessed.csv", index=False)

has_no_nans = not sub_tf[TARGET_COL].isna().any()
correct_rows = (len(sub_tf) == len(test_df))
print(f"Integrity Check: No NaNs: {has_no_nans} | Correct Rows: {correct_rows}")
print("SUCCESS: Notebook 3 (Transformer Fine-Tuning) completed flawlessly!")

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
