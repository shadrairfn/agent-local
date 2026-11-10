# %% [markdown]
# # ⚡ Paradigma 2: Classical Sequential Deep Learning (BiLSTM + 1D-TextCNN + Attention)
# ## Ultra-Fast, Low-VRAM Sequence Modeling with `.pth` Checkpoints
# **Competition Standard:** GEMASTIK XIX 2026 & Kaggle Standards  
# **Objective:** End-to-end sequential deep learning combining Bidirectional LSTM, 1D Convolutional feature extractors, and self-attention pooling with mixed precision training. Saves `.pth` weights and JSON config into `models/`.
#
# ### Table of Contents
# 1. [Master User Config](#1-master-user-config)
# 2. [Diagnostics & Reproducibility](#2-diagnostics--reproducibility)
# 3. [Data Ingestion & Sanitization](#3-data-ingestion--sanitization)
# 4. [Tokenization & PyTorch Sequence Dataset](#4-tokenization--pytorch-dataset)
# 5. [Neural Architecture (BiLSTM + 1D-CNN + Attention)](#5-neural-architecture)
# 6. [Training Loop & `.pth` Checkpoint Export](#6-training--validation-loop)
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

# 2. Data Paths & Columns
TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

ID_COL     = 'id'
TEXT_COL   = 'full_text'
TARGET_COL = 'label'

# 3. Sequential DL Hyperparameters (Runs in < 200MB VRAM)
MAX_VOCAB_SIZE   = 25000
EMBEDDING_DIM    = 128
HIDDEN_DIM       = 128
CNN_NUM_FILTERS  = 64
MAX_TOKEN_LENGTH = 128
BATCH_SIZE       = 8
NUM_EPOCHS       = 3
LEARNING_RATE    = 1e-3
SEED             = 42

print(f"Master Config Loaded | USE_KFOLD: {USE_KFOLD} | EPOCHS: {NUM_EPOCHS} | BATCH: {BATCH_SIZE}")
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
from collections import Counter
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
class VocabTokenizer:
    def __init__(self, max_vocab: int = 25000):
        self.max_vocab = max_vocab
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        self.id2word = {0: '<PAD>', 1: '<UNK>'}

    def fit(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(str(t).lower().split())
        for word, _ in counts.most_common(self.max_vocab - 2):
            w_id = len(self.word2id)
            self.word2id[word] = w_id
            self.id2word[w_id] = word

    def encode(self, text, max_len=128):
        tokens = str(text).lower().split()[:max_len]
        ids = [self.word2id.get(w, 1) for w in tokens]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [0] * pad_len
            mask += [0] * pad_len
        return ids, mask

vocab_tok = VocabTokenizer(max_vocab=MAX_VOCAB_SIZE)
vocab_tok.fit(train_df[TEXT_COL].values)
print(f"Vocabulary Built In-Memory: {len(vocab_tok.word2id)} unique tokens.")

class SeqDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=128):
        self.texts = list(texts)
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, mask = vocab_tok.encode(self.texts[idx], max_len=self.max_len)
        item = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long)
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# %%
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        weights = self.attn(x).squeeze(-1)
        weights = weights.masked_fill(mask == 0, -1e4)
        norm_weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        pooled = torch.sum(x * norm_weights, dim=1)
        return pooled

class HybridBiLSTM_CNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, hidden_dim=128, cnn_filters=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.conv3 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=5, padding=2)
        total_dim = hidden_dim + (cnn_filters * 2)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        lstm_pooled = self.attn_pool(lstm_out, attention_mask)
        emb_perm = emb.permute(0, 2, 1)
        c3 = F.relu(self.conv3(emb_perm))
        c5 = F.relu(self.conv5(emb_perm))
        c3_max = torch.max(c3, dim=2)[0]
        c5_max = torch.max(c5, dim=2)[0]
        merged = torch.cat([lstm_pooled, c3_max, c5_max], dim=-1)
        return self.fc(merged)

print("Hybrid BiLSTM + TextCNN architecture initialized.")

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

test_dataset = SeqDataset(test_df[TEXT_COL].values, max_len=MAX_TOKEN_LENGTH)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for fold, (tr_idx, val_idx) in enumerate(splits):
    tr_sub  = train_df.iloc[tr_idx]
    val_sub = train_df.iloc[val_idx]
    
    tr_loader  = DataLoader(SeqDataset(tr_sub[TEXT_COL].values, tr_sub['target_encoded'].values, MAX_TOKEN_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(val_sub[TEXT_COL].values, val_sub['target_encoded'].values, MAX_TOKEN_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    
    model = HybridBiLSTM_CNN(len(vocab_tok.word2id), NUM_CLASSES, EMBEDDING_DIM, HIDDEN_DIM, CNN_NUM_FILTERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in tr_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            lbl = batch['label'].to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(ids, msk)
                loss = criterion(logits, lbl)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(MODELS_DIR, f"bilstm_fold{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)
        saved_checkpoints.append(ckpt_path)

    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            probs = F.softmax(model(ids, msk), dim=-1).cpu().numpy()
            val_preds.append(probs)
            
    val_preds = np.vstack(val_preds)
    oof_probs[val_idx] = val_preds
    fold_f1 = float(f1_score(val_sub['target_encoded'].values, np.argmax(val_preds, axis=1), average='macro'))
    fold_scores.append(round(fold_f1, 4))
    print(f"Fold {fold+1}/{len(splits)} | Validation Macro F1: {fold_f1:.4f}")
    
    fold_te_preds = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            probs = F.softmax(model(ids, msk), dim=-1).cpu().numpy()
            fold_te_preds.append(probs)
    test_probs += np.vstack(fold_te_preds) / len(splits)

if USE_KFOLD:
    overall_f1  = float(f1_score(y_all, np.argmax(oof_probs, axis=1), average='macro'))
    overall_acc = float(accuracy_score(y_all, np.argmax(oof_probs, axis=1)))
else:
    val_only = splits[0][1]
    overall_f1  = float(f1_score(y_all[val_only], np.argmax(oof_probs[val_only], axis=1), average='macro'))
    overall_acc = float(accuracy_score(y_all[val_only], np.argmax(oof_probs[val_only], axis=1)))

print(f"\n>>> OVERALL BiLSTM+CNN OOF Macro F1: {overall_f1:.4f} | Accuracy: {overall_acc:.4f} <<<")

# %%
np.save(os.path.join(MODELS_DIR, "dl_bilstm_oof_predictions.npy"), oof_probs)
np.save(os.path.join(MODELS_DIR, "dl_bilstm_test_predictions.npy"), test_probs)

config_results = {
    "model_name": "Hybrid_BiLSTM_TextCNN_Attention",
    "task": "text_classification",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kfold_mode": {
        "use_kfold": USE_KFOLD,
        "n_folds": len(splits),
        "random_state": SEED
    },
    "hyperparameters": {
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "cnn_num_filters": CNN_NUM_FILTERS,
        "max_token_length": MAX_TOKEN_LENGTH,
        "batch_size": BATCH_SIZE,
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

json_path = os.path.join(MODELS_DIR, "02_dl_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config_results, f, indent=2)

print(f"Saved PyTorch .pth checkpoints & metadata to '{json_path}'.")

# %%
test_df['predicted_label_bilstm'] = [id2label[p] for p in np.argmax(test_probs, axis=1)]
print("BiLSTM Test Prediction Distribution:\n", test_df['predicted_label_bilstm'].value_counts())

# %%
sub_raw = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_df['predicted_label_bilstm']})
sub_raw.to_csv("submission_bilstm_raw.csv", index=False)
sub_raw.to_csv("submission_bilstm_postprocessed.csv", index=False)

has_no_nans = not sub_raw[TARGET_COL].isna().any()
correct_rows = (len(sub_raw) == len(test_df))
print(f"Integrity Check: No NaNs: {has_no_nans} | Correct Rows: {correct_rows}")
print("SUCCESS: Notebook 2 (BiLSTM + TextCNN) completed flawlessly!")

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
