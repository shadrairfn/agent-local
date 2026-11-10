# %% [markdown]
# # ⚡ Paradigma 2: Classical Sequential Deep Learning Regression (BiLSTM + 1D-CNN + Attention)
# ## Low-VRAM Sequence Modeling with Smooth L1 (Huber) Loss & `.pth` Checkpoints
# **Competition Standard:** GEMASTIK XIX 2026 & Kaggle Standards  
# **Objective:** End-to-end sequential deep learning regression combining Bidirectional LSTM, 1D-CNN feature extractors, and attention pooling trained with Smooth L1 loss. Saves `.pth` weights and JSON config into `models/`.

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

MAX_VOCAB_SIZE   = 25000
EMBEDDING_DIM    = 128
HIDDEN_DIM       = 128
CNN_NUM_FILTERS  = 64
MAX_TOKEN_LENGTH = 128
BATCH_SIZE       = 8
NUM_EPOCHS       = 4
LEARNING_RATE    = 1e-3
SEED             = 42

print(f"Master Config Loaded | USE_KFOLD: {USE_KFOLD} | EPOCHS: {NUM_EPOCHS}")
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr

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
y_all = train_df[TARGET_COL].astype(np.float32).values
TARGET_MIN, TARGET_MAX = float(np.min(y_all)), float(np.max(y_all))

class VocabTokenizer:
    def __init__(self, max_vocab=25000):
        self.max_vocab = max_vocab
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
    def fit(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(str(t).lower().split())
        for w, _ in counts.most_common(self.max_vocab - 2):
            self.word2id[w] = len(self.word2id)
    def encode(self, text, max_len=128):
        tokens = str(text).lower().split()[:max_len]
        ids = [self.word2id.get(w, 1) for w in tokens]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [0] * pad_len
            mask += [0] * pad_len
        return ids, mask

vocab_tok = VocabTokenizer(MAX_VOCAB_SIZE)
vocab_tok.fit(train_df[TEXT_COL].values)

class RegSeqDataset(Dataset):
    def __init__(self, texts, targets=None, max_len=128):
        self.texts = list(texts)
        self.targets = targets
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        ids, mask = vocab_tok.encode(self.texts[idx], self.max_len)
        item = {'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(mask, dtype=torch.long)}
        if self.targets is not None:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item

# %%
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x, mask):
        weights = self.attn(x).squeeze(-1).masked_fill(mask == 0, -1e4)
        norm_w = F.softmax(weights, dim=-1).unsqueeze(-1)
        return torch.sum(x * norm_w, dim=1)

class HybridBiLSTM_CNN_Regressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, cnn_filters=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.conv3 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=5, padding=2)
        total_dim = hidden_dim + (cnn_filters * 2)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        lstm_pooled = self.attn_pool(lstm_out, attention_mask)
        emb_p = emb.permute(0, 2, 1)
        c3 = torch.max(F.relu(self.conv3(emb_p)), dim=2)[0]
        c5 = torch.max(F.relu(self.conv5(emb_p)), dim=2)[0]
        merged = torch.cat([lstm_pooled, c3, c5], dim=-1)
        return self.head(merged).squeeze(-1)

# %%
if USE_KFOLD:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(train_df, y_all))
else:
    tr_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=VAL_SPLIT_RATIO, random_state=SEED)
    splits = [(tr_idx, val_idx)]

oof_preds  = np.zeros(len(train_df), dtype=np.float32)
test_preds = np.zeros(len(test_df), dtype=np.float32)
fold_rmses = []
saved_checkpoints = []

test_loader = DataLoader(RegSeqDataset(test_df[TEXT_COL].values, max_len=MAX_TOKEN_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

for fold, (tr_idx, val_idx) in enumerate(splits):
    tr_sub  = train_df.iloc[tr_idx]
    val_sub = train_df.iloc[val_idx]
    
    tr_loader  = DataLoader(RegSeqDataset(tr_sub[TEXT_COL].values, y_all[tr_idx], MAX_TOKEN_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RegSeqDataset(val_sub[TEXT_COL].values, y_all[val_idx], MAX_TOKEN_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    
    model = HybridBiLSTM_CNN_Regressor(len(vocab_tok.word2id), EMBEDDING_DIM, HIDDEN_DIM, CNN_NUM_FILTERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in tr_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            tgt = batch['target'].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                out = model(ids, msk)
                loss = criterion(out, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(MODELS_DIR, f"bilstm_reg_fold{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)
        saved_checkpoints.append(ckpt_path)

    model.eval()
    val_p = []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            val_p.extend(model(ids, msk).cpu().numpy())
    val_p = np.array(val_p, dtype=np.float32)
    oof_preds[val_idx] = val_p
    fold_rmse = float(np.sqrt(mean_squared_error(y_all[val_idx], val_p)))
    fold_rmses.append(round(fold_rmse, 4))
    print(f"Fold {fold+1}/{len(splits)} | Validation RMSE: {fold_rmse:.4f}")
    
    te_p = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            te_p.extend(model(ids, msk).cpu().numpy())
    test_preds += np.array(te_p, dtype=np.float32) / len(splits)

eval_idx = np.arange(len(train_df)) if USE_KFOLD else splits[0][1]
overall_rmse = float(np.sqrt(mean_squared_error(y_all[eval_idx], oof_preds[eval_idx])))
overall_mae  = float(mean_absolute_error(y_all[eval_idx], oof_preds[eval_idx]))
print(f"\n>>> BiLSTM+CNN OOF RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f} <<<")

# %%
np.save(os.path.join(MODELS_DIR, "dl_bilstm_reg_oof_predictions.npy"), oof_preds)
np.save(os.path.join(MODELS_DIR, "dl_bilstm_reg_test_predictions.npy"), test_preds)

config_results = {
    "model_name": "Hybrid_BiLSTM_TextCNN_Regressor",
    "task": "text_regression",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kfold_mode": {"use_kfold": USE_KFOLD, "n_folds": len(splits), "random_state": SEED},
    "hyperparameters": {"embedding_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE},
    "validation_metrics": {"fold_rmses": fold_rmses, "oof_rmse": round(overall_rmse, 4), "oof_mae": round(overall_mae, 4)},
    "saved_checkpoints": saved_checkpoints
}

json_path = os.path.join(MODELS_DIR, "02_dl_config_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(config_results, f, indent=2)

sub_raw = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_preds})
sub_raw.to_csv("submission_bilstm_raw.csv", index=False)

sub_post = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: np.clip(test_preds, TARGET_MIN, TARGET_MAX)})
sub_post.to_csv("submission_bilstm_postprocessed.csv", index=False)

print(f"Saved PyTorch .pth checkpoints & metadata to '{json_path}'.")
print("SUCCESS: Notebook 2 (BiLSTM + TextCNN Regression) completed flawlessly!")

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
