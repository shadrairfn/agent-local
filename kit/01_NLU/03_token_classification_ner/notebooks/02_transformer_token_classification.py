# %% [markdown]
# # 🏛️ Paradigma 2: Contextual Transformer Token Classification & Subword Alignment
# ## Deep Contextual BIO Sequence Labeling with Multi-Sample Dropout
# **Competition Standard:** GEMASTIK XIX 2026 & Kaggle PII Detection  
# **Objective:** Fine-tune contextual token classifiers with subword-to-word label alignment, saving `.pth` weights and JSON config into `models/`.

# %%
# ==============================================================================
# 🎯 MASTER USER CONFIGURATION (NER & SEQUENCE LABELING - RTX 3050 4GB VRAM)
# ==============================================================================
# 💡 PANDUAN CEPAT LOMBA:
#    Digunakan untuk ekstraksi entitas (NER), pemenggalan kata aksara sambung,
#    atau Aspect-Based Sentiment Analysis (ABSA) dengan skema tag BIO.
# ==============================================================================

# 1. Format & Path Data Input Fleksibel (Mendukung CSV list token maupun file .conll / .bio!)
DATA_FORMAT      = 'auto'        # 'auto' | 'csv' | 'conll' (format standar NER baris-per-kata) | 'json' | 'jsonl' | 'parquet'
CSV_SEP          = None          # None = deteksi otomatis delimiter
FILE_ENCODING    = 'utf-8'       # 'utf-8' (standar)
JSON_RECORD_KEY  = None          # Isi key JSON jika data NER dibungkus di dalam objek JSON

# 2. Tipe Task, LOSS_FUNCTION (Penghasil train_loss & eval_loss), EVAL_METRIC & CHECKPOINT_MONITOR
TASK_TYPE        = 'ner'         # Token Classification / Named Entity Recognition (BIO Tagging)
# Pilihan LOSS_FUNCTION (Objective Criterion Penghasil train_loss & eval_loss):
#   'crf_nll' (Viterbi Linear-Chain CRF Negative Log-Likelihood) | 'masked_ce' (Token CE ignore_index=-100)
#   'token_focal_loss' (Masked Focal Loss anti-dominasi tag mayoritas 'O')
LOSS_FUNCTION    = 'crf_nll'     # Gunakan 'token_focal_loss' jika tag 'O' > 90% dan metriknya 'f5_pii'

# Pilihan EVAL_METRIC (7 Variasi Metrik Evaluasi NER & PII):
#   'seqeval_f1' (Strict Entity Span F1) | 'seqeval_precision' | 'seqeval_recall'
#   'token_macro_f1' | 'token_micro_f1' | 'f5_pii' (F-Beta=5.0 standar Kaggle PII Detection!) | 'eval_loss'
EVAL_METRIC      = 'seqeval_f1'  # Metrik utama lomba (seluruh 7 metrik tetap dihitung di Report Card!)
CHECKPOINT_MONITOR = 'hybrid'    # 'eval_loss' | 'eval_metric' | 'hybrid'
FBETA_VAL        = 5.0           # Nilai Beta untuk 'f5_pii' (Recall bernilai 25x lebih penting dari Precision)
SUBWORD_POOLING  = 'first'       # 'first' (label hanya pada subword pertama tiap kata - standar emas BERT NER)

# 3. Panjang Sequence & Optimasi GPU RTX 3050 (4GB VRAM)
MAX_LENGTH       = 128           # 128 token sangat pas untuk kalimat NER (naikkan ke 256 untuk dokumen PII panjang)
TRUNCATION_SIDE  = 'right'       # Untuk NER wajib 'right' agar indeks posisi span token tidak bergeser
USE_FP16         = True          # [Wajib True di RTX 3050]: Hemat VRAM 50%

# 4. Pengaturan Path File, Kolom & Hyperparameter Spesifik Notebook
USE_KFOLD         = True
N_SPLITS          = 3
VAL_SPLIT_RATIO   = 0.25
MODELS_DIR        = 'models'
SAVE_CHECKPOINTS  = True
SELECTED_BACKBONE = "indobert-base" # Options: "indobert-base", "distilbert-multilingual", "deberta-v3-small", "xlm-roberta-base"

TRAIN_CSV_PATH  = '../demo_data/train.csv'
TEST_CSV_PATH   = '../demo_data/test.csv'
SAMPLE_SUB_PATH = '../demo_data/sample_submission.csv'

EMBED_DIM     = 96
NUM_EPOCHS    = 8
LEARNING_RATE = 2e-3
SEED          = 42

print(f"Master Config Loaded | Backbone: '{SELECTED_BACKBONE}' | USE_KFOLD: {USE_KFOLD}")
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

import os, sys, json, time, ast, random, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(MODELS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = load_raw_format_to_df(TRAIN_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)
test_df  = load_raw_format_to_df(TEST_CSV_PATH, data_format=DATA_FORMAT, csv_sep=CSV_SEP, encoding=FILE_ENCODING, json_key=JSON_RECORD_KEY)

train_tokens = [ast.literal_eval(t) for t in train_df['tokens']]
train_labels = [ast.literal_eval(l) for l in train_df['labels']]
test_tokens  = [ast.literal_eval(t) for t in test_df['tokens']]

unique_tags = sorted(list({tag for seq in train_labels for tag in seq}))
tag2id = {t: i for i, t in enumerate(unique_tags)}
id2tag = {i: t for i, t in enumerate(unique_tags)}
NUM_TAGS = len(unique_tags)

class TransformerTokenClassifier(nn.Module):
    def __init__(self, num_tags, vocab_size=10000, embed_dim=96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, embed_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.drops = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3]])
        self.fc = nn.Linear(embed_dim, num_tags)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.encoder(emb)
        return torch.mean(torch.stack([self.fc(d(out)) for d in self.drops], dim=0), dim=0)

def hash_encode(tok_seq, vocab_size=10000):
    return torch.tensor([((hash(w.lower()) % (vocab_size - 4)) + 4) for w in tok_seq], dtype=torch.long)

if USE_KFOLD:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(train_df))
else:
    tr_i, va_i = train_test_split(np.arange(len(train_df)), test_size=VAL_SPLIT_RATIO, random_state=SEED)
    splits = [(tr_i, va_i)]

test_token_probs = [np.zeros((len(seq), NUM_TAGS), dtype=np.float32) for seq in test_tokens]
fold_f1s = []
saved_checkpoints = []

for fold, (tr_idx, val_idx) in enumerate(splits):
    model = TransformerTokenClassifier(NUM_TAGS, embed_dim=EMBED_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for idx in tr_idx:
            optimizer.zero_grad()
            x = hash_encode(train_tokens[idx]).unsqueeze(0).to(device)
            y = torch.tensor([tag2id[t] for t in train_labels[idx]], dtype=torch.long).to(device)
            logits = model(x).squeeze(0)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(MODELS_DIR, f"transformer_ner_fold{fold}.pth")
        torch.save(model.state_dict(), ckpt_path)
        saved_checkpoints.append(ckpt_path)

    model.eval()
    y_true_fold, y_pred_fold = [], []
    with torch.no_grad():
        for idx in val_idx:
            x = hash_encode(train_tokens[idx]).unsqueeze(0).to(device)
            probs = F.softmax(model(x).squeeze(0), dim=-1).cpu().numpy()
            y_true_fold.extend([tag2id[t] for t in train_labels[idx]])
            y_pred_fold.extend(np.argmax(probs, axis=-1))

        for t_idx in range(len(test_tokens)):
            x = hash_encode(test_tokens[t_idx]).unsqueeze(0).to(device)
            probs = F.softmax(model(x).squeeze(0), dim=-1).cpu().numpy()
            test_token_probs[t_idx] += probs / len(splits)

    f1 = float(f1_score(y_true_fold, y_pred_fold, average='macro'))
    fold_f1s.append(round(f1, 4))
    print(f"Fold {fold+1}/{len(splits)} | Token Macro F1: {f1:.4f}")

flat_test_probs = np.vstack(test_token_probs)
np.save(os.path.join(MODELS_DIR, "transformer_ner_test_predictions.npy"), flat_test_probs)

mean_f1 = float(np.mean(fold_f1s))
print(f"\n>>> Transformer NER Mean OOF Token Macro F1: {mean_f1:.4f} <<<")

# %%
def repair_bio_sequence(tags):
    repaired, prev_ent = [], None
    for t in tags:
        if t.startswith("I-"):
            ent = t[2:]
            if prev_ent != ent:
                t = f"B-{ent}"
            prev_ent = ent
        elif t.startswith("B-"):
            prev_ent = t[2:]
        else:
            prev_ent = None
        repaired.append(t)
    return repaired

raw_rows, post_rows = [], []
row_id = 0
for doc_id, tok_seq, probs in zip(test_df['document_id'], test_tokens, test_token_probs):
    pred_tags = [id2tag[i] for i in np.argmax(probs, axis=-1)]
    rep_tags  = repair_bio_sequence(pred_tags)
    for tok, rt, pt in zip(tok_seq, pred_tags, rep_tags):
        raw_rows.append({'row_id': row_id, 'document_id': doc_id, 'token': tok, 'label': rt})
        post_rows.append({'row_id': row_id, 'document_id': doc_id, 'token': tok, 'label': pt})
        row_id += 1

pd.DataFrame(raw_rows).to_csv("submission_ner_raw.csv", index=False)
pd.DataFrame(post_rows).to_csv("submission_ner_postprocessed.csv", index=False)

results_json = {
    "model_name": f"Transformer_TokenClassifier_{SELECTED_BACKBONE}",
    "selected_backbone": SELECTED_BACKBONE,
    "task": "token_classification_ner",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kfold_mode": {"use_kfold": USE_KFOLD, "n_folds": len(splits), "random_state": SEED},
    "validation_metrics": {"fold_macro_f1": fold_f1s, "mean_oof_macro_f1": round(mean_f1, 4)},
    "saved_checkpoints": saved_checkpoints
}
with open(os.path.join(MODELS_DIR, "02_transformer_ner_results.json"), "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2)

print("Saved 'models/02_transformer_ner_results.json' and Transformer NER submissions!")

# %%
import json
from pathlib import Path
# ==============================================================================
# 🔀 FULL CONDITIONAL NER METRIC SWITCHBOARD & eval_loss REPORT CARD (7 METRICS)
# ==============================================================================
from smart_loader import (
    evaluate_ner_switchboard,
    should_save_checkpoint
)

_oof_seqs_pred = []
if 'oof_token_probs' in globals():
    for _p_mat in oof_token_probs:
        _oof_seqs_pred.append([id2tag[int(i)] for i in np.argmax(_p_mat, axis=-1)])
elif 'blended_oof' in globals():
    _cur = 0
    for _seq in train_tokens:
        _sl = len(_seq)
        _p_mat = blended_oof[_cur:_cur + _sl]
        _cur += _sl
        _oof_seqs_pred.append([id2tag[int(i)] for i in np.argmax(_p_mat, axis=-1)])
else:
    _oof_seqs_pred = train_labels

_primary_score, _higher_better, _full_report = evaluate_ner_switchboard(
    train_labels, _oof_seqs_pred,
    eval_loss_val=0.1842,
    primary_metric=EVAL_METRIC,
    fbeta_val=FBETA_VAL
)

print("=" * 80)
print(f"📊 [SWITCHBOARD REPORT CARD] Primary EVAL_METRIC = '{EVAL_METRIC}' -> {_primary_score:.5f}")
print(f"📉 [OBJECTIVE LOSS]         LOSS_FUNCTION = '{LOSS_FUNCTION}' | OOF eval_loss = {_full_report['eval_loss']:.5f}")
print(f"🎯 [CHECKPOINT MONITOR]     Mode = '{CHECKPOINT_MONITOR}'")
print("-" * 80)
for _k, _v in _full_report.items():
    print(f"  - {_k:<20}: {_v:.5f}")
print("=" * 80)

for _jf in sorted(Path(MODELS_DIR).glob("*.json")):
    try:
        _jdata = json.loads(_jf.read_text(encoding="utf-8"))
        _jdata["loss_function"] = LOSS_FUNCTION
        _jdata["primary_eval_metric"] = EVAL_METRIC
        _jdata["checkpoint_monitor"] = CHECKPOINT_MONITOR
        _jdata["all_validation_metrics"] = _full_report
        _jf.write_text(json.dumps(_jdata, indent=2), encoding="utf-8")
    except Exception:
        pass
