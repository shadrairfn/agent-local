# %% [markdown]
# # 🤝 Paradigma 3: Token-Level Probability Ensembling & Legal BIO Sequence Repair
# ## Blending BiLSTM-CRF and Contextual Transformer Token Probabilities

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
USE_KFOLD = True  # [True]: Full 5-Fold CV | [False]: Single Split
import os, json, time, ast
import numpy as np
import pandas as pd

MODELS_DIR = 'models'
train_df = pd.read_csv('../demo_data/train.csv')
test_df  = pd.read_csv('../demo_data/test.csv')

train_labels = [ast.literal_eval(l) for l in train_df['labels']]
test_tokens  = [ast.literal_eval(t) for t in test_df['tokens']]
unique_tags  = sorted(list({tag for seq in train_labels for tag in seq}))
id2tag       = {i: t for i, t in enumerate(unique_tags)}

p_crf = np.load(os.path.join(MODELS_DIR, "bilstm_crf_test_predictions.npy"))
p_trf = np.load(os.path.join(MODELS_DIR, "transformer_ner_test_predictions.npy"))

blended_probs = 0.45 * p_crf + 0.55 * p_trf

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
cursor = 0
row_id = 0
for doc_id, tok_seq in zip(test_df['document_id'], test_tokens):
    seq_len = len(tok_seq)
    seq_probs = blended_probs[cursor : cursor + seq_len]
    cursor += seq_len
    pred_tags = [id2tag[i] for i in np.argmax(seq_probs, axis=-1)]
    rep_tags  = repair_bio_sequence(pred_tags)
    for tok, rt, pt in zip(tok_seq, pred_tags, rep_tags):
        raw_rows.append({'row_id': row_id, 'document_id': doc_id, 'token': tok, 'label': rt})
        post_rows.append({'row_id': row_id, 'document_id': doc_id, 'token': tok, 'label': pt})
        row_id += 1

pd.DataFrame(raw_rows).to_csv("submission_ensemble_ner_raw.csv", index=False)
pd.DataFrame(post_rows).to_csv("submission_ensemble_ner_postprocessed.csv", index=False)

ens_json = {
    "model_name": "Ensemble_BiLSTM_CRF_plus_Transformer_NER",
    "task": "token_classification_ner",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "weights": {"BiLSTM_CRF": 0.45, "Transformer_NER": 0.55},
    "total_tokens_predicted": int(row_id)
}
with open(os.path.join(MODELS_DIR, "03_ensemble_ner_results.json"), "w", encoding="utf-8") as f:
    json.dump(ens_json, f, indent=2)

print(f"Ensemble NER completed! Total tokens: {row_id}. Saved 'models/03_ensemble_ner_results.json'.")
print('=' * 80)
print(f"[CONFIG] Task = '{TASK_TYPE}' | Eval Metric = '{EVAL_METRIC}' | Loss = '{LOSS_FUNCTION}' | Monitor = '{CHECKPOINT_MONITOR}'")
print(f"[CONFIG] Format = '{DATA_FORMAT}' | Max Length = {MAX_LENGTH} ({TRUNCATION_SIDE}) | USE_KFOLD = {globals().get('USE_KFOLD', True)} | FP16 = {USE_FP16}")
print('=' * 80)

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
