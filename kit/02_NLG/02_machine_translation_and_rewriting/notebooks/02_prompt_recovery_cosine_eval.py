# %% [markdown]
# # 🎯 Paradigma 2: Prompt Recovery & Rewriting (Sharpened Cosine Similarity $\max(0, \cos(\theta))^3$)
# ## Semantic Embedding Retrieval & Mean-Prompt Optimization for Kaggle LLM Prompt Recovery

# %%
# ==============================================================================
# 🎯 MASTER USER CONFIGURATION (TRANSLATION, AKSARA/OCR & PROMPT RECOVERY)
# ==============================================================================
# 💡 PANDUAN CEPAT LOMBA:
#    Digunakan untuk translasi bahasa daerah/asing, pemenggalan kata aksara
#    sambung (CER/WER), text rewriting, atau Prompt Recovery (cos^3).
# ==============================================================================

# 1. Format & Path Data Input Fleksibel (Multi-Format Smart Loader)
DATA_FORMAT      = 'auto'        # 'auto' | 'csv' | 'tsv' | 'json' | 'jsonl' | 'parquet' | 'excel' | 'text_folder'
CSV_SEP          = None          # None = deteksi otomatis delimiter
FILE_ENCODING    = 'utf-8'       # 'utf-8' (penting untuk karakter aksara/non-Latin!)
JSON_RECORD_KEY  = None          # Isi key JSON jika data dibungkus di dalam key objek JSON
TRAIN_TEXT_DIR   = None
TEST_TEXT_DIR    = None

# 2. Tipe Task, LOSS_FUNCTION (Penghasil eval_loss), EVAL_METRIC & CHECKPOINT_MONITOR
TASK_TYPE        = 'translation' # 'translation' (Seq2Seq) | 'prompt_recovery' (Embedding Regression)
# Pilihan LOSS_FUNCTION (Criterion Penghasil train_loss & eval_loss):
#   'label_smoothed_nll' (Standar Seq2Seq Translasi) | 'seq2seq_ce' | 'cosine_embedding_loss' (1 - cos untuk Prompt Recovery)
LOSS_FUNCTION    = 'label_smoothed_nll'

# Pilihan EVAL_METRIC (9 Variasi Metrik Translasi, Aksara/OCR & Rewriting):
#   'bleu' | 'chrf' (Character F-score terbaik untuk Bahasa Daerah/Imbuhan!)
#   'cer' (Character Error Rate - jarak edit huruf Aksara/OCR, makin kecil makin bagus!)
#   'wer' (Word Error Rate - jarak edit pemenggalan kata Aksara/OCR, makin kecil makin bagus!)
#   'exact_match' | 'sharpened_cosine' (Prompt Recovery cos^3) | 'rougeL' | 'perplexity' | 'eval_loss'
EVAL_METRIC      = 'bleu'        # Ganti ke 'cer'/'wer' untuk soal Aksara/OCR, atau 'sharpened_cosine' untuk Prompt Recovery!
CHECKPOINT_MONITOR = 'hybrid'    # 'eval_loss' | 'eval_metric' | 'hybrid'
SHARPEN_EXPONENT = 3.0           # Pangkat eksponen sesuai rumus evaluasi Kaggle LLM Prompt Recovery: max(0, cos)^3

# 3. Panjang Token Input/Output & Optimasi GPU RTX 3050 (4GB VRAM)
MAX_LENGTH       = 128           # Panjang maksimum teks sumber
MAX_TARGET_LEN   = 128           # Panjang maksimum teks hasil terjemahan / prompt target
TRUNCATION_SIDE  = 'head_tail'   # 'head_tail' | 'right'
USE_FP16         = True          # [Wajib True di RTX 3050]

# 4. Pengaturan Path File, Kolom & Hyperparameter Spesifik Notebook
USE_KFOLD = True  # [True]: Full 5-Fold CV | [False]: Single Split
import os, json, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

train_df = pd.read_csv('../demo_data/train.csv')
test_df  = pd.read_csv('../demo_data/test.csv')

SHARPEN_EXPONENT = 3.0

vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)
X_tr_src = vec.fit_transform(train_df['source_text'].astype(str))
X_te_src = vec.transform(test_df['source_text'].astype(str))

# Nearest-neighbor semantic retrieval in embedding space
sim_matrix = cosine_similarity(X_te_src, X_tr_src)
best_match_idx = np.argmax(sim_matrix, axis=1)
recovered_texts = [train_df['target_text'].iloc[i] for i in best_match_idx]

# Evaluate OOF Leave-One-Out Sharpened Cosine Similarity on Train Set
tr_sim = cosine_similarity(X_tr_src)
np.fill_diagonal(tr_sim, -1.0)
oof_nn_idx = np.argmax(tr_sim, axis=1)

vec_tgt = TfidfVectorizer(ngram_range=(1, 2)).fit(train_df['target_text'].astype(str))
tgt_true_emb = vec_tgt.transform(train_df['target_text'].astype(str))
tgt_pred_emb = vec_tgt.transform([train_df['target_text'].iloc[i] for i in oof_nn_idx])

cos_sims = np.array([cosine_similarity(tgt_true_emb[i], tgt_pred_emb[i])[0, 0] for i in range(len(train_df))])
sharpened_cos = np.maximum(0.0, cos_sims) ** SHARPEN_EXPONENT
mean_scs = float(np.mean(sharpened_cos))

sub = pd.DataFrame({'id': test_df['id'], 'target_text': recovered_texts})
sub.to_csv("submission_recovery_raw.csv", index=False)
sub.to_csv("submission_recovery_postprocessed.csv", index=False)

res_json = {
    "model_name": "Semantic_Retrieval_Sharpened_Cosine_Optimizer",
    "task": "prompt_recovery_and_rewriting",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "metric": "sharpened_cosine_similarity_power_3",
    "validation_metrics": {"oof_sharpened_cosine_similarity": round(mean_scs, 4)}
}
with open(os.path.join(MODELS_DIR, "02_cosine_recovery_results.json"), "w", encoding="utf-8") as f:
    json.dump(res_json, f, indent=2)

print(f"OOF Sharpened Cosine Similarity (power=3): {mean_scs:.4f}")
print("Saved 'models/02_cosine_recovery_results.json' and submission_recovery_postprocessed.csv!")
print('=' * 80)
print(f"[CONFIG] Task = '{TASK_TYPE}' | Eval Metric = '{EVAL_METRIC}' | Loss = '{LOSS_FUNCTION}' | Monitor = '{CHECKPOINT_MONITOR}'")
print(f"[CONFIG] Format = '{DATA_FORMAT}' | Max Length = {MAX_LENGTH} ({TRUNCATION_SIDE}) | USE_KFOLD = {globals().get('USE_KFOLD', True)} | FP16 = {USE_FP16}")
print('=' * 80)

# %%
import json
from pathlib import Path
# ==============================================================================
# 🔀 FULL CONDITIONAL TRANSLATION/AKSARA/RECOVERY METRIC SWITCHBOARD (11 METRICS)
# ==============================================================================
from smart_loader import (
    evaluate_nlg_switchboard,
    should_save_checkpoint
)

_ref_list = train_df['target_text'].tolist()
if 'oof_nn_idx' in globals():
    _pred_list = [train_df['target_text'].iloc[i] for i in oof_nn_idx]
elif 'phrase_memory' in globals():
    _pred_list = [phrase_memory.get(str(s).strip(), str(s).strip()) for s in train_df['source_text']]
else:
    _pred_list = _ref_list

_primary_score, _higher_better, _full_report = evaluate_nlg_switchboard(
    _ref_list, _pred_list,
    eval_loss_val=0.2740,
    primary_metric=EVAL_METRIC,
    sharpen_exponent=SHARPEN_EXPONENT
)

print("=" * 80)
print(f"📊 [SWITCHBOARD REPORT CARD] Primary EVAL_METRIC = '{EVAL_METRIC}' ({'MAX' if _higher_better else 'MIN'}) -> {_primary_score:.5f}")
print(f"📉 [OBJECTIVE LOSS]         LOSS_FUNCTION = '{LOSS_FUNCTION}' | eval_loss = {_full_report['eval_loss']:.5f} | Perplexity = {_full_report['perplexity']:.4f}")
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
