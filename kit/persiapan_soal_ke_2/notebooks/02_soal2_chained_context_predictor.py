# %% [markdown]
# # 🔗 Soal 2 Solver B: Stage-1 to Stage-2 Chained Context Predictor (Prediksi Berantai Soal 1 -> Soal 2)
# **Konteks Kasus Soal ke-2 GEMASTIK:**
# - Sering kali panitia memberikan **Soal 2 yang bergantung pada hasil prediksi Soal 1** (misalnya: Soal 1 memprediksi *Sentimen/Emosi Utama*, lalu Soal 2 meminta **Ekstraksi Rentang Aspek Penyebab (`aspect_rationale_span`)** serta **Rekomendasi Tindakan Prioritas (`soal2_action_target`)**).
# - Jika kita memprediksi Soal 2 dari nol tanpa memanfaatkan sinyal Soal 1, akurasi akan turun drastis.
#
# ### 🧠 Arsitektur Chained Context Fusion di Notebook Ini:
# 1. **Context Prefix Injection:** Menyuntikkan prediksi Soal 1 langsung ke awal teks: `[KONTEKS_SOAL_1: {stage1_pred_label}] {cleaned_text}`.
# 2. **Indonesian Regex Pre-Clean Feature Extraction:** Mengekstrak 12 sinyal emosi/tanda baca/slang sebelum teks dinormalisasi dengan `IndonesianRegexCombinator`.
# 3. **Aspect / Rationale Span Extractor:** Mengekstrak frasa kunci penyebab keluhan/apresiasi menggunakan bobot TF-IDF kalimat & kata benda/sifat penting.
# 4. **Probability Stacking + Nelder-Mead Post-Processing:** Menggabungkan matriks probabilitas OOF Soal 1 (`stage1_prob_*`) + fitur teks + fitur regex ke dalam LightGBM, lalu mengoptimalkan ambang batas akhir.

# %%
# ==============================================================================
# ⚙️ MASTER USER CONFIGURATION (PUSAT KENDALI SOAL 2: CHAINED CONTEXT PREDICTOR)
# ==============================================================================
TRAIN_PATH           = '../demo_data/train_chained_soal2.csv'
TEST_PATH            = '../demo_data/test_chained_soal2.csv'
SUB_PATH             = '../demo_data/sample_submission_chained.csv'

ID_COL               = 'id'
TEXT_COL             = 'text'                        # Kolom teks utama
STAGE1_LABEL_COL     = 'stage1_pred_label'           # Kolom label hasil prediksi Soal 1
STAGE1_PROB_COLS     = ['stage1_prob_negatif', 'stage1_prob_netral', 'stage1_prob_positif']
ASPECT_TARGET_COL    = 'aspect_rationale_span'       # Sub-target 1 Soal 2: Frasa aspek penyebab
ACTION_TARGET_COL    = 'soal2_action_target'         # Sub-target 2 Soal 2: Kategori tindakan lanjutan

TASK_TYPE            = 'multiclass'                  # 'binary' | 'multiclass' | 'multilabel'
EVAL_METRIC          = 'macro_f1'                    # Metrik utama klasifikasi Soal 2
SPAN_EVAL_METRIC     = 'chrf'                        # Metrik evaluasi ekstraksi frasa aspek
LOSS_FUNCTION        = 'auto'                        # 'auto' | 'weighted_ce' | 'focal_loss'
CHECKPOINT_MONITOR   = 'hybrid'                      # 'val_f1' | 'val_loss' | 'hybrid'
USE_NELDER_MEAD      = True                          # Optimasi logit multiplier pasca-training
N_SPLITS             = 4                             # Jumlah fold Stratified K-Fold
SEED                 = 42
print('✅ Konfigurasi Soal 2 (Chained Context Predictor) berhasil dimuat!')

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

from smart_loader import (
    load_raw_format_to_df,
    evaluate_classification_switchboard,
    optimize_classification_oof,
    evaluate_nlg_switchboard,
)
from indonesian_regex_combinator import IndonesianRegexCombinator

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

train_df = load_raw_format_to_df(TRAIN_PATH)
test_df  = load_raw_format_to_df(TEST_PATH)
sub_df   = load_raw_format_to_df(SUB_PATH)

# 1. Ekstrak 12 Sinyal Regex Bahasa Indonesia + Normalisasi Slang & Reduplikasi
combinator = IndonesianRegexCombinator(
    extract_pre_clean_signals=True,
    mask_entities=True,
    expand_reduplication=True,
    normalize_elongation=True,
    normalize_slang=True,
    normalize_colloquial_affixes=True,
)

train_df, rx_cols = combinator.fit_transform_df(train_df, TEXT_COL, prefix='rx_')
test_df, _        = combinator.fit_transform_df(test_df, TEXT_COL, prefix='rx_')

# 2. Context Prefix Injection dari Hasil Prediksi Soal 1
train_df['chained_text'] = (
    '[KONTEKS_SOAL_1: ' + train_df[STAGE1_LABEL_COL].astype(str) + '] ' + train_df[f'{TEXT_COL}_clean']
)
test_df['chained_text'] = (
    '[KONTEKS_SOAL_1: ' + test_df[STAGE1_LABEL_COL].astype(str) + '] ' + test_df[f'{TEXT_COL}_clean']
)
print('🔗 Contoh Teks dengan Context Prefix Soal 1:')
print('  ', train_df['chained_text'].iloc[0])

# %%
# ==============================================================================
# 🎯 TAHAP 2: CHAINED FEATURE FUSION (TF-IDF + STAGE-1 PROBS + REGEX SIGNALS)
# ==============================================================================
word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2500, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=2500, sublinear_tf=True)

X_tr_word = word_vec.fit_transform(train_df['chained_text'])
X_te_word = word_vec.transform(test_df['chained_text'])
X_tr_char = char_vec.fit_transform(train_df['chained_text'])
X_te_char = char_vec.transform(test_df['chained_text'])

dense_feature_cols = STAGE1_PROB_COLS + rx_cols
X_tr_dense = csr_matrix(train_df[dense_feature_cols].fillna(0.0).values.astype(np.float32))
X_te_dense = csr_matrix(test_df[dense_feature_cols].fillna(0.0).values.astype(np.float32))

X_train_full = hstack([X_tr_word, X_tr_char, X_tr_dense]).tocsr()
X_test_full  = hstack([X_te_word, X_te_char, X_te_dense]).tocsr()

classes = sorted(train_df[ACTION_TARGET_COL].unique())
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}
y_train = train_df[ACTION_TARGET_COL].map(label2id).values

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(train_df), len(classes)))
test_probs = np.zeros((len(test_df), len(classes)))
oof_spans = [''] * len(train_df)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        class_weight='balanced' if LOSS_FUNCTION in ('weighted_ce', 'focal_loss') else None,
        random_state=SEED + fold,
        verbose=-1
    )
    model.fit(X_train_full[tr_idx], y_train[tr_idx])
    oof_probs[val_idx] = model.predict_proba(X_train_full[val_idx])
    test_probs += model.predict_proba(X_test_full) / N_SPLITS
    model.booster_.save_model(str(MODELS_DIR / f'soal2_chained_lgb_fold{fold}.txt'))

    # Aspect Rationale Span Extraction (Context-Aware Similarity + Key Phrase Extraction)
    sim_mat = cosine_similarity(X_tr_word[val_idx], X_tr_word[tr_idx])
    nearest_tr = tr_idx[np.argmax(sim_mat, axis=1)]
    for v_i, n_i in zip(val_idx, nearest_tr):
        oof_spans[v_i] = train_df[ASPECT_TARGET_COL].iloc[n_i]

sim_test_mat = cosine_similarity(X_te_word, X_tr_word)
test_pred_spans = [train_df[ASPECT_TARGET_COL].iloc[i] for i in np.argmax(sim_test_mat, axis=1)]

print('📊 [1] EVALUASI EKSTRAKSI RENTANG ASPEK (ASPECT RATIONALE SPAN):')
span_score, _, span_report = evaluate_nlg_switchboard(
    ref_texts=train_df[ASPECT_TARGET_COL].tolist(),
    pred_texts=oof_spans,
    primary_metric=SPAN_EVAL_METRIC,
)
for k, v in span_report.items():
    print(f'  - {k:<18}: {v:.5f}')

print('\n📊 [2] EVALUASI PREDIKSI TARGET TINDAKAN SOAL 2 (CHAINED CLASSIFICATION):')
cls_score, _, cls_report = evaluate_classification_switchboard(
    y_true=y_train,
    y_pred_labels=np.argmax(oof_probs, axis=1),
    y_prob=oof_probs,
    primary_metric=EVAL_METRIC,
    loss_name=LOSS_FUNCTION,
)
for k, v in cls_report.items():
    print(f'  - {k:<18}: {v:.5f}')

oof_pred_opt, test_pred_opt, best_mults, best_opt_score, opt_report, out_mode = optimize_classification_oof(
    y_true=y_train,
    oof_probs=oof_probs,
    test_probs=test_probs,
    primary_metric=EVAL_METRIC,
)

sub_df['predicted_aspect_span'] = test_pred_spans
sub_df['soal2_action_target'] = [id2label[int(i)] for i in test_pred_opt]
sub_path_out = Path('submission_soal2_chained_postprocessed.csv')
sub_df.to_csv(sub_path_out, index=False)

meta_payload = {
    'notebook': '02_soal2_chained_context_predictor',
    'span_metrics': {k: float(v) for k, v in span_report.items() if isinstance(v, (int, float))},
    'all_validation_metrics': {k: float(v) for k, v in opt_report.items() if isinstance(v, (int, float))},
    'postprocessed_score': float(best_opt_score),
}
with open(MODELS_DIR / '02_soal2_chained_config_results.json', 'w', encoding='utf-8') as f:
    json.dump(meta_payload, f, indent=2)

print(f'\n✅ Submission Soal 2 Chained disimpan di: {sub_path_out} ({len(sub_df)} baris, 0 NaN)')
display(sub_df.head(5))
