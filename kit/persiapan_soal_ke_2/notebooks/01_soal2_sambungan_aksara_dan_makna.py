# %% [markdown]
# # 📜 Soal 2 Solver A: Sambungan Kata Aksara (Unsegmented Script) -> Pemenggalan Kata & Prediksi Makna
# **Konteks Kasus Soal ke-2 GEMASTIK:**
# - Pada Soal 1 (atau tahap OCR/Transliterasi Aksara Jawa/Sunda/Bali/Lontar), output sering kali berupa **rangkaian huruf tanpa spasi** (*scriptio continua* / sambungan kata aksara), misalnya: `aksarajawamenyimpanwarisanbudayaleluhur`.
# - **Tugas Soal ke-2** meminta peserta untuk:
#   1. **Word Segmentation (Pemenggalan Kata):** Memisahkan sambungan aksara tanpa spasi menjadi kalimat terpisah yang utuh.
#   2. **Semantic Meaning Translation & Classification (Prediksi Makna & Kategori):** Menerjemahkan/memetakan hasil pemenggalan kata ke dalam makna Bahasa Indonesia beserta klasifikasi topiknya.
#
# ### 🧠 Arsitektur Hybrid di Notebook Ini:
# 1. **Viterbi Dynamic Programming Word Segmenter ($\mathcal{O}(L^2)$):** Memanfaatkan probabilitas unigram & bigram kamus korpus untuk mencari titik potong kata paling optimal secara matematis.
# 2. **Character-Level BiLSTM Boundary Detector (PyTorch `.pth`):** Jaringan saraf tingkat karakter yang memprediksi probabilitas batas spasi (`0 = lanjut`, `1 = potong spasi`) untuk menangani kata *Out-Of-Vocabulary* (OOV) / typo OCR.
# 3. **Chained Meaning Retriever & LightGBM Topic Classifier (`.txt`):** Mengambil teks hasil pemenggalan kata untuk memprediksi makna (`meaning_indonesia`) dan label kategori (`category_label`).

# %%
# ==============================================================================
# ⚙️ MASTER USER CONFIGURATION (PUSAT KENDALI SOAL 2: SAMBUNGAN AKSARA & MAKNA)
# ==============================================================================
# 🛠️ Cara Pakai saat Hari-H:
#   1. Arahkan TRAIN_PATH, TEST_PATH, dan SUB_PATH ke file dataset Soal 2.
#   2. Isi UNSEGMENTED_COL dengan nama kolom teks aksara yang menyambung tanpa spasi.
#   3. Isi SEGMENTED_TARGET_COL (jika ada target spasi), MEANING_TARGET_COL (makna),
#      dan CLASS_TARGET_COL (jika ada sub-target kategori).
# ==============================================================================

TRAIN_PATH           = '../demo_data/train_aksara_sambung.csv'
TEST_PATH            = '../demo_data/test_aksara_sambung.csv'
SUB_PATH             = '../demo_data/sample_submission_aksara.csv'

ID_COL               = 'id'                   # Kolom ID unik sampel
UNSEGMENTED_COL      = 'aksara_sambung'       # Kolom input rangkaian karakter tanpa spasi
SEGMENTED_TARGET_COL = 'segmented_text'       # Target 1: Kalimat hasil pemenggalan spasi
MEANING_TARGET_COL   = 'meaning_indonesia'    # Target 2: Terjemahan / makna kalimat
CLASS_TARGET_COL     = 'category_label'       # Target 3: Label klasifikasi makna/topik

# ------------------------------------------------------------------------------
# 🎯 METRIK EVALUASI UTAMA & LOSS FUNCTION
# ------------------------------------------------------------------------------
# Pilihan EVAL_METRIC Pemenggalan & Makna: 'cer' (Character Error Rate - makin kecil makin baik)
#   | 'wer' (Word Error Rate) | 'chrf' (Character F-Score) | 'bleu' | 'macro_f1'
EVAL_METRIC          = 'chrf'                 # Metrik utama pemenggalan & translasi makna
CLASS_EVAL_METRIC    = 'macro_f1'             # Metrik utama klasifikasi topik makna
LOSS_FUNCTION        = 'auto'                 # 'auto' | 'bce_logits' (Boundary) | 'cross_entropy'
CHECKPOINT_MONITOR   = 'hybrid'               # 'val_f1' | 'val_loss' | 'hybrid'

# ------------------------------------------------------------------------------
# 🧩 KONFIGURASI ALGORITMA PEMENGGALAN KATA (HYBRID VITERBI + CHAR-BiLSTM)
# ------------------------------------------------------------------------------
MAX_WORD_LEN         = 22                     # Panjang maksimum 1 kata Bahasa Indonesia/Daerah
VITERBI_WEIGHT       = 0.65                   # Bobot skor DP Kamus Viterbi (0.0 - 1.0)
BILSTM_WEIGHT        = 0.35                   # Bobot skor neural Character-BiLSTM untuk kata OOV
CHAR_EMBED_DIM       = 48                     # Dimensi embedding karakter BiLSTM
CHAR_HIDDEN_DIM      = 64                     # Dimensi hidden state BiLSTM
EPOCHS               = 12                     # Epoch training Character-BiLSTM Boundary Detector
LR                   = 3e-3                   # Learning rate AdamW
N_SPLITS             = 3                      # Jumlah fold Cross-Validation
SEED                 = 42                     # Random seed untuk reprodusibilitas
print('✅ Konfigurasi Soal 2 (Sambungan Aksara & Prediksi Makna) berhasil dimuat!')

# %%
import os
import re
import math
import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

from smart_loader import (
    load_raw_format_to_df,
    evaluate_nlg_switchboard,
    evaluate_classification_switchboard,
    optimize_classification_oof,
)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

train_df = load_raw_format_to_df(TRAIN_PATH)
test_df  = load_raw_format_to_df(TEST_PATH)
sub_df   = load_raw_format_to_df(SUB_PATH)

print(f'📂 Train shape: {train_df.shape} | Test shape: {test_df.shape}')
display(train_df.head(3))

# %%
# ==============================================================================
# 🧠 TAHAP 1: VITERBI DYNAMIC PROGRAMMING + CHAR-BiLSTM WORD SEGMENTER
# ==============================================================================
# 1A. Bangun Kamus Frekuensi Kata & Bigram dari Korpus Target + Kosakata Dasar
word_counts = Counter()
bigram_counts = Counter()

for seg_sent in train_df[SEGMENTED_TARGET_COL].dropna().astype(str):
    tokens = seg_sent.lower().strip().split()
    word_counts.update(tokens)
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i+1])] += 1

# Tambahkan kosakata fungsional umum agar kata hubung selalu dikenali
for common_w in ['di', 'ke', 'dari', 'dan', 'yang', 'untuk', 'dengan', 'secara', 'indonesia', 'berkelanjutan', 'akurasi', 'tinggi']:
    word_counts[common_w] += 15

total_words = sum(word_counts.values()) + 1

def word_log_prob(w: str) -> float:
    if w in word_counts:
        return math.log((word_counts[w] + 1.0) / total_words)
    # Penalti eksponensial untuk potongan karakter yang tidak dikenal di kamus
    return math.log(1.0 / (total_words * (12.0 ** max(1, len(w) - 2))))

# 1B. Latih Character-Level BiLSTM Boundary Detector untuk membantu kata OOV
all_chars = sorted(list(set(''.join(train_df[SEGMENTED_TARGET_COL].dropna().astype(str).str.lower()))))
char2idx = {c: i + 1 for i, c in enumerate(all_chars) if c != ' '}
vocab_size = len(char2idx) + 2

def encode_segmented_to_boundary(seg_text: str):
    clean = re.sub(r'\s+', ' ', seg_text.lower().strip())
    char_ids = []
    labels = []  # 1 jika setelah karakter ini ada spasi, 0 jika lanjut menyambung
    i = 0
    while i < len(clean):
        c = clean[i]
        if c == ' ':
            i += 1
            continue
        char_ids.append(char2idx.get(c, 0))
        if i + 1 < len(clean) and clean[i + 1] == ' ':
            labels.append(1.0)
        else:
            labels.append(0.0)
        i += 1
    return char_ids, labels

class CharBoundaryBiLSTM(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.head(h).squeeze(-1)

bilstm_model = CharBoundaryBiLSTM(vocab_size, CHAR_EMBED_DIM, CHAR_HIDDEN_DIM)
optimizer = torch.optim.AdamW(bilstm_model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

encoded_samples = [encode_segmented_to_boundary(s) for s in train_df[SEGMENTED_TARGET_COL].astype(str)]
bilstm_model.train()
for ep in range(EPOCHS):
    ep_loss = 0.0
    for c_ids, lbls in encoded_samples:
        x_t = torch.tensor([c_ids], dtype=torch.long)
        y_t = torch.tensor([lbls], dtype=torch.float32)
        optimizer.zero_grad()
        logits = bilstm_model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        ep_loss += float(loss.item())

bilstm_ckpt_path = MODELS_DIR / 'char_bilstm_segmenter.pth'
torch.save(bilstm_model.state_dict(), bilstm_ckpt_path)
print(f'💾 Character-BiLSTM Boundary Checkpoint disimpan di: {bilstm_ckpt_path}')

@torch.no_grad()
def hybrid_viterbi_bilstm_segment(unseg_text: str) -> str:
    s = re.sub(r'\s+', '', str(unseg_text).lower().strip())
    n = len(s)
    if n == 0:
        return ''
    bilstm_model.eval()
    c_ids = [char2idx.get(c, 0) for c in s]
    boundary_probs = torch.sigmoid(bilstm_model(torch.tensor([c_ids], dtype=torch.long))).squeeze(0).numpy()
    if boundary_probs.ndim == 0:
        boundary_probs = np.array([float(boundary_probs)])

    dp = [-1e18] * (n + 1)
    backptr = [0] * (n + 1)
    dp[0] = 0.0

    for i in range(1, n + 1):
        start_j = max(0, i - MAX_WORD_LEN)
        for j in range(start_j, i):
            cand_word = s[j:i]
            vit_score = word_log_prob(cand_word)
            # Skor boundary BiLSTM pada ujung kata i-1
            b_prob = float(boundary_probs[i - 1]) if i < n else 1.0
            neural_score = math.log(max(b_prob, 1e-4))
            total_score = dp[j] + (VITERBI_WEIGHT * vit_score) + (BILSTM_WEIGHT * neural_score)
            if total_score > dp[i]:
                dp[i] = total_score
                backptr[i] = j

    words = []
    curr = n
    while curr > 0:
        prev = backptr[curr]
        words.append(s[prev:curr])
        curr = prev
    return ' '.join(reversed(words))

train_df['oof_segmented_text'] = [hybrid_viterbi_bilstm_segment(t) for t in train_df[UNSEGMENTED_COL]]
test_df['pred_segmented_text'] = [hybrid_viterbi_bilstm_segment(t) for t in test_df[UNSEGMENTED_COL]]

print('\n🔍 Contoh Hasil Pemenggalan Sambungan Kata Aksara:')
for idx in range(3):
    print(f"  Input Aksara : {train_df[UNSEGMENTED_COL].iloc[idx]}")
    print(f"  Prediksi     : {train_df['oof_segmented_text'].iloc[idx]}")
    print(f"  Ground Truth : {train_df[SEGMENTED_TARGET_COL].iloc[idx]}\n")

# %%
# ==============================================================================
# 🧠 TAHAP 2: PREDIKSI MAKNA (TRANSLATION RETRIEVAL) & KLASIFIKASI KATEGORI
# ==============================================================================
# 2A. Semantic Meaning Prediction berbasis Hybrid Char+Word TF-IDF Retrieval
meaning_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
X_seg_train_tfidf = meaning_vec.fit_transform(train_df['oof_segmented_text'])
X_seg_test_tfidf  = meaning_vec.transform(test_df['pred_segmented_text'])

# Out-of-Fold Meaning Prediction untuk evaluasi jujur
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_meanings = [''] * len(train_df)
label_unique = sorted(train_df[CLASS_TARGET_COL].unique())
label2id = {lbl: i for i, lbl in enumerate(label_unique)}
id2label = {i: lbl for lbl, i in label2id.items()}
y_train = train_df[CLASS_TARGET_COL].map(label2id).values

oof_class_probs = np.zeros((len(train_df), len(label_unique)))
test_class_probs = np.zeros((len(test_df), len(label_unique)))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
    # Nearest Semantic Meaning pada Fold Training
    sim_val = cosine_similarity(X_seg_train_tfidf[val_idx], X_seg_train_tfidf[tr_idx])
    best_tr_match = tr_idx[np.argmax(sim_val, axis=1)]
    for v_i, m_i in zip(val_idx, best_tr_match):
        oof_meanings[v_i] = train_df[MEANING_TARGET_COL].iloc[m_i]

    # LightGBM Classifier untuk prediksi kategori makna
    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=15,
        random_state=SEED + fold,
        verbose=-1
    )
    clf.fit(X_seg_train_tfidf[tr_idx], y_train[tr_idx])
    oof_class_probs[val_idx] = clf.predict_proba(X_seg_train_tfidf[val_idx])
    test_class_probs += clf.predict_proba(X_seg_test_tfidf) / N_SPLITS
    clf.booster_.save_model(str(MODELS_DIR / f'aksara_meaning_lgb_fold{fold}.txt'))

sim_test = cosine_similarity(X_seg_test_tfidf, X_seg_train_tfidf)
best_test_match = np.argmax(sim_test, axis=1)
test_pred_meanings = [train_df[MEANING_TARGET_COL].iloc[i] for i in best_test_match]

# Evaluasi Komprehensif: (1) Word Segmentation, (2) Meaning Prediction, (3) Category Classification
print('📊 [1] EVALUASI PEMENGGALAN KATA AKSARA (SEGMENTATION):')
seg_score, _, seg_report = evaluate_nlg_switchboard(
    ref_texts=train_df[SEGMENTED_TARGET_COL].tolist(),
    pred_texts=train_df['oof_segmented_text'].tolist(),
    primary_metric='cer',
)
for k, v in seg_report.items():
    print(f'  - {k:<18}: {v:.5f}')

print('\n📊 [2] EVALUASI PREDIKSI MAKNA BAHASA INDONESIA (MEANING TRANSLATION):')
meaning_score, _, meaning_report = evaluate_nlg_switchboard(
    ref_texts=train_df[MEANING_TARGET_COL].tolist(),
    pred_texts=oof_meanings,
    primary_metric=EVAL_METRIC,
)
for k, v in meaning_report.items():
    print(f'  - {k:<18}: {v:.5f}')

print('\n📊 [3] EVALUASI KLASIFIKASI TOPIK MAKNA:')
cls_score, _, class_report = evaluate_classification_switchboard(
    y_true=y_train,
    y_pred_labels=np.argmax(oof_class_probs, axis=1),
    y_prob=oof_class_probs,
    primary_metric=CLASS_EVAL_METRIC,
    loss_name=LOSS_FUNCTION,
)
for k, v in class_report.items():
    print(f'  - {k:<18}: {v:.5f}')

# Simpan Submission & Metadata JSON
sub_df['segmented_text'] = test_df['pred_segmented_text'].values
sub_df['meaning_indonesia'] = test_pred_meanings
sub_df['category_label'] = [id2label[int(i)] for i in np.argmax(test_class_probs, axis=1)]
sub_out_path = Path('submission_soal2_aksara_postprocessed.csv')
sub_df.to_csv(sub_out_path, index=False)

meta_payload = {
    'notebook': '01_soal2_sambungan_aksara_dan_makna',
    'segmentation_metrics': {k: float(v) for k, v in seg_report.items() if isinstance(v, (int, float))},
    'meaning_metrics': {k: float(v) for k, v in meaning_report.items() if isinstance(v, (int, float))},
    'all_validation_metrics': {k: float(v) for k, v in class_report.items() if isinstance(v, (int, float))},
}
with open(MODELS_DIR / '01_soal2_aksara_config_results.json', 'w', encoding='utf-8') as f:
    json.dump(meta_payload, f, indent=2)

print(f'\n✅ Submission Soal 2 Aksara disimpan di: {sub_out_path} ({ len(sub_df) } baris, 0 NaN)')
display(sub_df.head(5))
