# %% [markdown]
# # 📊 General Text EDA: Group Leakage & Duplicates Detection
# ### Universal Diagnostic for All Text Competitions (NLU & NLG)
# Tujuan: Mendeteksi teks duplikat dan kebocoran data kelompok (*group leakage* pada kolom prompt/author/topic) untuk memilih skema Cross-Validation yang benar.

# %%
# 1. Master Config
TRAIN_PATH = "demo_data/train.csv"
TEST_PATH = "demo_data/test.csv"
TEXT_COL = "full_text"
GROUP_CANDIDATES = ['prompt_id', 'author_id', 'topic_id', 'user_id', 'group_id']

# %%
# 2. Imports & Duplicate Detection
import pandas as pd
import hashlib

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# MD5 Hash for exact match
train_hashes = train_df[TEXT_COL].astype(str).str.strip().apply(lambda x: hashlib.md5(x.encode()).hexdigest())
test_hashes = test_df[TEXT_COL].astype(str).str.strip().apply(lambda x: hashlib.md5(x.encode()).hexdigest())

train_dups = train_hashes.duplicated().sum()
leak_dups = set(train_hashes).intersection(set(test_hashes))

print("=== DUPLICATE & LEAKAGE AUDIT ===")
print(f"Exact Duplicates inside Train: {train_dups}")
print(f"Exact Leaks between Train & Test: {len(leak_dups)}")

# %%
# 3. Group Column Leakage Audit
resolved_group = None
for col in GROUP_CANDIDATES:
    if col in train_df.columns:
        resolved_group = col
        break

if resolved_group:
    train_groups = set(train_df[resolved_group].dropna())
    test_groups = set(test_df[resolved_group].dropna()) if resolved_group in test_df.columns else set()
    overlap = train_groups.intersection(test_groups)
    
    print(f"\nDetected Group Column: '{resolved_group}'")
    print(f"Train Groups: {len(train_groups)}, Test Groups: {len(test_groups)}")
    print(f"Overlapping Groups in Train & Test: {len(overlap)}")
    if len(overlap) > 0:
        print(f"[WARNING] Group overlap detected! Standard KFold will cause optimistic leak.")
        print(f"   -> MUST USE GroupKFold(groups=df['{resolved_group}'])")
else:
    print("\nNo explicit group column detected. Standard StratifiedKFold is safe.")

# %% [markdown]
# ### 📋 Master Decision Matrix (Validation Strategy):
# 1. **Jika ditemukan `prompt_id` / `author_id` yang tumpang tindih:** **WAJIB gunakan `GroupKFold`**. KFold biasa akan menyebabkan model menghafal prompt/gaya penulis, menghasilkan skor CV tinggi palsu yang anjlok di leaderboard.
# 2. **Jika terdapat duplikasi teks identik di Train:** Hapus duplikasi jika label targetnya bertentangan (*label noise*).
# 3. **Jika sampel sepenuhnya independen:** Gunakan `StratifiedKFold` (untuk klasifikasi/ordinal) atau `KFold(shuffle=True)` (untuk regresi kontinu).
