# %% [markdown]
# # 🎯 Task-Specific EDA: Continuous & Ordinal Text Regression
# ## Target Distribution, Skewness, Length Correlation & Optimization Metric Diagnostics
# **Kit:** `01_NLU/02_text_regression` | **Standard:** GEMASTIK XIX 2026 & Kaggle NLP  
# **Focus:** Continuous values, essay grading (AES), readability index, sentiment scores, and ordinal metrics (RMSE, MAE, QWK).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

train_path = '../demo_data/train.csv'
test_path  = '../demo_data/test.csv'

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
train_df.head()

# %%
# Detect target column (numeric column not in test)
candidate_targets = [col for col in train_df.columns if col not in test_df.columns and pd.api.types.is_numeric_dtype(train_df[col])]
TARGET_COL = candidate_targets[0] if candidate_targets else 'score'
print(f"Detected Target Column: '{TARGET_COL}'")

target = train_df[TARGET_COL].dropna()
mean_val = target.mean()
median_val = target.median()
std_val = target.std()
skew_val = target.skew()
kurt_val = target.kurtosis()

print("--- TARGET DESCRIPTIVE STATISTICS ---")
print(f"Count    : {len(target)}")
print(f"Mean     : {mean_val:.4f}")
print(f"Median   : {median_val:.4f}")
print(f"Std Dev  : {std_val:.4f}")
print(f"Min / Max: {target.min():.4f} / {target.max():.4f}")
print(f"Skewness : {skew_val:.4f} (Rule: |skew| > 1.0 indicates severe skew)")
print(f"Kurtosis : {kurt_val:.4f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# 1. Raw Target Distribution
sns.histplot(target, kde=True, ax=axes[0], color='royalblue', bins=15)
axes[0].axvline(mean_val, color='red', linestyle='--', label=f'Mean ({mean_val:.2f})')
axes[0].axvline(median_val, color='green', linestyle=':', label=f'Median ({median_val:.2f})')
axes[0].set_title(f"Raw Target (Skew: {skew_val:.2f})")
axes[0].legend()

# 2. Log1p Transformation
if (target >= 0).all():
    log_target = np.log1p(target)
    sns.histplot(log_target, kde=True, ax=axes[1], color='darkorange', bins=15)
    axes[1].set_title(f"log1p(y) Transform (Skew: {log_target.skew():.2f})")
else:
    axes[1].text(0.5, 0.5, "Negative values present\n(log1p skipped)", ha='center')
    axes[1].set_title("log1p(y) (N/A)")

# 3. QQ-Plot for Normality
stats.probplot(target, dist="norm", plot=axes[2])
axes[2].set_title("Normal Q-Q Plot")

plt.tight_layout()
plt.show()

# %%
# Check if target is discrete or integer-stepped (common in essay scoring 1-6)
is_integer_stepped = (target.round() == target).all()
unique_steps = sorted(target.unique())

print(f"Is Target Discrete/Integer-stepped? : {is_integer_stepped}")
print(f"Unique Target Values ({len(unique_steps)}) : {unique_steps}")

if is_integer_stepped and len(unique_steps) <= 10:
    print("\n🎯 DIAGNOSTIC: Target is DISCRETE / ORDINAL (e.g. Essay Grading 1-6).")
    print("-> Recommendation: Train regression with MSE / SmoothL1, then calibrate with Nelder-Mead QWK Cutter!")
else:
    print("\n🎯 DIAGNOSTIC: Target is CONTINUOUS (e.g. Readability / Sentiment continuous index).")
    print("-> Recommendation: Train regression with MSE / Huber Loss, apply Bounded Clipping Post-Processing.")

# %%
text_candidates = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != 'id']
TEXT_COL = text_candidates[0] if text_candidates else 'full_text'

train_df['word_count'] = train_df[TEXT_COL].astype(str).apply(lambda x: len(x.split()))
train_df['char_count'] = train_df[TEXT_COL].astype(str).apply(len)

pearson_wc = train_df['word_count'].corr(train_df[TARGET_COL], method='pearson')
spearman_wc = train_df['word_count'].corr(train_df[TARGET_COL], method='spearman')

print(f"Text Column: '{TEXT_COL}'")
print(f"Word Count vs Target -> Pearson: {pearson_wc:.4f} | Spearman: {spearman_wc:.4f}")

plt.figure(figsize=(8, 4))
sns.regplot(data=train_df, x='word_count', y=TARGET_COL, scatter_kws={'alpha': 0.6}, line_kws={'color': 'crimson'})
plt.title(f"Word Count vs Target Score (Correlation: {pearson_wc:.2f})")
plt.xlabel("Word Count")
plt.ylabel(TARGET_COL)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

if abs(pearson_wc) > 0.4:
    print("🚨 HIGH LEAKAGE / HIGH SIGNAL: Text length is strongly correlated with target!")
    print("-> GBDT with text length features alone will provide a massive baseline boost!")

# %% [markdown]
# ### 📋 Master Decision Matrix (Text Regression):
# | Diagnostic Signal | Condition | Recommended Strategy |
# | :--- | :--- | :--- |
# | **Skewness** | $|\text{skew}| > 1.0$ | Apply `np.log1p(y)` target transformation during training, `np.expm1(p)` during inference. |
# | **Ordinal Steps** | Target $\in \{1, 2, 3, 4, 5, 6\}$ | Train continuous MSE/Huber, apply Nelder-Mead threshold cutter for QWK. |
# | **Length Correlation** | Pearson $|r| > 0.40$ | Heavily leverage surface text length features in GBDT baseline. |
# | **Loss Function** | Extreme outliers present | Switch from MSE to Huber Loss ($\delta=1.0$) to avoid gradient explosion. |
