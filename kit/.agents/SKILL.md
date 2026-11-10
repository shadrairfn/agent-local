---
name: clean-code-ds-notebook
description: Build ML/data-science notebooks (NLP, CV, tabular, time series, Kaggle baselines) in the user's clean-code style: fixed 9 sections, no library wrappers, no assert, minimal config. Use instead of ds-notebook-structure.
---

# Clean-Code DS Notebook

Use this skill EVERY TIME the user asks for a notebook (.ipynb), a modelling pipeline, a competition baseline (Kaggle, GEMASTIK, etc.), an ML experiment, or a data-analysis template. It replaces `ds-notebook-structure`: the 9-section flow is the same, but the code style rules below are stricter and take priority over anything in that skill.

The notebook (markdown, comments, docstrings, printed messages, variable names) is written in **English**. Chat replies to the user stay in the user's language.

## Workflow

1. **Inspect the data first.** Read the head of train/test/sample_submission, check shapes, dtypes, the label column, and what the metric is. If no data is attached, generate a small demo dataset inside the notebook so every cell can still run.
2. **Pick the simplest sensible model** (TF-IDF + linear model, LightGBM on handcrafted features, a small CNN, etc.). The notebook is a clean baseline and template; the model can be upgraded later with the same structure.
3. **Build the notebook with a builder script** (nbformat; keep it in the scratchpad, not in the user's outputs). Use `r'''...'''` for code strings and never put three single quotes in the code.
4. **Put the data files next to the notebook** in a run folder, execute with `jupyter nbconvert --to notebook --execute --output` (same name, so outputs are stored), and scan every cell output for errors.
5. **Verify the style rules mechanically** (grep the code cells): zero `assert` lines, no `def read_*` wrappers, no `Path(` / `*_PATH` / `*_PARAMS` config constants, every section header present.
6. **Deliver**: copy the executed notebook to the outputs folder and send it. In the reply, state the main score (baseline vs model), the recommended submission file, and any deviation from the style rules with the reason. Do not recap every step.

## Fixed notebook structure (never skip, merge, or reorder)

First cell: title, objective, main metric, table of contents. Then a short "coding rules" markdown, imports, and the config cell. Then these sections, each starting with `## N. Name` plus 1-3 sentences of purpose, and ending with a short summary:

1. **Read Data**: `pd.read_csv` directly, shapes, head, dtypes, memory; printed sanity checks (ids unique, test columns = train columns minus label, submission ids = test ids, missing values).
2. **EDA**: label distribution, duplicates, group structure, lengths/sizes, train vs test comparison. End with a markdown cell **"EDA findings -> decisions"**.
3. **Preprocessing**: cleaning plus the validation split. One function per purpose (see rules). State why the split strategy fits the data (StratifiedKFold, GroupKFold / StratifiedGroupKFold when rows share a group such as an anchor, user, or patient, time-based split for time series).
4. **Feature Engineering**: one function per feature group; show the feature lists and final shapes.
5. **Modelling**: a no-training or simple baseline, then the main model through ONE training function that returns OOF predictions, fold-averaged test predictions, and feature importance. Save fold models to disk.
6. **Evaluation**: ONE evaluation function used for every model and every post-processing candidate. Per-fold metrics, mean and std, overall OOF, comparison with the baseline, error analysis (largest errors, per-group scores, feature importance or confusion matrix).
7. **Test Set Prediction with the Pure Model**: no clipping, rounding, thresholds, or rules. Save `submission_raw.csv` (format identical to sample_submission) and the raw predictions as `.npy`. Printed format checks.
8. **Post Processing**: one `apply_post_processing(predictions, method)` function. Candidates are chosen on OOF only, never on test. Keep a method only if it improves overall OOF by a minimum gain AND improves more than half of the folds; otherwise keep the raw predictions and say so honestly.
9. **Test Set Prediction after Post Processing**: apply the exact same function to the raw test predictions, save `submission_postprocessed.csv`, run the same checks, show how many predictions changed, then a FINAL SUMMARY cell (baseline OOF, pure model OOF, chosen post-processing, OOF after, recommended file) and a Conclusion cell with next steps.

## Code style rules (hard rules)

### 1. Never wrap a library function
If a library already does it, call it directly where it is needed.

```python
# BAD
def read_data(path_data):
    return pd.read_csv(path_data)

# GOOD
train_df = pd.read_csv('train.csv')
```
Write your own function only when it combines several steps for one purpose and is reused (typically so train and test are processed identically).

### 2. No `assert`
Sanity checks are printed as readable lines such as `print(f"Train ids are unique: {train_df['id'].is_unique}")`, so the notebook never stops halfway and the reader sees every result.

### 3. The config cell holds global settings only
Only `RANDOM_SEED`, `NUM_FOLDS`, and similar global switches, plus seeding of `random`, `numpy` (and `torch` if used) and `PYTHONHASHSEED`. Everything else is written where it is used:
- file names inside `pd.read_csv('train.csv')` and `to_csv('submission_raw.csv', index=False)`; mention in markdown that on Kaggle the strings become `/kaggle/input/...` paths
- model hyperparameters inside the constructor call in the training function (`lgb.LGBMRegressor(n_estimators=..., learning_rate=..., random_state=RANDOM_SEED)`), including early stopping rounds
- vectorizer / transformer settings inside the `TfidfVectorizer(...)` call
- column names as plain strings (`'anchor'`, `'score'`); lookup tables (e.g. code -> title dict) inside the function that uses them; thresholds as lowercase local variables in the cell that uses them
- folder creation with `os.makedirs('models', exist_ok=True)` inside the training function

No `DATA_DIR`, `TRAIN_PATH`, `*_PARAMS`, `*_COLUMN`, or `EARLY_STOPPING_ROUNDS` constants.

### 4. Feature types come from the data
```python
CATEGORICAL_FEATURE = train_model_input_df.select_dtypes(include=['object']).columns.to_list()
NUMERICAL_FEATURE = train_model_input_df.select_dtypes(include=['number']).columns.to_list()
ALL_FEATURE = NUMERICAL_FEATURE + CATEGORICAL_FEATURE
```
First drop the non-feature columns (ids, raw text, label, fold, duplicate readable columns). Use `include=['number']` instead of `['float']` when count or flag features are int, otherwise they are silently skipped; if the user insists on `['float']`, cast those features to float. Print both lists and check that every input column landed in exactly one group. Convert categorical columns to `category` AFTER selecting, and give the test set the same categories as train (`pd.Categorical(..., categories=train[col].cat.categories)`).

### 5. Split work by purpose
- **Preprocessing**: one function per purpose (for example `normalize_phrases`, `decode_context_code`, `create_validation_folds`). Never one big `preprocess()`.
- **Feature engineering**: one function per feature group (`add_length_features`, `add_word_overlap_features`, `add_fuzzy_similarity_features`, `add_tfidf_cosine_features`).
- **Training**: exactly ONE function (for example `train_lightgbm_with_cv`) that loops over folds and returns a dict with named keys (`oof_predictions`, `test_predictions`, `feature_importance_df`). No score printing inside it beyond progress lines.
- **Evaluation**: exactly ONE function (`evaluate_predictions(true, predicted, fold_ids, model_name)`) returning a table with per-fold, mean, std and overall OOF; reused for baseline, model, and each post-processing candidate.

### 6. Functions
Short docstring and type hints on every function. Inputs come in as arguments (reading global constants such as `NUM_FOLDS` and `RANDOM_SEED` is fine). Work on `dataframe.copy()` and return the new object; never modify the input in place. The same function handles train and test. Prefer returning a dict with named keys over a tuple of many items. After defining a function, test it on a tiny example or print before/after shape checks.

### 7. Naming
| What | Rule | Examples |
|---|---|---|
| Variables, functions | `snake_case`; functions start with a verb | `train_df`, `add_length_features()` |
| Global settings, feature lists | `UPPER_SNAKE_CASE` | `RANDOM_SEED`, `NUM_FOLDS`, `NUMERICAL_FEATURE` |
| DataFrames | end with `_df`; a NEW name for each processing step | `train_df` -> `train_df_normalized` -> `train_df_decoded` -> `train_df_with_folds` -> `train_features_df` |
| Booleans | start with `is_` / `should_` / `has_` | `is_valid_row`, `should_use_post_processing` |
| Prediction stages | prefix by stage | `raw_oof_predictions`, `raw_test_predictions`, `postprocessed_test_predictions` |
| Loop variables | descriptive | `fold_index`, `post_processing_method` |

Forbidden throwaway names: `a`, `b`, `df`, `df1`, `df2`, `tmp`, `res`, `x1`, `f()`, bare `i`/`j` in real loops. Conventional `X_train`, `y_train`, `X_valid`, `X_test` are allowed. Reusing a name inside a chain that only adds columns (`train_features_df = add_x(train_features_df)`) is fine and should be explained once in markdown.

### 8. Leakage and validation
- Anything that learns from data (vectorizers, scalers, encoders, categories, thresholds, post-processing choice) is fitted on train / OOF only.
- The test set is never used for fitting or tuning.
- Folds must respect the data's structure (group-aware or time-aware when needed); verify with a printed check that no group is split across folds.
- Report OOF predictions as the honest score; fold mean +- std next to overall OOF.

## Per-cell habits
- Define functions in one cell, call them in the next; the calling cell reads like a short story.
- Print `shape` and a sanity line after every important transformation.
- Plot with pandas / matplotlib directly; no plotting wrapper functions.
- Add a short markdown note when a naming trap exists in the dataset (for example a column called `target` that is not the label).
- If a code competition has a tiny public test.csv, say in Section 1 that every step must work for any number of test rows.

## Environment gotchas
- pandas 3: `select_dtypes(include=['object'])` still selects string columns but emits a deprecation warning; keep `warnings.filterwarnings('ignore')` in the imports cell. It works unchanged on pandas 2 (Kaggle).
- Install missing libraries with `pip install --break-system-packages` (nbformat, nbconvert, ipykernel, lightgbm, rapidfuzz, etc.).
- The shell often cannot reach arbitrary websites, so never rely on downloading data or pretrained weights inside the notebook unless the user supplies them.

## Checklist before delivering
1. All 9 section headers present and in order, each with purpose and summary.
2. Notebook ran top to bottom without errors and outputs are stored.
3. Zero `assert`; zero wrapper functions around library calls.
4. Config cell contains only global settings; paths, model settings, and column names are inline.
5. `CATEGORICAL_FEATURE` / `NUMERICAL_FEATURE` come from `select_dtypes` and cover every model input.
6. Preprocessing split by purpose; one training function; one evaluation function reused everywhere.
7. No leakage; post-processing decided on OOF only; `submission_raw.csv` and `submission_postprocessed.csv` both saved and format-checked.
8. All names descriptive; new name per processing step.
