"""
=============================================================================
COMPETITION TRIAGE & STRATEGY ADVISOR (GEMASTIK XIX & KAGGLE STANDARDS)
=============================================================================
Inspects raw competition files, diagnoses granularity level (NLU vs NLG),
identifies target type, recommends the exact notebook path, audits GPU VRAM
(RTX 3050), and outputs a copy-paste ready Master User Config block.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Import Smart Universal Loader
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
from smart_loader import SmartUniversalDataLoader


def run_competition_triage(train_path: str, test_path: str, sub_path: str = None):
    print("=" * 75)
    print(" [GEMASTIK & KAGGLE AUTOMATIC COMPETITION TRIAGE & STRATEGY ADVISOR]")
    print("=" * 75)

    train_file = Path(train_path)
    test_file = Path(test_path)

    if not train_file.exists() or not test_file.exists():
        print(f"[Error] Files not found. Train: {train_file.exists()}, Test: {test_file.exists()}")
        return

    # Ingest with Smart Universal Loader
    bundle = SmartUniversalDataLoader.load_competition_bundle(
        train_path=train_file,
        test_path=test_file,
        sample_sub_path=sub_path
    )

    train_df = bundle["train_df"]
    test_df = bundle["test_df"]
    id_col = bundle["id_col"]
    text_col = bundle["text_col"]
    pair_col = bundle["pair_col"]
    group_col = bundle["group_col"]
    target_cols = bundle["target_cols"]
    meta_num = bundle["meta_num_cols"]
    meta_cat = bundle["meta_cat_cols"]

    # 1. Granularity & Task Diagnosis
    domain = "NLU"
    task_category = "Text Classification"
    task_type = "single_label"
    recommended_nb = "01_NLU/01_text_classification/notebooks/01_ml_gbdt_classification.ipynb"

    # Check Token Classification / NER (Level B)
    is_token_level = False
    for col in target_cols:
        sample_vals = train_df[col].dropna().astype(str).head(20).tolist()
        if any("B-" in val or "I-" in val for val in sample_vals):
            is_token_level = True
            break
        if any("token" in c.lower() for c in train_df.columns) and any(kw in col.lower() for kw in ["label", "tag", "bio", "entity"]):
            is_token_level = True
            break

    if is_token_level:
        domain = "NLU"
        task_category = "Token Classification & NER"
        task_type = "token_classification"
        recommended_nb = "01_NLU/03_token_classification_ner/notebooks/01_token_classification_ner.ipynb"

    # Check Seq2Seq / Summarization (NLG Level C)
    elif target_cols and train_df[target_cols[0]].dtype == "object":
        avg_target_len = train_df[target_cols[0]].astype(str).str.split().str.len().mean()
        if avg_target_len > 6:
            domain = "NLG"
            task_category = "Text Summarization & Generation"
            task_type = "seq2seq_generation"
            recommended_nb = "02_NLG/01_text_summarization/notebooks/01_seq2seq_summarization.ipynb"

    # Check Whole-Text Regression vs Classification (NLU Level A)
    elif target_cols:
        s_target = train_df[target_cols[0]].dropna()
        if len(target_cols) > 1:
            # Multi-target
            if np.issubdtype(s_target.dtype, np.number):
                domain = "NLU"
                task_category = "Text Regression & Ranking"
                task_type = "multi_regression"
                recommended_nb = "01_NLU/02_text_regression/notebooks/01_ml_gbdt_regression.ipynb"
            else:
                domain = "NLU"
                task_category = "Text Classification"
                task_type = "multi_label"
                recommended_nb = "01_NLU/01_text_classification/notebooks/01_ml_gbdt_classification.ipynb"
        else:
            # Single target
            if np.issubdtype(s_target.dtype, np.floating):
                domain = "NLU"
                task_category = "Text Regression & Ranking"
                if s_target.min() >= 0.0 and s_target.max() <= 1.0 and len(s_target.unique()) > 10:
                    task_type = "bounded_regression"
                else:
                    task_type = "single_regression"
                recommended_nb = "01_NLU/02_text_regression/notebooks/01_ml_gbdt_regression.ipynb"
            elif np.issubdtype(s_target.dtype, np.integer):
                if 3 <= s_target.nunique() <= 10:
                    domain = "NLU"
                    task_category = "Text Regression & Ranking"
                    task_type = "ordinal_grading"
                    recommended_nb = "01_NLU/02_text_regression/notebooks/01_ml_gbdt_regression.ipynb"
                elif s_target.min() >= 0 and s_target.max() > 20:
                    domain = "NLU"
                    task_category = "Text Regression & Ranking"
                    task_type = "counting_volume"
                    recommended_nb = "01_NLU/02_text_regression/notebooks/01_ml_gbdt_regression.ipynb"
                else:
                    domain = "NLU"
                    task_category = "Text Classification"
                    task_type = "single_label"
                    recommended_nb = "01_NLU/01_text_classification/notebooks/01_ml_gbdt_classification.ipynb"
            else:
                domain = "NLU"
                task_category = "Text Classification"
                task_type = "single_label"
                recommended_nb = "01_NLU/01_text_classification/notebooks/01_ml_gbdt_classification.ipynb"

    # 2. Hardware Budget
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)) if cuda_available else 0

    if vram_mb >= 6000:
        recommended_batch = 16
        recommended_max_len = 512
    elif vram_mb >= 3800:
        recommended_batch = 8
        recommended_max_len = 384
    else:
        recommended_batch = 4
        recommended_max_len = 256

    print("\n" + "=" * 75)
    print(" [TRIAGE DIAGNOSTIC REPORT]")
    print("=" * 75)
    print(f" * Problem Domain:         {domain}")
    print(f" * Detected Task Category: {task_category}")
    print(f" * Specific Task Type:     '{task_type}'")
    print(f" * Recommended Notebook:   {recommended_nb}")
    print(f" * Target Columns:         {target_cols} (Count: {len(target_cols)})")
    print(f" * Primary Text Column:    '{text_col}'")
    print(f" * Pair Text Context:      '{pair_col}'")
    print(f" * Group / Leakage Column: '{group_col}'")
    if meta_num or meta_cat:
        print(f" * Tabular Metadata:       Numeric: {meta_num} | Categorical: {meta_cat}")
    print(f" * Hardware Environment:   {device_name} (VRAM: {vram_mb} MB)")
    print(f" * Recommended Batch Size: {recommended_batch} (Max Token Len: {recommended_max_len})")

    # 3. Ready-to-paste Master Config Block
    print("\n" + "=" * 75)
    print(" [COPY-PASTE READY MASTER USER CONFIG (CELL 1)]")
    print("=" * 75)
    print(f"""# =============================================================================
# MASTER USER CONFIGURATION (AUTO-GENERATED BY TRIAGE ADVISOR)
# =============================================================================
SMOKE_TEST = True             # True = 10 detik fast smoke run; False = Full 5-Fold
TASK_TYPE = '{task_type}'

TRAIN_CSV_PATH = r'{train_path}'
TEST_CSV_PATH = r'{test_path}'
SAMPLE_SUB_PATH = {repr(sub_path)}

ID_COL = '{id_col}'
TEXT_COL = '{text_col}'
TEXT_PAIR_COL = {repr(pair_col)}
TARGET_COLS = {target_cols}
GROUP_COL = {repr(group_col)}

BATCH_SIZE = {recommended_batch}
MAX_TOKEN_LENGTH = {recommended_max_len}
MIXED_PRECISION = True
N_SPLITS = 5
""")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition Triage & Strategy Advisor")
    parser.add_argument("--train", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--test", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--sub", type=str, default=None, help="Path to sample submission")
    args = parser.parse_args()

    run_competition_triage(args.train, args.test, args.sub)

