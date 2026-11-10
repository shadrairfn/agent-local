# ⚡ 00_QUICK_START: Competition Onboarding & Battle Protocol

Welcome to the **Quick Start** suite. This directory is your mission control center during the first 15 minutes of any NLP hackathon (GEMASTIK XIX 2026, SATRIA DATA, Kaggle NLP).

---

## 🛠️ Tooling & Scripts

### 1. `verify_environment.py` (Hardware & Library Health Check)
Run this immediately upon opening the competition environment (laptop or cloud VM):
```bash
python 00_QUICK_START/verify_environment.py
```
- **Audits GPU Hardware:** Confirms CUDA availability, GPU device name, and exact VRAM capacity (e.g. 4096 MB on RTX 3050).
- **Verifies Core Packages:** Checks `torch`, `transformers`, `datasets`, `lightgbm`, `scikit-learn`, `scipy`.
- **Outputs Execution Budget:** Recommends batch size (4-8) and mixed-precision flags.

### 2. `competition_triage.py` (10-Second Auto-Triage & Master Config)
Give it the train and test CSV files, and it will automatically diagnose the problem:
```bash
python 00_QUICK_START/competition_triage.py --train <path/to/train.csv> --test <path/to/test.csv>
```
- **Infers Problem Family:** Detects whether the challenge is NLU (Classification, Regression, NER) or NLG (Summarization, Translation).
- **Auto-Resolves Roles:** Finds ID column, primary text, target column(s), and group leakage columns (`prompt_id`, `author_id`).
- **Generates Cell 1 Code:** Outputs a complete copy-paste ready Master User Config cell formatted for the appropriate kit notebook.

### 3. `SOP_5_JAM_GEMASTIK.md` (Tactical 5-Hour Battle Plan)
A minute-by-minute operational playbook:
- **Phase 1 (00:00 - 00:20):** Diagnostic triage & 00_GENERAL_TEXT_EDA.
- **Phase 2 (00:20 - 00:45):** GBDT first baseline submission (leaderboard anchor).
- **Phase 3 (00:45 - 02:30):** Deep Transformer modeling with FP16 mixed precision.
- **Phase 4 (02:30 - 04:00):** Feature engineering & error analysis drilldown.
- **Phase 5 (04:00 - 05:00):** Blending ensemble, threshold tuning, and final verification.

### 4. `smart_loader.py` (Universal Resilient Ingestion)
The underlying automated column resolver that prevents hardcoded column errors across any tabular text dataset.

