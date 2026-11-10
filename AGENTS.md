# GEMASTIK NLP workspace (offline, no internet)

Never use web tools or `pip install` from the internet. Answer short. Read files with small line ranges.

## Layout
- `kit/` template kit. Start at `kit/README.md`, `kit/00_QUICK_START/SOP_5_JAM_GEMASTIK.md`, `kit/00_USER_CONFIG_DEEP_DIVE/00_PANIC_BUTTON_SAFE_DEFAULTS.md`.
- `kit/01_NLU/01_text_classification/notebooks/` 5 tiers: 01 sparse+GBDT (fast baseline), 02 BiLSTM, 03 transformer, 04 embeddings+GBDT, 05 ensemble.
- `kit/00_QUICK_START/competition_triage.py --train X --test Y` detects the task type. `smart_loader.py` loads data and metrics.
- `kit/persiapan_soal_ke_2/indonesian_regex_combinator.py` Indonesian text signals.
- `tools/check_submission.py SUB SAMPLE` validates a submission file.

## Rules
1. Never read or edit `.ipynb` files (huge JSON, reading is blocked). Every kit notebook has a ready `.py` twin with the same name next to it: copy that `.py` (e.g. `cp kit/01_NLU/01_text_classification/notebooks/01_ml_sparse_dense_gbdt.py uji_opencode/baseline_a.py`), edit the copy. To get a notebook back: `python kit/.agents/skills/notebook-converter/scripts/convert.py DEST.py -o DEST.ipynb`.
   Shell commands already start in the workspace root. NEVER pass a `workdir` or cwd parameter to the bash tool; leave it out. Use relative paths like `uji_opencode/file.py`. If a command fails, change it, never repeat it unchanged.
   Read `.py` files in ranges of at most 150 lines (the config cell is near the top).
2. Run a notebook/script with `SMOKE_TEST=1` first (small sample), then the full run.
3. Code style: no `assert` (print checks), no wrappers around library calls, one training function, one evaluation function, descriptive names, new DataFrame name per step.
4. Fit anything learned (vectorizers, thresholds) on train only. Split must respect duplicates and groups.
5. Always save `submission_raw.csv` first, then `submission_postprocessed.csv`; validate both with `tools/check_submission.py`.
6. Baselines MUST start from a kit template: convert the template notebook to .py into the task folder, copy `smart_loader.py` next to it, edit the Master User Config cell first, then adapt only the parts that do not fit the task (keep the template structure and its output files). Test on a small sample before the full run.
7. GPU contest PC: pretrained weights are in `kit/03_OFFLINE_WEIGHTS/weights/` (load with local path, offline env vars `HF_HUB_OFFLINE=1`).
