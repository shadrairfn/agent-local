---
name: notebook-converter
description: Two-way conversion between Jupyter Notebook (.ipynb) and Python script (.py) or Markdown (.md). Use this skill to convert .ipynb to .py for editing, inspect notebook contents, convert .py to .ipynb, or clear notebook outputs.
---

# Notebook Converter Skill (`notebook-converter`)

A robust, two-way conversion tool between Jupyter Notebooks (`.ipynb`) and Python scripts (`.py`, Percent Format `# %%`) or Markdown (`.md`).

This skill solves the challenge of editing notebooks in AI environments (where direct edits to `.ipynb` are often restricted or prone to JSON schema corruption) by providing clean, lossless roundtrip conversions.

---

## 1. Quick CLI Usage

All conversion operations are handled by [convert.py](./scripts/convert.py) or the Windows shortcut [convert.bat](./scripts/convert.bat).

### A. Convert `.ipynb` to `.py` (Python Script)
Auto-detects `.ipynb` and converts to percent-formatted Python script:
```powershell
python .agents/skills/notebook-converter/scripts/convert.py notebook_submission_final.ipynb
# Output: notebook_submission_final.py
```

### B. Convert `.py` to `.ipynb` (Jupyter Notebook)
Auto-detects `.py` and reconstructs a Jupyter notebook with proper code & markdown cells:
```powershell
python .agents/skills/notebook-converter/scripts/convert.py script.py
# Output: script.ipynb
```

### C. Convert `.ipynb` to `.md` (Markdown)
Exports the notebook into a readable Markdown document with fenced python blocks:
```powershell
python .agents/skills/notebook-converter/scripts/convert.py notebook.ipynb --to md
# Output: notebook.md
```

### D. Convert `.md` to `.ipynb`
Builds a Jupyter Notebook from Markdown documentation:
```powershell
python .agents/skills/notebook-converter/scripts/convert.py README.md -o output.ipynb
```

### E. Clear Notebook Outputs
Strips large cell outputs, plots, and execution counts to keep git repositories clean:
```powershell
python .agents/skills/notebook-converter/scripts/convert.py notebook.ipynb --clear-outputs
```

---

## 2. Cell Syntax Standards (Percent Format `# %%`)

The converted Python script uses the standard **Percent Format** supported by VS Code Interactive Window, PyCharm, Spyder, and Jupytext:

### Code Cells
Start with `# %%` on its own line:
```python
# %%
import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")
print(df.shape)
```

### Markdown Cells
Start with `# %% [markdown]` or `# %% [md]` on its own line, with markdown lines commented:
```python
# %% [markdown]
# # Section 1: Exploratory Data Analysis
#
# Here we analyze the distribution of ticket sales over time.
# - Feature 1: D1..D3
# - Target: D4..D10
```

### Raw Cells
Start with `# %% [raw]`:
```python
# %% [raw]
# Raw unformatted text block
```

---

## 3. Best Practice: Agent Workflow for Editing Notebooks

Because `.ipynb` files are complex JSON files with strict schema and cell ID requirements:

```mermaid
graph LR
    A[".ipynb Notebook"] -->|convert.py| B[".py Script (# %%)"]
    B -->|view / edit code| C["Modified .py"]
    C -->|convert.py| D["Updated .ipynb Notebook"]
```

1. **Convert to Python**:
   ```powershell
   python .agents/skills/notebook-converter/scripts/convert.py notebook_submission_final.ipynb
   ```
2. **Inspect & Edit**:
   Inspect `notebook_submission_final.py` using `view_file` and edit code or markdown using `replace_file_content` or `write_to_file`.
3. **Rebuild Notebook**:
   ```powershell
   python .agents/skills/notebook-converter/scripts/convert.py notebook_submission_final.py -o notebook_submission_final.ipynb
   ```
4. **Clean up**:
   Remove the temporary `.py` file if no longer needed, or keep it tracked in version control for cleaner git diffs.

---

## 4. CLI Arguments Reference

| Argument | Description |
| :--- | :--- |
| `input_file` | Path to source file (`.ipynb`, `.py`, or `.md`) |
| `-o, --output` | Custom destination file path |
| `--to {ipynb,py,md}` | Explicit target format |
| `--clear-outputs` | Clear cell execution counts and output blobs |
| `--kernel` | Name of Jupyter kernel (default: `python3`) |
