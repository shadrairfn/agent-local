# Notebook Workflow Rule

When handling Jupyter Notebook (`.ipynb`) files in this repository:
1. Do not attempt direct JSON edits on `.ipynb` files.
2. Use the `notebook-converter` skill to convert `.ipynb` to Python percent format (`.py`).
3. Make changes to the `.py` script, then convert back to `.ipynb` using `scripts/convert.py`.
4. Ensure valid schema generation with cell IDs.
