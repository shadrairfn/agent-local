# 🔍 Confusion Matrix & High-Confidence Error Drilldown

## Purpose
Post-training audit to find where the model fails most severely.

## Actions Based on Findings
- **High-confidence false predictions ($p > 0.9$ but wrong):** Often indicate mislabeled ground-truth in the competition training set or severe dataset sarcasm.
- **Class pair leakage (e.g. Class A frequently confused with Class B):** Indicates overlapping semantic definitions; consider feature engineering pairwise similarity or focal loss targeting that pair.
