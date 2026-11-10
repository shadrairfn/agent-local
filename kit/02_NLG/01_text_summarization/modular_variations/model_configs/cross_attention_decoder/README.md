# 🎛️ Beam Search Generation Tuning

## Overview
Default greedy decoding (`num_beams=1`) frequently leads to repetitive hallucination or premature stopping.

## Key Hyperparameters:
- `num_beams = 4`: Balances sequence search quality with inference speed.
- `no_repeat_ngram_size = 3`: Strictly forbids trigram looping (e.g. *"pada saat itu pada saat itu"*).
- `length_penalty = 1.0`: Encourages models to produce sufficiently detailed summaries without truncation penalty.
