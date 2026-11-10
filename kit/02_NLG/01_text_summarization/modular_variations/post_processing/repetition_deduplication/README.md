# 🧹 Repetition Deduplication & Punctuation Polish

## Purpose
Post-processing generative text outputs:
1. Strips identical sentences generated when the decoder enters an attractor state.
2. Appends missing period/terminal punctuation so evaluation tokenizers calculate sentence boundaries accurately.
