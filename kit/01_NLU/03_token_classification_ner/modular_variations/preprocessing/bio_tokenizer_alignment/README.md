# 🔗 Subword-to-Word BIO Tokenizer Alignment

## Rationale
Pretrained models (BERT, RoBERTa, DeBERTa) use BPE/WordPiece tokenizers that split words into subwords (e.g., *"Kemenkes"* $\to$ `["Ke", "##men", "##kes"]`).
- If each subword is predicted independently, boundary confusion occurs.
- Standard convention: assign true entity label (`B-ORG`) to the *first* subword, and assign `ignore_index = -100` to subsequent subwords (`##men`, `##kes`).
