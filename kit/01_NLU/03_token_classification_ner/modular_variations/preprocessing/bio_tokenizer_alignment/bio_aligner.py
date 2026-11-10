"""
Subword-to-Word BIO Label Aligner.
Aligns word-level tokens with subword tokenizer IDs (HuggingFace FastTokenizer).
"""
from typing import List, Dict, Any

def align_labels_with_subwords(
    words: List[str],
    labels: List[str],
    tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 512,
    ignore_index: int = -100
) -> Dict[str, Any]:
    """
    Tokenizes words using FastTokenizer and aligns label sequence.
    First subword of a token inherits the true label; trailing subwords receive ignore_index (-100).
    """
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    word_ids = tokenized.word_ids(batch_index=0)
    aligned_labels = []
    prev_word_id = None

    for w_id in word_ids:
        if w_id is None:
            aligned_labels.append(ignore_index)
        elif w_id != prev_word_id:
            # First subword: inherit true label
            tag = labels[w_id] if w_id < len(labels) else "O"
            aligned_labels.append(label2id.get(tag, 0))
        else:
            # Continuation subword: ignore in loss
            aligned_labels.append(ignore_index)
        prev_word_id = w_id

    tokenized["labels"] = aligned_labels
    return tokenized
