"""
Beam Search & Generation Configuration Builder.
Generates optimal decoding parameters for Sequence-to-Sequence models (BART, T5, IndoBART).
"""
from typing import Dict, Any

def get_summarization_gen_kwargs(
    min_length: int = 15,
    max_length: int = 64,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3
) -> Dict[str, Any]:
    """Returns optimal keyword arguments for model.generate() in competitive summarization."""
    return {
        "min_length": min_length,
        "max_length": max_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": True
    }
