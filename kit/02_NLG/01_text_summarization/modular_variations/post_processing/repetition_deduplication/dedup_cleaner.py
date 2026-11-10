"""
Repetition Deduplication & Terminal Punctuation Repair.
Cleans raw Seq2Seq output strings to maximize ROUGE-L precision.
"""
import re

def clean_generated_summary(summary: str) -> str:
    """Removes duplicate consecutive sentences and ensures terminal punctuation."""
    if not isinstance(summary, str) or not summary.strip():
        return ""
    
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary.strip()) if s.strip()]
    deduped = []
    seen = set()

    for s in sentences:
        s_lower = s.lower()
        if s_lower not in seen:
            deduped.append(s)
            seen.add(s_lower)

    result = " ".join(deduped)
    # Ensure terminal punctuation
    if result and result[-1] not in {'.', '!', '?'}:
        result += '.'
    return result
