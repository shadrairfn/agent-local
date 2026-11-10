"""
BIO Sequence Repair & Consistency Auditor.
Converts invalid sequence transitions (e.g., orphan I-* tags) into valid BIO spans.
"""
from typing import List

def repair_bio_sequence(tags: List[str]) -> List[str]:
    """
    Corrects illegal transitions in predicted BIO tags:
    - If I-Tag appears after O or after a different class B-Other, convert it to B-Tag.
    """
    repaired = []
    prev_tag = "O"

    for tag in tags:
        if tag.startswith("I-"):
            entity_type = tag[2:]
            valid_prev = {f"B-{entity_type}", f"I-{entity_type}"}
            if prev_tag not in valid_prev:
                # Convert orphan I-Tag into a new span opener B-Tag
                tag = f"B-{entity_type}"
        repaired.append(tag)
        prev_tag = tag

    return repaired
