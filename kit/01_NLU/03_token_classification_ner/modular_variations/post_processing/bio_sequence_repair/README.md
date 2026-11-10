# 🔧 BIO Sequence Repair

## Rationale
Raw argmax predictions from sequence classifiers can occasionally output orphan `I-*` tags (e.g. `O` $\to$ `I-PER`). In competition scoring frameworks (`seqeval`), invalid spans are discarded, directly hurting Precision and F1. Converting orphan `I-*` to valid `B-*` guarantees syntactically correct spans.
