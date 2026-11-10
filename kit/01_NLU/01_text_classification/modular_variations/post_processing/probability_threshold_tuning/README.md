# 🎯 Probability Threshold Optimizer for Macro F1

## Why Argmax is Suboptimal
Standard `argmax(P)` assigns classes based on a 0.5 decision boundary. In imbalanced datasets evaluated by Macro F1:
- A rare class with 5% prevalence may never reach $P > 0.5$.
- Scaling rare class probabilities elevates their recall and significantly boosts overall Macro F1.
