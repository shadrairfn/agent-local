# ⚖️ Focal Loss for Imbalanced Datasets

## Overview
Standard Cross Entropy loss $\text{CE} = -\log(p_t)$ applies equal gradient pressure to easy and hard examples. In imbalanced competitions, majority class examples dominate the total gradient.

## Mathematical Formulation
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **$\gamma = 2.0$ (Focusing Parameter):** When an easy example has $p_t = 0.99$, $(1 - 0.99)^2 = 0.0001$, scaling down its loss by $10000\times$. Hard examples ($p_t = 0.2$) retain $(1 - 0.2)^2 = 0.64$ scale.
- **$\alpha$ (Class Balancing Weights):** Inverse class frequency tensor to balance rare classes.

## When to Use
- Class imbalance ratio $\ge 3:1$.
- Evaluation metric is Macro F1 or Balanced Accuracy.
