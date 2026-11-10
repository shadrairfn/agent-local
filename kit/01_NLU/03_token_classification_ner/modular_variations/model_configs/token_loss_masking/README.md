# 🛡️ Masked Token Loss

## Rationale
Only genuine token boundaries should contribute to model gradients. Masking padding tokens (`PAD`) and continuation subwords with `-100` ensures that $> 90\%$ background zero tokens do not bias the sequence loss.
