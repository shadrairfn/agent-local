# 🌡️ Temperature Scaling for Model Calibration

## Rationale
Modern deep neural networks often suffer from overconfidence ($P \approx 1.0$ even when uncertain).
Calibrating probabilities using learned temperature $T$:
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
ensures probabilities reflect true empirical frequencies, making ensembling with GBDTs substantially more effective.
