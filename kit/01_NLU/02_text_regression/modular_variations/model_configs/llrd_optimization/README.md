# 📉 Model Config: Layerwise Learning Rate Decay (LLRD)

## Deskripsi & Rationale
Transformer terlatih (*pretrained*) telah memiliki pemahaman tata bahasa dan kosakata dasar yang sangat kuat pada layer-layer bawahnya (dekat embedding). Menyesuaikan seluruh layer dengan learning rate yang sama sering kali merusak (*catastrophic forgetting*) bobot-bobot representasi dasar tersebut.

LLRD menerapkan **penurunan laju pembelajaran geometrik** dari layer atas menuju layer bawah:
$$\text{LR}_k = \text{Base LR} \times \xi^{(N - k)}$$
di mana $\xi \approx 0.90$ adalah faktor peluruhan (*decay factor*), dan layer linear head menerima learning rate tertinggi.

## Cara Menggunakan
```python
from modular_variations.model_configs.llrd_optimization.llrd import get_llrd_optimizer_params
import torch.optim as optim

param_groups = get_llrd_optimizer_params(model, base_lr=2e-5, decay_rate=0.90)
optimizer = optim.AdamW(param_groups, weight_decay=0.01)
```

