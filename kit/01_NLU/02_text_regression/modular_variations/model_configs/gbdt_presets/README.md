# 🌲 Model Config: GBDT Presets (LightGBM & CatBoost)

## Deskripsi & Rationale
Menyediakan preset hyperparameter yang telah teruji (*battle-tested*) untuk model tabular Gradient Boosting (LightGBM dan CatBoost) yang disesuaikan secara otomatis dengan `task_type` aktif:
- `fast`: Eksekusi 30–60 detik untuk submission patokan (*leaderboard anchor*).
- `thorough`: 800–1000 iterasi dengan subsampling 0.8 dan regularisasi L1/L2 untuk model final ensembling.

## Cara Menggunakan
```python
from modular_variations.model_configs.gbdt_presets.gbdt_configs import get_lightgbm_preset
import lightgbm as lgb

params = get_lightgbm_preset(task_type='single_regression', mode='thorough')
model = lgb.LGBMRegressor(**params)
```

