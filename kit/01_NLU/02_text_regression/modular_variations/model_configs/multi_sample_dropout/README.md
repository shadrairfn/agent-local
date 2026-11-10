# 🎲 Model Config: Multi-Sample Dropout (MSD)

## Deskripsi & Rationale
Multi-Sample Dropout (Inoue, 2019) adalah teknik regularisasi terbukti di podium Kaggle di mana output pooling dilewatkan melalui 5 layer dropout paralel dengan probabilitas berbeda ($p = [0.1, 0.2, 0.3, 0.4, 0.5]$). Prediksi dari seluruh jalur dirata-ratakan.

## Keuntungan Teknis:
1. **Konvergensi 2x Lebih Cepat:** Menghasilkan update gradient yang lebih halus dan stabil.
2. **Resistensi Overfitting:** Bertindak seperti ansambel mini di dalam satu model tunggal.
3. **Zero Inference Overhead:** Saat evaluasi/inferensi (`model.eval()`), seluruh dropout dinonaktifkan sehingga kecepatan inferensi identik dengan layer linear biasa.

## Cara Menggunakan
```python
from modular_variations.model_configs.multi_sample_dropout.ms_dropout import MultiSampleDropoutHead

# Gantikan nn.Linear konvensional dengan MSD:
self.head = MultiSampleDropoutHead(in_features=768, out_features=spec.out_dim)
logits = self.head(pooled_output)
```

