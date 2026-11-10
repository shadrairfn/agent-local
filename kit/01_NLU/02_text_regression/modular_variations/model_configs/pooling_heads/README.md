# 🏊 Model Config: Advanced Pooling Heads

## Deskripsi & Rationale
Transformer menghasilkan representasi tersembunyi berdimensi `(batch_size, seq_len, hidden_size)`. Untuk tugas Level A (regresi atau klasifikasi teks utuh), sekuens token tersebut harus diringkas menjadi satu vektor tunggal `(batch_size, hidden_size)` sebelum diteruskan ke linear regression head.

## Opsi Pooling yang Tersedia:
- **`mean` (Rekomendasi Default):** Merata-ratakan seluruh token dengan bobot attention mask. Sangat stabil dan tahan noise.
- **`cls`:** Mengambil token pertama `[CLS]`. Cepat namun sering kali kurang menangkap nuansa teks panjang.
- **`max`:** Mengambil aktivasi terbesar di sepanjang sekuens.
- **`attention`:** Menggunakan layer linear kecil untuk mempelajari bobot perhatian kata mana yang paling penting.
- **`gem` (Generalized Mean):** Pembobotan berbasis pangkat p yang dapat dipelajari (*learnable parameter*).

## Cara Menggunakan
```python
from modular_variations.model_configs.pooling_heads.pooling_layers import get_pooling_layer

self.pooling = get_pooling_layer(pooling_type='mean', hidden_size=768)
pooled_output = self.pooling(last_hidden_state, attention_mask)
```

