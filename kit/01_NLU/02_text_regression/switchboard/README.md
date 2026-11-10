# 🔀 Dynamic NLP Task Switchboard

## Deskripsi & Rationale
Dynamic Task Switchboard bertindak sebagai pusat kendali (*central routing switch*) yang membaca parameter `TASK_TYPE` dari sel Master Config dan secara dinamis mengonfigurasi:
1. **Dimensi Output Linear Head (`out_dim`)**
2. **Fungsi Loss Neural Network (`criterion`)**
3. **Layer Aktivasi Output (`activation`)**
4. **Metrik Evaluasi Utama & Tambahan (`evaluate()`)**
5. **Logika Post-Processing Kalibrasi Prediksi (`post_process_predictions()`)**

Komponen ini menghapus risiko *copy-paste traps* di mana fungsi loss klasifikasi tertinggal di pemodelan regresi.

---

## 📑 Matriks Konfigurasi Tugas NLP

| `TASK_TYPE` | Dimensi Output | Fungsi Loss Neural | Layer Aktivasi | Metrik Utama | Strategi Post-Processing |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `single_regression` | 1 | `SmoothL1Loss(beta=1.0)` | `Identity()` | **RMSE** & MAE | Min-Max Clipping |
| `multi_regression` | $K$ | `SmoothL1Loss(beta=1.0)` | `Identity()` | **MCRMSE** | Per-column Range Clipping |
| `bounded_regression`| $K$ | `BCEWithLogitsLoss()` | `Sigmoid()` | **RMSE / BCE** | Clamp $[0.0, 1.0]$ |
| `ordinal_grading` | 1 | `MSELoss()` / `SmoothL1` | `Identity()` | **QWK** | **Nelder-Mead Cut-points Optimization** |
| `counting_volume` | $K$ | `L1Loss()` / Poisson | `ReLU()` | **RMSLE** | Floor non-negative & round integer |
| `pair_similarity` | $K$ | `BCEWithLogitsLoss()` | `Sigmoid()` | **Pearson $r$** | Min-max boundary scaling |
| `single_label` | $C$ | `CrossEntropyLoss()` | `Softmax(dim=1)` | **Macro F1** | Argmax class selection |
| `multi_label` | $K$ | `BCEWithLogitsLoss()` | `Sigmoid()` | **Mean ROC-AUC**| Optimal thresholding ($p \ge 0.5$) |

---

## 💻 Contoh Penggunaan

```python
from switchboard.task_switchboard import NLPTaskSpecification, OptimizedQWKCutter

# Inisialisasi Switchboard
spec = NLPTaskSpecification(task_type='ordinal_grading', target_cols=['grade'], train_df=train_df)

# Di dalam PyTorch Model:
# model.head = nn.Linear(backbone_hidden_size, spec.out_dim)
# loss = spec.criterion(preds, targets)
# output_probs = spec.activation(preds)

# Evaluasi OOF:
metrics = spec.evaluate(oof_true, oof_pred)
print(f"Hasil Evaluasi: {metrics}")

# Kalibrasi Post-Processing (misal optimasi QWK):
cutter = OptimizedQWKCutter(num_classes=6, min_val=1.0).fit(oof_true, oof_pred)
final_test_preds = spec.post_process_predictions(raw_test_preds, cutter=cutter)
```

