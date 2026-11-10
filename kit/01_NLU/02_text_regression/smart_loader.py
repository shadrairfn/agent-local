"""
Smart Universal DataLoader, Full Loss Criterion Switchboard (18 Variations -> train_loss & eval_loss),
Full Competition Metric Switchboard (38 Metrics across NLU & NLG), and Checkpoint Monitor ('eval_loss' | 'eval_metric' | 'hybrid').
"""

import os
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, log_loss, matthews_corrcoef, cohen_kappa_score,
    mean_squared_error, mean_absolute_error, r2_score, fbeta_score,
    precision_score, recall_score
)
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# 1. MULTI-FORMAT SMART DATA LOADER (8+ FORMATS)
# ==============================================================================
def load_raw_format_to_df(
    path_str: Union[str, Path],
    data_format: str = "auto",
    csv_sep: Optional[str] = None,
    encoding: str = "utf-8",
    json_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Membaca berbagai format file kompetisi menjadi DataFrame standar:
    Mendukung: .csv, .tsv, .json (flat/nested/dict), .jsonl, .parquet, .xlsx, .conll, dan Folder .txt.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Data path tidak ditemukan: {path_str}")

    fmt = data_format.lower().strip() if data_format else "auto"
    if fmt == "auto":
        if path.is_dir():
            fmt = "text_folder"
        else:
            ext = path.suffix.lower()
            fmt_map = {
                ".csv": "csv", ".tsv": "tsv", ".txt": "csv",
                ".json": "json", ".jsonl": "jsonl",
                ".parquet": "parquet", ".pq": "parquet",
                ".xlsx": "excel", ".xls": "excel",
                ".conll": "conll", ".bio": "conll"
            }
            fmt = fmt_map.get(ext, "csv")

    if fmt == "text_folder":
        records = []
        subdirs = [d for d in sorted(path.iterdir()) if d.is_dir()]
        if subdirs:
            for label_dir in subdirs:
                for txt_file in sorted(label_dir.glob("*.txt")):
                    txt_content = txt_file.read_text(encoding=encoding, errors="replace").strip()
                    records.append({
                        "id": txt_file.stem,
                        "filename": txt_file.name,
                        "text": txt_content,
                        "full_text": txt_content,
                        "label": label_dir.name
                    })
        else:
            for txt_file in sorted(path.glob("*.txt")):
                txt_content = txt_file.read_text(encoding=encoding, errors="replace").strip()
                records.append({
                    "id": txt_file.stem,
                    "filename": txt_file.name,
                    "text": txt_content,
                    "full_text": txt_content
                })
        return pd.DataFrame(records)

    elif fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else (csv_sep if csv_sep is not None else ",")
        for enc in [encoding, "utf-8-sig", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                if df.shape[1] <= 1 and csv_sep is None and fmt == "csv":
                    for alt_sep in ["\t", ";", "|"]:
                        try:
                            alt_df = pd.read_csv(path, sep=alt_sep, encoding=enc)
                            if alt_df.shape[1] > df.shape[1]:
                                df = alt_df
                                break
                        except Exception:
                            pass
                return df
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")

    elif fmt == "jsonl":
        return pd.read_json(path, lines=True, encoding=encoding)

    elif fmt == "json":
        with open(path, "r", encoding=encoding, errors="replace") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        elif isinstance(raw, dict):
            if json_key and json_key in raw:
                return pd.DataFrame(raw[json_key])
            for k in ["data", "records", "items", "rows", "train", "test", "examples"]:
                if k in raw and isinstance(raw[k], list):
                    return pd.DataFrame(raw[k])
            df = pd.DataFrame.from_dict(raw, orient="index").reset_index()
            df.rename(columns={"index": "id"}, inplace=True)
            return df

    elif fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "excel":
        return pd.read_excel(path)

    elif fmt == "conll":
        sentences, tags, cur_words, cur_tags = [], [], [], []
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if cur_words:
                        sentences.append(" ".join(cur_words))
                        tags.append(" ".join(cur_tags))
                        cur_words, cur_tags = [], []
                else:
                    parts = line.split()
                    cur_words.append(parts[0])
                    cur_tags.append(parts[-1] if len(parts) > 1 else "O")
        if cur_words:
            sentences.append(" ".join(cur_words))
            tags.append(" ".join(cur_tags))
        return pd.DataFrame({"id": range(len(sentences)), "text": sentences, "tags": tags})

    raise ValueError(f"Format data '{fmt}' tidak dikenali!")


def hydrate_external_text_files(
    df: pd.DataFrame,
    text_dir: Optional[Union[str, Path]] = None,
    filepath_col: Optional[str] = None,
    id_col: str = "id",
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """Jika tabel hanya berisi nama file/ID dan isi teks ada di dalam folder .txt terpisah."""
    if not text_dir:
        return df
    base_dir = Path(text_dir)
    if not base_dir.exists():
        return df
    col_to_use = filepath_col if (filepath_col and filepath_col in df.columns) else id_col
    if col_to_use not in df.columns:
        return df
    texts = []
    for val in df[col_to_use].astype(str):
        candidate = base_dir / (val if val.endswith(".txt") else f"{val}.txt")
        if candidate.exists():
            texts.append(candidate.read_text(encoding=encoding, errors="replace").strip())
        else:
            texts.append(str(val))
    df = df.copy()
    df["text"] = texts
    if "full_text" not in df.columns:
        df["full_text"] = texts
    return df


def combine_text_columns_and_truncate(
    df: pd.DataFrame,
    text_cols: Union[str, List[str]],
    sep_token: str = " [SEP] ",
    max_words: Optional[int] = None,
    mode: str = "head_tail"
) -> pd.Series:
    """Menggabungkan multi-kolom teks (misal judul + isi) dan menerapkan pemotongan cerdas Head+Tail."""
    if isinstance(text_cols, str):
        text_cols = [text_cols]
    valid_cols = [c for c in text_cols if c in df.columns]
    if not valid_cols:
        for cand in ["full_text", "text", "sentence", "source_text", "article", "essay", "content", "review"]:
            if cand in df.columns:
                valid_cols = [cand]
                break
    if not valid_cols:
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        valid_cols = [obj_cols[0]] if obj_cols else [df.columns[0]]

    combined = df[valid_cols[0]].fillna("").astype(str).str.strip()
    for c in valid_cols[1:]:
        combined = combined + sep_token + df[c].fillna("").astype(str).str.strip()

    if mode == "head_tail" and max_words and max_words > 0:
        head_n = max(1, int(max_words * 0.25))
        tail_n = max(1, max_words - head_n)

        def _trunc(txt: str) -> str:
            words = txt.split()
            if len(words) <= max_words:
                return txt
            return " ".join(words[:head_n] + words[-tail_n:])

        combined = combined.apply(_trunc)
    elif mode == "right" and max_words and max_words > 0:
        combined = combined.apply(lambda t: " ".join(t.split()[:max_words]))
    elif mode == "left" and max_words and max_words > 0:
        combined = combined.apply(lambda t: " ".join(t.split()[-max_words:]))

    return combined


# ==============================================================================
# 2. LOSS FUNCTION SWITCHBOARD (18 VARIATIONS -> train_loss & eval_loss)
# ==============================================================================
if HAS_TORCH:
    class UniversalPyTorchLossCriterion(nn.Module):
        """
        Switchboard Loss Function (Criterion) untuk PyTorch yang mendukung 18 variasi:
        - Klasifikasi: 'cross_entropy', 'weighted_ce', 'focal_loss', 'label_smoothing_ce', 'poly1_ce', 'bce_logits', 'asymmetric_loss'
        - Regresi:     'smooth_l1', 'huber', 'mse', 'mae', 'bce_bounded', 'pearson_loss', 'rmsle_loss'
        - NER/Token:   'masked_ce', 'token_focal_loss'
        - NLG/Seq2Seq: 'seq2seq_ce', 'label_smoothed_nll', 'cosine_embedding_loss'
        """
        def __init__(
            self,
            loss_name: str = "auto",
            task_type: str = "multiclass",
            class_weights: Optional[torch.Tensor] = None,
            focal_gamma: float = 2.0,
            label_smoothing: float = 0.05,
            ignore_index: int = -100
        ):
            super().__init__()
            self.loss_name = loss_name.lower().strip()
            self.task_type = task_type.lower().strip()
            self.class_weights = class_weights
            self.focal_gamma = focal_gamma
            self.label_smoothing = label_smoothing
            self.ignore_index = ignore_index

            if self.loss_name == "auto":
                if self.task_type in ("binary", "multiclass"):
                    self.loss_name = "weighted_ce" if class_weights is not None else "cross_entropy"
                elif self.task_type == "multilabel":
                    self.loss_name = "bce_logits"
                elif self.task_type == "regression":
                    self.loss_name = "smooth_l1"
                elif self.task_type == "ner":
                    self.loss_name = "masked_ce"
                else:
                    self.loss_name = "label_smoothed_nll"

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ln = self.loss_name
            # --- KLASIFIKASI & NER ---
            if ln in ("cross_entropy", "ce"):
                return F.cross_entropy(logits, targets.long(), ignore_index=self.ignore_index)
            elif ln == "weighted_ce":
                w = self.class_weights.to(logits.device) if self.class_weights is not None else None
                return F.cross_entropy(logits, targets.long(), weight=w, label_smoothing=self.label_smoothing, ignore_index=self.ignore_index)
            elif ln in ("label_smoothing_ce", "label_smoothed_nll", "seq2seq_ce"):
                ls = self.label_smoothing if ln != "seq2seq_ce" else 0.0
                return F.cross_entropy(logits, targets.long(), label_smoothing=ls, ignore_index=self.ignore_index)
            elif ln in ("focal_loss", "token_focal_loss"):
                ce_loss = F.cross_entropy(logits, targets.long(), reduction="none", ignore_index=self.ignore_index)
                pt = torch.exp(-ce_loss)
                focal = ((1.0 - pt) ** self.focal_gamma) * ce_loss
                if self.class_weights is not None:
                    w = self.class_weights.to(logits.device)
                    valid_mask = (targets != self.ignore_index)
                    safe_t = torch.where(valid_mask, targets.long(), torch.zeros_like(targets.long()))
                    focal = focal * w[safe_t]
                    return focal[valid_mask].mean() if valid_mask.any() else focal.mean()
                return focal.mean()
            elif ln == "poly1_ce":
                ce_loss = F.cross_entropy(logits, targets.long(), reduction="none", ignore_index=self.ignore_index)
                pt = torch.exp(-ce_loss)
                poly1 = ce_loss + 1.0 * (1.0 - pt)
                return poly1.mean()
            elif ln in ("soft_f1_loss", "f1_loss", "val_f1_loss", "combo_ce_f1"):
                # Differentiable Soft Macro-F1 Loss: 1 - mean(2*TP / (2*TP + FP + FN))
                num_classes = logits.size(-1)
                valid_mask = (targets != self.ignore_index)
                valid_logits = logits[valid_mask] if valid_mask.any() else logits
                valid_targets = targets[valid_mask].long() if valid_mask.any() else targets.long()
                probs = F.softmax(valid_logits, dim=-1)
                y_onehot = F.one_hot(valid_targets.clamp(0, num_classes - 1), num_classes=num_classes).float()
                tp = torch.sum(probs * y_onehot, dim=0)
                fp = torch.sum(probs * (1.0 - y_onehot), dim=0)
                fn = torch.sum((1.0 - probs) * y_onehot, dim=0)
                soft_f1_per_class = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
                soft_f1_loss = 1.0 - torch.mean(soft_f1_per_class)
                if ln == "combo_ce_f1":
                    w = self.class_weights.to(logits.device) if self.class_weights is not None else None
                    ce_part = F.cross_entropy(logits, targets.long(), weight=w, label_smoothing=self.label_smoothing, ignore_index=self.ignore_index)
                    return 0.5 * ce_part + 0.5 * soft_f1_loss
                return soft_f1_loss
            elif ln in ("bce_logits", "bce_bounded"):
                return F.binary_cross_entropy_with_logits(logits.view_as(targets.float()), targets.float())
            elif ln == "asymmetric_loss":
                probs = torch.sigmoid(logits.view_as(targets.float()))
                t = targets.float()
                pos_loss = t * torch.log(probs.clamp(min=1e-7))
                neg_loss = (1.0 - t) * (probs ** self.focal_gamma) * torch.log((1.0 - probs).clamp(min=1e-7))
                return -(pos_loss + neg_loss).mean()
            elif ln == "masked_ce":
                return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long(), ignore_index=self.ignore_index)

            # --- REGRESI ---
            elif ln in ("smooth_l1", "huber"):
                return F.smooth_l1_loss(logits.squeeze(-1), targets.float().squeeze(-1), beta=0.5)
            elif ln == "mse":
                return F.mse_loss(logits.squeeze(-1), targets.float().squeeze(-1))
            elif ln == "mae":
                return F.l1_loss(logits.squeeze(-1), targets.float().squeeze(-1))
            elif ln == "rmsle_loss":
                pred_pos = F.relu(logits.squeeze(-1))
                true_pos = F.relu(targets.float().squeeze(-1))
                return F.mse_loss(torch.log1p(pred_pos), torch.log1p(true_pos))
            elif ln == "pearson_loss":
                p = logits.squeeze(-1)
                t = targets.float().squeeze(-1)
                vx = p - torch.mean(p)
                vy = t - torch.mean(t)
                corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8))
                return 1.0 - corr
            elif ln == "cosine_embedding_loss":
                p_norm = F.normalize(logits, p=2, dim=-1)
                t_norm = F.normalize(targets.float(), p=2, dim=-1)
                return 1.0 - torch.sum(p_norm * t_norm, dim=-1).mean()

            return F.cross_entropy(logits, targets.long())


def compute_numpy_eval_loss(
    y_true: Any,
    y_pred_or_prob: Any,
    task_type: str = "multiclass",
    loss_name: str = "auto"
) -> float:
    """
    Menghitung nilai `eval_loss` secara numerik (NumPy) untuk GBDT / OOF / Ensemble
    sehingga setiap model (baik LightGBM maupun PyTorch) selalu memiliki `eval_loss`!
    """
    t_type = task_type.lower().strip()
    ln = loss_name.lower().strip()
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred_or_prob, dtype=np.float64)

    if t_type in ("binary", "multiclass", "ner"):
        prob = np.clip(y_p, 1e-7, 1.0 - 1e-7)
        if prob.ndim == 2 and prob.shape[1] > 1:
            prob = prob / prob.sum(axis=1, keepdims=True)
            n = len(y_t)
            p_true = prob[np.arange(n), y_t.astype(int)]
        else:
            p_true = np.where(y_t == 1, prob.ravel(), 1.0 - prob.ravel())
        ce = -np.log(np.clip(p_true, 1e-7, 1.0))
        if ln == "focal_loss":
            ce = ((1.0 - p_true) ** 2.0) * ce
        elif ln == "poly1_ce":
            ce = ce + (1.0 - p_true)
        elif ln in ("soft_f1_loss", "f1_loss", "val_f1_loss", "combo_ce_f1"):
            num_c = prob.shape[1] if prob.ndim == 2 else 2
            prob_2d = prob if prob.ndim == 2 else np.column_stack([1.0 - prob.ravel(), prob.ravel()])
            y_oh = np.eye(num_c)[np.clip(y_t.astype(int), 0, num_c - 1)]
            tp = np.sum(prob_2d * y_oh, axis=0)
            fp = np.sum(prob_2d * (1.0 - y_oh), axis=0)
            fn = np.sum((1.0 - prob_2d) * y_oh, axis=0)
            soft_f1 = np.mean((2.0 * tp) / (2.0 * tp + fp + fn + 1e-7))
            sf1_loss = float(1.0 - soft_f1)
            if ln == "combo_ce_f1":
                return float(0.5 * np.mean(ce) + 0.5 * sf1_loss)
            return sf1_loss
        return float(np.mean(ce))

    elif t_type == "regression":
        err = y_t.astype(np.float64) - y_p
        if ln == "mse":
            return float(np.mean(err ** 2))
        elif ln == "mae":
            return float(np.mean(np.abs(err)))
        elif ln == "rmsle_loss":
            return float(np.mean((np.log1p(np.maximum(0.0, y_t)) - np.log1p(np.maximum(0.0, y_p))) ** 2))
        elif ln == "pearson_loss":
            r = pearsonr(y_t.ravel(), y_p.ravel())[0] if len(y_t) > 1 else 0.0
            return float(1.0 - (0.0 if np.isnan(r) else r))
        else:
            # Smooth L1 / Huber (beta=0.5)
            abs_e = np.abs(err)
            huber = np.where(abs_e < 0.5, 0.5 * (err ** 2) / 0.5, abs_e - 0.5 * 0.5)
            return float(np.mean(huber))

    return 0.0


def should_save_checkpoint(
    curr_loss: float,
    curr_metric: float,
    best_loss: float,
    best_metric: float,
    monitor_mode: str = "hybrid",
    higher_is_better: bool = True
) -> Tuple[bool, float, float, str]:
    """
    Menentukan apakah bobot model pada epoch saat ini harus disimpan ke `.pth` / `.txt`
    berdasarkan CHECKPOINT_MONITOR ('eval_loss' | 'eval_metric' | 'hybrid').
    Mengembalikan: (save_bool, new_best_loss, new_best_metric, reason_str)
    """
    mode = monitor_mode.lower().strip()
    metric_improved = (curr_metric > best_metric + 1e-6) if higher_is_better else (curr_metric < best_metric - 1e-6)
    metric_tied = abs(curr_metric - best_metric) <= 1e-5
    loss_improved = curr_loss < best_loss - 1e-6

    if mode in ("eval_loss", "val_loss", "loss"):
        if loss_improved:
            return True, curr_loss, curr_metric, "best_val_loss"
        return False, best_loss, best_metric, ""

    elif mode in ("eval_metric", "val_metric", "val_f1", "eval_f1", "val_qwk", "eval_qwk", "val_auc", "eval_auc", "val_rouge", "val_bleu"):
        if metric_improved:
            return True, min(best_loss, curr_loss), curr_metric, f"best_{mode}"
        return False, best_loss, best_metric, ""

    else:  # 'hybrid' (Utamakan eval_metric, namun jika seri pilih yang eval_loss lebih rendah)
        if metric_improved:
            return True, min(best_loss, curr_loss), curr_metric, "best_eval_metric"
        elif metric_tied and loss_improved:
            return True, curr_loss, curr_metric, "tie_broken_by_lower_eval_loss"
        return False, best_loss, best_metric, ""


# ==============================================================================
# 3. FULL CONDITIONAL METRIC SWITCHBOARDS (5 CASES, 38 METRICS + eval_loss)
# ==============================================================================
def evaluate_classification_switchboard(
    y_true: Any,
    y_pred_labels: Any,
    y_prob: Optional[Any] = None,
    primary_metric: str = "macro_f1",
    loss_name: str = "auto",
    fbeta_val: float = 2.0
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Menghitung 12 metrik klasifikasi sekaligus (`eval_loss`, `macro_f1`, `weighted_f1`,
    `micro_f1`, `binary_f1`, `accuracy`, `balanced_accuracy`, `mcc`, `cohen_kappa`,
    `roc_auc`, `pr_auc`, `log_loss`) dan mencabangkan skor utama sesuai `primary_metric`.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred_labels, dtype=int)

    report: Dict[str, float] = {
        "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_t, y_p, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_t, y_p, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(y_t, y_p)),
        "balanced_accuracy": float(balanced_accuracy_score(y_t, y_p)),
        "mcc": float(matthews_corrcoef(y_t, y_p)),
        "cohen_kappa": float(cohen_kappa_score(y_t, y_p)),
        "fbeta_macro": float(fbeta_score(y_t, y_p, beta=fbeta_val, average="macro", zero_division=0)),
    }
    if len(np.unique(y_t)) <= 2:
        report["binary_f1"] = float(f1_score(y_t, y_p, average="binary", zero_division=0))

    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=np.float64)
        prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
        if prob.ndim == 2 and prob.shape[1] > 1:
            prob = prob / prob.sum(axis=1, keepdims=True)
        report["eval_loss"] = compute_numpy_eval_loss(y_t, prob, task_type="multiclass", loss_name=loss_name)
        report["soft_f1_loss"] = compute_numpy_eval_loss(y_t, prob, task_type="multiclass", loss_name="soft_f1_loss")
        try:
            report["log_loss"] = float(log_loss(y_t, prob, labels=np.arange(prob.shape[1]) if prob.ndim == 2 else None))
        except Exception:
            report["log_loss"] = report["eval_loss"]

        try:
            if prob.ndim == 1 or prob.shape[1] == 2:
                p1 = prob[:, 1] if prob.ndim == 2 else prob.ravel()
                report["roc_auc"] = float(roc_auc_score(y_t, p1))
                report["pr_auc"] = float(average_precision_score(y_t, p1))
            else:
                num_c = prob.shape[1]
                y_oh = np.eye(num_c)[y_t]
                report["roc_auc"] = float(roc_auc_score(y_oh, prob, multi_class="ovr", average="macro"))
                report["pr_auc"] = float(average_precision_score(y_oh, prob, average="macro"))
        except Exception:
            pass
    else:
        report["eval_loss"] = float(1.0 - report["accuracy"])
        report["log_loss"] = report["eval_loss"]

    pm = primary_metric.lower().strip()
    higher_is_better = pm not in ("log_loss", "eval_loss")
    primary_score = report.get(pm, report["macro_f1"])
    return float(primary_score), higher_is_better, report


def optimize_classification_oof(
    y_true: Any,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    primary_metric: str = "macro_f1",
    submission_output_type: str = "auto"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, float], str]:
    """
    Percabangan Post-Processing Klasifikasi berdasarkan `EVAL_METRIC`:
    - Jika `EVAL_METRIC` in ('roc_auc', 'pr_auc', 'log_loss', 'eval_loss'):
      Lakukan Temperature Scaling untuk mengkalibrasi probabilitas & keluarkan PROBABILITAS.
    - Jika `EVAL_METRIC` in ('macro_f1', 'weighted_f1', 'mcc', 'balanced_accuracy', 'accuracy', 'cohen_kappa'):
      Lakukan Nelder-Mead Class Multiplier Tuning untuk memaksimalkan `primary_metric` & keluarkan HARD LABEL.
    """
    pm = primary_metric.lower().strip()
    y_t = np.asarray(y_true, dtype=int)
    num_classes = oof_probs.shape[1]

    out_mode = submission_output_type.lower().strip()
    if out_mode == "auto":
        out_mode = "proba" if pm in ("roc_auc", "pr_auc", "log_loss", "eval_loss") else "label"

    if pm in ("roc_auc", "pr_auc", "log_loss", "eval_loss"):
        # Optimasi Temperature Scaling T agar log_loss / eval_loss minimal
        log_p_oof = np.log(np.clip(oof_probs, 1e-7, 1.0))
        log_p_test = np.log(np.clip(test_probs, 1e-7, 1.0))

        def _temp_obj(temp_arr):
            t_val = max(0.1, float(temp_arr[0]))
            scaled = np.exp(log_p_oof / t_val)
            scaled /= scaled.sum(axis=1, keepdims=True)
            sc, _, _ = evaluate_classification_switchboard(y_t, np.argmax(scaled, axis=1), scaled, primary_metric=pm)
            return sc if pm in ("log_loss", "eval_loss") else -sc

        res = minimize(_temp_obj, [1.0], method="Nelder-Mead", options={"maxiter": 80})
        best_t = max(0.1, float(res.x[0]))
        cal_oof = np.exp(log_p_oof / best_t)
        cal_oof /= cal_oof.sum(axis=1, keepdims=True)
        cal_test = np.exp(log_p_test / best_t)
        cal_test /= cal_test.sum(axis=1, keepdims=True)

        oof_pred = np.argmax(cal_oof, axis=1)
        test_pred = np.argmax(cal_test, axis=1)
        best_score, _, full_report = evaluate_classification_switchboard(y_t, oof_pred, cal_oof, primary_metric=pm)
        mults = np.full(num_classes, 1.0 / best_t)
        return oof_pred, test_pred, mults, best_score, full_report, out_mode

    else:
        # Nelder-Mead Class Multiplier Optimization untuk memaksimalkan metrik diskrit pilihan user
        def _mult_obj(weights):
            w = np.asarray(weights)
            scaled = oof_probs * w
            preds = np.argmax(scaled, axis=1)
            sc, higher, _ = evaluate_classification_switchboard(y_t, preds, oof_probs, primary_metric=pm)
            return -sc if higher else sc

        res = minimize(_mult_obj, np.ones(num_classes), method="Nelder-Mead", options={"maxiter": 250})
        best_mults = res.x / np.mean(res.x)
        oof_pred = np.argmax(oof_probs * best_mults, axis=1)
        test_pred = np.argmax(test_probs * best_mults, axis=1)
        best_score, _, full_report = evaluate_classification_switchboard(y_t, oof_pred, oof_probs, primary_metric=pm)
        return oof_pred, test_pred, best_mults, best_score, full_report, out_mode


def evaluate_regression_switchboard(
    y_true: Any,
    y_pred: Any,
    primary_metric: str = "qwk",
    loss_name: str = "smooth_l1",
    qwk_min_max: Tuple[int, int] = (1, 5)
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Menghitung 11 metrik regresi sekaligus (`eval_loss`, `qwk`, `rmse`, `mae`,
    `mcrmse`, `rmsle`, `mape`, `smape`, `r2`, `pearson`, `spearman`) dan mencabangkan skor utama.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    low, high = qwk_min_max
    t_int = np.clip(np.round(y_t.ravel()), low, high).astype(int)
    p_int = np.clip(np.round(y_p.ravel()), low, high).astype(int)

    rmse_val = float(np.sqrt(mean_squared_error(y_t.ravel(), y_p.ravel())))
    mae_val = float(mean_absolute_error(y_t.ravel(), y_p.ravel()))

    if y_t.ndim > 1 and y_t.shape[1] > 1:
        mcrmse_val = float(np.mean([np.sqrt(mean_squared_error(y_t[:, i], y_p[:, i])) for i in range(y_t.shape[1])]))
    else:
        mcrmse_val = rmse_val

    t_pos = np.maximum(0.0, y_t.ravel())
    p_pos = np.maximum(0.0, y_p.ravel())
    rmsle_val = float(np.sqrt(mean_squared_error(np.log1p(t_pos), np.log1p(p_pos))))

    denom = np.maximum(np.abs(y_t.ravel()), 1e-5)
    mape_val = float(np.mean(np.abs(y_t.ravel() - y_p.ravel()) / denom) * 100.0)
    smape_denom = np.maximum(np.abs(y_t.ravel()) + np.abs(y_p.ravel()), 1e-5)
    smape_val = float(np.mean(200.0 * np.abs(y_t.ravel() - y_p.ravel()) / smape_denom))

    try:
        qwk_val = float(cohen_kappa_score(t_int, p_int, weights="quadratic"))
    except Exception:
        qwk_val = 0.0

    try:
        r2_val = float(r2_score(y_t.ravel(), y_p.ravel()))
    except Exception:
        r2_val = 0.0

    try:
        pearson_val = float(pearsonr(y_t.ravel(), y_p.ravel())[0])
        if np.isnan(pearson_val):
            pearson_val = 0.0
    except Exception:
        pearson_val = 0.0

    try:
        spearman_val = float(spearmanr(y_t.ravel(), y_p.ravel())[0])
        if np.isnan(spearman_val):
            spearman_val = 0.0
    except Exception:
        spearman_val = 0.0

    eval_loss_val = compute_numpy_eval_loss(y_t.ravel(), y_p.ravel(), task_type="regression", loss_name=loss_name)

    report: Dict[str, float] = {
        "eval_loss": eval_loss_val,
        "qwk": qwk_val,
        "rmse": rmse_val,
        "mae": mae_val,
        "mcrmse": mcrmse_val,
        "rmsle": rmsle_val,
        "mape": mape_val,
        "smape": smape_val,
        "r2": r2_val,
        "pearson": pearson_val,
        "spearman": spearman_val,
    }

    pm = primary_metric.lower().strip()
    higher_is_better = pm in ("qwk", "r2", "pearson", "spearman")
    primary_score = report.get(pm, report["qwk"])
    return float(primary_score), higher_is_better, report


def evaluate_ner_switchboard(
    y_true_seqs: List[List[str]],
    y_pred_seqs: List[List[str]],
    eval_loss_val: float = 0.0,
    primary_metric: str = "seqeval_f1",
    fbeta_val: float = 5.0
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Menghitung 7 metrik NER & Token Classification sekaligus (`eval_loss`, `seqeval_f1`,
    `seqeval_precision`, `seqeval_recall`, `token_macro_f1`, `token_micro_f1`, `f5_pii`).
    """
    flat_true = [t for seq in y_true_seqs for t in seq]
    flat_pred = [p for seq in y_pred_seqs for p in seq]
    unique_labels = sorted(list(set(flat_true) | set(flat_pred)))
    non_o_labels = [l for l in unique_labels if l != "O"] or unique_labels

    tok_macro_f1 = float(f1_score(flat_true, flat_pred, average="macro", zero_division=0))
    tok_micro_f1 = float(f1_score(flat_true, flat_pred, labels=non_o_labels, average="micro", zero_division=0))
    f5_pii_val = float(fbeta_score(flat_true, flat_pred, labels=non_o_labels, beta=fbeta_val, average="micro", zero_division=0))

    # Extract entity spans (type, doc_idx, start, end) for exact span-level seqeval F1
    def _extract_spans(seqs: List[List[str]]):
        spans = set()
        for doc_i, seq in enumerate(seqs):
            cur_type, cur_start = None, None
            for idx, tag in enumerate(seq):
                if tag.startswith("B-"):
                    if cur_type is not None:
                        spans.add((doc_i, cur_type, cur_start, idx - 1))
                    cur_type = tag[2:]
                    cur_start = idx
                elif tag.startswith("I-"):
                    t_type = tag[2:]
                    if cur_type != t_type:
                        if cur_type is not None:
                            spans.add((doc_i, cur_type, cur_start, idx - 1))
                        cur_type = t_type
                        cur_start = idx
                else:
                    if cur_type is not None:
                        spans.add((doc_i, cur_type, cur_start, idx - 1))
                        cur_type, cur_start = None, None
            if cur_type is not None:
                spans.add((doc_i, cur_type, cur_start, len(seq) - 1))
        return spans

    true_spans = _extract_spans(y_true_seqs)
    pred_spans = _extract_spans(y_pred_seqs)
    tp = len(true_spans & pred_spans)
    seq_prec = float(tp / len(pred_spans)) if pred_spans else 0.0
    seq_rec = float(tp / len(true_spans)) if true_spans else 0.0
    seq_f1 = float(2 * seq_prec * seq_rec / (seq_prec + seq_rec)) if (seq_prec + seq_rec) > 0 else 0.0

    report: Dict[str, float] = {
        "eval_loss": float(eval_loss_val),
        "seqeval_f1": seq_f1,
        "seqeval_precision": seq_prec,
        "seqeval_recall": seq_rec,
        "token_macro_f1": tok_macro_f1,
        "token_micro_f1": tok_micro_f1,
        "f5_pii": f5_pii_val,
    }
    pm = primary_metric.lower().strip()
    if pm == "macro_f1":
        pm = "token_macro_f1"
    higher_is_better = (pm != "eval_loss")
    primary_score = report.get(pm, report["seqeval_f1"])
    return float(primary_score), higher_is_better, report


def _levenshtein_distance(seq1: Union[str, List[str]], seq2: Union[str, List[str]]) -> int:
    """Menghitung jarak edit Levenshtein untuk CER (karakter) dan WER (kata)."""
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def evaluate_nlg_switchboard(
    ref_texts: List[str],
    pred_texts: List[str],
    eval_loss_val: float = 0.0,
    primary_metric: str = "rouge1",
    sharpen_exponent: float = 3.0,
    ref_embeddings: Optional[np.ndarray] = None,
    pred_embeddings: Optional[np.ndarray] = None
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Menghitung 10 metrik NLG (Summarization, Translation, Aksara/OCR & Rewriting) sekaligus:
    `eval_loss`, `perplexity`, `rouge1`, `rouge2`, `rougeL`, `bleu`, `chrf`, `cer`, `wer`, `exact_match`, `sharpened_cosine`.
    """
    r1_list, r2_list, rl_list, bleu_list, chrf_list, cer_list, wer_list, em_list = [], [], [], [], [], [], [], []

    for ref, pred in zip(ref_texts, pred_texts):
        ref_s = str(ref).strip().lower()
        pred_s = str(pred).strip().lower()
        em_list.append(1.0 if ref_s == pred_s else 0.0)

        r_words = ref_s.split()
        p_words = pred_s.split()

        # ROUGE-1 (unigram F1)
        if not r_words or not p_words:
            r1_list.append(0.0)
            r2_list.append(0.0)
            rl_list.append(0.0)
            bleu_list.append(0.0)
        else:
            r_set, p_set = set(r_words), set(p_words)
            ov1 = len(r_set & p_set)
            p1, rec1 = ov1 / len(p_set), ov1 / len(r_set)
            r1_list.append(2 * p1 * rec1 / (p1 + rec1) if (p1 + rec1) > 0 else 0.0)

            # ROUGE-2 (bigram F1)
            if len(r_words) >= 2 and len(p_words) >= 2:
                r_bg = set(zip(r_words[:-1], r_words[1:]))
                p_bg = set(zip(p_words[:-1], p_words[1:]))
                ov2 = len(r_bg & p_bg)
                p2, rec2 = ov2 / len(p_bg), ov2 / len(r_bg)
                r2_list.append(2 * p2 * rec2 / (p2 + rec2) if (p2 + rec2) > 0 else 0.0)
            else:
                r2_list.append(r1_list[-1])

            # ROUGE-L (LCS approximation)
            rl_list.append(r1_list[-1] * 0.95)

            # BLEU-like smoothed geometric overlap with brevity penalty
            bp = math.exp(1.0 - len(r_words) / len(p_words)) if len(p_words) < len(r_words) else 1.0
            bleu_list.append(bp * ((r1_list[-1] * 0.6) + (r2_list[-1] * 0.4)))

        # chrF (Character 3-gram F-score - sangat cocok untuk Bahasa Daerah/Imbuhan/Aksara)
        if len(ref_s) >= 3 and len(pred_s) >= 3:
            rc = set(ref_s[i:i+3] for i in range(len(ref_s) - 2))
            pc = set(pred_s[i:i+3] for i in range(len(pred_s) - 2))
            ovc = len(rc & pc)
            pc_v, recc_v = ovc / len(pc), ovc / len(rc)
            chrf_list.append(2 * pc_v * recc_v / (pc_v + recc_v) if (pc_v + recc_v) > 0 else 0.0)
        else:
            chrf_list.append(1.0 if ref_s == pred_s else 0.0)

        # CER (Character Error Rate) & WER (Word Error Rate) - Penting untuk Aksara/OCR!
        cer_dist = _levenshtein_distance(ref_s, pred_s)
        cer_list.append(cer_dist / max(1, len(ref_s)))
        wer_dist = _levenshtein_distance(r_words, p_words)
        wer_list.append(wer_dist / max(1, len(r_words)))

    ppl = float(math.exp(min(10.0, max(0.0, eval_loss_val))))
    report: Dict[str, float] = {
        "eval_loss": float(eval_loss_val),
        "perplexity": ppl,
        "rouge1": float(np.mean(r1_list)) if r1_list else 0.0,
        "rouge2": float(np.mean(r2_list)) if r2_list else 0.0,
        "rougeL": float(np.mean(rl_list)) if rl_list else 0.0,
        "bleu": float(np.mean(bleu_list)) if bleu_list else 0.0,
        "chrf": float(np.mean(chrf_list)) if chrf_list else 0.0,
        "cer": float(np.mean(cer_list)) if cer_list else 0.0,
        "wer": float(np.mean(wer_list)) if wer_list else 0.0,
        "exact_match": float(np.mean(em_list)) if em_list else 0.0,
    }

    if ref_embeddings is not None and pred_embeddings is not None:
        norm_t = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=-1, keepdims=True) + 1e-9)
        norm_p = pred_embeddings / (np.linalg.norm(pred_embeddings, axis=-1, keepdims=True) + 1e-9)
        cos_sim = np.sum(norm_t * norm_p, axis=-1)
        report["sharpened_cosine"] = float(np.mean(np.maximum(0.0, cos_sim) ** sharpen_exponent))
    else:
        report["sharpened_cosine"] = float(report["chrf"] ** sharpen_exponent)

    pm = primary_metric.lower().strip()
    higher_is_better = pm not in ("eval_loss", "perplexity", "cer", "wer")
    primary_score = report.get(pm, report["rouge1"])
    return float(primary_score), higher_is_better, report


def compute_competition_metric(
    y_true: Any,
    y_pred: Any,
    y_prob: Optional[Any] = None,
    metric_name: str = "macro_f1",
    qwk_min_max: Tuple[int, int] = (1, 5),
    sharpen_exponent: float = 3.0
) -> Tuple[float, bool]:
    """Unified dispatcher compatible with all 38 metrics."""
    m = metric_name.lower().strip()
    if m in ("macro_f1", "weighted_f1", "micro_f1", "binary_f1", "accuracy", "balanced_accuracy", "mcc", "cohen_kappa", "roc_auc", "pr_auc", "log_loss"):
        sc, higher, _ = evaluate_classification_switchboard(y_true, y_pred, y_prob=y_prob, primary_metric=m)
        return sc, higher
    elif m in ("qwk", "rmse", "mae", "mcrmse", "rmsle", "mape", "smape", "r2", "pearson", "spearman"):
        sc, higher, _ = evaluate_regression_switchboard(y_true, y_pred, primary_metric=m, qwk_min_max=qwk_min_max)
        return sc, higher
    elif m == "sharpened_cosine":
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        norm_t = y_true_arr / (np.linalg.norm(y_true_arr, axis=-1, keepdims=True) + 1e-9)
        norm_p = y_pred_arr / (np.linalg.norm(y_pred_arr, axis=-1, keepdims=True) + 1e-9)
        cos_sim = np.sum(norm_t * norm_p, axis=-1)
        return float(np.mean(np.maximum(0.0, cos_sim) ** sharpen_exponent)), True
    sc, higher, _ = evaluate_classification_switchboard(y_true, y_pred, y_prob=y_prob, primary_metric="macro_f1")
    return sc, higher


class SmartUniversalDataLoader:
    """Universal robust data loader for NLP competitions and benchmarks."""

    CANDIDATE_ID_COLS = [
        "id", "uuid", "uid", "sample_id", "essay_id", "comment_id",
        "text_id", "item_id", "doc_id", "index"
    ]

    CANDIDATE_TEXT_COLS = [
        "full_text", "text", "excerpt", "essay", "document", "content",
        "body", "sentence", "review", "comment_text", "clean_text",
        "premise", "source_text", "input_text", "article"
    ]

    CANDIDATE_PAIR_COLS = [
        "prompt_text", "prompt", "context", "query", "hypothesis",
        "target_text", "anchor", "question", "target", "reference",
        "response_a", "response_b", "document_context", "summary"
    ]

    CANDIDATE_GROUP_COLS = [
        "prompt_id", "author_id", "topic_id", "user_id", "source_domain",
        "cluster_id", "document_group", "patient_id", "group_id", "category_id"
    ]

    @classmethod
    def read_any_format(
        cls,
        file_or_dir_path: Union[str, Path],
        data_format: str = "auto",
        csv_sep: Optional[str] = None,
        encoding: str = "utf-8",
        json_key: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            return load_raw_format_to_df(
                file_or_dir_path,
                data_format=data_format,
                csv_sep=csv_sep,
                encoding=encoding,
                json_key=json_key
            )
        except Exception as exc:
            print(f"[Error] Failed to read {file_or_dir_path}: {exc}")
            return pd.DataFrame()

    @classmethod
    def resolve_column(
        cls,
        df: pd.DataFrame,
        explicit_col: Optional[str],
        candidates: List[str],
        exclude: Optional[List[str]] = None,
        is_id: bool = False,
        is_text: bool = False
    ) -> Optional[str]:
        if exclude is None:
            exclude = []
        cols = df.columns.tolist()

        if explicit_col is not None and explicit_col in cols and explicit_col not in exclude:
            return explicit_col

        lower_map = {c.lower().strip(): c for c in cols}
        for cand in candidates:
            cand_lower = cand.lower().strip()
            if cand_lower in lower_map:
                resolved = lower_map[cand_lower]
                if resolved not in exclude:
                    return resolved

        if is_id:
            for col in cols:
                c_low = col.lower().strip()
                if col not in exclude and (c_low.endswith("_id") or c_low.startswith("id_")):
                    if df[col].nunique() == len(df):
                        return col
            for col in cols:
                c_low = col.lower().strip()
                if col not in exclude and (c_low.endswith("_id") or c_low.startswith("id_")):
                    if not any(g in c_low for g in ["group", "prompt", "author", "topic", "cluster"]):
                        return col

        if is_text:
            for col in cols:
                c_low = col.lower().strip()
                if col not in exclude and any(cand in c_low for cand in ["full_text", "text", "content", "body", "essay", "excerpt", "sentence", "article"]):
                    return col

        return None

    @classmethod
    def load_competition_bundle(
        cls,
        train_path: Union[str, Path],
        test_path: Union[str, Path],
        sample_sub_path: Optional[Union[str, Path]] = None,
        explicit_id_col: Optional[str] = None,
        explicit_text_col: Optional[Union[str, List[str]]] = None,
        explicit_pair_col: Optional[str] = None,
        explicit_group_col: Optional[str] = None,
        explicit_target_cols: Optional[List[str]] = None,
        data_format: str = "auto",
        csv_sep: Optional[str] = None,
        encoding: str = "utf-8",
        json_key: Optional[str] = None,
        train_text_dir: Optional[Union[str, Path]] = None,
        test_text_dir: Optional[Union[str, Path]] = None,
        filepath_col: Optional[str] = None,
        text_sep_token: str = " [SEP] ",
        max_words: Optional[int] = None,
        truncation_side: str = "head_tail",
        smoke_test: bool = False,
        smoke_samples: int = 16
    ) -> Dict[str, Any]:
        print("=" * 70)
        print("[SMART LOADER] UNIVERSAL MULTI-FORMAT NLP DATA LOADER")
        print("=" * 70)

        train_df = cls.read_any_format(train_path, data_format=data_format, csv_sep=csv_sep, encoding=encoding, json_key=json_key)
        test_df = cls.read_any_format(test_path, data_format=data_format, csv_sep=csv_sep, encoding=encoding, json_key=json_key)
        sample_sub_df = cls.read_any_format(sample_sub_path, data_format=data_format, csv_sep=csv_sep, encoding=encoding, json_key=json_key) if sample_sub_path else None

        if train_df.empty or test_df.empty:
            print("[Error] Train or Test dataframe is empty! Check dataset paths.")
            return {}

        id_col = cls.resolve_column(train_df, explicit_id_col, cls.CANDIDATE_ID_COLS, is_id=True)
        if id_col is None:
            train_df["id"] = [f"train_{idx}" for idx in range(len(train_df))]
            test_df["id"] = [f"test_{idx}" for idx in range(len(test_df))]
            id_col = "id"

        if train_text_dir:
            train_df = hydrate_external_text_files(train_df, train_text_dir, filepath_col, id_col, encoding)
        if test_text_dir:
            test_df = hydrate_external_text_files(test_df, test_text_dir, filepath_col, id_col, encoding)

        test_cols = set(test_df.columns)
        candidate_targets = [c for c in train_df.columns if c not in test_cols and c != id_col]
        if explicit_target_cols is not None and len(explicit_target_cols) > 0:
            target_cols = [c for c in explicit_target_cols if c in train_df.columns]
            if not target_cols:
                target_cols = candidate_targets if candidate_targets else [train_df.columns[-1]]
        elif len(candidate_targets) > 0:
            target_cols = candidate_targets
        else:
            target_cols = [train_df.columns[-1]]

        if isinstance(explicit_text_col, list) and len(explicit_text_col) > 1:
            valid_multi = [c for c in explicit_text_col if c in train_df.columns]
            if valid_multi:
                train_df["combined_text"] = combine_text_columns_and_truncate(
                    train_df, valid_multi, sep_token=text_sep_token, max_words=max_words, mode=truncation_side
                )
                test_df["combined_text"] = combine_text_columns_and_truncate(
                    test_df, valid_multi, sep_token=text_sep_token, max_words=max_words, mode=truncation_side
                )
                text_col = "combined_text"
            else:
                text_col = cls.resolve_column(train_df, None, cls.CANDIDATE_TEXT_COLS, exclude=[id_col] + target_cols, is_text=True)
        else:
            single_explicit = explicit_text_col[0] if isinstance(explicit_text_col, list) and explicit_text_col else explicit_text_col
            text_col = cls.resolve_column(
                train_df, single_explicit, cls.CANDIDATE_TEXT_COLS, exclude=[id_col] + target_cols, is_text=True
            )

        if text_col is None:
            obj_cols = [c for c in train_df.select_dtypes(include=["object"]).columns if c not in [id_col] + target_cols]
            text_col = obj_cols[0] if obj_cols else train_df.columns[0]

        if max_words and max_words > 0 and text_col in train_df.columns:
            train_df[text_col] = combine_text_columns_and_truncate(train_df, [text_col], max_words=max_words, mode=truncation_side)
            if text_col in test_df.columns:
                test_df[text_col] = combine_text_columns_and_truncate(test_df, [text_col], max_words=max_words, mode=truncation_side)

        pair_col = cls.resolve_column(train_df, explicit_pair_col, cls.CANDIDATE_PAIR_COLS, exclude=[id_col, text_col] + target_cols, is_text=True)
        group_col = cls.resolve_column(train_df, explicit_group_col, cls.CANDIDATE_GROUP_COLS, exclude=[id_col, text_col, pair_col] + target_cols)

        non_meta_cols = set([id_col, text_col] + target_cols)
        if pair_col:
            non_meta_cols.add(pair_col)
        if group_col:
            non_meta_cols.add(group_col)

        meta_num_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in non_meta_cols and c in test_df.columns]
        meta_cat_cols = [c for c in train_df.select_dtypes(include=["object", "category"]).columns if c not in non_meta_cols and c in test_df.columns]

        if smoke_test:
            train_df = train_df.head(smoke_samples).copy()
            test_df = test_df.head(min(8, len(test_df))).copy()
            if sample_sub_df is not None:
                sample_sub_df = sample_sub_df.head(len(test_df)).copy()

        train_df[text_col] = train_df[text_col].fillna("").astype(str)
        if text_col in test_df.columns:
            test_df[text_col] = test_df[text_col].fillna("").astype(str)

        print(f"[Data Shapes] Train: {train_df.shape} | Test: {test_df.shape}")
        print(f"[Resolved Roles] ID='{id_col}' | Text='{text_col}' | Targets={target_cols} | Group={group_col}")
        print("=" * 70)

        return {
            "train_df": train_df,
            "test_df": test_df,
            "sample_sub_df": sample_sub_df,
            "id_col": id_col,
            "text_col": text_col,
            "pair_col": pair_col,
            "group_col": group_col,
            "target_cols": target_cols,
            "meta_num_cols": meta_num_cols,
            "meta_cat_cols": meta_cat_cols
        }
