"""
Layerwise Learning Rate Decay (LLRD) Optimizer Parameter Builder.
Applies geometric learning rate decay from top transformer layers down to lower layers,
preserving low-level grammatical representations while aggressively adapting top semantic heads.
"""

from typing import List, Dict, Any
import torch.nn as nn


def get_llrd_optimizer_params(
    model: nn.Module,
    base_lr: float = 2e-5,
    decay_rate: float = 0.90,
    head_lr_mult: float = 2.5,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """Builds parameter groups with layerwise decaying learning rates for HuggingFace Transformers."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    # Separate head parameters from backbone
    backbone = getattr(model, "backbone", getattr(model, "transformer", None))
    if backbone is None:
        # Fallback to standard parameter group
        return [{"params": model.parameters(), "lr": base_lr, "weight_decay": weight_decay}]

    # Identify layer depth in transformer encoder
    encoder = getattr(backbone, "encoder", backbone)
    layers = getattr(encoder, "layer", getattr(encoder, "layers", []))
    num_layers = len(layers)

    params_groups = []

    # 1. Embeddings (lowest rate: base_lr * (decay_rate ** num_layers))
    emb_rate = base_lr * (decay_rate ** num_layers)
    emb_params_decay = []
    emb_params_no_decay = []
    for name, p in getattr(backbone, "embeddings", backbone).named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            emb_params_no_decay.append(p)
        else:
            emb_params_decay.append(p)

    if emb_params_decay:
        params_groups.append({"params": emb_params_decay, "lr": emb_rate, "weight_decay": weight_decay})
    if emb_params_no_decay:
        params_groups.append({"params": emb_params_no_decay, "lr": emb_rate, "weight_decay": 0.0})

    # 2. Intermediate Encoder Layers
    for layer_idx, layer in enumerate(layers):
        layer_rate = base_lr * (decay_rate ** (num_layers - layer_idx - 1))
        p_decay = []
        p_no_decay = []
        for name, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                p_no_decay.append(p)
            else:
                p_decay.append(p)
        if p_decay:
            params_groups.append({"params": p_decay, "lr": layer_rate, "weight_decay": weight_decay})
        if p_no_decay:
            params_groups.append({"params": p_no_decay, "lr": layer_rate, "weight_decay": 0.0})

    # 3. Prediction Head (Highest learning rate)
    head_rate = base_lr * head_lr_mult
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    if head_params:
        params_groups.append({"params": head_params, "lr": head_rate, "weight_decay": weight_decay})

    return params_groups

