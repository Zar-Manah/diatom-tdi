"""
vws_ops.py — Operaciones matemáticas en el espacio de pesos para EfficientNet-B3.
Transformaciones analíticas cerradas sin gradientes ni backpropagation.
"""
from __future__ import annotations

import torch
from typing import Dict, Any, Tuple, List


def validate_tensor(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise ValueError(f"Tensor {name} contiene valores no finitos (NaN o Inf).")


def apply_bias_tune(
    state_dict: Dict[str, torch.Tensor],
    class_idx: int,
    delta: float,
    layer_name: str = "classifier.bias",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Ajusta analíticamente el umbral de activación (bias) de una clase."""
    if layer_name not in state_dict:
        raise KeyError(f"Capa {layer_name} no encontrada en state_dict.")
    
    bias = state_dict[layer_name].clone()
    num_classes = bias.shape[0]
    if not (0 <= class_idx < num_classes):
        raise IndexError(f"class_idx={class_idx} fuera de rango [0, {num_classes - 1}].")
    
    prev_val = float(bias[class_idx].item())
    bias[class_idx] += float(delta)
    validate_tensor(bias, layer_name)
    
    meta = {
        "op": "bias_tune",
        "layer": layer_name,
        "class_idx": class_idx,
        "delta": float(delta),
        "prev_value": prev_val,
        "new_value": float(bias[class_idx].item()),
    }
    return bias, meta


def apply_temperature_scaling(
    state_dict: Dict[str, torch.Tensor],
    temperature: float,
    weight_layer: str = "classifier.weight",
    bias_layer: str = "classifier.bias",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Escala la temperatura de los logits: z' = z / T.
    Equivale analíticamente a W' = W / T y b' = b / T.
    Calibra la incertidumbre y mitiga la sobreconfianza sin alterar los argmax directos.
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperatura debe ser positiva: T={temperature}")
    if weight_layer not in state_dict or bias_layer not in state_dict:
        raise KeyError(f"Capas clasificadoras no encontradas: {weight_layer}, {bias_layer}")
    
    w = state_dict[weight_layer].clone() / float(temperature)
    b = state_dict[bias_layer].clone() / float(temperature)
    validate_tensor(w, weight_layer)
    validate_tensor(b, bias_layer)
    
    meta = {
        "op": "temperature_scaling",
        "temperature": float(temperature),
        "weight_layer": weight_layer,
        "bias_layer": bias_layer,
    }
    return w, b, meta


def apply_prototype_blend(
    state_dict: Dict[str, torch.Tensor],
    class_idx: int,
    target_vector: torch.Tensor,
    alpha: float = 0.2,
    layer_name: str = "classifier.weight",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Combina el vector prototipo de la clase con un vector objetivo preservando la norma original.
    """
    if layer_name not in state_dict:
        raise KeyError(f"Capa {layer_name} no encontrada en state_dict.")
    
    weight = state_dict[layer_name].clone()
    num_classes, feat_dim = weight.shape
    if not (0 <= class_idx < num_classes):
        raise IndexError(f"class_idx={class_idx} fuera de rango [0, {num_classes - 1}].")
    
    target = target_vector.to(weight.device, dtype=weight.dtype).view(-1)
    if target.shape[0] != feat_dim:
        raise ValueError(f"Dimensión de target ({target.shape[0]}) != feat_dim ({feat_dim}).")
    
    old_row = weight[class_idx]
    old_norm = torch.norm(old_row, p=2)
    target_norm = torch.norm(target, p=2)
    
    if target_norm == 0:
        raise ValueError("El vector target tiene norma 0.")
    
    normed_target = (target / target_norm) * old_norm
    blended = (1.0 - alpha) * old_row + alpha * normed_target
    
    # Preservar exactamente la norma L2 original para evitar derivas globales de escala
    blended_norm = torch.norm(blended, p=2)
    if blended_norm > 0:
        new_row = blended * (old_norm / blended_norm)
    else:
        new_row = old_row

    weight[class_idx] = new_row
    validate_tensor(weight, layer_name)
    
    meta = {
        "op": "prototype_blend",
        "layer": layer_name,
        "class_idx": class_idx,
        "alpha": float(alpha),
        "norm_conserved": float(old_norm.item()),
    }
    return weight, meta


def apply_margin_expansion(
    state_dict: Dict[str, torch.Tensor],
    class_true: int,
    class_confused: int,
    eta: float = 0.05,
    layer_name: str = "classifier.weight",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Expande el margen de decisión entre dos clases hiper-confundidas (A y B)
    empujando sus prototipos en direcciones opuestas en la esfera unitaria.
    """
    if layer_name not in state_dict:
        raise KeyError(f"Capa {layer_name} no encontrada en state_dict.")
    
    weight = state_dict[layer_name].clone()
    w_t = weight[class_true]
    w_c = weight[class_confused]
    
    norm_t = torch.norm(w_t, p=2)
    norm_c = torch.norm(w_c, p=2)
    
    diff = w_t - w_c
    norm_diff = torch.norm(diff, p=2)
    if norm_diff == 0:
        return weight, {"op": "margin_expansion", "status": "skipped_identical"}
    
    direction = diff / norm_diff
    
    # Empujar clase verdadera hacia fuera del competidor
    new_w_t = w_t + eta * direction * norm_t
    new_w_t = new_w_t * (norm_t / torch.norm(new_w_t, p=2)) # Renormalizar
    
    # Empujar clase competidora hacia atrás levemente (la mitad del impacto)
    new_w_c = w_c - (eta * 0.5) * direction * norm_c
    new_w_c = new_w_c * (norm_c / torch.norm(new_w_c, p=2)) # Renormalizar
    
    weight[class_true] = new_w_t
    weight[class_confused] = new_w_c
    validate_tensor(weight, layer_name)
    
    meta = {
        "op": "margin_expansion",
        "class_true": class_true,
        "class_confused": class_confused,
        "eta": float(eta),
        "norm_t_conserved": float(norm_t.item()),
        "norm_c_conserved": float(norm_c.item()),
    }
    return weight, meta


def apply_nullspace_proj(
    state_dict: Dict[str, torch.Tensor],
    noise_vector: torch.Tensor,
    layer_name: str = "classifier.weight",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Proyecta las filas del clasificador al complemento ortogonal del vector de ruido (detritus).
    w_i = w_i - (w_i . u) * u, donde u = v / ||v||
    Preserva la norma para no alterar el balance relativo de predicción.
    """
    if layer_name not in state_dict:
        raise KeyError(f"Capa {layer_name} no encontrada en state_dict.")
    
    weight = state_dict[layer_name].clone()
    num_classes, feat_dim = weight.shape
    
    u = noise_vector.to(weight.device, dtype=weight.dtype).view(-1)
    if u.shape[0] != feat_dim:
        raise ValueError(f"Dimensión de noise_vector ({u.shape[0]}) != feat_dim ({feat_dim}).")
    
    norm_u = torch.norm(u, p=2)
    if norm_u == 0:
        raise ValueError("noise_vector tiene norma 0.")
    u = u / norm_u
    
    old_norms = torch.norm(weight, p=2, dim=1, keepdim=True) # (N, 1)
    proj = torch.matmul(weight, u).unsqueeze(1) # (N, 1)
    weight_proj = weight - proj * u.unsqueeze(0)
    
    # Renormalizar por fila para conservar energía exacta
    new_norms = torch.norm(weight_proj, p=2, dim=1, keepdim=True) + 1e-8
    weight_proj = weight_proj * (old_norms / new_norms)
    validate_tensor(weight_proj, layer_name)
    
    meta = {
        "op": "nullspace_proj",
        "layer": layer_name,
        "noise_dim": feat_dim,
        "mean_proj_energy": float(torch.mean(torch.abs(proj)).item()),
    }
    return weight_proj, meta


def apply_channel_prune(
    state_dict: Dict[str, torch.Tensor],
    channel_indices: List[int],
    bn_layer: str = "bn2.weight",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Anula canales de ruido en batchnorm/proyección."""
    if bn_layer not in state_dict:
        raise KeyError(f"Capa {bn_layer} no encontrada en state_dict.")
    
    scale = state_dict[bn_layer].clone()
    num_channels = scale.shape[0]
    for ch in channel_indices:
        if not (0 <= ch < num_channels):
            raise IndexError(f"Canal {ch} fuera de rango [0, {num_channels - 1}].")
        scale[ch] = 0.0
        
    validate_tensor(scale, bn_layer)
    meta = {
        "op": "channel_prune",
        "layer": bn_layer,
        "pruned_channels": channel_indices,
        "count": len(channel_indices),
    }
    return scale, meta
