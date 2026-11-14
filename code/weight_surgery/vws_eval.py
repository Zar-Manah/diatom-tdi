"""
vws_eval.py — Evaluador pareado integral para modelos base vs. editados.
Verifica la estabilidad en espacio latente y evalúa el rendimiento real
sobre el benchmark sellado de 7.375 células y los 294 sitios de muestreo de río.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image

# Importación robusta de componentes de diatomeas
diat_root = Path.home() / "Desktop/OpenCode/diatomeas"
if diat_root.exists():
    sys.path.insert(0, str(diat_root / "code"))

try:
    from tdi import DiatomClassifier, TdiTable, compute_tdi, Detection
    from validate_tdi import site_of, tdi_from_counts, merge
    DIAT_AVAILABLE = True
except ImportError:
    DIAT_AVAILABLE = False


def fast_synthetic_check(base_ckpt: Path, edited_ckpt: Path, num_samples: int = 50) -> dict:
    """Verificación rápida de derivas en espacio latente con entradas sintéticas."""
    b_blob = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    e_blob = torch.load(edited_ckpt, map_location="cpu", weights_only=False)

    b_state = b_blob.get("model", b_blob)
    e_state = e_blob.get("model", e_blob)

    diffs = []
    for k in b_state:
        if k in e_state:
            diff = torch.abs(b_state[k] - e_state[k]).max().item()
            if diff > 1e-7:
                diffs.append((k, diff, b_state[k].shape))

    print(f"\n[vws_eval] Tensores modificados: {len(diffs)}")
    for name, diff, shape in diffs:
        print(f"  * {name:<30} shape={str(shape):<18} max_diff={diff:.6f}")

    feat_dim = 1536
    torch.manual_seed(42)
    synthetic_feats = torch.randn(num_samples, feat_dim)

    b_logits = F.linear(synthetic_feats, b_state["classifier.weight"], b_state["classifier.bias"])
    b_preds = torch.argmax(b_logits, dim=-1)

    e_logits = F.linear(synthetic_feats, e_state["classifier.weight"], e_state["classifier.bias"])
    e_preds = torch.argmax(e_logits, dim=-1)

    agreement = (b_preds == e_preds).float().mean().item()
    max_logit_drift = torch.abs(b_logits - e_logits).max().item()
    mean_logit_drift = torch.abs(b_logits - e_logits).mean().item()

    print(f"\nPrueba Rápida de Espacio Latente (N={num_samples}):")
    print(f"  * Concordancia de predicciones: {agreement * 100:.2f}%")
    print(f"  * Deriva media de logits:      {mean_logit_drift:.4f}")
    print(f"  * Deriva máxima de logits:    {max_logit_drift:.4f}")

    return {
        "modified_tensors": len(diffs),
        "agreement": agreement,
        "mean_logit_drift": mean_logit_drift,
        "max_logit_drift": max_logit_drift,
    }


def evaluate_real_benchmark(
    base_ckpt: Path,
    edited_ckpt: Path,
    max_cells: int | None = None,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> dict:
    """Evalúa de forma pareada el modelo base vs editado sobre los datos de validación/test reales."""
    if not DIAT_AVAILABLE:
        raise RuntimeError("Los módulos de Diatomeas (tdi.py) no están accesibles.")

    split_path = diat_root / "state" / "split_f1.json"
    if not split_path.exists():
        raise FileNotFoundError(f"No se encontró el split sellado: {split_path}")

    split = json.load(split_path.open())
    test_entries = split["test"]
    if max_cells:
        test_entries = test_entries[:max_cells]

    table = TdiTable.load()
    print(f"\n[vws_eval] Cargando {len(test_entries)} imágenes del benchmark para evaluación pareada...")

    imgs = []
    truth = []
    sites = []
    for rel, label in test_entries:
        p = diat_root / "data" / rel
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
            truth.append(int(label))
            sites.append(site_of(rel.split("/")[-1]))

    print(f"Total imágenes listas: {len(imgs)}")

    # 1. Evaluar Base
    print(f"\n--> Evaluando Modelo BASE: {base_ckpt.name}...")
    b_clf = DiatomClassifier(ckpt=base_ckpt, device=device)
    t0 = time.perf_counter()
    b_probs = b_clf.classify(imgs, batch_size=64)
    dt_base = time.perf_counter() - t0
    b_preds = b_probs.argmax(dim=1).tolist()
    b_top1 = sum(p == t for p, t in zip(b_preds, truth)) / len(truth)

    # 2. Evaluar Editado
    print(f"--> Evaluando Modelo EDITADO: {edited_ckpt.name}...")
    e_clf = DiatomClassifier(ckpt=edited_ckpt, device=device)
    t0 = time.perf_counter()
    e_probs = e_clf.classify(imgs, batch_size=64)
    dt_edited = time.perf_counter() - t0
    e_preds = e_probs.argmax(dim=1).tolist()
    e_top1 = sum(p == t for p, t in zip(e_preds, truth)) / len(truth)

    # 3. Métricas TDI por sitio y global
    b_site_counts = defaultdict(lambda: defaultdict(int))
    e_site_counts = defaultdict(lambda: defaultdict(int))
    true_site_counts = defaultdict(lambda: defaultdict(int))

    for s, bp, ep, tr in zip(sites, b_preds, e_preds, truth):
        b_site_counts[s][table.classes[bp]] += 1
        e_site_counts[s][table.classes[ep]] += 1
        true_site_counts[s][table.classes[tr]] += 1

    # Agregación global
    b_global_tdi = compute_tdi([Detection(p, 0.9, (0, 0, 1, 1)) for p in b_preds], table).tdi
    e_global_tdi = compute_tdi([Detection(p, 0.9, (0, 0, 1, 1)) for p in e_preds], table).tdi
    true_global_tdi = compute_tdi([Detection(t, 0.9, (0, 0, 1, 1)) for t in truth], table).tdi

    # Errores por sitio
    unique_sites = list(true_site_counts.keys())
    b_site_errors = []
    e_site_errors = []
    b_tier_matches = 0
    e_tier_matches = 0

    for s in unique_sites:
        t_res = tdi_from_counts(true_site_counts[s], table)
        b_res = tdi_from_counts(b_site_counts[s], table)
        e_res = tdi_from_counts(e_site_counts[s], table)

        if t_res["tdi"] is not None:
            if b_res["tdi"] is not None:
                b_site_errors.append(abs(b_res["tdi"] - t_res["tdi"]))
                if b_res["quality"] == t_res["quality"]:
                    b_tier_matches += 1
            if e_res["tdi"] is not None:
                e_site_errors.append(abs(e_res["tdi"] - t_res["tdi"]))
                if e_res["quality"] == t_res["quality"]:
                    e_tier_matches += 1

    b_mae = sum(b_site_errors) / len(b_site_errors) if b_site_errors else 0.0
    e_mae = sum(e_site_errors) / len(e_site_errors) if e_site_errors else 0.0

    b_tier_acc = b_tier_matches / len(unique_sites) if unique_sites else 0.0
    e_tier_acc = e_tier_matches / len(unique_sites) if unique_sites else 0.0

    print("\n" + "=" * 65)
    print("RESUMEN PAREADO DE RENDIMIENTO (BENCHMARK REAL)")
    print("=" * 65)
    print(f"{'Métrica':<30} {'BASE':<16} {'EDITADO':<16} {'DIFERENCIA':<12}")
    print("-" * 65)
    diff_acc = (e_top1 - b_top1) * 100
    print(f"{'Top-1 Test Accuracy':<30} {b_top1*100:>6.2f}%         {e_top1*100:>6.2f}%         {diff_acc:>+6.2f}%")
    
    b_tdi_err = abs(b_global_tdi - true_global_tdi) if b_global_tdi else 0.0
    e_tdi_err = abs(e_global_tdi - true_global_tdi) if e_global_tdi else 0.0
    print(f"{'Error TDI Global':<30} {b_tdi_err:>7.4f}          {e_tdi_err:>7.4f}          {e_tdi_err - b_tdi_err:>+7.4f}")
    
    print(f"{'MAE TDI por Sitio (294)':<30} {b_mae:>7.4f}          {e_mae:>7.4f}          {e_mae - b_mae:>+7.4f}")
    print(f"{'Concordancia Clase Agua':<30} {b_tier_acc*100:>6.2f}%         {e_tier_acc*100:>6.2f}%         {(e_tier_acc - b_tier_acc)*100:>+6.2f}%")
    print("=" * 65)

    return {
        "base_top1": b_top1,
        "edited_top1": e_top1,
        "diff_top1": e_top1 - b_top1,
        "base_mae_tdi": b_mae,
        "edited_mae_tdi": e_mae,
        "diff_mae_tdi": e_mae - b_mae,
        "base_tier_acc": b_tier_acc,
        "edited_tier_acc": e_tier_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluador pareado de cirugías de pesos")
    parser.add_argument("--base", type=str, required=True, help="Ruta al checkpoint base")
    parser.add_argument("--edited", type=str, required=True, help="Ruta al checkpoint editado")
    parser.add_argument("--eval-real", action="store_true", help="Ejecutar evaluación sobre el benchmark real de diatomeas")
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de células para prueba rápida")
    args = parser.parse_args()

    fast_synthetic_check(Path(args.base), Path(args.edited))

    if args.eval_real:
        evaluate_real_benchmark(Path(args.base), Path(args.edited), max_cells=args.limit)


if __name__ == "__main__":
    main()
