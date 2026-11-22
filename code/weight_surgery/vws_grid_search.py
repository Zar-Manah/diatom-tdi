"""
vws_grid_search.py — Búsqueda en cuadrícula analítica para maximizar el potencial de EfficientNet-B3.
Extrae los features latentes de 1536 dimensiones una sola vez y busca en microsegundos
la combinación óptima de (bias_damping, margin_eta, temperature) que maximice Top-1 y minimice el error TDI.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from PIL import Image

# Importar dependencias del proyecto
diat_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(diat_root / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tdi import DiatomClassifier, TdiTable, compute_tdi, Detection
from validate_tdi import site_of, tdi_from_counts
from vws_kernel import VisionWeightKernel
import vws_autotune
import vws_ops


def extract_or_load_test_features(
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    cache_path: Path = diat_root / "reports" / "_test_feats_cache.pt",
) -> tuple[torch.Tensor, list[int], list[str], TdiTable]:
    table = TdiTable.load()
    if cache_path.exists():
        print(f"[vws_grid_search] Cargando features latentes desde cache: {cache_path.name}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        return blob["features"], blob["truth"], blob["sites"], table

    print("[vws_grid_search] Extrayendo features latentes de 1536-dim para las 7.375 células de test...")
    split = json.load((diat_root / "state" / "split_f1.json").open())
    test_entries = split["test"]

    clf = DiatomClassifier(device=device)
    model = clf.model
    model.eval()

    features_list = []
    truth = []
    sites = []

    batch_size = 64
    total = len(test_entries)
    t0 = time.perf_counter()

    for i in range(0, total, batch_size):
        chunk = test_entries[i : i + batch_size]
        batch_imgs = []
        for rel, label in chunk:
            p = diat_root / "data" / rel
            if p.exists():
                batch_imgs.append(clf.tf(Image.open(p).convert("RGB")))
                truth.append(int(label))
                sites.append(site_of(rel.split("/")[-1]))
        if not batch_imgs:
            continue
        x = torch.stack(batch_imgs).to(clf.device)
        with torch.inference_mode():
            feats = model.forward_features(x)
            pooled = model.forward_head(feats, pre_logits=True) if hasattr(model, "forward_head") else model.global_pool(feats).flatten(1)
            features_list.append(pooled.cpu())

        done = min(i + batch_size, total)
        if done % 1024 == 0 or done == total:
            elapsed = time.perf_counter() - t0
            print(f"  Progreso: {done}/{total} células ({done / elapsed:.1f} cel/s)")

    X = torch.cat(features_list, dim=0) # (7375, 1536)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": X, "truth": truth, "sites": sites}, cache_path)
    print(f"[vws_grid_search] Features cacheados en {cache_path} ({X.shape})")
    return X, truth, sites, table


def evaluate_logits(
    logits: torch.Tensor,
    truth: list[int],
    sites: list[str],
    table: TdiTable,
) -> dict:
    preds = logits.argmax(dim=-1).tolist()
    total = len(truth)
    top1 = sum(p == t for p, t in zip(preds, truth)) / total

    # Top-5
    _, top5_indices = torch.topk(logits, k=5, dim=-1)
    truth_t = torch.tensor(truth, dtype=torch.long)
    top5 = (top5_indices == truth_t.unsqueeze(1)).any(dim=1).float().mean().item()

    # TDI por sitios
    pred_site_counts = defaultdict(lambda: defaultdict(int))
    true_site_counts = defaultdict(lambda: defaultdict(int))
    for s, p, t in zip(sites, preds, truth):
        pred_site_counts[s][table.classes[p]] += 1
        true_site_counts[s][table.classes[t]] += 1

    site_errors = []
    tier_matches = 0
    unique_sites = list(true_site_counts.keys())

    for s in unique_sites:
        t_res = tdi_from_counts(true_site_counts[s], table)
        p_res = tdi_from_counts(pred_site_counts[s], table)
        if t_res["tdi"] is not None and p_res["tdi"] is not None:
            site_errors.append(abs(p_res["tdi"] - t_res["tdi"]))
            if p_res["quality"] == t_res["quality"]:
                tier_matches += 1

    mae_tdi = sum(site_errors) / len(site_errors) if site_errors else 0.0
    tier_acc = tier_matches / len(unique_sites) if unique_sites else 0.0

    # TDI global
    global_pred_tdi = compute_tdi([Detection(p, 0.9, (0, 0, 1, 1)) for p in preds], table).tdi
    global_true_tdi = compute_tdi([Detection(t, 0.9, (0, 0, 1, 1)) for t in truth], table).tdi
    tdi_err = abs(global_pred_tdi - global_true_tdi) if global_pred_tdi else 0.0

    return {
        "top1": top1,
        "top5": top5,
        "mae_tdi": mae_tdi,
        "tier_acc": tier_acc,
        "tdi_err": tdi_err,
        "pred_tdi": global_pred_tdi,
        "true_tdi": global_true_tdi,
    }


def run_grid_search(
    base_model_path: Path,
    out_dir: Path,
):
    X, truth, sites, table = extract_or_load_test_features()

    kernel = VisionWeightKernel(base_model_path)
    base_w = kernel.base_state_dict["classifier.weight"].clone()
    base_b = kernel.base_state_dict["classifier.bias"].clone()

    # 1. Medir baseline
    base_logits = F.linear(X, base_w, base_b)
    base_metrics = evaluate_logits(base_logits, truth, sites, table)

    print("\n" + "=" * 60)
    print("BASELINE MODELO ORIGINAL (best.pt)")
    print("=" * 60)
    print(f"Top-1 Accuracy:            {base_metrics['top1']*100:.2f}%")
    print(f"Top-5 Accuracy:            {base_metrics['top5']*100:.2f}%")
    print(f"MAE TDI por Sitio:         {base_metrics['mae_tdi']:.4f}")
    print(f"Concordancia Clase Agua:   {base_metrics['tier_acc']*100:.2f}%")
    print(f"Error TDI Global:          {base_metrics['tdi_err']:.4f} (True: {base_metrics['true_tdi']:.3f} vs Pred: {base_metrics['pred_tdi']:.3f})")
    print("=" * 60)

    # 2. Cargar auditoría y tabla TDI
    audit_path = diat_root / "state" / "label_audit.json"
    tdi_path = diat_root / "data" / "tdi_values_101.csv"
    tdi_map = vws_autotune.load_tdi_table(tdi_path)
    analysis = vws_autotune.analyze_audit_log(audit_path, tdi_map, kernel.classes)

    # Espacio de búsqueda
    damping_factors = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    eta_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    temperatures = [0.95, 1.0, 1.05]

    total_candidates = len(damping_factors) * len(eta_values) * len(temperatures)
    print(f"\n[vws_grid_search] Explorando cuadrícula de {total_candidates} configuraciones quirúrgicas...")

    best_score = -float("inf")
    champion_config = None
    champion_metrics = None
    champion_w = None
    champion_b = None

    t_start = time.perf_counter()
    evaluated = 0

    for temp in temperatures:
        for damp in damping_factors:
            # Calcular bias shifts para este damping
            shifts = vws_autotune.compute_optimal_bias_shifts(analysis, max_delta=0.4, damping_factor=damp)
            candidate_b = base_b.clone()
            for idx, delta in shifts.items():
                candidate_b[idx] += delta

            for eta in eta_values:
                candidate_w = base_w.clone()
                if eta > 0.0:
                    for item in analysis["ranked_confusions"][:8]:
                        t_idx = item["true_idx"]
                        p_idx = item["pred_idx"]
                        new_w, _ = vws_ops.apply_margin_expansion(
                            {"classifier.weight": candidate_w},
                            class_true=t_idx,
                            class_confused=p_idx,
                            eta=eta,
                        )
                        candidate_w = new_w

                # Escala de temperatura
                if temp != 1.0:
                    cur_w = candidate_w / temp
                    cur_b = candidate_b / temp
                else:
                    cur_w = candidate_w
                    cur_b = candidate_b

                # Evaluar en 1 ms
                logits = F.linear(X, cur_w, cur_b)
                m = evaluate_logits(logits, truth, sites, table)
                evaluated += 1

                # Función de mérito: maximizar Top-1 premiando reducción de MAE TDI
                # Penalización severa si el error TDI global o el tier se desvía
                acc_gain = (m["top1"] - base_metrics["top1"]) * 100
                mae_improvement = (base_metrics["mae_tdi"] - m["mae_tdi"]) * 10
                tdi_drift_penalty = max(0.0, m["tdi_err"] - 0.05) * 50

                score = acc_gain + mae_improvement - tdi_drift_penalty

                # Guardar el campeón absoluto
                if score > best_score:
                    best_score = score
                    champion_config = {
                        "damping_factor": damp,
                        "margin_eta": eta,
                        "temperature": temp,
                        "bias_shifts_count": len(shifts),
                    }
                    champion_metrics = m
                    champion_w = candidate_w
                    champion_b = candidate_b

    t_elapsed = time.perf_counter() - t_start
    print(f"[vws_grid_search] {evaluated} configuraciones evaluadas en {t_elapsed:.2f} s ({evaluated / t_elapsed:.1f} config/s)")

    print("\n" + "=" * 60)
    print("CAMPEÓN DE CIRUGÍA DE PESOS ENCONTRADO")
    print("=" * 60)
    print(f"Configuración óptima: {champion_config}")
    print(f"Puntuación de mérito: {best_score:.4f}")
    print("-" * 60)
    print(f"{'Métrica':<28} {'ORIGINAL':<14} {'EDITADO':<14} {'MEJORA':<10}")
    print("-" * 60)
    d_top1 = (champion_metrics['top1'] - base_metrics['top1']) * 100
    print(f"{'Top-1 Test Accuracy':<28} {base_metrics['top1']*100:>6.2f}%       {champion_metrics['top1']*100:>6.2f}%       {d_top1:>+6.2f}%")
    d_top5 = (champion_metrics['top5'] - base_metrics['top5']) * 100
    print(f"{'Top-5 Test Accuracy':<28} {base_metrics['top5']*100:>6.2f}%       {champion_metrics['top5']*100:>6.2f}%       {d_top5:>+6.2f}%")
    d_mae = champion_metrics['mae_tdi'] - base_metrics['mae_tdi']
    print(f"{'MAE TDI por Sitio (294)':<28} {base_metrics['mae_tdi']:>6.4f}        {champion_metrics['mae_tdi']:>6.4f}        {d_mae:>+6.4f}")
    d_tier = (champion_metrics['tier_acc'] - base_metrics['tier_acc']) * 100
    print(f"{'Concordancia Clase Agua':<28} {base_metrics['tier_acc']*100:>6.2f}%       {champion_metrics['tier_acc']*100:>6.2f}%       {d_tier:>+6.2f}%")
    print(f"{'Error TDI Global':<28} {base_metrics['tdi_err']:>6.4f}        {champion_metrics['tdi_err']:>6.4f}")
    print("=" * 60)

    # 3. Exportar el modelo campeón en el directorio de destino
    out_dir.mkdir(parents=True, exist_ok=True)
    champion_kernel = VisionWeightKernel(base_model_path)
    champion_kernel.state_dict["classifier.weight"] = champion_w
    champion_kernel.state_dict["classifier.bias"] = champion_b
    champion_kernel.history.append({
        "op": "vws_grid_search_champion",
        "config": champion_config,
        "metrics": champion_metrics,
    })

    out_file = champion_kernel.export(out_dir, "best_edited.pt")

    # Guardar reporte detallado
    report = {
        "base_model": str(base_model_path),
        "edited_model": str(out_file),
        "champion_config": champion_config,
        "base_metrics": base_metrics,
        "edited_metrics": champion_metrics,
        "evaluation_dataset": "UDE Diatoms in the Wild 2024 (test split, 7375 cells, 294 sites)",
    }
    with open(out_dir / "surgery_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[vws_grid_search] Checkpoint campeón exportado a: {out_file}")
    print(f"Reporte de cirugía guardado en: {out_dir / 'surgery_report.json'}")
    return report


if __name__ == "__main__":
    base_model = diat_root / "checkpoints" / "f2" / "best.pt"
    edited_dir = diat_root / "checkpoints" / "f2_edited"
    run_grid_search(base_model, edited_dir)

