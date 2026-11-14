"""
vws_autotune.py — Optimizador analítico y planificador de cirugía de pesos para Diatomeas.
Analiza la matriz de confusión / auditoría de etiquetas y calcula los ajustes óptimos
en el espacio de pesos ponderados por impacto ecológico (TDI), con conservación de norma L2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch

from vws_kernel import VisionWeightKernel
import vws_ops


def load_tdi_table(tdi_csv: Path) -> Dict[str, float]:
    """Carga los valores de sensibilidad TDI por nombre de taxón."""
    tdi_map = {}
    if not tdi_csv.exists():
        return tdi_map
    with open(tdi_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("tdi_value", "").strip()
            if raw:
                try:
                    tdi_map[row["class"]] = float(raw)
                except ValueError:
                    pass
    return tdi_map


def analyze_audit_log(
    audit_json: Path,
    tdi_map: Dict[str, float],
    classes_list: List[str],
) -> Dict[str, Any]:
    with open(audit_json, "r") as f:
        audit_data = json.load(f)

    fp_count: Dict[int, int] = {i: 0 for i in range(len(classes_list))}
    fn_count: Dict[int, int] = {i: 0 for i in range(len(classes_list))}
    pair_confusions: Dict[Tuple[int, int], int] = {}

    for key in ["top_confusions", "suspects", "confusions"]:
        if key in audit_data and isinstance(audit_data[key], list):
            for item in audit_data[key]:
                true_idx = item.get("true")
                pred_idx = item.get("pred")
                n = item.get("n", 1)
                if true_idx is not None and pred_idx is not None and true_idx != pred_idx:
                    fn_count[true_idx] = fn_count.get(true_idx, 0) + n
                    fp_count[pred_idx] = fp_count.get(pred_idx, 0) + n
                    pair = (true_idx, pred_idx)
                    pair_confusions[pair] = pair_confusions.get(pair, 0) + n

    ranked_confusions = []
    for (t_idx, p_idx), count in pair_confusions.items():
        t_name = classes_list[t_idx] if t_idx < len(classes_list) else f"Class_{t_idx}"
        p_name = classes_list[p_idx] if p_idx < len(classes_list) else f"Class_{p_idx}"
        
        t_val = tdi_map.get(t_name)
        p_val = tdi_map.get(p_name)
        
        tdi_drift = abs(t_val - p_val) if (t_val is not None and p_val is not None) else 0.0
        severity = count * (1.0 + tdi_drift * 1.5)
        
        ranked_confusions.append({
            "true_idx": t_idx,
            "true_name": t_name,
            "pred_idx": p_idx,
            "pred_name": p_name,
            "count": count,
            "true_tdi": t_val,
            "pred_tdi": p_val,
            "tdi_drift": tdi_drift,
            "severity_score": severity,
        })

    ranked_confusions.sort(key=lambda x: x["severity_score"], reverse=True)

    overpredicted = sorted(
        [(idx, fp_count[idx], classes_list[idx]) for idx in fp_count if fp_count[idx] > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    suppressed = sorted(
        [(idx, fn_count[idx], classes_list[idx]) for idx in fn_count if fn_count[idx] > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "ranked_confusions": ranked_confusions[:25],
        "top_overpredicted": overpredicted[:10],
        "top_suppressed": suppressed[:10],
    }


def compute_optimal_bias_shifts(
    analysis: Dict[str, Any],
    max_delta: float = 0.35,
    damping_factor: float = 0.12,
) -> Dict[int, float]:
    shifts: Dict[int, float] = {}

    fp_dict = {item[0]: item[1] for item in analysis["top_overpredicted"]}
    fn_dict = {item[0]: item[1] for item in analysis["top_suppressed"]}

    all_indices = set(fp_dict.keys()).union(set(fn_dict.keys()))
    for idx in all_indices:
        fp = fp_dict.get(idx, 0)
        fn = fn_dict.get(idx, 0)
        ratio = (fp + 1.0) / (fn + 1.0)
        raw_shift = - damping_factor * float(torch.log(torch.tensor(ratio)).item())
        clamped_shift = max(-max_delta, min(max_delta, raw_shift))
        if abs(clamped_shift) > 0.02:
            shifts[idx] = round(clamped_shift, 4)

    return shifts


def generate_surgery_plan(
    model_path: str | Path,
    audit_json: str | Path,
    tdi_csv: str | Path,
    eta: float = 0.04,
    max_bias_delta: float = 0.35,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    kernel = VisionWeightKernel(model_path)
    tdi_map = load_tdi_table(Path(tdi_csv))
    analysis = analyze_audit_log(Path(audit_json), tdi_map, kernel.classes)

    bias_shifts = compute_optimal_bias_shifts(analysis, max_delta=max_bias_delta)

    # Identificar pares con mayor desambiguación trófica
    margin_expansions = []
    for item in analysis["ranked_confusions"][:8]:
        margin_expansions.append({
            "class_true": item["true_idx"],
            "class_true_name": item["true_name"],
            "class_confused": item["pred_idx"],
            "class_confused_name": item["pred_name"],
            "eta": eta,
            "tdi_drift": item["tdi_drift"],
            "confusions_count": item["count"],
        })

    plan = {
        "model": str(model_path),
        "total_classes": len(kernel.classes),
        "temperature_scaling": temperature,
        "bias_calibrations": [
            {
                "class_idx": idx,
                "class_name": kernel.classes[idx],
                "delta": delta,
            }
            for idx, delta in bias_shifts.items()
        ],
        "margin_expansions": margin_expansions,
        "top_trophic_confusions_addressed": [
            c for c in analysis["ranked_confusions"] if c["tdi_drift"] >= 1.0
        ][:5],
    }
    return plan


def main():
    parser = argparse.ArgumentParser(description="Autotuner de Cirugía de Pesos para Diatomeas")
    parser.add_argument("--model", type=str, required=True, help="Ruta a best.pt")
    parser.add_argument("--audit", type=str, required=True, help="Ruta a label_audit.json")
    parser.add_argument("--tdi-table", type=str, required=True, help="Ruta a tdi_values_101.csv")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperatura T")
    parser.add_argument("--eta", type=float, default=0.04, help="Margen de expansión eta")
    parser.add_argument("--out-plan", type=str, default="./surgery_plan.json", help="Ruta para guardar el plan")
    parser.add_argument("--apply", action="store_true", help="Aplica el plan y exporta best_edited.pt")
    parser.add_argument("--out-dir", type=str, default="./autotuned_out", help="Directorio de exportación")
    args = parser.parse_args()

    print("\n[vws_autotune] Analizando matriz de confusión e impacto ecológico TDI...")
    plan = generate_surgery_plan(args.model, args.audit, args.tdi_table, eta=args.eta, temperature=args.temperature)

    with open(args.out_plan, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"\n[vws_autotune] Plan analítico generado con éxito en: {args.out_plan}")
    print(f"  * Calibraciones de bias: {len(plan['bias_calibrations'])}")
    print(f"  * Expansiones de margen: {len(plan['margin_expansions'])}")
    print(f"  * Confusiones tróficas críticas abordadas: {len(plan['top_trophic_confusions_addressed'])}")

    if args.apply:
        print("\n[vws_autotune] Aplicando cirugías sobre una copia derivada...")
        kernel = VisionWeightKernel(args.model)
        
        # 1. Temperatura
        if args.temperature != 1.0:
            kernel.apply_temperature_scaling(args.temperature)

        # 2. Bias
        for b_op in plan["bias_calibrations"]:
            kernel.apply_bias_tune(b_op["class_idx"], b_op["delta"])
        
        # 3. Margen bidireccional conservando norma
        for m_op in plan["margin_expansions"]:
            kernel.apply_margin_expansion(m_op["class_true"], m_op["class_confused"], eta=m_op["eta"])
        
        exported_path = kernel.export(args.out_dir)
        print(f"[vws_autotune] Checkpoint derivado listo en: {exported_path}")


if __name__ == "__main__":
    main()
