"""
vws_kernel.py — Kernel de cirugía de pesos para EfficientNet-B3 (Diatomeas).
Permite inspección, aplicación de parches en memoria, reversión y exportación auditada.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import vws_ops


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class VisionWeightKernel:
    def __init__(self, checkpoint_path: str | Path):
        self.ckpt_path = Path(checkpoint_path).resolve()
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {self.ckpt_path}")
        
        self.base_sha256 = sha256_file(self.ckpt_path)
        self.blob = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        
        if "model" in self.blob and isinstance(self.blob["model"], dict):
            self.base_state_dict = {k: v.clone() for k, v in self.blob["model"].items()}
            self.is_full_blob = True
        else:
            self.base_state_dict = {k: v.clone() for k, v in self.blob.items()}
            self.is_full_blob = False
            
        self.state_dict = {k: v.clone() for k, v in self.base_state_dict.items()}
        self.classes = self.blob.get("classes", [])
        self.history: List[Dict[str, Any]] = []

    def revert(self) -> None:
        """Restaura el estado de los tensores al checkpoint base original."""
        self.state_dict = {k: v.clone() for k, v in self.base_state_dict.items()}
        self.history.clear()

    def inspect_summary(self) -> Dict[str, Any]:
        """Devuelve resumen del modelo sin modificar nada."""
        layers = []
        for name, t in self.state_dict.items():
            layers.append({
                "name": name,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "mean": float(t.mean().item()) if t.is_floating_point() else None,
                "std": float(t.std().item()) if t.is_floating_point() and t.numel() > 1 else None,
            })
        return {
            "model_path": str(self.ckpt_path),
            "sha256": self.base_sha256,
            "is_full_blob": self.is_full_blob,
            "num_classes": len(self.classes),
            "total_tensors": len(self.state_dict),
            "layers": layers,
        }

    def print_inspect(self) -> None:
        info = self.inspect_summary()
        print(f"\n[VisionWeightKernel] Checkpoint: {info['model_path']}")
        print(f"SHA-256: {info['sha256']}")
        print(f"Clases registradas: {info['num_classes']}")
        print(f"Total tensores: {info['total_tensors']}")
        print("\n--- Capas Clave ---")
        for layer in info["layers"]:
            if any(k in layer["name"] for k in ["classifier", "bn2", "conv_head"]):
                mean_str = f"{layer['mean']:.4f}" if layer['mean'] is not None else "N/A"
                print(f"  * {layer['name']:<25} shape={str(layer['shape']):<18} mean={mean_str}")

    def apply_bias_tune(self, class_idx: int, delta: float) -> None:
        new_bias, meta = vws_ops.apply_bias_tune(self.state_dict, class_idx, delta)
        self.state_dict["classifier.bias"] = new_bias
        self.history.append(meta)

    def apply_temperature_scaling(self, temperature: float) -> None:
        w, b, meta = vws_ops.apply_temperature_scaling(self.state_dict, temperature)
        self.state_dict["classifier.weight"] = w
        self.state_dict["classifier.bias"] = b
        self.history.append(meta)

    def apply_prototype_blend(self, class_idx: int, target_vector: torch.Tensor, alpha: float = 0.2) -> None:
        new_weight, meta = vws_ops.apply_prototype_blend(self.state_dict, class_idx, target_vector, alpha)
        self.state_dict["classifier.weight"] = new_weight
        self.history.append(meta)

    def apply_margin_expansion(self, class_true: int, class_confused: int, eta: float = 0.05) -> None:
        new_weight, meta = vws_ops.apply_margin_expansion(self.state_dict, class_true, class_confused, eta)
        self.state_dict["classifier.weight"] = new_weight
        self.history.append(meta)

    def apply_nullspace_proj(self, noise_vector: torch.Tensor) -> None:
        new_weight, meta = vws_ops.apply_nullspace_proj(self.state_dict, noise_vector)
        self.state_dict["classifier.weight"] = new_weight
        self.history.append(meta)

    def apply_channel_prune(self, channel_indices: List[int]) -> None:
        new_bn, meta = vws_ops.apply_channel_prune(self.state_dict, channel_indices)
        self.state_dict["bn2.weight"] = new_bn
        self.history.append(meta)

    def export(self, out_dir: str | Path, filename: str = "best_edited.pt") -> Path:
        out_path_dir = Path(out_dir).resolve()
        out_path_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_path_dir / filename
        manifest_file = out_path_dir / "manifest.json"

        # Preparar blob manteniendo exactamente todos los metadatos
        if self.is_full_blob:
            new_blob = dict(self.blob)
            new_blob["model"] = self.state_dict
            new_blob["edited_by"] = "diatom-weight-surgery"
            new_blob["surgery_history"] = self.history
        else:
            new_blob = self.state_dict

        torch.save(new_blob, out_file)
        new_sha256 = sha256_file(out_file)

        # Generar manifiesto
        manifest = {
            "base_checkpoint": str(self.ckpt_path),
            "base_sha256": self.base_sha256,
            "edited_checkpoint": str(out_file),
            "edited_sha256": new_sha256,
            "operations_applied": self.history,
        }
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n[VisionWeightKernel] Checkpoint editado guardado en: {out_file}")
        print(f"Nuevo SHA-256: {new_sha256}")
        print(f"Manifiesto: {manifest_file}")
        return out_file


def main():
    parser = argparse.ArgumentParser(description="Diatom Weight Surgery Kernel")
    parser.add_argument("--model", type=str, required=True, help="Ruta al checkpoint .pt base")
    parser.add_argument("--inspect", action="store_true", help="Inspeccionar tensores y salir")
    parser.add_argument("--op", type=str, choices=["bias_tune", "nullspace_proj", "prototype_blend", "channel_prune", "margin_expansion", "temperature"])
    parser.add_argument("--class-idx", type=int, help="Índice de clase para bias_tune o prototype_blend")
    parser.add_argument("--class-true", type=int, help="Índice de clase real para margin_expansion")
    parser.add_argument("--class-confused", type=int, help="Índice de clase confundida para margin_expansion")
    parser.add_argument("--delta", type=float, default=0.0, help="Delta para bias_tune")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha para prototype_blend")
    parser.add_argument("--eta", type=float, default=0.05, help="Eta para margin_expansion")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperatura T > 0")
    parser.add_argument("--channels", type=str, help="Canales separados por coma para channel_prune")
    parser.add_argument("--vector-path", type=str, help="Ruta a tensor .pt para nullspace o prototype")
    parser.add_argument("--out-dir", type=str, default="./edited_out", help="Directorio de salida")
    args = parser.parse_args()

    kernel = VisionWeightKernel(args.model)

    if args.inspect:
        kernel.print_inspect()
        return

    if args.op == "bias_tune":
        if args.class_idx is None:
            sys.exit("Error: --class-idx es obligatorio para bias_tune")
        kernel.apply_bias_tune(args.class_idx, args.delta)
    elif args.op == "temperature":
        kernel.apply_temperature_scaling(args.temperature)
    elif args.op == "margin_expansion":
        if args.class_true is None or args.class_confused is None:
            sys.exit("Error: --class-true y --class-confused son obligatorios para margin_expansion")
        kernel.apply_margin_expansion(args.class_true, args.class_confused, eta=args.eta)
    elif args.op == "channel_prune":
        if not args.channels:
            sys.exit("Error: --channels es obligatorio para channel_prune (ej: --channels 5,12,89)")
        chans = [int(c.strip()) for c in args.channels.split(",")]
        kernel.apply_channel_prune(chans)
    elif args.op in ("prototype_blend", "nullspace_proj"):
        if not args.vector_path:
            sys.exit(f"Error: --vector-path requerido para {args.op}")
        vec = torch.load(args.vector_path, map_location="cpu")
        if args.op == "prototype_blend":
            if args.class_idx is None:
                sys.exit("Error: --class-idx es obligatorio para prototype_blend")
            kernel.apply_prototype_blend(args.class_idx, vec, alpha=args.alpha)
        else:
            kernel.apply_nullspace_proj(vec)

    kernel.export(args.out_dir)


if __name__ == "__main__":
    main()
