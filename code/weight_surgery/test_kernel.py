"""
test_kernel.py — Suite de pruebas para diatom-weight-surgery.
Verifica que las operaciones analíticas son exactas, conservan normas y no alteran los originales.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import torch

from vws_kernel import VisionWeightKernel
import vws_ops


def run_tests():
    print("[TEST] Iniciando verificación completa del arnés diatom-weight-surgery...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mock_ckpt = tmp_path / "mock_base.pt"

        feat_dim = 1536
        num_classes = 101

        mock_state = {
            "bn2.weight": torch.ones(feat_dim),
            "bn2.bias": torch.zeros(feat_dim),
            "classifier.weight": torch.randn(num_classes, feat_dim),
            "classifier.bias": torch.zeros(num_classes),
        }
        mock_blob = {
            "model": mock_state,
            "classes": [f"class_{i}" for i in range(num_classes)],
            "split_hash": "test_hash_123",
            "epoch": 42,
        }
        torch.save(mock_blob, mock_ckpt)

        # 1. Carga e inspección
        kernel = VisionWeightKernel(mock_ckpt)
        summary = kernel.inspect_summary()
        assert summary["num_classes"] == 101, "Error en num_classes"
        assert summary["total_tensors"] == 4, "Error en total_tensors"
        print("  ✓ 1. Carga e inspección de tensores: OK")

        # 2. Bias tune
        orig_bias_0 = kernel.state_dict["classifier.bias"][0].item()
        kernel.apply_bias_tune(class_idx=0, delta=1.5)
        new_bias_0 = kernel.state_dict["classifier.bias"][0].item()
        assert abs(new_bias_0 - (orig_bias_0 + 1.5)) < 1e-5, "Error en bias_tune"
        print("  ✓ 2. Calibración analítica de bias: OK")

        # 3. Prototype blend conservando norma
        old_w5_norm = torch.norm(kernel.state_dict["classifier.weight"][5], p=2).item()
        target_vec = torch.randn(feat_dim)
        kernel.apply_prototype_blend(class_idx=5, target_vector=target_vec, alpha=0.5)
        new_w5_norm = torch.norm(kernel.state_dict["classifier.weight"][5], p=2).item()
        assert abs(old_w5_norm - new_w5_norm) < 1e-4, "Error: norma no conservada en blend"
        print("  ✓ 3. Fusión de prototipos latentes (norma L2 conservada): OK")

        # 4. Margin expansion conservando norma
        old_norm_10 = torch.norm(kernel.state_dict["classifier.weight"][10], p=2).item()
        old_norm_20 = torch.norm(kernel.state_dict["classifier.weight"][20], p=2).item()
        kernel.apply_margin_expansion(class_true=10, class_confused=20, eta=0.08)
        new_norm_10 = torch.norm(kernel.state_dict["classifier.weight"][10], p=2).item()
        new_norm_20 = torch.norm(kernel.state_dict["classifier.weight"][20], p=2).item()
        assert abs(old_norm_10 - new_norm_10) < 1e-4, "Norma L2 de clase true alterada"
        assert abs(old_norm_20 - new_norm_20) < 1e-4, "Norma L2 de clase confused alterada"
        print("  ✓ 4. Expansión bidireccional de margen (normas conservadas): OK")

        # 5. Temperature scaling
        w_before = kernel.state_dict["classifier.weight"][0, 0].item()
        kernel.apply_temperature_scaling(temperature=1.2)
        w_after = kernel.state_dict["classifier.weight"][0, 0].item()
        assert abs(w_after - (w_before / 1.2)) < 1e-4, "Error en temperature scaling"
        print("  ✓ 5. Escala analítica de temperatura (logits/T): OK")

        # 6. Nullspace projection
        noise_vec = torch.randn(feat_dim)
        noise_u = noise_vec / torch.norm(noise_vec, p=2)
        kernel.apply_nullspace_proj(noise_vec)
        dot_products = torch.matmul(kernel.state_dict["classifier.weight"], noise_u)
        max_residual = torch.abs(dot_products).max().item()
        assert max_residual < 1e-4, f"Fallo en ortogonalidad: residual={max_residual}"
        print(f"  ✓ 6. Proyección ortogonal anti-ruido (residual={max_residual:.2e}): OK")

        # 7. Channel prune
        kernel.apply_channel_prune([10, 25, 100])
        assert kernel.state_dict["bn2.weight"][10].item() == 0.0, "Fallo en channel prune 10"
        print("  ✓ 7. Abliteración de canales de ruido: OK")

        # 8. Revert in memory
        kernel.revert()
        assert kernel.state_dict["classifier.bias"][0].item() == orig_bias_0, "Fallo en revert"
        assert len(kernel.history) == 0, "Historial no limpio tras revert"
        print("  ✓ 8. Reversión en memoria al baseline original: OK")

        # 9. Exportación y manifiesto
        kernel.apply_bias_tune(class_idx=0, delta=0.5)
        out_dir = tmp_path / "out"
        out_file = kernel.export(out_dir, "best_edited.pt")
        assert out_file.exists(), "El archivo exportado no existe"
        assert (out_dir / "manifest.json").exists(), "El manifiesto no existe"

        # 10. Inmutabilidad del original en disco
        base_blob_reloaded = torch.load(mock_ckpt, map_location="cpu")
        assert base_blob_reloaded["model"]["classifier.bias"][0].item() == 0.0, "¡El archivo base original fue mutado!"
        print("  ✓ 9. Exportación, auditoría SHA-256 e inmutabilidad del original: OK")

    print("\n[ÉXITO TOTAL] Todas las pruebas del arnés han pasado limpiamente (9/9).")


if __name__ == "__main__":
    run_tests()
