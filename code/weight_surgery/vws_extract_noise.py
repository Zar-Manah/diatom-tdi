"""
vws_extract_noise.py — Extractor de vectores de firma latente para detritus/artefactos.
Toma una carpeta de imágenes de fondo/detritus y calcula el vector propio principal (SVD)
en el espacio latente de 1536 dimensiones de EfficientNet-B3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from PIL import Image

try:
    from tdi import DiatomClassifier, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    # Si tdi.py está en el repo de diatomeas
    diat_code = Path(__file__).resolve().parents[1]
    if diat_code.exists():
        sys.path.insert(0, str(diat_code))
        from tdi import DiatomClassifier, IMAGENET_MEAN, IMAGENET_STD
    else:
        DiatomClassifier = None


def extract_noise_signature(
    image_paths: list[Path],
    ckpt_path: Path,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
) -> torch.Tensor:
    if DiatomClassifier is None:
        raise RuntimeError("No se pudo importar DiatomClassifier de tdi.py.")

    clf = DiatomClassifier(ckpt=ckpt_path, device=device)
    model = clf.model
    model.eval()

    latent_vectors = []
    print(f"[vws_extract_noise] Procesando {len(image_paths)} imágenes de detritus...")

    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i : i + batch_size]
        batch = []
        for p in chunk:
            try:
                im = Image.open(p).convert("RGB")
                batch.append(clf.tf(im))
            except Exception as e:
                print(f"  Advertencia: error al leer {p.name}: {e}")
        if not batch:
            continue
        x = torch.stack(batch).to(clf.device)
        with torch.inference_mode():
            # Extraer características antes del clasificador lineal
            feats = model.forward_features(x)
            pooled = model.global_pool(feats) if hasattr(model, "global_pool") else torch.mean(feats, dim=(2, 3))
            pooled = pooled.flatten(1)
            latent_vectors.append(pooled.cpu())

    if not latent_vectors:
        raise ValueError("No se extrajeron vectores latentes válidos.")

    X = torch.cat(latent_vectors, dim=0) # (N, 1536)
    print(f"Matriz de ruido extraída: {X.shape}")

    # Centrar y obtener el 1er componente principal vía SVD
    X_mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - X_mean
    U, S, V = torch.pca_lowrank(X_centered, q=1)
    principal_noise_vector = V[:, 0]

    # Asegurar norma 1
    principal_noise_vector = principal_noise_vector / torch.norm(principal_noise_vector, p=2)
    return principal_noise_vector


def main():
    parser = argparse.ArgumentParser(description="Extractor de vector de ruido para supresión en espacio nulo")
    parser.add_argument("--images-dir", type=str, required=True, help="Directorio con imágenes de detritus/artefactos")
    parser.add_argument("--model", type=str, required=True, help="Ruta al checkpoint base .pt")
    parser.add_argument("--out-vector", type=str, default="./detritus_vector.pt", help="Archivo de salida .pt")
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(list(img_dir.glob(ext)))

    if not files:
        sys.exit(f"Error: no se encontraron imágenes en {img_dir}")

    vec = extract_noise_signature(files, Path(args.model))
    torch.save(vec, args.out_vector)
    print(f"[vws_extract_noise] Vector propio de detritus guardado en: {args.out_vector} (dim={vec.shape[0]})")


if __name__ == "__main__":
    main()

