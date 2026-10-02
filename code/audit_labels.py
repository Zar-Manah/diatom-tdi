#!/usr/bin/env python3
"""Auditoría de ruido de etiquetas — Fase 0 MAX.

Carga best.pt de F2 (EfficientNet-B3@300), infiere sobre train+val del split
sellado y produce label_audit.json:
  - suspects: pred != true con prob >= 0.85 (confident-learning-lite)
  - genus_mismatch: género del pred != género del true (flag fuerte)
  - class_errors: tasa de error por clase sobre train
  - class_weights: peso por clase (mean 1, clip [1,3]) derivado del error
  - confusions: top true->pred para el informe
  - val_suspects: reportados aparte (NO se cuarentenan)
Salida: /content/ckpt/label_audit.json
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CKPT = Path("/content/ckpt")
SPLIT = Path("/content/split_f1.json")
BEST = CKPT / "best.pt"
OUT = CKPT / "label_audit.json"
BATCH = 64
PROB_MIN = 0.85


class ListDataset(Dataset):
    def __init__(self, root, items, tfm):
        self.root = Path(root)
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        img = Image.open(self.root / rel).convert("RGB")
        return self.tfm(img), label


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CKPT / "audit.log", "a") as f:
        f.write(line + "\n")


def ensure_ude(data_root: Path, zip_path: Path) -> None:
    inner = data_root / "UDE Diatoms in the Wild 2024" / "images"
    if inner.exists() and any(inner.iterdir()):
        log(f"UDE ya presente: {len(list(inner.glob('*')))} imgs")
        return
    url = (
        "https://zenodo.org/api/records/10410655/files/"
        "UDE%20Diatoms%20in%20the%20Wild%202024.zip/content"
    )
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists() or zip_path.stat().st_size < 2_000_000_000:
        log("descargando UDE...")
        rc = __import__("os").system(
            f'wget -c -O "{zip_path}" "{url}" >> /content/ckpt/uedl.log 2>&1'
        )
        if rc != 0:
            raise RuntimeError("wget UDE failed")
    data_root.mkdir(parents=True, exist_ok=True)
    log("unzip UDE...")
    rc = __import__("os").system(
        f'unzip -o -q "{zip_path}" -d "{data_root}" >> /content/ckpt/uedl.log 2>&1'
    )
    if rc != 0:
        raise RuntimeError("unzip UDE failed")
    n = len(list(inner.glob("*")))
    if n < 1000:
        raise RuntimeError(f"extract vacío: {n}")
    log(f"UDE ready: {n} imgs")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    with open(SPLIT) as f:
        split = json.load(f)
    classes = split["classes"]
    n_cls = len(classes)
    genus = split.get("genus", {})
    log(f"split {split['hash'][:12]} classes={n_cls} "
        f"train={len(split['train'])} val={len(split['val'])}")

    ensure_ude(Path("/content"), Path("/content/ude.zip"))

    ck = torch.load(BEST, map_location="cpu")
    model_name = ck.get("args", {}).get("model", "efficientnet_b3")
    import timm

    model = timm.create_model(model_name, pretrained=False, num_classes=n_cls)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    log(f"modelo {model_name} cargado (best_acc={ck.get('best_acc')})")

    tfm = transforms.Compose(
        [
            transforms.Resize(int(300 * 1.14)),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    suspects, val_suspects = [], []
    class_wrong = defaultdict(int)
    class_total = defaultdict(int)
    confusions = defaultdict(lambda: defaultdict(int))

    for part, items in (("train", split["train"]), ("val", split["val"])):
        ds = ListDataset("/content", items, tfm)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2,
                        pin_memory=True)
        n_wrong = 0
        t0 = time.time()
        idx = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                probs = F.softmax(logits.float(), dim=1)
                p_pred, i_pred = probs.max(1)
                for j in range(yb.size(0)):
                    rel, y = items[idx]
                    y = int(y)
                    pr = float(p_pred[j])
                    ip = int(i_pred[j])
                    if part == "train":
                        class_total[y] += 1
                    if ip != y:
                        if part == "train":
                            class_wrong[y] += 1
                            confusions[y][ip] += 1
                        g_true = (genus.get(classes[y], "") or "").lower()
                        g_pred = (genus.get(classes[ip], "") or "").lower()
                        gmm = bool(g_true and g_pred and g_true != g_pred)
                        rec = dict(rel=rel, true=y, true_name=classes[y],
                                   pred=ip, pred_name=classes[ip],
                                   prob=round(pr, 4), genus_mismatch=gmm)
                        if pr >= PROB_MIN:
                            if part == "train":
                                suspects.append(rec)
                            else:
                                val_suspects.append(rec)
                    idx += 1
        log(f"{part} ok: {len(items)} imgs, wrong={idx and ''}"
            f"{n_wrong} {time.time()-t0:.0f}s")

    class_errors = {
        str(c): round(class_wrong[c] / class_total[c], 4)
        for c in class_total
        if class_total[c] > 0
    }
    weights = {}
    for c in class_total:
        if class_total[c] == 0:
            weights[str(c)] = 1.0
            continue
        err = class_wrong[c] / class_total[c]
        weights[str(c)] = round(min(3.0, 1.0 + 2.0 * err), 4)
    mean_w = sum(weights.values()) / max(1, len(weights))
    weights = {k: round(v / mean_w, 4) for k, v in weights.items()}

    top_conf = []
    for y, d in confusions.items():
        for ip, cnt in sorted(d.items(), key=lambda kv: -kv[1])[:3]:
            top_conf.append(dict(true=y, true_name=classes[y],
                                 pred=ip, pred_name=classes[ip], n=cnt))
    top_conf.sort(key=lambda r: -r["n"])

    audit = dict(
        model=model_name,
        best_acc=float(ck.get("best_acc", 0)),
        split_hash=split["hash"],
        prob_min=PROB_MIN,
        n_train=len(split["train"]),
        n_val=len(split["val"]),
        n_suspects=len(suspects),
        n_val_suspects=len(val_suspects),
        suspects=suspects,
        val_suspects=val_suspects,
        class_errors=class_errors,
        class_weights=weights,
        top_confusions=top_conf[:50],
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(audit))
    tmp.replace(OUT)
    gm = sum(1 for s in suspects if s["genus_mismatch"])
    log(f"DONE suspects={len(suspects)} (genus_mismatch={gm}) "
        f"val_suspects={len(val_suspects)} -> {OUT}")
    (CKPT / "audit_done.flag").write_text("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
