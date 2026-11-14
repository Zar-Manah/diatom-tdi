#!/usr/bin/env python3
"""Ensemble MAX — promedio de probabilidades de las 9 corridas (EMA) con y
sin TTA (TenCrop x2 escalas) sobre el test sellado + OOF val por fold.
Salida: /content/ckpt/metrics_ensemble.json
"""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CK = Path("/content/ckpt")
ARCHS = ["convnext_tiny", "efficientnetv2_rw_s", "swin_tiny_patch4_window7_224"]
FOLDS = [0, 1, 2]
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ListDS(Dataset):
    def __init__(self, root, items, size):
        self.root = Path(root)
        self.items = items
        self.tfm = transforms.Compose(
            [transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
             transforms.ToTensor(), NORM])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        return self.tfm(Image.open(self.root / rel).convert("RGB")), y


class TTADS(Dataset):
    def __init__(self, root, items, size):
        self.root = Path(root)
        self.items = items
        self.tfm = transforms.Compose(
            [transforms.Resize(int(size * 1.15)), transforms.TenCrop(size)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        crops = self.tfm(Image.open(self.root / rel).convert("RGB"))
        return torch.stack(crops), y


def load_model(rid: str, device):
    from train_max import make_model
    ck = torch.load(CK / rid / "best.pt", map_location="cpu")
    name = ck["args"]["model"] if "args" in ck else ck["model_name"]
    n_cls = len(ck["classes"])
    size = ck.get("img", 320)
    m = make_model(name, n_cls, size, 0.0, pretrained=False)
    sd = ck.get("ema") or ck["model"]
    m.load_state_dict(sd, strict=False)
    missing = m.load_state_dict(sd, strict=False)
    m = m.to(device).eval()
    return m, size, missing


def probs(model, root, items, size, device, tta, batch):
    ds = TTADS(root, items, size) if tta else ListDS(root, items, size)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    out = []
    with torch.no_grad():
        for xb in dl:
            if tta:
                b, n, c, h, w = xb.shape
                x = xb.view(b * n, c, h, w).to(device)
                with torch.amp.autocast("cuda"):
                    lg = model(x)
                lg = lg.view(b, n, -1).float().mean(1)
                out.append(lg.softmax(1).cpu())
            else:
                x = xb.to(device)
                with torch.amp.autocast("cuda"):
                    lg = model(x)
                out.append(lg.float().softmax(1).cpu())
    return torch.cat(out)


def acc(p: torch.Tensor, y) -> float:
    return 100.0 * (p.argmax(1) == torch.tensor(y)).float().mean().item()


def main() -> int:
    device = torch.device("cuda")
    with open("/content/split_max.json") as f:
        split = json.load(f)
    test = split["test"]
    y_test = [y for _, y in test]
    log = (CK / "ensemble.log")

    def L(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    runs = [f"{a.split('_')[0]}_f{k}" for a in ARCHS for k in FOLDS]
    P_plain, P_tta, singles = [], [], []
    for rid in runs:
        if not (CK / rid / "best.pt").exists():
            L(f"skip {rid} (sin best.pt)")
            continue
        model, size, miss = load_model(rid, device)
        L(f"{rid} cargado (size={size}, missing={len(miss.missing_keys)})")
        b = {192: 64, 320: 32, 448: 16, 512: 8}.get(size, 16)
        pp = probs(model, "/content", test, size, device, False, b)
        pt = probs(model, "/content", test, size, device, True, max(2, b // 4))
        singles.append(dict(run=rid, plain=acc(pp, y_test),
                            tta=acc(pt, y_test)))
        L(f"  {rid}: plain={singles[-1]['plain']:.2f}% "
          f"tta={singles[-1]['tta']:.2f}%")
        P_plain.append(pp)
        P_tta.append(pt)
        del model
        torch.cuda.empty_cache()

    if not P_plain:
        L("FATAL: 0 corridas disponibles")
        return 2

    ens_plain = torch.stack(P_plain).mean(0)
    ens_tta = torch.stack(P_tta).mean(0)

    arch_groups = {}
    for a in ARCHS:
        pre = a.split("_")[0]
        idx = [i for i, r in enumerate(runs)
               if r.startswith(pre) and (CK / r / "best.pt").exists()]
        if idx:
            arch_groups[a] = dict(
                plain=acc(torch.stack([P_plain[i] for i in idx]).mean(0), y_test),
                tta=acc(torch.stack([P_tta[i] for i in idx]).mean(0), y_test))

    oof = []
    for k in FOLDS:
        rid = next((r for r in runs if r.endswith(f"_f{k}")
                    and (CK / r / "best.pt").exists()), None)
        if rid is None:
            continue
        model, size, _ = load_model(rid, device)
        items = split["folds"][k]["val"]
        yv = [y for _, y in items]
        b = {192: 64, 320: 32, 448: 16, 512: 8}.get(size, 16)
        pv = probs(model, "/content", items, size, device, False, b)
        oof.append(dict(fold=k, run=rid, oof_val=acc(pv, yv)))
        L(f"OOF fold{k} ({rid}): {oof[-1]['oof_val']:.2f}%")
        del model
        torch.cuda.empty_cache()

    metrics = dict(
        n_models=len(P_plain),
        singles=singles,
        arch_groups=arch_groups,
        ensemble_plain=acc(ens_plain, y_test),
        ensemble_tta=acc(ens_tta, y_test),
        oof=oof,
        best_single_plain=max(s["plain"] for s in singles),
        best_single_tta=max(s["tta"] for s in singles),
        gate_min=85.0,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        phase="MAX-ENSEMBLE",
    )
    metrics["gate_pass"] = bool(max(metrics["ensemble_plain"],
                                    metrics["ensemble_tta"]) >= 85.0)
    (CK / "metrics_ensemble.json").write_text(json.dumps(metrics, indent=2))
    L(f"ENSEMBLE plain={metrics['ensemble_plain']:.2f}% "
      f"tta={metrics['ensemble_tta']:.2f}% gate={metrics['gate_pass']}")
    return 0


if __name__ == "__main__":
    main()
