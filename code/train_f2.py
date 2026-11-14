#!/usr/bin/env python3
"""F2 crash-safe trainer — salto 76 → ~85.

Cambios vs F1: efficientnet_b3 @300, RandAugment, Mixup, RandomErasing,
batch 32, lr 5e-4, gate 85. Resume con split_hash abort si no cuadra.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_argv = sys.argv[1:]
if sys.argv and not str(sys.argv[0]).endswith("train_f2.py"):
    _argv = []

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CKPT = Path("/content/ckpt")
CKPT.mkdir(parents=True, exist_ok=True)
LOG = CKPT / "train_log.csv"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CKPT / "train.log", "a") as f:
        f.write(line + "\n")


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


def build_tfms(train: bool, img_size: int = 300):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(0.35, 0.35, 0.35, 0.08),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                norm,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.18)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm,
        ]
    )


def ensure_ude_data(data_root: Path, zip_path: Path) -> None:
    inner = data_root / "UDE Diatoms in the Wild 2024" / "images"
    if inner.exists() and any(inner.iterdir()):
        log(f"UDE already present at {inner}")
        return
    url = (
        "https://zenodo.org/api/records/10410655/files/"
        "UDE%20Diatoms%20in%20the%20Wild%202024.zip/content"
    )
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists() or zip_path.stat().st_size < 2_000_000_000:
        log(f"downloading UDE -> {zip_path}")
        rc = os.system(f'wget -c -O "{zip_path}" "{url}" >> /content/ckpt/uedl.log 2>&1')
        if rc != 0:
            raise RuntimeError("wget UDE failed")
    data_root.mkdir(parents=True, exist_ok=True)
    log("unzipping UDE")
    rc = os.system(f'unzip -o -q "{zip_path}" -d "{data_root}" >> /content/ckpt/uedl.log 2>&1')
    if rc != 0:
        raise RuntimeError("unzip UDE failed")
    n = len(list(inner.glob("*"))) if inner.exists() else 0
    if n < 1000:
        raise RuntimeError(f"UDE extract looks empty: {n} images")
    log(f"UDE ready: {n} images at {inner}")


def append_log_row(row: dict) -> None:
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "lr",
                "best_acc",
                "ts",
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def save_ckpt(path: Path, **kw) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(kw, tmp)
    os.replace(tmp, path)


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[index]
    return mixed, y, y[index], lam


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="/content/split_f1.json")
    ap.add_argument("--data-root", default="/content")
    ap.add_argument("--zip", default="/content/ude.zip")
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--model", default="efficientnet_b3")
    ap.add_argument("--img-size", type=int, default=300)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--gate", type=float, default=85.0)
    args = ap.parse_args(_argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__} model={args.model} size={args.img_size}")

    with open(args.split) as f:
        split = json.load(f)
    split_hash = split["hash"]
    classes = split["classes"]
    n_cls = len(classes)
    log(
        f"split hash={split_hash[:12]} classes={n_cls} "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split.get('test', []))}"
    )

    ensure_ude_data(Path(args.data_root), Path(args.zip))

    import timm

    model = timm.create_model(args.model, pretrained=True, num_classes=n_cls)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    total_steps = max(1, (len(split["train"]) // args.batch) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        log(f"RESUME load {args.resume}")
        ck = torch.load(args.resume, map_location="cpu")
        if ck.get("split_hash") != split_hash:
            log("FATAL split_hash mismatch — aborting resume")
            return 2
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"])
        best_acc = float(ck.get("best_acc", 0.0))
        log(f"RESUME from epoch {start_epoch + 1} (best_acc={best_acc:.4f}) → {args.epochs} total")

    train_ds = ListDataset(args.data_root, split["train"], build_tfms(True, args.img_size))
    val_ds = ListDataset(args.data_root, split["val"], build_tfms(False, args.img_size))
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        run_loss = 0.0
        nb = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if args.mixup > 0 and np.random.rand() < 0.5:
                xb, ya, yb2, lam = mixup_data(xb, yb, args.mixup)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb2)
            else:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            nb += 1
        train_loss = run_loss / max(1, nb)

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss_sum += loss.item()
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = 100.0 * correct / max(1, total)
        val_loss = val_loss_sum / max(1, len(val_dl))
        lr_now = opt.param_groups[0]["lr"]
        epoch_num = epoch + 1
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc

        append_log_row(
            dict(
                epoch=epoch_num,
                train_loss=f"{train_loss:.5f}",
                val_loss=f"{val_loss:.5f}",
                val_acc=f"{val_acc:.4f}",
                lr=f"{lr_now:.3e}",
                best_acc=f"{best_acc:.4f}",
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
        )

        common = dict(
            epoch=epoch_num,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            scaler=scaler.state_dict(),
            best_acc=best_acc,
            classes=classes,
            split_hash=split_hash,
            args=vars(args),
            val_acc=val_acc,
        )
        save_ckpt(CKPT / "last.pt", **common)
        if improved:
            save_ckpt(CKPT / "best.pt", **common)

        log(
            f"epoch {epoch_num}/{args.epochs} train={train_loss:.4f} "
            f"val={val_acc:.2f}% best={best_acc:.2f}% lr={lr_now:.2e} "
            f"({'IMPROVED' if improved else 'ok'}) {time.time()-t0:.0f}s"
        )

    test_acc = None
    if split.get("test"):
        model.eval()
        test_ds = ListDataset(args.data_root, split["test"], build_tfms(False, args.img_size))
        test_dl = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers
        )
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        test_acc = 100.0 * correct / max(1, total)

    metrics = dict(
        best_val_acc=best_acc,
        test_acc=test_acc,
        gate_min=args.gate,
        gate_pass=bool(best_acc >= args.gate),
        epochs=args.epochs,
        model=args.model,
        classes=len(classes),
        split_hash=split_hash,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        device=str(device),
        phase="F2",
    )
    with open(CKPT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"DONE metrics={json.dumps(metrics)}")
    log("GATE " + ("PASS" if metrics["gate_pass"] else "FAIL"))
    return 0 if metrics["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
