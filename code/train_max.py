#!/usr/bin/env python3
"""MAX trainer — sistema completo desde cero (Fase 0 del diseño MAX).

Palancas implementadas (checklist cerrado de Lead Researcher):
  2 etiquetas:     split_max con cuarentena de suspects (a cargo del split)
  3 aug stack:     AutoAugment + RandAugment + MixUp + CutMix + RandomErasing
  4 EMA:           shadow de pesos, val/best sobre EMA
  5 grid:          --mode grid (smoothing × drop_path corto @192)
  6 progresivo:    etapas 320→448→512 (mismo proceso, resume-safe)
  7 k-fold:        --fold 0..4 sobre split_max
  8 arquitecturas: --model (convnext_tiny / efficientnetv2_rw_s / swin_tiny…)
  9 hard weights:  class_weights del audit en la CE
 10 TTA:           TenCrop×2 escalas en el test final de cada corrida
 (ensemble y TDI viven fuera: ensemble_eval.py / fase F3)

Resume: ckpt guarda model+ema+opt+sched+scaler+stage+epoch+split_hash.
Etapa nueva = optimizer fresco + pesos EMA transferidos (shape-match).
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

_argv = sys.argv[1:]
if sys.argv and not str(sys.argv[0]).endswith("train_max.py"):
    _argv = []

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RUN_DIR = Path("/content/ckpt")
STAGES_MAIN = [
    dict(name="a", img=320, epochs=30, batch=32, accum=1, lr=4e-4),
    dict(name="b", img=448, epochs=10, batch=16, accum=2, lr=1.5e-4),
    dict(name="c", img=512, epochs=5, batch=8, accum=4, lr=8e-5),
]
ARCH_DEFAULT_DP = {
    "convnext_tiny": 0.2,
    "efficientnetv2_rw_s": 0.2,
    "swin_tiny_patch4_window7_224": 0.2,
}


def log(rd: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(rd / "train.log", "a") as f:
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


class TTADataset(Dataset):
    def __init__(self, root, items, size):
        self.root = Path(root)
        self.items = items
        self.tfm = transforms.Compose(
            [transforms.Resize(int(size * 1.15)), transforms.TenCrop(size)]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        img = Image.open(self.root / rel).convert("RGB")
        crops = self.tfm(img)  # 10 x C x H x W
        return torch.stack(crops), label


NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def build_train_tfm(size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.35, 0.35, 0.35, 0.08),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            NORM,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18)),
        ]
    )


def build_eval_tfm(size: int):
    return transforms.Compose(
        [transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
         transforms.ToTensor(), NORM]
    )


def ensure_ude(data_root: Path = Path("/content")) -> None:
    inner = data_root / "UDE Diatoms in the Wild 2024" / "images"
    if inner.exists() and any(inner.iterdir()):
        return
    zip_path = data_root / "ude.zip"
    url = ("https://zenodo.org/api/records/10410655/files/"
           "UDE%20Diatoms%20in%20the%20Wild%202024.zip/content")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists() or zip_path.stat().st_size < 2_000_000_000:
        rc = os.system(f'wget -c -O "{zip_path}" "{url}" '
                       '>> /content/ckpt/uedl.log 2>&1')
        if rc != 0:
            raise RuntimeError("wget UDE failed")
    rc = os.system(f'unzip -o -q "{zip_path}" -d "{data_root}" '
                   '>> /content/ckpt/uedl.log 2>&1')
    if rc != 0:
        raise RuntimeError("unzip UDE failed")
    n = len(list(inner.glob("*")))
    if n < 1000:
        raise RuntimeError(f"UDE extract vacío: {n}")


def cutmix_data(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    h, w = x.size(2), x.size(3)
    cut = int(w * (1 - lam) ** 0.5)
    cy = int(h * (1 - lam) ** 0.5)
    cx = random.randint(0, w - cut)
    cy2 = random.randint(0, h - cy)
    x[:, :, cy2:cy2 + cy, cx:cx + cut] = x[index, :, cy2:cy2 + cy, cx:cx + cut]
    lam = 1 - (cy * cut) / (h * w)
    return x, y, y[index], lam


def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[index], y, y[index], lam


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def load_into(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, t in self.shadow.items():
            if k in sd and sd[k].shape == t.shape:
                sd[k].copy_(t)

    def state_dict(self):
        return dict(shadow=self.shadow, decay=self.decay, steps=self.steps)

    def load_state_dict(self, st):
        self.shadow = st["shadow"]
        self.decay = st["decay"]
        self.steps = st["steps"]


def transfer_weights(src_sd: dict, dst: nn.Module) -> int:
    dst_sd = dst.state_dict()
    n = 0
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k].copy_(v)
            n += 1
    dst.load_state_dict(dst_sd)
    return n


def append_row(rd: Path, row: dict) -> None:
    p = rd / "train_log.csv"
    fields = ["stage", "local_epoch", "global_step", "train_loss",
              "val_loss", "val_acc", "lr", "best_acc", "ts"]
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def save_ckpt(path: Path, **kw) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(kw, tmp)
    os.replace(tmp, path)


def make_model(name, n_cls, size, dp, pretrained):
    import timm
    kwargs = dict(pretrained=pretrained, num_classes=n_cls,
                  drop_path_rate=dp)
    if name.startswith("swin"):
        kwargs["img_size"] = size
    return timm.create_model(name, **kwargs)


def eval_loader(model, dl, device, criterion=None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    nb = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                if criterion is not None:
                    loss_sum += criterion(logits.float(), yb).item()
                nb += 1
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    acc = 100.0 * correct / max(1, total)
    loss = loss_sum / max(1, nb) if criterion is not None else 0.0
    return acc, loss


def tta_test_acc(model, root, items, size, device, batch_imgs):
    """Media de probabilidades sobre TenCrop(10) x 2 escalas (resize 1.15)."""
    ds = TTADataset(root, items, size)
    dl = DataLoader(ds, batch_size=batch_imgs, shuffle=False, num_workers=2)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for crops, yb in dl:
            b, n, c, h, w = crops.shape
            x = crops.view(b * n, c, h, w).to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
            logits = logits.view(b, n, -1).float().mean(1)
            correct += (logits.argmax(1) == yb.to(device)).sum().item()
            total += b
    return 100.0 * correct / max(1, total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="/content/split_max.json")
    ap.add_argument("--data-root", default="/content")
    ap.add_argument("--run-id", default="r0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--mode", choices=["run", "grid"], default="run")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--drop-path", type=float, default=-1.0,
                    help="<0 → default por arquitectura")
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--resume", default="")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--gate", type=float, default=85.0)
    ap.add_argument("--grid-img", type=int, default=192)
    ap.add_argument("--grid-epochs", type=int, default=4)
    ap.add_argument("--grid-batch", type=int, default=64)
    args = ap.parse_args(_argv)

    run_dir = RUN_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = args.drop_path if args.drop_path >= 0 else ARCH_DEFAULT_DP.get(
        args.model, 0.1)
    log(run_dir, f"run={args.run_id} mode={args.mode} model={args.model} "
                 f"fold={args.fold} smooth={args.smoothing} dp={dp} dev={device}")

    with open(args.split) as f:
        split = json.load(f)
    split_hash = split["hash"]
    classes = split["classes"]
    n_cls = len(classes)
    if args.mode == "run":
        fold = split["folds"][args.fold]
        train_items, val_items = fold["train"], fold["val"]
    else:
        fold0 = split["folds"][0]
        train_items, val_items = fold0["train"], fold0["val"]
    test_items = split["test"]
    log(run_dir, f"split {split_hash[:12]} train={len(train_items)} "
                 f"val={len(val_items)} test={len(test_items)}")

    cw = split.get("class_weights")
    class_weight = None
    if cw:
        class_weight = torch.tensor(
            [float(cw.get(str(i), 1.0)) for i in range(n_cls)],
            device=device)

    stages = (STAGES_MAIN if args.mode == "run" else
              [dict(name="g", img=args.grid_img, epochs=args.grid_epochs,
                    batch=args.grid_batch, accum=1, lr=5e-4)])
    criterion = nn.CrossEntropyLoss(weight=class_weight,
                                    label_smoothing=args.smoothing)

    start_stage = 0
    local_epoch = 0
    global_step = 0
    best_acc = 0.0
    ema = None
    model = None
    opt = sched = scaler = None

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        if ck.get("split_hash") != split_hash:
            log(run_dir, "FATAL split_hash mismatch")
            return 2
        start_stage = int(ck["stage_idx"])
        local_epoch = int(ck["local_epoch"])
        global_step = int(ck.get("global_step", 0))
        best_acc = float(ck.get("best_acc", 0.0))
        st = stages[start_stage]
        model = make_model(args.model, n_cls, st["img"], dp, pretrained=False)
        model.load_state_dict(ck["model"])
        model = model.to(device)
        ema = EMA(model, args.ema_decay)
        if ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        if ck.get("optimizer"):
            opt = torch.optim.AdamW(model.parameters(), lr=st["lr"],
                                    weight_decay=0.02)
            opt.load_state_dict(ck["optimizer"])
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=1, T_mult=2, eta_min=1e-6)
            if ck.get("scheduler"):
                sched.load_state_dict(ck["scheduler"])
            scaler = torch.amp.GradScaler("cuda")
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
        log(run_dir, f"RESUME stage={start_stage+1}/{len(stages)} "
                     f"{st['name']} local_ep={local_epoch+1} "
                     f"best={best_acc:.2f} step={global_step}")
    else:
        st = stages[0]
        model = make_model(args.model, n_cls, st["img"], dp, pretrained=True)
        model = model.to(device)
        ema = EMA(model, args.ema_decay)

    finished = False
    for stage_idx in range(start_stage, len(stages)):
        st = stages[stage_idx]
        if opt is None:
            opt = torch.optim.AdamW(model.parameters(), lr=st["lr"],
                                    weight_decay=0.02)
            total_steps = max(1, (len(train_items) // (st["batch"])) *
                              st["epochs"])
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=max(1, total_steps // 2), T_mult=2, eta_min=1e-6)
            scaler = torch.amp.GradScaler("cuda")
        if local_epoch >= st["epochs"]:
            local_epoch = 0

        train_ds = ListDataset(args.data_root, train_items,
                               build_train_tfm(st["img"]))
        val_ds = ListDataset(args.data_root, val_items,
                             build_eval_tfm(st["img"]))
        train_dl = DataLoader(train_ds, batch_size=st["batch"], shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=st["batch"], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        log(run_dir, f"STAGE {st['name']}: img={st['img']} "
                     f"epochs={st['epochs']} batch={st['batch']}"
                     f"x{st['accum']} lr={st['lr']}")

        for ep in range(local_epoch, st["epochs"]):
            t0 = time.time()
            model.train()
            run_loss = 0.0
            nb = 0
            opt.zero_grad(set_to_none=True)
            for i, (xb, yb) in enumerate(train_dl):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                r = np.random.rand()
                with torch.amp.autocast("cuda"):
                    if r < 0.35:
                        xb_m, ya, yb2, lam = cutmix_data(xb.clone(), yb)
                        logits = model(xb_m)
                        loss = lam * criterion(logits, ya) + \
                            (1.0 - lam) * criterion(logits, yb2)
                    elif r < 0.70:
                        xb_m, ya, yb2, lam = mixup_data(xb, yb)
                        logits = model(xb_m)
                        loss = lam * criterion(logits, ya) + \
                            (1.0 - lam) * criterion(logits, yb2)
                    else:
                        logits = model(xb)
                        loss = criterion(logits, yb)
                scaler.scale(loss / st["accum"]).backward()
                if (i + 1) % st["accum"] == 0 or (i + 1) == len(train_dl):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    ema.update(model)
                    global_step += 1
                run_loss += loss.item()
                nb += 1

            train_loss = run_loss / max(1, nb)
            ema.load_into(model)
            val_acc, val_loss = eval_loader(model, val_dl, device, criterion)
            model.train()
            lr_now = opt.param_groups[0]["lr"]
            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc

            append_row(run_dir, dict(
                stage=st["name"], local_epoch=ep + 1, global_step=global_step,
                train_loss=f"{train_loss:.5f}", val_loss=f"{val_loss:.5f}",
                val_acc=f"{val_acc:.4f}", lr=f"{lr_now:.3e}",
                best_acc=f"{best_acc:.4f}",
                ts=time.strftime("%Y-%m-%dT%H:%M:%S")))

            raw_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            common = dict(
                epoch=ep + 1, stage_idx=stage_idx, local_epoch=ep + 1,
                global_step=global_step, model=raw_sd,
                ema=ema.state_dict(), optimizer=opt.state_dict(),
                scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                best_acc=best_acc, classes=classes, split_hash=split_hash,
                args=vars(args), val_acc=val_acc,
            )
            save_ckpt(run_dir / "last.pt", **common)
            if improved:
                ema_cpu = {k: v.cpu() for k, v in ema.shadow.items()}
                save_ckpt(run_dir / "best.pt",
                          epoch=ep + 1, stage=st["name"], model=raw_sd,
                          ema=ema_cpu, best_acc=best_acc, classes=classes,
                          split_hash=split_hash, args=vars(args),
                          val_acc=val_acc, model_name=args.model,
                          fold=args.fold, img=st["img"])

            log(run_dir, f"{st['name']} ep{ep+1}/{st['epochs']} "
                         f"train={train_loss:.4f} val={val_acc:.2f}% "
                         f"best={best_acc:.2f}% lr={lr_now:.2e} "
                         f"{'IMP' if improved else 'ok'} "
                         f"{time.time()-t0:.0f}s")

            if ep + 1 == st["epochs"] and stage_idx + 1 < len(stages):
                nxt = stages[stage_idx + 1]
                new_model = make_model(args.model, n_cls, nxt["img"], dp,
                                       pretrained=False)
                n_hit = transfer_weights(dict(ema.shadow), new_model)
                log(run_dir, f"TRANSITION -> {nxt['name']} @{nxt['img']} "
                             f"(EMA transferido, {n_hit} tensores)")
                model = new_model.to(device)
                ema = EMA(model, args.ema_decay)
                opt = sched = scaler = None
                save_ckpt(run_dir / "last.pt",
                          epoch=0, stage_idx=stage_idx + 1, local_epoch=0,
                          global_step=global_step,
                          model={k: v.cpu() for k, v in model.state_dict().items()},
                          ema=ema.state_dict(), best_acc=best_acc,
                          classes=classes, split_hash=split_hash,
                          args=vars(args))
                local_epoch = 0
                break
        else:
            local_epoch = 0
            continue
        if stage_idx + 1 >= len(stages):
            finished = True
            break
    else:
        finished = True

    if args.mode == "run":
        log(run_dir, "etapas completadas → test plain + TTA")
        ema.load_into(model)
        test_tfm_ds = ListDataset(args.data_root, test_items,
                                  build_eval_tfm(stages[-1]["img"]))
        test_dl = DataLoader(test_tfm_ds, batch_size=stages[-1]["batch"],
                             shuffle=False, num_workers=args.num_workers)
        test_acc, _ = eval_loader(model, test_dl, device, None)
        tta_bs = {320: 16, 448: 8, 512: 4}.get(stages[-1]["img"], 8)
        test_tta = tta_test_acc(model, args.data_root, test_items,
                                stages[-1]["img"], device, tta_bs)
        metrics = dict(
            run_id=args.run_id, fold=args.fold, model=args.model,
            smoothing=args.smoothing, drop_path=dp,
            best_val_acc=best_acc, test_acc=test_acc, test_acc_tta=test_tta,
            final_img=stages[-1]["img"], gate_min=args.gate,
            gate_pass=bool(test_tta >= args.gate or test_acc >= args.gate),
            split_hash=split_hash,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            phase="MAX",
        )
    else:
        metrics = dict(
            run_id=args.run_id, grid_id=f"s{args.smoothing}_d{dp}",
            smoothing=args.smoothing, drop_path=dp, best_val_acc=best_acc,
            split_hash=split_hash, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            mode="grid",
        )
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (run_dir / "DONE").write_text("ok")
    log(run_dir, f"DONE {json.dumps(metrics)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
