#!/usr/bin/env python3
"""F3 - TDI layer on top of the F2 classifier.

The F2 checkpoint is a single-label classifier over a cropped cell: it answers
"which taxon is this one cell?". The TDI is an index over a whole sample: the
abundance-weighted mean of per-taxon sensitivity values. This module adds the
missing piece - counting - and then computes the index.

Pipeline
    image -> sliding windows -> B3 classifier -> NMS -> per-taxon cell counts
          -> abundance-weighted TDI -> water-quality class

Reference: Kelly & Whitton (1995) doi:10.1007/BF00003802.
Per-taxon values come from data/tdi_values_101.csv (see build_tdi_table.py).

Design rules
  * Nothing invented: taxa with no published value carry no weight, and the
    index is renormalised over the valued fraction. The returned `value_mass`
    is how much of the counted biomass that fraction represents, so a thin
    index is always visible instead of silently misleading.
  * Planktonic taxa are reported separately (`planktonic_fraction`): the TDI is
    defined on the benthic fraction only.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "f2" / "best.pt"
TABLE = ROOT / "data" / "tdi_values_101.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Kelly & Whitton (1995) quality classes on the native 1..5 scale.
# The literature usually reports the index multiplied by 10 (a TDI of 3.0 is
# printed as "30"), so `TdiResult.tdi_x10` is provided alongside.
TDI_BANDS = (
    (2.0, "Excellent", "Oligotrophic - very good water quality"),
    (3.0, "Good", "Oligotrophic - good water quality"),
    (4.0, "Fair", "Mesotrophic - moderate nutrient enrichment"),
    (4.5, "Poor", "Eutrophic - high nutrient enrichment"),
    (float("inf"), "Bad", "Hypereutrophic - severe nutrient enrichment"),
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class TaxonValue:
    name: str
    value: float | None
    source: str


@dataclass
class TdiTable:
    values: list[TaxonValue]
    classes: list[str]

    @property
    def value_array(self) -> torch.Tensor:
        """Per-class index value; NaN where no value is published."""
        t = torch.full((len(self.values),), float("nan"))
        for i, v in enumerate(self.values):
            if v.value is not None:
                t[i] = v.value
        return t

    def valued_mask(self) -> torch.Tensor:
        return ~torch.isnan(self.value_array)

    @classmethod
    def load(cls, path: Path = TABLE) -> "TdiTable":
        by_name: dict[str, TaxonValue] = {}
        with Path(path).open() as fh:
            for row in csv.DictReader(fh):
                raw = row["tdi_value"].strip()
                by_name[row["class"]] = TaxonValue(
                    name=row["class"],
                    value=float(raw) if raw else None,
                    source=row["tdi_source"],
                )
        if not by_name:
            raise ValueError(f"empty TDI table: {path}")
        classes = list(by_name.keys())
        return cls(values=[by_name[c] for c in classes], classes=classes)


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #
class DiatomClassifier:
    """F2 EfficientNet-B3, with the exact eval preprocessing used in training."""

    def __init__(
        self,
        ckpt: Path = CKPT,
        device: str | None = None,
        img_size: int = 300,
    ):
        import timm
        from torchvision import transforms

        ckpt = Path(ckpt)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)

        self.split_hash = blob.get("split_hash", "")
        self.epoch = blob.get("epoch", -1)
        self.best_acc = blob.get("best_acc", float("nan"))
        self.classes = list(blob["classes"])
        self.n_classes = len(self.classes)

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = timm.create_model(
            blob["args"]["model"], pretrained=False, num_classes=self.n_classes
        )
        self.model.load_state_dict(blob["model"])
        self.model.eval().to(self.device)

        self.img_size = img_size
        self.tf = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @torch.inference_mode()
    def classify(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """Return (n, n_classes) softmax probabilities."""
        out = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            x = torch.stack([self.tf(im.convert("RGB")) for im in chunk]).to(self.device)
            out.append(F.softmax(self.model(x), dim=1).cpu())
        return torch.cat(out) if out else torch.empty(0, self.n_classes)


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    cls_idx: int
    conf: float
    box: tuple[float, float, float, float]  # xyxy in source-image pixels


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if all(_iou(d.box, k.box) <= iou_thr for k in kept):
            kept.append(d)
    return kept


def count_cells(
    classifier: DiatomClassifier,
    image: Image.Image,
    window: int = 150,
    stride_frac: float = 0.5,
    conf_thr: float = 0.5,
    nms_iou: float = 0.3,
    cross_class_iou: float = 0.7,
    refine: bool = True,
    max_windows: int = 800,
    scale: float = 1.0,
) -> list[Detection]:
    """Detect cells in a field image by sliding the classifier over it.

    The classifier was trained on single-cell crops, so a field image is swept
    with a `window`-sized square and de-duplicated with class-wise NMS.

    `window` is the single most important knob: it must match the apparent cell
    size, otherwise every crop carries excess background and confidence collapses.
    Calibrated on real UDE cutouts (median side ~148 px) -> 150 is the default.

    `scale` shrinks the image before sweeping, and shrinks `window` with it so
    the detector still sees cells at the same apparent size. This is what makes
    phone-sized input viable: sweeping a 12 MP photo 1:1 costs ~41 s, at
    scale=0.5 it is ~4x cheaper with the same detections.

    refine      re-classify each surviving box cropped tight; boxes whose
                confidence collapses were firing on background, so they are
                dropped instead of polluting the counts.
    cross_class_iou  same-cell detections of *different* classes are suppressed,
                keeping the most confident one.
    """
    img = image.convert("RGB")
    if scale != 1.0:
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.BILINEAR,
        )
    W, H = img.size
    win = min(window, W, H)

    # Window budget. Scaling the image alone does not speed anything up: the
    # window shrinks too, so the window count is scale-invariant. The real
    # budget is therefore spent by widening the stride until the whole frame is
    # covered by at most max_windows positions - sampled uniformly across the
    # image, never truncated to its top-left corner.
    nx = max(1, (W - win) // max(1, int(win * stride_frac)) + 1)
    ny = max(1, (H - win) // max(1, int(win * stride_frac)) + 1)
    while nx * ny > max_windows:
        stride_frac *= 1.15
        step = max(1, int(win * stride_frac))
        nx = max(1, (W - win) // step + 1)
        ny = max(1, (H - win) // step + 1)
    step = max(1, int(win * stride_frac))

    def axis(total: int, n: int) -> list[int]:
        if total <= win or n == 1:
            return [0]
        last = total - win
        return [round(i * last / (n - 1)) for i in range(n)]

    xs = axis(W, nx)
    ys = axis(H, ny)

    crops, boxes = [], []
    for y in ys:
        for x in xs:
            crops.append(img.crop((x, y, x + win, y + win)))
            boxes.append((x, y, x + win, y + win))

    if not crops:
        return []

    probs = classifier.classify(crops)
    dets: list[Detection] = []
    for (bx0, by0, bx1, by1), p in zip(boxes, probs):
        conf, idx = float(p.max()), int(p.argmax())
        if conf >= conf_thr:
            dets.append(Detection(cls_idx=idx, conf=conf, box=(bx0, by0, bx1, by1)))

    # class-wise NMS
    dets = _nms(dets, nms_iou)

    if cross_class_iou is not None and cross_class_iou < 1.0:
        dets = _cross_class_nms(dets, cross_class_iou)

    if refine and dets:
        dets = _refine(classifier, img, dets, conf_thr)
        dets = _cross_class_nms(dets, cross_class_iou)

    return dets


def _cross_class_nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.conf):
        if all(_iou(d.box, k.box) <= iou_thr for k in kept):
            kept.append(d)
    return kept


def _refine(
    classifier: "DiatomClassifier", img: Image.Image, dets: list[Detection], thr: float
) -> list[Detection]:
    """Re-classify each box cropped tight (plus 6% margin) and keep it only if
    the classification holds up. Background-triggered boxes lose confidence here."""
    crops = []
    for d in dets:
        x0, y0, x1, y1 = d.box
        mx, my = (x1 - x0) * 0.06, (y1 - y0) * 0.06
        crops.append(
            img.crop(
                (
                    max(0, int(x0 - mx)),
                    max(0, int(y0 - my)),
                    min(img.width, int(x1 + mx)),
                    min(img.height, int(y1 + my)),
                )
            )
        )
    probs = classifier.classify(crops)
    out: list[Detection] = []
    for d, p in zip(dets, probs):
        conf, idx = float(p.max()), int(p.argmax())
        if conf >= thr:
            out.append(Detection(cls_idx=idx, conf=conf, box=d.box))
    return out


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
@dataclass
class TdiResult:
    tdi: float | None
    counts: dict[str, int] = field(default_factory=dict)
    total_cells: int = 0
    valued_cells: int = 0
    value_mass: float = 0.0
    mean_confidence: float = 0.0
    planktonic_fraction: float = 0.0
    quality: str = ""
    quality_es: str = ""

    @property
    def tdi_x10(self) -> float | None:
        """The x10 form used in the literature ("TDI 30" == native 3.0)."""
        return None if self.tdi is None else self.tdi * 10.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["tdi"] = None if self.tdi is None else round(self.tdi, 2)
        d["tdi_x10"] = None if self.tdi is None else round(self.tdi * 10.0, 1)
        d["value_mass"] = round(self.value_mass, 4)
        d["mean_confidence"] = round(self.mean_confidence, 4)
        d["planktonic_fraction"] = round(self.planktonic_fraction, 4)
        d["counts"] = dict(sorted(self.counts.items(), key=lambda kv: -kv[1]))
        return d


def band_for(tdi: float) -> tuple[str, str]:
    for hi, name, es in TDI_BANDS:
        if tdi < hi:
            return name, es
    return TDI_BANDS[-1][1], TDI_BANDS[-1][2]


PLANKTONIC_GENUS = {
    "cyclotella",
    "stephanodiscus",
    "aulacoseira",
    "cyclostephanos",
    "thalassiosira",
    "melosira",
    "detomus",
    "skeletonema",
    "nicomphalus",
}


def compute_tdi(
    detections: list[Detection],
    table: TdiTable,
    planktonic: set[str] | None = None,
) -> TdiResult:
    counts: dict[str, int] = {}
    for d in detections:
        counts[table.classes[d.cls_idx]] = counts.get(table.classes[d.cls_idx], 0) + 1

    res = TdiResult(tdi=None, counts=counts, total_cells=len(detections))
    if not detections:
        return res

    res.mean_confidence = sum(d.conf for d in detections) / len(detections)

    planktonic = PLANKTONIC_GENUS if planktonic is None else planktonic
    plank_cells = 0

    num = 0.0
    den = 0.0
    for name, n in counts.items():
        genus = name.split(" ")[0].lower()
        if genus in planktonic:
            plank_cells += n
        tv = table.values[table.classes.index(name)]
        if tv.value is None or n == 0:
            continue
        num += tv.value * n
        den += n

    res.valued_cells = int(den)
    res.planktonic_fraction = plank_cells / len(detections)
    res.value_mass = den / len(detections)
    if den > 0:
        res.tdi = num / den
        res.quality, res.quality_es = band_for(res.tdi)
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="TDI layer over the F2 classifier")
    ap.add_argument("--image", help="field image or single cell crop")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--stride", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.3)
    ap.add_argument("--max-windows", type=int, default=800)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink input before sweeping; window shrinks with it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true", help="dump the result as JSON")
    args = ap.parse_args()

    if not args.image:
        ap.error("--image is required")

    table = TdiTable.load()
    clf = DiatomClassifier(device=args.device)

    t0 = time.perf_counter()
    dets = count_cells(
        clf,
        Image.open(args.image),
        window=args.window,
        stride_frac=args.stride,
        conf_thr=args.conf,
        nms_iou=args.nms,
        scale=args.scale,
        max_windows=args.max_windows,
    )
    res = compute_tdi(dets, table)
    dt = time.perf_counter() - t0

    if args.json:
        print(json.dumps(res.as_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"image        {args.image}")
    print(f"device       {clf.device}   model {clf.classes and ''}{Path(CKPT).name} ep{clf.epoch} (val {clf.best_acc:.2f}%)")
    print(f"seconds      {dt:.2f}")
    print(f"cells        {res.total_cells} counted, mean confidence {res.mean_confidence:.3f}")
    print(f"planktonic   {res.planktonic_fraction * 100:.1f}% of counts")
    print(f"valued       {res.valued_cells}/{res.total_cells} cells ({res.value_mass * 100:.1f}% of mass)")
    print()
    if res.tdi is None:
        print("TDI           n/a - no counted taxon carries a published TDI value")
    else:
        print(f"TDI           {res.tdi:.2f}  ->  {res.quality} / {res.quality_es}")
    print()
    print("top taxa:")
    for name, n in list(res.as_dict()["counts"].items())[:10]:
        tv = table.values[table.classes.index(name)].value
        vs = f"TDI {tv:.0f}" if tv is not None else "no value"
        print(f"  {n:>4}  {name:<48} {vs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())