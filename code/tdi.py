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
Per-taxon values (sensitivity `s` and indicator weight `v`) come from
data/tdi_values_101.csv (see build_tdi_table.py), sourced from the `tdi`
dataset of the CRAN package `diathor`.

Index definition (the one used by the reference implementation,
`diathor/R/diat_tdi.R`):

    TDI_raw = sum(a_i * s_i * v_i) / sum(a_i * v_i)        # 1..5
    TDI_100 = 25 * TDI_raw - 25                            # 0..100

Kelly & Whitton quality classes are defined on the 0..100 form:
    <35 high, 35-50 good, 50-65 moderate, 65-85 poor, >85 bad

Design rules
  * Nothing invented: taxa with no published value (or no published indicator
    weight) carry no weight, and the index is renormalised over the valued
    fraction. The returned `value_mass` is how much of the counted biomass that
    fraction represents, so a thin index is always visible instead of silently
    misleading.
  * Planktonic taxa are excluded from the index (`planktonic_fraction` reports
    their share): the TDI is defined on the benthic fraction only.
  * The morphometric prior is only meaningful when the crop aspect ratio is the
    cell's aspect ratio. Sliding-window counting feeds square windows, where it
    degenerates into a constant per-class bias, so it is off by default.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Packaging hooks. When this module is imported straight out of the single-file
# .pyz, there is no repository around it, so __main__.py extracts the resources
# to a temp dir and points DIATOM_ROOT there (and DIATOM_CKPT at the checkpoint).
# Left unset, the normal repository layout applies.
import os

ROOT = Path(os.environ["DIATOM_ROOT"]) if os.environ.get("DIATOM_ROOT") else Path(__file__).resolve().parent.parent
_ckpt_env = os.environ.get("DIATOM_CKPT")
CKPT_EDITED = Path(_ckpt_env) if _ckpt_env else ROOT / "checkpoints" / "f2_edited" / "best_edited.pt"
CKPT_BASE = ROOT / "checkpoints" / "f2" / "best.pt"
CKPT = CKPT_EDITED if CKPT_EDITED.exists() else CKPT_BASE
TABLE = ROOT / "data" / "tdi_values_101.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Kelly & Whitton quality classes on the published 0..100 scale.
TDI_BANDS = (
    (35.0, "Excellent", "Oligotrophic - very good water quality"),
    (50.0, "Good", "Oligotrophic - good water quality"),
    (65.0, "Fair", "Mesotrophic - moderate nutrient enrichment"),
    (85.0, "Poor", "Eutrophic - high nutrient enrichment"),
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
    indicator: float | None = None


@dataclass
class TdiTable:
    values: list[TaxonValue]
    classes: list[str]

    @property
    def value_array(self) -> torch.Tensor:
        """Per-class sensitivity `s`; NaN where no value is published."""
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
                raw_v = (row.get("indicator_value") or "").strip()
                by_name[row["class"]] = TaxonValue(
                    name=row["class"],
                    value=float(raw) if raw else None,
                    source=row["tdi_source"],
                    indicator=float(raw_v) if raw_v else None,
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

        # Camera-ready morphometric prior. Only meaningful for single-cell crops,
        # where the image aspect ratio is the cell aspect ratio.
        self.morph_file = ROOT / "data" / "morphometric_prior_101.json"
        if self.morph_file.exists():
            try:
                m_data = json.loads(self.morph_file.read_text())
                self.morph_mu = torch.tensor(m_data["mu"], device=self.device)
                self.morph_sigma = torch.tensor(m_data["sigma"], device=self.device)
                self.morph_weight = float(m_data.get("weight", 0.20))
            except Exception:
                self.morph_mu = None
        else:
            self.morph_mu = None

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
    def classify(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        tta: bool = False,
        morph_prior: bool = False,
    ) -> torch.Tensor:
        """Return (n, n_classes) softmax probabilities.

        `tta` averages the prediction with its horizontal flip.
        `morph_prior` gates predictions with the biological aspect-ratio
        distribution; it must only be enabled when each image is a single cell
        crop, because the prior is a function of the crop aspect ratio.
        """
        out = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            x = torch.stack([self.tf(im.convert("RGB")) for im in chunk]).to(self.device)
            probs = F.softmax(self.model(x), dim=1)
            if tta:
                x_flip = torch.flip(x, dims=[-1])
                probs_flip = F.softmax(self.model(x_flip), dim=1)
                probs = (probs + probs_flip) * 0.5

            if morph_prior and self.morph_mu is not None:
                aspects = torch.tensor(
                    [max(im.size) / max(min(im.size), 1) for im in chunk],
                    device=self.device,
                    dtype=torch.float32,
                )
                z_scores = torch.abs(aspects.unsqueeze(1) - self.morph_mu.unsqueeze(0)) / self.morph_sigma.unsqueeze(0)
                penalty = torch.clamp(z_scores - 2.0, min=0.0) ** 2
                probs = F.softmax(torch.log(probs + 1e-9) - self.morph_weight * penalty, dim=1)

            out.append(probs.cpu())
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


def sweep_plan(
    width: int, height: int, window: int, stride_frac: float, max_windows: int
) -> tuple[list[int], list[int], int]:
    """Window origins that cover the whole frame within a budget.

    The naive sweep truncates at `max_windows`, so a 12 MP photo would only be
    inspected in its top-left corner. Instead the stride is widened until the
    frame fits the budget, then the origins are spread evenly from edge to edge
    (`axis`), so the last window always touches the opposite border.

    Returns (xs, ys, step_px). `step_px > int(window * stride_frac)` means the
    grid is sparser than requested - callers must report that, not hide it.
    """
    win = max(1, min(window, width, height))
    requested = max(1, int(win * stride_frac))
    nx = max(1, (width - win) // requested + 1)
    ny = max(1, (height - win) // requested + 1)
    step = requested
    frac = stride_frac
    while nx * ny > max_windows:
        frac *= 1.15
        step = max(1, int(win * frac))
        nx = max(1, (width - win) // step + 1)
        ny = max(1, (height - win) // step + 1)

    def axis(total: int, n: int) -> list[int]:
        if total <= win or n == 1:
            return [0]
        last = total - win
        return [round(i * last / (n - 1)) for i in range(n)]

    return axis(width, nx), axis(height, ny), step


def count_cells(
    classifier: DiatomClassifier,
    image: Image.Image,
    window: int = 150,
    stride_frac: float = 0.5,
    conf_thr: float = 0.5,
    nms_iou: float = 0.3,
    cross_class_iou: float = 0.7,
    merge: bool = True,
    refine: bool = False,
    max_windows: int = 1200,
    scale: float = 1.0,
) -> list[Detection]:
    """Detect cells in a field image by sliding the classifier over it.

    The classifier was trained on single-cell crops, so a field image is swept
    with a `window`-sized square and de-duplicated in three steps:

      1. class-wise NMS       removes overlapping boxes of the same class;
      2. `_merge_fragments`   joins the contiguous pieces of one cell that the
                              window cut apart (a cell wider than `window` is
                              seen in two or more adjacent windows);
      3. cross-class NMS      removes the remaining duplicates when the model
                              changed its mind between neighbouring windows.

    `window` is the single most important knob: it must match the apparent cell
    size, otherwise every crop carries excess background and confidence collapses.
    Calibrated on real UDE cutouts (median side ~148 px) -> 150 is the default.

    `scale` shrinks the image before sweeping, and shrinks `window` with it so
    the detector still sees cells at the same apparent size. This is what makes
    phone-sized input viable: sweeping a 12 MP photo 1:1 costs ~41 s, at
    scale=0.5 it is ~4x cheaper with the same detections.

    `refine` is off by default: re-cropping a box and re-classifying it loses
    valid cells whose confidence was already close to the threshold, and it
    deforms elongated boxes. It is kept as an option for callers that want it.

    The morphometric aspect-ratio prior is deliberately NOT applied here: every
    window is square, so the aspect ratio is 1.0 and the prior degenerates into
    a constant per-class bias.
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
    xs, ys, step = sweep_plan(W, H, win, stride_frac, max_windows)

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

    dets = _nms(dets, nms_iou)

    if merge and dets:
        dets = _merge_fragments(dets, win)

    if cross_class_iou is not None and cross_class_iou < 1.0:
        dets = _cross_class_nms(dets, cross_class_iou)

    if refine and dets:
        dets = _refine(classifier, img, dets, conf_thr)
        if merge:
            dets = _merge_fragments(dets, win)
        dets = _cross_class_nms(dets, cross_class_iou)

    return dets


def _merge_fragments(
    dets: list[Detection],
    win: int,
    gap_frac: float = 0.18,
    min_perp: float = 0.45,
    max_span: float = 4.0,
) -> list[Detection]:
    """Join the pieces of one cell that the sliding window cut apart.

    A cell larger than `win` is seen by two or more adjacent windows. Those
    boxes are contiguous rather than overlapping, so NMS keeps them and the cell
    is counted once per piece. Two boxes are joined when they carry the same
    class, sit within `gap_frac * win` of each other, and overlap on the
    perpendicular axis. The union is capped at `max_span * win` so a chain of
    background boxes cannot swallow the frame.
    """
    n = len(dets)
    if n < 2:
        return dets
    parent = list(range(n))
    box = {i: list(dets[i].box) for i in range(n)}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gap = gap_frac * win
    for i in range(n):
        for j in range(i + 1, n):
            if dets[i].cls_idx != dets[j].cls_idx:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            ax0, ay0, ax1, ay1 = box[ri]
            bx0, by0, bx1, by1 = box[rj]
            wa, ha = ax1 - ax0, ay1 - ay0
            wb, hb = bx1 - bx0, by1 - by0
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            near = (ax1 + gap >= bx0 and bx1 + gap >= ax0
                    and ay1 + gap >= by0 and by1 + gap >= ay0)
            if not near:
                continue
            if not (oy >= min_perp * min(ha, hb) or ox >= min_perp * min(wa, wb)):
                continue
            ux0, uy0 = min(ax0, bx0), min(ay0, by0)
            ux1, uy1 = max(ax1, bx1), max(ay1, by1)
            if ux1 - ux0 > max_span * win or uy1 - uy0 > max_span * win:
                continue
            parent[rj] = ri
            box[ri] = [ux0, uy0, ux1, uy1]

    groups: dict[int, list[Detection]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(dets[i])
    out: list[Detection] = []
    for g in groups.values():
        best = max(g, key=lambda d: d.conf)
        out.append(
            Detection(
                cls_idx=best.cls_idx,
                conf=best.conf,
                box=(
                    min(d.box[0] for d in g),
                    min(d.box[1] for d in g),
                    max(d.box[2] for d in g),
                    max(d.box[3] for d in g),
                ),
            )
        )
    return out


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
    quality_description: str = ""
    tdi100: float | None = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["tdi"] = None if self.tdi is None else round(self.tdi, 3)
        d["tdi100"] = None if self.tdi100 is None else round(self.tdi100, 1)
        d["value_mass"] = round(self.value_mass, 4)
        d["mean_confidence"] = round(self.mean_confidence, 4)
        d["planktonic_fraction"] = round(self.planktonic_fraction, 4)
        d["counts"] = dict(sorted(self.counts.items(), key=lambda kv: -kv[1]))
        return d


def band_for(tdi100: float) -> tuple[str, str]:
    """Kelly & Whitton quality class from the 0..100 index."""
    for hi, name, es in TDI_BANDS:
        if tdi100 < hi:
            return name, es
    return TDI_BANDS[-1][1], TDI_BANDS[-1][2]


# Planktonic genera are excluded from the benthic index. The dataset spells one
# genus as "Pantocksekiella"; both spellings are accepted on purpose.
PLANKTONIC_GENUS = {
    "aulacoseira",
    "conticribra",
    "cyclostephanos",
    "cyclotella",
    "discostella",
    "lindavia",
    "melosira",
    "pantocsekiella",
    "pantocksekiella",
    "skeletonema",
    "stephanocyclus",
    "stephanodiscus",
    "thalassiosira",
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
    used_cells = 0

    num = 0.0
    den = 0.0
    for name, n in counts.items():
        genus = name.split(" ")[0].lower()
        if genus in planktonic:
            plank_cells += n
            continue
        tv = table.values[table.classes.index(name)]
        if tv.value is None or tv.indicator is None or n == 0:
            continue
        num += tv.value * tv.indicator * n
        den += tv.indicator * n
        used_cells += n

    res.valued_cells = used_cells
    res.planktonic_fraction = plank_cells / len(detections)
    res.value_mass = used_cells / len(detections)
    if den > 0:
        res.tdi = num / den
        res.tdi100 = 25.0 * res.tdi - 25.0
        res.quality, res.quality_description = band_for(res.tdi100)
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
    ap.add_argument("--max-windows", type=int, default=1200)
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
    print(f"device       {clf.device}   model {Path(CKPT).name} ep{clf.epoch} (val {clf.best_acc:.2f}%)")
    print(f"seconds      {dt:.2f}")
    w, h = Image.open(args.image).size
    xs, ys, step = sweep_plan(w, h, args.window, args.stride, args.max_windows)
    print(f"sweep        {len(xs) * len(ys)} windows of {args.window}px, step {step}px "
          f"({'full frame' if step <= int(args.window * args.stride) + 1 else 'sparse grid (budget)'})")
    print(f"cells        {res.total_cells} counted, mean confidence {res.mean_confidence:.3f}")
    print(f"planktonic   {res.planktonic_fraction * 100:.1f}% of counts (excluded from the index)")
    print(f"valued       {res.valued_cells}/{res.total_cells} cells ({res.value_mass * 100:.1f}% of mass)")
    print()
    if res.tdi is None:
        print("TDI           n/a - no counted benthic taxon carries a published TDI value")
    else:
        print(f"TDI           {res.tdi100:.1f} / 100  (raw {res.tdi:.3f})  ->  {res.quality} / {res.quality_description}")
    print()
    print("top taxa:")
    for name, n in list(res.as_dict()["counts"].items())[:10]:
        tv = table.values[table.classes.index(name)].value
        vs = f"TDI s={tv:.0f}" if tv is not None else "no value"
        print(f"  {n:>4}  {name:<48} {vs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
