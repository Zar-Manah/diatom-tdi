# Diatom-TDI

**Deep learning estimation of the Trophic Diatom Index from light microscopy images**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24443739.svg)](https://doi.org/10.5281/zenodo.24443739)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**Zar Manah** — Independent Researcher, Madrid, Spain
ORCID: [0009-0007-8443-8999](https://orcid.org/0009-0007-8443-8999)

---

## Overview

Benthic diatoms are among the standard biological quality elements used to assess
eutrophication in rivers under the European Water Framework Directive (2000/60/EC).
The Trophic Diatom Index (TDI; Kelly & Whitton, 1995) summarises an assemblage as a
sensitivity-weighted mean of its taxa, but producing it still requires a trained
taxonomist to identify and count 300–500 frustules per slide under light microscopy.

This repository contains an end-to-end, offline pipeline that estimates the TDI
directly from microscopy images:

1. a budget-aware sliding-window detector locates candidate frustules in an
   unsegmented micrograph;
2. a fine-tuned EfficientNet-B3 classifies each window over 101 taxa;
3. three-stage de-duplication converts overlapping windows into cell counts;
4. the published Kelly & Whitton index is computed from per-taxon sensitivity
   (`s`) and indicator weight (`v`), with planktonic taxa excluded from the
   benthic index and an explicit `value_mass` report.

The scientific contribution is not only the classifier: the repository quantifies
how single-cell classification error propagates into the assemblage-level index,
and it releases a slide-grouped split that avoids the leakage present in the
original random split.

## Results

### Single-cell classification

Test partition of the UDE *Diatoms in the Wild 2024* benchmark (101 taxa, 7,375 cutouts):

| Model | Top-1 | SHA-256 (prefix) |
| :--- | :---: | :--- |
| EfficientNet-B0, phase 1 | 76.01% | `53d2c1a7…` |
| EfficientNet-B3, baseline | 77.90% | `8d4fd7c6…` |
| EfficientNet-B3, edited head | **78.16%** | `0557c765…` |

The two B3 checkpoints come from the same training run (epoch 28). They differ only
in `classifier.weight` and `classifier.bias`; the closed-form edit flips 19 test
cells and requires no gradient descent.

The split is hash-locked but not leakage-free: 45.61% of test cutouts are crops of a
micrograph that also appears in training. Accuracy is 80.98% on those cutouts and
75.79% on the 4,011 cutouts from unseen micrographs. The headline figure is therefore
an upper bound. `state/split_grouped.json` provides a slide-disjoint alternative
(0 shared slides; 59,184 / 7,528 / 7,482 cutouts) for retraining.

### Index propagation

Grouping the 7,375 test cutouts by slide/treatment token yields 294 groups. Using
the published index formula:

| Metric | Baseline | Edited head |
| :--- | :---: | :---: |
| Whole-set TDI, true (0–100) | 54.4 | 54.4 |
| Whole-set TDI, predicted (0–100) | 53.8 | 53.8 |
| Mean absolute error (raw 1–5) | **0.1249** | 0.1255 |
| RMSE (raw 1–5) | 0.2286 | 0.2334 |
| Pearson *r* | 0.9681 | 0.9665 |
| Spearman *ρ* | 0.9692 | 0.9667 |
| Exact 5-tier band | 245/294 (83.33%) | 245/294 (83.33%) |
| Within ±1 band | 291/294 (98.98%) | 290/294 (98.64%) |

`value_mass` over the test set is 78.3%; the planktonic share is 11.0%. Both are
reported by the command-line tool, because an index computed over 78% of the
counted biomass is not the same claim as an index over all of it.

### Latency

On Apple M4 with PyTorch MPS: a 902×948 micrograph is processed in 2.5 s
(121 dense windows), and a 5000×3000 field frame in 15.5 s over 1,118 windows
at a 114 px step. When the dense grid exceeds the 1,200-window budget, the stride
is widened to cover the frame edge to edge instead of truncating the sweep.

## The index

```
TDI_raw = Σ(aᵢ · sᵢ · vᵢ) / Σ(aᵢ · vᵢ)        # 1..5
TDI_100 = 25 · TDI_raw − 25                    # 0..100
```

* `aᵢ` is the counted abundance, `sᵢ ∈ [1,5]` the sensitivity, `vᵢ ∈ [1,3]` the
  indicator weight.
* Quality bands on the 0–100 scale: `<35` Excellent, `35–50` Good, `50–65` Fair,
  `65–85` Poor, `≥85` Bad.
* Planktonic genera (`Aulacoseira`, `Conticribra`, `Cyclostephanos`, `Cyclotella`,
  `Discostella`, `Lindavia`, `Melosira`, `Pantocsekiella`, `Skeletonema`,
  `Stephanocyclus`, `Stephanodiscus`, `Thalassiosira`) are removed from both the
  numerator and the denominator.
* `v` is the `tdi_v` column of the CRAN package `diathor` (Kelly & Whitton, 1995).
  Of the 101 classes, 79 carry a published `s` and 74 carry both `s` and `v`.
  Taxa without values carry no weight, and `value_mass` reports the represented
  fraction of the count.

## Data

The classifier is trained on the public **UDE Diatoms in the Wild 2024** dataset
(83,570 light-microscopy images, 611 taxa; CC0):
[doi:10.5281/zenodo.10410655](https://doi.org/10.5281/zenodo.10410655).

The 101 taxa with at least 100 examples are used. The raw dataset is not
redistributed here; the training scripts download it automatically from Zenodo.

Two deterministic splits are included:

| File | Cutouts (train/val/test) | Slides shared | SHA-256 (prefix) |
| :--- | :---: | :---: | :--- |
| `state/split_f1.json` | 59,444 / 7,375 / 7,375 | 294/294 | `b6a0cb94…` |
| `state/split_grouped.json` | 59,184 / 7,528 / 7,482 | 0 | `31d7ad22…` |

`split_f1.json` is the split used for the released checkpoints. It was built by
shuffling cutout filenames and is retained for reproducibility, not as a clean
generalisation benchmark. `split_grouped.json` groups by slide and should be used
for any new training run.

## Repository layout

```
.
├── code/
│   ├── tdi.py                  # detection, classification, counting, TDI
│   ├── train_f1.py             # EfficientNet-B0, phase 1
│   ├── train_f2.py             # EfficientNet-B3, phase 2
│   ├── validate_tdi.py         # index-level validation on the test split
│   ├── audit_leakage.py        # micrograph/slide leakage audit
│   ├── audit_labels.py         # label-noise audit
│   ├── build_split.py          # random split (released checkpoints)
│   ├── build_split_grouped.py  # slide-grouped split
│   ├── build_tdi_table.py      # per-taxon s/v table
│   ├── test_tdi.py             # index and counting checks
│   └── weight_surgery/         # closed-form classifier-head editing
├── data/                       # per-taxon values and source tables
├── state/                      # splits and label audit
├── reports/                    # validation and leakage reports
├── paper/                      # manuscript (LaTeX source and PDF)
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required. PyTorch runs on Apple Silicon MPS, NVIDIA CUDA or CPU.

## Model weights

The weights are archived on Zenodo together with the source release:
[doi:10.5281/zenodo.24443739](https://doi.org/10.5281/zenodo.24443739).

Download `diatom-tdi-checkpoints.zip`, verify it against `SHA256SUMS.txt`, and
extract it at the repository root. The archive restores:

```
checkpoints/f1/best.pt
checkpoints/f2/best.pt
checkpoints/f2_edited/best_edited.pt
checkpoints/f2_edited/detritus_vector.pt
```

| File | SHA-256 |
| :--- | :--- |
| `checkpoints/f1/best.pt` | `53d2c1a708ebb122c0bb99f9e0d148de0c5ee7ceab17defce5dd57e3d52ef540` |
| `checkpoints/f2/best.pt` | `8d4fd7c61b6a1e11f329847db902830bee8623862f53d7a521ebe6943f7676c1` |
| `checkpoints/f2_edited/best_edited.pt` | `0557c7655159532f597ceb14c530f8c767a8b717bd80026a8f042b7d87eedd94` |
| `checkpoints/f2_edited/detritus_vector.pt` | `67f7a62dbc1483ae02a1c9bcce61e122a8279a20b0747d42f7ab309953b03a8b` |

## Usage

```bash
# human-readable report
python3 code/tdi.py --image path/to/micrograph.png

# machine-readable output
python3 code/tdi.py --image photo.jpg --json

# adjust the window to the optical scale (150 px is calibrated to the UDE scans)
python3 code/tdi.py --image photo.jpg --window 120 --conf 0.48
```

The command prints per-taxon counts, the raw and 0–100 index, the quality band,
`value_mass`, the planktonic fraction and the sweep coverage.

## Reproducibility

The evaluation reports in `reports/` are produced by:

```bash
python3 code/test_tdi.py                     # index, tables, counting and sweep checks
python3 code/validate_tdi.py                 # -> reports/validacion_tdi.json
python3 code/validate_tdi.py --ckpt checkpoints/f2/best.pt
python3 code/audit_leakage.py \
    --ckpt checkpoints/f2_edited/best_edited.pt \
    --ckpt checkpoints/f2/best.pt            # -> reports/auditoria_fuga.json
python3 code/build_split_grouped.py          # -> state/split_grouped.json
```

`validate_tdi.py` and `audit_leakage.py` require the UDE images. The training
scripts download and extract the dataset automatically on first run. Predictions
are cached per checkpoint SHA-256, and validation aborts if the checkpoint's
`split_hash` or class order does not match the split.

## Limitations

* **Leaky benchmark.** The released checkpoints were trained on a split that
  shares micrographs between train and test. The index metrics are an upper
  bound; retraining on `state/split_grouped.json` is required for a clean number.
* **Counting.** The sliding-window detector is the weakest link. Cells larger
  than the window can be split into fragments that the merge step does not
  rejoin, and low-contrast phone photographs can fall below the confidence
  threshold. Instance segmentation is the natural replacement.
* **Optical scale.** The 150 px window is calibrated to the UDE scans. Adjust
  `--window` for other objectives or camera setups.
* **Taxonomic coverage.** The model covers 101 Central European taxa. Tropical,
  Arctic or hypersaline assemblages require regional fine-tuning.
* **Single field.** The pipeline processes one micrograph at a time and reports
  relative abundances; it does not stitch a full cover slip or estimate absolute
  cell density.

## Citation

```bibtex
@software{manah2026diatomtdi,
  title     = {Diatom-TDI: deep learning estimation of the Trophic Diatom Index
               from light microscopy images},
  author    = {Manah, Zar},
  year      = {2026},
  version   = {1.0.1},
  doi       = {10.5281/zenodo.24443739},
  url       = {https://github.com/Zar-Manah/diatom-tdi},
  license   = {MIT}
}
```

The manuscript is available in [`paper/manuscript.pdf`](paper/manuscript.pdf) and as
LaTeX source in [`paper/manuscript.tex`](paper/manuscript.tex).

## License

Code and documentation are released under the [MIT License](LICENSE).

## Acknowledgements

The benchmark used here was created by the authors of *UDE Diatoms in the Wild
2024*; the per-taxon index values follow Kelly & Whitton (1995) and the `diathor`
R package. Their public availability made this study possible.
