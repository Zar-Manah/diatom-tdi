<p align="center">
  <img src="diatomeas-icon.png" width="128" alt="Diatomeas icon" />
</p>

# Diatom-TDI 🧬 — AI Water Purity Assessment from Microscopy Images

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![WFD Compliant](https://img.shields.io/badge/WFD%20Directive-2000%2F60%2FEC-06B6D4.svg)](https://environment.ec.europa.eu/topics/water/water-framework-directive_en)
[![Offline Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline%20%2F%20Zero%20Cloud-success.svg)](#privacy--data-sovereignty)
[![Crafted in Spain](https://img.shields.io/badge/Crafted%20in-Spain%20%F0%9F%87%AA%F0%9F%87%B8-yellow.svg)](#)

> **AI for Water Purity Assessment:** a computational limnology engine that classifies benthic diatom frustules from microscopy images (101 taxa, EfficientNet-B3) and aggregates them into the Kelly & Whitton (1995) **Trophic Diatom Index (TDI)** with Water Framework Directive quality classes. Runs fully offline on a laptop or a phone-sized image, no cloud. Crafted in Spain 🇪🇸.

---

## ⚠️ Status: audited, corrected, and honest about what is not yet proven

This repository was audited end-to-end on 2026-11-14. The classifier and the per-taxon
table are real and reproduce; the *reported ecological metrics were not*. Everything below
is the measured truth after the corrections, with the raw artifacts linked.

| Claim in the previous README | Reality (measured) |
| :--- | :--- |
| Surgical model 78.29%, +29 cells, MAE 0.1020, r 0.9705 | The shipped `best_edited.pt` scores **78.16%**; the report was silently the **baseline** cache (`reports/validacion_tdi.json` reproduced base-without-prior exactly) |
| 90.14% exact band agreement, 99.32% within 1 | With the correct index: **83.33%** exact (245/294), **98.64%** within 1 (290/294) |
| TDI error 0.004 over 3.435 | The index formula itself was wrong (no `v` weight, ×10 scale, non-standard bands). Corrected below |
| "294 natural river sampling sites" | 294 **slides/treatments** (`NH4Cl`, `DW`, `Naphrax` preparations) of 7,375 cutouts |
| 97.33% ensemble / 93.80% TTA | **No checkpoint, no code, no artifact** in this repository — removed |
| 60.41% "UDE baseline" | That number is the **OOD-detection accuracy** of DS50 in Table 3 of the UDE paper, not a 101-class classifier |
| 11.2 s for 12 MP, 800 windows | 800 windows covered only the **top-left 42.5%** of a 12 MP frame. Now the grid spans edge to edge: 1,118 windows, **15.5 s** |

The audit trail is reproducible: `code/test_tdi.py` (70 checks), `code/validate_tdi.py`,
`code/audit_leakage.py`, `reports/validacion_tdi.json`, `reports/auditoria_fuga.json`.

---

## 🔬 Conceptual Overview

### Why Benthic Diatoms?
Benthic diatoms (unicellular siliceous algae of the class *Bacillariophyceae*) are premier
bioindicators mandated by the **European Water Framework Directive (2000/60/EC)**. Their
species-specific tolerances to nutrients, organic pollution and pH make community
composition an integrated measure of river water quality over weeks to months.

### The Operational Bottleneck
1. **Labour:** acid/peroxide digestion, mounting in Naphrax, and manually counting 300–500
   frustules per slide at 1000× oil immersion (2–4 h per sample).
2. **Taxonomist scarcity:** the expert pool is retiring faster than it is replaced.
3. **The software gap:** OMNIDIA, DARLEQ3 and DiaThor are downstream calculators — a human
   still has to count and type the species vector.

### What This Work Does
* **Raw image → index:** budget-aware sliding windows + three-stage de-duplication (class-wise
  NMS → fragment merge → cross-class NMS) locate frustules in an unsegmented micrograph,
  EfficientNet-B3 classifies each one, and the abundance-weighted index is computed from
  published per-taxon values.
* **The index is the published one.** `TDI = 25·Σ(aᵢ·sᵢ·vᵢ)/Σ(aᵢ·vᵢ) − 25`, exactly as in
  `diathor::tdi` (Kelly & Whitton 1995), with bands `<35 / 35–50 / 50–65 / 65–85 / ≥85`.
* **Planktonic taxa are excluded** from the index and reported separately
  (`planktonic_fraction`): the TDI is defined on the benthic fraction.
* **Nothing is hidden:** taxa without a published `s` or `v` carry no weight, and the app
  reports `value_mass` — how much of the counted biomass the index actually represents
  (78.3% on the UDE test set).

---

## ⚡ Quickstart: Run in 1 Click

### Option A: The universal single-file engine (`zar_manah_diatom_tdi.pyz`)
One self-contained file with the checkpoint, the table, the server and the UI:
```bash
python3 zar_manah_diatom_tdi.pyz      # macOS / Linux / Android (Termux)
python  zar_manah_diatom_tdi.pyz      # Windows
```
* Opens `http://127.0.0.1:8765`; installs missing dependencies on first run.
* The bundled `tdi.py` is byte-identical to `code/tdi.py` — the app cannot compute a
  different index than the tests. `BUILD.json` inside the zip lists the SHA-256 of every
  asset, so you can verify which checkpoint you are running.

### Option B: Native macOS app (`Diatomeas.app`)
Double-click. Isolated dark-themed window, hardware accelerated.

### Option C: Python CLI
```bash
python3 code/tdi.py --image path/to/sample.png --window 150 --conf 0.48
python3 code/tdi.py --image photo.jpg --json          # machine-readable
```

---

## 📊 Measured results

### 1. Single-cell classification (UDE *Diatoms in the Wild 2024*, 101 classes, 7,375 test cutouts)

| Model | Top-1 | Note |
| :--- | :---: | :--- |
| EfficientNet-B0 (phase 1) | 76.01% | reported by the training logs, not re-run here |
| EfficientNet-B3 `best.pt` | **77.90%** | re-run from scratch, `sha 8d4fd7c6…` |
| EfficientNet-B3 `best_edited.pt` (shipped) | **78.16%** | re-run from scratch, `sha 0557c765…` |

The two checkpoints are the same training run (`epoch=28`, `best_acc=78.0067`,
`split_hash=b6a0cb94…`); they differ only in `classifier.weight/bias` (`max|Δ| ≈ 0.147`),
which is why the "surgical" gain is +0.26 pp and not +0.39 pp.

**This number is optimistic.** `state/split_f1.json` was built by shuffling *cutout*
filenames, so a photomicrograph can appear in train and test:

* 3,364 of 7,375 test cutouts (**45.61%**) come from a micrograph that is also in train;
* 294 of 294 test slides appear in train;
* accuracy on those cells is **80.98%** vs **75.79%** on genuinely unseen micrographs —
  a **+5.18 pp** inflation (`reports/auditoria_fuga.json`).

`state/split_grouped.json` (`hash 31d7ad2280c818b4`) fixes this: slides are grouped, so
**0 slides are shared** between train and test (train 59,184 / val 7,528 / test 7,482;
5 classes lose test coverage, 15 have <5 test cells). **The shipped checkpoints were trained
on the leaky split and cannot be honestly evaluated on the grouped one — retraining is
required before any generalisation claim.** The training entry point accepts that split
unchanged; the honest number will be published with the retrained weights.

### 2. Index accuracy at slide level (corrected Kelly & Whitton formula, 294 slides)

| Metric | `best.pt` | `best_edited.pt` (shipped) |
| :--- | :---: | :---: |
| Test cutouts | 7,375 | 7,375 |
| Whole-set TDI, true (0–100) | 54.4 | 54.4 |
| Whole-set TDI, predicted (0–100) | 53.8 | 53.8 |
| Whole-set raw TDI (1–5), true | 3.1746 | 3.1746 |
| Whole-set raw TDI (1–5), predicted | 3.1525 | 3.1533 |
| Mean absolute error (raw) | 0.1249 | **0.1255** |
| RMSE (raw) | 0.2286 | 0.2334 |
| Pearson `r` | **0.9681** | 0.9665 |
| Spearman `ρ` | 0.9692 | 0.9667 |
| Exact 5-tier band | 245/294 (**83.33%**) | 245/294 (**83.33%**) |
| Within ±1 band | 291/294 (**98.98%**) | 290/294 (**98.64%**) |
| Slides biased up / down | 93 / 107 | 99 / 107 |

`value_mass` on the whole test set is **78.3%** and the planktonic share **11.0%** — both are
reported by the app, because an index computed over 78% of the counted biomass is not the
same claim as an index over all of it.

Note the reachable range: no benthic taxon in this 101-class set has `s = 1` (only the
planktonic *Thalassiosira* does), so the lowest reachable value is `s = 2` → index 25, not 0.

### 3. Latency (Apple M4, PyTorch MPS)

| Input | Windows | Stride | Wall clock |
| :--- | :---: | :---: | :---: |
| `app/sample.png` (902×948) | 121 | 75 px (dense) | **2.5 s** |
| 5000×3000 field frame | 1,118 | 114 px (full frame) | **15.5 s** |

Measured with `python3 code/tdi.py --image …` on Apple M4 / MPS, default budget of 1,200
windows. The 12 MP figure is the sweep itself (synthetic frame, 0 detections): it is dominated
by window classification, not I/O. The window budget widens the stride instead of truncating
the frame, and both CLI and app report `sweep.covers_full_frame` / `sweep.dense_grid`, so a
sparse grid is never passed off as a full sweep.

---

## 📐 The index, exactly as implemented

```
TDI_raw = Σ(aᵢ · sᵢ · vᵢ) / Σ(aᵢ · vᵢ)        # 1..5
TDI_100 = 25 · TDI_raw − 25                    # 0..100
```

* `aᵢ` counted abundance, `sᵢ` sensitivity 1–5, `vᵢ` indicator weight 1–3.
* Bands on the 0–100 form: `<35` Excellent, `35–50` Good, `50–65` Fair, `65–85` Poor, `≥85` Bad.
* Planktonic genera (`Aulacoseira`, `Conticribra`, `Cyclostephanos`, `Cyclotella`,
  `Discostella`, `Lindavia`, `Melosira`, `Pantocsekiella`, `Skeletonema`, `Stephanocyclus`,
  `Stephanodiscus`, `Thalassiosira`) are excluded from numerator **and** denominator.
* `v` comes from the `tdi_v` column of the CRAN package `diathor` (Kelly & Whitton 1995);
  74 of 101 taxa carry both `s` and `v`, and `s` matches `diathor$tdi_s` with **0 mismatches**.

Why the previous version was wrong: it used `Σ(a·s)/Σa` (no `v`), multiplied by 10, cut the
bands at 2/3/4/4.5 on the raw scale, and folded planktonic cells into the benthic average.

---

## 🔁 Reproducing everything

```bash
make check          # 70 unit checks + 50 artifact-consistency checks (~10 s)
make all            # regenerate every number, rebuild the engine, verify it

# or step by step
python3 code/test_tdi.py                                  # 70 checks, no checkpoint needed
python3 code/check_artifacts.py                           # pyz/.app/reports vs sources
python3 code/validate_tdi.py                              # shipped ckpt -> reports/validacion_tdi.json
python3 code/validate_tdi.py --ckpt checkpoints/f2/best.pt
python3 code/audit_leakage.py --ckpt checkpoints/f2_edited/best_edited.pt \
                             --ckpt checkpoints/f2/best.pt   # -> reports/auditoria_fuga.json
python3 code/build_split_grouped.py                       # -> state/split_grouped.json
python3 code/build_pyz.py                                 # -> zar_manah_diatom_tdi.pyz
```

`check_artifacts.py` exists because this repository shipped three drift bugs at once: the `.pyz` and the
`.app` bundle embedded older copies of the engine, and the validation report belonged to a different
checkpoint. The gate fails if any of that happens again — it compares the zip members byte for byte with
`code/tdi.py`, checks the app bundle, and verifies that every report carries the SHA-256 of the shipped
checkpoint and the hash of the split it was measured on.

`paper/MANUSCRIPT.md` and `paper/manuscript.tex` carry the corrections and the current numbers.
The old PDF was removed because it predated the audit; regenerate it from the `.tex` with a LaTeX
toolchain if a PDF is needed.

Predictions are cached per checkpoint **SHA-256**, and validation aborts if the checkpoint's
`split_hash` or class order does not match the split. That guard exists because a stale cache
from another checkpoint is precisely how the original report ended up labelled "surgical".

---

## 📦 Checkpoints & artifacts

| Artifact | SHA-256 |
| :--- | :--- |
| `checkpoints/f2_edited/best_edited.pt` (shipped) | `0557c7655159532f597ceb14c530f8c767a8b717bd80026a8f042b7d87eedd94` |
| `checkpoints/f2/best.pt` (baseline) | `8d4fd7c61b6a1e11f329847db902830bee8623862f53d7a521ebe6943f7676c1` |
| `checkpoints/f1/best.pt` | `53d2c1a7…d52ef540` |
| `data/tdi_values_101.csv` | see `BUILD.json` inside the `.pyz` |
| `state/split_f1.json` (random, used for training) | `b6a0cb94…` |
| `state/split_grouped.json` (slide-grouped, for retraining) | `31d7ad2280c818b4` |

No public DOI is assigned to this version yet. The previous Zenodo record was withdrawn and
must not be cited; the numbers in it are the superseded ones. A new record with the corrected
numbers, the audit report and `split_grouped.json` will replace it.

---

## 🔒 Privacy & Data Sovereignty
* **100% offline:** local MPS/CUDA/CPU inference, no telemetry, no uploads.
* **Auditable:** every number in this README is produced by a script in `code/`.

---

## 👤 Author
* **Author:** Zar Manah
* **Affiliation:** Independent Researcher, Madrid, Spain
* **ORCID:** [0009-0007-8443-8999](https://orcid.org/0009-0007-8443-8999)

---

## 📖 Citation

```bibtex
@software{manah2026diatomtdi,
  title={Diatom-TDI: benthic diatom classification and Trophic Diatom Index estimation from microscopy images},
  author={Manah, Zar},
  year={2026},
  note={Version with corrected Kelly \& Whitton index; see reports/validacion_tdi.json and reports/auditoria_fuga.json}
}
```

---

## ⚖️ License
Released under the [MIT License](LICENSE).
