# zar manah diatom tdi 🧬

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23108803.svg)](https://doi.org/10.5281/zenodo.23108803)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![WFD Compliant](https://img.shields.io/badge/WFD%20Directive-2000%2F60%2FEC-06B6D4.svg)](https://environment.ec.europa.eu/topics/water/water-framework-directive_en)
[![Offline Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline%20%2F%20Zero%20Cloud-success.svg)](#privacy--data-sovereignty)
[![Crafted in Spain](https://img.shields.io/badge/Crafted%20in-Spain%20%F0%9F%87%AA%F0%9F%87%B8-yellow.svg)](#)

> **AI for Water Purity Assessment:** Autonomous Computational Limnology Engine for deep learning-driven Trophic Diatom Index (TDI) estimation from microscopy images. An end-to-end, edge-deployable deep learning pipeline that directly estimates the Kelly & Whitton (1995) **Trophic Diatom Index (TDI)** and Water Framework Directive (WFD) water quality classes from raw light microscopy images and smartphone eyepiece photographs in under 11 seconds. Crafted in Spain 🇪🇸.

---

## 🔬 Conceptual Overview

### Why Benthic Diatoms?
Benthic diatoms (unicellular siliceous algae of the class *Bacillariophyceae*) are premier bioindicators mandated by international regulatory frameworks, most notably the **European Water Framework Directive (2000/60/EC)**. Due to narrow, species-specific tolerances to nutrient concentrations (phosphorus, nitrogen, organic pollution, and pH), their community composition reflects integrated river water quality over weeks and months.

### The Operational Bottleneck
Traditional diatom assessment is severely bottlenecked:
1. **Labor-Intensive Counting:** Requires chemical digestion (hot acid/peroxide), mounting in high-refractive-index media (Naphrax), and manually enumerating 300–500 frustules per slide under 1000× oil immersion microscopy (2–4 hours per sample).
2. **Contracting Expert Pool:** Global defunding and retirement of professional classical alpha-taxonomists.
3. **The Software Gap:** Commercial and academic tools (such as OMNIDIA, DARLEQ3, and DiaThor) are downstream calculators that require humans to manually count and enter species vectors. Until now, no pipeline closed the loop from raw field microscopy to ecological index verification.

### What This Work Achieves
* **End-to-End Autonomous Pipeline:** Takes raw unsegmented micrographs or 12 MP smartphone photos taken through microscope eyepieces, locates frustules via budget-aware sliding windows with dual-stage NMS, classifies 101 taxa via fine-tuned **EfficientNet-B3**, and computes abundance-weighted TDI scores.
* **Empirical Proof of Statistical Error Damping:** Evaluated across **294 natural river sampling sites** (7,375 field specimens) from the *UDE Diatoms in the Wild 2024* sealed benchmark. We demonstrate that single-cell classification error ($\sim 22\%$) cancels out during ecological aggregation:
  * Whole-dataset TDI error is **0.002** (3.435 ground truth vs. 3.433 predicted).
  * Pearson correlation $r = \mathbf{0.9703}$, Spearman $\rho = \mathbf{0.9658}$, $R^2 = \mathbf{0.9410}$.
  * **90.14% exact 5-tier WFD ecological quality band preservation** (265/294 sites) and **99.32% within $\le 1$ adjacent band**.
* **Correction of Critical Ecological Trap:** Exposes and resolves the widely confused `tdi_s` vs. `tdi_v` column ambiguity in `diathor` and enforces mathematical isolation of planktonic taxa to prevent benthic index corruption.

---

## ⚡ Quickstart: Run in 1 Click

### Option A: The Universal Single-File Engine (`zar_manah_diatom_tdi.pyz`)
A single, self-contained executable file containing the entire model (125 MB), local server, web interface, and TDI tables:
```bash
# macOS / Linux
python3 zar_manah_diatom_tdi.pyz

# Windows
python zar_manah_diatom_tdi.pyz

# Android (Termux)
python zar_manah_diatom_tdi.pyz
```
* On macOS, you can also double-click `zar_manah_diatom_tdi.command`.
* Automatically launches the local web browser at `http://127.0.0.1:8765`.
* Automatically installs any missing dependencies silently on first run.
* Auto-terminates the local process when the browser window is closed.

### Option B: Native macOS App (`Diatomeas.app`)
Double-click `Diatomeas.app` on macOS. Runs in an isolated, dark-themed native window with hardware acceleration.

### Option C: Python CLI
```bash
python3 code/tdi.py --image path/to/sample.png --window 150 --conf 0.48
```

---

## 📊 Benchmark & Validation Results

### 1. Classification Accuracy (UDE Diatoms in the Wild 2024 Benchmark)
| Model / Framework | Classes | Top-1 Accuracy | Gain over Baseline |
| :--- | :--- | :--- | :--- |
| UDE Baseline (Venkataramanan et al. 2024) | 101 | 60.41% | Baseline |
| Published Literature SOTA | 101 | 69.00% | +8.59 pp |
| **EfficientNet-B3 (This Work)** | **101** | **77.90%** | **+17.49 pp** |

### 2. Ecological Site-Level Validation (294 Natural River Assemblages)
| Validation Metric | Value | Limnological Meaning |
| :--- | :--- | :--- |
| **Sites Evaluated** | 294 river reaches | Reconstructed natural catchments across Germany |
| **Field Cutouts** | 7,375 | Fully sealed test split (`b6a0cb94...`) |
| **Dataset True TDI** | **3.435** | Ground-truth human taxonomist index |
| **Dataset Predicted TDI** | **3.433** | Autonomous model prediction |
| **Net Aggregate Error** | $\mathbf{-0.002}$ | **Near-zero systemic bias** |
| **Pearson Correlation ($r$)** | $\mathbf{0.9703}$ | Near-perfect linear tracking ($p < 10^{-15}$) |
| **Spearman Rank ($\rho$)** | $\mathbf{0.9658}$ | Monotonic ordering preserved ($p < 10^{-15}$) |
| **Mean Absolute Error (MAE)**| $\mathbf{0.1037}$ | $5\times$ smaller than quality class boundaries ($0.50$) |
| **Exact 5-Tier WFD Agreement**| $\mathbf{90.14\%}$ (265/294) | 9 out of 10 sites receive identical legal status |
| **Agreement within $\le 1$ Class**| $\mathbf{99.32\%}$ (292/294) | Only 2 sites deviate by $> 1$ adjacent class |

### 3. Inference Latency (Commodity Edge Hardware)
| Input Modality | Image Size | Windows Scanned | Wall-Clock Latency |
| :--- | :--- | :--- | :--- |
| Single Pre-Cropped Cell | $300 \times 300\text{ px}$ | 1 | **15 ms** |
| Standard Micrograph | $1600 \times 1200\text{ px}$ | 280 | **7.1 s** |
| Smartphone Eyepiece Photo | $4000 \times 3000\text{ px}$ (12 MP) | 800 (budget capped) | **11.2 s** |

---

## 📐 Mathematical Formulation of Error Damping

The Kelly & Whitton (1995) Trophic Diatom Index is:

$$\text{TDI} = \frac{\sum_{i=1}^{K} s_i \cdot N_i}{\sum_{i=1}^{K} N_i}$$

where $s_i \in [1, 5]$ is taxon ecological sensitivity and $N_i$ is counted abundance.

Under an imperfect classifier with confusion matrix $\mathbf{C}$, the expected net ecological bias $\Delta \text{TDI}$ is:

$$\mathbb{E}[\Delta \text{TDI}] = \sum_{j=1}^{K} p_j^* \left[ \sum_{i \neq j} C_{ij} (s_i - s_j) \right]$$

1. **Intra-Guild Invariance:** Closely related sister species (e.g. within *Navicula* or *Nitzschia*) that confuse neural networks inhabit identical ecological niches ($s_i = s_j$). Consequently, $(s_i - s_j) = 0$, contributing **zero net error** to the ecological verdict.
2. **Symmetric Error Cancellation:** Misclassifications across differing trophic guilds are symmetrically distributed ($90$ sites positive bias vs. $103$ sites negative bias), balancing each other out during abundance-weighted summation.

---

## 🔒 Privacy & Data Sovereignty
* **100% Offline Execution:** Runs locally on your device (GPU/MPS or CPU).
* **Zero Cloud Telemetry:** No images, coordinates, or data ever leave your machine.
* **Open Source & Auditable:** All code, weights, and validation scripts are fully open.

---

## 👤 Author & Affiliation
* **Author:** Zar Manah
* **Affiliation:** Independent Researcher, Madrid, Spain
* **ORCID:** [0009-0007-8443-8999](https://orcid.org/0009-0007-8443-8999)
* **Correspondence:** 

---

## 📖 Citation

If you use this software or empirical error damping findings in your research, please cite:

```bibtex
@article{manah2026bridging,
  title={Bridging Computer Vision and Freshwater Biomonitoring: Deep Learning-Driven Trophic Diatom Index Estimation and Empirical Error Damping Across 294 Natural River Assemblages},
  author={Manah, Zar},
  affiliation={Independent Researcher, Madrid, Spain},
  journal={bioRxiv / Zenodo Preprints},
  year={2026},
  doi={10.5281/zenodo.23108803}
}
```

---

## ⚖️ License
Released under the [MIT License](LICENSE).
