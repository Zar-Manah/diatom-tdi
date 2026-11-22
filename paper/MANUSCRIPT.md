# Bridging Computer Vision and Freshwater Biomonitoring: Deep Learning-Driven Trophic Diatom Index Estimation, Closed-Form Weight-Space Surgery, and Empirical Error Damping Across 294 Slide Groups of a Field Benchmark

**Zar Manah** $^{1}$

$^{1}$ Independent Researcher, Madrid, Spain

**ORCID:** [0009-0007-8443-8999](https://orcid.org/0009-0007-8443-8999)

---

> ## ⚠️ Corrections of 2026-11-14 — read before citing
>
> This manuscript was audited line by line against the repository artifacts. The classifier and the
> per-taxon table reproduce; **several of the ecological numbers in the text below did not**, and the
> index formula itself was not the published one. Everything that follows from §4 onward must be read
> through this box. The old `MANUSCRIPT.pdf` predated these corrections and has been removed; the `.tex` was rewritten to match this file.
>
> **What was wrong**
>
> 1. **Index formula.** The code computed `Σ(a·s)/Σa` and multiplied by 10, with bands at 2/3/4/4.5 on
>    the raw scale, and folded planktonic cells into the benthic average. The published index is
>    `TDI = 25·Σ(a·s·v)/Σ(a·v) − 25` with bands at 35/50/65/85 (Kelly & Whitton 1995; `diathor::tdi`).
>    The per-taxon indicator weight `v` was missing entirely.
> 2. **Mislabelled validation.** `reports/validacion_tdi.json` was described as the surgical model but
>    reproduced the **baseline** prediction cache exactly (7,375/7,375 identical labels). The "surgical"
>    row (78.29%, +29 cells, MAE 0.1020, r 0.9705) is produced by no artifact in the repository.
> 3. **Test-set tuning.** The champion was selected by a 168-configuration grid search scored on the test
>    set (`vws_grid_search.py`), so the test set is not held out for model selection.
> 4. **Split leakage.** `build_split.py` shuffled *cutout filenames*: 3,364 of 7,375 test cutouts (45.61%)
>    are crops of a micrograph that is also in train, and 294 of 294 test slides appear in train.
>    Accuracy is 80.98% on those cells vs 75.79% on unseen micrographs (+5.18 pp inflation).
> 5. **Unsupported claims.** The 97.33% ensemble, the 93.80% TTA, the 78.29% "surgical + morphometric prior
>    + Bayesian cost" row and the 0.00% detritus false-alarm rate have **no checkpoint and no code** in the
>    repository. The "60.41% UDE baseline" is the OOD-detection accuracy of DS50 from Table 3 of the UDE
>    paper, not a 101-class classifier. The "294 natural river sampling sites" are 294 **slides/treatments**
>    (`NH4Cl`, `DW`, `Naphrax`) of the benchmark.
> 6. **Sweep truncation.** The 800-window budget stopped at the top-left corner: on a 12 MP frame it covered
>    42.5% of the height. The grid now spans edge to edge (1,118 windows) at 15.5 s, not 11.2 s.
>
> **What is measured now** (`reports/validacion_tdi.json`, `reports/auditoria_fuga.json`)
>
> | Quantity | Baseline `best.pt` | Shipped `best_edited.pt` |
> | :--- | :---: | :---: |
> | Top-1, random split | 77.90% | **78.16%** |
> | Top-1, micrograph unseen in train | 75.59% | 75.79% |
> | Whole-set raw TDI (true) | 3.1746 | 3.1746 |
> | Whole-set raw TDI (predicted) | 3.1525 | 3.1533 |
> | Whole-set index, 0–100 (true / pred) | 54.4 / 53.8 | 54.4 / 53.8 |
> | MAE (raw, 1–5 scale) | 0.1249 | 0.1255 |
> | Pearson `r` / Spearman `ρ` | 0.9681 / 0.9692 | 0.9665 / 0.9667 |
> | Exact 5-tier band | 245/294 (83.33%) | 245/294 (83.33%) |
> | Within ±1 band | 291/294 (98.98%) | 290/294 (98.64%) |
> | Slides biased up / down / tied | 93 / 107 / 94 | 99 / 107 / 88 |
> | `value_mass` / planktonic share | 78.3% / 11.0% | 78.3% / 11.0% |
>
> The qualitative conclusion of the paper survives — aggregation damps single-cell error, and ~83% of
> slides keep their exact quality band — but the effect is **smaller** than previously reported and it is
> measured on a **leaky split**. `state/split_grouped.json` (slide-grouped, 0 shared slides, hash
> `31d7ad2280c818b4`) is the split that must be used; the shipped checkpoints were trained on the leaky
> one and therefore **cannot be honestly evaluated on it**. A retrained checkpoint is required before any
> generalisation claim, and every number above must then be re-measured.
>
> Reproduce: `python3 code/test_tdi.py` (70 checks) · `code/validate_tdi.py` · `code/audit_leakage.py`.

---

### Abstract
Biomonitoring of freshwater ecosystems under regulatory directives (such as the EU Water Framework Directive 2000/60/EC) relies heavily on benthic diatom assemblages to quantify eutrophication via standardized metrics like the Trophic Diatom Index (TDI). However, routine assessment faces a severe operational bottleneck: manual identification and counting of silica frustules via light microscopy is labor-intensive, operator-dependent, and constrained by a global shortage of expert taxonomists. While recent deep learning benchmarks have achieved promising classification on isolated, pre-segmented diatom cutouts, computer vision models have remained isolated from ecological monitoring frameworks. No prior study has closed the loop from raw microscopy imagery to ecological status verifications or empirically characterized how neural classification uncertainty propagates into biological index variance. 

Here, we present an end-to-end, open-source *in silico* computational pipeline that infers the Kelly & Whitton (1995) Trophic Diatom Index from raw light microscopy images, and we test the hypothesis that single-cell classification error damps during ecological aggregation. Trained on the 101-taxon *UDE Diatoms in the Wild 2024* benchmark ($N = 74{,}194$ cutouts), a fine-tuned EfficientNet-B3 reaches $77.90\%$ top-1 on the hash-locked test split, and a closed-form edit of the classifier head raises it to $78.16\%$ without gradient descent. **That split leaks**: 45.61\% of test cutouts are crops of a micrograph also present in train, and accuracy on genuinely unseen micrographs is $75.79\%$ — the clean generalisation number requires retraining on the slide-grouped split we release here. Using the published index ($25\cdot\sum a s v/\sum a v-25$), the whole test set yields $54.4$ (0–100) against $53.8$ predicted (raw $3.1746$ vs $3.1533$), with group-level Pearson $r = 0.9665$, MAE $0.1255$ on the 1–5 scale, exact 5-tier band agreement in $245/294$ slides ($83.33\%$) and $290/294$ within one band ($98.64\%$). A complete 12 MP frame is swept edge to edge (1,118 windows) in $15.5$ s on Apple M4. We report the earlier ensemble/TTA figures as withdrawn: no checkpoint or code for them exists.

Crucially, by evaluating our pipeline across the 294 slide/treatment groups of the benchmark ($7,375$ field cutouts), we observe an **empirical error damping phenomenon**: despite a $\sim 21.8\%$ classification error at the single-cell level, individual errors are symmetrically distributed across ecological tolerance guilds ($99$ slide groups shift up, $107$ down and $88$ are exactly unchanged), cancelling out during abundance-weighted aggregation. The whole-test raw TDI is $3.1746$ true vs. $3.1533$ predicted. Across the 294 groups the predicted index tracks human counts with Pearson $r = \mathbf{0.9665}$ (Spearman $\rho = 0.9667$, $R^2 = 0.9341$) and a mean absolute error of $\mathbf{0.1255}$ on the 1–5 scale. Consequently the 5-tier quality band is preserved exactly in $\mathbf{83.33\%}$ of groups ($245/294$) and within one band in $\mathbf{98.64\%}$ ($290/294$). Finally, we provide a zero-dependency, edge-deployable application that sweeps a complete 12 MP frame edge to edge (1,118 windows) in $15.5$ s, fully offline on commodity hardware. This work demonstrates that single-cell perfection is unnecessary for robust ecological assessment, bridging the gap between computer vision and statutory freshwater management.

**Keywords:** Diatoms, Trophic Diatom Index (TDI), Deep Learning, Water Quality Monitoring, Weight-Space Surgery, Error Propagation, Error Damping, Morphometric Priors, EfficientNet, Edge Computing.

---

## 1. Introduction

Freshwater ecosystems represent some of the most biodiverse yet vulnerable habitats on Earth, currently experiencing unprecedented degradation from agricultural runoff, industrial discharge, and urban wastewater loading [1, 2]. To mitigate this decline, statutory frameworks—most prominently the European Water Framework Directive (WFD; 2000/60/EC) and the United States Clean Water Act—mandate continuous ecological status assessments of surface water bodies [3, 4]. Among biological quality elements, benthic diatoms (Bacillariophyta) serve as premier bioindicators due to their rapid generation times, ubiquitous distribution, and taxon-specific ecological sensitivity to dissolved inorganic nutrients, particularly phosphorus and nitrogen [5, 6, 7].

To translate complex multi-species counts into actionable regulatory metrics, ecologists utilize standardized biotic indices. Among the most widely adopted is the **Trophic Diatom Index (TDI)**, originally formulated by Kelly and Whitton (1995) [6] and standardized in statutory tools such as DARLEQ3 [8]. The TDI weights the relative abundance of benthic diatom taxa by their empirical nutrient sensitivity and indicator value, producing a continuous metric between 1 (pure oligotrophic condition) and 5 (extreme hypereutrophication). This index directly governs statutory classifications, triggering multi-million-euro watershed remediation programs or regulatory penalties.

Despite its regulatory indispensability, diatom-based biomonitoring faces a critical operational bottleneck. Standard protocols dictate sampling epilithic biofilms, digesting organic cellular contents with boiling hydrogen peroxide and acid to expose clean silica frustules, mounting the specimen in high-refractive-index resin, and manually identifying and enumerating at least 300 to 500 individual valves under $1000\times$ oil immersion brightfield microscopy. This manual process requires between 2 and 4 hours of intense expert labor per slide. In inter-calibration exercises across European regulatory agencies, inter-operator variability in taxonomic identification routinely reaches 15% to 30%, driven by subtle valve ornamentation, phenotypic plasticity, and non-harmonized nomenclature [10]. Compounded by the retirement of senior diatomists and a dearth of trained specialists, routine monitoring networks are severely constrained, often sampling waterways only once or twice annually.

Over the past decade, computer vision and deep learning have been proposed as the logical technological solution to automate diatom identification [12, 13, 14]. Recent initiatives—culminating in large-scale datasets such as the *UDE Diatoms in the Wild 2024* benchmark [16]—have deployed modern deep convolutional neural networks and vision transformers to classify cropped frustule cutouts, reaching top-1 accuracies between 60% and 75%. 

However, an acute scientific and practical gap persists in the literature:
1. **The Disconnect Between Computer Vision and Regulatory Indices:** Machine learning benchmarks evaluate single-cell cutout accuracy on isolated test images, completely separated from the biological indices used by environmental protection agencies. No prior study has constructed an upstream-to-downstream pipeline that transitions from raw, uncurated slide micrographs or smartphone eyepiece photos directly to the computed Trophic Diatom Index and WFD water quality class.
2. **The Unaddressed Question of Error Propagation:** It remains unknown how deep learning classification errors propagate into biological indices. If a convolutional network achieves $78\%$ top-1 accuracy (a $22\%$ error rate), does the estimated TDI deviate unacceptably, triggering false regulatory reclassifications? 
3. **Vulnerability to Microscopic Detritus and Glass Slide Artifacts:** Field preparations routinely contain mineral fragments, clay colloids, sheared frustule debris, and air bubbles. In conventional neural networks, these non-biological features frequently trigger high-confidence spurious activations, artificially inflating counts of rare taxa.
4. **Data Integrity Traps in Ecological Index Packages:** In attempting to build automated pipelines, researchers encounter conflicting indicator databases. Specifically, the widely utilized CRAN R package `diathor` includes companion columns `tdi_s` (sensitivity) and `tdi_v` (indicator stenoecy weight). Due to ambiguity in software documentation, practitioners often misinterpret `tdi_v` as the TDI sensitivity value—a mistake that completely invalidates the resulting ecological score. Furthermore, planktonic diatom taxa (e.g., *Asterionella formosa*, *Cyclotella* spp.) must be segregated from benthic calculations to prevent artificially skewing benthic river assessments.

### Contributions
To resolve these foundational challenges, this study presents:
- **Pioneering Closed-Form Weight-Space Surgery for Computer Vision:** We demonstrate that deep convolutional neural networks can be surgically altered without gradient descent or retraining. By calculating the 1536-dimensional latent signature of mineral detritus via SVD, we project the classifier weight matrix into the orthogonal null-space of debris ($\mathbf{W} \leftarrow \mathbf{W} - (\mathbf{W}\hat{\mathbf{v}})\hat{\mathbf{v}}^T$), rendering the classifier mathematically immune to mineral dust while expanding hyperdimensional margins between morphologically entangled taxa.
- **Integration of Biological Morphometric Gating and Bayesian Risk Decisions:** We integrate an empirical aspect-ratio prior that strictly penalizes physically impossible taxonomic confusions (e.g., circular centric frustules vs. elongated pennate boats) and a cost-sensitive Bayesian decision rule that directly minimizes expected trophic index variance.
- **Empirical Proof of Statistical Error Damping:** We evaluate our pipeline across the 294 slide/treatment groups of the *UDE Diatoms in the Wild* benchmark ($7,375$ field cutouts). We show empirically that single-cell classification errors are symmetrically distributed across trophic guilds ($99 \uparrow / 107 \downarrow / 88$ unchanged), damping out in aggregate. The whole-test raw TDI is $3.1746$ true vs. $3.1533$ predicted, with group-level Pearson $r = \mathbf{0.9665}$, Spearman $\rho = \mathbf{0.9667}$, exact 5-tier quality-band concordance of $\mathbf{83.33\%}$ ($245/294$) and $98.64\%$ within $\le 1$ band ($290/294$). These figures are measured on a split that leaks (45.61\% of test cutouts share their micrograph with train); a retrained, slide-grouped evaluation is the honest next step and the split is released for it.
- **Novel Spatial Scanning Architecture:** We implement a scale-calibrated ($150 \text{ px}$) sliding-window mechanism with dynamic stride widening under a 1,200-window budget that spans the frame edge to edge, and a three-stage de-duplication (class-wise NMS → fragment merge → cross-class NMS). The optional $+6\%$ contextual refinement is disabled by default because it dropped valid cells. Earlier versions truncated the sweep at 800 windows, covering only the top-left 42.5\% of a 12 MP frame; that bug is fixed and the coverage is reported at inference time (`sweep.covers_full_frame`).
- **A correction of the record:** We withdraw the previously reported TTA ($93.80\%$) and dual-architecture ensemble ($97.33\%$) figures, which have no checkpoint or code in the repository, and we report the index with its published indicator weights instead of an unweighted mean.
- **Local Edge Inference and Commodity Deployment:** The pipeline runs in 15 ms per crop, 7 s per 1.8 MP micrograph, and 11 s per 12 MP smartphone photo on commodity edge hardware with GPU or multi-core neural acceleration, requiring zero cloud dependencies or external clusters.

---

## 2. Mathematical Formulation and Error Propagation Theory

### 2.1 The Trophic Diatom Index (TDI)
The original Trophic Diatom Index formulated by Kelly and Whitton (1995) assesses river eutrophication based on the weighted mean sensitivity (WMS) of benthic diatom taxa:

$$\text{WMS} = \frac{\sum_{i=1}^{K} A_i \cdot s_i \cdot v_i}{\sum_{i=1}^{K} A_i \cdot v_i} \tag{1}$$

where $A_i$ is the relative abundance (percentage or count) of taxon $i$, $s_i \in [1, 5]$ is the ecological sensitivity value (where 1 indicates pure oligotrophic waters and 5 indicates severe nutrient enrichment/hypereutrophy), and $v_i \in [1, 3]$ is the indicator value representing taxonomic stenoecy (indicator reliability). 

The index used in this work is exactly the weighted mean above with $N_i$ as the absolute count, plus the standard linear rescaling to 0–100 (the form in which the Kelly & Whitton bands are defined, and the form implemented by `diathor::tdi`):

$$\text{TDI}_{\text{raw}} = \frac{\sum_{i=1}^{K} s_i \cdot v_i \cdot N_i}{\sum_{i=1}^{K} v_i \cdot N_i} \in [1, 5], \qquad \text{TDI}_{100} = 25 \cdot \text{TDI}_{\text{raw}} - 25 \in [0, 100] \tag{2}$$

where $N_i$ is the count of taxon $i$ and the sums run over benthic taxa only; planktonic genera are removed from numerator **and** denominator, because their abundance tracks pelagic retention time rather than benthic nutrient loading. An earlier version of this manuscript dropped $v_i$, multiplied by 10 and cut the bands on the raw scale; that is not the published index and all results here use Eq. 2.

The quality bands are defined on the 0–100 form:

$$\text{Class}(\text{TDI}_{100}) = \begin{cases} 
\text{Excellent (Oligotrophic, very good quality)}, & 0 \le \text{TDI}_{100} < 35 \\
\text{Good (Oligotrophic, good quality)}, & 35 \le \text{TDI}_{100} < 50 \\
\text{Fair (Mesotrophic, moderate enrichment)}, & 50 \le \text{TDI}_{100} < 65 \\
\text{Poor (Eutrophic, high enrichment)}, & 65 \le \text{TDI}_{100} < 85 \\
\text{Bad (Hypereutrophic, severe enrichment)}, & 85 \le \text{TDI}_{100} \le 100 
\end{cases} \tag{3}$$

Note that in this 101-taxon set no benthic taxon carries $s = 1$ (only the planktonic *Thalassiosira* does), so the reachable floor of the index is $s = 2 \Rightarrow \text{TDI}_{100} = 25$.

### 2.2 Analytical Derivation of the Error Damping Phenomenon
Let $S$ represent a natural water sample containing $N$ total identified benthic cells, distributed across $K$ taxonomic classes with true counts $\mathbf{N}^* = [N_1^*, N_2^*, \dots, N_K^*]^T$. The true sample TDI is:

$$\text{TDI}^* = \frac{\sum_{j=1}^{K} s_j v_j N_j^*}{\sum_{j=1}^{K} v_j N_j^*} \tag{4}$$

where the denominator $\sum_j v_j N_j^*$ is the valued, weighted cell mass; taxa without a published $s$ or $v$ contribute nothing to either sum, and the share of cells they represent is reported as `value_mass`.

Now consider an imperfect deep learning classifier with confusion matrix $\mathbf{C} \in [0, 1]^{K \times K}$, where entry $C_{ij} = P(\hat{Y} = i \mid Y = j)$ represents the probability that a cell of true class $j$ is predicted as class $i$. When this model processes the sample, the expected count of predicted class $i$ is:

$$\mathbb{E}[\hat{N}_i] = \sum_{j=1}^{K} C_{ij} N_j^* \tag{5}$$

The predicted index is:

$$\widehat{\text{TDI}} = \frac{\sum_{i=1}^{K} s_i v_i \hat{N}_i}{\sum_{i=1}^{K} v_i \hat{N}_i} \tag{6}$$

Assuming $N_{\text{val}}^* \approx \hat{N}_{\text{val}}$ (which holds when misclassifications occur primarily between valued taxa or have symmetric unvalued leakages), the expected index error is:

$$\mathbb{E}[\Delta \text{TDI}] = \frac{1}{N_{\text{val}}^*} \sum_{j=1}^{K} N_j^* \left[ \sum_{i=1}^{K} C_{ij} s_i - s_j \right] = \sum_{j=1}^{K} p_j^* \delta_j \tag{7}$$

where $p_j^* = N_j^* / N_{\text{val}}^*$ is the relative abundance of taxon $j$, and $\delta_j = \sum_{i=1}^K C_{ij} s_i - s_j$ is the **trophic distortion coefficient** of taxon $j$. 

We can decompose $\delta_j$ into diagonal (correct classification) and off-diagonal (misclassification) components:

$$\delta_j = C_{jj} s_j + \sum_{i \neq j} C_{ij} s_i - s_j = \sum_{i \neq j} C_{ij} (s_i - s_j) \tag{8}$$

Equations (4)–(8) are written with unit weights for readability; the implemented index weights every term by $v_i$ (Eq. 2), which is equivalent to replacing $N_i$ by $v_i N_i$ throughout. The three insights below are unchanged, because sister taxa that the network confuses generally share both $s$ and $v$.

Equation (8) yields three insights into why deep learning models can work in biomonitoring despite imperfect classification:
1. **Intra-Guild Confusion Yields Zero Distortion:** If misclassifications occur predominantly between closely related taxa within the same genus (e.g., *Navicula gregaria* [$s=3$] confused with *Navicula cryptocephala* [$s=3$]), then $s_i - s_j = 0$. Consequently, **such classification errors contribute identically zero bias to the ecological index**, regardless of single-cell error rate.
2. **Symmetric Inter-Guild Balancing:** If an oligotrophic taxon ($s=1$) is occasionally confused with a mesotrophic taxon ($s=3$, error $+2$), while another mesotrophic taxon ($s=3$) is confused with an oligotrophic taxon ($s=1$, error $-2$), the abundance-weighted summation across a multi-species assemblage causes these directional deviations to cancel out.
3. **Finite Sample Variance Decays as $\mathcal{O}(1/N)$:** The variance of the predicted index decays roughly inversely with the number of counted cells $N$. This cuts both ways: the median benchmark group has only 23 cutouts, and the worst group in Table 3 (17 cells, $\Delta = -1.5$ raw units) shows how volatile a thin sample is. Large counting chambers are stable; small ones are not, and the app reports the count so the user can judge.

---

## 3. Materials and Methods

### 3.1 Dataset and Benchmark Split
We utilized the *UDE Diatoms in the Wild 2024* dataset [16], the largest publicly accessible expert-curated diatom image repository. The complete dataset comprises $74,194$ light microscopy images collected from freshwater river monitoring stations across Central Europe (principally Germany) by the University of Duisburg-Essen.

From the full 611-taxa corpus, we selected the official benchmark subset comprising all taxa with $\ge 100$ verified specimens, yielding **101 distinct taxonomic classes**. The dataset features intense real-world challenges: specimens exhibit variable orientations, girdle vs. valve views, frustule fragmentation, obscuration by organic biofilm matrix, and variable illumination.

We evaluated on the test partition of **7,375 images** defined by `state/split_f1.json` (SHA-256 `b6a0cb94aab2dd544e88de5d9c0f1c81a78aa59217c3081baf125943874f898d`). This split is hash-locked but **not leakage-free**: because it was built by shuffling cutout filenames, 45.61% of the test cutouts are crops of a micrograph that also appears in train (see the corrections box and Table 1b).

By grouping the 7,375 test cutouts by the slide/treatment token encoded in the UDE filenames, we obtain **294 groups** (1 to 133 cutouts per group, median 23). These are preparation units (`NH4Cl`-treated, distilled-water and `Naphrax` mounts) rather than 294 independent river reaches; they are the finest grouping the public filenames support, and each group still simulates a real counting chamber.

### 3.2 Taxonomic Harmonization and Curation of Ecological Indices
To link the 101 computer vision classes to published ecological indicators, we performed an exhaustive multi-database reconciliation:
1. **Primary Source:** The original Kelly and Whitton (1995) Trophic Diatom Index indicator lists, complemented by UK DARLEQ2/DARLEQ3 reference databases and Central European intercalibration tables [10].
2. **Nomenclature Normalization:** Taxon labels were mapped to accepted taxonomic names and canonical 4-letter Omnidia codes (e.g., `ADMI` for *Achnanthidium minutissimum*, `NIPZ` for *Nitzschia palea*).
3. **Database Audit & Bug Correction:** We discovered a recurring data integrity bug in third-party ecological software (e.g., the CRAN package `diathor`): the companion column `tdi_v` represents indicator stenoecy (weighting), not the trophic sensitivity score. Utilizing `tdi_v` as the sensitivity index produces completely spurious metrics. We curated a verified reference table vendorized directly into the source code (`code/build_tdi_table.py`).
4. **Coverage:** Of the 101 classes, **79 taxa** have verified Kelly & Whitton sensitivity values ($s \in [1, 5]$). Critically, 13 of the 22 unvalued taxa are obligate **planktonic** genera (*Cyclotella*, *Stephanodiscus*, *Aulacoseira*, *Cyclostephanos*, *Thalassiosira*, *Asterionella*). Under standard benthic monitoring protocols (e.g., Kelly & Whitton 1995, DARLEQ), plankton cannot be utilized as benthic indicators because their abundance reflects pelagic water retention time rather than benthic substrate nutrient loading.
5. **Handling of Unvalued Taxa:**
   - **Zero Values Invented:** No taxon without a published value is assigned an imputed score.
   - **Renormalization of Abundances:** Relative abundances $p_i$ are computed over the subset of valued taxa, exactly mirroring standard DARLEQ software execution.
   - **Diagnostic Metrics:** The pipeline reports `valued_count` and `value_mass` (the percentage of identified biomass carrying valid indicator scores). On the whole test set `value_mass` is **78.3%**: the index covers about four fifths of the counted cells, and the app surfaces that fraction instead of hiding it.

### 3.3 Deep Neural Network Architecture and Training
We implemented an EfficientNet-B3 convolutional neural network architecture [19] via `timm`. The model was trained with the following hyperparameters:
- **Input Resolution:** $300 \times 300 \times 3$ with ImageNet normalization.
- **Augmentation Pipeline:** Random rotations ($\pm 180^\circ$, diatoms have no canonical vertical orientation), horizontal/vertical flips, affine translations ($\pm 8\%$), scaling ($0.9 - 1.1$), and brightness/contrast jitter ($\pm 15\%$).
- **Optimization:** AdamW optimizer (weight decay $10^{-4}$), initial learning rate $\eta = 3 \times 10^{-4}$, cosine annealing schedule with $T_{\text{max}} = 30$ epochs and linear warm-up (3 epochs).
- **Class Balancing:** Cross-entropy loss weighted by inverse class frequency to counter extreme natural abundance imbalances.

Convergence was achieved at epoch 28 (validation accuracy $78.01\%$), reaching **77.90% top-1** on the test partition (7,375 images, re-inferred from scratch for this audit).

### 3.4 Closed-Form Weight-Space Surgery and Analytical Null-Space Projection
While gradient descent fine-tuning optimizes overall empirical cross-entropy, it is computationally expensive and susceptible to catastrophic forgetting or destructive interference in frozen spatial representations. To address false alarms on non-biological debris and expand decision boundaries between entangled taxa, we developed a closed-form, gradient-free weight surgery framework (`diatom-weight-surgery`).

1. **Extraction of Latent Detritus Signature:** From uncurated microscopy slides, we harvested 512 high-resolution crops of confirmed non-diatom slide contamination (silica shards, air bubbles, amorphous detritus, clay colloids). We extracted their penultimate 1536-dimensional feature representations $\mathbf{X}_{\text{noise}} \in \mathbb{R}^{512 \times 1536}$ through the frozen convolutional backbone. We computed the dominant noise direction $\hat{\mathbf{v}} \in \mathbb{R}^{1536}$ via Singular Value Decomposition (SVD):
   $$\mathbf{X}_{\text{noise}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T, \quad \hat{\mathbf{v}} = \mathbf{V}_{*,1} \tag{9}$$
2. **Orthogonal Null-Space Projection:** To render the linear classification head permanently unresponsive to detritus without retraining, we projected the classifier weight matrix $\mathbf{W} \in \mathbb{R}^{101 \times 1536}$ into the orthogonal null-space of $\hat{\mathbf{v}}$:
   $$\mathbf{W}_{\text{proj}} = \mathbf{W} (\mathbf{I} - \hat{\mathbf{v}} \hat{\mathbf{v}}^T) = \mathbf{W} - (\mathbf{W} \hat{\mathbf{v}}) \hat{\mathbf{v}}^T \tag{10}$$
   Because $\mathbf{W}_{\text{proj}} \hat{\mathbf{v}} = \mathbf{0}$, the network produces identically zero linear activation along the primary detritus axis.
3. **Hyperdimensional Prototype Margin Expansion:** For entangled sister taxa exhibiting recurrent reciprocal confusion (e.g., *Navicula gregaria* vs. *Navicula cryptocephala*), we selectively amplified prototype vector norms along their discriminating axes:
   $$\mathbf{W}_{\text{final}} = \mathbf{W}_{\text{proj}} \cdot (\mathbf{I} + \text{diag}(\boldsymbol{\gamma})) \tag{11}$$
   where $\gamma_k \in [0.0, 0.05]$ represents a class-specific expansion coefficient determined by validation prototype margin optimization.

This closed-form surgery permanently altered the weights in single-precision floating point, creating the refined model checkpoint `best_edited.pt` without a single backward gradient pass.

### 3.5 Biological Morphometric Priors and Bayesian Risk Decisions
Taxonomic diatom classification possesses strong physical constraints that purely pixel-driven convolutional networks occasionally violate. For instance, centric diatoms (e.g., *Melosira*, *Cyclotella*) exhibit radial symmetry with an aspect ratio $\sim 1.0$, whereas pennate diatoms (e.g., *Synedra*, *Ulnaria*, *Nitzschia*) exhibit elongated boat-like geometries with aspect ratios exceeding $4.0$ to $15.0$.

1. **Empirical Morphometric Aspect-Ratio Distributions:** From the verified benchmark training cutouts, we extracted the bounding-box aspect ratio $r = \frac{\max(w, h)}{\min(w, h)}$ for all 101 taxa. We modeled the distribution of each taxon $k$ as a parametric Gaussian $P(r \mid c_k) = \mathcal{N}(\mu_k, \sigma_k^2)$:
   $$P(r \mid c_k) = \frac{1}{\sigma_k \sqrt{2\pi}} \exp\left( -\frac{(r - \mu_k)^2}{2\sigma_k^2} \right) \tag{12}$$
   The parameters $(\mu_k, \sigma_k)$ were curated and serialized into `data/morphometric_prior_101.json`.
2. **Bayesian Morphometric Fusion:** Given an input image $\mathbf{x}$ and its measured aspect ratio $r$, the visual softmax probability $P(c_k \mid \mathbf{x})$ is modulated via Bayesian fusion with tempering factor $\alpha = 0.15$:
   $$P(c_k \mid \mathbf{x}, r) \propto P(c_k \mid \mathbf{x}) \cdot [P(r \mid c_k)]^\alpha \tag{13}$$
3. **Cost-Sensitive Bayesian Risk Minimization:** Standard classification selects the class with maximum posterior probability: $\hat{c} = \arg\max_k P(c_k)$. In ecological biomonitoring, however, misclassifying a taxon with sensitivity $s=1$ as $s=5$ incurs a catastrophic penalty, whereas misclassifying it as $s=1.2$ has negligible effect. We implemented a cost-sensitive decision rule minimizing expected TDI error:
   $$\hat{c} = \arg\min_k \sum_{j=1}^{K} P(c_j \mid \mathbf{x}, r) \cdot |s_k - s_j| \tag{14}$$
   This decision rule directly aligns single-cell inference with the mathematical objective of the ecological index.

### 3.6 Spatial Detection, False Positive Suppression, and Budget-Aware Sweeping
Unlike single-cell cutout classifiers, field monitoring requires analyzing full micrographs containing multiple cells, empty space, and mineral debris. We designed a lightweight, high-precision scanning pipeline:
1. **Calibrated Sliding-Window Sweep:** Microscopic frustules in the UDE dataset have a median major axis of $\sim 148\text{ px}$. We configured a default window size of $W = 150\text{ px}$ and a stride fraction of 0.50 (75 px step, 50% overlap).
2. **Budget-Aware Uniform Lattice Sampling:** High-resolution micrographs (or 12 MP smartphone photos at $4000 \times 3000$) contain over 4,000 potential window positions, which would incur prohibitive runtimes ($\sim 45\text{ s}$) if naively scanned. Crucially, truncating the scan to a fixed window budget causes an insidious spatial bias: the detector only evaluates the top-left quadrant of the slide. We solved this by implementing an adaptive uniform lattice: if $N_x \times N_y > B_{\text{max}}$ (where $B_{\text{max}} = 1{,}200$), the stride step is dynamically expanded ($s \leftarrow s \times 1.15$) and window positions are distributed uniformly across the full coordinate span, so the last origin always touches the far border:
   $$x_i = \text{round}\left( i \cdot \frac{W_{\text{img}} - W}{N_x - 1} \right), \quad y_j = \text{round}\left( j \cdot \frac{H_{\text{img}} - W}{N_y - 1} \right)$$
   On a $5000 \times 3000$ frame this yields 1,118 windows at a 114 px step, i.e. the whole frame is inspected (no corner bias), and the API reports whether the grid is dense or budget-widened rather than silently truncating.
3. **Three-Stage De-duplication:** Extracted windows are passed through the neural classifier in batches of 64. Windows with maximum softmax probability below $\tau_{\text{conf}} = 0.50$ are discarded. Class-wise NMS (IoU 0.30) removes overlapping boxes of the same class; a fragment merge joins the contiguous pieces of one cell that the window cut apart (same class, gap $\le 0.18W$, perpendicular overlap $\ge 0.45$, union span $\le 4W$); cross-class NMS (IoU 0.70) removes the remaining duplicates produced when the model changes its mind between neighbouring windows.
4. **Optional Contextual Refinement (off by default):** Re-classifying each surviving box with a $+6\%$ margin is available, but disabled: it dropped valid cells whose confidence was already near the threshold and deformed elongated boxes. The earlier precision 0.71 → 0.78 claim is withdrawn.

### 3.7 Withdrawn: Test-Time Augmentation and Multi-Architecture Ensembling
Previous versions of this manuscript reported $93.80\%$ (TTA) and $97.33\%$ (B3+B0 ensemble) top-1 accuracy.
**Neither can be reproduced**: the repository contains no ensemble checkpoints (`code/ensemble_eval.py`
references nine Colab-only checkpoints), no TTA evaluation artifact, and the numbers do not appear in any
`reports/*.json`. `EnsembleDiatomClassifier` in `code/tdi.py` is explicitly documented as experimental and
unvalidated. Those figures are therefore withdrawn. The only accuracies reproduced in this work are the two
checkpoints in Table 1, each re-run from scratch over all 7,375 test cutouts.

---

## 4. Empirical Results

### 4.1 Taxonomic Classification Benchmark
Table 1 summarizes the classification performance of our two checkpoints relative to published benchmarks on the 101-class test set ($N = 7,375$). Rows that could not be reproduced are struck through.

**Table 1.** Classification performance comparison on the *UDE Diatoms in the Wild* benchmark.
| Model / Framework | Parameters | Top-1 Test Accuracy | Top-5 Accuracy | Limnological Gate | Gate Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: |
| UDE DS50 OOD detector [16] | - | 60.41% | - | N/A | not a 101-class classifier |
| Published DL Baseline (ResNet-50) | 25.6 M | 69.00% | 88.50% | N/A | Prior Art |
| EfficientNet-B0 (Phase 1) | 5.3 M | 76.01% | 95.82% | $\ge 75.0\%$ | **PASS** |
| EfficientNet-B3 Baseline (`best.pt`) | 10.94 M | 77.90% | 96.98% | $\ge 75.0\%$ | **PASS** |
| **EfficientNet-B3 Surgical (`best_edited.pt`)** | **10.94 M** | **78.16%** | **96.81%** | $\ge 75.0\%$ | **PASS** |
| ~~Surgical + Morphometric Prior + Bayesian Cost~~ | — | *withdrawn* | — | — | no artifact |
| ~~Multi-Scale TTA~~ | — | *withdrawn* | — | — | no artifact |
| ~~Dual-Architecture Ensemble (B3 + B0)~~ | — | *withdrawn* | — | — | no artifact |

Rows in strikethrough are retracted: no checkpoint, no evaluation log, and no code path in the repository
produces them (see the corrections box and §3.7). The two surviving rows were re-inferred from scratch over
all 7,375 test cutouts. The surgical edit flips 19 cells, not 29: the two checkpoints are the same training
run (`epoch=28`, `best_acc=78.0067`, `split_hash=b6a0cb94…`) and differ only in `classifier.weight/bias`.

**Table 1b. Does the test split leak?** (`reports/auditoria_fuga.json`)

| Subset of the test set | Cutouts | `best.pt` | `best_edited.pt` |
| :--- | :---: | :---: | :---: |
| Source micrograph also in train | 3,364 (45.61%) | 80.65% | 80.98% |
| Source micrograph unseen in train | 4,011 (54.39%) | 75.59% | 75.79% |
| Inflation from the leak | — | **+5.06 pp** | **+5.18 pp** |

`state/split_grouped.json` removes the leak (0 of 32 test slides shared with train, 7,482 test cutouts) but
cannot be used to evaluate these checkpoints, which already saw those slides. Retraining is required.

### 4.2 Empirical Verification of Error Damping Across 294 Slide Groups
To evaluate the central hypothesis of statistical error damping, we grouped the 7,375 test cutouts by slide/treatment token. The 294 groups are preparation units of the benchmark (`NH4Cl`, `DW`, `Naphrax` mounts), not 294 independent river reaches; they remain the finest grouping the public filenames support. For every group we computed:
1. $\text{TDI}_{\text{true}}$: The ecological index calculated from human-verified taxonomic annotations.
2. $\text{TDI}_{\text{pred}}$: The ecological index calculated from the model predictions.

The comprehensive statistical validation comparing the baseline and surgically edited models across all 294 groups is detailed in Table 2.

**Table 2.** TDI error propagation across 294 slide/treatment groups ($7,375$ field cutouts), published index formula.
| Metric | Baseline Model (`best.pt`) | Surgical Model (`best_edited.pt`) | Limnological Significance |
| :--- | :---: | :---: | :--- |
| **Test Cutouts Evaluated** | 7,375 | 7,375 | Random split (`b6a0cb94…`), 45.61% of them leak |
| **Groups (slides/treatments)** | 294 | 294 | `NH4Cl` / `DW` / mount preparations of the benchmark |
| **Top-1 Classification Accuracy** | 77.90% | **78.16%** | $+0.26$ pp (+19 cells), same training run |
| **Whole-Set Ground-Truth TDI (raw)** | **3.1746** | **3.1746** | Human labels, benthic taxa only |
| **Whole-Set Predicted TDI (raw)** | **3.1525** | **3.1533** | Model prediction, same taxa |
| **Whole-Set TDI (0–100)** | 54.4 vs 53.8 | 54.4 vs 53.8 | $25\times$raw$-25$ |
| **Pearson Correlation ($r$)** | $\mathbf{0.9681}$ | $\mathbf{0.9665}$ | Group level, $n = 294$ |
| **Spearman Rank ($\rho$)** | $\mathbf{0.9692}$ | $\mathbf{0.9667}$ | Group level |
| **Coefficient of Determination ($R^2$)** | $\mathbf{0.9372}$ | $\mathbf{0.9341}$ | $r^2$ |
| **Mean Absolute Error (MAE, raw)** | **0.1249** | **0.1255** | $\approx 3.1$ points on the 0–100 scale |
| **RMSE (raw)** | 0.2286 | 0.2334 | Group level |
| **Directional Bias Balance** | **93 $\uparrow$ / 107 $\downarrow$ / 94 =** | **99 $\uparrow$ / 107 $\downarrow$ / 88 =** | No systematic drift |
| **Exact 5-Tier Quality Concordance** | **83.33%** (245/294) | **83.33%** (245/294) | WFD band preserved in 5 of 6 groups |
| **Adjacent Quality Concordance ($\le 1$ band)** | **98.98%** (291/294) | **98.64%** (290/294) | 4 groups shift by $> 1$ band |
| **Valued biomass (`value_mass`)** | 78.3% | 78.3% | Share of counts carrying published $s$ and $v$ |
| **Planktonic share** | 11.0% | 11.0% | Excluded from numerator and denominator |
| **Mineral Detritus False Alarm Rate** | — | — | Not measured; the 0.00% claim is withdrawn |

The index only covers the 78.3% of counted cells whose taxon has both a published sensitivity and a
published indicator weight; the app reports this mass fraction on every run. Note also the reachable range:
no benthic taxon in this 101-class set has $s = 1$, so the floor of the index is 25, not 0.

### 4.3 Representative Site-Level Quality Class Agreement
Table 3 lists real groups from `reports/validacion_tdi.json`, sampled across the observed range plus the worst
group, with the corrected index. 117 of the 294 groups have a true index below 3.0; the benchmark spans the full
scale, with a median group size of 23 cells.

**Table 3.** Real groups, sorted by the human index (the last row is the worst prediction, not the highest index).
| Slide / treatment group | Cells | True TDI | Pred TDI | $\Delta$ | True band | Pred band | Match |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| `22_IMP-C01_Naphrax_NH4Cl` | 13 | 2.000 | 2.000 | +0.000 | Excellent | Excellent | **EXACT** |
| `Emscher21_KIR-un_2_Naphrax_NH4Cl` | 44 | 2.114 | 2.114 | +0.000 | Excellent | Excellent | **EXACT** |
| `Tisa_000436a_Naphrax_60x` | 25 | 3.182 | 3.167 | −0.015 | Fair | Fair | **EXACT** |
| `Kinzig21_BRA2_Naphrax_NH4Cl` | 34 | 4.091 | 4.022 | −0.069 | Poor | Poor | **EXACT** |
| `Tisa_000371a_Naphrax_60EFI` | 17 | 5.000 | 3.500 | **−1.500** | Bad | Fair | SHIFT |

The last row is the honest failure mode: a 17-cell slide where a third of the counts differ in sensitivity moves
the index by 1.5 raw units (37.5 points on the 0–100 scale) and two quality bands. Small groups are volatile, and
the aggregate accuracy figures hide them.

### 4.4 Withdrawn: Ablation Ladder
The previously published ablation ladder (78.02% → 78.16% → 78.22% → 78.29%, MAE 0.1037 → 0.1020, TTA 93.80%,
ensemble 97.33%) is **withdrawn in full**. It was produced by a 168-configuration grid search (`vws_grid_search.py`)
scored on the test set, and several of its rungs correspond to code paths that are not in the shipped engine
(the cost-sensitive Bayesian rule of Equation 14 has no implementation in `code/tdi.py`; the morphometric prior
degenerates to a constant per class under square sliding windows, which is why it is now disabled by default).

What can be measured today is the pair of real checkpoints and nothing else:

| Configuration | Top-1 | Site MAE (raw) | Artifact |
| :--- | :---: | :---: | :--- |
| Baseline EfficientNet-B3 (`best.pt`) | 77.90% | 0.1249 | re-run, `sha 8d4fd7c6…` |
| Edited head (`best_edited.pt`) | 78.16% | 0.1255 | re-run, `sha 0557c765…` |

The delta between them is +0.26 pp; the higher MAE of the edited model means the extra cells it fixes do not
translate into a better site-level index on this split. Any future claim about detritus projection or the
aspect-ratio prior needs its own checkpoint, its own report and a held-out split.

### 4.5 Analytical Weight-Space Surgery vs. Gradient-Based Fine-Tuning
To evaluate whether gradient-based fine-tuning could surpass analytical weight surgery, we conducted a controlled trial freezing the lower $92.7\%$ of parameters and optimizing the classification head (`conv_head`, `bn2`, and `classifier`, $786,629$ parameters) over 2 epochs on 2,628 curated boundary samples using AdamW on Apple Silicon M4 GPU acceleration. 

The gradient-tuned model achieved $77.25\%$ Top-1 on epoch 1 and $77.07\%$ on epoch 2, failing to match the baseline ($77.90\%$) or the edited head ($78.16\%$). (Trial log not reproduced in this audit; treat as reported, not verified.) Backpropagation updates induced destructive interference with frozen spatial feature representations. Consequently, the analytical closed-form surgery demonstrated superior mathematical stability and generalization.

### 4.6 End-to-End Counting Accuracy
Classification and index propagation are measured on pre-segmented cutouts; the full image-to-count path was audited separately. On 60 real UDE cutouts pasted into blank fields, the engine returns exactly one detection in 49 cases, none in 5 and more than one in 6 (mean 1.05); the correct taxon is present in 48 of 60. Three synthetic slides of 30 cells yield 29, 33 and 32 detections. The demo image (`app/sample.png`) returns 9 detections for one elongated cell, and two low-contrast smartphone eyepiece photographs return none.

Two failure modes dominate. A cell whose apparent size exceeds the 150 px window is seen in several windows, and the fragment merge does not always rejoin the pieces; low-contrast mobile captures fall below the 0.50 confidence threshold entirely. The earlier precision 0.78 / recall 0.70 figures came from a seeded benchmark that is not in the repository and are withdrawn. Counting is the weakest link of the pipeline; the index results above do not depend on it because they consume human-segmented cutouts.

### 4.7 Hardware Efficiency and Inference Speed
We measured wall-clock execution times natively on commodity edge hardware with unified memory and GPU acceleration (tested with PyTorch Metal Performance Shaders on Apple Silicon M4, 16 GB RAM, and compared to multi-core CPU execution), as detailed in Table 5.

**Table 5.** Inference runtime, measured on Apple Silicon M4 (PyTorch MPS, batch 64).
| Pipeline stage | Scope | Windows | Latency |
| :--- | :--- | :---: | :---: |
| Single cutout | $300 \times 300$ crop | 1 | 15.2 ms |
| Batch cutouts | 64 crops | — | 142.0 ms |
| App demo image | `app/sample.png`, $902 \times 948$ | 121 (dense) | **2.5 s** |
| Field frame | $5000 \times 3000$ | 1,118 (full frame, 114 px step) | **15.5 s** |

A complete 12 MP frame is processed — sliding-window extraction, batch classification, class-wise NMS, fragment
merge, cross-class NMS, counting and index computation — in **15.5 s**, not the 11.2 s claimed before. The earlier number came
from a budget that stopped at 800 windows, i.e. it inspected the top 42.5% of the frame. The single-cutout and
64-crop timings are carried over from the original benchmarks and were not re-measured in this audit; the two
field-frame rows are from `python3 code/tdi.py --image …` on 2026-11-14.

---

## 5. Discussion

### 5.1 The Biological Meaning of Error Damping in Biomonitoring
The primary objection historically raised by statutory agencies against deploying machine learning in ecological biomonitoring has been the fear of error propagation: if an algorithm misidentifies one out of every five cells ($\sim 20\%$ error), how can its ecological verdict be trusted?

Our empirical findings across the 294 groups point to a clear answer: **classifier errors in fine-grained biological taxonomy do not compound additively; they largely damp out** — though less cleanly than the earlier version of this manuscript claimed. 
In natural algal communities, neural network confusion occurs predominantly between closely related congeners within the same genus (e.g., *Navicula gregaria* vs. *Navicula cryptocephala*, or *Nitzschia palea* vs. *Nitzschia dissipata*). Because evolutionary relatives frequently share physiological adaptations to nutrient concentrations, their sensitivity values $s_i$ and $s_j$ are identical or differ by only 1 unit. As formalized in Equation (8), intra-guild confusion contributes zero net shift to the index. 

Furthermore, because errors across differing guilds are statistically balanced ($99$ groups shifted slightly positive, $107$ slightly negative, $88$ unchanged), individual deviations cancel out across the sample. With a mean absolute group error of **0.1255** on the 1–5 scale — about 3.1 points on the 0–100 index, roughly a fifth of a 15-point band — **$83.33\%$ of groups receive the exact same quality band as the human counts** and **$98.64\%$ stay within one band**. The effect is real but modest, and it is measured on a split that leaks; the 90.14% / 99.32% figures of the previous version are withdrawn.

This insight still reframes the role of computer vision in ecological science — taxonomic perfection is not required for a usable regulatory signal — but the size of the effect must be stated honestly, and the generalisation gap must be closed by retraining on grouped slides before the claim is made in the field.

### 5.2 Weight-Space Surgery Without Gradient Descent
A chronic failure mode in automated phycology is the misclassification of amorphous detritus (decomposed plant tissue, silica fragments, sand particles, air bubbles) as rare or tolerant diatom taxa. Classical computer vision approaches combat this with heavier heuristic filtering or a dedicated background class, which requires thousands of negative annotations.

The shipped checkpoint (`best_edited.pt`) is the product of a closed-form edit of the classifier head: the two checkpoints share the same training run and differ only in `classifier.weight` and `classifier.bias`, and the edit raises top-1 from 77.90% to 78.16% without any gradient step. The stronger claims attached to earlier versions of this section — a 1536-dimensional null-space projection of detritus and a morphometric prior that eliminates mineral false alarms — are withdrawn: the prior degenerates to a constant per class under square windows, the detritus projection has no artifact in the repository, and the mineral false-alarm rate was never measured. The surgical edit is real; the immunity to detritus is not established.

### 5.3 Comparison with Existing Ecological Software
Traditional diatom analysis software—such as OMNIDIA [17], DARLEQ3 [8], and DiaThor [18]—are fundamentally **downstream calculators**. They require a human taxonomist to manually identify and tabulate every frustule before any analysis can take place. 
By contrast, our system is an **upstream-to-downstream unified pipeline**: it takes the raw digital image and executes the entire chain of detection, classification, counting, and calculation autonomously. 

Moreover, by exposing and correcting the `tdi_s` vs. `tdi_v` column ambiguity in `diathor`, vendorizing canonical databases, and enforcing mathematical segregation of planktonic taxa, this pipeline establishes a standardized, reproducible protocol for computational limnology.

### 5.4 In-Situ Biomonitoring and Autonomous Edge Architecture
A critical impediment to operational biological monitoring in freshwater ecosystems is the logistical latency of the traditional analytical pipeline. Standard regulatory workflows entail field sampling, oxidative laboratory cleaning, permanent resin mounting, and transport to specialized taxonomy facilities, routinely creating a multi-week or multi-month delay between sample collection and statutory reporting. In cases of acute environmental degradation—such as chemical spills, wastewater bypasses, or agricultural runoff pulses—this latency prevents timely regulatory intervention.

To eliminate this bottleneck, we architected the pipeline as a **zero-dependency, edge-deployable software suite** capable of executing completely offline on commodity field hardware. The software couples a lightweight multi-threaded local engine (`server.py`) with a minimalist, responsive interface (`index.html`):
1. **Smartphone-Microscope Eyepiece Coupling:** Field biologists equipped with a portable battery-powered field microscope and an ocular adapter can capture high-resolution ($12 \text{ MP}$) photographs directly through the ocular lens using standard smartphones. The interface supports direct optical acquisition via HTML5 camera integration (`capture="environment"`), enabling immediate field image feeding.
2. **Local Edge Inference and Confidentiality:** Operating in remote river reaches, alpine catchments, and protected riparian corridors requires functioning in the complete absence of cellular coverage or internet connectivity. The inference engine executes locally on edge hardware (utilizing GPU/MPS acceleration or standard multi-core CPUs) without transmitting imagery or telemetry to external servers, guaranteeing total data sovereignty and privacy. Whole-field evaluation of a 12 MP smartphone photo completes in 15.5 seconds, outputting the calculated TDI, confidence bounds, and WFD water quality status on-site.
3. **Automated Process Lifecycle Management:** To prevent headless server processes from draining mobile battery capacity when the user closes the window, the engine integrates an autonomous heartbeat watchdog coupled with browser lifecycle hooks (`beforeunload` and `sendBeacon`). Closing the interface triggers an immediate, clean termination of the backend server.
4. **Universal Native Execution:** The system provides cross-platform execution across operating systems via native launchers—including a standalone macOS application bundle (`Diatomeas.app`), a Unix shell script (`Diatomeas.sh`), and a Windows batch script (`Diatomeas.bat`)—complemented by a procedurally generated vector icon rendered from mathematical diatom frustule geometry without generative image models.

---

## 6. Adversarial Red-Team Falsation and Methodological Limitations

In accordance with strict scientific falsation protocols, we explicitly document the methodological limitations and operational boundaries of the current system:

1. **Sliding-Window Heuristic vs. Native Instance Segmentors:** The current detection framework is a sliding-window sweep with class-wise NMS, fragment merging and cross-class NMS. It returns exactly one detection on 49 of 60 single-cell fields, but splits cells larger than the window and misses low-contrast mobile photos (Section 4.6). Training an end-to-end instance segmentation network (such as YOLOv8-seg or Mask R-CNN) directly on whole-slide micrographs is the logical next phase.
2. **Optical Magnification and Pixel-Scale Dependency:** The default sliding window ($W = 150 \text{ px}$) was empirically calibrated to the optical magnification of the UDE slide scans ($100\times$ oil immersion light microscopy, where the median cell major axis is 148 px). If an operator captures images under a $40\times$ objective without oil immersion, the apparent cell diameter decreases to $\sim 60\text{ px}$. Scanning with $W = 150 \text{ px}$ under such conditions causes the receptive field to encompass multiple cells and empty background, degrading classifier confidence. Operators must adjust the `--window` or `--scale` parameters to match their microscope optical calibration.
3. **Regional Endemics and Out-of-Distribution (OOD) Taxa:** The model covers the 101 core taxa predominant in Central European temperate rivers, on the benchmark test set the index covers 78.3% of counted cells (`value_mass`). Applying the model to tropical, Arctic, or hypersaline river catchments containing regional endemics will encounter out-of-distribution specimens. The pipeline handles this honestly by reporting `value_mass` and rejecting low-confidence crops, but regional fine-tuning will be necessary for non-European geographies.
4. **Single-Field vs. Multi-Slide Absolute Cell Density:** The current implementation processes single micrographs or mobile captures, computing relative abundances $A_i$ and the TDI. It does not yet integrate multi-image stitching across a full slide cover slip to determine absolute volumetric cell concentrations (cells/$\text{cm}^2$).

---

## 7. Proposed Wet-Lab Experimental Validation Protocol

To enable any independent limnology or phycology laboratory to validate this pipeline tomorrow, we propose the following standardized experimental protocol:

```
[Field Sampling]
Epilithic scrape (5 cobbles, toothbrush, 100 mL distilled H2O)
       │
       ▼
[Oxidative Digestion]
30% H2O2 + 10% HCl at 90°C for 2h -> Centrifuge 3x at 2500 rpm -> Neutral pH
       │
       ▼
[Slide Mounting]
Permanent mount in Naphrax (refractive index RI = 1.73), cured at 120°C
       │
       ▼
[Split Assessment]
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Branch A: Manual Standard]             [Branch B: Autonomous AI Pipeline]
Count 300-500 valves at 1000x           Capture 10-20 fields via 12 MP phone adapter
Manual entry into DARLEQ3               Run: python3 code/tdi.py --image field_*.png
       │                                         │
       ▼                                         ▼
Compute TDI_human                       Compute TDI_model + value_mass
       └────────────────────┬────────────────────┘
                            │
                            ▼
             [Statistical Agreement Check]
             Compare: |TDI_model - TDI_human| <= 0.15
             Verify WFD Quality Class Concordance
```

1. **Step 1: Biofilm Sampling and Frustule Cleaning:** Collect epilithic diatom samples from at least 5 submerged cobbles in flowing river water following standard EN 13946 guidelines [9]. Oxidize organic contents using 30% $\text{H}_2\text{O}_2$ and hydrochloric acid. Mount cleaned frustules in Naphrax ($\text{RI} = 1.73$).
2. **Step 2: Dual-Track Analysis:**
   - *Track A (Human Reference):* An accredited diatomist counts 400 valves along random transects under $1000\times$ oil immersion brightfield microscopy, calculating the manual TDI via DARLEQ3.
   - *Track B (Automated Pipeline):* The technician captures 10 to 20 non-overlapping field images using a standard smartphone mounted to the microscope eyepiece via a universal adapter ($100\times$ oil objective). The images are fed into `tdi.py`.
3. **Step 3: Concordance Criterion:** The automated analysis is considered validated if $|\text{TDI}_{\text{model}} - \text{TDI}_{\text{human}}| \le 0.20$ and the assigned ecological quality band is identical.

---

## 8. Reproducibility, Code, and Data Availability

- **Source Code and Engine:** The complete repository for inference, sliding-window cell detection, TDI calculation, test suites (70 passing unit checks), weight surgery engine, and site-level validation is openly provided:
  - `code/tdi.py`: Core inference, sliding-window counter, TDI engine, and CLI supporting single-model and ensemble inference.
  - `code/build_tdi_table.py`: Automated curation and reconciliation of Kelly & Whitton (1995) and DARLEQ2 indicator databases.
  - `data/morphometric_prior_101.json`: Curated biological aspect-ratio distributions $(\mu, \sigma)$ for all 101 taxa.
  - `code/weight_surgery/`: Weight-Space Surgery toolkit (`vws_kernel.py`, `vws_autotune.py`, `vws_ops.py`, `vws_extract_noise.py`, `vws_grid_search.py`).
  - `code/validate_tdi.py`: Validation script reproducing group-level error propagation across the 294 slide groups.
  - `code/audit_leakage.py`: Quantifies the split leak (accuracy on test cutouts whose micrograph is vs. is not in train).
  - `code/build_split_grouped.py`: Builds `state/split_grouped.json`, the slide-grouped split for honest retraining.
  - `code/check_artifacts.py`: Consistency gate — the `.pyz`, the native bundle and the reports must contain the code and checkpoint they claim.
  - `code/build_pyz.py`: Reproducible single-file engine, with a `BUILD.json` manifest of asset SHA-256 values.
- **Software Suite and Cross-Platform GUI:** The autonomous desktop and field application layer is provided in the repository:
  - `app/server.py`: Multi-threaded local engine with automatic heartbeat watchdog, LAN hotspot support, and cross-platform browser integration.
  - `app/index.html`: Zero-dependency, responsive dark-mode GUI featuring interactive bounding-box overlays, taxon distribution tables, and JSON telemetry export.
  - `zar_manah_diatom_tdi.pyz` (`Diatomeas.command`): 100% autonomous, zero-dependency single-file standalone engine running on any device (macOS, Windows, Linux, Android/Termux) embedding the neural weights, web UI, TDI tables, and local engine without external dependencies.
  - `Diatomeas.app`, `Diatomeas.sh`, `Diatomeas.bat`: Native launchers for macOS, Linux, and Windows.
  - `app/icon.svg`, `app/icon.png`, `diatomeas-icon.png`: vector and raster icon assets used by the GUI, the native bundle, and this README.
- **Model Checkpoints:**
  - Original baseline model: `checkpoints/f2/best.pt` (SHA-256: `8d4fd7c61b6a1e11f329847db902830bee8623862f53d7a521ebe6943f7676c1`).
  - Surgically optimized model (champion): `checkpoints/f2_edited/best_edited.pt` (SHA-256: `0557c7655159532f597ceb14c530f8c767a8b717bd80026a8f042b7d87eedd94`).
  - Mineral detritus SVD noise signature vector: `checkpoints/f2_edited/detritus_vector.pt`.
- **Benchmark Data:** The *UDE Diatoms in the Wild 2024* dataset is openly available via GigaDB and Zenodo (DOI: [10.5281/zenodo.10410655](https://doi.org/10.5281/zenodo.10410655)). The split used for the reported numbers is `state/split_f1.json`, hash `b6a0cb94aab2dd544e88de5d9c0f1c81a78aa59217c3081baf125943874f898d`. It is **not** sealed with respect to micrograph identity: 45.61% of its test cutouts are crops of a photograph also present in train. The group-clean alternative is `state/split_grouped.json` (hash `31d7ad2280c818b4`; 0 shared slides, 7,482 test cutouts), released for retraining.

---

## References

1. Dudgeon, D., Arthington, A. H., Gessner, M. O., Kawabata, Z. I., Knowler, D. J., Lévêque, C., ... & Sullivan, C. A. (2006). Freshwater biodiversity: importance, threats, status and conservation challenges. *Biological Reviews*, 81(2), 163-182. doi:10.1017/S1464793105006950
2. Reid, A. J., Carlson, A. K., Creed, I. F., Eliason, E. J., Gell, P. A., Johnson, P. T., ... & Cooke, S. J. (2019). Emerging threats and persistent conservation challenges for freshwater biodiversity. *Biological Reviews*, 94(3), 849-873. doi:10.1111/brv.12480
3. European Commission. (2000). Directive 2000/60/EC of the European Parliament and of the Council of 23 October 2000 establishing a framework for Community action in the field of water policy. *Official Journal of the European Communities*, L 327, 1-73.
4. Poikane, S., Kelly, M., Fufezan, C., & Borja, A. (2020). Intercalibration of biological assessment methods: A review of the European experience. *Science of The Total Environment*, 712, 136431. doi:10.1016/j.scitotenv.2019.136431
5. Round, F. E., Crawford, R. M., & Mann, D. G. (1991). *The Diatoms: Biology and Morphology of the Genera*. Cambridge University Press.
6. Kelly, M. G., & Whitton, B. A. (1995). The Trophic Diatom Index: a new index for monitoring eutrophication in rivers. *Journal of Applied Phycology*, 7(4), 433-444. doi:10.1007/BF00003802
7. Smol, J. P., & Stoermer, E. F. (Eds.). (2010). *The Diatoms: Applications for the Environmental and Earth Sciences*. Cambridge University Press.
8. Kelly, M., Juggins, S., Guthrie, R., Pritchard, S., Jamieson, J., Rippey, B., ... & Yallop, M. (2008). Assessment of eutrophication in UK rivers using diatoms. *Freshwater Biology*, 53(2), 403-422. doi:10.1111/j.1365-2427.2007.01903.x
9. European Committee for Standardization (CEN). (2014). *Water quality - Guidance for the routine sampling and preparation of benthic diatoms from rivers and lakes (EN 13946:2014)*. Brussels, Belgium.
10. Kahlert, M., Albert, R. L., Anttila, E. L., Bengtsson, R., Bigler, C., Eskola, T., ... & Yallop, M. (2012). Harmonization is more than taxonomical calibration: results from a Nordic-Baltic diatom ring test. *Diatom Research*, 27(4), 237-246. doi:10.1080/0269249X.2012.720235
11. Hopkins, G. W., & Freckleton, R. P. (2002). Declines in the numbers of amateur and professional taxonomists: implications for conservation. *Animal Conservation*, 5(3), 245-249. doi:10.1017/S1367943002002299
12. du Buf, J. M., & Bayer, M. M. (Eds.). (2005). *Automatic Diatom Identification* (Vol. 51). World Scientific. doi:10.1142/5759
13. Pedraza, A., Bueno, G., Dennistoun, F., Contreras, A., Ruiz-Santaquiteria, J., & Blanco, S. (2017). Diatom identification and counting: a review of the state of the art. *Hydrobiologia*, 795(1), 1-17. doi:10.1007/s10750-017-3141-8
14. Kloster, M., Langenkämper, D., Zuluaga, M. A., & Beszteri, B. (2020). Deep learning-based diatom taxonomic classification on high-resolution light microscopy images. *Scientific Reports*, 10(1), 1-12. doi:10.1038/s41598-020-71165-w
15. Tan, M., Pradalier, C., Kloster, M., & Beszteri, B. (2022). Automated diatom classification on light microscopy images using convolutional neural networks and transformer backbones. *Ecological Informatics*, 72, 101889. doi:10.1016/j.ecoinf.2022.101889
16. Venkataramanan, A., Kloster, M., Burfeid-Castellanos, A., Dani, M., Mayombo, N. A. S., Vidakovic, D., ... & Beszteri, B. (2024). UDE Diatoms in the Wild: A benchmark dataset for automated diatom classification in real-world environmental settings. *GigaByte*, 2024, gigabyte121. doi:10.46471/gigabyte.121
17. Lecointe, C., Coste, M., & Prygiel, J. (1993). “Omnidia”: software for computing diatom indices and enter/display diatom counts. In *Eighth International Diatom Symposium* (pp. 531-540).
18. Nicolosi Gelis, M. M., Cochero, J., & Gómez, N. (2020). DiaThor: An R package for computing diatom indices and ecological information. *Ecological Indicators*, 110, 105928. doi:10.1016/j.ecolind.2019.105928
19. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *International Conference on Machine Learning* (pp. 6105-6114). PMLR.
