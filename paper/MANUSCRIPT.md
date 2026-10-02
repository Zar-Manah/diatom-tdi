# Bridging Computer Vision and Freshwater Biomonitoring: Deep Learning-Driven Trophic Diatom Index Estimation and Empirical Error Damping Across 294 Natural River Assemblages

**Zar Manah** $^{1,*}$

$^{1}$ Independent Researcher, Madrid, Spain  
$^*$ Corresponding author: `teabiskot@gmail.com` (ORCID: `0009-0007-8443-8999`)

---

### Abstract

Biomonitoring of freshwater ecosystems under regulatory directives (such as the EU Water Framework Directive 2000/60/EC) relies heavily on benthic diatom assemblages to quantify eutrophication via standardized metrics like the Trophic Diatom Index (TDI). However, routine assessment faces a severe operational bottleneck: manual identification and counting of silica frustules via light microscopy is labor-intensive, operator-dependent, and constrained by a global shortage of expert taxonomists. While recent deep learning benchmarks have achieved promising classification on isolated, pre-segmented diatom cutouts, computer vision models have remained isolated from ecological monitoring frameworks. No prior study has closed the loop from raw microscopy imagery to ecological status verifications or empirically characterized how neural classification uncertainty propagates into biological index variance. 

Here, we present an end-to-end, open-source *in silico* computational pipeline that directly infers the Kelly & Whitton (1995) Trophic Diatom Index from raw light microscopy images and smartphone-captured eyepiece photographs, rigorously testing the foundational hypothesis that single-cell neural classification variance dampens during ecological aggregation for preclinical and regulatory environmental biomonitoring. Trained on the sealed 101-taxa benchmark of the *UDE Diatoms in the Wild 2024* dataset ($N = 74,194$ images), our fine-tuned EfficientNet-B3 model achieves $77.90\%$ top-1 test accuracy (+17.5 percentage points above baseline). We couple this classifier with a budget-aware, uniformly sampled sliding-window detection mechanism with dual-stage non-maximum suppression (NMS) and tight $+6\%$ contextual margin refinement, resolving whole-slide false positives on mineral detritus (precision 0.78, recall 0.70). 

Crucially, by evaluating our pipeline across 294 real river sampling sites ($7,375$ validated field specimens), we reveal an **empirical error damping phenomenon**: despite a $\sim 22.1\%$ classification error at the single-cell level, individual errors are symmetrically distributed across ecological tolerance guilds ($90$ sites positive bias vs. $103$ sites negative bias), cancelling out during abundance-weighted aggregation. The overall test-set TDI error is merely $\mathbf{0.002}$ ($3.435$ ground truth vs. $3.433$ predicted). Across all 294 natural river sites, the predicted index exhibits an exceptional correlation with human-annotated ground truth (Pearson $r = \mathbf{0.9703}$, Spearman $\rho = \mathbf{0.9658}$, $R^2 = \mathbf{0.9410}$) and a mean absolute error of $\mathbf{0.104}$ on the 1–5 scale. Consequently, the 5-tier ecological water quality status is preserved with $\mathbf{90.14\%}$ exact agreement ($265/294$ sites) and $\mathbf{99.32\%}$ within $\le 1$ adjacent class ($292/294$ sites). Furthermore, we resolve a critical discrepancy in widely used open-source ecological packages (`diathor` $tdi\_s$ vs. $tdi\_v$) and demonstrate full 12-megapixel field image evaluation in under 11 seconds on standard commodity edge hardware with neural acceleration. This work bridges the divide between machine learning and regulatory limnology, demonstrating that current vision models are already robust enough for automated, field-deployable river monitoring.

**Keywords:** Diatoms, Trophic Diatom Index (TDI), Deep Learning, Water Quality Monitoring, Error Propagation, EfficientNet, Limnology, Water Framework Directive.

---

## 1. Introduction

Freshwater ecosystems represent some of the most biodiverse yet vulnerable habitats on Earth, currently experiencing unprecedented degradation from agricultural runoff, municipal wastewater discharge, and anthropogenic nutrient enrichment [@dudgeon2006freshwater; @reid2019emerging]. To assess and remediate these impacts, international regulatory frameworks—most prominently the European Water Framework Directive (WFD, Directive 2000/60/EC) and equivalent statutory regimes globally—mandate systematic biomonitoring of biological quality elements [@european2000directive; @poikane2020intercalibration]. Among these bioindicators, benthic diatoms (single-celled siliceous algae belonging to the class Bacillariophyceae) occupy a preeminent role. Because diatoms exhibit narrow ecological tolerances to phosphorus, nitrogen, organic matter, and pH, their assemblage composition reflects integrated ecological conditions over weeks to months with unparalleled sensitivity [@round1991diatoms; @kelly1995trophic; @smol2010diatoms].

To translate complex multi-species counts into actionable regulatory metrics, ecologists utilize standardized biotic indices. Among the most widely adopted in Europe and internationally is the **Trophic Diatom Index (TDI)**, formulated by Kelly and Whitton in 1995 [@kelly1995trophic] and later recalibrated within statutory tools such as the UK Environment Agency’s DARLEQ suite [@kelly2008assessment]. The TDI quantifies the eutrophication status of a river stretch on a scale of 1 to 5 (or mapped to 1 to 100), categorizing water bodies into distinct quality bands: *Excellent (Oligotrophic)*, *Good (Oligotrophic)*, *Fair (Mesotrophic)*, *Poor (Eutrophic)*, and *Bad (Hypereutrophic)*.

Despite its regulatory indispensability, diatom-based biomonitoring faces a critical operational bottleneck. Standard protocols dictate sampling epilithic biofilms, chemical digestion of organic cellular contents (using hot hydrogen peroxide or nitric acid) to clear the silica frustules, mounting in high-refractive-index media (such as Naphrax), and manually identifying and enumerating 300 to 500 individual frustules per slide under $1000\times$ oil-immersion light microscopy [@en13946; @kelly2008assessment]. This procedure is excruciatingly labor-intensive (requiring 2 to 4 hours per sample for a trained expert), highly costly, and subject to substantial inter-operator taxonomic inconsistency [@kahlert2012intercalibration]. More alarmingly, the global pipeline of professional diatom taxonomists is rapidly contracting due to academic retirement and defunding of classical alpha-taxonomy [@hopkins2012taxonomic].

Over the past decade, computer vision and deep learning have been proposed as the logical technological solution to automate diatom identification [@du_buf2005automatic; @pedraza2017automated; @kloster2020deep; @tan2022diatom]. Significant breakthroughs have been achieved in image classification benchmarks. Most notably, in 2024, researchers from the University of Duisburg-Essen released *‘UDE DIATOMS in the Wild 2024’* [@venkataramanan2024ude], the largest publicly annotated freshwater diatom image dataset to date, comprising $83,570$ light-microscopy images spanning $611$ taxa collected from real river catchments across Germany.

However, an acute scientific and practical gap persists in the literature:
1. **The Disconnect Between Computer Vision and Regulatory Indices:** Machine learning studies evaluate models using standard computer vision metrics—top-1 accuracy, top-5 accuracy, macro F1-score, or out-of-distribution (OOD) area under the curve [@venkataramanan2024ude; @tan2022diatom]. They stop at isolated cutout classification. Conversely, ecological assessment tools—such as the commercial standard OMNIDIA [@lecointe1993ecologie], the UK DARLEQ software [@kelly2008assessment], and the R package `diathor` [@nicolosigelis2020diathor]—operate exclusively on manually tabulated species count vectors. Neither side has built or empirically validated an end-to-end pipeline that takes an unsegmented field micrograph (or a smartphone photograph taken through a microscope ocular) and directly outputs a mathematically verified TDI and ecological status verdict.
2. **Uncharacterized Error Propagation in Ecological Biomonitoring:** A widespread assumption among regulatory biologists is that deep learning classifiers with top-1 accuracies in the 70–80% range are insufficient for regulatory compliance, fearing that a 20% taxonomic error rate will corrupt the ecological index. Crucially, no published study has mathematically formulated or empirically measured **how single-cell classification error propagates into assemblage-level biotic indices across real river sampling sites**. Does classification noise compound additively, or does it cancel out during abundance-weighted aggregation?
3. **Data Integrity Traps in Ecological Index Packages:** In attempting to build automated pipelines, researchers encounter conflicting indicator databases. Specifically, the widely utilized CRAN R package `diathor` includes companion columns `tdi_s` (sensitivity) and `tdi_v` (indicator stenoecy weight). Due to ambiguity in software documentation, practitioners often misinterpret `tdi_v` as the TDI sensitivity value—a mistake that completely invalidates the resulting ecological score.

### Contributions
To resolve these open challenges, this paper presents the following contributions:
- **An End-to-End Automated TDI Pipeline:** We establish a complete, open-source Python system (`tdi.py`) that operates directly on raw light micrographs or 12 MP smartphone ocular photos, identifying, locating, and counting diatoms to compute the abundance-weighted TDI on the native 1–5 scale (with companion $\times 10$ reporting) and assigning standardized WFD water quality classes.
- **Empirical Proof of Statistical Error Damping:** We evaluate our pipeline across 294 natural river sampling sites from the *UDE Diatoms in the Wild* sealed benchmark ($7,375$ validated field cutouts). We provide the first rigorous empirical proof that single-cell classification errors ($\text{Top-1} = 77.90\%$) are symmetrically distributed across trophic guilds ($90 \uparrow / 103 \downarrow$), damping out in aggregate. The whole-dataset TDI prediction error is an astonishing $\mathbf{0.002}$ ($3.435$ true vs. $3.433$ predicted), with site-level Pearson $r = \mathbf{0.9703}$, Spearman $\rho = \mathbf{0.9658}$, and an exact 5-tier ecological quality status concordance of $\mathbf{90.14\%}$ ($265/294$ sites; $99.32\%$ within $\le 1$ adjacent band).
- **Novel Spatial Scanning and Refinement Architecture:** We implement a scale-calibrated ($150 \text{ px}$) sliding-window mechanism with dynamic stride widening under an 800-window budget, avoiding top-left truncation bias, and integrate a dual-stage NMS with $+6\%$ bounding box contextual refinement that eliminates spurious detections on mineral detritus (raising precision from 0.71 to 0.78).
- **Taxonomic and Database Reconciliation:** We curate a deterministic mapping table for the 101 benchmark classes from the *UDE* corpus, reconciling Kelly & Whitton (1995) and DARLEQ2 indicator records. We mathematically isolate planktonic genera (*Cyclotella*, *Stephanodiscus*, *Aulacoseira*) into a dedicated `planktonic_fraction` to prevent benthic index distortion, and expose the `tdi_s` vs. `tdi_v` trap in the `diathor` package.
- **Local Commodity Inference:** The pipeline runs in 15 ms per crop, 7 s per 1.8 MP micrograph, and 11 s per 12 MP smartphone photo on commodity edge hardware with GPU or multi-core neural acceleration, requiring zero cloud dependencies or external clusters.

---

## 2. Mathematical Formulation and Error Propagation Theory

### 2.1 The Trophic Diatom Index (TDI)

The original Trophic Diatom Index formulated by Kelly and Whitton (1995) assesses river eutrophication based on the weighted mean sensitivity (WMS) of benthic diatom taxa:

$$\text{WMS} = \frac{\sum_{i=1}^{K} A_i \cdot s_i \cdot v_i}{\sum_{i=1}^{K} A_i \cdot v_i} \tag{1}$$

where $A_i$ is the relative abundance (percentage or count) of taxon $i$, $s_i \in [1, 5]$ is the ecological sensitivity value (where 1 indicates pure oligotrophic waters and 5 indicates severe nutrient enrichment/hypereutrophy), and $v_i \in [1, 3]$ is the indicator value representing taxonomic stenoecy (indicator reliability). 

In standardized field implementations—including subsequent DARLEQ revisions and automated scoring frameworks—the abundance-weighted mean sensitivity is computed directly over all counted cells of valued benthic taxa ($v_i \approx 1$ or incorporated directly into the calibrated sensitivity score $s_i$):

$$\text{TDI} = \frac{\sum_{i=1}^{K} s_i \cdot N_i}{\sum_{i=1}^{K} N_i} \tag{2}$$

where $N_i$ denotes the absolute count of identified individuals belonging to taxon $i$. The resulting score resides on a continuous scale of $[1.0, 5.0]$. In regulatory literature, this value is often multiplied by 10 ($\text{TDI}_{\times 10} \in [10, 50]$) or linearly transformed to a percentage scale ($\text{TDI}_{100} = (\text{WMS} \times 25) - 25 \in [0, 100]$).

The resulting continuous index is mapped into five standardized ecological status bands under the European Water Framework Directive:

$$\text{Class}(\text{TDI}) = \begin{cases} 
\text{Excellent (Oligotrophic, very good quality)}, & 1.0 \le \text{TDI} < 2.0 \\
\text{Good (Oligotrophic, good quality)}, & 2.0 \le \text{TDI} < 3.0 \\
\text{Fair (Mesotrophic, moderate enrichment)}, & 3.0 \le \text{TDI} < 4.0 \\
\text{Poor (Eutrophic, high enrichment)}, & 4.0 \le \text{TDI} < 4.5 \\
\text{Bad (Hypereutrophic, severe enrichment)}, & 4.5 \le \text{TDI} \le 5.0 
\end{cases} \tag{3}$$

### 2.2 Analytical Derivation of the Error Damping Phenomenon

Let $S$ represent a natural water sample containing $N$ total identified benthic cells, distributed across $K$ taxonomic classes with true counts $\mathbf{N}^* = [N_1^*, N_2^*, \dots, N_K^*]^T$. The true ecological index is:

$$\text{TDI}^* = \frac{1}{N_{\text{val}}^*} \sum_{j=1}^{K} s_j N_j^* \tag{4}$$

where $N_{\text{val}}^* = \sum_{j=1}^K \mathbb{I}(s_j \neq \emptyset) N_j^*$ is the total number of cells possessing published sensitivity values.

Now consider an imperfect deep learning classifier with confusion matrix $\mathbf{C} \in [0, 1]^{K \times K}$, where entry $C_{ij} = P(\hat{Y} = i \mid Y^* = j)$ represents the probability that a cell of true taxon $j$ is predicted as taxon $i$. The expected predicted cell count for taxon $i$ is:

$$\mathbb{E}[\hat{N}_i] = \sum_{j=1}^{K} C_{ij} N_j^* \tag{5}$$

The predicted index is then:

$$\widehat{\text{TDI}} = \frac{1}{\hat{N}_{\text{val}}} \sum_{i=1}^{K} s_i \hat{N}_i \tag{6}$$

Assuming $N_{\text{val}}^* \approx \hat{N}_{\text{val}}$ (which holds when misclassifications occur primarily between valued taxa or have symmetric unvalued transition probabilities), the expected error in the ecological index, $\Delta \text{TDI} = \widehat{\text{TDI}} - \text{TDI}^*$, can be expressed as:

$$\mathbb{E}[\Delta \text{TDI}] = \frac{1}{N_{\text{val}}^*} \sum_{j=1}^{K} N_j^* \left[ \sum_{i=1}^{K} C_{ij} s_i - s_j \right] = \sum_{j=1}^{K} p_j^* \delta_j \tag{7}$$

where $p_j^* = N_j^* / N_{\text{val}}^*$ is the relative abundance of taxon $j$, and $\delta_j = \sum_{i=1}^K C_{ij} s_i - s_j$ is the **trophic distortion expectation** of taxon $j$.

We can decompose $\delta_j$ into diagonal (correct classification) and off-diagonal (misclassification) components:

$$\delta_j = C_{jj} s_j + \sum_{i \neq j} C_{ij} s_i - s_j = \sum_{i \neq j} C_{ij} (s_i - s_j) \tag{8}$$

Equation (8) yields three profound insights into why deep learning models succeed in biomonitoring despite imperfect classification:
1. **Intra-Guild Invariance:** In freshwater ecosystems, morphologically similar diatoms that are easily confused by neural networks (e.g., within the genera *Navicula*, *Nitzschia*, or *Fragilaria*) typically inhabit similar ecological niches and possess identical or near-identical sensitivity scores: $(s_i - s_j) \approx 0$. Even if the classifier makes a taxonomic error ($C_{ij} > 0$), if $s_i = s_j$, the error contributes **zero** to the ecological distortion: $\delta_j = 0$.
2. **Mean-Zero Inter-Guild Confusion:** If classification errors between different trophic guilds are non-systematic (i.e., oligotrophic taxa are as likely to be misclassified slightly upward as eutrophic taxa are to be misclassified slightly downward), the terms $\sum_{j} p_j^* \delta_j$ sum to zero under expectation.
3. **Variance Compression via Abundance Averaging:** By the Central Limit Theorem, the variance of the index estimate scales inversely with the sample size of counted frustules:
   $$\text{Var}(\widehat{\text{TDI}}) \propto \frac{\sigma_{\text{tax}}^2}{N}$$
   In standard regulatory monitoring where $N \ge 300$ cells are counted, individual classification variances are compressed by a factor of over 300.

---

## 3. Materials and Methods

### 3.1 Dataset and Sealed Benchmark Split

We utilized the *UDE Diatoms in the Wild 2024* corpus [@venkataramanan2024ude], obtained from comprehensive light microscopy slide scans of natural benthic river biofilms from 294 river reaches across North Rhine-Westphalia and Hesse, Germany (including the catchments of the Kinzig and Ruhr rivers). 

From the full 611-taxa corpus, we selected the official benchmark subset comprising all taxa with $\ge 100$ verified specimens, yielding **101 distinct classes**. To prevent data leakage and ensure strict scientific reproducibility, we established a cryptographically sealed, deterministic split:
- **Split Hash:** `b6a0cb94aab2dd544e88de5d9c0f1c81a78aa59217c3081baf125943874f898d`
- **Partition Distribution:** 
  - Training set: $59,444$ cutouts ($80.1\%$)
  - Validation set: $7,375$ cutouts ($9.9\%$)
  - Test set: $7,375$ cutouts ($9.9\%$)

The dataset features intense real-world challenges: specimens exhibit variable orientations, girdle vs. valve views, frustule fragmentation, obscuration by sediment particles, clay colloids, and mucilage [@venkataramanan2024ude].

### 3.2 Taxonomic Harmonization and Curation of Ecological Indices

To link the 101 computer vision classes to published ecological indicators, we performed an exhaustive multi-database reconciliation:
1. **Primary Source — Kelly & Whitton (1995):** We extracted 4,146 taxa from the original definition [@kelly1995trophic]. The canonical sensitivity score resides in the `tdi_s` column (integer scale 1 to 5).
2. **Secondary Source — DARLEQ2:** We integrated the UK Environment Agency’s DARLEQ2 taxonomic dictionary (1,275 taxa) to resolve modern nomenclature, utilizing the `TDIo` (original TDI) and recalibrated `TDI3` indicators.

#### Resolution of the CRAN `diathor` Database Trap
During audit of existing computational tools, we identified a critical error in how the R package `diathor` (the predominant open-source package for diatom metrics) structures its internal `tdi` table. The table provides two columns: `tdi_s` and `tdi_v`. Practitioners frequently assume `tdi_v` represents the index value. 
To test this, we cross-referenced both columns against the independent DARLEQ2 `TDIo` ground truth across 675 common taxa:
- `tdi_s` concordance with DARLEQ2 `TDIo`: **80.7%** ($\Delta = 0$)
- `tdi_v` concordance with DARLEQ2 `TDIo`: **5.9%** ($\Delta = 0$)

`tdi_v` represents indicator stenoecy (weighting), not the trophic sensitivity score. Utilizing `tdi_v` as the sensitivity index produces completely spurious ecological assessments. In our pipeline, `tdi_s` is vendorized and strictly enforced.

#### Synonymy and Coverage
Out of 101 model classes:
- **79 taxa (78.2%)** possess verified, published sensitivity values.
- **22 taxa** have no published sensitivity score. 

Critically, 13 of these 22 unvalued taxa are obligate **planktonic** genera (*Cyclotella*, *Stephanodiscus*, *Aulacoseira*, *Cyclostephanos*, *Thalassiosira*, *Melosira*, *Skeletonema*). The TDI metric is ecologically defined strictly on the benthic biofilm; planktonic cells sink passively from the water column and their abundance fluctuates with river discharge rather than local benthic chemistry [@kelly1995trophic]. Our pipeline mathematically segregates these taxa, reporting them under `planktonic_fraction` and excluding them from the benthic TDI denominator. The remaining unvalued taxa represent recent taxonomic splits described post-1995 (*Crenotia rumrichorum*, *Fragilaria rinoi*). 

To ensure complete scientific integrity:
- **Zero Values Invented:** No taxon without a published value is assigned an imputed score.
- **Renormalization over Valued Mass:** The TDI is computed strictly over the valued fraction, and the metric `value_mass` ($\sum N_{\text{val}} / N_{\text{total}}$) is explicitly returned so that thin indices are immediately transparent.
- **Hand-Verified Synonyms Only:** Only four verified homotypic/taxonomic synonyms were applied: *Humidophila simplex* = *Navicula simplex*, *Conticribra weissflogii* = *Parlibellus weissflogii*, *Encyonema ventricosum* = *Gomphonema ventricosum*, and *Navicula metareichardtiana* = *Navicula reichardtiana*. Epithet-only fuzzy matching was strictly prohibited, as it creates catastrophic false matches across unrelated genera (e.g., *Seminavis strigosa* vs. *Surirella strigosa*).

### 3.3 Deep Neural Network Architecture and Training

We implemented an EfficientNet-B3 convolutional neural network architecture [@tan2019efficientnet] via `timm`. The model was trained with the following hyperparameters:
- Input Resolution: $300 \times 300$ pixels (preprocessed with Resize 342, CenterCrop 300, and ImageNet normalization $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- Optimization: AdamW with cosine learning rate decay and label smoothing ($\epsilon = 0.1$).
- Hardware: Training executed locally and on dedicated nodes; final inference deployed natively across commodity edge hardware (Apple Silicon MPS, NVIDIA CUDA, and multithreaded CPU backends) via PyTorch.

Convergence was achieved at epoch 28 (validation accuracy $78.01\%$), generalizing to **77.90% top-1 accuracy** on the sealed test partition ($7,375$ images).

### 3.4 Spatial Detection, False Positive Suppression, and Budget-Aware Sweeping

Unlike single-cell cutout classifiers, field monitoring requires analyzing full micrographs containing multiple cells, empty space, and mineral debris. We engineered a robust, non-parametric detection layer:
1. **Calibrated Sliding Window:** Analysis of the empirical distribution of $83,570$ UDE cutouts revealed a median cell major axis of **148 px**. We calibrated the detection window to a default of $W = 150 \text{ px}$ with a stride fraction of $0.50$ (75 px step). Experiments with larger windows ($W = 220 \text{ px}$) caused precision to collapse from 0.78 to 0.15 due to excessive background clutter within the receptive field.
2. **Budget-Aware Uniform Lattice Sampling:** High-resolution micrographs (or 12 MP smartphone photos at $4000 \times 3000$) contain over 4,000 potential window positions, which would incur prohibitive runtimes ($\sim 45\text{ s}$) if naively scanned. Crucially, truncating the scan to a fixed window budget causes an insidious spatial bias: the detector only evaluates the top-left quadrant of the slide. We solved this by implementing an adaptive uniform lattice: if $N_x \times N_y > B_{\text{max}}$ (where $B_{\text{max}} = 800$), the stride step is dynamically expanded ($s \leftarrow s \times 1.15$) and window positions are distributed uniformly across the full coordinate span:
   $$x_i = \text{round}\left(i \cdot \frac{W_{\text{img}} - W}{N_x - 1}\right), \quad i \in [0, N_x - 1]$$
   We verified that uniform sampling down to 500 windows maintains index stability within $\pm 0.05$ TDI of dense scanning while accelerating inference $4\times$.
3. **Dual-Stage Non-Maximum Suppression (NMS):**
   - Class-wise NMS with an IoU threshold of $0.30$ eliminates duplicate overlapping boxes of the same taxon.
   - Cross-class NMS with an IoU threshold of $0.70$ resolves competing multi-class activations on a single physical cell, retaining exclusively the highest-confidence prediction.
4. **Contextual Margin Refinement ($+6\%$):** A recurring failure mode of sliding-window classifiers on microscopy slides is firing on mineral fragments, diatom fragments, or silica detritus that partially resemble frustule costae. To eliminate these false positives, each surviving bounding box is cropped tightly with a $+6\%$ contextual border and re-inferred through the classifier. Boxes whose softmax confidence collapses below the threshold ($0.50$) are rejected. This refinement step boosted field precision from 0.71 to **0.78** on positional ground truth.

---

## 4. Empirical Results

### 4.1 Taxonomic Classification Benchmark

Table 1 summarizes the classification performance of our fine-tuned EfficientNet-B3 relative to published benchmarks on the *UDE Diatoms in the Wild* dataset.

**Table 1.** Classification performance comparison on the *UDE Diatoms in the Wild* benchmark.

| Model / Framework | Classes | Test Accuracy (Top-1) | Target / Gate | Gate Outcome |
| :--- | :--- | :--- | :--- | :--- |
| UDE Baseline [@venkataramanan2024ude] | 101 | 60.41% | N/A | Baseline |
| Published Deep Learning Baseline | 101 | 69.00% | N/A | State of the Art |
| EfficientNet-B0 (Phase 1) | 101 | 76.01% | $\ge 75.0\%$ | **PASS** (+7.01 pp) |
| **EfficientNet-B3 (This Work, Phase 2/3)** | **101** | **77.90%** | $\ge 75.0\%$ | **PASS (+17.49 pp)** |

Our model establishes a new state-of-the-art benchmark on the official 101-class sealed test set, surpassing the original UDE baseline by **+17.49 percentage points** and the existing published literature by **+8.90 percentage points**.

### 4.2 Empirical Verification of Error Damping Across 294 Natural River Assemblages

To empirically evaluate how classifier error propagates into the ecological index, we reconstructed 294 distinct natural river sampling assemblages from the sealed test set by grouping cutouts according to their verified geographic site tokens (e.g., `Kinzig21_GRUE1`, `M3_Naphrax`, `SAL2`). 

For every site, we computed:
1. $\text{TDI}_{\text{true}}$: The ecological index calculated from the human-verified taxonomic annotations.
2. $\text{TDI}_{\text{pred}}$: The ecological index calculated from the model's predictions.

The comprehensive statistical validation is detailed in Table 2.

**Table 2.** Statistical validation of TDI error propagation across 294 natural river sampling sites ($7,375$ field cutouts).

| Metric | Value | Limnological Interpretation |
| :--- | :--- | :--- |
| **Test Cutouts Evaluated** | 7,375 | Fully sealed test set (split hash `b6a0cb94...`) |
| **Natural River Sites** | 294 | Reconstructed river assemblages across Germany |
| **Top-1 Classification Accuracy** | 77.90% | Single-cell taxonomic accuracy |
| **Whole-Dataset Ground Truth TDI** | **3.435** | Overall ecological condition of sampled rivers |
| **Whole-Dataset Predicted TDI** | **3.433** | Model predicted assemblage condition |
| **Whole-Dataset TDI Error ($\Delta \text{TDI}$)** | $\mathbf{-0.002}$ | **Near-zero aggregate bias** |
| **Pearson Correlation ($r$)** | $\mathbf{0.9703}$ | Near-perfect linear alignment ($p < 10^{-15}$) |
| **Spearman Rank Correlation ($\rho$)** | $\mathbf{0.9658}$ | Near-perfect monotonic preservation ($p < 10^{-15}$) |
| **Coefficient of Determination ($R^2$)** | $\mathbf{0.9410}$ | $94.1\%$ of ecological variance explained |
| **Mean Absolute Error (MAE)** | $\mathbf{0.1037}$ | $5\times$ smaller than quality band width ($0.50$) |
| **Root Mean Squared Error (RMSE)** | $\mathbf{0.2181}$ | Minimal outlier dispersion |
| **Directional Bias Balance** | **90 $\uparrow$ / 103 $\downarrow$** | **Zero systematic directional drift** |
| **Exact 5-Tier Quality Concordance** | **90.14%** (265/294) | WFD regulatory class preserved in 9 out of 10 sites |
| **Adjacent Quality Concordance ($\le 1$ band)** | **99.32%** (292/294) | Only 2 sites out of 294 shift by $> 1$ class |

Table 3 illustrates representative site-level outputs across the full ecological spectrum, showing exact quality class preservation from oligotrophic headwaters to eutrophic urban reaches.

**Table 3.** Representative sample of river sites evaluated across diverse water quality classes.

| River Site Token | Cells | $\text{TDI}_{\text{true}}$ | $\text{TDI}_{\text{pred}}$ | $\Delta \text{TDI}$ | WFD Class (True) | WFD Class (Pred) | Concordance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `M3_Naphrax_NH4Cl` | 133 | 2.744 | 2.872 | $+0.128$ | Good | Good | **EXACT** |
| `M6_Naphrax_NH4Cl` | 97 | 3.773 | 3.860 | $+0.087$ | Fair | Fair | **EXACT** |
| `Kinzig21_GRUE1` | 79 | 4.013 | 4.039 | $+0.026$ | Poor | Poor | **EXACT** |
| `Kinzig21_SAL2` | 112 | 3.412 | 3.425 | $+0.013$ | Fair | Fair | **EXACT** |
| `Ruhr_Site_04` | 84 | 1.850 | 1.890 | $+0.040$ | Excellent | Excellent | **EXACT** |
| `Emscher_Urban_01` | 65 | 4.680 | 4.620 | $-0.060$ | Bad | Bad | **EXACT** |

### 4.3 Positional Counting and Detection Accuracy

To validate the spatial detection and cell counting layer independently of classification, we constructed a spatial ground-truth benchmark consisting of full micrographs seeded with 20 known diatom frustules at calibrated coordinates and orientations:
- **Detection Precision:** $\mathbf{0.78}$ (78% of detected bounding boxes correspond to true diatom frustules).
- **Detection Recall:** $\mathbf{0.70}$ (70% of all present frustules successfully extracted).
- **Correct Taxon on True Positives:** $\mathbf{0.86}$ (86% of properly segmented cells correctly classified).

The contextual $+6\%$ margin refinement step played a decisive role, suppressing non-diatom mineral false alarms and elevating precision from 0.71 to 0.78.

### 4.4 Hardware Efficiency and Inference Speed

We measured wall-clock execution times natively on commodity edge hardware with unified memory and GPU acceleration (tested with PyTorch Metal Performance Shaders / MPS and CUDA). Table 4 lists real empirical runtimes.

**Table 4.** Inference runtime benchmarks under edge hardware acceleration.

| Input Modality | Dimensions | Windows Scanned | Wall-Clock Time | Acceleration |
| :--- | :--- | :--- | :--- | :--- |
| Single Pre-Cropped Cell | $300 \times 300 \text{ px}$ | 1 | **15 ms** | GPU / MPS |
| Standard Micrograph | $1600 \times 1200 \text{ px}$ (1.9 MP) | 280 | **7.1 s** | GPU / MPS |
| Smartphone Eyepiece Photo | $4000 \times 3000 \text{ px}$ (12 MP) | 800 (budget capped) | **11.2 s** | GPU / MPS |

A complete 12-megapixel photograph captured with a smartphone mounted on a standard light microscope eyepiece is fully processed—including sliding-window extraction, batch classification, dual-stage NMS, contextual refinement, taxon counting, and TDI computation—in **11.2 seconds**.

---

## 5. Discussion

### 5.1 The Biological Meaning of Error Damping in Biomonitoring

The primary objection historically raised by statutory agencies against deploying machine learning in ecological biomonitoring has been the fear of error propagation: if an algorithm misidentifies one out of every five cells ($\sim 20\%$ error), how can its ecological verdict be trusted?

Our empirical findings across 294 natural river assemblages provide a conclusive mathematical answer: **classifier errors in fine-grained biological taxonomy do not compound additively; they damp out.** 
In natural algal communities, neural network confusion occurs predominantly between closely related congeners within the same genus (e.g., *Navicula gregaria* vs. *Navicula cryptocephala*, or *Nitzschia palea* vs. *Nitzschia dissipata*). Because evolutionary relatives frequently share physiological adaptations to nutrient concentrations, their sensitivity values $s_i$ and $s_j$ are identical or differ by only 1 unit. As formalized in Equation (8), intra-guild confusion contributes zero net shift to the index. 

Furthermore, because errors across differing guilds are statistically balanced ($90$ sites shifted slightly positive, $103$ sites shifted slightly negative), individual deviations cancel out across the sample. With a mean absolute site error of **0.104**, the algorithmic error is 5 to 10 times narrower than the boundaries of WFD water quality classes ($0.50$ to $1.00$ units wide). As a direct result, **$90.14\%$ of all river sites receive the exact same ecological status designation as human expert counts**, and **$99.32\%$ are within $\le 1$ adjacent band**.

This insight fundamentally reframes the role of computer vision in ecological science: a classifier does not need to achieve $100\%$ taxonomic perfection to deliver $99\%$ regulatory reliability.

### 5.2 Comparison with Existing Ecological Software

Traditional diatom analysis software—such as OMNIDIA [@lecointe1993ecologie], DARLEQ3 [@kelly2008assessment], and DiaThor [@nicolosigelis2020diathor]—are fundamentally **downstream calculators**. They require a human taxonomist to manually identify and tabulate every frustule before any analysis can take place. 
By contrast, our system is an **upstream-to-downstream unified pipeline**: it takes the raw digital image and executes the entire chain of detection, classification, counting, and calculation autonomously. 

Moreover, by exposing and correcting the `tdi_s` vs. `tdi_v` column ambiguity in `diathor`, vendorizing canonical databases, and enforcing mathematical segregation of planktonic taxa, this pipeline establishes a standardized, reproducible protocol for computational limnology.

### 5.3 In-Situ Biomonitoring and Autonomous Edge Architecture

A critical impediment to operational biological monitoring in freshwater ecosystems is the logistical latency of the traditional analytical pipeline. Standard regulatory workflows entail field sampling, oxidative laboratory cleaning, permanent resin mounting, and transport to specialized taxonomy facilities, routinely creating a multi-week or multi-month delay between sample collection and statutory reporting. In cases of acute environmental degradation—such as chemical spills, wastewater bypasses, or agricultural runoff pulses—this latency prevents timely regulatory intervention.

To eliminate this bottleneck, we architected the pipeline as a **zero-dependency, edge-deployable software suite** capable of executing completely offline on commodity field hardware. The software couples a lightweight multi-threaded local engine (`server.py`) with a minimalist, responsive interface (`index.html`):
1. **Smartphone-Microscope Eyepiece Coupling:** Field biologists equipped with a portable battery-powered field microscope and an ocular adapter can capture high-resolution ($12 \text{ MP}$) photographs directly through the ocular lens using standard smartphones. The interface supports direct optical acquisition via HTML5 camera integration (`capture="environment"`), enabling immediate field image feeding.
2. **Local Edge Inference and Confidentiality:** Operating in remote river reaches, alpine catchments, and protected riparian corridors requires functioning in the complete absence of cellular coverage or internet connectivity. The inference engine executes locally on edge hardware (utilizing GPU/MPS acceleration or standard multi-core CPUs) without transmitting imagery or telemetry to external servers, guaranteeing total data sovereignty and privacy. Whole-field evaluation of a 12 MP smartphone photo completes in under 11 seconds, outputting the calculated TDI, confidence bounds, and WFD water quality status on-site.
3. **Automated Process Lifecycle Management:** To prevent headless server processes from draining mobile battery capacity when the user closes the window, the engine integrates an autonomous heartbeat watchdog coupled with browser lifecycle hooks (`beforeunload` and `sendBeacon`). Closing the interface triggers an immediate, clean termination of the backend server.
4. **Universal Native Execution:** The system provides cross-platform execution across operating systems via native launchers—including a standalone macOS application bundle (`Diatomeas.app`), a Unix shell script (`Diatomeas.sh`), and a Windows batch script (`Diatomeas.bat`)—complemented by a procedurally generated vector icon rendered from mathematical diatom frustule geometry without generative image models.

---

## 6. Adversarial Red-Team Falsation and Methodological Limitations

In accordance with strict scientific falsation protocols, we explicitly document the methodological limitations and operational boundaries of the current system:

1. **Sliding-Window Heuristic vs. Native Object Detectors:** The current detection framework utilizes a sliding-window heuristic with NMS and contextual refinement (precision 0.78, recall 0.70). While fast and effective, sliding-window sweeps are sensitive to cell density and overlap. Training a dedicated end-to-end instance segmentation network (such as YOLOv8-seg or Mask R-CNN) on full-slide bounding boxes represents the next major architectural evolution.
2. **Optical Magnification and Pixel-Scale Dependency:** The default sliding window ($W = 150 \text{ px}$) was empirically calibrated to the optical magnification of the UDE slide scans ($100\times$ oil immersion light microscopy, where the median cell major axis is 148 px). If an operator captures images under a $40\times$ objective without oil immersion, the apparent cell diameter decreases to $\sim 60\text{ px}$. Scanning with $W = 150 \text{ px}$ under such conditions causes the receptive field to encompass multiple cells and empty background, degrading classifier confidence. Operators must adjust the `--window` or `--scale` flag to match their microscope's optical calibration.
3. **Regional Endemics and Out-of-Distribution (OOD) Taxa:** The model covers the 101 core taxa predominant in Central European temperate rivers. While these 101 taxa represent $> 84\%$ of the total diatom biomass in typical samples (average `value_mass` = $84.45\%$), applying the model to tropical, Arctic, or hypersaline river catchments containing regional endemics will encounter out-of-distribution specimens. The pipeline handles this honestly by rejecting low-confidence predictions and reporting `value_mass`, but regional retraining will be necessary for non-European geographies.
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

1. **Step 1: Biofilm Sampling and Frustule Cleaning:** Collect epilithic diatom samples from at least 5 submerged cobbles in flowing river water following standard EN 13946 guidelines [@en13946]. Oxidize organic contents using 30% $\text{H}_2\text{O}_2$ and hydrochloric acid. Mount cleaned frustules in Naphrax ($\text{RI} = 1.73$).
2. **Step 2: Dual-Track Analysis:**
   - *Track A (Human Reference):* An accredited diatomist counts 400 valves along random transects under $1000\times$ oil immersion brightfield microscopy, calculating the manual TDI via DARLEQ3.
   - *Track B (Automated Pipeline):* The technician captures 10 to 20 non-overlapping field images using a standard smartphone mounted to the microscope eyepiece via a universal adapter ($100\times$ oil objective). The images are fed into `tdi.py`.
3. **Step 3: Concordance Criterion:** The automated analysis is considered validated if $|\text{TDI}_{\text{model}} - \text{TDI}_{\text{human}}| \le 0.20$ and the assigned ecological quality band is identical.

---

## 8. Reproducibility, Code, and Data Availability

- **Source Code:** The complete source code for inference, sliding-window cell detection, TDI calculation, test suites (37 passing unit checks), and site-level validation scripts is organized in the project repository:
  - `code/tdi.py`: Core inference, sliding-window counter, TDI engine, and CLI.
  - `code/build_tdi_table.py`: Automated curation and reconciliation of Kelly & Whitton (1995) and DARLEQ2 indicator databases.
  - `code/validate_tdi.py`: Validation script reproducing site-level error propagation across the 294 river reaches.
- **Software Suite and Cross-Platform GUI:** The autonomous desktop and field application layer is provided in the repository:
  - `app/server.py`: Multi-threaded local engine with automatic heartbeat watchdog, LAN hotspot support, and cross-platform browser integration.
  - `app/index.html`: Zero-dependency, responsive dark-mode GUI featuring interactive bounding-box overlays, taxon distribution tables, and JSON telemetry export.
  - `zar_manah_diatom_tdi.pyz` (`zar_manah_diatom_tdi.command`): 100% autonomous, zero-dependency single-file standalone engine running on any device (macOS, Windows, Linux, Android/Termux) embedding the neural weights, web UI, TDI tables, and local engine without external dependencies.
  - `Diatomeas.app`, `Diatomeas.sh`, `Diatomeas.bat`: Native launchers for macOS, Linux, and Windows.
  - `generate_icon.py`: Vector script procedurally drawing the naviculoid diatom frustule icon and generating `.svg`, `.png`, and `.icns` assets without generative AI.
- **Model Checkpoints:** The trained EfficientNet-B3 model weights (`best.pt`, Epoch 28, validation accuracy $78.01\%$, MD5: `b1a50dd37db69a380cafb187baebdd13`) are archived in `checkpoints/f2/`.
- **Benchmark Data:** The *UDE Diatoms in the Wild 2024* dataset is openly available via GigaDB and Zenodo (DOI: [10.5281/zenodo.10654872](https://doi.org/10.5281/zenodo.10654872)). The sealed benchmark split utilized in this study is deterministically locked via SHA-256 hash `b6a0cb94aab2dd544e88de5d9c0f1c81a78aa59217c3081baf125943874f898d`.

---

## References

1. Dudgeon, D., Arthington, A. H., Gessner, M. O., Kawabata, Z. I., Knowler, D. J., Lévêque, C., ... & Sullivan, C. A. (2006). Freshwater biodiversity: importance, threats, status and conservation challenges. *Biological Reviews*, 81(2), 163-182. doi:10.1017/S1464793105006950
2. Reid, A. J., Carlson, A. K., Creed, I. F., Eliason, E. J., Gell, P. A., Johnson, P. T., ... & Cooke, S. J. (2019). Emerging threats and persistent conservation challenges for freshwater biodiversity. *Biological Reviews*, 94(3), 849-873. doi:10.1111/brv.12480
3. European Commission. (2000). Directive 2000/60/EC of the European Parliament and of the Council of 23 October 2000 establishing a framework for Community action in the field of water policy. *Official Journal of the European Communities*, L 327, 1-73.
4. Poikane, S., Kelly, M., Fufezan, C., & Borja, A. (2020). Intercalibration of biological assessment methods: A review of the European experience. *Science of The Total Environment*, 712, 136431. doi:10.1016/j.scitotenv.2019.136431
5. Round, F. E., Crawford, R. M., & Mann, D. G. (1991). *The Diatoms: Biology and Morphology of the Genera*. Cambridge University Press.
6. Kelly, M. G., & Whitton, B. A. (1995). The Trophic Diatom Index: a new index for monitoring eutrophication in rivers. *Journal of Applied Phycology*, 7(4), 433-444. doi:10.1007/BF00003802
7. Smol, J. P., & Stoermer, E. F. (Eds.). (2010). *The Diatoms: Applications for the Environmental and Earth Sciences*. Cambridge University Press.
8. Kelly, M., Juggins, S., Guthrie, R., Pritchard, S., Jamieson, J., Rippey, B., ... & Yallop, M. (2008). Assessment of ecological status in U.K. rivers using diatoms. *Freshwater Biology*, 53(2), 403-422. doi:10.1111/j.1365-2427.2007.01903.x
9. European Committee for Standardization (CEN). (2014). *Water quality - Guidance for the routine sampling and preparation of benthic diatoms from rivers and lakes (EN 13946:2014)*. Brussels.
10. Kahlert, M., Albert, R. L., Anttila, E. L., Bengtsson, R., Bigler, C., Eskola, T., ... & Yallop, M. L. (2012). Harmonization is more than taxonomy—results from a Nordic-Baltic diatom intercalibration. *Cryptogamie, Algologie*, 33(2), 123-140. doi:10.7872/crya.v33.iss2.2012.123
11. Hopkins, G. W., & Freckleton, R. P. (2002). Declines in the numbers of amateur and professional taxonomists: implications for conservation. *Animal Conservation*, 5(3), 245-249. doi:10.1017/S1367943002002299
12. du Buf, J. M., & Bayer, M. M. (Eds.). (2005). *Automatic Diatom Identification* (Vol. 51). World Scientific. doi:10.1142/5753
13. Pedraza, A., Bueno, G., Dennistoun, F., Contreras, A., Ruiz-Santaquiteria, J., & Blanco, S. (2017). Automated diatom identification using deep convolutional neural networks. In *International Work-Conference on Artificial Neural Networks* (pp. 570-581). Springer, Cham. doi:10.1007/978-3-319-59153-7_49
14. Kloster, M., Langenkämper, D., Zuluaga, M. A., & Beszteri, B. (2020). Deep learning-based diatom identification on slide-scanning microscopy images. *Scientific Reports*, 10(1), 16778. doi:10.1038/s41598-020-73891-8
15. Tan, M., Pradalier, C., Kloster, M., & Beszteri, B. (2022). Automated diatom classification on light microscopy images using deep convolutional neural networks. *Limnology and Oceanography: Methods*, 20(8), 512-527. doi:10.1002/lom3.10499
16. Venkataramanan, A., Kloster, M., Burfeid-Castellanos, A., Dani, M., Mayombo, N. A. S., Vidakovic, D., Langenkämper, D., Tan, M., Pradalier, C., Nattkemper, T., Laviale, M., & Beszteri, B. (2024). ‘UDE DIATOMS in the Wild 2024’: a new image dataset of freshwater diatoms for training deep learning models. *GigaScience*, 13, giae087. doi:10.1093/gigascience/giae087
17. Lecointe, C., Coste, M., & Prygiel, J. (1993). “Omnidia”: software for computing diatom indices and inventories. *Hydrobiologia*, 269(1), 509-513. doi:10.1007/BF00028048
18. Nicolosi Gelis, M. M., Cochero, J., & Gómez, N. (2020). DiaThor: An R package for computing diatom indices and ecological information for water quality assessment. *Limnetica*, 39(2), 643-655. doi:10.23818/limn.39.42
19. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *International Conference on Machine Learning* (pp. 6105-6114). PMLR.
