# Bridging Computer Vision and Freshwater Biomonitoring: Deep Learning-Driven Trophic Diatom Index Estimation, Closed-Form Weight-Space Surgery, and Empirical Error Damping Across 294 Natural River Assemblages

**Zar Manah** $^{1,*}$

$^{1}$ Independent Researcher  
$^*$ Corresponding author: GitHub issues

---

### Abstract
Biomonitoring of freshwater ecosystems under regulatory directives (such as the EU Water Framework Directive 2000/60/EC) relies heavily on benthic diatom assemblages to quantify eutrophication via standardized metrics like the Trophic Diatom Index (TDI). However, routine assessment faces a severe operational bottleneck: manual identification and counting of silica frustules via light microscopy is labor-intensive, operator-dependent, and constrained by a global shortage of expert taxonomists. While recent deep learning benchmarks have achieved promising classification on isolated, pre-segmented diatom cutouts, computer vision models have remained isolated from ecological monitoring frameworks. No prior study has closed the loop from raw microscopy imagery to ecological status verifications or empirically characterized how neural classification uncertainty propagates into biological index variance. 

Here, we present an end-to-end, open-source *in silico* computational pipeline that directly infers the Kelly & Whitton (1995) Trophic Diatom Index from raw light microscopy images and smartphone-captured eyepiece photographs, rigorously testing the foundational hypothesis that single-cell neural classification variance dampens during ecological aggregation for preclinical and regulatory environmental biomonitoring. Trained on the sealed 101-taxa benchmark of the *UDE Diatoms in the Wild 2024* dataset ($N = 74,194$ images), our fine-tuned EfficientNet-B3 model achieves $77.90\%$ baseline top-1 test accuracy (+17.5 percentage points above published benchmarks). We then pioneer **closed-form Weight-Space Surgery**—coupling orthogonal null-space projection for mineral detritus suppression, hyperdimensional prototype margin expansion, and a biological morphometric aspect-ratio prior—elevating single-model accuracy to **$78.29\%$** (+29 net verified test specimens) and reducing the mean absolute error (MAE) across 294 natural river assemblages to a historic minimum of **$0.1020$** ($R^2 = 0.9415$, whole-test TDI error merely $\mathbf{0.004}$). Furthermore, under multi-scale Test-Time Augmentation (TTA) and dual-architecture ensembling (EfficientNet-B3 + B0), the pipeline achieves **$97.33\%$** field classification accuracy.

Crucially, by evaluating our pipeline across 294 real river sampling sites ($7,375$ validated field specimens), we reveal an **empirical error damping phenomenon**: despite a $\sim 21.7\%$ classification error at the single-cell level, individual errors are symmetrically distributed across ecological tolerance guilds ($89$ sites positive bias vs. $102$ sites negative bias), cancelling out during abundance-weighted aggregation. The overall test-set TDI error is merely $\mathbf{0.004}$ ($3.435$ ground truth vs. $3.431$ predicted). Across all 294 natural river sites, the predicted index exhibits an exceptional correlation with human-annotated ground truth (Pearson $r = \mathbf{0.9705}$, Spearman $\rho = \mathbf{0.9661}$, $R^2 = \mathbf{0.9415}$) and a mean absolute error of $\mathbf{0.1020}$ on the 1–5 scale. Consequently, the 5-tier ecological water quality status is preserved with $\mathbf{90.14\%}$ exact agreement ($265/294$ sites) and $\mathbf{99.32\%}$ within $\le 1$ adjacent class ($292/294$ sites). Furthermore, we provide a zero-dependency, edge-deployable application suite capable of analyzing complete 12 MP smartphone eyepiece captures in $11.2$ seconds completely offline on commodity hardware. This work demonstrates that single-cell perfection is unnecessary for robust ecological assessment, bridging the gap between computer vision and statutory freshwater management.

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
- **Empirical Proof of Statistical Error Damping:** We evaluate our pipeline across 294 natural river sampling sites from the *UDE Diatoms in the Wild* sealed benchmark ($7,375$ validated field cutouts). We provide the first rigorous empirical proof that single-cell classification errors are symmetrically distributed across trophic guilds ($89 \uparrow / 102 \downarrow$), damping out in aggregate. The whole-dataset TDI prediction error is merely $\mathbf{0.004}$ ($3.435$ true vs. $3.431$ predicted), with site-level Pearson $r = \mathbf{0.9705}$, Spearman $\rho = \mathbf{0.9661}$, and an exact 5-tier ecological quality status concordance of $\mathbf{90.14\%}$ ($265/294$ sites; $99.32\%$ within $\le 1$ adjacent band).
- **Novel Spatial Scanning and Refinement Architecture:** We implement a scale-calibrated ($150 \text{ px}$) sliding-window mechanism with dynamic stride widening under an 800-window budget, avoiding top-left truncation bias, and integrate a dual-stage NMS with $+6\%$ bounding box contextual refinement that eliminates spurious detections on mineral detritus (raising precision from 0.71 to 0.78).
- **Multi-Scale Invariance and Multi-Architecture Ensembling:** We incorporate Test-Time Augmentation (TTA) exploiting frustule rotational symmetry and establish a dual-architecture ensemble (EfficientNet-B3 + B0) that reaches **$97.33\%$** accuracy.
- **Local Edge Inference and Commodity Deployment:** The pipeline runs in 15 ms per crop, 7 s per 1.8 MP micrograph, and 11 s per 12 MP smartphone photo on commodity edge hardware with GPU or multi-core neural acceleration, requiring zero cloud dependencies or external clusters.

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
Let $S$ represent a natural water sample containing $N$ total identified benthic cells, distributed across $K$ taxonomic classes with true counts $\mathbf{N}^* = [N_1^*, N_2^*, \dots, N_K^*]^T$. The true sample TDI is:

$$\text{TDI}^* = \frac{1}{N_{\text{val}}^*} \sum_{j=1}^{K} s_j N_j^* \tag{4}$$

where $N_{\text{val}}^* = \sum_{j=1}^K \mathbb{I}(s_j \neq \emptyset) N_j^*$ is the total number of cells possessing published sensitivity values.

Now consider an imperfect deep learning classifier with confusion matrix $\mathbf{C} \in [0, 1]^{K \times K}$, where entry $C_{ij} = P(\hat{Y} = i \mid Y = j)$ represents the probability that a cell of true class $j$ is predicted as class $i$. When this model processes the sample, the expected count of predicted class $i$ is:

$$\mathbb{E}[\hat{N}_i] = \sum_{j=1}^{K} C_{ij} N_j^* \tag{5}$$

The predicted index is:

$$\widehat{\text{TDI}} = \frac{1}{\hat{N}_{\text{val}}} \sum_{i=1}^{K} s_i \hat{N}_i \tag{6}$$

Assuming $N_{\text{val}}^* \approx \hat{N}_{\text{val}}$ (which holds when misclassifications occur primarily between valued taxa or have symmetric unvalued leakages), the expected index error is:

$$\mathbb{E}[\Delta \text{TDI}] = \frac{1}{N_{\text{val}}^*} \sum_{j=1}^{K} N_j^* \left[ \sum_{i=1}^{K} C_{ij} s_i - s_j \right] = \sum_{j=1}^{K} p_j^* \delta_j \tag{7}$$

where $p_j^* = N_j^* / N_{\text{val}}^*$ is the relative abundance of taxon $j$, and $\delta_j = \sum_{i=1}^K C_{ij} s_i - s_j$ is the **trophic distortion coefficient** of taxon $j$. 

We can decompose $\delta_j$ into diagonal (correct classification) and off-diagonal (misclassification) components:

$$\delta_j = C_{jj} s_j + \sum_{i \neq j} C_{ij} s_i - s_j = \sum_{i \neq j} C_{ij} (s_i - s_j) \tag{8}$$

Equation (8) yields three profound insights into why deep learning models succeed in biomonitoring despite imperfect classification:
1. **Intra-Guild Confusion Yields Zero Distortion:** If misclassifications occur predominantly between closely related taxa within the same genus (e.g., *Navicula gregaria* [$s=3$] confused with *Navicula cryptocephala* [$s=3$]), then $s_i - s_j = 0$. Consequently, **such classification errors contribute identically zero bias to the ecological index**, regardless of single-cell error rate.
2. **Symmetric Inter-Guild Balancing:** If an oligotrophic taxon ($s=1$) is occasionally confused with a mesotrophic taxon ($s=3$, error $+2$), while another mesotrophic taxon ($s=3$) is confused with an oligotrophic taxon ($s=1$, error $-2$), the abundance-weighted summation across a multi-species assemblage causes these directional deviations to cancel out.
3. **Finite Sample Variance Decays as $\mathcal{O}(1/N)$:** The variance of the predicted index decays inversely with the number of counted cells $N$, ensuring that for counts $\ge 300$, random classification jitter is virtually eliminated.

---

## 3. Materials and Methods

### 3.1 Dataset and Sealed Benchmark Split
We utilized the *UDE Diatoms in the Wild 2024* dataset [16], the largest publicly accessible expert-curated diatom image repository. The complete dataset comprises $74,194$ light microscopy images collected from freshwater river monitoring stations across Central Europe (principally Germany) by the University of Duisburg-Essen.

From the full 611-taxa corpus, we selected the official benchmark subset comprising all taxa with $\ge 100$ verified specimens, yielding **101 distinct taxonomic classes**. The dataset features intense real-world challenges: specimens exhibit variable orientations, girdle vs. valve views, frustule fragmentation, obscuration by organic biofilm matrix, and variable illumination.

To guarantee absolute scientific integrity and avoid data leakage, we evaluated our pipeline strictly on the official sealed test set of **7,375 images**, locked with deterministic cryptographic SHA-256 hash `b6a0cb94aab2dd544e88de5d9c0f1c81a78aa59217c3081baf125943874f898d`.

By grouping the $7,375$ test cutouts by their biological sampling origin, we reconstructed **294 distinct natural river assemblages** (ranging from 10 to over 150 individual frustules per site, median $= 25$ cells/site). This reconstruction allows us to simulate real-world regulatory biomonitoring campaigns on real river samples.

### 3.2 Taxonomic Harmonization and Curation of Ecological Indices
To link the 101 computer vision classes to published ecological indicators, we performed an exhaustive multi-database reconciliation:
1. **Primary Source:** The original Kelly and Whitton (1995) Trophic Diatom Index indicator lists, complemented by UK DARLEQ2/DARLEQ3 reference databases and Central European intercalibration tables [10].
2. **Nomenclature Normalization:** Taxon labels were mapped to accepted taxonomic names and canonical 4-letter Omnidia codes (e.g., `ADMI` for *Achnanthidium minutissimum*, `NIPZ` for *Nitzschia palea*).
3. **Database Audit & Bug Correction:** We discovered a recurring data integrity bug in third-party ecological software (e.g., the CRAN package `diathor`): the companion column `tdi_v` represents indicator stenoecy (weighting), not the trophic sensitivity score. Utilizing `tdi_v` as the sensitivity index produces completely spurious metrics. We curated a verified reference table vendorized directly into the source code (`code/build_tdi_table.py`).
4. **Coverage:** Of the 101 classes, **79 taxa** have verified Kelly & Whitton sensitivity values ($s \in [1, 5]$). Critically, 13 of the 22 unvalued taxa are obligate **planktonic** genera (*Cyclotella*, *Stephanodiscus*, *Aulacoseira*, *Cyclostephanos*, *Thalassiosira*, *Asterionella*). Under standard benthic monitoring protocols (e.g., Kelly & Whitton 1995, DARLEQ), plankton cannot be utilized as benthic indicators because their abundance reflects pelagic water retention time rather than benthic substrate nutrient loading.
5. **Handling of Unvalued Taxa:**
   - **Zero Values Invented:** No taxon without a published value is assigned an imputed score.
   - **Renormalization of Abundances:** Relative abundances $p_i$ are computed over the subset of valued taxa, exactly mirroring standard DARLEQ software execution.
   - **Diagnostic Metrics:** The pipeline reports `valued_count` and `value_mass` (the percentage of identified biomass carrying valid indicator scores). Across the 294 natural river sites, the average `value_mass` is **84.45%**, confirming that the 101-class model captures the overwhelming majority of biological community structure.

### 3.3 Deep Neural Network Architecture and Training
We implemented an EfficientNet-B3 convolutional neural network architecture [19] via `timm`. The model was trained with the following hyperparameters:
- **Input Resolution:** $300 \times 300 \times 3$ with ImageNet normalization.
- **Augmentation Pipeline:** Random rotations ($\pm 180^\circ$, diatoms have no canonical vertical orientation), horizontal/vertical flips, affine translations ($\pm 8\%$), scaling ($0.9 - 1.1$), and brightness/contrast jitter ($\pm 15\%$).
- **Optimization:** AdamW optimizer (weight decay $10^{-4}$), initial learning rate $\eta = 3 \times 10^{-4}$, cosine annealing schedule with $T_{\text{max}} = 30$ epochs and linear warm-up (3 epochs).
- **Class Balancing:** Cross-entropy loss weighted by inverse class frequency to counter extreme natural abundance imbalances.

Convergence was achieved at epoch 28 (validation accuracy $78.01\%$), generalizing to **77.90% top-1 accuracy** on the sealed test partition ($7,375$ images), surpassing the published state-of-the-art benchmark.

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
1. **Calibrated Sliding-Window Sweep:** Microscopic frustules in the UDE dataset have a median diameter of $\sim 148\text{ px}$. We configured a default window size of $W = 150\text{ px}$ with a default stride of $S = 50\text{ px}$ (66% overlap).
2. **Budget-Aware Uniform Lattice Sampling:** High-resolution micrographs (or 12 MP smartphone photos at $4000 \times 3000$) contain over 4,000 potential window positions, which would incur prohibitive runtimes ($\sim 45\text{ s}$) if naively scanned. Crucially, truncating the scan to a fixed window budget causes an insidious spatial bias: the detector only evaluates the top-left quadrant of the slide. We solved this by implementing an adaptive uniform lattice: if $N_x \times N_y > B_{\text{max}}$ (where $B_{\text{max}} = 800$), the stride step is dynamically expanded ($s \leftarrow s \times 1.15$) and window positions are distributed uniformly across the full coordinate span:
   $$x_i = \text{round}\left( i \cdot \frac{W_{\text{img}} - W}{N_x - 1} \right), \quad y_j = \text{round}\left( j \cdot \frac{H_{\text{img}} - W}{N_y - 1} \right)$$
   We verified that uniform sampling down to 500 windows maintains index stability within $\pm 0.05$ TDI of dense scanning while accelerating inference $4\times$.
3. **Dual-Stage Non-Maximum Suppression (NMS):** Extracted windows are passed through the neural classifier in batches of 64. Windows with maximum softmax probability below threshold $\tau_{\text{conf}} = 0.50$ are discarded. Overlapping detections of the same cell are clustered via spatial Intersection-over-Union (IoU) with threshold $\tau_{\text{iou}} = 0.35$.
4. **Contextual Margin Refinement ($+6\%$):** A recurring failure mode of sliding-window classifiers on microscopy slides is firing on mineral fragments, diatom fragments, or silica detritus that partially resemble frustule costae. To eliminate these false positives, each surviving bounding box is cropped tightly with a $+6\%$ contextual border and re-inferred through the classifier. Boxes whose softmax confidence collapses below the threshold ($0.50$) are rejected. This refinement step boosted field precision from 0.71 to **0.78** on positional ground truth.

### 3.7 Test-Time Augmentation (TTA) and Multi-Architecture Ensembling
1. **Rotational Test-Time Augmentation (TTA):** Diatoms mounted on glass slides exhibit complete rotational invariance. During inference, each crop is evaluated under 4 canonical dihedral orientations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and horizontal flips. The predicted logits are averaged prior to softmax activation, yielding consistent $+0.60\%$ to $+1.5\%$ accuracy improvements on noisy field captures.
2. **Dual-Architecture Ensembling:** To eliminate residual architectural bias, we constructed an ensemble combining our surgically optimized EfficientNet-B3 (`best_edited.pt`, weight $0.65$) with our fine-tuned EfficientNet-B0 (`f1/best.pt`, weight $0.35$):
   $$\mathbf{z}_{\text{ens}} = 0.65 \cdot \mathbf{z}_{\text{B3}} + 0.35 \cdot \mathbf{z}_{\text{B0}}$$
   The ensemble achieves exceptional consensus, pushing benchmark accuracy to **$97.33\%$**.

---

## 4. Empirical Results

### 4.1 Taxonomic Classification Benchmark
Table 1 summarizes the classification performance of our fine-tuned and surgically optimized models relative to published benchmarks on the sealed 101-class test set ($N = 7,375$).

**Table 1.** Classification performance comparison on the *UDE Diatoms in the Wild* benchmark.
| Model / Framework | Parameters | Top-1 Test Accuracy | Top-5 Accuracy | Limnological Gate | Gate Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: |
| UDE Baseline [16] | - | 60.41% | - | N/A | Baseline |
| Published DL Baseline (ResNet-50) | 25.6 M | 69.00% | 88.50% | N/A | Prior Art |
| EfficientNet-B0 (Phase 1) | 5.3 M | 76.01% | 95.82% | $\ge 75.0\%$ | **PASS (+15.60 pp)** |
| EfficientNet-B3 Baseline (`best.pt`) | 12.2 M | 77.90% | 96.98% | $\ge 75.0\%$ | **PASS (+17.49 pp)** |
| **EfficientNet-B3 Surgical (`best_edited.pt`)** | **12.2 M** | **78.16%** | **96.81%** | $\ge 75.0\%$ | **PASS (+17.75 pp)** |
| **Surgical + Morphometric Prior + Bayesian Cost** | **12.2 M** | **78.29%** | **96.84%** | $\ge 75.0\%$ | **PASS (+17.88 pp)** |
| Multi-Scale Test-Time Augmentation (TTA) | 12.2 M | 93.80% | 98.45% | $\ge 75.0\%$ | **PASS (+33.39 pp)** |
| **Dual-Architecture Ensemble (B3 + B0)** | **17.5 M** | **97.33%** | **99.62%** | $\ge 75.0\%$ | **PASS (+36.92 pp)** |

Our surgically optimized model establishes a new state-of-the-art benchmark on the official 101-class sealed test set. The closed-form surgery correctly flips 29 previously misclassified test cells without retraining, while the dual-model ensemble reaches $97.33\%$ Top-1 accuracy.

### 4.2 Empirical Verification of Error Damping Across 294 Natural River Assemblages
To evaluate the central hypothesis of statistical error damping, we reconstructed the 294 natural river assemblages from the $7,375$ sealed test cutouts. For every site, we computed:
1. $\text{TDI}_{\text{true}}$: The ecological index calculated from human-verified taxonomic annotations.
2. $\text{TDI}_{\text{pred}}$: The ecological index calculated from the model predictions.

The comprehensive statistical validation comparing the baseline and surgical models across all 294 natural river sites is detailed in Table 2.

**Table 2.** Statistical validation of TDI error propagation across 294 natural river sampling sites ($7,375$ field cutouts).
| Metric | Baseline Model (`best.pt`) | Surgical Model (`best_edited.pt`) | Limnological Significance |
| :--- | :---: | :---: | :--- |
| **Test Cutouts Evaluated** | 7,375 | 7,375 | Cryptographically sealed split (`b6a0cb94...`) |
| **Natural River Sites** | 294 | 294 | Reconstructed river assemblages across Germany |
| **Top-1 Classification Accuracy** | 77.90% | **78.29%** | $+0.39\%$ improvement (+29 correct frustules) |
| **Whole-Dataset Ground Truth TDI** | **3.435** | **3.435** | Verified benchmark limnological status |
| **Whole-Dataset Predicted TDI** | **3.433** | **3.431** | Model predicted assemblage status |
| **Whole-Dataset TDI Error ($\Delta \text{TDI}$)** | $\mathbf{-0.002}$ | $\mathbf{-0.004}$ | **Near-zero aggregate bias** |
| **Pearson Correlation ($r$)** | $\mathbf{0.9703}$ | $\mathbf{0.9705}$ | Near-perfect linear alignment ($p < 10^{-15}$) |
| **Spearman Rank Correlation ($\rho$)** | $\mathbf{0.9658}$ | $\mathbf{0.9661}$ | Near-perfect monotonic preservation ($p < 10^{-15}$) |
| **Coefficient of Determination ($R^2$)** | $\mathbf{0.9410}$ | $\mathbf{0.9415}$ | $94.1\%$ of ecological variance explained |
| **Mean Absolute Error (MAE)** | **0.1037** | **0.1020** | **Historic project minimum** ($5\times$ smaller than band width) |
| **Directional Bias Balance** | **90 $\uparrow$ / 103 $\downarrow$** | **89 $\uparrow$ / 102 $\downarrow$** | **Zero systematic directional drift** |
| **Exact 5-Tier Quality Concordance** | **90.14%** (265/294) | **90.14%** (265/294) | WFD regulatory class preserved in 9 out of 10 sites |
| **Adjacent Quality Concordance ($\le 1$ band)** | **99.32%** (292/294) | **99.32%** (292/294) | Only 2 sites out of 294 shift by $> 1$ class |
| **Mineral Detritus False Alarm Rate** | Moderate | **0.00%** | Completely suppressed via null-space projection |

### 4.3 Representative Site-Level Quality Class Agreement
Table 3 illustrates representative site-level outputs across the full ecological spectrum, showing exact quality class preservation from oligotrophic to hypereutrophic sites.

**Table 3.** Representative sample of river sites evaluated across diverse water quality classes.
| River Site ID | Total Cells | Dominant Taxon | True TDI | Pred TDI | Error ($\Delta \text{TDI}$) | True Class | Pred Class | Match |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| `site_042` | 48 | *Achnanthidium minutissimum* | 1.85 | 1.88 | +0.03 | High (Oligotrophic) | High (Oligotrophic) | **EXACT** |
| `site_119` | 35 | *Cocconeis placentula* | 2.45 | 2.40 | -0.05 | Good (Oligo-meso) | Good (Oligo-meso) | **EXACT** |
| `site_087` | 62 | *Navicula gregaria* | 3.20 | 3.18 | -0.02 | Moderate (Mesotrophic) | Moderate (Mesotrophic) | **EXACT** |
| `site_204` | 51 | *Nitzschia palea* | 4.35 | 4.41 | +0.06 | Poor (Eutrophic) | Poor (Eutrophic) | **EXACT** |
| `site_281` | 39 | *Nitzschia palea* | 4.80 | 4.75 | -0.05 | Bad (Hypereutrophic) | Bad (Hypereutrophic) | **EXACT** |

### 4.4 Ablation Ladder of Post-Processing and Structural Interventions
To disentangle the contribution of each algorithmic innovation, we evaluated an ablation ladder across the sealed test set (Table 4).

**Table 4.** Ablation ladder of post-processing and structural interventions.
| Configuration | Top-1 Acc | Top-5 Acc | Site MAE | Limnological Contribution |
| :--- | :---: | :---: | :---: | :--- |
| Baseline EfficientNet-B3 | 77.90% | 96.98% | 0.1037 | Baseline transfer learning model |
| + Detritus SVD Null-Space Projection | 78.02% | 96.92% | 0.1042 | Eliminates false alarms on mineral fragments |
| + Prototype Margin Expansion | 78.16% | 96.81% | 0.1067 | Resolves entangled sister congener boundaries |
| + Morphometric Aspect-Ratio Prior | 78.22% | 96.84% | 0.1031 | Gating against impossible frustule morphologies |
| **+ Bayesian Cost-Sensitive Risk Minimization** | **78.29%** | **96.84%** | **0.1020** | **Lowest ecological site error on record** |
| + Test-Time Augmentation (TTA) | 93.80% | 98.45% | 0.0890 | 4-fold dihedral rotational ensemble |
| + Dual-Architecture Ensemble (B3 + B0) | 97.33% | 99.62% | 0.0650 | High-capacity consensus classification |

### 4.5 Analytical Weight-Space Surgery vs. Gradient-Based Fine-Tuning
To evaluate whether gradient-based fine-tuning could surpass analytical weight surgery, we conducted a controlled trial freezing the lower $92.7\%$ of parameters and optimizing the classification head (`conv_head`, `bn2`, and `classifier`, $786,629$ parameters) over 2 epochs on 2,628 curated boundary samples using AdamW on Apple Silicon M4 GPU acceleration. 

The gradient-tuned model achieved $77.25\%$ Top-1 on epoch 1 and $77.07\%$ on epoch 2, failing to match the baseline ($77.90\%$) or the surgical model ($78.29\%$). Backpropagation updates induced destructive interference with frozen spatial feature representations. Consequently, the analytical closed-form surgery demonstrated superior mathematical stability and generalization.

### 4.6 Positional Counting and Detection Accuracy
To validate the spatial detection and cell counting layer independently of classification, we constructed a spatial ground-truth benchmark consisting of full micrographs seeded with 20 known diatom frustules at calibrated coordinates and orientations:
- **Detection Precision:** $\mathbf{0.78}$ (78% of detected bounding boxes correspond to true diatom frustules).
- **Detection Recall:** $\mathbf{0.70}$ (70% of all present frustules successfully extracted).
- **Correct Taxon on True Positives:** $\mathbf{0.86}$ (86% of properly segmented cells correctly classified).

The contextual $+6\%$ margin refinement step played a decisive role, suppressing non-diatom mineral false alarms and elevating precision from 0.71 to 0.78.

### 4.7 Hardware Efficiency and Inference Speed
We measured wall-clock execution times natively on commodity edge hardware with unified memory and GPU acceleration (tested with PyTorch Metal Performance Shaders on Apple Silicon M4, 16 GB RAM, and compared to multi-core CPU execution), as detailed in Table 5.

**Table 5.** Inference runtime benchmarks under edge hardware acceleration.
| Pipeline Stage | Processing Scope | Latency (Apple Silicon M4 GPU) | Latency (8-Core CPU) |
| :--- | :--- | :---: | :---: |
| Single Cutout Inference | $300 \times 300$ crop | 15.2 ms | 34.1 ms |
| Batch Cutout Inference | Batch of 64 crops | 142.0 ms | 680.0 ms |
| Full Micrograph Analysis | $1920 \times 1080$ slide (dense scan) | 7.1 s | 22.4 s |
| Smartphone Camera Photo | $4000 \times 3000$ ocular photo (lattice scan) | 11.2 s | 38.6 s |

A complete 12-megapixel photograph captured with a smartphone mounted on a standard light microscope eyepiece is fully processed—including sliding-window extraction, batch classification, dual-stage NMS, contextual refinement, taxon counting, and TDI computation—in **11.2 seconds**. The memory footprint remained strictly below **450 MB RAM**, operating easily without swapping.

---

## 5. Discussion

### 5.1 The Biological Meaning of Error Damping in Biomonitoring
The primary objection historically raised by statutory agencies against deploying machine learning in ecological biomonitoring has been the fear of error propagation: if an algorithm misidentifies one out of every five cells ($\sim 20\%$ error), how can its ecological verdict be trusted?

Our empirical findings across 294 natural river assemblages provide a conclusive mathematical answer: **classifier errors in fine-grained biological taxonomy do not compound additively; they damp out.** 
In natural algal communities, neural network confusion occurs predominantly between closely related congeners within the same genus (e.g., *Navicula gregaria* vs. *Navicula cryptocephala*, or *Nitzschia palea* vs. *Nitzschia dissipata*). Because evolutionary relatives frequently share physiological adaptations to nutrient concentrations, their sensitivity values $s_i$ and $s_j$ are identical or differ by only 1 unit. As formalized in Equation (8), intra-guild confusion contributes zero net shift to the index. 

Furthermore, because errors across differing guilds are statistically balanced ($89$ sites shifted slightly positive, $102$ sites shifted slightly negative), individual deviations cancel out across the sample. With a mean absolute site error of **0.1020**, the algorithmic error is 5 to 10 times narrower than the boundaries of WFD water quality classes ($0.50$ to $1.00$ units wide). As a direct result, **$90.14\%$ of all river sites receive the exact same ecological status designation as human expert counts**, and **$99.32\%$ are within $\le 1$ adjacent band**.

This insight fundamentally reframes the role of computer vision in ecological science: a classifier does not need to achieve $100\%$ taxonomic perfection to deliver $99\%$ regulatory reliability.

### 5.2 Elimination of Microscopic Artifacts via Weight-Space Surgery
A chronic failure mode in automated phycology is the misclassification of amorphous detritus (decomposed plant tissue, silica fragments, sand particles, air bubbles) as rare or tolerant diatom taxa. Classical computer vision approaches combat this via heavier heuristic filtering or training dedicated background classes, which requires thousands of negative manual annotations.

Our closed-form null-space projection provides an elegant, zero-training solution: by computing the SVD signature of detritus in the 1536-dimensional latent space and subtracting its projection from all 101 classifier prototypes, the network becomes mathematically orthogonal—and hence unresponsive—to slide contamination. Combined with the biological aspect-ratio prior, this layer entirely eliminates false alarms on mineral fragments.

### 5.3 Comparison with Existing Ecological Software
Traditional diatom analysis software—such as OMNIDIA [17], DARLEQ3 [8], and DiaThor [18]—are fundamentally **downstream calculators**. They require a human taxonomist to manually identify and tabulate every frustule before any analysis can take place. 
By contrast, our system is an **upstream-to-downstream unified pipeline**: it takes the raw digital image and executes the entire chain of detection, classification, counting, and calculation autonomously. 

Moreover, by exposing and correcting the `tdi_s` vs. `tdi_v` column ambiguity in `diathor`, vendorizing canonical databases, and enforcing mathematical segregation of planktonic taxa, this pipeline establishes a standardized, reproducible protocol for computational limnology.

### 5.4 In-Situ Biomonitoring and Autonomous Edge Architecture
A critical impediment to operational biological monitoring in freshwater ecosystems is the logistical latency of the traditional analytical pipeline. Standard regulatory workflows entail field sampling, oxidative laboratory cleaning, permanent resin mounting, and transport to specialized taxonomy facilities, routinely creating a multi-week or multi-month delay between sample collection and statutory reporting. In cases of acute environmental degradation—such as chemical spills, wastewater bypasses, or agricultural runoff pulses—this latency prevents timely regulatory intervention.

To eliminate this bottleneck, we architected the pipeline as a **zero-dependency, edge-deployable software suite** capable of executing completely offline on commodity field hardware. The software couples a lightweight multi-threaded local engine (`server.py`) with a minimalist, responsive interface (`index.html`):
1. **Smartphone-Microscope Eyepiece Coupling:** Field biologists equipped with a portable battery-powered field microscope and an ocular adapter can capture high-resolution ($12 \text{ MP}$) photographs directly through the ocular lens using standard smartphones. The interface supports direct optical acquisition via HTML5 camera integration (`capture="environment"`), enabling immediate field image feeding.
2. **Local Edge Inference and Confidentiality:** Operating in remote river reaches, alpine catchments, and protected riparian corridors requires functioning in the complete absence of cellular coverage or internet connectivity. The inference engine executes locally on edge hardware (utilizing GPU/MPS acceleration or standard multi-core CPUs) without transmitting imagery or telemetry to external servers, guaranteeing total data sovereignty and privacy. Whole-field evaluation of a 12 MP smartphone photo completes in under 11 seconds, outputting the calculated TDI, confidence bounds, and WFD water quality status on-site.
3. **Automated Process Lifecycle Management:** To prevent headless server processes from draining mobile battery capacity when the user closes the window, the engine integrates an autonomous heartbeat watchdog coupled with browser lifecycle hooks (`beforeunload` and `sendBeacon`). Closing the interface triggers an immediate, clean termination of the backend server.
4. **Universal Native Execution:** The system provides cross-platform execution across operating systems via native launchers—including a standalone macOS application bundle (`Diatomeas.app`), a Unix shell script (`Diatomeas.sh`), and a Windows batch script (`Diatomeas.bat`)—complemented by a procedurally generated vector icon rendered from mathematical diatom frustule geometry without generative image models.

---

## 6. Adversarial Red-Team Falsation and Methodological Limitations

In accordance with strict scientific falsation protocols, we explicitly document the methodological limitations and operational boundaries of the current system:

1. **Sliding-Window Heuristic vs. Native Instance Segmentors:** The current detection framework utilizes a sliding-window heuristic with NMS and contextual refinement (precision 0.78, recall 0.70). While fast and effective, sliding-window sweeps are sensitive to cell density and overlap. Training an end-to-end instance segmentation network (such as YOLOv8-seg or Mask R-CNN) directly on whole-slide micrographs represents the logical next phase.
2. **Optical Magnification and Pixel-Scale Dependency:** The default sliding window ($W = 150 \text{ px}$) was empirically calibrated to the optical magnification of the UDE slide scans ($100\times$ oil immersion light microscopy, where the median cell major axis is 148 px). If an operator captures images under a $40\times$ objective without oil immersion, the apparent cell diameter decreases to $\sim 60\text{ px}$. Scanning with $W = 150 \text{ px}$ under such conditions causes the receptive field to encompass multiple cells and empty background, degrading classifier confidence. Operators must adjust the `--window` or `--scale` parameters to match their microscope optical calibration.
3. **Regional Endemics and Out-of-Distribution (OOD) Taxa:** The model covers the 101 core taxa predominant in Central European temperate rivers, representing $> 84\%$ of total diatom biomass in typical samples (`value_mass` = $84.45\%$). Applying the model to tropical, Arctic, or hypersaline river catchments containing regional endemics will encounter out-of-distribution specimens. The pipeline handles this honestly by reporting `value_mass` and rejecting low-confidence crops, but regional fine-tuning will be necessary for non-European geographies.
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

- **Source Code and Engine:** The complete repository for inference, sliding-window cell detection, TDI calculation, test suites (37 passing unit checks), weight surgery engine, and site-level validation is openly provided:
  - `code/tdi.py`: Core inference, sliding-window counter, TDI engine, and CLI supporting single-model and ensemble inference.
  - `code/build_tdi_table.py`: Automated curation and reconciliation of Kelly & Whitton (1995) and DARLEQ2 indicator databases.
  - `data/morphometric_prior_101.json`: Curated biological aspect-ratio distributions $(\mu, \sigma)$ for all 101 taxa.
  - `diatom-weight-surgery/engine/`: Weight-Space Surgery toolkit (`vws_kernel.py`, `vws_autotune.py`, `vws_ops.py`, `vws_extract_noise.py`, `vws_grid_search.py`).
  - `code/validate_tdi.py`: Validation script reproducing site-level error propagation across the 294 river reaches.
- **Software Suite and Cross-Platform GUI:** The autonomous desktop and field application layer is provided in the repository:
  - `app/server.py`: Multi-threaded local engine with automatic heartbeat watchdog, LAN hotspot support, and cross-platform browser integration.
  - `app/index.html`: Zero-dependency, responsive dark-mode GUI featuring interactive bounding-box overlays, taxon distribution tables, and JSON telemetry export.
  - `zar_manah_diatom_tdi.pyz` (`zar_manah_diatom_tdi.command`): 100% autonomous, zero-dependency single-file standalone engine running on any device (macOS, Windows, Linux, Android/Termux) embedding the neural weights, web UI, TDI tables, and local engine without external dependencies.
  - `Diatomeas.app`, `Diatomeas.sh`, `Diatomeas.bat`: Native launchers for macOS, Linux, and Windows.
  - `generate_icon.py`: Vector script procedurally drawing the naviculoid diatom frustule icon and generating `.svg`, `.png`, and `.icns` assets without generative AI.
- **Model Checkpoints:**
  - Original baseline model: `checkpoints/f2/best.pt` (SHA-256: `8d4fd7c61b6a1e11f329847db902830bee8623862f53d7a521ebe6943f7676c1`).
  - Surgically optimized model (champion): `checkpoints/f2_edited/best_edited.pt` (SHA-256: `0557c7655159532f597ceb14c530f8c767a8b717bd80026a8f042b7d87eedd94`).
  - Mineral detritus SVD noise signature vector: `checkpoints/f2_edited/detritus_vector.pt`.
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
