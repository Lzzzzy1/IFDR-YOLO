---
title: "Novelty collision matrix A — P2, dynamic/selective pyramids, QueryDet, and ESOD"
task_id: task-a-p2-pyramids
search_date: 2026-08-10
timezone: Asia/Shanghai
as_of: 2026-08-10
status: complete
source_policy: original papers and author/official repositories only
verdict: "DEFENSIBLE GAP (provisional), with material collisions; this slice alone does not authorize implementation"
---

# Novelty collision matrix A: P2 and selective high-resolution detection

## Scope and frozen comparator

This note audits only the following prior-art slice: FPN/P2 high-resolution heads, the official ordinary YOLOv8-P2 configuration, dynamic/selective feature pyramids, QueryDet, ESOD, and one directly relevant 2025 sparse-pyramid successor (VISO). It does **not** replace the parallel audits of zoom/coarse-to-fine, road geometry, uncertainty-guided computation, NWD/RFLA, or early-exit/dynamic inference.

The comparator is the frozen F4 design in the [2026-08-10 value-of-resolution P2 specification](../../superpowers/specs/2026-08-10-hard-budget-value-of-resolution-p2-design.md):

1. run leakage-free plain P2 F0 once;
2. form GT-free road candidates;
3. estimate each candidate's *counterfactual marginal detection utility* from road-geometry confidence plus first-pass detector residual/uncertainty;
4. under a deterministic per-frame budget, choose at most one fixed half-image crop, with choosing none allowed; and
5. when every predicted utility is non-positive, return the unmodified F0 output (the declared no-harm fallback).

Here, “value of resolution” (VoR) means predicted *base-versus-re-observation utility*, not objectness, classification confidence, a feature-level assignment loss, an attention mask, or generic saliency.

## Executive verdict

**Scoped verdict: DEFENSIBLE GAP, PROVISIONAL / PASS WITH CONDITIONS.** None of the primary sources in this slice contains the full joint mechanism of road-geometry confidence + first-pass residual/uncertainty + candidate-specific counterfactual VoR + deterministic at-most-one-crop budget + exact F0 fallback. That narrow conjunction is therefore still defensible against this slice.

The gap is narrow. There are material collisions:

- **Ordinary P2/FPN and official YOLOv8-P2 are exact prior art for a high-resolution P2 head.** F0 is a baseline, not a contribution.
- **QueryDet is a high collision** for coarse-to-fine query-guided sparse computation on high-resolution P2 features.
- **ESOD is the strongest collision in this slice** for object-seeking, adaptive feature slicing, background removal, and sparse high-resolution detection, including an official YOLO-family implementation.
- **DyFPN collides with resource-aware conditional feature-pyramid computation**, although it gates feature branches rather than an image re-observation.
- **FSAF collides with per-instance feature-level selection**, although its selection is a supervised training assignment and all levels are used at inference.
- **VISO reinforces that learned multi-scale masks and threshold-controlled sparse pyramid/head inference are already prior art.**

Accordingly, a claim framed as “dynamic P2,” “selective high-resolution detection,” “query/crop small-object detection,” “adaptive slicing,” or “budget-aware feature pyramids” is not defensible. A claim may only be pursued around the precise estimand and decision rule of *counterfactual base-versus-re-observation utility under a hard one-crop budget with exact base-output fallback*, subject to the other collision-matrix slices.

### Candidate falsifiable contribution statement

> Under the same images, seed, candidate set, and deterministic budget of at most one fixed half-image crop per frame, estimating each GT-free road candidate's counterfactual base-versus-re-observation detection utility from road-geometry confidence and first-pass residual/uncertainty yields higher KITTI Pedestrian/Cyclist Moderate macro AP_R40 than geometry-only, uncertainty-only, and value-agnostic selection at identical compute; when all predicted utilities are non-positive, the method emits the exact F0 result.

This is a **hypothesis/design claim, not an observed result**. It is falsified by the frozen ablations or stopping gates. The overall research targets (+4.0 absolute AP points for F4 versus F0 and +5.0 for the full system versus original P3–P5 YOLO) remain targets, never promises.

## Primary-source register

All factual method claims below are grounded in the listed original papers or author/official repositories. Search-result snippets were used only to locate these sources. No survey, blog, aggregator, or unofficial implementation is used as evidence.

| ID | Primary source | Source type and authority | Accessibility checked 2026-08-10 | Official/author code and visible license |
|---|---|---|---|---|
| S1 | Lin et al., [*Feature Pyramid Networks for Object Detection*](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html), CVPR 2017 ([PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf), [arXiv](https://arxiv.org/abs/1612.03144)) | Original peer-reviewed paper; CVF open-access copy | HTML/PDF/arXiv accessible | [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron), official FAIR research platform explicitly listing FPN; Apache-2.0 visible; archived/read-only since 2023 |
| S2 | Ultralytics, [`yolov8-p2.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml) | Official project configuration, not a separate paper | File and raw view accessible | Same official repository; file header visibly states AGPL-3.0 |
| S3 | Zhu et al., [*Dynamic Feature Pyramid Networks for Object Detection*](https://arxiv.org/abs/2012.00779), 2020 | Original arXiv preprint; no peer-reviewed venue established from primary sources in this search | Abstract/PDF accessible | [Mingjian-Zhu/DyFPN](https://github.com/Mingjian-Zhu/DyFPN), author repository; Apache-2.0 visible |
| S4 | Zhu, He, and Savvides, [*Feature Selective Anchor-Free Module for Single-Shot Object Detection*](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.html), CVPR 2019 ([PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.pdf), [arXiv](https://arxiv.org/abs/1903.00621)) | Original peer-reviewed paper; CVF open-access copy | HTML/PDF/arXiv accessible | No author/official repository located in the recorded searches; unofficial ports excluded |
| S5 | Yang, Huang, and Wang, [*QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection*](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html), CVPR 2022 ([PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf), [arXiv](https://arxiv.org/abs/2103.09136)) | Original peer-reviewed paper; CVF open-access copy | HTML/PDF/arXiv accessible | [ChenhongyiYang/QueryDet-PyTorch](https://github.com/ChenhongyiYang/QueryDet-PyTorch), author-designated official implementation; MIT visible |
| S6 | Liu et al., [*ESOD: Efficient Small Object Detection on High-Resolution Images*](https://arxiv.org/abs/2407.16424), IEEE TIP (author repo cites vol. 34, 2025), [DOI 10.1109/TIP.2024.3501853](https://doi.org/10.1109/TIP.2024.3501853) | Original author manuscript plus canonical DOI; repository supplies publication citation | arXiv accessible; publisher landing page was not retrievable in this environment, so no claim depends on its contents | [alibaba/esod](https://github.com/alibaba/esod), official repository; GPL-3.0 visible; README says a large part derives from YOLO and includes a third-party SAM tree |
| S7 | Wang and Qiu, [*VISO: Accelerating In-orbit Object Detection with Language-Guided Mask Learning and Sparse Inference*](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_VISO_Accelerating_In-orbit_Object_Detection_with_Language-Guided_Mask_Learning_and_ICCV_2025_paper.html), ICCV 2025 ([PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_VISO_Accelerating_In-orbit_Object_Detection_with_Language-Guided_Mask_Learning_and_ICCV_2025_paper.pdf)) | Original peer-reviewed paper; CVF open-access copy | HTML search record/PDF accessible | [joannahuadu/VISO](https://github.com/joannahuadu/VISO), author-designated official implementation; **no top-level license was visible** in the inspected file listing |

## Component-by-component collision matrix

Legend: **Y** = substantive match; **P** = partial/adjacent match but a different signal, target, granularity, or constraint; **N** = not part of the method as specified in the primary source; **—** = not applicable because there is no optional re-observation path. “Hard budget” means the frozen deterministic *zero-or-one fixed crop per frame*, not expected FLOPs, a learned resource loss, or a tunable threshold.

| Prior | Core mechanism | Road-geometry confidence | First-pass detection residual / uncertainty | Candidate-specific counterfactual VoR | Selective high-resolution computation | Hard zero-or-one crop budget | Exact unmodified-base fallback | Strongest collision and scoped verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| FPN (S1) | Top-down pyramid with lateral connections gives semantically strong features at multiple resolutions; P2 is the finest normal pyramid level and detection is dense across levels. | N | N | N | N (P2 is always evaluated) | N | — | **Material collision:** P2/high-resolution semantic heads and multi-scale detection are established prior art. |
| Official YOLOv8-P2 (S2) | Static YOLOv8 neck/head upsamples and concatenates P2, then predicts densely from P2/P3/P4/P5. | N | N | N | N (static dense head) | N | — | **Exact F0 collision:** a plain P2 YOLO configuration cannot be called original. |
| DyFPN (S3) | Per-level gates conditionally execute a heavier inception-style lateral branch, retain a light 1×1 path, and use a computational-cost/resource objective. | N | N | N (gate is pooled feature evidence, not a base-vs-reobserve effect) | P (conditional feature branch, not image pixels/crop) | N (resource target/average computation, not per-frame cardinality) | N (internal skip path is not exact result fallback) | **Material partial collision:** “dynamic/resource-aware FPN” and skipping expensive computation for easy inputs are prior. |
| FSAF (S4) | An anchor-free branch is attached to each FPN level; during training, each instance is assigned to the level with minimum classification-plus-regression loss; inference uses all levels. | N | N | P (instance/level loss assignment, not inference-time marginal pixel value) | P (training selection only; inference is dense) | N | — | **Material partial collision:** per-instance content/loss-based feature-level selection is prior. |
| QueryDet (S5) | Coarse query heads predict small-object locations and cascade sparse queries to finer high-resolution maps, computing detections only at queried positions. | N | N (small-object objectness is not residual/uncertainty) | P (location objectness, not counterfactual gain over the base result) | **Y** (sparse P2/fine-level computation) | N (query count varies with threshold; no max-one image crop) | N | **High collision:** coarse-to-fine query-guided sparse P2 computation and accuracy/speed thresholding are prior; the paper also evaluates a local “Crop Query” feature alternative. |
| ESOD (S6) | Reuses early detector features to predict an objectness mask; AdaSlicer creates feature patches around object centers and suppresses backgrounds; SparseHead detects sparsely at retained centers. | N | N (objectness mask is not first-pass residual/uncertainty) | P (object-seeking score, not base-vs-crop utility) | **Y** (adaptive feature slicing/sparse head) | N (adaptive, variable number of patches) | N | **Very high / strongest collision:** object-seeking, adaptive slicing, background discard, sparse high-resolution detection, and YOLO-family use are all prior. |
| VISO (S7) | Language-guided learned masks at multiple pyramid levels suppress irrelevant regions; dense features are converted to sparse features for VL-PAN/head inference, with threshold-controlled AP/FLOPs. | N (satellite/object-language prior, not road perspective geometry) | N | P (semantic mask relevance, not counterfactual re-observation gain) | **Y** (feature-level sparse pyramid/head) | N (sparsity threshold, not one crop) | N | **Material partial collision:** learned multi-scale masks, sparse pyramid/head execution, and adjustable AP/FLOPs are prior. |

## Method notes and exact collision boundaries

### FPN and ordinary P2

FPN (S1) constructs a top-down pathway with lateral connections so that high-resolution maps receive strong semantics. Its standard pyramid includes P2, and predictions are made independently at pyramid levels. The paper's analysis also observes that a P2-only configuration creates many anchors without solving the problem merely by anchor count. Thus the existence of a P2 level, higher spatial resolution, lateral fusion, and “better small-object detail” are not novel mechanisms.

The current official Ultralytics configuration (S2) is even more direct: its header describes “P2/4 - P5/32 outputs”; the head upsamples and concatenates the P2 backbone feature and ends with `Detect(P2, P3, P4, P5)`. That file is the appropriate structural comparator for F0. It provides no dynamic selector, scene geometry, uncertainty estimator, utility target, per-frame budget, or fallback.

**Claim boundary:** F0 can be reported as a strong leakage-free baseline, but neither “adding P2” nor its dense multi-scale fusion may appear in the originality statement.

### DyFPN

DyFPN (S3) first expands FPN lateral processing with multiple receptive fields, then places a learned gate at each level. The gate uses the current feature representation (global pooling and a small prediction layer; Gumbel-style discrete selection in the paper) to choose whether to execute the more expensive branch. A computational resource loss steers the aggregate cost.

This overlaps the *general* idea of content-conditional, resource-aware pyramid computation. It differs at all decisive F4 points: the decision is level/branch execution rather than a road candidate; the signal is not a frozen detector's residual/uncertainty; the target is not the counterfactual gain from extra image pixels; its cost objective is not a deterministic zero-or-one crop; and the light skip route does not restore an unchanged base detection result.

**Claim boundary:** do not claim conditional FPN computation, resource-constrained feature routing, or a gate/skip pattern. If F4's “hard budget” is weakened into an expected-FLOPs regularizer, the distinction from DyFPN materially collapses.

### FSAF as a selective-pyramid representative

FSAF (S4) makes feature-level assignment content-dependent. For each training instance, it evaluates losses across feature levels and selects the level with the smallest combined classification and regression loss as the online feature-selection target. At inference, however, every level produces outputs and detections are merged; it does not save computation by selecting a level.

This is the closest collision in this slice to *per-instance choice among resolutions*, but the estimand is different. It chooses the currently best feature level for supervised assignment. F4 must instead predict, without inference-time GT, the *marginal benefit of an optional second observation compared with the already available F0 result*.

**Claim boundary:** avoid broad wording such as “learns which resolution is best for each object.” The defensible wording must retain “counterfactual base-versus-re-observation gain under a deployment hard budget.”

### QueryDet

QueryDet (S5) explicitly adds high-resolution pyramid features for small objects while avoiding dense computation. A query head on a coarser map predicts where small objects are likely; thresholded positions are propagated to neighboring positions at the next finer level, and sparse convolution evaluates only those queried locations. The query target is small-object presence/location derived from annotations, not failure residual, predictive uncertainty, or marginal improvement over an existing detection.

The paper tunes a query threshold to trade accuracy against speed; that produces a variable number of sparse locations rather than a deterministic at-most-one crop. Its ordinary coarse detections and sparse fine detections are integrated, so it has no exact-F0 “select none” output path. The paper also includes a “Crop Query” alternative using local feature patches, which makes a generic query/crop formulation especially unsafe even though it is not the frozen image-space one-crop policy.

**Claim boundary:** do not claim query-guided coarse-to-fine computation, sparse P2 evaluation, threshold-based accuracy/compute control, or local crop queries. The proposed work has to show that the selector estimates *incremental outcome value* rather than merely objectness.

### ESOD

ESOD (S6) attacks redundant computation in high-resolution small-object detection by reusing early detector features for object seeking. Its ObjSeeker predicts an objectness mask, AdaSlicer forms feature-map patches around predicted centers and adjusts them to cover clusters, and SparseHead operates on the retained sparse centers. Background regions are discarded. The official repository provides detector-family integrations and explicitly includes YOLO-derived code.

The closest overlap is the selective allocation of high-resolution computation to likely object regions. The decisive differences are that ESOD uses objectness rather than a frozen first pass's residual/uncertainty or counterfactual utility; slicing happens in feature space within a high-resolution detector; the number of patches is adaptive rather than deterministically capped at one fixed image crop; and there is no declared exact-base fallback when slicing is unhelpful. The paper reports that simplified slicing can slightly reduce AP because of truncation, which further distinguishes it from an explicit result fallback.

**Claim boundary:** do not claim object-seeking, adaptive feature slicing, patching around centers, background suppression, or sparse high-resolution heads. If F4's score degenerates to objectness/low confidence and its crop count becomes variable, the proposed mechanism is substantively close to ESOD regardless of renaming.

### VISO (current sparse-pyramid stress test)

VISO (S7) uses language-guided multi-scale masks to focus a satellite detector on object-relevant regions. The learned masks guide sparse conversion of feature maps and sparse inference in later pyramid/head layers; a threshold is varied to balance AP and FLOPs. This is not road geometry and does not estimate the marginal value of re-observing candidate image pixels, but it demonstrates that learned relevance masks plus sparse multi-scale detector execution remain active prior art through ICCV 2025.

**Claim boundary:** a learned mask, relevance map, or thresholded sparse feature pyramid is not novel by itself. The official repository has no visible top-level license, so its code must not be copied or incorporated without an explicit license/permission audit.

## Claims that are forbidden versus potentially defensible

### Forbidden by this slice

- “We introduce P2 / a high-resolution head for small-object detection.”
- “We introduce a dynamic/selective feature pyramid.”
- “We query coarse features to run fine-resolution detection sparsely.”
- “We crop/slice likely object regions or remove background to accelerate a high-resolution detector.”
- “We learn an objectness/relevance mask for sparse high-resolution inference.”
- “We control detection compute with a gate, threshold, or accuracy/FLOPs trade-off.”
- Any suggestion that an internal residual/skip connection is the proposed no-harm fallback.

### Potentially defensible, pending the remaining matrix

- a road-scene-specific geometry-confidence signal used jointly with first-pass detector residual/uncertainty;
- a candidate label and predictor whose estimand is the counterfactual *difference between frozen F0 and the same frame after fixed crop re-observation*, not objectness or expected difficulty;
- a deterministic cardinality constraint of zero or one crop per frame, held identical across ablations;
- an explicit decision to execute no crop when every predicted marginal utility is non-positive; and
- returning the byte-/value-equivalent unmodified F0 detections on that path, rather than fusing a second head and hoping not to hurt.

The combination is not automatically non-obvious merely because each component differs from a single paper. The global audit must still test whether geometry-aware detection, uncertainty-guided computation, and coarse-to-fine re-detection already teach the same combination.

## Code provenance and license gate

| Repository | Status on 2026-08-10 | Reuse implication |
|---|---|---|
| [facebookresearch/Detectron](https://github.com/facebookresearch/Detectron) | Apache-2.0 visible; archived | May be studied under its terms, but there is no reason to transplant its legacy implementation for this design. Preserve notices if any code is reused. |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) / `yolov8-p2.yaml` | AGPL-3.0 stated in the file | Treat as a baseline/configuration dependency under the project's existing license posture; do not copy it into differently licensed code without review. |
| [Mingjian-Zhu/DyFPN](https://github.com/Mingjian-Zhu/DyFPN) | Apache-2.0 visible | Independent implementation is preferred; cite the paper and retain required notices if code is reused. |
| FSAF | No author/official repository found in this search | Do not use unofficial ports as provenance; implement only independently from the paper if ever needed. |
| [ChenhongyiYang/QueryDet-PyTorch](https://github.com/ChenhongyiYang/QueryDet-PyTorch) | MIT visible | Paper citation and license attribution are still required for any reused code; this project does not need its implementation. |
| [alibaba/esod](https://github.com/alibaba/esod) | GPL-3.0 visible; README acknowledges substantial YOLO-derived code; includes third-party tree | **Do not copy or transplant** into this project without a full license and per-file provenance review. Use the paper as conceptual prior only. |
| [joannahuadu/VISO](https://github.com/joannahuadu/VISO) | No top-level license visible in inspected listing; contains `third_party/` | No permission to reuse is established. Treat all code as unavailable for copying unless authors supply a license and third-party terms are audited. |

No paper wording, equations, figures, diagrams, or code structure should be copied. Public ideas must be cited. Any implementation of the surviving gap should be independently written against this project's interfaces, with a later provenance review before release.

## Search log

Searches were run on **2026-08-10 (Asia/Shanghai)**. Queries are recorded verbatim below. Search results were discovery aids; only sources S1–S7 and their official repositories support the findings.

### FPN and official YOLOv8-P2

1. `Feature Pyramid Networks for Object Detection official paper CVPR 2017 official code Detectron`
2. `site:github.com/ultralytics/ultralytics yolov8-p2.yaml official`
3. `site:docs.ultralytics.com YOLOv8 P2 model yaml official`
4. `site:github.com/ultralytics/ultralytics P2/4 yolov8-p2`

### DyFPN and venue/code verification

5. `"Dynamic Feature Pyramid Networks for Object Detection" official paper code authors`
6. `arXiv 2012.00779 official code Dynamic Feature Pyramid Networks GitHub`
7. `site:openaccess.thecvf.com "Dynamic Feature Pyramid Networks"`
8. `site:github.com "Dynamic Feature Pyramid Networks for Object Detection"`
9. `Mingjian Zhu Kai Han Changbin Yu Yunhe Wang DyFPN GitHub`
10. `site:github.com/MingjianZhu DyFPN`
11. `site:github.com/Westlake-AI "DyFPN"`
12. `site:github.com/huawei-noah DyFPN object detection`
13. `"Dynamic Feature Pyramid Networks for Object Detection" BMVC`
14. `"Dynamic Feature Pyramid Networks for Object Detection" conference 2021`
15. `Mingjian Zhu DyFPN publication venue`

### Selective feature pyramids / FSAF

16. `Feature Selective Anchor-Free Module for Single-Shot Object Detection official paper code FSAF`
17. `site:openaccess.thecvf.com "Feature Selective Anchor-Free"`
18. `site:github.com FSAF official implementation Zhu He Savvides`
19. `"selective feature pyramid" object detection official paper`
20. `site:github.com/Chenchen-Zhu FSAF`
21. `site:github.com/chenchenzhu FSAF`
22. `site:github.com/yihui-he FSAF`
23. `site:github.com "Feature Selective Anchor-Free Module"`

### QueryDet

24. `QueryDet official paper code Cascade Sparse Query high-resolution small object detection`
25. `site:openaccess.thecvf.com QueryDet small object detection`
26. `site:arxiv.org QueryDet Cascade Sparse Query`
27. `site:github.com QueryDet official implementation`

### ESOD and publisher metadata

28. `ESOD Efficient Small Object Detection high resolution official paper code`
29. `site:openaccess.thecvf.com ESOD small object detection`
30. `site:arxiv.org ESOD efficient small object detection`
31. `site:github.com ESOD small object detection official`
32. `"ESOD: Efficient Small Object Detection on High-Resolution Images" IEEE Transactions on Image Processing DOI`
33. `site:ieeexplore.ieee.org "ESOD: Efficient Small Object Detection"`
34. `10.1109/TIP.2024.3501853` (domain restriction: `ieeexplore.ieee.org`)
35. `"3501853" ESOD` (domain restriction: `ieeexplore.ieee.org`)
36. `"ESOD: Efficient Small Object Detection on High-Resolution Images"` (domain restriction: `ieee.org`)

### 2025 sparse-pyramid stress test / VISO

37. `site:openaccess.thecvf.com ICCV 2025 VISO Accelerating In-orbit Object Detection Language-Guided Mask Learning official code`
38. `"VISO: Accelerating In-orbit Object Detection" official code`
39. `site:github.com "VISO" "In-orbit Object Detection"`
40. `site:openaccess.thecvf.com/content/ICCV2025 VISO Accelerating In-orbit Object Detection Language-Guided Mask Learning Sparse Inference`
41. `site:github.com/joannahuadu/VISO license LICENSE`

## Gaps and failed searches

1. **DyFPN venue:** the original paper and author repository identify it as arXiv:2012.00779. Searches for a CVF/BMVC/conference version did not locate a primary peer-reviewed record. It is therefore classified here as an original preprint, not silently upgraded to a conference paper.
2. **FSAF code:** no repository owned by the named authors or an explicitly designated official organization was located with queries 18 and 20–23. Several third-party ports surfaced but were excluded under the source policy.
3. **ESOD publisher page:** the DOI was identified, but the IEEE landing page was blocked/unavailable in this environment. Method claims use the author manuscript and official Alibaba repository; no inaccessible full text is treated as read.
4. **VISO license:** the official repository was accessible, but its top-level listing contained no `LICENSE` file or visible GitHub license badge. Absence of a visible license is recorded as “no reuse permission established,” not as a claim about the authors' intent.
5. **Mutable repository heads:** official GitHub `main`/`master` pages were inspected as they existed on the search date. A publication-grade archive should later pin source commits or preserve PDFs/config snapshots and hashes.
6. **Literal-term searches:** absence of words such as “uncertainty,” “geometry,” or “fallback” was not used as proof of novelty. Differences above come from the described mechanisms, signals, targets, and inference constraints.
7. **Scope boundary:** this note cannot determine whether road-geometry confidence, uncertainty-guided re-observation, zoom/crop detection, NWD/RFLA, or early-exit work closes the surviving gap. The global gate remains pending those primary-source reviews.

## Mandatory counter-review

Before treating the gap as publishable, attempt to falsify it with the following checks:

1. **Aggregation/obviousness risk:** even if no single paper contains all five components, the combination of known road priors, known uncertainty routing, QueryDet/ESOD-style selection, and a cardinality constraint may be an obvious aggregation. A global cross-category matrix is mandatory.
2. **Estimand-collapse risk:** if the F4 target is implemented as objectness, low confidence, entropy alone, or “contains a small object,” it collapses toward QueryDet/ESOD/VISO. Training labels must measure the candidate's causal/matched base-versus-re-observation utility and must be auditable for leakage.
3. **Budget-collapse risk:** if the budget becomes an expected FLOPs penalty, variable crop count, or threshold-chosen average compute, the distinction from DyFPN/QueryDet/ESOD weakens materially. The deterministic zero-or-one constraint must be enforced and reported per frame.
4. **Fallback-collapse risk:** feature skips, NMS fusion, and confidence gating are not an exact base-output fallback. Tests must prove that the non-positive path emits the unmodified F0 result and does not silently alter other detections.
5. **Evaluation risk:** a selective method can improve small/far AP while increasing false positives, latency variance, or harm outside selected regions. AP_R40, small/far strata, false positives, realized budget, FLOPs, and latency must all be reported; no favorable-metric cherry-picking.
6. **Result risk:** no source review establishes that the proposed mechanism will achieve +4.0 or +5.0 AP. O1 and the frozen ablations may produce zero or negative results, which must be retained.

## Final scoped decision

- **Defensible gap:** **yes, provisionally**, only for the joint counterfactual-VoR decision problem and exact hard-budget/fallback semantics described above.
- **Material collision:** **yes**, strongest with ESOD and QueryDet; exact baseline collision with official YOLOv8-P2/FPN; additional partial collisions with DyFPN, FSAF, and VISO.
- **Insufficient evidence:** **yes for global clearance** because this is one literature slice and the other required categories are outside its ownership.
- **Does this file authorize coding?** **No.** It supports `PASS WITH CONDITIONS` for this slice only. Sol/ultra must merge all collision rows, resolve any material cross-category isomorphism, and freeze the final falsifiable claim before implementation begins.
