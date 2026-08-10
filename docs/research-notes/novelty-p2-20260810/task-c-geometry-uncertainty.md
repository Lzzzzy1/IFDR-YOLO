---
task_id: task-c-geometry-uncertainty
owner: /root/novelty_geometry_uncertainty
status: complete
as_of: 2026-08-10
search_date: 2026-08-10
scope: perspective/road-geometry-aware detection; uncertainty-guided computation; dynamic inference/early exit; NWD/RFLA
evidence_policy: original papers and author/official repositories only
implementation_or_server_actions: none
skills_used: deep-research; academic-writing; fact-checker
---

# P2 novelty collision notes: geometry, uncertainty, dynamic inference, NWD/RFLA

## 1. Executive verdict

**Assigned-slice verdict: PROVISIONAL DEFENSIBLE GAP; CONDITIONALLY ALLOW ENCODING.** I did not find, in the primary literature reviewed below as of 2026-08-10, one method that jointly does all five of the following:

1. represents **road-geometry reliability/confidence**, rather than merely using a deterministic perspective prior;
2. uses first-pass detection residual/uncertainty to predict the **candidate-specific incremental detection gain caused by an actually observed high-resolution counterfactual**;
3. ranks local candidates under an **exact hard per-frame crop budget**;
4. performs local high-resolution re-observation; and
5. has an explicit **no-harm fallback** that retains the coarse result when the re-observation is unsafe or unhelpful.

This is not proof of worldwide priority. It is a bounded search result. The final project-wide novelty gate must also merge the separate ESOD/ZoomDet/coarse-to-fine/P2 searches; a structural collision there can still change this verdict to NO-GO.

The gap is narrow because several component-level collisions are already strong:

- **Uzkent et al. (WACV 2020)** already learn where to acquire/use high-resolution patches with an accuracy-versus-HR-cost reward.
- **Adaptive Feeding (ICCV 2017)** already trains a router from the observed per-image performance difference between fast and accurate detectors, including SSD300 versus SSD500.
- **DynamicDet (CVPR 2023)** already supervises an image router with the loss difference between its first and second detectors.
- **AdaDet (TCDS 2024)** already uses per-object localization uncertainty, aggregated to image reliability, to allocate detection computation through early exit.
- **GUPNet/GUPNet++ (ICCV 2021/TPAMI 2024)** already combine perspective geometry and per-object uncertainty/confidence in KITTI road-object detection.
- **QR-DETR (ACCV 2024)** already performs learned per-query dynamic computation/early exit.

Therefore the project must not claim novelty for “geometry + uncertainty,” “uncertainty-guided computation,” “predicting whether a more accurate/higher-resolution detector helps,” “adaptive high-resolution patches,” “dynamic routing,” or “candidate/query-level routing” by themselves.

### Falsifiable contribution statement

> Under a fixed per-frame re-observation budget, test whether a selector trained on candidate-level high- versus low-resolution detection deltas can use calibrated road-geometry reliability and first-pass detection residual uncertainty to improve KITTI Pedestrian/Cyclist Moderate macro AP_R40, while an explicit fallback bounds harm to unselected and non-improving detections.

This statement is falsified if the selector does not outperform leakage-free plain P2 under the frozen image list/seed/budget/evaluator, if the measured gain comes from more total computation rather than allocation, or if fallback does not reduce a predeclared harm metric.

### Encoding gate for this slice

Encoding is scientifically defensible only if all of the following are present in the frozen specification:

- the supervision target is an **actual paired candidate-level low-resolution versus high-resolution detection delta**, not confidence, object size, difficulty, or road position renamed as “value”;
- selection obeys an exact per-frame `K` (or an exactly specified equivalent resource cap), not an average crop rate or soft penalty alone;
- geometry is accompanied by an observable reliability/confidence variable and a geometry-disabled control;
- uncertainty/residual features are tested independently from geometry;
- fallback has a predeclared decision rule and harm outcome; “merge both outputs and hope” is not a fallback claim;
- the final global collision matrix finds no structurally equivalent ESOD/ZoomDet/coarse-to-fine prior.

If the implementation collapses to a perspective prior, confidence-threshold crop, image-level easy/hard router, soft HR-cost regularizer, or generic uncertainty refinement, the result is **NO-GO as an originality claim**, even if its AP improves.

## 2. Frozen comparison axes

The matrix uses the following exact axes. A check means primary evidence for the component; `partial` means a related but materially weaker/different form; `no` means it was not found in the paper material reviewed.

| Axis | Frozen project meaning |
|---|---|
| **G** | road/perspective geometry with an explicit confidence or reliability estimate |
| **U** | detection residual/localization/class uncertainty from the first pass |
| **ΔHR** | target or policy signal tied to incremental improvement under a real high-resolution observation, at candidate level |
| **K** | exact hard local re-observation budget per frame |
| **F** | explicit no-harm fallback retaining the coarse detection when refinement is unsafe/unhelpful |

## 3. Novelty collision matrix

| Prior work (primary source) | Source/accessibility | Official code/license as of search date | Method core | G | U | ΔHR | K | F | Component overlap and decisive difference | Collision risk |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|
| **Perspective Aware Road Obstacle Detection**, Lis et al., RA-L 2023 ([paper](https://arxiv.org/abs/2210.01779), [accepted PDF](https://infoscience.epfl.ch/record/301426/files/PerspectiveAwareRoadObstacleDetection.pdf)) | Original paper; open arXiv and EPFL accepted manuscript | No author/official repository located by exact-title/author searches; license N/A | From camera calibration and a planar-road model, computes at every pixel the apparent pixel width of a hypothetical 1 m object; uses the perspective scale map for synthetic obstacle injection and at multiple FPN decoder levels. Intended to detect distant small road obstacles and suppress nearby small-irregularity false positives. | partial | no | no | no | no | Direct collision with “road perspective helps small/far detection and false-positive control.” It uses a deterministic calibrated map, not learned geometry confidence, counterfactual resolution value, compute allocation, or fallback. | **HIGH** for geometry claims; lower for full F4 |
| **Geometry-Aware Video Object Detection for Static Cameras**, Xu, Xie, Zisserman, BMVC 2019 ([paper](https://arxiv.org/abs/1909.03140), [author PDF](https://www.robots.ox.ac.uk/~vgg/publications/2019/xu19/xu19.pdf)) | Original paper; open arXiv and Oxford author copy | No author/official repository located; license N/A | Uses object scale as pseudo-depth/2.5D scene geometry and conditions dynamic multi-scale feature selection/fusion in a video detector. | partial | no | no | no | no | Direct collision with “geometry-aware dynamic multi-scale selection.” It targets static-camera video features, not moving-road-camera crop value or a measured HR counterfactual. | **HIGH** |
| **FoveaNet: Perspective-Aware Urban Scene Parsing**, Li et al., ICCV 2017 ([CVF paper](https://openaccess.thecvf.com/content_ICCV_2017/html/Li_FoveaNet_Perspective-Aware_Urban_ICCV_2017_paper.html), [arXiv](https://arxiv.org/abs/1708.02421)) | Original conference paper; open CVF/arXiv | No author/official repository located; license N/A | Estimates scene perspective from contextual objects and compensates for projective scale in urban scene parsing. | partial | no | no | no | no | Establishes older perspective-aware urban-scene precedent. It is parsing, not candidate-level detection re-observation or budgeted computation. | **MEDIUM** |
| **GUPNet** and **GUPNet++**, Lu et al., ICCV 2021 / TPAMI 2024 ([GUPNet paper](https://arxiv.org/abs/2107.13774), [CVF record](https://openaccess.thecvf.com/content/ICCV2021/html/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.html), [GUPNet++ paper](https://arxiv.org/abs/2310.15624)) | Original papers; open arXiv/CVF metadata | [GUPNet official repo](https://github.com/SuperMHP/GUPNet), MIT; [GUPNet++ official repo](https://github.com/SuperMHP/GUPNet_Plus), MIT | Probabilistic perspective-geometry projection models error amplification in monocular depth, yielding per-object depth distributions/uncertainty and reliable 3D confidence; GUPNet++ propagates geometry uncertainty downstream. | partial | check | no | no | no | **Very strong terminology/component collision:** geometry and uncertainty are already jointly modeled for KITTI objects. Their uncertainty concerns projected 3D depth quality, not the causal gain of local HR pixels, and they do not route computation. | **VERY HIGH**; forbids “first geometry + uncertainty detector” |
| **Adaptive Feeding**, Zhou, Gao, Wu, ICCV 2017 ([CVF paper](https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Adaptive_Feeding_Achieving_ICCV_2017_paper.html), [arXiv](https://arxiv.org/abs/1707.06399)) | Original conference paper; open CVF/arXiv | No author/official implementation located; license N/A | Runs a lightweight detector, extracts top-proposal class/confidence/box/size features, and uses an SVM to route each image to a fast or accurate detector. Training “hard” labels are derived from the observed per-image performance difference; an explicit example is SSD300 versus SSD500. Cost-sensitive training/thresholds trade speed for accuracy. | no | partial | **partial** | no | no | Closest precedent for learning whether a higher-resolution/more accurate route helps from actual detector outcomes. Difference is **image-level binary routing**, not candidate-level local gain/ranking; no road geometry, exact crop budget, or coarse-result fallback. | **CRITICAL** |
| **Efficient Object Detection in Large Images Using Deep Reinforcement Learning**, Uzkent, Yeh, Ermon, WACV 2020 ([CVF paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.pdf), [arXiv](https://arxiv.org/abs/1912.03966)) | Original paper; open CVF/arXiv | Paper promises release, but no author/official repository was located by exact-title/author searches; license N/A | Two policy networks perform coarse and fine search over large images, choosing patches/subpatches for high-resolution acquisition/detection. Reinforcement reward trades detection accuracy against HR acquisition/use; evaluated on xView and also reports Caltech Pedestrian efficiency. | no | no | partial | partial | no | Direct collision with adaptive HR patch acquisition and accuracy-cost routing. However the action is spatial-policy based, the budget is a soft reward/average sampled-area outcome rather than exact per-frame `K`, and the paper does not learn a candidate-specific paired HR improvement target or a no-harm merge. This row overlaps the separate crop/zoom review and must be reconciled there. | **CRITICAL** for “adaptive HR crop” claims |
| **DynamicDet**, Lin et al., CVPR 2023 ([CVF paper](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_DynamicDet_A_Unified_Dynamic_Architecture_for_Object_Detection_CVPR_2023_paper.html), [arXiv](https://arxiv.org/abs/2304.05552)) | Original conference paper; open CVF/arXiv | [Official repo](https://github.com/VDIGPKU/DynamicDet); custom/non-OSI notice: free for academic research, commercial use requires authorization | Cascades two detectors and learns a multi-scale-feature router. The router’s difficulty signal is tied to the per-image detection-loss difference between the first and second detector; thresholds produce variable-speed inference. | no | partial | partial | no | no | Strong collision with detection-residual/loss-based allocation of extra detector computation. It is image-level and routes to a larger detector, not local HR candidate selection or AP-gain supervision; no geometry/fallback. | **CRITICAL** |
| **AdaDet: An Adaptive Object Detection System Based on Early-Exit Neural Networks**, Yang et al., TCDS 2024 ([IEEE/DOI](https://doi.org/10.1109/TCDS.2023.3274214)) | Original publisher record/abstract accessible; full text was not accessible in this environment | No author/official repository located; license N/A | Models localization predictions as stochastic variables, estimates per-object detection uncertainty, aggregates it with an entropy criterion into whole-image reliability, and early-exits high-confidence images while hard images finish the graph. | no | **check** | no | no | no | Directly invalidates a generic “detection uncertainty guides computation” novelty claim. Its uncertainty estimates current-result reliability, not incremental benefit caused by HR re-observation; routing is whole-image depth, not local crops. | **CRITICAL** |
| **QR-DETR: Query Routing for Detection Transformer**, Senthivel, Vu, ACCV 2024 ([CVF record](https://openaccess.thecvf.com/content/ACCV2024/html/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.html), [PDF](https://openaccess.thecvf.com/content/ACCV2024/papers/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.pdf)) | Original CVF record/PDF indexed; direct fetch returned 403 during this audit | No author/official repository located; license N/A | Learned entry-exit routing decides per object query whether to traverse or skip each DETR decoder layer; selected queries receive more processing and are scattered back to their positions. | no | partial | no | no | no | Important candidate-level dynamic-compute collision. Its unit is a latent DETR query/layer, not an image crop or measured value of HR pixels; no road geometry, HR counterfactual, or crop budget. | **HIGH** |
| **Uncertainty-Aware Gradient Stabilization for Small Object Detection (UGS)**, Sun et al., ICCV 2025 ([CVF paper](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html), [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.pdf)) | Original conference paper; open CVF | No author/official repository located; license N/A | Reformulates localization for stable gradients, adds uncertainty minimization, and refines high-uncertainty small-object regions through perturbation-based refinement. | no | check | no | no | no | Direct collision with “uncertainty-guided refinement of small-object regions.” It does not acquire new high-resolution evidence or estimate its counterfactual benefit; no exact crop budget/fallback. | **HIGH** |
| **AnytimeYOLO / You Only Look Once at Anytime**, Kuhse et al., 2025 ([arXiv](https://arxiv.org/abs/2503.17497)) | Original open preprint | No author/official repository linked or located; license N/A | Adds and optimizes early exits for interruptible/anytime YOLO inference. | no | partial | no | no | no | Dynamic/early-exit YOLO is already prior art; the exit-depth problem is different from choosing local high-resolution evidence. | **MEDIUM-HIGH** |
| **DualRes: Production-ready Dynamic Object Detection**, Hassani et al., WACV 2026 ([CVF record](https://openaccess.thecvf.com/content/WACV2026/html/Hassani_DualRes_Production-ready_Dynamic_Object_Detection_WACV_2026_paper.html), [DOI](https://doi.org/10.1109/WACV61042.2026.00757)) | Original CVF abstract indexed; full PDF retrieval failed in this environment | No author/official repository located; license N/A | Accessible abstract establishes a dynamic detector, a routing-efficacy metric, deployment/ONNX analysis, and Pareto models. | unknown | unknown | unknown | unknown | unknown | Freshness warning only. The title must not be interpreted as proof of input-resolution routing; accessible primary text was insufficient for component-level adjudication. | **UNRESOLVED**; obtain full paper before a priority claim |
| **A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (NWD)**, Wang et al., 2021/2022 ([paper](https://arxiv.org/abs/2110.13389)) | Original open paper | [Official repo](https://github.com/jwwangchn/NWD), Apache-2.0 | Models boxes as 2-D Gaussians; normalized Wasserstein distance gives a smooth, scale-insensitive alternative to IoU for tiny-box matching and can be used in assignment, loss, and NMS. | no | no | no | no | no | Strong tiny-box robustness baseline, but it neither observes new pixels nor allocates inference computation. If used, it must be cited as a public assignment/loss metric, not called a resolution mechanism. | **LOW** for F4; **HIGH** if Gaussian tiny-box matching is claimed |
| **RFLA: Gaussian Receptive Field Based Label Assignment for Tiny Object Detection**, Xu et al., ECCV 2022 ([paper](https://arxiv.org/abs/2208.08738), [ECVA PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf)) | Original open paper | [Official repo](https://github.com/Chasel-Tsui/mmdet-rfla), MIT | Models feature receptive fields and ground truths as Gaussians, introduces Receptive Field Distance and Hierarchical Label Assignment to give tiny objects better positive samples. | no | no | no | no | no | Assignment baseline, not a selector or re-observer. Official repo explicitly notes that the tiny-object issue is not a generic medium/large-object method. | **LOW** for F4; **HIGH** if Gaussian RF assignment is claimed |

## 4. Strongest collision analysis

### 4.1 High-resolution value is not a blank concept

Adaptive Feeding and Uzkent et al. make the broadest tempting claim unavailable. Adaptive Feeding derives an image-level route label from the empirical accuracy difference between two detectors and explicitly evaluates SSD300 versus SSD500. Uzkent et al. select locations for HR imagery/detection using an accuracy-versus-HR-use objective. Thus “learn when/where higher resolution is useful” is prior art.

The remaining defensible distinction must be demonstrated, not asserted:

- **unit:** detection candidate rather than whole image or generic spatial cell;
- **target:** paired change in matched detection outcome attributable to the actual HR crop, rather than size, confidence, loss, or a soft reward proxy;
- **constraint:** exact per-frame `K`, not an average HR rate;
- **context:** calibrated road-geometry reliability plus detection residual uncertainty;
- **safety:** retain coarse evidence on predicted/observed harm.

### 4.2 Geometry plus uncertainty is already occupied

GUPNet/GUPNet++ are especially dangerous because they use the exact words “geometry” and “uncertainty,” produce per-object reliable confidence, run on KITTI, and report pedestrian/cyclist results in the official repository. Perspective Aware Road Obstacle Detection and the static-camera geometry-aware detector separately occupy perspective maps and geometry-conditioned multi-scale selection. The project contribution cannot be framed as adding road geometry and uncertainty to detection. It must be framed and tested as estimating **incremental value of acquiring/processing higher-resolution evidence under a fixed allocation problem**.

### 4.3 Uncertainty-guided computation is already occupied

AdaDet directly uses detection uncertainty for early exit. DynamicDet uses first-versus-second detector loss difference. QR-DETR routes individual object queries. Consequently, uncertainty or residual features are not novel merely because they drive compute. The project must show that these features predict a different estimand: candidate-level `ΔHR`, with calibration/ranking quality reported independently of downstream AP.

### 4.4 NWD/RFLA are controls, not substitutes for the mechanism

NWD and RFLA can improve tiny-object optimization without any second observation. They are appropriate strong baselines or orthogonal controls. Adding either to F4 would not establish the value-of-resolution mechanism and would confound attribution unless placed in a separate factorial experiment. Third-party code should not be copied into the project without preserving license and attribution; independent implementation of only the needed public equation is simpler and safer.

## 5. Primary source and code/license register

| ID | Primary evidence | Access status on 2026-08-10 | Official code/license evidence |
|---|---|---|---|
| S1 | Lis et al., [arXiv:2210.01779](https://arxiv.org/abs/2210.01779); [EPFL accepted PDF](https://infoscience.epfl.ch/record/301426/files/PerspectiveAwareRoadObstacleDetection.pdf) | Open | No official repo located; paper says code would be made public, which is not equivalent to an available licensed release. |
| S2 | Xu et al., [arXiv:1909.03140](https://arxiv.org/abs/1909.03140); [Oxford author PDF](https://www.robots.ox.ac.uk/~vgg/publications/2019/xu19/xu19.pdf) | Open | No official repo located. |
| S3 | Li et al., [CVF ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Li_FoveaNet_Perspective-Aware_Urban_ICCV_2017_paper.html) | Open | No official repo located. |
| S4 | Lu et al., [CVF ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.html) | Open | [Author repo](https://github.com/SuperMHP/GUPNet), MIT. Repo warns that released code contains tricks not described in the paper and some nondeterminism. |
| S5 | Lu et al., [arXiv:2310.15624](https://arxiv.org/abs/2310.15624) | Open | [Author repo](https://github.com/SuperMHP/GUPNet_Plus), MIT. Repo records a known beta-NLL implementation issue; do not treat repository numbers as automatically paper-identical. |
| S6 | Zhou et al., [CVF ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Adaptive_Feeding_Achieving_ICCV_2017_paper.html) | Open | No official repo located. |
| S7 | Uzkent et al., [CVF WACV 2020 PDF](https://openaccess.thecvf.com/content_WACV_2020/papers/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.pdf) | Open | No official repo located despite paper release statement. |
| S8 | Lin et al., [CVF CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_DynamicDet_A_Unified_Dynamic_Architecture_for_Object_Detection_CVPR_2023_paper.html) | Open | [Author repo](https://github.com/VDIGPKU/DynamicDet); README grants free academic research use and requires commercial authorization. No standard permissive license was displayed. |
| S9 | Yang et al., [DOI 10.1109/TCDS.2023.3274214](https://doi.org/10.1109/TCDS.2023.3274214) | Publisher abstract/metadata accessible; full text not retrieved | No official repo located. |
| S10 | Senthivel and Vu, [CVF ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Senthivel_QR-DETR__Query_Routing_for_Detection_Transformer_ACCV_2024_paper.html) | Indexed primary record/PDF; direct fetch returned 403 during audit | No official repo located. |
| S11 | Sun et al., [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html) | Open | No official repo located. |
| S12 | Kuhse et al., [arXiv:2503.17497](https://arxiv.org/abs/2503.17497) | Open preprint | No official repo located. |
| S13 | Hassani et al., [CVF WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Hassani_DualRes_Production-ready_Dynamic_Object_Detection_WACV_2026_paper.html) | Abstract indexed; full paper not retrieved | No official repo located. |
| S14 | Wang et al., [arXiv:2110.13389](https://arxiv.org/abs/2110.13389) | Open | [Author repo](https://github.com/jwwangchn/NWD), Apache-2.0. |
| S15 | Xu et al., [arXiv:2208.08738](https://arxiv.org/abs/2208.08738); [ECVA PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf) | Open | [Author repo](https://github.com/Chasel-Tsui/mmdet-rfla), MIT. |

No third-party implementation, survey, blog, aggregator, or code mirror is used as evidence in this note. Search results from such pages were used only to navigate to the original paper/repository and were then excluded.

## 6. Search log

All searches below were run on **2026-08-10**. Exact-title searches were paired with author/lab GitHub searches when checking official code. Negative search results mean “not located in this bounded search,” not “does not exist.”

### Perspective/road geometry

- `site:openaccess.thecvf.com perspective geometry aware pedestrian detection road scene small objects`
- `site:arxiv.org geometry-aware road object detection perspective small object detection`
- `site:openaccess.thecvf.com perspective map uncertainty object detection road geometry confidence`
- `site:arxiv.org "geometry uncertainty" object detection road scene perspective`
- `"Perspective Aware Road Obstacle Detection" official code github`
- `"Geometry-Aware Video Object Detection for Static Cameras" official code github`
- `"FoveaNet: Perspective-Aware Urban Scene Parsing" official code github`
- `"Geometry Uncertainty Projection Network" official code github license`

### Uncertainty-guided computation and dynamic/early-exit detection

- `site:openaccess.thecvf.com uncertainty guided dynamic computation object detection adaptive resolution`
- `site:arxiv.org object detection uncertainty guided computation crop budget high resolution`
- `object detection early exit bounding box uncertainty criterion paper official`
- `"AdaDet" "3274214" pdf`
- `"Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors" paper`
- `"Adaptive Feeding" object detector github official`
- `"DynamicDet: A Unified Dynamic Architecture for Object Detection" GitHub license`
- `"QR-DETR" "Query Routing for Detection Transformer" official GitHub`
- `"AnytimeYOLO" original paper official code`
- `"Uncertainty-Aware Gradient Stabilization for Small Object Detection" official code`
- `site:github.com "Uncertainty-Aware Gradient Stabilization"`
- `"DualRes: Production-ready Dynamic Object Detection" code github`
- `"Efficient Object Detection in Large Images Using Deep Reinforcement Learning" arxiv official code`
- `site:github.com Uzkent Efficient Object Detection Large Images reinforcement learning`

### NWD/RFLA

- `"A Normalized Gaussian Wasserstein Distance for Tiny Object Detection" official code github`
- `"RFLA" tiny object detection official code github paper`
- `site:github.com/SuperMHP/GUPNet license MIT`
- `site:github.com/SuperMHP/GUPNet_Plus license MIT`
- `site:github.com/VDIGPKU/DynamicDet license academic research`
- `site:github.com/jwwangchn/NWD license Apache`

### Exact-combination / disconfirmation queries

- `"value of resolution" object detection geometry uncertainty crop budget`
- `"counterfactual" "high-resolution" object detection uncertainty`
- `"hard budget" crop object detection geometry uncertainty`
- `"no-harm" object detection re-observation`
- `site:openaccess.thecvf.com object detection high resolution crop uncertainty routing budget`
- `site:arxiv.org object detection high resolution crop uncertainty routing budget geometry`
- `site:openaccess.thecvf.com object detection counterfactual resolution routing`
- `site:arxiv.org object detection "hard budget" "high resolution" crop`

These exact-combination queries returned no primary paper matching all five frozen axes. They did surface component precedents, notably GUPNet, Uzkent et al., DynamicDet, QR-DETR, and unrelated uses of “counterfactual”; therefore the absence of an exact phrase is not treated as novelty proof.

## 7. Claims that must be prohibited

The paper/design must not say or imply any of the following:

- “We are the first to use perspective/road geometry for small or far road-object detection.”
- “We are the first to combine geometry and uncertainty in road-object/KITTI detection.”
- “We are the first to use detection uncertainty to allocate computation.”
- “We are the first to learn whether a more accurate or higher-resolution detector will help.”
- “We are the first to dynamically select object candidates/queries for more computation.”
- “We are the first to select high-resolution patches for efficient object detection.”
- “NWD/RFLA/P2 is our value-of-resolution mechanism.”

Permissible wording must describe a tested combination and estimand, for example: “candidate-level measured resolution delta under an exact per-frame crop budget, conditioned on road-geometry reliability and first-pass residual uncertainty, with an explicit fallback.” Even this wording remains conditional on the project-wide collision review and experimental evidence.

## 8. Missing evidence, adversarial review, and unresolved risks

1. **Search cannot establish absolute absence.** Index terms vary, 2026 papers may not be indexed consistently, patents and unpublished industrial systems were outside scope, and exact-title GitHub searches can miss renamed repositories.
2. **DualRes remains unresolved.** Only the primary abstract/metadata was accessible; the title alone cannot support a claim that it routes input resolutions. The full paper should be obtained before any “first dynamic resolution” statement.
3. **The closest high-resolution precedents straddle another review slice.** Uzkent et al. and likely ESOD/ZoomDet/coarse-to-fine methods must be compared at implementation granularity in the final matrix. This task alone cannot grant the global GO.
4. **Candidate-level `ΔHR` needs a strict matching definition.** If labels are generated using ground truth at selector inference, split-shared crop outcomes, or validation/test results, the scientific gap is irrelevant because the experiment leaks. If candidate identity changes across resolutions, the matching policy and negative outcomes must be frozen before training.
5. **Uncertainty is not automatically value.** High entropy can indicate ambiguity that higher resolution cannot fix; low confidence can also reflect occlusion/truncation. The selector must report ranking/calibration against observed `ΔHR`, not only correlation with object size or AP.
6. **Geometry reliability may be a renamed perspective prior.** It needs its own target/calibration evidence and failure modes (non-planar road, hills, pitch change, junctions, calibration error). Otherwise S1/S2/GUPNet absorb the claim.
7. **No-harm is ambiguous without a metric.** A deterministic merge rule is not itself novel or safe. Predeclare whether harm means candidate matched loss, new false positive, loss of an existing true positive, class flip, overall AP drop, or a bound on a paired per-image statistic, and include fallback-off/always-replace controls.
8. **Hard budget must be literal.** Average FLOPs, a crop penalty, or an expected route rate repeats prior soft-cost routing. Report actual crops/frame distribution, worst-case FLOPs/latency, and any budget violation count.
9. **AP alone can hide damage.** The contribution question requires Moderate macro AP_R40 plus small/far strata, false positives, recall, crops/frame, FLOPs, and latency. A favorable stratum cannot replace the frozen primary endpoint.
10. **Code reuse risk.** The reviewed official repositories have mixed licenses. DynamicDet is not permissively licensed for unrestricted reuse. No paper phrasing, diagrams, or unlicensed code should be copied; public ideas should be cited and the project mechanism implemented independently.

## 9. Final decision record for parent matrix

- **Defensible gap within this slice:** yes, but only the five-part candidate-level value-of-resolution allocation problem described in Section 1.
- **Strongest collision points:** Uzkent HR patch policy; Adaptive Feeding detector-performance-delta router; DynamicDet detector-loss-delta router; AdaDet uncertainty-driven detection compute; GUPNet geometry-plus-uncertainty; QR-DETR per-query routing.
- **Is coding allowed from this slice alone?** **Conditional yes** for an independent implementation of the frozen mechanism, after the parent matrix confirms no exact crop/zoom/ESOD/ZoomDet collision. It is not permission to claim novelty or positive performance.
- **Immediate NO-GO triggers:** image-level routing; soft average HR budget only; confidence/size/perspective threshold relabeled as value; no actual paired HR target; no fallback control; or a structurally equivalent method found in the remaining primary-literature slices.
- **Experimental result status:** no result was generated or inferred in this task. Positive, zero, and negative outcomes remain possible and must be retained with hashes under the project evidence gate.

