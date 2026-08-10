# Novelty Collision Matrix and Decision

- Search date / AS_OF: 2026-08-10 (Asia/Shanghai)
- Evidence policy: original papers and author/official repositories only
- Full query logs and source notes:
  - `task-a-p2-pyramids.md`
  - `task-b-zoom-redetect.md`
  - `task-c-geometry-uncertainty.md`
- This record is a novelty decision, not an experiment result or a claim of absolute priority.

## 1. Decision

**Broad value-of-resolution claim: NO-GO. Narrow risk-constrained formulation: PROVISIONAL GO for independent implementation and falsification.**

The following ideas are established prior art and cannot be claimed as original: P2/FPN high-resolution heads; dynamic or sparse feature pyramids; coarse-to-fine queries; adaptive slicing; crop/zoom re-detection; geometry/perspective-driven pixel allocation; uncertainty-driven compute; and prediction of coarse-versus-fine detection gain. In particular, Dynamic Zoom-In (CVPR 2018) and Uzkent et al. (WACV 2020) directly invalidate any claim that predicting the detection value of extra resolution is itself new.

The bounded search did not find one primary source that jointly uses all of the following as a single candidate-level decision: paired benefit **and harm** outcomes from one actual high-resolution re-observation; calibrated current-frame road-geometry reliability; first-pass detection residual uncertainty; a literal zero-or-one crop action space; a risk/lower-bound abstention rule; and exact F0 output when the rule abstains. This narrow conjunction is a defensible research gap, not proof of being first.

### Falsifiable contribution statement

> Under an identical zero-or-one high-resolution crop budget, does a selector that estimates separate paired benefit and harm outcomes of one additional observation, conditions their reliability on current-frame road geometry and first-pass detection residual uncertainty, and abstains to the exact F0 output when a calibrated lower-bound net benefit is non-positive, improve KITTI Pedestrian/Cyclist Moderate macro AP_R40 over geometry-only, uncertainty-only, and prior-art-style expected-gain selection without increasing registered false-positive or near/medium-object harm?

The statement is falsified if the expected-gain F3 matches F4, F4 fails the +2.0 screening target over F0, a registered safety slice degrades, or the effect is not directionally consistent over three seeds.

## 2. Frozen component definitions after collision review

| Variant | Definition | Unique causal question |
|---|---|---|
| F0 | Leakage-free plain YOLOv8m-P2; no re-observation | What does a strong P2 baseline achieve? |
| F1 | Geometry-only selection under the same zero-or-one crop budget | Does calibrated road geometry add useful allocation evidence? |
| F2 | Detection-residual/uncertainty-only selection under the same budget | Does first-pass uncertainty identify useful extra pixels? |
| F3 | Prior-art-style expected coarse-versus-fine gain selector under the same hard budget and candidate pool | Is ordinary expected-gain zoom sufficient? |
| F4 | Separate benefit/harm outcome prediction, geometry-reliability conditioning, and calibrated lower-bound abstention to exact F0 | Does risk-constrained joint evidence outperform F1-F3 without harm? |

F3 is a strong prior-art control, not a claimed contribution. F4 may not be replaced by a weighted average of size, confidence, entropy, or geometry thresholds.

## 3. Collision matrix

Legend: `same` = substantive mechanism collision; `partial` = related but different estimand/granularity/constraint; `different` = not present in the primary source.

| Prior work | Core prior mechanism | Collision with frozen design | What remains different | Maximum risk |
|---|---|---|---|---|
| FPN, CVPR 2017 | Dense semantic pyramid including high-resolution P2 | `same`: P2/high-resolution multi-scale head | No optional observation, value estimate, hard crop budget, or fallback | P2 cannot be a contribution |
| Official Ultralytics YOLOv8-P2 | Dense P2/P3/P4/P5 detector | `same`: exact structural F0 precedent | Project only adapts, trains, and audits it | Exact F0 collision |
| DyFPN, 2020 | Feature-dependent conditional pyramid branches with resource objective | `partial`: dynamic/resource-aware pyramid | Feature branch rather than one image crop; no paired harm outcome or exact base fallback | “Dynamic P2/FPN” wording prohibited |
| QueryDet, CVPR 2022 | Coarse queries guide sparse high-resolution P2 computation | `same`: coarse-to-fine selective high-resolution detection | Objectness queries, variable locations, no paired benefit/harm or one-crop fallback | High |
| ESOD, TIP 2024/2025 | Object-seeking mask, adaptive feature slicing, sparse high-resolution head | `same`: selective sparse high-resolution small-object detection | Objectness/feature slicing, variable patches, no exact F0 risk abstention | Very high |
| Dynamic Zoom-In, CVPR 2018 | Learns proposal-level fine-minus-coarse accuracy gain and selects regions/scales under pixel cost | `same`: actual detection value of extra resolution | Sequential multi-action; no calibrated road reliability; no separate harm model; no exact F0 abstention | Critical; broad VoR novelty invalid |
| Uzkent et al., WACV 2020 | HR patch policy trained from fine-minus-coarse recall and acquisition/compute cost | `same`: patch-level detection utility plus budget | Multi-patch binary policy; recall ignores FP; no road reliability or risk fallback | Critical |
| AdaZoom, TMM 2022 | Detector misses/low confidence guide adaptive multi-region zoom | `partial/same`: uncertainty/difficulty-guided zoom | Proxy difficulty rather than paired benefit/harm; no zero-or-one safety gate | High |
| FOVEA, ICCV 2021 | Horizon/spatial or temporal prior controls foveated magnification | `same`: road/spatial resolution allocation as a broad idea | Global warp; no geometry reliability, paired outcome, or risk fallback | High |
| Two-Plane Perspective Prior, CVPR 2023 | Ground-plane/vanishing-point prior reallocates pixels to distant objects | `same`: perspective-aware road pixel allocation | Global resampling; no uncertainty/value/hard action/fallback | High |
| AutoFocus / CornerNet-Saccade / ClusDet / Cascaded Zoom-In | Objectness, centers, clusters, or density select high-resolution regions for another detector pass | `same`: detector-guided crop/re-detection skeleton | Mostly multiple regions and proxy selection, not paired risk-controlled action | High for generic crop claims |
| SAHI, ICIP 2022 | Uniform overlapping sliced inference and fusion | `same`: crop/coordinate-map/fusion mechanics | No learned selector; all slices processed | Crop and fusion are controls only |
| ZoomDet, 2026 | Learned global non-uniform warp with box magnification objective | `partial`: adaptive zoom | Global warp, not candidate benefit/harm or hard one-crop action | Medium-high; name collision |
| Perspective-aware road detection / FoveaNet | Perspective maps and scale-aware road-object processing | `same`: geometry-aware road detection broadly | No paired HR treatment outcome or abstention | High for geometry claim |
| GUPNet / GUPNet++ | Per-object geometry plus uncertainty on KITTI 3D detection | `same`: geometry+uncertainty broadly | Depth/3D confidence, not incremental HR evidence allocation | High for phrase-level claim |
| Adaptive Feeding, ICCV 2017 | Routes images between detectors/resolutions using empirical accuracy difference | `same`: performance-delta routing | Whole-image routing, no local crop or explicit paired harm | Critical for “learn when resolution helps” |
| DynamicDet, CVPR 2023 | Routes between detector stages using detector-loss difference | `partial/same`: residual-driven dynamic detection | Whole-image/stage computation, not HR crop risk | High |
| AdaDet / QR-DETR / AnytimeYOLO | Uncertainty or query-level early exit/dynamic compute | `same`: uncertainty-guided computation broadly | Different action and no paired HR treatment outcome | High for uncertainty-routing claim |
| NWD, 2021/2022 | Gaussian Wasserstein tiny-box metric for assignment/loss/NMS | `different` for F4; `same` if box metric claimed | Does not allocate pixels or observations | Baseline only |
| RFLA, ECCV 2022 | Gaussian receptive-field label assignment for tiny objects | `different` for F4; `same` if label assignment claimed | Does not allocate inference computation | Baseline only |

## 4. Primary-source and official-code register

| Work | Primary source | Author/official code | Visible license / reuse rule |
|---|---|---|---|
| FPN | https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html | https://github.com/facebookresearch/Detectron | Apache-2.0 |
| YOLOv8-P2 | https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml | same official repository | AGPL-3.0 |
| DyFPN | https://arxiv.org/abs/2012.00779 | https://github.com/Mingjian-Zhu/DyFPN | Apache-2.0 |
| QueryDet | https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html | https://github.com/ChenhongyiYang/QueryDet-PyTorch | MIT |
| ESOD | https://arxiv.org/abs/2407.16424 | https://github.com/alibaba/esod | GPL-3.0 plus third-party attribution |
| Dynamic Zoom-In | https://openaccess.thecvf.com/content_cvpr_2018/html/Gao_Dynamic_Zoom-In_Network_CVPR_2018_paper.html | no author repository located | No code reuse permitted from an unlicensed copy |
| Uzkent WACV 2020 | https://openaccess.thecvf.com/content_WACV_2020/html/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.html | https://github.com/uzkent/EfficientObjectDetection | No visible LICENSE; do not copy |
| AdaZoom | https://arxiv.org/abs/2106.10409 | no author repository located | No code reuse |
| FOVEA | https://openaccess.thecvf.com/content/ICCV2021/html/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.html | https://github.com/tchittesh/fovea | MIT |
| Two-Plane Prior | https://openaccess.thecvf.com/content/CVPR2023/html/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.html | https://github.com/geometriczoom/two-plane-prior | MIT; derived-code attribution applies |
| SAHI | https://arxiv.org/abs/2202.06934 | https://github.com/obss/sahi | MIT |
| ZoomDet | https://arxiv.org/abs/2602.07512 | https://github.com/twangnh/zoomdet_code | Apache-2.0 plus upstream attribution |
| GUPNet | https://openaccess.thecvf.com/content/ICCV2021/html/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.html | https://github.com/SuperMHP/GUPNet | MIT |
| DynamicDet | https://openaccess.thecvf.com/content/CVPR2023/html/Lin_DynamicDet_A_Unified_Dynamic_Architecture_for_Object_Detection_CVPR_2023_paper.html | https://github.com/VDIGPKU/DynamicDet | Academic-use terms; no unrestricted copy |
| NWD | https://arxiv.org/abs/2110.13389 | https://github.com/jwwangchn/NWD | Apache-2.0 |
| RFLA | https://arxiv.org/abs/2208.08738 | https://github.com/Chasel-Tsui/mmdet-rfla | MIT |

No third-party implementation is copied into the proposed F4. Public concepts and equations must be cited; implementation remains independent. Paper wording and diagrams must be newly authored.

## 5. Representative search log

All queries were executed on 2026-08-10. Full logs are preserved in the three task notes.

- `P2 high resolution detection head official YOLOv8 FPN`
- `dynamic selective feature pyramid object detection official code`
- `QueryDet ESOD sparse high resolution small object detection`
- `coarse fine detection accuracy gain zoom crop object detection`
- `counterfactual high resolution object detection candidate utility`
- `hard budget one crop object detection no harm fallback`
- `road perspective geometry uncertainty high resolution detection`
- `uncertainty guided dynamic computation object detection early exit`
- `NWD RFLA tiny object detection official code`
- `value of resolution detection geometry uncertainty crop budget`

## 6. Counter-review

1. **Obvious-combination risk:** road geometry, uncertainty, gain prediction, and crop inference all have strong independent precedents. F4 is not defensible merely because no single paper uses the exact list.
2. **Engineering-only risk:** top-1 selection and retaining the base output may be viewed as ordinary deployment constraints. F4 must test a distinct paired benefit/harm estimand and calibrated abstention, not just append `topk=1` and NMS.
3. **More-compute confound:** any AP gain may come from a second detector pass. Always-crop, expected-gain F3, and identical-budget F1/F2 controls are mandatory.
4. **Geometry rename risk:** vertical position, horizon, vanishing point, or a static KDE alone is prior art. Geometry reliability must be calibrated per frame and allowed to disable its own contribution.
5. **Uncertainty-not-value risk:** low confidence or high entropy may reflect occlusion that extra pixels cannot fix. Report ranking/calibration against observed paired benefit and harm, not only downstream AP.
6. **Safety overclaim risk:** exact F0 fallback only applies on abstention; selected crops can still harm. Report added FP, lost TP, class flips, near/medium slices, and budget violations.
7. **Search-limit risk:** bounded public search cannot prove absolute novelty. Do not use “first”; state the tested difference and cite the closest methods directly.

## 7. Coding authorization

- Ordinary P2, generic zoom, expected detection gain, geometry-aware magnification, and uncertainty routing: **NO-GO as novelty claims**.
- Independent implementation of leakage-free F0 and O1/O2 feasibility experiments: **GO**, because they test whether the route has any reachable AP value and do not claim novelty.
- New F4 code: **CONDITIONAL GO only under the narrowed definition in Section 2**, after O1 and O2 each retain at least +2.0 absolute AP_R40 and pass no-harm feasibility. Until then, reuse the existing local oracle and do not add modules.

## 8. Integrity

Positive, zero, and negative outcomes must be retained with input/config/code/result hashes. O1/O2 are upper-bound screens, not deployable results. A single-seed gain is preliminary; a positive claim requires three directionally consistent seeds, paired uncertainty, far/small improvement, near/medium no-harm, and complete latency/FLOPs/budget reporting.
