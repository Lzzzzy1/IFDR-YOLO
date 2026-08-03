---
task_id: e
role: Prior-art and reproducibility auditor
status: complete
sources_found: 12
---

## Sources

[1] LER-YOLO: Reliability-Aware Expert Routing for Misaligned RGB-Infrared UAV Detection | https://arxiv.org/abs/2605.20667 | Source-Type: academic | As Of: 2026-05 | Authority: 7/10
[2] RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR for Robust 3D Object Detection in Adverse Weather | https://arxiv.org/abs/2607.04587 | Source-Type: academic | As Of: 2026-07 | Authority: 7/10
[3] TOOD: Task-aligned One-stage Object Detection | https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf | Source-Type: academic | As Of: 2021-10 | Authority: 10/10
[4] QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection | https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf | Source-Type: academic | As Of: 2022-06 | Authority: 10/10
[5] ESOD: Efficient Small Object Detection on High-Resolution Images | https://arxiv.org/abs/2407.16424 | Source-Type: academic | As Of: 2024-07 | Authority: 7/10
[6] Rethinking Intersection Over Union for Small Object Detection in Few-Shot Regime | https://arxiv.org/abs/2307.09562 | Source-Type: academic | As Of: 2023-07 | Authority: 7/10
[7] CEM-FBGTinyDet: Context-Enhanced Foreground Balance with Gradient Tuning for Tiny Objects | https://arxiv.org/abs/2506.09897 | Source-Type: academic | As Of: 2025-06 | Authority: 6/10
[8] Scale-Aware Relay and Scale-Adaptive Loss for Tiny Object Detection in Aerial Images | https://arxiv.org/abs/2511.09891 | Source-Type: academic | As Of: 2025-11 | Authority: 7/10
[9] Wise-IoU Official Code | https://github.com/Instinct323/Wise-IoU | Source-Type: official | As Of: 2026-07-28 | Authority: 8/10
[10] QueryDet-PyTorch Official Code | https://github.com/ChenhongyiYang/QueryDet-PyTorch | Source-Type: official | As Of: 2026-07-28 | Authority: 9/10
[11] TOOD Official Code | https://github.com/fcjian/TOOD | Source-Type: official | As Of: 2026-07-28 | Authority: 9/10
[12] CN116758340A: Small Target Detection Method Based on Super-Resolution Feature Pyramid and Attention Mechanism | https://patents.google.com/patent/CN116758340A/en | Source-Type: official | As Of: 2023-09 | Authority: 6/10

## Findings

- Direct collision: LER-YOLO predicts a local correspondence-reliability map and uses it to gate aligned features and route among RGB-dominant, infrared-dominant, and interactive-fusion experts, so a generic claim of “reliability-conditioned feature routing” is occupied even though its routing is cross-modal rather than cross-scale. [1]
- Reproducibility caveat: LER-YOLO calls its module sparse top-\(k\) routing, but its reported implementation sets \(k=3\) for exactly three experts, so every expert executes and the demonstrated mechanism is conditional weighting rather than compute-sparse selection. [1]
- RAF independently supervises per-pixel reliability to suppress weather-corrupted camera cues before multimodal 3-D fusion, further occupying the broad claim that an uncertainty or reliability field should control feature fusion. [2]
- Strong partial collision: TOOD computes one alignment latent \(t=s^\alpha u^\beta\) from classification confidence and IoU, then reuses it for positive assignment, soft classification targets, and regression weighting while its task-aligned head also learns spatial prediction alignment. [3]
- TOOD's public MMDetection implementation provides configs, checkpoints, training commands, and an Apache-2.0 codebase, and its repository records adoption of Task Alignment Learning by PP-YOLOE, YOLOv6, YOLOv8, and YOLOv10, making “shared quality for assignment and loss” a reproducible baseline rather than a new component. [11]
- Direct sparse-P2 collision: QueryDet predicts coarse small-object locations on low-resolution pyramid levels and sparsely executes high-resolution \(P_3/P_2\) heads around them, reducing their reported FLOP share to about 1% and increasing inference from 4.85 to 14.88 FPS with only 0.17 AP loss. [4]
- QueryDet also shows that adding \(P_2\) changes the sample distribution enough to require level-wise loss rebalancing, and its maintained public code exposes separate dense and Cascade Sparse Query inference modes for COCO and VisDrone. [4][10]
- ESOD reuses backbone features for object-seeking and patch slicing and applies a sparse head to target-containing high-resolution regions, so region-selective high-resolution computation is occupied beyond QueryDet even without an explicit \(P_2\) name. [5]
- No scale term appears in the official WIoU focusing code beyond IoU geometry and the global running mean \(L_{IoU}/\overline{L}_{IoU}\), but scale-adaptive IoU, gradient-equilibrium DCLoss, and area-weighted SAL already occupy the functional ingredients of “scale-balanced WIoU.” [6][7][8][9]
- Patent collision is broad but relevant: CN116758340A describes a super-resolution pyramid producing \(P_2\) through \(P_6\) plus attention-based region proposals for small targets, although Google Patents records the application as withdrawn and it does not disclose reliability-conditioned sparse routing. [12]

## Deep Read Notes

### Source [1]: LER-YOLO: Reliability-Aware Expert Routing for Misaligned RGB-Infrared UAV Detection
Key data: \(R=\sigma(\phi([F_{ir},\widetilde F_{rgb}]))\) is trained with a feature-discrepancy plus \(-\log R\) anti-collapse term, modulates aligned RGB features, and enters a softmax top-\(k\) router; three-seed AP50 is \(89.7\pm0.2\).
Key insight: This is the closest functional collision to reliability-conditioned routing, but its present \(k=3\) over three experts provides no execution sparsity and its reliability measures cross-modal alignment rather than cross-scale detection uncertainty.
Useful for: reliability-map definition, router collision, matched-parameter controls, and code/reproducibility caveats.

### Source [3]: TOOD: Task-aligned One-stage Object Detection
Key data: \(t=s^\alpha u^\beta\) selects the top-\(m\) positives, its normalized value replaces binary positive classification labels, and the same quality weights bounding-box regression; TOOD reports 51.1 COCO AP.
Key insight: A single learned quality proxy already coordinates assignment and two losses, so novelty requires extending a calibrated reliability state into feature routing rather than merely reusing an IoU-confidence product.
Useful for: shared-latent collision, YOLOv8 TaskAlignedAssigner baseline, and separation between predictive quality and uncertainty.

### Source [4]: QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection
Key data: coarse query maps recursively activate high-resolution locations; reported \(P_3/P_2\) cost falls to about 1%, FPS rises 4.85 to 14.88, and explicit pyramid-level loss weights counter the \(P_2\) sample-count shift.
Key insight: Sparse high-resolution routing and scale balancing are already coupled operationally, but the query is an objectness/size proxy rather than a calibrated distance, occlusion, truncation, or uncertainty state.
Useful for: sparse-\(P_2\) collision, compute-matched controls, query-recall failure analysis, and official-code reproduction.

## Gaps

- Bounded gap: five targeted searches through 2026-07-28 did not identify one primary detector in which the same calibrated latent controls cross-scale fusion or execution, positive assignment, and uncertainty-aware regression simultaneously; this is a scoped search result, not an exhaustive novelty claim.
- Candidate surviving distinction: use one explicitly supervised or probabilistically calibrated road-scene reliability state across pyramid routing, TaskAlignedAssigner modification, and regression-gradient allocation, then prove that each reuse contributes beyond separate ASFF/DyFPN, TOOD, WIoU/SAL, and QueryDet modules.
- No reviewed sparse-\(P_2\) source conditions execution jointly on estimated distance, occlusion, and truncation; QueryDet and ESOD route primarily from small-object/object-seeking evidence, so a road-scene-conditioned policy remains a bounded functional gap. [4][5]
- Reliability semantics remain unresolved: LER-YOLO's map measures RGB-infrared correspondence, RAF's map measures modality degradation, TOOD's \(t\) measures classification-localization alignment, and none is automatically equivalent to calibrated localization or label uncertainty. [1][2][3]
- Scale-balanced WIoU is weak as a standalone novelty claim because SIoU, DCLoss, and SAL supply explicit scale or gradient balancing and the public WIoU module makes their composition straightforward. [6][7][8][9]
- Reproducibility gap: TOOD, QueryDet, and WIoU have public author repositories, while no author code repository was linked from the accessed LER-YOLO arXiv record; any comparison should therefore distinguish reproduced code from paper-only reimplementation. [1][9][10][11]
- Patent limitation: this scan used public keyword-accessible patent records and is not a freedom-to-operate or legal novelty opinion; claims, priority families, non-English records, and unpublished applications require professional patent searching.
- Counterclaim: reviewers may characterize the full proposal as LER-YOLO reliability routing plus TOOD task alignment plus QueryDet sparse \(P_2\) plus SAL/WIoU weighting, so defensibility depends on a new probabilistic coupling mechanism and interaction ablations, not the combination itself.
