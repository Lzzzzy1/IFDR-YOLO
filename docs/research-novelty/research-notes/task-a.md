---
task_id: a
role: Small-object architecture researcher
status: complete
sources_found: 10
---

## Sources

[1] Feature Pyramid Networks for Object Detection | https://arxiv.org/abs/1612.03144 | Source-Type: academic | As Of: 2017-04 | Authority: 9/10
[2] Path Aggregation Network for Instance Segmentation | https://arxiv.org/abs/1803.01534 | Source-Type: academic | As Of: 2018-09 | Authority: 9/10
[3] EfficientDet: Scalable and Efficient Object Detection | https://arxiv.org/abs/1911.09070 | Source-Type: academic | As Of: 2020-07 | Authority: 9/10
[4] Learning Spatial Fusion for Single-Shot Object Detection | https://arxiv.org/abs/1911.09516 | Source-Type: academic | As Of: 2019-11 | Authority: 7/10
[5] Fine-Grained Dynamic Head for Object Detection | https://arxiv.org/abs/2012.03519 | Source-Type: academic | As Of: 2020-12 | Authority: 8/10
[6] Dynamic Feature Pyramid Networks for Object Detection | https://arxiv.org/abs/2012.00779 | Source-Type: academic | As Of: 2020-12 | Authority: 7/10
[7] HRDNet: High-resolution Detection Network for Small Objects | https://arxiv.org/abs/2006.07607 | Source-Type: academic | As Of: 2020-06 | Authority: 7/10
[8] QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection | https://arxiv.org/abs/2103.09136 | Source-Type: academic | As Of: 2022-03 | Authority: 9/10
[9] Enhancing Aerial Pedestrian Detection via High-Resolution P2 Feature Integration in YOLOv12 | https://openaccess.thecvf.com/content/CVPR2026W/AERO-HPR/papers/S_Enhancing_Aerial_Pedestrian_Detection_via_High-Resolution_P2_Feature_Integration_in_CVPRW_2026_paper.pdf | Source-Type: academic | As Of: 2026-06 | Authority: 7/10
[10] From Structural Degradation to Semantic Misalignment: A Unified Frequency-Aware Compensation Framework for Remote Sensing Object Detection | https://www.mdpi.com/2072-4292/18/5/777 | Source-Type: academic | As Of: 2026-03 | Authority: 7/10

## Findings

- The established pyramid progression is FPN's top-down pathway with lateral connections, PANet's added bottom-up path for shorter localization-signal propagation, and BiFPN's efficient weighted bidirectional fusion, so another static bidirectional neck is a saturated architectural claim. [1][2][3]
- ASFF already learns spatially varying fusion that filters conflicting information among aligned pyramid levels in a one-stage YOLOv3 detector, making generic "per-pixel adaptive pyramid fusion" an occupied claim. [4]
- Fine-Grained Dynamic Head conditionally selects a pixel-level combination of FPN scales for each instance and couples it to a spatial gate and sparse convolution, narrowing the remaining room beyond ASFF to more specific conditioning, routing, or deployment objectives. [5]
- DyFPN places an image-conditioned gate before multi-kernel lateral branches and reported about 40% lower FLOPs than its inception-FPN comparator at similar accuracy, so conditional branch execution and explicit compute regularization are also established patterns. [6]
- HRDNet processes high-resolution input with shallow backbones and lower-resolution input with deeper backbones, then fuses multi-stream and multi-level features in MS-FPN; it reported more than 4.9 AP-small improvement over then-recent COCO models. [7]
- QueryDet predicts coarse small-object locations on low-resolution features and sparsely computes high-resolution detection around them, reporting +2.0 AP-small and 3.0x high-resolution inference speed on COCO plus 2.3x acceleration on VisDrone. [8]
- A 2026 CVPR workshop paper still improves aerial pedestrian detection by adding a P2 branch and bidirectional fusion, which is evidence that "add P2 for tiny objects" remains active engineering but is not, by itself, a defensible doctoral novelty claim. [9]
- A 2026 remote-sensing detector combines shallow-guided high-frequency calibration, cross-scale spatial/semantic alignment, and cascaded gated fusion, so generic frequency compensation plus adaptive fusion is already represented in adjacent small-object literature. [10]
- Across these sources, spatially dense fusion, image-level conditional computation, multi-resolution branches, and sparse high-resolution querying are distinct prior-art axes; a credible YOLOv8m/KITTI contribution must specify a new joint conditioning signal, routing granularity, objective, or deployment constraint rather than merely recombining their labels. [4][5][6][7][8]

## Deep Read Notes

### Source [6]: Dynamic Feature Pyramid Networks for Object Detection
Key data: DyFPN gates the multi-kernel lateral block per input image and reports roughly 40% FLOP reduction versus inception FPN while preserving similar accuracy.
Key insight: Its adaptivity is image-level execution of a receptive-field enrichment branch, not object-region routing among P2/P3/P4/P5.
Useful for: bounding conditional-computation novelty and motivating matched-FLOP, static-gate, and latency controls.

### Source [7]: HRDNet: High-resolution Detection Network for Small Objects
Key data: MD-IPN uses shallow networks for high-resolution streams and deeper networks for lower-resolution streams; MS-FPN propagates semantics both across hierarchy levels and across resolution streams.
Key insight: It separates detail retention from semantic depth architecturally, showing that a persistent high-resolution branch is older than YOLOv8 and not equivalent to simply attaching a P2 head.
Useful for: high-resolution-branch prior art, compute/detail trade-offs, and small-versus-large-object ablations.

## Gaps

- Bounded-search gap: none of the reviewed primary sources jointly evaluates a YOLOv8m neck on KITTI with routing conditioned on road-scene distance/scale, occlusion, and truncation while enforcing an explicit expected-FLOP or measured-latency budget; this is a candidate research gap, not an exhaustive novelty claim.
- Defensible direction: route semantic enrichment and/or P2 computation only to regions whose features predict distant, small, or occluded KITTI objects, then evaluate AP by class and KITTI difficulty together with AP-small, calibration, latency, memory, and energy under parameter/FLOP-matched static controls.
- Defensible frequency direction: make frequency compensation conditional and localized to P2/P3 candidate regions, with an anti-aliasing or noise-suppression objective and cross-condition tests; merely adding FFT/wavelet/high-frequency attention is weak because adjacent 2026 work already combines frequency calibration with gated cross-scale fusion. [10]
- Evidence gap: the reviewed high-resolution and sparse-routing studies are primarily COCO, VisDrone, aerial, or remote-sensing evaluations, so transfer to KITTI's forward-camera scale distribution and class imbalance must be demonstrated rather than assumed. [7][8][9][10]
- Counterclaim: reviewers can reasonably characterize a proposed region router as a combination of Fine-Grained Dynamic Head's pixel/instance scale selection, DyFPN's conditional branches, and QueryDet's sparse high-resolution querying; the work therefore needs a technically distinct conditioning mechanism or optimization target plus evidence that gains are not due to extra resolution, parameters, or training budget. [5][6][8]
- Counterclaim: P2 may improve distant pedestrians/cyclists but can amplify texture and aliasing noise and increase memory; any benefit should survive equal-compute controls, per-size error analysis, and removal/replacement ablations for P2, the router, and frequency/context modules.
- Search limitation: four targeted searches covered synonyms including adaptive spatial fusion, pixel-level pyramid fusion, dynamic branch routing, sparse high-resolution querying, and frequency-aware fusion as of 2026-07-28, but they do not justify claiming that no unreviewed paper implements a similar combination.
