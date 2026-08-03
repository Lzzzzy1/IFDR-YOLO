---
task_id: c
role: Autonomous-driving benchmark researcher
status: complete
sources_found: 7
---

## Sources

[1] KITTI Object Detection and Orientation Estimation Evaluation (2D) | https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d | Source-Type: official | As Of: 2026-07-28 | Authority: 10/10
[2] KITTI Download and Submission Policy | https://www.cvlibs.net/datasets/kitti/user_login.php | Source-Type: official | As Of: 2026-07-28 | Authority: 10/10
[3] Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite | https://www.cvlibs.net/publications/Geiger2012CVPR.pdf | Source-Type: academic | As Of: 2012-06 | Authority: 10/10
[4] Detecting the Objects on the Road Using Modular Lightweight Network | https://arxiv.org/pdf/1811.06641 | Source-Type: academic | As Of: 2018-11 | Authority: 8/10
[5] Research on Road Object Detection Algorithm Based on Improved YOLOX | https://arxiv.org/pdf/2302.08156 | Source-Type: academic | As Of: 2023-02 | Authority: 7/10
[6] SOD-YOLOv8: Enhancing YOLOv8 for Small Object Detection in Traffic Scenes | https://arxiv.org/pdf/2408.04786 | Source-Type: academic | As Of: 2024-08 | Authority: 8/10
[7] YOLO-APD: Enhancing YOLOv8 for Robust Pedestrian Detection on Complex Road Geometries | https://arxiv.org/pdf/2507.05376 | Source-Type: academic | As Of: 2025-07 | Authority: 7/10

## Findings

- KITTI 2D has 7,481 labeled training images and 7,518 hidden-label test images but no official validation partition, so the exact image IDs and sequence-grouping method of every train/validation/test split must be published and the server used only once for the final selected model. [1][2][3]
- Official 2D matches require IoU 0.70 for Car and 0.50 for Pedestrian/Cyclist, with Easy defined by height at least 40 px, no occlusion, and truncation at most 15%, Moderate by height at least 25 px, partial occlusion, and truncation at most 30%, and Hard by height at least 25 px, difficult visibility, and truncation at most 50%, while ranking uses Moderate and Hard has an approximately 98% human-recall ceiling. [1]
- Since 8 October 2019 KITTI reports AP at 40 recall positions rather than the earlier 11, making pre-2019 AP11, generic YOLO mAP@0.5, COCO AP, and current class-specific KITTI AP40 Easy/Moderate/Hard numerically non-interchangeable without re-evaluation. [1][4][5]
- The official evaluator does not count detections in DontCare regions or detections below the difficulty-specific minimum height as false positives, so a YOLO conversion that discards DontCare/ignored ground truth or evaluates every box with ordinary COCO/VOC code does not reproduce KITTI and can especially distort small/far-object conclusions. [1]
- Split practice is visibly non-uniform: the benchmark delegates validation design to participants, the improved-YOLOX study randomly uses 7:1:2, and YOLO-APD uses 90:10, while KITTI’s original benchmark construction explicitly kept source sequences out of both official train and test, so results from undocumented random splits are not directly comparable and may not preserve sequence independence. [2][3][5][7]
- As of 2026-07-28 the all-input 2D leaderboard reports 80.13 AP40 Moderate for Pedestrian and 85.30 for Cyclist, but it mixes RGB, stereo, LiDAR, cross-modal, and extra-data methods, so those ranks cannot be used as an unqualified RGB-only YOLO baseline. [1]
- Multi-scale/context fusion for far road users was already claimed by the 2018 MFFD work, while loss redesign and occlusion-aware box separation were claimed by improved YOLOX and a P2 head, feature fusion, attention, and IoU-loss replacement were combined in SOD-YOLOv8, making “add attention + extra small-object head + better IoU + fusion” a saturated recipe rather than a dissertation-level novelty claim by itself. [4][5][6]
- A rigorous dissertation evaluation should pair official per-class AP40 Easy/Moderate/Hard and PR curves with fixed-split multi-seed ablations, uncertainty intervals, latency/FPS/parameters/FLOPs/memory on fixed hardware, and custom recall/error breakdowns by pixel-height, distance proxy, occlusion level, truncation bin, class frequency, and crowded overlap while clearly labeling all non-official metrics. [1][3][4][5]
- RGB-only KITTI can support a narrowly scoped benchmark or efficiency contribution, but broad claims about small/far pedestrian and cyclist robustness require at least one external dataset or cross-domain test because KITTI’s primary protocol ignores sub-25-px objects and recent YOLO evidence shows both dataset-specific protocols and measurable synthetic-to-real/domain-shift effects. [1][6][7]

## Deep Read Notes

### Source [1]: KITTI Object Detection and Orientation Estimation Evaluation (2D)
Key data: 7,481 train/7,518 test images; Pedestrian/Cyclist IoU 0.50; Easy/Moderate/Hard thresholds are 40/25/25 px, occlusion 0/1/2, and truncation 15/30/50%; AP40 replaced AP11 in 2019.
Key insight: Official difficulty entangles size, occlusion, and truncation and suppresses DontCare/minimum-height detections, so one headline mAP cannot diagnose small/far road-user performance.
Useful for: canonical evaluation rules, current leaderboard saturation, AP11-versus-AP40 comparability, and the required devkit check.

### Source [3]: Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite
Key data: KITTI used 1,392×512 cameras before rectification, annotated cars/vans/trucks/pedestrians/cyclists/trams with visibility and truncation states, and ensured images from one source sequence did not enter both official train and test.
Key insight: Sequence separation was a deliberate benchmark design property, so random image-level validation splits require an explicit leakage audit rather than being treated as equivalent.
Useful for: dataset provenance, occlusion/truncation interpretation, split governance, and limits on generalization claims.

### Source [5]: Research on Road Object Detection Algorithm Based on Improved YOLOX
Key data: The study collapses labels to Car/Cyclist/Pedestrian, divides all 7,481 labeled images 7:1:2, trains for 500 epochs, evaluates mAP/mAR at IoU 0.5 plus COCO-style size/AP metrics, and reports 88.9/89.1 mAP for improved YOLOX-s/m.
Key insight: Its high headline mAP is not official per-class KITTI AP40 and illustrates how split, label mapping, and metric choices can dominate apparent comparability.
Useful for: prior-art saturation in loss/occlusion handling and a counterexample to unqualified “KITTI mAP” comparisons.

## Gaps

- Searched but did not find an official KITTI source that designates the widely reused 3,712/3,769 “Chen split” as the canonical 2D split; if used, it must be cited as a non-official inherited protocol with the exact image lists.
- Searched but did not find a 2020-2026 peer-reviewed YOLOv8 study reporting exact official KITTI AP40 Easy/Moderate/Hard for both Pedestrian and Cyclist together with public split files, DontCare handling, code, seeds, and uncertainty estimates.
- Searched but did not establish exhaustive novelty, because relevant work may be indexed under general small-object, pedestrian, monocular-3D, or multimodal detection rather than “YOLO + KITTI.”
- Alternative interpretation: weak Hard results can reflect KITTI’s coupled size/occlusion/truncation definition and annotation ceiling rather than a detector’s isolated inability to recognize small objects.
- Alternative interpretation: leaderboard gains may come from LiDAR/stereo, external pretraining, additional data, input resolution, or test-time processing rather than the claimed architecture, so modality- and data-matched baselines are mandatory.
- Limitation: KITTI’s official Moderate/Hard protocol excludes targets below 25 px from the primary score, so extreme-far performance needs a separately specified custom validation protocol and cannot be advertised as an official KITTI leaderboard improvement.
