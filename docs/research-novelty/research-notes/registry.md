# Citation Registry

Built from: `task-a.md`, `task-b.md`, `task-c.md`, `task-d.md`, `task-e.md`, `task-f.md`

AS_OF: 2026-08-04

Paper-facing claim-to-citation commitments are maintained in
`../citation-commitments.md`. That file is authoritative for methods actually
used or compared in IFDR-YOLO; this registry retains the broader novelty search.

## Approved Sources

[1] KITTI — Object Detection and Orientation Estimation Evaluation (2D) | https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d | Source-Type: official | Accessibility: public | As Of: 2026-07-28 | Auth: 10 | From: task-c
[2] KITTI — Download and Submission Policy | https://www.cvlibs.net/datasets/kitti/user_login.php | Source-Type: official | Accessibility: public | As Of: 2026-07-28 | Auth: 10 | From: task-c
[3] Instinct323 — Wise-IoU Official Code | https://github.com/Instinct323/Wise-IoU | Source-Type: official | Accessibility: public | As Of: 2026-07-28 | Auth: 8 | From: task-e
[4] Yang et al. — QueryDet-PyTorch Official Code | https://github.com/ChenhongyiYang/QueryDet-PyTorch | Source-Type: official | Accessibility: public | As Of: 2026-07-28 | Auth: 9 | From: task-e
[5] Feng et al. — TOOD Official Code | https://github.com/fcjian/TOOD | Source-Type: official | Accessibility: public | As Of: 2026-07-28 | Auth: 9 | From: task-e
[6] CN116758340A — Small Target Detection Method Based on Super-Resolution Feature Pyramid and Attention Mechanism | https://patents.google.com/patent/CN116758340A/en | Source-Type: official | Accessibility: public | As Of: 2023-09 | Auth: 6 | From: task-e
[7] Liu et al. — Learning Spatial Fusion for Single-Shot Object Detection | https://arxiv.org/abs/1911.09516 | Source-Type: academic | Accessibility: public | As Of: 2019-11 | Auth: 7 | From: task-a
[8] Zhu et al. — Dynamic Feature Pyramid Networks for Object Detection | https://arxiv.org/abs/2012.00779 | Source-Type: academic | Accessibility: public | As Of: 2020-12 | Auth: 7 | From: task-a
[9] Yang et al. — QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection | https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf | Source-Type: academic | Accessibility: public | As Of: 2022-06 | Auth: 10 | From: task-a/task-e
[10] Sun et al. — Uncertainty-Aware Gradient Stabilization for Small Object Detection | https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html | Source-Type: academic | Accessibility: public | As Of: 2025-10 | Auth: 10 | From: task-b/task-d
[11] Tong et al. — Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism | https://arxiv.org/abs/2301.10051 | Source-Type: academic | Accessibility: public | As Of: 2023-04 | Auth: 7 | From: task-b
[12] Scale-Aware Relay and Scale-Adaptive Loss for Tiny Object Detection in Aerial Images | https://arxiv.org/abs/2511.09891 | Source-Type: academic | Accessibility: public | As Of: 2025-11 | Auth: 7 | From: task-b/task-d
[13] Li et al. — Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection | https://proceedings.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html | Source-Type: academic | Accessibility: public | As Of: 2020-12 | Auth: 10 | From: task-b
[14] Feng et al. — Labels Are Not Perfect: Inferring Spatial Uncertainty in Object Detection | https://arxiv.org/abs/2012.12195 | Source-Type: academic | Accessibility: public | As Of: 2020-12 | Auth: 7 | From: task-b
[15] The Importance of Anti-Aliasing in Tiny Object Detection | https://arxiv.org/abs/2310.14221 | Source-Type: academic | Accessibility: public | As Of: 2023-10 | Auth: 8 | From: task-d
[16] Learn Discriminative Features for Small Object Detection through Multi-Scale Image Degradation with Contrastive Learning | https://www.jstage.jst.go.jp/article/transinf/E108.D/4/E108.D_2024EDP7204/_article/-char/en | Source-Type: academic | Accessibility: public | As Of: 2025-04 | Auth: 8 | From: task-d
[17] LER-YOLO: Reliability-Aware Expert Routing for Misaligned RGB-Infrared UAV Detection | https://arxiv.org/abs/2605.20667 | Source-Type: academic | Accessibility: public | As Of: 2026-05 | Auth: 7 | From: task-e
[18] Feng et al. — TOOD: Task-aligned One-stage Object Detection | https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.pdf | Source-Type: academic | Accessibility: public | As Of: 2021-10 | Auth: 10 | From: task-e
[19] Geiger et al. — Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite | https://www.cvlibs.net/publications/Geiger2012CVPR.pdf | Source-Type: academic | Accessibility: public | As Of: 2012-06 | Auth: 10 | From: task-c
[20] Pedestrian Emergence Estimation and Occlusion-Aware Risk Assessment for Urban Autonomous Driving | https://arxiv.org/abs/2107.02326 | Source-Type: academic | Accessibility: public | As Of: 2021-07 | Auth: 7 | From: task-d

## Dropped

x FPN | https://arxiv.org/abs/1612.03144 | Source-Type: academic | Auth: 9 | Reason: foundational but less specific than approved adaptive-fusion sources
x PANet | https://arxiv.org/abs/1803.01534 | Source-Type: academic | Auth: 9 | Reason: foundational but less specific than approved adaptive-fusion sources
x EfficientDet | https://arxiv.org/abs/1911.09070 | Source-Type: academic | Auth: 9 | Reason: BiFPN is established background; registry prioritizes closer functional collisions
x Fine-Grained Dynamic Head | https://arxiv.org/abs/2012.03519 | Source-Type: academic | Auth: 8 | Reason: DyFPN, ASFF, and QueryDet cover the selected routing claims
x HRDNet | https://arxiv.org/abs/2006.07607 | Source-Type: academic | Auth: 7 | Reason: superseded in the report by closer sparse-routing evidence
x Enhancing Aerial Pedestrian Detection via High-Resolution P2 Feature Integration in YOLOv12 | https://openaccess.thecvf.com/content/CVPR2026W/AERO-HPR/papers/S_Enhancing_Aerial_Pedestrian_Detection_via_High-Resolution_P2_Feature_Integration_in_CVPRW_2026_paper.pdf | Source-Type: academic | Auth: 7 | Reason: workshop engineering evidence, stronger functional collisions retained
x Unified Frequency-Aware Compensation Framework | https://www.mdpi.com/2072-4292/18/5/777 | Source-Type: academic | Auth: 7 | Reason: adjacent remote-sensing evidence, not needed for core claims
x Normalized Wasserstein Distance for Tiny Objects | https://www.sciencedirect.com/science/article/pii/S0924271622001599 | Source-Type: academic | Auth: 9 | Reason: scale-adaptive loss source retained instead
x ATSS | https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html | Source-Type: academic | Auth: 10 | Reason: TOOD is the closer shared-quality collision
x Bounding Box Regression With Uncertainty | https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.html | Source-Type: academic | Auth: 10 | Reason: label-uncertainty and GFL sources retained for the selected uncertainty distinction
x Modular Lightweight Network for Road Objects | https://arxiv.org/pdf/1811.06641 | Source-Type: academic | Auth: 8 | Reason: older application evidence not required for core protocol claims
x Improved YOLOX Road Object Detection | https://arxiv.org/pdf/2302.08156 | Source-Type: academic | Auth: 7 | Reason: protocol counterexample summarized in task notes but not needed in final registry
x SOD-YOLOv8 | https://arxiv.org/pdf/2408.04786 | Source-Type: academic | Auth: 8 | Reason: module-stacking collision is already supported by stronger method-level prior art
x YOLO-APD | https://arxiv.org/pdf/2507.05376 | Source-Type: academic | Auth: 7 | Reason: application preprint with weaker reproducibility evidence
x SET | https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html | Source-Type: academic | Auth: 10 | Reason: anti-aliasing source retained for the selected mechanistic argument
x HS-FPN | https://ojs.aaai.org/index.php/AAAI/article/download/32740/34895 | Source-Type: academic | Auth: 9 | Reason: frequency stacking is not the recommended primary research program
x FSDETR | https://arxiv.org/abs/2604.14884 | Source-Type: academic | Auth: 6 | Reason: recent preprint and not essential to the final claims
x RAF | https://arxiv.org/abs/2607.04587 | Source-Type: academic | Auth: 7 | Reason: LER-YOLO retained as the closer reliability-router collision
x ESOD | https://arxiv.org/abs/2407.16424 | Source-Type: academic | Auth: 7 | Reason: QueryDet retained as the stronger sparse-high-resolution reference
x Scale-Adaptive IoU for Few-Shot Small Objects | https://arxiv.org/abs/2307.09562 | Source-Type: academic | Auth: 7 | Reason: scale-adaptive loss source retained instead
x CEM-FBGTinyDet | https://arxiv.org/abs/2506.09897 | Source-Type: academic | Auth: 6 | Reason: lower-authority preprint and redundant gradient-balancing evidence

## Stats

- Total evaluated: 41
- Approved: 20
- Dropped: 21
- Unique approved domains: 8
- Source type: official 6 / academic 14
- Official share: 6/20 = 30% (pass)
- Maximum single-domain share: arxiv.org 5/20 = 25% (pass)
- Private or privileged sources rejected: 0
- Duplicate URLs in approved registry: 0
