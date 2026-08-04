# IFDR-YOLO Citation Commitments

Updated: 2026-08-04

This file is the paper-facing citation ledger. A checked-in method must not be
described as original when it appears under `Prior work we must cite`.

## Prior work we must cite

| Topic used in IFDR-YOLO | Citation | What it supports | Originality boundary |
|---|---|---|---|
| KITTI data, labels and evaluation | Geiger et al., *Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012. https://www.cvlibs.net/publications/Geiger2012CVPR.pdf | Dataset and road-scene benchmark | KITTI and its difficulty definitions are not ours |
| BDD100K cross-domain validation | Yu et al., *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html | Diverse driving-domain validation | BDD100K and its diversity claims are not ours |
| Feature pyramids | Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017. https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html | Multi-scale top-down/lateral features | Pyramid detection is not ours |
| Bottom-up pyramid aggregation | Liu et al., *Path Aggregation Network for Instance Segmentation*, CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Path_Aggregation_Network_CVPR_2018_paper.html | PAN/FPN background used by YOLO necks | Bidirectional paths alone are not ours |
| Weighted bidirectional fusion | Tan et al., *EfficientDet: Scalable and Efficient Object Detection*, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html | BiFPN-style learnable fusion weights | Generic adaptive weighted fusion is not ours |
| High-resolution small-object processing | Yang et al., *QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection*, CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html | High-resolution features help small objects | Adding or sparsely using high-resolution features is not ours |
| Scale-specific training | Singh and Davis, *An Analysis of Scale Invariance in Object Detection (SNIP)*, CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Singh_An_Analysis_of_CVPR_2018_paper.html | Scale-conditioned gradient selection | Scale curricula alone are not ours |
| Online hard-example mining | Shrivastava et al., *Training Region-Based Object Detectors with Online Hard Example Mining*, CVPR 2016. https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html | Non-uniform training allocation based on current example difficulty | Generic hard-example replay is not ours |
| Dense hard-example focusing | Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017. https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html | Down-weighting easy dense-detection examples | Focal weighting is not ours |
| Adaptive positive assignment | Zhang et al., *ATSS*, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html | Object-statistics-based positive selection | Adaptive assignment alone is not ours |
| Task-aligned assignment/head | Feng et al., *TOOD*, ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html | Classification-localization alignment | Task alignment alone is not ours |
| Class-balanced reweighting | Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html | Effective-number class weighting | Fixed class reweighting is not ours |
| Tail-gradient suppression | Tan et al., *Equalization Loss for Long-Tailed Object Recognition*, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_Equalization_Loss_for_Long-Tailed_Object_Recognition_CVPR_2020_paper.html | Rare classes suffer discouraging gradients | Equalization loss is not ours |
| Long-tailed detection equilibrium | Feng et al., *Exploring Classification Equilibrium in Long-Tailed Object Detection*, ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Feng_Exploring_Classification_Equilibrium_in_Long-Tailed_Object_Detection_ICCV_2021_paper.html | Improving tail categories without sacrificing head categories | LOCE/memory sampling is not ours |
| Step-wise long-tailed detection | Dong et al., *Boosting Long-tailed Object Detection via Step-wise Learning on Smooth-tail Data*, ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Boosting_Long-tailed_Object_Detection_via_Step-wise_Learning_on_Smooth-tail_Data_ICCV_2023_paper.html | Head-preserving replay followed by tail-dominant transfer | Staged replay is not ours |
| Head-to-tail staged transfer | Tran, *SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection*, CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Tran_SimLTD_Simple_Supervised_and_Semi-Supervised_Long-Tailed_Object_Detection_CVPR_2025_paper.html | Pre-train, tail transfer and balanced fine-tuning | Multi-stage head-to-tail transfer is not ours |
| Cross-dataset detection difficulty | Wang et al., *Train in Germany, Test in the USA: Making 3D Object Detectors Generalize*, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Train_in_Germany_Test_in_the_USA_Making_3D_Object_CVPR_2020_paper.html | Camera-aware difficulty analysis using depth, occlusion and truncation | Metadata-based difficulty definitions are not ours |
| Image-level degradation representation | Becker et al., *Self-Aware Object Detection via Degradation Manifolds*, arXiv:2602.18394, 2026. https://arxiv.org/abs/2602.18394 | Degradation-structured detector features and cross-dataset reliability monitoring | Image-level degradation manifolds are not ours; this source is a non-peer-reviewed preprint |
| Dynamic focusing IoU | Tong et al., *Wise-IoU*, 2023. https://arxiv.org/abs/2301.10051 | Quality-relative regression gradient allocation | WIoU is not ours |
| Small-object localization gradients | Sun et al., *Uncertainty-Aware Gradient Stabilization for Small Object Detection*, ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html | Small boxes can have sharper localization curvature | Generic small-object gradient stabilization is not ours |
| Gradient-conflict projection | Yu et al., *Gradient Surgery for Multi-Task Learning (PCGrad)*, NeurIPS 2020. https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html | Conflicting auxiliary gradients can be projected | Generic gradient surgery is not ours |
| Validation-aware negative-transfer control | Jiang et al., *ForkMerge*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/60f9118a849e8e9a0c67e2a36ad80ebf-Abstract-Conference.html | Auxiliary tasks can hurt the target despite optimization fixes | Generic auxiliary-task filtering is not ours |
| Multi-objective conflict control | Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning*, NeurIPS 2021. https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html | Worst-task-aware conflict control and Pareto motivation | CAGrad and generic Pareto optimization are not ours |

## Candidate baselines that require citation if implemented

- Inner-IoU: https://arxiv.org/abs/2311.02877
- Alpha-IoU: https://arxiv.org/abs/2110.13675
- GradNorm: https://proceedings.mlr.press/v80/chen18a.html
- ScaleKD: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html
- Data-centric long-tailed sample selection (LTTSS): https://openaccess.thecvf.com/content/ACCV2022/html/Xu_Boosting_Dense_Long-Tailed_Object_Detection_from_Data-Centric_View_ACCV_2022_paper.html
- Counterfactual and invariant visual data generation: https://openaccess.thecvf.com/content/CVPR2021/html/Chang_Towards_Robust_Classification_Model_by_Counterfactual_and_Invariant_Data_Generation_CVPR_2021_paper.html

## Literature-backed time-saving route

The preferred next intervention is a degradation-aware staged fine-tuning
protocol initialized from the accepted protected IFDR checkpoint:

1. Reuse the accepted all-class checkpoint as the shared road-scene
   representation instead of restarting from ImageNet weights.
2. Perform a short Cyclist-focused stage whose sampling priority is conditioned
   on KITTI scale, depth, occlusion and truncation, not class identity alone.
3. Perform a low-learning-rate balanced recovery stage over all classes.
4. Select the checkpoint with an explicit no-harm gate over Car, Pedestrian and
   Cyclist, then confirm a winning recipe with three seeds.

This protocol is inspired by SimLTD's head-to-tail transfer, LOCE's
head/tail-equilibrium goal, SNIP's scale-conditioned training and LTTSS's
rare-class positive allocation. The joint degradation score, protected IFDR
initialization, instance-level road-scene conditioning and no-harm gate are the
parts that must be evaluated as IFDR-YOLO contributions.

Required controls are: equal-budget ordinary fine-tuning, class-only sampling,
degradation-only sampling without balanced recovery, and the full staged
protocol. PCGrad or CAGrad is a strong optimization baseline for semantic
protection but must not silently replace the existing protected/unprotected
matched control.

## IFDR-YOLO contribution boundary

The paper may claim the following only after the corresponding controlled
experiments support it:

1. A KITTI-specific degradation representation jointly derived from object
   scale, depth/distance, occlusion and truncation instead of class frequency
   or scale alone.
2. Degradation-conditioned control of rare-target training and localization,
   with P2 reliability evidence rather than an unconditional module stack.
3. Semantic protection designed around the measured conflict between the main
   detector and IFDR factor/counterfactual auxiliary paths.
4. A no-harm objective and evaluation protocol: improve distant Cyclist while
   bounding the loss on Car and Pedestrian.
5. A closed evidence loop: stratified diagnosis, targeted intervention,
   matched controls, multiple seeds, uncertainty intervals and cross-domain
   validation.

The paper must not claim P2, BiFPN, WIoU, class reweighting, curriculum
learning, adaptive assignment or gradient surgery individually as new.

## Citation and code-use rules

- Cite the original paper wherever an adopted equation, mechanism or baseline
  is first introduced.
- State `adapted from` or `implemented following` when our implementation is a
  modification rather than a clean-room independent idea.
- Record repository URL, commit and license before copying any source code.
- Do not copy figures, tables or prose; redraw figures from our own model and
  report our own measurements.
- Every experiment configuration must record which cited baseline it realizes.
- Use `counterfactual-inspired intervention` rather than a causal claim unless
  the intervention validity and causal assumptions are explicitly tested.
