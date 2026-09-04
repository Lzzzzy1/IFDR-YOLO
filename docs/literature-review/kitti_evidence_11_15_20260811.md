# KITTI 冻结清单 K11–K15：原始证据摘录（独立副读）

> 任务角色：只为主代理提供逐篇可追溯证据，不替代最终综述判断。  
> 检索/核验日期（AS_OF）：2026-08-12。  
> 来源规则：只采用论文的 CVF 正式页面/官方 PDF、arXiv 原稿与论文作者仓库；没有在原文中明确写出的 AP11/AP40、seed、CI、参数量、FLOPs 一律记为 `unknown`。  
> 比较禁令：以下五篇主要研究 KITTI **单目或双目 3D/BEV Car 检测**，不能把其中数值与本项目 RGB 2D Pedestrian/Cyclist Moderate macro AP_R40 相减或相加。

## 本地来源编号

[S11-P] Qian et al., *End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection*, CVPR 2020, pp. 5881–5890 | https://openaccess.thecvf.com/content_CVPR_2020/html/Qian_End-to-End_Pseudo-LiDAR_for_Image-Based_3D_Object_Detection_CVPR_2020_paper.html | Source-Type: academic (CVF official proceedings) | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10  
[S11-A] arXiv:2004.03080（作者原稿与源码） | https://arxiv.org/abs/2004.03080 | Source-Type: academic | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10  
[S11-C] 作者官方代码 `mileyan/pseudo-LiDAR_e2e` | https://github.com/mileyan/pseudo-LiDAR_e2e | Source-Type: official | Accessibility: public | License: MIT（GitHub API 于 2026-08-12 核验） | Authority: 9/10

[S12-P] Chen et al., *MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation*, CVPR 2021, pp. 10379–10388 | https://openaccess.thecvf.com/content/CVPR2021/html/Chen_MonoRUn_Monocular_3D_Object_Detection_by_Reconstruction_and_Uncertainty_Propagation_CVPR_2021_paper.html | Source-Type: academic (CVF official proceedings) | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10  
[S12-A] arXiv:2103.12605（含正文及 supplementary 源码） | https://arxiv.org/abs/2103.12605 | Source-Type: academic | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10  
[S12-C] 作者官方代码 `tjiiv-cprg/MonoRUn` | https://github.com/tjiiv-cprg/MonoRUn | Source-Type: official | Accessibility: public | License: MIT（GitHub API 于 2026-08-12 核验） | Authority: 9/10

[S13-P] Lu et al., *Geometry Uncertainty Projection Network for Monocular 3D Object Detection*, ICCV 2021, pp. 3111–3121 | https://openaccess.thecvf.com/content/ICCV2021/html/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.html | Source-Type: academic (CVF official proceedings) | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10  
[S13-A] arXiv:2107.13774（作者原稿与源码） | https://arxiv.org/abs/2107.13774 | Source-Type: academic | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10  
[S13-C] 作者官方代码 `SuperMHP/GUPNet` | https://github.com/SuperMHP/GUPNet | Source-Type: official | Accessibility: public | License: MIT（GitHub API 于 2026-08-12 核验） | Authority: 9/10

[S14-P] Ma et al., *Delving Into Localization Errors for Monocular 3D Object Detection*, CVPR 2021, pp. 4721–4730 | https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Delving_Into_Localization_Errors_for_Monocular_3D_Object_Detection_CVPR_2021_paper.html | Source-Type: academic (CVF official proceedings) | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10  
[S14-A] arXiv:2103.16237（作者原稿、正文与 supplementary 源码） | https://arxiv.org/abs/2103.16237 | Source-Type: academic | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10  
[S14-C] 作者官方代码 `xinzhuma/monodle` | https://github.com/xinzhuma/monodle | Source-Type: official | Accessibility: public | License: MIT（GitHub API 于 2026-08-12 核验） | Authority: 9/10

[S15-P] Zhang et al., *MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection*, ICCV 2023, pp. 9155–9166 | https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_MonoDETR_Depth-guided_Transformer_for_Monocular_3D_Object_Detection_ICCV_2023_paper.html | Source-Type: academic (CVF official proceedings) | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10  
[S15-S] MonoDETR official supplementary material | https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhang_MonoDETR_Depth-guided_Transformer_ICCV_2023_supplemental.pdf | Source-Type: academic (CVF official supplement) | Accessibility: public | As Of: 2026-08-12 | PDF SHA256: `42D8C1C5EF21EDD15B42F52EAB4E66442230C5554EC55F7BB7CC84DBA6EEF03E` | Authority: 10/10  
[S15-A] arXiv:2203.13310（v5 于 2025-02-13 更新；本卡以 ICCV camera-ready 与官方 supplement 为数值主来源） | https://arxiv.org/abs/2203.13310 | Source-Type: academic | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10  
[S15-C] 作者官方代码 `ZrrSkywalker/MonoDETR` | https://github.com/ZrrSkywalker/MonoDETR | Source-Type: official | Accessibility: public | License: MIT；核验 HEAD `6994b9f512400b258c6edb75f77423beb9c126f2`（2026-08-12） | Authority: 9/10

---

## K11. End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection

### A. 元数据与原始入口

- Rui Qian, Divyansh Garg, Yan Wang, Yurong You, Serge Belongie, Bharath Hariharan, Mark Campbell, Kilian Q. Weinberger, Wei-Lun Chao；CVPR 2020，正式论文页 5881–5890；arXiv:2004.03080。[S11-P][S11-A]
- 正式论文页直接给出作者代码；仓库许可证为 MIT。借代码仍须保留许可证和作者归属，不能把 CoR、soft quantization 或联合深度—检测训练改名为本项目原创。[S11-P][S11-C]

### B. 研究问题与真正 gap

- 既有 pseudo-LiDAR 是两段式训练：先用像素级目标训练 stereo depth estimator，再冻结它、训练 LiDAR-style 3D detector。两段优化目标并不对齐，最终检测误差无法告诉深度网络“哪些深度错误真正伤害目标检测”。[S11-P，Sec. 3]
- 作者的定量观察是约 90% 像素属于背景，车与行人/骑行者合计约 10%，其中人仅小于 1%；普通像素深度目标因而偏向背景/近景，远目标又只有很少 pseudo points，最容易被漏检或定位不准。[S11-A，Fig. 2；Sec. 3]
- 真正 gap 不是“增加高分辨率层”，而是让检测损失穿过 representation change，反向修正对检测有用的深度区域，同时控制不同损失的梯度覆盖和量级。

### C. 输入、网络流程、训练目标与后处理

1. 输入是**左右双目图像**，不是单张 RGB；Stereo Depth Network（SDN）输出 dense depth map。
2. 深度经 differentiable Change of Representation（CoR）变成 pseudo-LiDAR：
   - point-input 路线接 PointRCNN；
   - voxel 路线接带视觉信息的 PIXOR。硬体素化不可导，作者用对 27 个相邻 bins 的 RBF soft quantization 扩大梯度可达区域。
3. 训练按三步进行：SDN 预训练/微调；冻结 SDN 从头训练 3D detector；最后联合微调，使 detection loss 回传到 depth network。SDN 先在 Scene Flow 预训练，再用 KITTI 3,712 train 图像及投影 LiDAR 稀疏深度微调。[S11-A，Sec. 4.1]
4. 总体包含 depth loss 与 detector 原生损失；作者不是简单等权相加，而是查看它们对 predicted depth map 的梯度后设权重：PointRCNN 路线 `lambda_depth=1, lambda_det=0.01`，PIXOR 路线 `1, 0.1`。[S11-A，Sec. 4.1]
5. Table 1 的梯度诊断：depth loss、PointRCNN loss、PIXOR loss 覆盖深度图像素比例分别为 3%、4%、70%；梯度均值约为 `1e-5/1e-3/1e-5`，梯度和约为 `0.1/10/1`。作者明确称“平衡梯度和”对稳定联合训练关键。[S11-A，Table 1]
6. 论文复用 PointRCNN/PIXOR 的 detector、assignment/NMS，并未提出面向 YOLO 2D head 的新 matching 或 NMS。

### D. KITTI 数据与 split

- 原文采用常见 3,712 train / 3,769 validation / 7,518 official test，即后来常称的 Chen split；论文写出数目，但没有在该段声明 sequence-level independence。[S11-A，Sec. 4.1]
- KITTI 为每幅图提供 64-beam Velodyne、右相机图像和相机标定；本方法训练时用右图/双目和由 LiDAR 投影得到的 depth ground truth，输入/监督信息明显多于本项目单幅 RGB 2D 检测。[S11-A，Sec. 4.1]
- 论文既报告 3,769 val，也向官方隐藏 test 提交；test 可视为 benchmark confirmation，但没有多 seed 或置信区间。

### E. 类别、任务、IoU、AP 口径与难度

- 类别仅 **Car**；任务是 `AP_BEV` 与 `AP_3D`，不是 2D AP。validation 同时报 IoU=0.5 和 0.7；official test 表报 IoU=0.7；均分 Easy/Moderate/Hard。[S11-A，Tables 2–3]
- 论文表头只写 `AP`，没有在正文/表注明确写 AP11 或 AP40。因此这里把 recall sampling 版本记为 **unknown**；即使 official server 当时采用某版本，也不能凭年代替论文补写。
- Pedestrian/Cyclist 只出现在像素占比背景分析中，论文没有给两类检测 AP；不存在可供本项目 Ped/Cyc macro 直接比较的数据。

### F. 基线、完整方法、消融、效率（绝对值与 delta）

**同协议 PointRCNN 关键对照（Table 2/3）：**

| split/task | baseline PL++: PointRCNN | E2E: PointRCNN | 同列绝对 AP 点差 |
|---|---:|---:|---:|
| val, IoU .7, Moderate BEV | 64.0 | 65.7 | +1.7 |
| val, IoU .7, Moderate 3D | 50.1 | 51.7 | +1.6 |
| test, IoU .7, Moderate BEV | 58.0 | 58.8 | +0.8 |
| test, IoU .7, Moderate 3D | 42.4 | 43.9 | +1.5 |
| test, IoU .7, Hard BEV / 3D | 51.3 / 37.0 | 52.1 / 38.1 | +0.8 / +1.1 |

**PointRCNN 联合训练消融（Table 4，val IoU=.7 Moderate，BEV/3D）：**

- 不再训练任何子网：`64.0/50.1`；只继续训 Depth：`64.5/50.5`；只训 RPN：`63.9/50.0`；只训 RCNN：`63.4/50.4`。
- RPN+RCNN：`64.0/51.2`；Depth+RPN：`64.7/50.7`；Depth+RCNN：`65.5/51.0`；Depth+RPN+RCNN：`65.7/51.7`。这说明全联合在该指标最好，但各部分收益**不是可相加常数**，且有列由非完整组合取得最好值。

**PIXOR（Table 2/3/5）：**

- val Moderate，PL++ hard-quantized PIXOR → E2E soft-quantized PIXOR：IoU=.5 BEV `78.4→84.8`（+6.4）；IoU=.7 BEV `61.1→64.3`（+3.2）。
- Table 5 内部 soft-quantized pipeline 的全冻结 → Depth+Detector 联训：Moderate IoU=.5/.7 BEV `78.3/57.7→84.8/64.3`（+6.5/+6.6）。这个 delta 同时包含“允许两个子网协同更新”的效应，不应拆成两个模块各自的独立贡献。
- test Moderate BEV，PL++ PIXOR `48.3→51.7`（+3.4）。
- 速度：单 GPU、PyTorch，E2E PointRCNN 0.49 s/frame，E2E PIXOR 0.55 s/frame；其中 SDN 约 0.39 s/frame。参数量、FLOPs、FPS 的统一硬件对账未报告。[S11-A，Sec. 4.5]

### G. Ped/Cyc、small/far/occlusion 证据

- 没有 Pedestrian/Cyclist AP、没有按尺寸或距离的数值分层，也没有 CI。
- 论文在 Fig. 7 仅给定性例子：E2E 减少远车漏检、改善远车定位并减少一个场景的 FP；结论部分也承认 KITTI 标注的远目标少，认为更高分辨率且更多远车的数据可能带来更大收益。这只能算机制提示，不能当作 far AP 证据。[S11-A，Fig. 7；Conclusion]

### H. 失败、局限、泄漏与不可比风险

- 训练需要 stereo 和投影 LiDAR 深度监督，计算主要耗在 SDN；与单目 RGB YOLOv8 的部署预算不等价。
- validation/official test 有明显差距（例如 PointRCNN Moderate 3D 51.7 val vs 43.9 test），说明 3,769 val 上的方案选择不等于独立泛化证明。
- 没有 seed、方差、CI；AP recall 采样版本未明确；没有参数/FLOPs；只报 Car。这些都限制“稳定提升”的外推。
- 作者只说联合训练稳定需要平衡梯度和，并未排除学习率、训练时长、soft quantization 本身等替代解释。

### I. 与本项目 P3–P5、plain-P2、DCLI、fusion 的逐项区别

- 本文不使用 YOLOv8、P2/FPN、Ped/Cyc 2D head、DCLI 或门控融合；其 `PL` 是 pseudo-LiDAR，不是本项目的 P2 特征层。
- 可比的只有**科学结构**：两个原本分开优化的子系统联合后，梯度能否抵达正确位置、各损失量级是否失衡，决定最终协同。它支持我们记录 P2/P3/P4/P5 上的梯度覆盖、范数与方向，而不支持直接复制其 AP 或宣称发现“多模块负交互”首创。

### J. 可借鉴思想与引用/复用边界

- 可借鉴：在合并 DCLI 与 fusion 前，先量化每个损失在共享参数上的非零梯度覆盖率、梯度和/范数、层级分布；再让干预只对应已证实的冲突。
- 必须引用：end-to-end pseudo-LiDAR、differentiable CoR、soft quantization、检测目标反向塑造深度、按梯度量级调 loss weight。
- 代码为 MIT，可依法复用，但本项目若只需诊断应独立实现极小 hook；不得复制论文表述/结构图或把其机制包装成“原创 P2”。

### K. 不抄袭、可证伪的项目假设与最小实验

- **H1：** clean C/B/AB 中的 `AB-B<0` 主要集中在共享 P2 语义/校准参数；DCLI 启用后，fusion loss 与 DCLI calibration loss 在这些参数上的 norm ratio 或负 cosine 显著恶化，而 P3–P5 不出现同等变化。
- **H0：** 梯度覆盖、范数比、cosine 与 AB 退化无稳定对应；负交互应转向样本收益重叠、assignment 或评测方差解释。
- **最小实验：** 冻结现有 split/seed/schedule，不新训大模型；在同一 30-epoch C/B/AB screen 中按 epoch 记录 P2–P5 每分支梯度 support、L2 norm、pairwise cosine、激活样本数，并以 DCLI ramp 前后作预注册断点。只有诊断指标与 AP/FP/FN 的时序和层级同时对应，才测试一个最小 stop-gradient/ownership separation；否则该修复 NO-GO。

### L. 原始表、页码/路径与一句证据结论

- 正式论文：CVPR 2020 pp. 5881–5890；Methods/CoR 为 Sec. 3，实验为 Sec. 4；核心原始证据为 Fig. 2–4、Tables 1–5、Fig. 7。[S11-P][S11-A]
- arXiv 原稿源码中对应 `approach.tex` 与 `exp.tex`；本笔记所有数值均从表格源码逐列转录，未采用排行榜聚合网站。
- **证据结论（高置信）：** 这篇论文直接证明“联合优化不是把独立收益相加；梯度覆盖和量级会决定协同成败”，但它没有证明本项目当前负交互的根因，也没有给可比较的 Ped/Cyc 2D AP_R40。

---

## K12. MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation

### A. 元数据与原始入口

- Hansheng Chen, Yuyao Huang, Wei Tian, Zhong Gao, Lu Xiong；CVPR 2021，正式论文页 10379–10388；arXiv:2103.12605。[S12-P][S12-A]
- 官方代码由论文脚注明示，仓库许可证为 MIT。[S12-C]

### B. 研究问题与真正 gap

- 单目 3D 定位缺乏显式深度。传统 2D–3D 几何约束依赖少量 keypoints/box edges，在遮挡、截断时约束变少；dense correspondence 更稳，但既往依赖每个实例的完整 3D 模型/表面标签，真实道路类别级目标难以获得。[S12-A，Sec. 1]
- MonoRUn 的 gap 是：仅用普通 3D box/pose 标注，通过 re-projection 自监督学习每个 RoI 的 dense normalized object coordinates（NOC），并使网络预测的不确定性贯穿 reconstruction→PnP→pose covariance→最终 score。
- 论文还明确指出普通 Gaussian KL 在多任务训练中有一个优化缺陷：随着预测 `sigma` 变小，均值分支梯度权重增大，容易造成 loss imbalance。这与本项目 DCLI 和 fusion 的负交互属于直接相关的先例，不能再把“发现异方差损失会压制另一任务”泛化声称为首创。[S12-A，Sec. 3.3]

### C. 输入、网络流程、损失、匹配与后处理

1. 单张 RGB 输入；2D 基座是预训练 Faster R-CNN + ResNet-101，使用六层 FPN，并额外加入一个 upsampled level。每个预测 2D box 经 RoI Align 后进入 3D branch。[S12-A，Sec. 4.2]
2. global extractor 从高层 FPN 的 7×7 RoI 特征预测尺寸和 16-channel global latent；NOC decoder 将 latent 与局部卷积特征合并，输出 dense NOC 和 reprojection standard-deviation maps。latent 显式承载形状、遮挡、截断等全局信息。[S12-A，Fig. 3–4；Sec. 3.2]
3. 用 ground-truth pose 和相机内参把预测 object coordinates 投影回图像；Robust KL（RKL）按 aleatoric uncertainty 加权 reprojection error，使背景/异常点得到较低权重，无需 foreground mask。[S12-A，Eq. 2–7]
4. RKL 先把 Gaussian/Laplacian 思路组合为 Huber-like mixed KL，再用 batch 内 `E[1/sigma]` 的 EMA 对权重归一；网络预测 `log sigma` 避免直接优化 sigma 的梯度爆炸。[S12-A，Sec. 3.3]
5. 推理用 uncertainty-weighted PnP 求 pose，并以 inverse Hessian/Gauss–Newton 近似 pose covariance；一个 4维 calibration vector 校准 covariance。关键细节是 `p*` 在 calibration loss 中 **detach**，该 loss 只更新 calibration vector，原文已明确做“校准梯度隔离”。[S12-A，Eq. 8–11]
6. 3D localization score 由 pose uncertainty 与 network feature 经 MLP 预测，再乘 2D confidence；score target 是 3D IoU 的截断线性函数。最终用极低阈值 `3D NMS IoU=0.01`；不是 YOLO 2D NMS。[S12-A，Sec. 3.5、4.2]
7. 默认 loss 为 `L_2D + L_proj(RKL) + L_dim + L_score + 0.01 L_calib`；带 LiDAR 版本再加 sparse NOC supervision。可微 PnP 的 end-to-end training 仅用于消融，且是在自监督训练后追加 refinement，不是最终默认路线。[S12-A，Sec. 3.6]

### D. KITTI 数据与 split

- 原始 KITTI Object：7,481 training images、7,518 hidden test、80,256 个八类标注实例；研发用常见 3,712 train / 3,769 val（Chen/3DOP split）。[S12-A，Sec. 4.1]
- 主模型按 full training set 训练 32 epochs；split 版本训练 50 epochs；两张 RTX 2080 Ti、batch=6、AdamW、base LR 2e-4、cosine decay。E2E refinement 另加 15 epochs、LR 3e-5。[S12-A，Sec. 4.2]
- 带 LiDAR 版本使用 KITTI 前景点生成 sparse NOC targets，属于额外训练监督；不带额外监督版本才是纯 RGB+3D box 自监督 reconstruction。

### E. 类别、任务、IoU、AP 与难度

- 原文明示采用 KITTI official **40-point interpolated AP（AP_R40）**，按 Easy/Moderate/Hard；Car 3D IoU 阈值 0.7，Pedestrian/Cyclist 3D IoU 阈值 0.5。[S12-A，Sec. 4.1]
- 主 Table 1 是 Car `AP_3D`，不是 2D AP；Table 2 是 official test Ped/Cyc `AP_3D`。Supplementary Tables 6–7 另列 2D、orientation、3D、BEV，不能跨任务混算。
- 单 seed；无 CI；主消融 `mAP` 是 Car 的六个数（IoU=.5/.7 × Easy/Mod/Hard）平均值，不是 COCO mAP，也不是本项目两类 Moderate macro AP_R40。

### F. 基线、完整方法、消融、成本（绝对值与 delta）

**Car：无额外监督与 +LiDAR（Table 1）：**

| split/metric | w/o extra supervision | + LiDAR supervision | 绝对 AP 点差 |
|---|---:|---:|---:|
| test AP3D@.7 Easy | 16.04 | 19.65 | +3.61 |
| test AP3D@.7 Moderate | 10.53 | 12.30 | +1.77 |
| test AP3D@.7 Hard | 9.11 | 10.58 | +1.47 |
| val AP3D@.7 Easy | 17.26 | 20.02 | +2.76 |
| val AP3D@.7 Moderate | 12.27 | 14.65 | +2.38 |
| val AP3D@.7 Hard | 10.41 | 12.61 | +2.20 |

**核心消融（Table 3；Car 六指标平均 mAP）：**

- RKL + LiDAR + latent（默认）：31.21；RKL 自监督、无 LiDAR：28.57（-2.64）；只有 LiDAR、无自监督 reprojection：18.84（-12.37）。论文据此认为 LiDAR 更像 shape regularizer，自监督才是主干。
- SmoothL1 / Laplacian KL / mixed KL / RKL：26.35 / 29.47 / 30.05 / 31.21；RKL 比 LapKL +1.74，但这是同一六指标均值、单次消融。
- **负交互原始证据：** LapKL baseline 加 E2E refinement `29.47→29.73`（+0.26）；更强 RKL baseline 加同一 E2E `31.21→31.09`（-0.12）。说明一个附加优化路径可帮助弱 loss，却伤害强 loss，不能默认“更端到端”必然叠加。
- epistemic dropout 只采样 dimension branch：`31.21→31.47`（+0.26）；采样完整 reconstruction network：`31.21→31.16`（-0.05）。取消 global latent：`31.21→29.78`（-1.43）；将 latent 置零对 RKL 模型影响 -1.98，而对无 aleatoric 的 SmoothL1 仅 -0.02（Table 4）。
- scoring supplementary：2D score only 25.40；Monte Carlo uncertainty score 28.19（+2.79）；MLP 融合 uncertainty+feature 31.47（再 +3.28）。但这与其他组件同时存在，不应拆成论文外的可加总收益。
- 推理 0.070 s/frame（含 Monte Carlo 与 PnP；约 14.3 FPS 的算术换算，不是作者另测 FPS）。参数量与 FLOPs 未报告；速度硬件对齐不足。[S12-A，Table 1]

### G. Ped/Cyc、small/far/occlusion 证据

- official test AP3D@.5：Pedestrian `10.88/6.78/5.83`；Cyclist `1.01/0.61/0.48`（Easy/Moderate/Hard）。Ped Moderate 仅比 MonoPair 6.68 高 +0.10；Cyclist 远低于 MonoPSR 的 4.74 Moderate。作者明确将 Cyclist 失败归因于训练样本不足。[S12-A，Table 2]
- supplementary 中 +LiDAR vs self-only：Ped Moderate 3D `6.78 vs 6.53`（+0.25），Cyclist `0.61 vs 0.38`（+0.23）；但 Cyclist 2D Moderate 反而 `49.13 vs 50.32`（-1.19）。这证明额外 3D supervision 对不同类别/任务并非同向。
- 方法设计上 global latent 与 uncertainty 面向遮挡/截断；Fig. 7 用颜色展示不同 occlusion level，并画 95% pose ellipse，但没有按 occlusion、distance、small/far 的 AP 表。

### H. 失败、局限、泄漏与不可比风险

- 作者主动指出 DORN depth estimator 的训练集与 KITTI-Object validation 有重叠，故把 AM3D/PatchNet/D4LCN 相应 val 数字标灰，不能用作公平强基线。这是非常直接的数据用途警告。[S12-A，Table 1 caption]
- covariance calibration 在 train 上进行；论文承认模型在 unseen val 上过度自信，需经验 covariance scaling。它证明“预测 uncertainty”不等于“校准可靠”，也留下 scaling 是否利用 val 选择的风险。[S12-A，Fig. 8；Sec. 4.5]
- 50次 MC dropout、CPU Ceres PnP、3D NMS 与本项目实时 2D YOLO 的预算不同；没有参数/FLOPs、multi-seed、CI 或 independent confirmation。
- Ped/Cyc test 数值很低且 Cyclist 样本不足；不能从 Car 的消融平均外推 Ped/Cyc。

### I. 与本项目 plain P3–P5、plain-P2、DCLI、fusion 的逐项区别

- MonoRUn 的六层 FPN 额外上采样层是公开多尺度结构，不等于我们的 plain-P2 创新；主任务是 RoI-based monocular 3D detection，不是 YOLO 2D head。
- Robust KL、heteroscedastic uncertainty、batchwise uncertainty weight normalization和 calibration gradient detach 均已有直接先例。若 DCLI 含相同思想，必须引用且不能声称这些算子本身原创。
- 与我们最强的机制联系是两点：其一，RKL 明确解决 uncertainty loss 随 sigma 收缩造成的梯度量级失衡；其二，论文已把 pose 从 calibration loss detach，说明“语义/定位预测与校准所有权分离”至少在局部算子上已有先例。

### J. 可借鉴思想与引用/复用边界

- 可借鉴：对 DCLI uncertainty weight 做 EMA mean normalization；记录 sigma/weight 分布和 loss-gradient norm；把纯 calibration target detach，只允许校准小参数更新；用 confidence calibration 与 AP 两条证据共同验收。
- 必须引用：RKL 的 mixed Gaussian/Laplacian 形式、`E[1/sigma]` 归一化、log-sigma、uncertainty PnP/covariance、detached pose calibration、uncertainty-conditioned scoring。
- MIT 允许代码复用，但本项目应优先独立实现最小公式并保留引用；不能复制其结构图或把 stop-gradient 本身写成首创。

### K. 不抄袭、可证伪的项目假设与最小实验

- **H1：** DCLI 的校准梯度通过共享 reliability/semantic parameters 改变 fusion 的门控排序，造成 `AB-B<0`；将 calibration target 与 shared semantic anchor 隔离，只更新 task-specific calibration adapter，会在不损害 B 的前提下提高 AB-B，并同时降低对应共享参数的冲突频率。
- **H0：** isolation 不改善 AB-B，或 AP 改善但梯度/排序诊断不变；则主要根因不是 calibration gradient ownership，而更可能是样本收益重叠/assignment。
- **最小实验：** 同 seed、split、30 epochs 比 `B`、`AB`、`AB+calibration-stopgrad` 三条件；只改变 calibration gradient path。预注册主比较是 isolated-AB vs AB 和 isolated-AB vs B；同时记录 P2/P3/P4/P5 cosine/norm、门控样本集合 Jaccard、small/far FP/FN。若 AP 与机制指标不共同改善，NO-GO。

### L. 原始表、页码/路径与一句证据结论

- 正式论文：CVPR 2021 pp. 10379–10388；Methods Sec. 3，实验 Sec. 4；核心 Tables 1–4、Figs. 7–8；supplementary 的 scoring 与完整 official benchmark 表均包含在 arXiv 原稿。[S12-P][S12-A]
- 关键定位：Table 1 明写 AP_R40 与 DORN overlap 风险；Table 2 给 Ped/Cyc；Table 3 给 loss/E2E/epistemic/latent 消融；Eq. 11 明写 `p* is detached`。
- **证据结论（高置信）：** MonoRUn 是本项目 DCLI/负交互最直接的先例之一：它既给出 uncertainty loss 导致多任务梯度失衡的数学原因，也实际观察到更强 RKL 与 E2E 联用的 -0.12 退化；因此我们可以借其诊断和隔离原则，但不能把 KL、weight normalization 或 stop-gradient 本身主张为原创。

---

## K13. Geometry Uncertainty Projection Network（GUPNet）

### A. 元数据与作者纠正

- 正确作者是 **Yan Lu, Xinzhu Ma, Lei Yang, Tianzhu Zhang, Yating Liu, Qi Chu, Junjie Yan, Wanli Ouyang**；不是 Simonelli。ICCV 2021，正式论文页 3111–3121；arXiv:2107.13774。[S13-P][S13-A]
- 作者官方代码为 `SuperMHP/GUPNet`，MIT 许可证。[S13-C]

### B. 研究问题与真正 gap

- 单目 3D detector 常用几何投影 `depth=focal_length×3D_height/2D_height`。这个先验有用，但上游 2D/3D height 的小误差会被除法放大成大 depth error，既使 inference confidence 不可信，也会在训练早期把错误梯度传给下游 depth task。[S13-A，Sec. 1、3]
- GUPNet 的两个回答是：用概率形式传播 height/bias uncertainty，得到可解释的 depth uncertainty；按任务依赖图让下游任务在前置任务学好后逐步启用，而不是把所有 loss 从 epoch 1 同权合并。
- 这是“几何+不确定性+阶段化多任务训练”的直接先例；本项目不能泛称这些元素或 delayed loss 为原创。

### C. 输入、网络流程、损失、匹配与后处理

1. 单目 RGB→DLA-34→CenterNet 2D heatmap/offset/size heads；RoIAlign 后裁剪 object feature，并拼接 normalized coordinate map（CM）补回位置/尺度信息。
2. 基础 3D heads 预测 projected 3D-center offset、alpha orientation 与 3D dimensions。
3. GUP：令预测 3D height 为 Laplace `La(mu_h,sigma_h)`；按 `d_p=f*h_3d/h_2d` 把均值和方差解析传播到 projection depth，再加独立 Laplace bias `La(mu_b,sigma_b)`：`mu_d=mu_p+mu_b`，`sigma_d=sqrt(sigma_p^2+sigma_b^2)`。[S13-A，Eq. 1–4]
4. depth 用 Laplacian heteroscedastic loss `sqrt(2)|mu_d-d_gt|/sigma_d + log sigma_d`；inference confidence 为 `p_depth=exp(-sigma_d)`，最终 3D score `p_3d=p_depth*p_2d`。[S13-A，Eq. 5–7]
5. Hierarchical Task Learning（HTL）把任务分三级：2D detection→basic 3D heads→depth；每个 loss 权重从 0 到 1，以前置任务最近 K 个 epoch 的 loss-derivative 收敛程度调整。K=5；训练曲线显示 depth stage 约 epoch 11 才开始明显增权。[S13-A，Sec. 3.5、Fig. 5/7]
6. 论文沿用 CenterNet 检测/后处理；没有在正文给独立 NMS、label-assignment 或 score threshold 规格，记为 `unknown/inherited`，不能把其结果归因于新 NMS。

### D. KITTI 数据与 split

- 7,481 train、7,518 hidden test；ablation 用常见 3,712 train / 3,769 val，最终模型在 7,481 全训练集上训练并提交 official server。[S13-A，Sec. 4.1]
- 输入 380×1280、feature stride 4；DLA-34；batch 32、3×Titan XP、140 epochs；LR 1.25e-3，在 epoch 90/120 ×0.1；前 5 epochs linear warm-up；HTL K=5。[S13-A，Sec. 4.1]
- 作者称 GUPNet 本身 `Extra data=None`；只使用 KITTI 单目图与标准框/标定，不用额外 LiDAR/depth/temporal 数据。

### E. 类别、任务、IoU、AP 与难度

- 原文明示使用 **AP40** 而非偏置更大的旧 AP11；Car 阈值 IoU=.7，Pedestrian/Cyclist=.5，报告 3D/BEV 与 Easy/Moderate/Hard。[S13-A，Sec. 4.1]
- Table 1 为 official test `AP3D_R40` 三类；Table 2 为 val Car 的 3D/BEV@.7/.5；Table 3/4 为 val Car AP3D/APBEV@.7 消融。
- 单 seed、无 CI；不能把 official test 的不同方法排序当配对统计。

### F. 基线、完整方法、消融、效率（绝对值与 delta）

**official test AP3D_R40（Table 1）：**

| 类别 | GUPNet Easy/Mod/Hard | 最佳无额外数据基线 | GUPNet 对其绝对 AP 点差 |
|---|---:|---:|---:|
| Car @.7 | 20.11 / 14.20 / 11.77 | RAR-Net 16.37 / 11.01 / 9.52 | +3.74 / +3.19 / +2.25 |
| Ped @.5 | 14.72 / 9.53 / 7.87 | MonoPair 10.02 / 6.68 / 5.53 | +4.70 / +2.85 / +2.34 |
| Cyclist @.5 | 4.18 / 2.65 / 2.09 | MonoPair 3.79 / 2.12 / 1.83 | +0.39 / +0.53 / +0.26 |

- 摘要写的“Car +3.74、Ped +4.7”是各自 **Easy** 对最佳 no-extra-data 方法的差值，不是 Moderate，也不是统一 macro。Moderate 对应 +3.19/+2.85。
- 对使用 LiDAR supervision 的 CaDDN，Moderate 差值是 Car +0.79、Ped +1.39、Cyclist **-2.09**；不能说三类全面领先。[S13-A，Table 1]

**val 定位精度证据（Table 2）：**

- GUPNet vs MonoPair，Car AP3D@.7 Moderate `16.46 vs 12.30`（+4.16），APBEV@.7 `22.94 vs 18.17`（+4.77）。
- 但在更宽松 IoU=.5，AP3D Moderate `42.33 vs 42.39`（-0.06），APBEV `47.06 vs 47.63`（-0.57）。因此增益主要是**高 IoU 定位质量/排序**，不是宽松覆盖率整体增加；这一区分对本项目很重要。

**组件消融（Table 3，val Car Moderate）：**

| 设置 | 3D@.7 | BEV@.7 | 相对上一因果对照 |
|---|---:|---:|---|
| (a) baseline | 11.00 | 16.43 | — |
| (b) +CM | 12.44 | 18.32 | +1.44 / +1.89 |
| (c) +CM+UnC | 13.53 | 19.00 | 对(b) +1.09 / +0.68 |
| (d) +CM+GeP | 12.79 | 18.73 | 对(b) +0.35 / +0.41 |
| (e) +CM+UnC+GeP | 13.57 | 19.19 | 对(c) +0.04 / +0.19；Easy/Hard 3D反而 -1.46/-0.11 |
| (f) +CM+UnC+GeP+GeU | 15.70 | 20.80 | 对(e) +2.13 / +1.61 |
| (g) +CM+UnC+GeP+HTL | 15.63 | 21.32 | 对(e) +2.06 / +2.13 |
| (h) 全部 | 16.46 | 22.94 | 对(f) +0.76/+2.14；对(g) +0.83/+1.62 |

- 这张表本身含负/近零交互：在已经有 UnC 时仅加入 deterministic GeP，Moderate 几乎不变，Easy/Hard 3D下降；只有显式 GeU 或 HTL 才恢复并提高。这不是“模块 AP 可相加”的证据。
- loss controller（Table 4）：GradNorm / task uncertainty / HTL 的 Moderate 3D 为 `10.49/13.94/16.46`，BEV `14.74/19.45/22.94`。作者解释通用 controller 假设任务独立，但其网络是层级依赖，且 error amplification 使 loss scale 跨训练期大变。
- 单 Titan XP 平均 29.4 FPS；参数量、FLOPs、延迟分解未报告。

### G. Ped/Cyc、small/far/occlusion 证据

- 论文给出 train-val 实例总数：Car/Ped/Cyclist 约 `14,357/2,207/734`；作者用 Cyclist 样本少解释其相对弱表现。[S13-A，Sec. 4.2]
- 没有 small/far 的 AP 表。Fig. 8 定性展示遮挡和远距离 bad cases：GUPNet 对坏框给更大 uncertainty，baseline projection 仍可能给低 uncertainty；这是 calibration/案例证据，不是分层显著性。
- Heavy occlusion/truncation 下，Table 3 文本指出普通 projection 的 hard 指标下降；完整 GeU/HTL 改善，但无独立 occlusion-bin 统计。

### H. 失败、局限、泄漏与不可比风险

- 全部 ablation 在单一 3,769 val 上，未报告 seed/CI；official test 用 7,481 full training 后提交，测试虽隐藏但方法已在 val 反复选择。
- 摘要挑 Easy 类别的最大 +3.74/+4.7；若误读为 Moderate 会高估。Cyclist 对强 LiDAR baseline 明显落后。
- HTL 同时改变各 loss 的全程权重/有效训练时长，Table 3 不能完全分离“合理依赖顺序”与“更合适总优化预算”。
- 3D Car/Ped/Cyc AP 与本项目 2D Ped/Cyc macro AP_R40 不可直接比较；29.4 FPS 的 Titan XP 条件也不能直接作为 YOLO 延迟。

### I. 与本项目 plain P3–P5、plain-P2、DCLI、fusion 的逐项区别

- GUPNet 无 YOLO P2/FPN 检测头；CM 是 RoI 坐标图，GeP 是相机几何 depth projection，GeU 是解析 uncertainty propagation，均不是“原创 P2”。
- 它已公开组合 geometry、heteroscedastic uncertainty、confidence product 和 staged task learning；我们若采用这些公共思想必须引用。
- 对我们最直接的因果启示是：若 DCLI 依赖 fusion/2D 语义稳定，epoch 1 同时启用可能把尚未成形的上游误差放大；schedule 应由依赖关系和新鲜训练信号定义，而不是为了追正数任意延后。

### J. 可借鉴思想与引用/复用边界

- 可借鉴：把任务依赖图写清；基于前置任务 loss-slope/稳定度触发 DCLI ramp；分别看宽松/严格 IoU 来判断“覆盖”还是“定位/排序”；将 uncertainty 用于 score 前必须做校准验证。
- 必须引用：`d=f*h3d/h2d` uncertainty propagation、GeU、`exp(-sigma)` confidence、HTL 动态 loss schedule、CM/RoI geometry。
- MIT 允许复用，但本项目不需要搬入整套 3D projection；最多借鉴最小 schedule/diagnostic，并在原创声明中排除 delayed training 和 geometry uncertainty 本身。

### K. 不抄袭、可证伪的项目假设与最小实验

- **H1：** fusion 的语义/可靠性估计在早期尚未稳定时，DCLI calibration 会放大错误样本权重；仅在预先定义的 fusion-fit loss slope/排序稳定度过门后 ramp DCLI，可使 `AB-B` 回升，并减少 DCLI 启用点附近 P2 shared-gradient 冲突。
- **H0：** delayed/rule-based ramp 不能同时改善 `AB-B`、strict-IoU localization 与机制诊断；则训练时序不是主要根因。
- **最小实验：** 保持总 30 epochs 与其他超参一致，比较既有 schedule 与一个预注册 delayed schedule；trigger 只使用 fit loss EMA/排名 Jaccard，不能看 development AP。报告 AP_R40、IoU/定位误差分层、small/far、FP/FN、gradient norm/cosine；单 seed 只作筛查。

### L. 原始表、页码/路径与一句证据结论

- 正式论文：ICCV 2021 pp. 3111–3121；Methods Sec. 3；Experiments Sec. 4；Table 1（test三类）、Table 2（val多IoU）、Table 3（组件消融）、Table 4（loss controller），Figs. 4–8。[S13-P][S13-A]
- **证据结论（高置信）：** GUPNet 提供了“上游误差放大→下游不确定性/训练不稳→解析 uncertainty + dependency-aware schedule 修复”的完整先例；它支持我们优先验证时序与 calibration，而不支持把 geometry、uncertainty、delayed DCLI 或它的 AP 当成本项目原创/可加收益。

## K14. Delving Into Localization Errors for Monocular 3D Object Detection（MonoDLE）

### A. 元数据与原始入口

- Xinzhu Ma, Yinmin Zhang, Dan Xu, Dongzhan Zhou, Shuai Yi, Haojie Li, Wanli Ouyang；CVPR 2021，正式论文页 4721–4730；arXiv:2103.16237。[S14-P][S14-A]
- 作者官方实现为 `xinzhuma/monodle`，许可证为 MIT。许可允许在保留声明的条件下复用代码，但 projected-center supervision、远样本降权/删除和 IoU-oriented size loss 都已有直接先例，不能改名声称原创。[S14-C]

### B. 研究问题与真正 gap

- 论文先做 oracle replacement，而不是先堆模块：对 anchor-free 单目 3D baseline 的每个输出分支分别用 GT 替换，诊断究竟是 projected center、depth、3D size 还是 orientation 限制严格 IoU 的 3D AP。[S14-A，Sec. 3.2，Table 1]
- 真正 gap 是单目 3D 位置由像平面 projected center 与 depth 共同反投影，几像素中心误差在远距离会放大到超过 Car@IoU=.7 可容许的纵向误差；同时直接用完整 3D IoU loss 会让大尺度、难解的 localization 项压倒 size 等支路，使训练崩溃。[S14-A，Secs. 3.2、3.5]
- 作者据此只提出三个对应干预：以 projected 3D center 监督 coarse center，同时保留 2D 辅助任务；删除/降权极远训练样本；只对 3D size 重分配 IoU-oriented loss。它不是 P2/FPN、YOLO 2D Ped/Cyc 或选择性复检方法。

### C. 输入、网络流程、训练目标、匹配与后处理

1. 输入是单幅 RGB；baseline 采用 CenterNet 式 anchor-free one-stage detector、标准 DLA-34 backbone 和共享特征上的七个轻量预测 heads（每个 `3x3 conv + 1x1 conv`）。[S14-A，Sec. 3.1]
2. 2D 分支预测类别 heatmap、coarse center、2D center offset 和 2D size；3D 分支预测 projected 3D center offset、depth、3D size 与 orientation，再依据相机内参恢复 3D center。
3. 分类用 modified focal loss；2D center/size、3D center refinement 用 L1；depth 用 uncertainty modeling；orientation 用 12 个互不重叠 bins；baseline 3D size 用 L1，完整方法改用 `|| (s-s*)/s ||_1` 并动态补偿 loss magnitude，使其成为标准 L1 的维度重分配。所有 loss item 权重设为 1。[S14-A，Secs. 3.1、3.5]
4. projected-center 设计用 projected 3D center 监督 coarse center，同时仍从同一点回归真实 2D box center offset 和 2D box size；2D detection 是辅助任务，不是后处理模块。[S14-A，Sec. 3.3]
5. 远样本采用两种 object weight：hard rule 在 `d>60 m` 时置 0；soft rule 为 `1/(1+exp((d-c)/T))`，默认 `c=60,T=1`。[S14-A，Sec. 3.4]
6. 推理只在 center heatmap 上用 `3x3` max-pooling NMS，恢复 2D/3D boxes，并删除 confidence `<0.2` 的预测；没有 box-level NMS、二次 crop 或 P2 re-observation。[S14-A，Implementation]

### D. KITTI 数据、split、训练与独立性

- KITTI 含 7,481 trainval 和 7,518 hidden test；作者按常见 Chen split 划成 3,712 train / 3,769 validation。全部消融在该 validation 上，最终模型用完整 7,481 trainval 训练并提交官方 test。[S14-A，Sec. 4.1]
- 训练为两个 GTX 1080Ti、batch 16、140 epochs、Adam，初始 LR `1.25e-3`，epoch 90/120 各降 10 倍，weight decay `1e-5`，前 5 epochs warmup，约 9 小时；增强为 horizontal flip 与仅对 2D detection 的 random crop/scale。[S14-A，Implementation]
- 原文没有 seed 数、CI、图像 ID manifest 或 sequence-level independence 报告，均记为 `unknown`。同一 3,769 val 被用于 oracle、阈值与模块选择，不能当独立 confirmation。
- 作者明确指出部分竞品使用 DORN 预训练深度模型，而 DORN 的训练集与 KITTI 3D validation 重叠，因此在 validation 表中排除这些方法；这是原文直接的数据重叠警告。[S14-A，Sec. 4.2]

### E. 类别、任务、IoU、AP 与难度口径

- 论文明确使用 **AP40（40 recall positions）**，报告 3D、BEV 和部分 AOS；分 Easy/Moderate/Hard。[S14-A，Sec. 4.1]
- Car 的默认 IoU=.7，Pedestrian/Cyclist 的默认 IoU=.5；validation Car 还同时报告宽松 IoU=.5，用来区分覆盖与严格定位。[S14-A，Tables 3–5]
- 这是单目 **3D/BEV** 检测；Table 7 虽含 2D AP40 作为辅助任务诊断，仍不能与本项目 KITTI RGB 2D Ped/Cyc Moderate macro AP_R40 直接相减。

### F. 基线、完整方法、逐项消融与成本

**Oracle error diagnosis（Table 1，val Car Moderate 3D AP40）：**

| 替换条件 | AP40 | 相对 baseline |
|---|---:|---:|
| baseline | 11.12 | — |
| GT projected center | 23.90 | +12.78 |
| GT depth | 38.01 | +26.89 |
| GT 3D location | 78.84 | +67.72 |
| GT size | 11.96 | +0.84 |
| GT orientation | 11.88 | +0.76 |
| all GT | 99.97 | +88.85 |

- 右半表从全 GT 逐项换回预测：without GT projected center/depth/location/size/orientation 分别 `46.33/25.25/12.13/80.50/70.89`。这确认位置误差主导，但 oracle 绝不是可实现增益。[S14-A，Table 1]
- **原文内部矛盾需保留：** Table 1 的 GT projected-center 是 `23.90`，紧随表后的正文却写 `18.97`；不能静默替作者选一个。以上表格按正式 Table 1 记录，正文值另标为冲突。

**定位放大（Table 2）：** KITTI Car 平均 `[h,w,l]=[1.53,1.63,3.53]m`；在其他量正确且误差沿 length 的最宽容假设下，IoU=.7 最大纵向误差约 `0.62m`。60m 处像平面偏移 `(du,dv)=(2,2)/(4,2)/(6,2)/(6,4)/(8,2)/(8,6)` 对应 `0.24/0.38/0.54/0.61/0.70/0.85m`，说明 1–2 个 stride-4 feature pixels 已可能令远车失配。[S14-A，Table 2]

**official test Car（Table 3，AP40@IoU=.7）：**

- MonoDLE 3D Easy/Moderate/Hard=`17.23/12.26/10.29`；BEV=`24.79/18.89/16.00`；AOS=`93.46/90.23/80.11`。
- 相对表中无额外数据的先前最佳列：3D `+2.47/+2.27/+1.64`，BEV `+3.77/+4.06/+3.11`，AOS `+0.52/+3.21/+2.99`。正文把 Moderate AOS 增益写成 `+4.12`，但 Table 3 的显式 improvement row 是 `+3.21`，这是第二处需保留的内部不一致。
- 竞品并非全等协议：表中 Decoupled-3D/AM3D/PatchNet/D4LCN/Kinematic3D 标为 extra data=Yes；作者只在 No-extra-data 子组内给“previous best”差值。

**Car validation 与阈值敏感性（Table 4，MonoDLE 对 MonoPair）：**

| task / IoU / Moderate | MonoPair | MonoDLE | delta |
|---|---:|---:|---:|
| 3D @ .7 | 12.30 | 13.66 | +1.36 |
| BEV @ .7 | 18.17 | 19.33 | +1.16 |
| 3D @ .5 | 42.39 | 43.42 | +1.03 |
| BEV @ .5 | 47.63 | 46.87 | -0.76 |

- 全列 delta 为：3D@.7 `+1.17/+1.36/+1.26`；BEV@.7 `+.85/+1.16/+1.25`；3D@.5 `+.03/+1.03/-.18`；BEV@.5 `-.33/-.76/-.03`（表中 printed improvement row 对 Moderate 写 `-0.80`，由显示绝对值相减为 `-0.76`，应视作四舍五入/表内不一致）。收益主要出现在严格 IoU，不能概括为所有覆盖指标都提升。

**累计消融（Table 6，val Car BEV/3D AP40）：**

| variant | Easy | Moderate | Hard | Moderate 3D 增量 |
|---|---:|---:|---:|---:|
| baseline | 20.29 / 14.51 | 16.15 / 11.12 | 14.07 / 9.97 | — |
| + projected-center `p` | 23.10 / 15.78 | 18.15 / 12.65 | 16.11 / 10.62 | +1.53 |
| + `p` + IoU-size `I` | 23.89 / 16.12 | 18.34 / 12.97 | 16.69 / 10.99 | +0.32 |
| + `p` + `I` + sample rule `s` | 24.97 / 17.45 | 19.33 / 13.66 | 17.01 / 11.68 | +0.69 |

- 完整模型相对 baseline 的 Moderate 3D/BEV 是 `+2.54/+3.18`，但这是固定顺序的累计消融，不是三项独立 AP 可以任意相加。
- Table 7 证明任务关系敏感：no projected-center/no refinement + multitask 的 2D/3D Moderate=`92.31/1.04`；加 refinement=`92.31/10.81`；projected-center+refinement 但无 2D multitask=`88.79/10.30`；三者都有=`92.74/13.04`。共享 2D 辅助任务改善 3D，但移除它同时改变了监督与表征，不能直接映射为我们的 fusion 贡献。
- 速度为单 GTX1080Ti、batch1、`40ms/frame≈25 FPS`；训练约 9h。参数量与 FLOPs 未报告，记为 `unknown`。[S14-A，Table 3；Latency analysis]

### G. Ped/Cyc、small/far/occlusion 证据

- official test Pedestrian（Table 5，BEV/3D AP40@.5）Easy=`10.73/9.64`、Moderate=`6.96/6.55`、Hard=`6.20/5.44`；相对 MonoPair 的 Moderate `7.04/6.68` 为 `-0.08/-0.13`，没有改善。
- Cyclist Easy=`5.34/4.59`、Moderate=`3.28/2.66`、Hard=`2.83/2.45`；相对 MonoPair Moderate `2.87/2.12` 为 `+0.41/+0.54`。作者明确警告 Ped/Cyc training samples 少，结果会波动；无 seed/CI，不能把 Cyclist 正点估计当稳定机制证据。[S14-A，Table 5]
- 没有 KITTI 2D `small` AP，也没有按距离报告最终 AP。Table 8 反而显示作者从训练目标中删除远样本：7,481 trainval 的 28,742 个目标中，`>60m` 1,301（4.5%），`>65m` 767（2.7%）。因此总体 AP 上升不能证明远目标获益。
- Table 8 的 Car 3D AP40：baseline=`16.12/12.97/10.99`；hard `s=40`=`14.25/11.25/9.63`（全负）；hard `s=60`=`17.45/13.66/11.68`；soft `(c=40,T=1)`=`14.50/11.74/9.95`；soft `(60,1)`=`17.50/13.54/11.32`；soft `(60,5)`=`17.25/13.03/11.01`。方向对 threshold 高度敏感。
- Easy/Moderate/Hard 只由 2D height、occlusion、truncation 规则定义，不能当纯距离分层。Fig. 6/qualitative 有距离和遮挡案例，但不是独立 far/occlusion 统计。

### H. 失败、局限、泄漏、饱和与不可比风险

- `s=60` 是在同一 val 上选择的训练规则；Table 8 显示 40m 设置明显退化，且没有报告被删除 far 对象的 no-harm。这与本项目“改善 far/small 且 no-harm”目标存在根本冲突，不能把“删除难样本”作为主路线追总体正数。
- Oracle replacement 同时解除模型误差并使用 GT，回答上限/瓶颈而不回答可实现方法因果；不应写成模块提升。
- 论文无多 seed、paired CI 或独立 confirmation；Ped/Cyc 特别小且作者自报波动，不能据单次榜单得出稳定类间结论。
- test 表把多种 backbone、额外训练数据和不同后处理模型放在一起；只有作者累计 ablation 是较可控的内部对照。
- 3D Car AP40、BEV/AP40、AOS 和我们 2D Ped/Cyc macro AP_R40 任务不同；禁止数字相减或把 `+2.54` 当作我们的可实现 P2 增益。
- 论文定性案例承认仍有 FP，主要来自不精确 depth/center；未提供 FP/FN、calibration 或 latency variance。

### I. 与本项目 plain P3–P5、plain-P2、DCLI、fusion 的逐项区别

- MonoDLE 没有 P2 feature head、FPN/PAN fusion 或 multi-scale 2D assignment；其 projected-center、depth 和 3D size 都服务于相机坐标 3D box。
- MonoDLE 已公开“共享表征中的一个难 localization 子任务可能压倒其他 loss、直接完整 IoU 导致 collapse、应隔离到 size-specific objective”这一结构性观察。我们不能声称“多任务 loss 会负交互”本身首次发现。
- 它与 DCLI 的最相近点是：localization objective 必须匹配唯一可控误差源，并平衡 loss magnitude；与 fusion 的最相近点是：保留上游 2D 辅助语义可帮助下游 3D。不同点是本项目需证明 P2/fusion/DCLI 在 2D Ped/Cyc 上的具体互动，不能借 MonoDLE 的 3D 结果代证。
- 它的 strict-vs-loose IoU 结果提供一个诊断模板：若 P2 在宽松 IoU recall 不差、严格 IoU AP 下跌，则问题更像 localization/ranking；若宽松 recall 也跌，则优先查 coverage/assignment，而非继续换回归 loss。

### J. 可借鉴思想与引用/复用边界

- 可借鉴：先做 output substitution/固定预测诊断；同时报告宽松与严格 IoU；把 center error 按 object size/distance 归一化；在共享 loss 上记录范数支配；任何 sample filtering 都必须报告被过滤 stratum 的 no-harm。
- 必须引用：projected 3D-center supervision、2D auxiliary task、far-sample hard/soft weighting、IoU-oriented 3D-size loss，以及“full 3D IoU 被 localization error 压倒”的论证。
- 不应复用为主方案：为了提高 aggregate AP 删除远目标。我们研究对象正是 far/small Ped/Cyc，这会偷换目标；最多把 `drop-far` 作为预注册负对照，验证“总体指标能被牺牲目标子群抬高”的选择性报告风险。
- 官方代码虽为 MIT，若用任何实现须保留 license/attribution；更合理的是独立实现适用于 2D 的诊断统计，而不是移植整套 3D detector。

### K. 不抄袭、可证伪的项目假设与最小实验

- **H1（最高信息价值、无需训练）：** plain-P2 的总体 Moderate macro AP_R40 退化主要不是“没有新增小目标覆盖”，而是新增 P2 候选的定位/排序质量不足；相对 P3–P5，它在 small/far 的低 IoU (`.3/.5`) matched recall 上升，但在更严格 IoU、precision 或 score calibration 上丢失。
- **H0：** P2 在低 IoU recall 也没有增加，或错误不集中于 P2 candidates；则 localization/DCLI 不是首要修复，应转查 receptive field、assignment 与噪声覆盖。
- **最小实验：** 只复用同一 371 development 的冻结预测，不改权重；对 P3–P5、plain-P2、C/B/AB 按 class×small/far 输出 GT coverage、max recall@IoU `.3/.5/.7`、matched center/size error、FP/FN 和 score rank/calibration，并以 image-paired bootstrap 给差值 CI。先证明错误位置，再决定是否允许一个 localization/assignment 30-epoch screen。
- **负对照：** 仅在统计层模拟删除 `far` 或极小目标，展示 aggregate AP 是否虚增；该结果绝不算模型提升，而用于约束论文 no-harm 与防止复刻 MonoDLE 的目标牺牲。

### L. 原始表、页码/路径与一句证据结论

- 正式论文：CVPR 2021 pp. 4721–4730；Methods Sec. 3；Experiments Sec. 4；Table 1（oracle error analysis）、Table 2（center shift→meter error）、Table 3（official test Car）、Table 4（val 多 IoU）、Table 5（official test Ped/Cyc）、Table 6（累计消融）、Table 7（center/multitask）、Table 8（training-sample threshold）。[S14-P][S14-A]
- 原始 TeX 已逐项核对；所有显示 delta 均由同表绝对值复算，并保留 Table 1/正文、Table 3/正文及 Table 4 printed delta 的内部不一致。
- **证据结论（高置信）：** MonoDLE 最有价值的不是其 `+2.54`，而是一个可复用的研究顺序——oracle 确认瓶颈、严格/宽松 IoU 分开定位、唯一对应干预、并揭示 aggregate AP 可通过忽略远样本提高。它支持我们先做 P2 coverage/localization/ranking 审计，也强烈反对用删 far 或跨任务 AP 包装本项目提升。

## K15. MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection

### A. 元数据与原始入口

- ICCV camera-ready 作者为 Renrui Zhang, Han Qiu, Tai Wang, Ziyu Guo, Yiwen Tang, Xuanzhuo Xu, Ziteng Cui, Yu Qiao, Hongsheng Li, Peng Gao；ICCV 2023，正式页 9155–9166。主 PDF SHA256=`1669F27AB9CB72DE6736E6691FF5786828C8D51E0C84947B05202493110788C4`。[S15-P]
- official supplement 共 8 页，SHA256 已列于 [S15-S]。supplement 标题页仍列较早的 8 作者版本；arXiv v5 又是 10 作者。学术引用以正式 ICCV camera-ready 为准，不混用旧 README 的作者字段。[S15-S][S15-A]
- 官方仓库 `ZrrSkywalker/MonoDETR` 当前 HEAD=`6994b9f...`，许可证明确为 MIT（Copyright 2024 Renrui Zhang）。MIT 允许修改/分发但必须保留版权与许可；仓库还明确致谢 Deformable-DETR 和 MonoDLE。[S15-C]

### B. 研究问题与真正 gap

- 既有 center-guided 单目 3D 检测先定位 2D/投影中心，再用中心附近的局部特征回归 3D 属性；作者认为这缺少 scene-level spatial context 和长程 inter-object depth relations。[S15-P，pp. 9155–9157]
- 真正 gap 是：能否在不使用 dense depth/LiDAR 辅助监督、不用 anchors/NMS 的情况下，把 object-wise depth cues 作为全局查询的引导，并用 DETR 的一对一匹配直接输出 3D 对象。[S15-P，Abstract；Sec. 2]
- 论文的另一项关键观察不是“更多 loss 更好”，而是训练早期 3D 属性不准，`L3D` 会污染 Hungarian assignment；因此只允许较稳定的 `L2D` 决定语义所有权，再在匹配完成后优化全部 3D loss。[S15-P，p. 9159；S15-S，p. 5 Table 5]

### C. 输入、网络流程、损失、匹配与后处理

1. **输入与视觉支路。** 单幅 RGB 经 ResNet-50 得到 stride 8/16/32 特征；最高层 stride-32 作为 visual feature。当前官方代码把输入 affine resize 到 `1280×384`，但正式论文正文没有把这个尺寸绑定为 camera-ready run identity，因此“论文输入尺寸”严格记为未在正文声明，代码尺寸只作实现证据。[S15-P，pp. 9158–9159；S15-C]
2. **深度支路。** stride 8/16/32 特征统一到 stride 16 后逐元素相加，再经两个 `3×3 conv` 生成 depth feature；一个 `1×1 conv` 预测 `k+1` 类 foreground depth map。2D box 内像素被赋同一 object depth，重叠处取离相机最近对象；无需 dense depth map。[S15-P，p. 9158]
3. **深度离散。** `dmin=0m,dmax=60m,k=80`，采用 LID（linear-increasing discretization），让远处 bin 更宽；最后一类为 background。该设计承认远处误差更大，但并没有报告独立 far AP。[S15-P，pp. 9158、9160]
4. **双 encoder。** visual encoder 3 blocks，depth encoder 1 block；前者用 deformable attention 节省内存，后者用 vanilla global attention 建模非局部深度关系。visual embedding 为 stride-32，depth embedding 为 stride-16。[S15-P，pp. 9158–9160]
5. **depth-guided decoder。** 50 个 learnable object queries；每个 3-block decoder 按 `depth cross-attention → inter-query self-attention → visual cross-attention → FFN` 排列。depth positional encoding 是每米一个 learnable embedding，再按预测 depth 插值，而非普通 2D sinusoidal encoding。[S15-P，pp. 9158–9159]
6. **预测头。** query 输出 category、2D side distances、projected 3D center、depth、3D size、orientation；无预定义 mean-size anchor。supplement 说明 category 用 linear head，depth/size/orientation 用 two-layer MLP，2D size/center 用 three-layer MLP。[S15-S，p. 3]
7. **三路 depth。** 最终 `dpred=(dreg+dgeo+dmap(x3D,y3D))/3`；`dgeo=f*h3D/(t+b)`，`dmap` 从 foreground map 采样；用 Laplacian aleatoric uncertainty loss `sqrt(2)/sigma * |dgt-dpred| + log(sigma)`。[S15-S，p. 3, Eqs. 1–3]
8. **匹配的语义隔离。** Hungarian cost 只用 `L2D=2Lclass+10Lxy3D+5Llrtb+2LGIoU`；`L3D=Lsize3D+Lorien+Ldepth` 不参与 matching。匹配后总 loss 才是 paired `L2D+L3D`，再加 foreground depth-map focal loss。[S15-P，p. 9159 Eq. 6；S15-S，p. 4 Eqs. 4–5]
9. **推理。** 只保留 category confidence `>=0.2` 的 queries，用相机参数恢复 3D box；不使用 NMS 或 anchors。[S15-P，p. 9160]

### D. KITTI split、训练、模型选择与独立性

- KITTI 为 7,481 trainval / 7,518 hidden test；作者从 trainval 划 3,769 val，剩余 3,712 train，即常用 Chen split。主消融与结构选择都在 3,769 val，最终向 official test 提交。[S15-P，p. 9160]
- 训练：单 RTX3090、195 epochs、batch 16、AdamW，LR=`2e-4`，weight decay=`1e-4`，epoch 125/165 各乘 0.1；8 attention heads、hidden/MLP 256、50 queries。[S15-P，p. 9160]
- 为稳定训练，作者删除 depth `<2m` 或 `>65m` 的训练对象；foreground map 最大深度又设为 60m。没有报告这些距离 strata 的 no-harm。[S15-P，p. 9160]
- 论文没有多 seed、paired CI 或独立 confirmation。当前官方 config 写 `random_seed:444`，但不能证明全部 camera-ready 表格均由该 seed 产生；3712/3769 的 ID/SHA、sequence independence 未报告。[S15-C]

### E. 类别、任务、IoU、AP40 与难度口径

- 主指标为 **AP at 40 recall positions**，分别是 `AP3D` 与 `APBEV`，Easy/Moderate/Hard；主表以 Car 为主。[S15-P，p. 9160]
- Car official 3D/BEV 使用 IoU=.7；supplement 另报 val Car 的 BEV@.7、3D@.5、BEV@.5。Pedestrian/Cyclist official test 为 AP3D@IoU=.5。[S15-S，p. 4 Tables 2–3]
- 这不是 RGB 2D AP，也不是本项目 Ped/Cyc Moderate macro AP_R40；即使都叫 AP40，任务空间、类别和 IoU 仍不同，禁止直接相减。

### F. 绝对结果、逐项消融、负交互与成本

**主性能（main Table 2，p. 9161，Car AP40）：**

| split/task | Easy | Moderate | Hard | 表中对 second-best delta |
|---|---:|---:|---:|---:|
| official test AP3D@.7 | 25.00 | 16.47 | 13.58 | +2.53 / +1.08 / +0.85 |
| official test APBEV@.7 | 33.60 | 22.11 | 18.60 | +2.94 / +1.73 / +1.41 |
| val AP3D@.7 | 28.84 | 20.61 | 16.38 | +4.32 / +2.04 / +0.81 |

- 这些 `second-best` 不是同代码基线：test/val Moderate 的 second-best 多为使用 LiDAR dense depth 的 MonoDTR（15.39/18.57），其他列也可能是 CAD/extra-data 方法；是 cross-paper leaderboard gap，不是单组件 causal delta。
- supplement Table 2（p. 4）补充 val：Car Moderate `APBEV@.7=26.95`、`AP3D@.5=48.92`、`APBEV@.5=53.10`；对表中 MonoDTR 分别 `+1.60/+1.60/+0.63`。仍是 cross-method comparison。

**深度引导整体（main Table 5，p. 9162，val Car AP3D@.7）：**

| architecture | Easy | Moderate | Hard | full 对该变体 Moderate delta |
|---|---:|---:|---:|---:|
| full MonoDETR | 28.84 | 20.61 | 16.38 | — |
| w/o depth-guided transformer（同时删 depth predictor 与 transformer） | 19.69 | 15.15 | 13.93 | +5.46 |
| w/o transformer（仍有隐式 foreground-depth supervision） | 20.19 | 16.05 | 14.18 | +4.56 |
| w/o depth guidance（视觉 DETR，无 depth predictor/encoder/CA） | 24.14 | 17.81 | 15.60 | +2.80 |

- `+5.46` 同时改变架构、深度支路、local/global aggregation 和训练目标，不能称单一 depth 模块增益，也不能与其他列相加。

**主文 Tables 6–9（pp. 9162–9163）的结构敏感性：**

- depth encoder（Table 6）Moderate：global SA `20.61`；deformable SA `18.91`；two `3×3 conv` `18.36`；无 encoder `18.38`。full 对无 encoder `+2.23`。
- decoder 顺序（Table 7）Moderate：`D→I→V 20.61`；`I→D→V 19.28`；`I→V→D 18.85`；先把 D+V 相加再统一 CA `18.41`。相同组件仅顺序不同可差 `1.33–2.20 AP`，直接说明交互/信息所有权重要。
- depth map（Table 8）Moderate：foreground LID `20.61`；dense LID `19.85`；foreground UD `18.90`；foreground SID `18.95`。相对 dense `+0.76`，但官方仓库自报约 ±1 AP 随机性，单点不足以证明小增益稳定。
- positional encoding（Table 9）Moderate：meter-wise `20.61`；k-bin `19.68`；depth sin/cos `19.57`；2D sin/cos `18.63`；无 encoding `18.94`。相对无 encoding `+1.67`，仍无 seed/CI。

**supplement 直接负交互（p. 5 Table 5）：**

| Hungarian matching cost | Easy | Moderate | Hard | 对仅 L2D 的 Moderate 变化 |
|---|---:|---:|---:|---:|
| only `L2D` | 28.84 | 20.61 | 16.38 | — |
| `L2D + Lsize3D` | 27.13 | 19.21 | 15.93 | -1.40 |
| `L2D + Lorien` | 25.78 | 18.63 | 15.12 | -1.98 |
| `L2D + Ldepth` | collapse | collapse | collapse | 训练崩溃 |
| `L2D + L3D` | collapse | collapse | collapse | 训练崩溃 |

- 这是对“把不稳定辅助目标直接塞入共享 assignment 会伤害甚至崩溃”的强直接先例；matching 隔离本身不能声称是我们的首创。

**其他 supplement 消融：**

- depth prediction（p. 5 Table 4）Moderate：三路平均 `20.61`；only direct regression `16.83`；无 geometric depth `18.74`；无 depth-map sample `18.89`；无 uncertainty sigma `16.98`。
- transformer 容量（p. 5 Table 6）：visual blocks 2/3/4 的 Moderate=`18.73/20.61/20.04`；decoder blocks 2/3/4=`18.58/20.61/18.29`；FFN 256/512/1024=`20.61/18.93/19.07`。更深/更宽反而更差，作者归因 KITTI 训练样本有限；这反对无假设堆模块。
- depth error（p. 7 Table 7）：full `AP3D=20.61, mean depth error=1.35±2.07m`；w/o entire depth-guided transformer=`15.15,1.54±2.29m`。这里的 `±` 是对象 depth-error dispersion，不是 seed CI。
- 效率（main Table 3，p. 9161）：RTX3090、batch1，MonoDETR `38ms`、`62.12 GFLOPs`、test Moderate AP3D `16.47`；对比 MonoDLE/GUPNet/MonoDTR 为 `40/34/37ms`、`79.12/62.32/120.48 GFLOPs`。参数量未报告，记为 `unknown`。

### G. Ped/Cyc、small/far/occlusion 证据

- supplement Table 3（p. 4，official test AP3D@.5）：Pedestrian Easy/Moderate/Hard=`12.54/7.89/6.65`；Cyclist=`7.33/4.18/2.92`。
- 这不是 own-baseline ablation。Ped Moderate 低于 LiDAR-extra MonoDTR `10.18`，但高于表中无 extra-data 的 MonoPair `6.68`；Cyclist Moderate 高于列出的所有方法（CaDDN 3.41、MonoDTR 3.27、无-extra MonoGeo 2.93）。没有 seed/CI，作者同时承认两类训练样本稀少、直接从头学 size 较难。
- 主文声称 stride-8/16/32 融合有助于 small pattern，LID 有助于抑制远距离误差；但没有 small AP、distance-bin AP、far recall 或 no-harm 表。相反训练删除 `>65m`，所以不能声称 far 检测被验证改善。
- p. 9163 attention map 和 supplement p. 7 只有遮挡/截断与长距离 inter-object attention 的定性可视化；不是 occlusion/far 分层统计。

### H. 失败、局限、数据选择与不可比风险

- 官方 README 明确警告 KITTI 单目训练随机性可达 **±1 AP3D**，并公开三次 “stable version” val Moderate=`20.83/20.64/20.14`。因此主文中 `+0.63/+0.76/+0.81/+0.93` 量级的单点差，若无 paired seeds/CI，不能被我们当作稳定收益证据。[S15-C]
- 所有 architecture/depth/matching 选择都用同一 3,769 val；无独立 confirmation。official test 只确认 final system，并不能给每个消融独立性。
- 训练排除 `<2m/>65m` 且 depth map 截到 60m；对远目标的性能主张存在目标删减风险。论文没有报告被删 strata 或近/中目标 no-harm。
- 主要 `+1.08` test Moderate 是对不同实现且使用 LiDAR extra depth 的 MonoDTR，而 `+5.46` 是多因素删改；两者都不是我们所需的同协议单机制估计。
- main paper 的明确 limitation 是未讨论如何引入 multi-modal input；对本项目更关键但原文未解决的局限包括 Ped/Cyc 稀少、无多 seed/CI、无 distance/small strata。
- 版本信息有漂移：formal main、supplement、arXiv v5、当前官方 repo 的作者/稳定实现并非完全同一快照；复用时必须固定 commit，不能只写“官方代码”。

### I. 与本项目 plain P3–P5、plain-P2、DCLI、fusion 的逐项区别

- MonoDETR 没有 YOLO P2 detection head、PAN/FPN gate、DCLI 或二次复检；其 stride-8 feature 只参与 stride-16 depth feature fusion，最终 visual query 使用 stride-32 semantics。不能把它当“原创 P2”先例或把 3D AP 搬到 2D。
- 与 fusion 最接近的公开思想是 multi-scale additive feature fusion、foreground reliability cue 和 depth-first decoder ordering；它们都必须引用，不能换名。
- 与 DCLI/负交互最接近的直接先例是：不稳定 3D localization/depth loss 若获得 Hungarian matching 的语义所有权，会损失 `1.40–1.98 AP` 或训练崩溃；只让稳定 `L2D` 决定 assignment，再让 `L3D` 在已匹配 pairs 上学习。
- 我们仍可能有可辩护差异，但必须是经实验支持的**特定场景**：P2 可靠性融合与 DCLI calibration 在 2D road-small-target head 上共享估计器时的负交互、其时序/梯度/样本分配证据，以及最小语义所有权分离修复；“loss isolation”“2D stable matching”“depth/uncertainty”本身不是原创。

### J. 可借鉴思想与引用/复用边界

- 可借鉴：把“谁有权改变 assignment/gate”与“谁只优化已分配样本”分开；记录 assignment churn；不稳定信号先 stop-gradient/adapter 隔离；不同次序用同组件对照；较小数据优先轻量 1–3 block，而不是堆深网络。
- 必须引用：foreground object-wise depth map、LID、depth positional encoding、depth-first decoder、三路 depth average、Laplacian uncertainty、仅 `L2D` Hungarian cost、任何对应网络/损失实现。
- 不应直接移植：完整 DETR/深度 transformer。它会改 backbone、head、assignment、训练 195 epochs 和计算图，无法回答当前 fusion×DCLI 的唯一因果问题，也违反最小改动边界。
- 若借官方代码，必须保留 MIT notice/attribution并锁定 commit `6994b9f...`；但当前仓库为 2024+ stable implementation，仍需核对与 ICCV paper snapshot 的差异。

### K. 不抄袭、可证伪的项目假设与最小实验

- **H1：** 当前 fusion×DCLI 负交互的一部分来自 DCLI calibration/factor signal 反向改变了 fusion gate 或 shared semantic anchor 的样本所有权；让 detection/fusion 语义分支独占 gate/assignment，DCLI 只在已分配样本上经 task-specific adapter 学习，并 stop-gradient 回 shared reliability path，可降低 assignment churn/负梯度频率，同时使 `AB-B` 转正。
- **H0：** ownership isolation 不降低 P2/P3–P5 的 assignment churn、梯度冲突或 DCLI 启用点的 ranking shift，且 `AB-B` 不改善；则负交互不在共享 assignment/gate，应停止此修复并查样本收益重叠/容量竞争。
- **最小 30-epoch screen：** 只比较冻结 AB 与一个 `calibration-stop-gradient + tiny task adapter` AB-isolated；保持 3341/371 split、seed/初始化/预算一致，不引入 attention/新 neck。训练期记录 P2–P5 assignment Jaccard/churn、三路 gradient cosine/norm；结果报告 `AB-B`、Ped/Cyc macro AP_R40、small/far、FP/FN、strict/loose IoU、延迟。单 seed 只筛查，过门后才 seeds 17/29/41 与 paired CI。
- **成功因果门：** 机制指标（churn/冲突）按预注册方向改善且 `AB-B` 同时改善；只 AP 正或只 cosine 好均不能写因果。MonoDETR 只提供先例/设计约束，不提供我们的正结果。

### L. 原始页表、源码位置与一句证据结论

- 正式论文：Methods 从 p. 9157 开始；feature/depth transformer pp. 9158–9159；matching/loss p. 9159；settings p. 9160；Tables 2–3 p. 9161；Tables 4–8 p. 9162；Tables 7–9/limitations p. 9163。[S15-P]
- Supplement：Table 1 p. 1；attribute/loss p. 3；Car/Ped/Cyc Tables 2–3 与 loss weights p. 4；depth/matching/capacity Tables 4–6 p. 5；depth-error Table 7 p. 7。[S15-S]
- 官方代码实现证据：`configs/monodetr.yaml` 固定 seed 444、Car train target、loss/matcher weights；`lib/datasets/kitti/kitti_dataset.py` 固定 `1280×384`、三类定义并调用其 Python KITTI evaluator。它们是当前 HEAD 的实现事实，不自动等同 camera-ready run provenance。[S15-C]
- **证据结论（高置信）：** MonoDETR 最直接支持的不是“上 transformer 就能给我们 +5.46”，而是一个已公开且被负消融验证的原则：**不稳定定位目标不能直接取得共享 assignment 的语义所有权**。这与我们 fusion×DCLI 冲突高度相关，允许做最小 ownership-isolation 检验；同时它构成 novelty collision，禁止把 stop-gradient/稳定匹配/不确定性或普通多尺度融合本身写成首创。

## 增量进度

- [x] K11 End-to-End Pseudo-LiDAR：A–L 完成，原始 Tables 1–5 已逐项核对。
- [x] K12 MonoRUn：A–L 完成，主文 Tables 1–4 与 supplementary 完整 benchmark/scoring 已核对。
- [x] K13 GUPNet：A–L 完成，作者/协议已纠正，Tables 1–4 逐列核对。
- [x] K14 MonoDLE：A–L 完成，Tables 1–8、正文内部数值冲突与远样本风险已逐项核对。
- [x] K15 MonoDETR：A–L 完成，main pp. 9155–9166、supplement pp. 1–8、Tables 1–9/1–7 与官方代码许可/HEAD 已核对。
