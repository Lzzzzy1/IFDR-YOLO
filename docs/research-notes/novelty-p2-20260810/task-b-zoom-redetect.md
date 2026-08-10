# Task B — coarse-to-fine / zoom / crop / re-detection novelty collision audit

- **Task ID:** B / zoom-redetect
- **检索与核验日期:** 2026-08-10
- **知识截止:** `AS_OF=2026-08-10`
- **状态:** COMPLETE（用于 novelty 硬门；不是实验结果）
- **证据范围:** 原始论文（CVF、NeurIPS、arXiv）与作者/课题组官方代码仓库；未用综述、博客或聚合站作为结论证据。
- **本轮 Skills:** `deep-research`（来源分级、查询日志、反证审查、可访问性记录）；`academic-writing`（区分事实、推断、研究缺口与不可声称内容）。
- **冻结待比较设计:** 道路几何置信度 + 检测残差/不确定性 + 每个候选“多给像素会带来多少检测收益”的反事实价值 + 每帧至多一个 crop 的硬预算 + no-harm fallback。

## 0. 先给结论

**可辩护缺口：有条件存在，但当前宽泛声明已经撞车。是否允许按当前声明编码：NO-GO。**

不能再把下列思想声称为本项目原创：

1. “根据粗检测预测某区域在高分辨率重检后的检测收益”；Gao 等人的 **Dynamic Zoom-In Network（CVPR 2018）** 已用配对的 coarse/fine detections 监督 proposal 级 accuracy gain，并让区域动作最大化检测增益减像素成本。
2. “根据 fine-vs-coarse 检测性能差和计算/获取成本选择高分辨率 patch”；Uzkent 等人的 **Efficient Object Detection in Large Images Using Deep Reinforcement Learning（WACV 2020）** 已直接用每 patch 的 `Recall_fine - Recall_coarse` 与成本训练一次前向的 patch policy。
3. “用检测器难度/低置信度驱动 adaptive zoom”；**AdaZoom（TMM 2022）** 的 collaborative training 已把漏检/低置信检测反馈给 zoom policy。
4. “用地平线、道路/地面透视或 vanishing point 分配分辨率”；**FOVEA（ICCV 2021）** 的 horizon/spatial prior 与 **Two-Plane Perspective Prior（CVPR 2023）** 的 ground-plane / vanishing-point 几何重采样已覆盖这一大类。

仍可能辩护的窄缺口只能是一个**可证伪且非平凡的联合机制**：在道路场景中，对几何估计本身输出校准置信度；用严格 leakage-free 的候选级 potential-outcome / detection-risk estimand 估计“只允许一次复看”时的净收益；在每帧 `0-or-1 crop` 硬约束下允许 abstain；并用明确、可测的 fallback 保留 F0 结果以限制新增 FP/伤害。这个组合必须逐项对照 Dynamic Zoom-In、Uzkent 2020、AdaZoom、FOVEA/Two-Plane；“把这些已有模块串起来”本身不足以成为原创贡献。

## 1. 判定口径

为避免把相似词误判为相同机制，本审查固定五个组件：

| 组件 | 本审查中的严格定义 |
|---|---|
| G：道路几何置信度 | 当前帧道路走廊/地面透视的估计及其可靠性，而非仅固定地平线、纵向位置或 dataset KDE |
| U：检测残差/不确定性 | 来源于 F0 检测输出的残差、漏检风险或校准不确定性，用于选 action；不是论文中只做误差分析的 variance |
| V：候选级额外像素反事实价值 | 训练目标或 reward 显式比较同一候选/patch 在 coarse/F0 与 fine/re-observed 条件下的检测效用差；objectness、density、smallness 只是 proxy，不计为 V |
| B：每帧硬预算 | 推理时动作空间被硬限制为 `0 或 1 crop/frame`；软 FLOPs 惩罚、阈值停止、多 crop 固定 T 均不算 |
| N：no-harm fallback | 选错或 fine 结果较差时有显式 abstention/保留 F0 输出的机制与伤害指标；仅“平均 AP 没下降”或普通 NMS 不算 |

## 2. Collision matrix

符号：`✓` 已有同构/近同构；`△` 有相关先例但定义不同；`—` 未发现。风险是相对冻结设计的最大碰撞风险，不代表论文质量。

| 主要先例 | 核心机制 / selector 信号 | G | U | V | B | N | 最大撞车风险 |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Dynamic Zoom-In Network**, CVPR 2018 | R-net 从配对 coarse/fine detections 学 proposal accuracy gain；Q-net 选区域/尺度，reward=检测增益−像素成本 | — | △（用 coarse quality；并以 entropy 作对照） | **✓** | —（顺序多动作，阈值停止） | △（可预测负 gain，但选中后直接用 fine 替换 coarse） | **CRITICAL：V 本身直接撞车** |
| **Efficient Object Detection in Large Images Using DRL**, WACV 2020 | LR 图一次输出多个 HR patch 动作；每 patch reward 显式使用 fine/coarse recall 差、对象数和成本 | — | △（coarse/fine 输出与 entropy baseline） | **✓** | —（16 个 coarse patch + 4 个 subpatch 的二进制动作） | △（未选 patch 用 LR；选错时无显式回退） | **CRITICAL：V+成本+一次前向选择直接撞车** |
| **AdaZoom**, TMM 2022 | policy-gradient 生成可变位置/尺度/长宽比的多个 focus regions；object distribution reward；collaborative detector feedback | — | **△/✓**（漏检及低置信难例反馈） | △（difficulty proxy，不是同候选 fine−coarse outcome） | —（固定 T 个区域，论文示例为多区域） | — | **HIGH：U/difficulty+adaptive zoom 撞车** |
| **FOVEA**, ICCV 2021 | 固定输出尺寸的可微非均匀全局 warp；dataset spatial/horizon KDE 或前帧 detections 产生 saliency | △（固定 horizon/spatial prior） | — | — | △（一次全局 warp，不是一个 crop） | — | **HIGH：若 G 只是 horizon/纵向先验则撞车** |
| **Two-Plane Perspective Prior**, CVPR 2023 | ground plane + 上方平面 + vanishing point 构造 learnable geometry saliency，对远小目标分配更多采样 | **✓/△**（几何分辨率分配已存在；未见显式置信门） | — | — | △（一次全局 warp） | — | **HIGH：road/perspective-aware resolution allocation 撞车** |
| **Cascaded Zoom-In Detector**, CVPRW 2023 | 把 density crop 标成一个新类；首遍检测 crop，放大后第二遍检测并融合 | — | — | —（density/objectness） | —（可检出多个 crop） | △（融合 whole-image 与 crop detections，但无伤害保证） | **HIGH：两遍 detector-guided crop 的结构撞车** |
| **AutoFocus**, ICCV 2019 | coarse scale 预测 category-agnostic FocusPixels，形成 FocusChips 后只在细尺度运行 detector | — | — | —（small-object likelihood） | —（多 chip、多尺度） | — | MEDIUM-HIGH |
| **CornerNet-Saccade**, 2019 | 低分辨率 attention maps 预测 small/medium/large object locations，再取高分辨率 crop 检测 | — | — | —（object center/scale supervision） | —（多 crop） | — | MEDIUM |
| **ClusDet**, ICCV 2019 | CPNet 提 object-cluster regions，ScaleNet 估尺度，scale-normalized crop 交给 DetecNet | — | — | —（cluster density/scale） | —（多 cluster） | — | MEDIUM |
| **SAHI**, ICIP 2022 | 均匀重叠切片、逐片推理、坐标映射与 NMS；可选 whole-image inference | — | — | —（无选择器） | —（处理全部切片） | △（可融合全图，但非候选 no-harm） | LOW-MEDIUM；crop 本身绝非原创 |
| **ZoomDet**, 2026 | OffsetNet 学全局非均匀 warp；box magnification loss；预测框逆变换回原图 | — | — | —（GT box magnification，不是检测收益） | △（单张 warped image，不是 crop budget） | — | MEDIUM；“adaptive zoom”命名/全局 warp 撞车 |
| **SNIPER**, NeurIPS 2018 | 训练时按 GT 尺度选择 positive chips，并从 RPN proposals 采 negative chips；不是选择性推理 | — | — | — | — | — | LOW（训练/chip 先例） |
| **PatchDrop / Learning When and Where to Zoom**, CVPR 2020 | LR 图经 RL 选择多个 HR patches，优化分类准确率与像素成本 | — | △（分类 ambiguity） | △（任务效用，但不是 detection candidate utility） | — | △（two-stream 分类，不是检测 fallback） | MEDIUM（跨任务动态像素预算先例） |

## 3. 决定性证据：是否已经有人预测 detection utility/value？

### 3.1 是：Dynamic Zoom-In Network（Gao et al., CVPR 2018）

这是最直接的否证，不应弱化为“普通 zoom”或“objectness”。

- coarse detector 先处理下采样图；R-net 的输入来自 coarse detection results，论文明确称其输出为 zoom-in **accuracy gain**。
- 训练时同时运行 coarse 和 fine detectors，并配对相同 proposal；proposal `k` 的监督量是
  `|g_k-p_k^l| - |g_k-p_k^h|`。高分辨率分数更接近 ground truth 时为正，fine 更差时为负。
- 区域动作的即时 reward 是区域内 proposal gain 之和减去 `lambda * selected_pixels / total_pixels`；Q-net 学 cost-aware 长期 action value，并在多个位置与尺度间选择。
- 论文还把 coarse output entropy 作为 baseline。作者明确指出 entropy 只说明粗检测不确定，不能判断 zoom 后是否真的改善；R-net 通过 coarse/fine 相关性学习实际改善方向。
- 与冻结设计的差别：没有道路几何；不是 `≤1 crop/frame`，而是阈值停止的顺序多 crop；选中区域后直接用 fine detections 替换 coarse detections，没有显式输出级 fallback。作者只把 scene segmentation/context 作为未来可加入的 cue。
- **结论:** “candidate-specific counterfactual value of extra pixels”不能作为新概念声称；最多只能提出不同的 estimand、约束、校准或安全机制。

### 3.2 是：Efficient Object Detection in Large Images Using DRL（Uzkent et al., WACV 2020）

这是第二个独立、同样直接的否证。

- agent 只观察 LR image，在一次前向中输出多个 HR patch 的 binary actions；再由 coarse 或 fine detector 处理对应 LR/HR patch。
- 论文 reward 对每个 patch 显式比较 `Recall(fine, GT)` 与 `Recall(coarse, GT)`，再乘该 patch 的 object count，并加入 HR patch 数量/运行与采购成本。论文原式的核心为
  `sum_i (Recall_fine_i - (Recall_coarse_i + beta)) * N_i + cost(a)`。
- 作者明确写出：当 fine detector 在 HR patch 上的 gain 高于 coarse detector 在 LR patch 上的 gain 时 zoom；因此不是纯 objectness/density selector。
- 与冻结设计的差别：无 road geometry；动作是多 patch binary array，不是每帧至多一个；指标偏 recall（论文明确未计 precision），因而没有针对新增 FP 的 no-harm；两级 coarse/fine search 也不同于单一候选。
- **结论:** “检测效用差 + 计算预算 + 非顺序一次前向选择”也已有明确先例。

### 3.3 部分是：AdaZoom（Xu et al., TMM 2022）

- 基础 reward 以区域内 object distribution、尺度和布局质量驱动 policy-gradient 的 focus-region 生成。
- collaborative training 把检测器当前表现反馈给 zoom 模块：漏检得到最困难权重，低置信正确检测得到较高权重。它已经覆盖“用 detection residual/difficulty 而非单纯密度改善 zoom policy”的方向。
- 但它没有为同一候选构造 observed fine-minus-coarse outcome，也不估计额外像素的个体处理效应；因此是强 proxy 碰撞，而不是 V 的完全同构。

### 3.4 否：其余主要 zoom/crop 方法预测的是什么？

- **AutoFocus / CornerNet-Saccade:** 预测小目标存在位置、中心与尺度（objectness/smallness），不是“zoom 会增加多少检测效用”。
- **ClusDet / Cascaded Zoom-In Detector:** 预测 cluster/density crop，重视拥挤小目标，不比较同一 patch 的 coarse/fine detection utility。
- **FOVEA / Two-Plane Prior:** 用空间、时间或透视几何 saliency 分配像素；是 geometry/where prior，不是 candidate-specific treatment value。
- **ZoomDet:** 通过 GT boxes 的放大目标学全局变形；论文中的 detector variance 属于坐标变换误差分析，不是 selector uncertainty，也不是 gain target。
- **SAHI:** 无自适应 selector，全部切片处理。
- **PatchDrop:** 学分类任务的像素效用/ambiguity，证明动态像素预算的一般思想更早存在，但不等同 detection utility。

## 4. 方法逐项核验与官方代码/许可证

以下“代码未找到”只表示按本页查询在 `AS_OF` 日期未定位到论文作者明确链接的仓库，不证明互联网上绝对不存在。

| 方法 | 原始论文（Source-Type / Accessibility / Authority） | 作者/官方代码与许可证（若可见） | 核心同/异点 |
|---|---|---|---|
| Dynamic Zoom-In Network | [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Gao_Dynamic_Zoom-In_Network_CVPR_2018_paper.html), [arXiv](https://arxiv.org/abs/1711.05187)；Academic paper / Public / 10 | 未定位到官方仓库；WACV 2020 论文也明确称其 code 当时不公开；license N/A | V 直接相同；无 G/B/N，U 仅相关 |
| Efficient Object Detection in Large Images Using DRL | [WACV 2020 paper](https://openaccess.thecvf.com/content_WACV_2020/html/Uzkent_Efficient_Object_Detection_in_Large_Images_Using_Deep_Reinforcement_Learning_WACV_2020_paper.html), [arXiv](https://arxiv.org/abs/1912.03966)；Academic paper / Public / 10 | [author repo](https://github.com/uzkent/EfficientObjectDetection)；Author repository / Public / 9；根目录无 LICENSE，按默认版权处理，**不得复制复用** | patch-level V+cost 直接相同；无 G/B/N |
| AdaZoom | [arXiv](https://arxiv.org/abs/2106.10409)（对应 IEEE TMM 2022）；Academic paper / Public / 9 | 定向查询未定位到作者代码；license N/A | U/difficulty+adaptive zoom 强相关；V 仅 proxy |
| FOVEA | [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.pdf), [arXiv](https://arxiv.org/abs/2108.12102)；Academic paper / Public / 10 | [author repo](https://github.com/tchittesh/fovea), [LICENSE](https://raw.githubusercontent.com/tchittesh/fovea/main/LICENSE)；MIT | horizon/spatial prior 与 G 有碰撞；全局 warp，无 U/V/B/N |
| Learned Two-Plane Perspective Prior | [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.html), [arXiv](https://arxiv.org/abs/2303.14311)；Academic paper / Public / 10 | [author repo](https://github.com/geometriczoom/two-plane-prior), [LICENSE](https://raw.githubusercontent.com/geometriczoom/two-plane-prior/main/LICENSE)；MIT；README 明确致谢/派生自 FOVEA 部分代码 | road/ground-plane/perspective allocation 已有；未见 geometry confidence gate、V/B/N |
| Cascaded Zoom-In Detector | [CVPRW 2023 paper](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Meethal_Cascaded_Zoom-In_Detector_for_High_Resolution_Aerial_Images_CVPRW_2023_paper.pdf), [arXiv](https://arxiv.org/abs/2303.08747)；Academic paper / Public / 10 | [author repo](https://github.com/akhilpm/DroneDetectron2), [LICENSE](https://raw.githubusercontent.com/akhilpm/DroneDetectron2/main/LICENSE)；MIT，版权注明 Facebook 与 LIVIA-ETS | detector 本身预测 density crop 并二次推理；无 V/B/保证式 N |
| AutoFocus | [ICCV 2019 paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html), [arXiv](https://arxiv.org/abs/1812.01600)；Academic paper / Public / 10 | [official joint SNIPER/AutoFocus repo](https://github.com/MahyarNajibi/SNIPER), [LICENSE](https://raw.githubusercontent.com/MahyarNajibi/SNIPER/master/LICENSE)；Apache-2.0，且含 MXNet 等第三方来源，复用需保留归属 | object-likelihood/smallness，多 chip；非 V |
| CornerNet-Saccade | [CornerNet-Lite paper](https://arxiv.org/abs/1904.08900)；Academic paper / Public / 9 | [official Princeton repo](https://github.com/princeton-vl/CornerNet-Lite), [LICENSE](https://raw.githubusercontent.com/princeton-vl/CornerNet-Lite/master/LICENSE)；BSD-3-Clause | attention/object scale selector，多 crop；非 V |
| ClusDet | [ICCV 2019 paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Clustered_Object_Detection_in_Aerial_Images_ICCV_2019_paper.html), [arXiv](https://arxiv.org/abs/1904.08008)；Academic paper / Public / 10 | 论文只写代码将放在作者 GitHub；本次未定位到对应明确 repo；license N/A | cluster/density + scale，不是 detection gain |
| SAHI | [ICIP 2022/arXiv paper](https://arxiv.org/abs/2202.06934)；Academic paper / Public / 9 | [official OBSS repo](https://github.com/obss/sahi), [LICENSE](https://raw.githubusercontent.com/obss/sahi/main/LICENSE)；MIT | uniform slicing；无 selector；可选全图融合不构成 no-harm |
| ZoomDet | [arXiv 2026 paper](https://arxiv.org/abs/2602.07512)；Academic paper / Public / 9 | [paper-linked repo](https://github.com/twangnh/zoomdet_code), [LICENSE](https://raw.githubusercontent.com/twangnh/zoomdet_code/master/LICENSE)；Apache-2.0；license/代码保留 OpenMMLab/MMDetection 来源，复用必须核第三方归属 | box-driven 全局 warp；非 U/V/B/N |
| SNIPER | [NeurIPS 2018 paper](https://papers.nips.cc/paper/2018/file/166cee72e93a992007a89b39eb29628b-Paper.pdf), [arXiv](https://arxiv.org/abs/1805.09300)；Academic paper / Public / 10 | [official repo](https://github.com/MahyarNajibi/SNIPER), [LICENSE](https://raw.githubusercontent.com/MahyarNajibi/SNIPER/master/LICENSE)；Apache-2.0/第三方归属同上 | selective training chips；不等于推理时 V |
| PatchDrop | [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Uzkent_Learning_When_and_Where_to_Zoom_With_Deep_Reinforcement_Learning_CVPR_2020_paper.pdf), [arXiv](https://arxiv.org/abs/2003.00425)；Academic paper / Public / 10 | [official Ermon group repo](https://github.com/ermongroup/PatchDrop)；未见 LICENSE，按默认版权处理，**不得复制复用** | 分类任务的动态 HR patch policy；非 detection V |

## 5. 精确检索日志

所有查询日期均为 **2026-08-10**。搜索结果页仅用于发现来源；上文结论只绑定原始论文、作者主页和作者/官方仓库。

### 5.1 Dynamic value / coarse-to-fine

1. `site:arxiv.org "Dynamic Zoom-in Network" object detection large images`
2. `"Dynamic Zoom-in Network for Fast Object Detection in Large Images" GitHub Mingfei Gao`
3. `github "Dynamic Zoom-in Network" R-net Q-net`
4. `site:github.com "Qnet-CNN" "Rnet" zoom-in detection`
5. `Mingfei Gao publications Dynamic Zoom-in Network project page`
6. `site:umiacs.umd.edu "Dynamic Zoom-in" code`
7. `site:openaccess.thecvf.com/content_WACV_2020 "Efficient Object Detection in Large Images Using Deep Reinforcement Learning"`
8. `site:arxiv.org "Efficient Object Detection in Large Images Using Deep Reinforcement Learning"`
9. `site:github.com Uzkent EfficientObjectDetection large images reinforcement learning`
10. `"Learning When and Where to Zoom" object detection deep reinforcement learning official paper`
11. `site:github.com Uzkent PatchDrop "Learning When and Where to Zoom"`

### 5.2 Adaptive zoom / crop / cascade / density

12. `site:arxiv.org small object detection adaptive zoom crop re-detection high resolution primary paper`
13. `"AdaZoom" official GitHub Adaptive Zoom Network`
14. `site:arxiv.org/abs/2106.10409 AdaZoom "1-c_i" false negative reward`
15. `"AdaZoom" "false negative" confidence score reward`
16. `"Adaptive Zoom Network for Multi-Scale Object Detection" confidence score reward`
17. `"cascaded zoom-in" detector high resolution aerial images paper`
18. `site:github.com/akhilpm/DroneDetectron2 LICENSE`
19. `site:openaccess.thecvf.com ClusDet small object detection large images paper`
20. `"Clustered Object Detection in Aerial Images" official code GitHub authors`
21. `site:github.com/fyangneil ClusDet code`
22. `"CornerNet-Saccade" paper official code`
23. `site:github.com/princeton-vl CornerNet-Lite LICENSE`
24. `"AutoFocus: Efficient Multi-Scale Inference" official code GitHub`
25. `site:github.com "AutoFocus" "FocusPixels" Najibi`
26. `site:github.com/MahyarNajibi SNIPER LICENSE`
27. `"SNIPER: Efficient Multi-Scale Training" official GitHub`

### 5.3 Slicing / warping / geometry

28. `"Slicing Aided Hyper Inference" paper SAHI official GitHub`
29. `"FOVEA: Foveated Image Magnification for Autonomous Navigation" arxiv`
30. `"FOVEA: Foveated Image Magnification" GitHub tchittesh`
31. `site:openaccess.thecvf.com FOVEA Foveated Image Magnification Autonomous Navigation`
32. `"FOVEA" "KDE-SI" "KDE-SD" saliency`
33. `"FOVEA" "anti-cropping" "temporal" object predictions`
34. `"Learned Two-Plane Perspective Prior Based Image Resampling" official code`
35. `site:github.com "Two-Plane Perspective" object detection resampling`
36. `site:openaccess.thecvf.com "Learned Two-Plane Perspective Prior"`
37. `Anurag Ghosh CVPR 2023 geometric zoom GitHub`
38. `"geometric_zoom" CVPR2023 github`
39. `"ZoomDet" small object detection paper official code`

## 6. 原创性与代码合规硬门

1. **表述:** 不得把 crop、P2、zoom、coarse-to-fine、density selection、geometry-aware sampling、uncertainty-guided compute 或 fine-vs-coarse utility prediction 单独写成原创。
2. **引用:** F4 中只要出现预测 zoom gain / detection value，必须至少正面区分并引用 Gao 2018 与 Uzkent 2020；出现 detector-difficulty selector 必须区分 AdaZoom；出现道路/透视像素分配必须区分 FOVEA 与 Two-Plane Prior。
3. **代码:** 本项目应独立实现。无 LICENSE 的 Dynamic Zoom-In、Uzkent EfficientObjectDetection/PatchDrop 等仓库不得复制代码；MIT/BSD/Apache 仓库即使允许复用，也必须保留许可、版权和第三方来源，并在决定复用前逐文件核验。
4. **结构图与文字:** 不复制既有论文结构图、公式表述或贡献文字；可以引用后重新推导自己的 estimand 与实验设计。
5. **实验:** 上述先例的论文数字不能转写成项目结果；本项目任何 +AP 只能绑定自身原始结果、配置、输入清单和 hash，负结果必须保留。

## 7. 允许重新进入编码门之前必须改写的贡献

当前“道路几何 + uncertainty + value-of-resolution + budget + fallback”的并列式声明不足以越过 collision gate。至少要先落下并可证伪地回答：

1. **Estimand 不同在哪里？** 不能仍是 Gao 的 proposal score-error delta 或 Uzkent 的 patch recall delta 换名。候选价值需预先定义是否为 Moderate Ped/Cyc AP surrogate、FN/FP risk delta、或其他可验证量，并说明为何适合一 crop 决策。
2. **几何置信如何改变决策？** 不能只是 horizon/vertical prior；要有当前帧 road-corridor/vanishing-point 可靠度与退避条件，并以 F1 唯一检验其因果作用。
3. **硬预算是否内生？** action space 必须原生为 `abstain 或恰好一个候选`，而不是训练后 top-1 截断一个原本多 crop 的 selector。
4. **no-harm 是什么机制与指标？** 必须明确 F0 预测何时保留、fine 预测何时拒绝/融合，并报告全体与 small/far 分层的 FP、AP、延迟；不能把平均 AP 上升称为 no-harm。
5. **为何联合不是显然拼装？** 每个新增组件只能回答一个独立假设；若没有区别于四个关键先例的技术关系或风险约束，则应 NO-GO，而不是换名。

可供主审进一步收敛的一句**候选、尚未获准**的可证伪声明是：

> 在固定为每帧至多一次高分辨率复看的道路检测中，校准的道路几何可靠度与 leakage-free 的候选净检测风险差，能否比 objectness/density、entropy、geometry-only 及既有 coarse–fine gain 基线更准确地选择唯一 action，并通过显式保留 F0 的退避规则降低新增 false positives，而不是只提高被选小目标的召回？

这句话是研究问题，不是原创性或正结果保证。

## 8. 反证审查与边界

- **最强反方 1：组合可能显而易见。** Dynamic Zoom-In 已建议把 scene segmentation/context 加入 gain prediction；把 road geometry 接到已有 gain selector 上，可能只是显然特化，不能靠“首次组合”自动成立。
- **最强反方 2：硬 top-1 与 fallback 可能只是工程约束。** 若没有新的风险建模、可证明/可检验的 abstention 规则或新的 estimand，`≤1 crop` 和保留 baseline 可能不足以构成论文方法贡献。
- **最强反方 3：现有 selector 已覆盖多种信号。** Uzkent 2020 已有 detector utility + cost，AdaZoom 已有 detector difficulty，FOVEA/Two-Plane 已有 spatial/temporal/geometry；审稿人很可能把 F4 看成这些先例的道路场景拼装。
- **边界:** 本审查覆盖英文公开原始论文及作者代码，未声称穷尽专利、非英文论文、未公开工业系统或 `AS_OF` 之后成果。代码“未找到”不是不存在证明。
- **结果边界:** 文献缺口只决定是否值得做实验，不保证 F4 对 F0 `+4.0 AP` 或完整系统对原始 P3–P5 `+5.0 AP`；O1/后续 GO/NO-GO 仍须由项目原始数据决定。

## 9. 最终门判定

| 问题 | 判定 |
|---|---|
| 是否已有“真实检测收益而非 objectness/density/geometry”的预测？ | **有。** Gao 2018 为 proposal score-error fine−coarse gain；Uzkent 2020 为 patch recall fine−coarse gain。 |
| 冻结设计是否与单篇论文完全同构？ | 未发现单篇同时具备校准道路几何、candidate net value、严格 `0-or-1`、显式 no-harm fallback。 |
| 当前宽泛贡献声明能否辩护？ | **不能。** 核心 V 与 geometry 两条支柱分别有直接先例，U 也有强先例。 |
| 是否允许按当前声明开始 F1–F4 新实现？ | **NO-GO。** 先由 Sol/ultra 将贡献收窄为不同 estimand + geometry-confidence abstention + 原生 one-action risk constraint，并逐条通过上述反证。 |
| 若无法提出非平凡差异怎么办？ | 保存本矩阵并停止原创路线；不得改名包装已有 gain selector / crop pipeline。 |

