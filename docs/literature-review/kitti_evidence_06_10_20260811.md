# KITTI 冻结清单 K06–K10：原始证据卡

- 任务：为主代理的 15 篇 KITTI 双人独立深读提供 K06–K10 原始证据；不代替主结论。
- AS_OF：2026-08-11（Asia/Shanghai）
- 模式：General Research / Standard deep read
- 来源门：仅公开可访问的原论文、会议官方 PDF、作者官方代码库及 KITTI 官方页面；博客、排行榜二手转述和检索摘要不作为数值证据。
- 口径门：2D/BEV/3D、Car/Pedestrian/Cyclist、IoU、AP11/AP40、validation/test、Easy/Moderate/Hard 必须分开；不同协议的 AP 禁止相减。
- 写作门：普通多尺度、P2、动态卷积、Wasserstein 损失或语义融合不改名冒充原创；代码复用必须核许可证。
- 反方审查：论文提升可能来自不同传感器、额外监督、较弱基线、旧 AP11、官方 test/私有 split 差异，或只在 Car 而非 Pedestrian/Cyclist 上成立。

## 进度

| ID | 论文 | 状态 | 下一步 |
|---|---|---|---|
| K06 | Song et al., TLL + Temporal Feature Aggregation | **完成／建议替换** | 仅把 KITTI 当作标注偏差案例，没有完整 KITTI 检测实验 |
| K07 | Han et al., Wasserstein Loss-Based Deep Object Detection | **完成** | KITTI 为自定义随机三折 7 类 mAP，非官方 AP_R40 |
| K08 | Simon et al., Complexer-YOLO | **完成** | RGB+LiDAR 3D/BEV official AP11，非 RGB 2D P2 |
| K09 | Brazil & Liu, M3D-RPN | **完成** | 严格拆分 paper AP11 与官网当前 AP_R40 |
| K10 | Ding et al., D4LCN | 进行中 | 原文 Methods/Experiments |

## 来源登记（局部编号）

> 每条来源格式：标题｜URL｜Source-Type｜Accessibility｜出版/版本时间｜检索日期｜Authority。

[1] Song et al., *Small-scale Pedestrian Detection Based on Somatic Topology Localization and Temporal Feature Aggregation*｜https://arxiv.org/pdf/1807.01438｜Source-Type: academic（ECCV 2018 原论文公开版）｜Accessibility: public｜2018-07-04｜检索：2026-08-11｜Authority: 10/10

[2] ECCV 2018 Open Access：Song et al. 论文页｜https://openaccess.thecvf.com/content_ECCV_2018/html/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.html｜Source-Type: official｜Accessibility: public｜2018｜检索：2026-08-11｜Authority: 10/10

[3] Mao et al., *What Can Help Pedestrian Detection?*｜https://arxiv.org/pdf/1705.02757｜Source-Type: academic（CVPR 2017 原论文公开版）｜Accessibility: public｜2017-05-08｜检索：2026-08-11｜Authority: 10/10

[4] CVPR 2017 Open Access：Mao et al. 论文页｜https://openaccess.thecvf.com/content_cvpr_2017/html/Mao_What_Can_Help_CVPR_2017_paper.html｜Source-Type: official｜Accessibility: public｜2017｜检索：2026-08-11｜Authority: 10/10

[5] Han et al., *Wasserstein Loss-Based Deep Object Detection*｜https://openaccess.thecvf.com/content_CVPRW_2020/papers/w60/Han_Wasserstein_Loss-Based_Deep_Object_Detection_CVPRW_2020_paper.pdf｜Source-Type: academic（CVPR Workshops 2020 原论文）｜Accessibility: public｜2020-06｜检索：2026-08-12｜Authority: 10/10

[6] CVPRW 2020 Open Access：Han et al. 论文页｜https://openaccess.thecvf.com/content_CVPRW_2020/html/w60/Han_Wasserstein_Loss-Based_Deep_Object_Detection_CVPRW_2020_paper.html｜Source-Type: official｜Accessibility: public｜2020｜检索：2026-08-12｜Authority: 10/10

[7] Simon et al., *Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds*｜https://openaccess.thecvf.com/content_CVPRW_2019/papers/Autonomous%20Driving/Simon_Complexer_YOLO_Real-Time_3D_Object_Detection_and_Tracking_on_Semantic_CVPRW_2019_paper.pdf｜Source-Type: academic（CVPR Workshops 2019 原论文）｜Accessibility: public｜2019-06｜检索：2026-08-12｜Authority: 10/10

[8] CVPRW 2019 Open Access：Simon et al. 论文页｜https://openaccess.thecvf.com/content_CVPRW_2019/html/Autonomous_Driving/Simon_Complexer_YOLO_Real-Time_3D_Object_Detection_and_Tracking_on_Semantic_CVPRW_2019_paper.html｜Source-Type: official｜Accessibility: public｜2019｜检索：2026-08-12｜Authority: 10/10

[9] KITTI Object Evaluation 官方页｜https://www.cvlibs.net/datasets/kitti/eval_object.php｜Source-Type: official benchmark protocol｜Accessibility: public｜持续更新；2019-10-08 起由 11 个 recall positions 攀至 40 个｜检索：2026-08-12｜Authority: 10/10

[10] Brazil & Liu, *M3D-RPN: Monocular 3D Region Proposal Network for Object Detection*｜https://openaccess.thecvf.com/content_ICCV_2019/papers/Brazil_M3D-RPN_Monocular_3D_Region_Proposal_Network_for_Object_Detection_ICCV_2019_paper.pdf｜Source-Type: academic（ICCV 2019 原论文）｜Accessibility: public｜2019-10｜检索：2026-08-12｜Authority: 10/10

[11] Brazil & Liu 作者官方 M3D-RPN 代码库｜https://github.com/brazgardian/M3D-RPN｜Source-Type: author official code｜Accessibility: public｜仓库公开于 2020；master｜检索：2026-08-12｜Authority: 10/10

[12] M3D-RPN 官方代码许可证｜https://github.com/brazgardian/M3D-RPN/blob/master/LICENSE｜Source-Type: author official code license｜Accessibility: public｜MIT License，Copyright 2020 Garrick Brazil｜检索：2026-08-12｜Authority: 10/10

[13] KITTI 官方 M3D-RPN method detail｜https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=cfac9f339695af62bc10f88dbfacb168b5c50e3d｜Source-Type: official benchmark result｜Accessibility: public｜submission 2019-08-11；当前页面｜检索：2026-08-12｜Authority: 10/10

## 逐篇证据卡

### K06 — Song et al., TLL + Temporal Feature Aggregation（ECCV 2018）

**结论先行：不满足“在 KITTI 上完成目标检测实验”的冻结纳入条件，应保留否定检索证据并替换。** 原文只在 KITTI validation 图上展示定性例子，并报出一个无法完整追溯协议的 Pedestrian Moderate AP=38.72%；训练、主表、消融和正式比较全部发生在 Caltech/CityPersons。它不能作为“KITTI 方法提升”的数值先例。[1][2]

**A. 元数据与代码**

- Tao Song, Leiyu Sun, Di Xie, Haiming Sun, Shiliang Pu；ECCV 2018；正式 Springer DOI：`10.1007/978-3-030-01234-2_33`；官方开放论文页见 [2]，可逐页检索的作者公开版见 [1]。
- 截至本次检索，没有在论文、官方开放页或作者机构页面中核验到作者官方实现；因此代码与许可证均记为 `not found / unknown`，不能从同名非作者仓库推定官方性。

**B. 研究问题与真正 gap**

- 论文的问题不是“怎样改 KITTI 的 P2”，而是：小尺度行人框内背景比例高、边界模糊，bounding-box 标注把不确定背景灌入分类器；能否改用更一致的身体中轴拓扑线，并用跨帧信息恢复模糊/微小目标（PDF p.1–2，Introduction）。[1]
- 它将 gap 定义为**标注表示与小尺度信噪比**，而非普通高分辨率特征层；这与我们的 P2 噪声假设有关，但并不直接证明我们的 P2 或 DCLI 负交互。

**C. 输入、网络、目标、匹配与后处理**

- 输入为图像；有视频时再取相邻帧。ResNet-50 改为 FCN，取消 Conv5x 下采样并用 dilation 保持感受野，最终 stride=16；Conv3d/Conv4f/Conv5c 经 deconvolution 对齐到输入的 1/4，再 skip-connect（PDF p.6，Sec. 4.1，Fig. 3）。[1]
- 监督不回归普通框：上/下顶点分别表示为 Gaussian confidence peaks，连接二者的身体中轴表示为有方向的 link field；候选顶点经 link confidence 配对，先用 bipartite graph matching/Hungarian algorithm 得到实例（PDF pp.6–8，Eq. 1–5）。[1]
- 拥挤时用 MRF：虚拟框之间的 IoU 形成相邻势，抑制不同人的顶/底点错配；优化后再做 BGM。它是**显式结构化匹配后处理**，不是 NMS 融合或 YOLO label assignment（PDF pp.9–10，Sec. 4.3，Eq. 6–8）。[1]
- 视频版把多层特征送入 256-channel Conv-LSTM，unroll 5 个时间步；作者明确说堆叠两个或更多 Conv-LSTM 不获益，说明“更多时序模块”并非单调变好（PDF pp.10、13，Sec. 4.4/5.3）。[1]

**D. KITTI split 与训练/测试身份**

- 原文实验设置明确写“we examine ... Caltech and CityPersons”；Caltech 使用每 3 帧采样得到 42,782 张训练图和标准 4,024 张 test，CityPersons 用全部 3,000 train 和 500 validation（PDF pp.11–13，Sec. 5.1–5.2）。[1]
- KITTI 只出现于 Fig. 7(a) 的“KITTI validation set”可视化和一句 38.72% AP；没有说明 KITTI train/val ID、是否 Chen split、训练图数量、是否 official test、随机种子或 checkpoint 角色（PDF p.11，Fig. 7(a) 前后）。[1]
- 因而不能把 38.72 当作可复现实验，更不能与我们 3341/371 development 的 AP_R40 相减。

**E. 任务、类别、IoU、AP11/AP40、难度**

- TLL 主实验是 **2D Pedestrian detection**，Caltech/CityPersons 指标是 log-average miss rate（MR，FPPI 范围 `10^-2`–`10^0`），不是 KITTI AP_R40（PDF p.11，Sec. 5.1）。[1]
- KITTI 只给 Pedestrian `Moderate AP=38.72%`，没有 Easy/Hard、Cyclist、IoU 阈值、PR 采样数，也没有说明 AP11/AP40。2018 年时代背景不能代替论文明确口径，因此 AP 版本标为 `unknown`，不可擅自写成 AP11。

**F. baseline、完整方法、消融、效率**

- 表 1 是 Caltech MR：ADM Far 74.53；TLL 68.03（降低 6.50 MR 点）；TLL+MRF 67.69；TLL+MRF+FGFA 63.28；TLL+MRF+LSTM 60.79。Reasonable 为 TLL 8.45、+MRF 8.01、+FGFA 7.92、+LSTM 7.40（PDF p.12，Table 1）。这些是**越低越好**的 Caltech MR，不是 AP 增益。[1]
- CityPersons Table 2：TLL 15.5/53.6/17.2/10.0（Reasonable/Heavy/Partial/Bare），TLL+MRF 14.4/52.0/15.9/9.2；依然不是 KITTI（PDF p.12，Table 2）。[1]
- 原文没有参数量、FLOPs、端到端 FPS/延迟；只说把 Conv-LSTM 前特征降为 256 channels 以控制内存/计算，不能据此声称部署成本可控。

**G. Pedestrian/Cyclist、小/远、遮挡证据**

- 对 Pedestrian 的 small/far 证据很强但仅来自 Caltech：Far MR 74.53→60.79（-13.74 MR 点），Middle 30.82→22.92（-7.90）；Near 接近饱和且 TLL 不是明显最佳（PDF pp.12–14，Table 1/Fig. 9）。[1]
- MRF 在 CityPersons heavy occlusion 53.6→52.0，仅 -1.6 MR 点；这提示结构匹配收益需按遮挡层报告，不能用综合数掩盖。[1]
- 没有 KITTI Cyclist 检测结果，也没有 KITTI small/far 分层；不适合作为我们 Pedestrian/Cyclist macro 的直接最近邻结果。

**H. 失败、限制与不可比风险**

- 作者自己报告：Conv2 语义过弱，两个以上 Conv-LSTM 难训且不获益；大目标的 line annotation 会包含更多背景，Near 不优于其他方法（PDF pp.8、12–13）。[1]
- 从预测 line 自动生成固定 aspect ratio=0.41 的框；KITTI 中姿态导致实际宽高比变化，正确人体也可能因 IoU 低被同时计为 FP/FN。论文将 38.72 解释为 annotation bias，但未给替代控制（例如同预测、不同框恢复方式），故这只是支持性观察，不是因果证明（PDF p.11，Fig. 7/正文）。[1]
- 最核心不可比风险：Caltech MR 与 KITTI AP 不同；视频 5 帧与我们的单帧 YOLO 输入不同；论文没有 KITTI 完整协议、KITTI 消融、统计区间或效率表。

**I. 与本项目逐项差异**

- 与 plain P3–P5/plain-P2：TLL 改的是输出表示、监督和实例匹配，不是增加 P2 检测头。
- 与 fusion：TLL 的 skip-connected 多层特征是公共多尺度思想；MRF 处理顶/底点错配，不是我们门控融合的收益/伤害估计。
- 与 DCLI：TLL 用 Gaussian vertex/link-field 监督和 Hungarian/MRF；DCLI 是检测回归校准路径，目标函数与正样本分配不同。
- 因此可引用它支持“边界模糊时，框内背景与语义/定位可能互相干扰”，但不能说它发现或解决了我们 fusion×DCLI 的负交互。

**J. 可借鉴与不可复制边界**

- 可借鉴思想：对 P2 特有候选做**尺度/遮挡/框内背景比例分层**，验证负迁移是否集中在边界不确定的小目标；这不需要复制 TLL 结构。
- 必须引用：top-bottom line supervision、Gaussian vertex maps、link fields、MRF top/bottom matching、5-frame Conv-LSTM 组合均已有明确先例。[1]
- 没有核验到许可的官方代码，因此不得复制网络实现或后处理代码；只能独立实现不同假设下的最小诊断。

**K. 面向本项目的可证伪假设与最小实验**

- H1：plain-P2 的退化若主要来自高分辨率分支把框内背景/邻近实例噪声灌入共享语义，则 P2 特有预测的坏例应显著集中在“小高度 + 高遮挡/高邻框 IoU + 低前景占比”，且关闭 P2 因素路径后这些层的 FP/定位错配会下降；H0：坏例不随这些变量集中。
- 最小实验：复用同一 371-development 的固定预测，不训练；按 GT height、occlusion、邻近 GT IoU、预测框内前景代理量分桶，对 P3–P5、plain-P2、B、AB 输出 TP/FP/FN、duplicate/assignment/localization errors 和 paired bootstrap CI。只有关联模式与预注册方向一致才进入 30-epoch 修复筛查。
- 这项实验借鉴的是“标注/边界不确定性应分层诊断”，不是复制 line detector；若不成立，应否定该解释而非堆 TLL 模块。

**L. 原始定位**

- 方法：PDF pp.6–10，Sec. 4.1–4.4，Fig. 3–6，Eq. 1–8。[1]
- KITTI 唯一证据：PDF p.11，Sec. 5.1，Fig. 7(a)，`38.72% average precision for the moderate test set`；同页同时明确主实验数据集是 Caltech/CityPersons。[1]
- 真正数值表：PDF p.12，Table 1（Caltech）与 Table 2（CityPersons）；PDF pp.13–14，Sec. 5.3/Fig. 9。[1]

#### K06 替换建议（不默默替换）

建议将冻结清单 K06 替换为 **Mao et al., “What Can Help Pedestrian Detection?” (CVPR 2017)**，理由是它确实把 KITTI 用作主要受控实验：7,481 labeled / 7,518 test，训练集按 Chen et al. 拆为 train/validation；以改过 anchors、移除 conv5 保留高分辨率图的 VGG-16 Faster R-CNN 为共同基线（PDF p.3，Sec. 3.1）。[3][4]

- Table 3（KITTI validation，Pedestrian）：1x baseline Moderate 68.96；直接加 segmentation 70.80（+1.84 AP 点），edge 69.49（+0.53）；2x baseline 71.21；segmentation 72.54（+1.33），edge 72.34（+1.13）。它同时显示 heatmap/optical-flow 在 2x 几乎无益或负向，正适合研究“高分辨率细节与辅助语义并不总互补”（PDF p.5，Table 3）。[3]
- Table 2：在 70% precision 下，height≤80 px 的 recall 从 21.3% 升到 35.6%（+14.3 recall 点），所有尺度从 70.0% 到 74.0%；这是切片召回，不是 AP，不能与 AP 相加（PDF p.4，Table 2）。[3]
- Table 4（HyperLearner）：1x baseline Moderate 69.80，segmentation-supervised 71.15（+1.35），edge-supervised 71.25（+1.45）；2x baseline 71.73，分别 72.35（+0.62）与 72.51（+0.78）。CFN 移除的 HyperNet 为 69.72/72.23，表明辅助监督的净增益可被单独隔离（PDF p.7，Table 4）。[3]
- 它仍有局限：只报告 KITTI validation、旧时代 AP 版本未明示、没有 Cyclist AP/seed/CI；分割 channel 来自 MS-COCO 预训练模型，光流和视差需要额外输入。因此可作为 K06 的真正 KITTI 近邻，但不能直接给我们的 AP_R40 预期增益。

### K07 — Han et al., Wasserstein Loss-Based Deep Object Detection（CVPRW 2020）

**结论先行：这篇确实在 KITTI labeled images 上做了 2D 检测，但协议是作者自定义的随机三折、7 类 mAP，不是 KITTI 官方 Pedestrian/Cyclist Moderate AP_R40。** 它对 YOLOv3 的最佳点估计是 +1.1 mAP，对 SSD 是 +1.9 mAP；这些数不能直接移植或与本项目数字相减。[5][6]

**A. 元数据与代码**

- Yuzhuo Han, Xiaofeng Liu, Zhenfei Sheng, Yutao Ren, Xu Han, Jane You, Risheng Liu, Zhongxuan Luo；CVPR Workshops 2020，IEEE 页码 4299–4305；DOI `10.1109/CVPRW50498.2020.00507`；原论文与官方会议页见 [5][6]。
- 论文及官方页没有代码链接；本次按标题、WLOD 和作者名检索也未核验到作者官方仓库。因此 `official code = not found`、`license = unknown`，不得从非官方实现复制。

**B. 研究问题与真正 gap**

- CE/BCE 对一个样本只强调真实类概率：若真实 Bike 的正确类概率相同，把剩余概率放到 Motor 或 Car 会得到相同 CE/BCE，却可能有不同交通安全后果。论文 gap 是**类别间相似性/误分类严重度没有进入检测分类损失**（PDF pp.1–2，Fig. 1）。[5]
- 这不是小目标定位问题，也不是用 Wasserstein 距离比较两个框。必须与 NWD/Gaussian Wasserstein bbox regression 分开：WLOD 的 transport ground matrix 位于**类别概率空间**。

**C. 输入、网络、损失、匹配与后处理**

- 原检测多任务损失保持 `L_location + L_conf + L_class`，新增 `λ L_WD(P,P*,W)`；对 softmax prediction 与 one-hot target，最优 transport 可化成预测误差向量与真实类所在的 ground-matrix 行的内积（PDF p.4，Eq. 2–4）。[5]
- `W` 不是数据学出的道路几何，而是作者依据外观相似性和误判严重度**预先手工分组/赋距**。它鼓励错误概率落在真实类的“邻近类”，同时仍保留 CE/BCE 来提高真实类概率。
- YOLOv3 版输入 resize 为 416×416，backbone Darknet-53；输出三个尺度 `M=13,26,52`、每 cell 3 anchors，新增损失只作用于类别 cells（PDF p.4，Sec. 3.3）。SSD 版同理替换/增强分类项。
- 论文不改 label assignment、bbox regression 或 NMS，也未给独立后处理算法；这些沿用 YOLOv3/SSD，不能把 mAP 差归因于定位或 P2。

**D. KITTI split 与独立性**

- 使用 7,481 张 labeled images，随机分为三折 `(2495, 2493, 2493)` 做 three-fold cross-validation；排除 `Misc`（PDF p.5，Sec. 4）。[5]
- 论文写 KITTI 有 7,512 test images，而 KITTI 官方对象基准通常为 7,518；这一原文内部数量异常必须如实保留，不能替作者修正后冒充完整可复现协议。[5]
- 未公布随机划分 ID/seed、每折训练与验证轮换细节、每折原始 AP 或方差；也没有 official hidden test submission。三折均值不等于三个独立训练 seed，更不等于独立 confirmation。

**E. 任务、类别、IoU、AP11/AP40、难度**

- 任务是 RGB **2D bounding-box detection**。七类为 Car、Van、Truck、Tram、Pedestrian、Person(sitting)、Cyclist；作者把 Cyclist/Pedestrian/Person 置于同一语义组（PDF p.5）。[5]
- Table 2 只称 `mAP`；没有写 KITTI Easy/Moderate/Hard，没有逐类 AP，没有说明 KITTI mAP 的 IoU 阈值或 11/40 recall positions。论文前一段仅对 BDD100K 明确提到 AP50，不能据此自动把 KITTI Table 2 标成 AP50。
- 因此 K07 的 KITTI 数字只能记作 `custom 7-class 3-fold mAP (IoU/AP sampling unspecified)`，严禁标为 AP_R40。

**F. baseline、完整方法、消融、效率**

- KITTI Table 2（PDF p.5）：YOLOv3 68.3；`W0.5-y` 69.2（+0.9 mAP）；`W1-y` 69.4（+1.1）。SSD 72.8；`W0.5-s` 74.7（+1.9）；`W1-s` 74.4（+1.6）。这里 `0.5/1` 是 λ，不是 IoU。[5]
- λ 的方向并不一致：YOLOv3 从 0.5 到 1 再 +0.2，而 SSD 从 0.5 到 1 反而 -0.3，说明权重需按骨干校准，不能说“Wasserstein 越强越好”。
- 论文没有 `CE/BCE-only + 同等参数控制` 之外的多因子消融，没有 ground matrix 随机/均匀/数据驱动对照，也没有每类 AP、统计区间、参数、FLOPs、FPS、训练时长或额外内存；推理理论上未加分支，但“零成本”未经实测。

**G. Pedestrian/Cyclist、小/远、遮挡证据**

- Pedestrian 和 Cyclist 被纳入 7 类 mAP，却没有任何逐类绝对 AP 或 delta；无法知道 +1.1 是否来自 Car/Van/Truck，还是来自我们关心的 Ped/Cyc。
- 没有 small/far/height/occlusion/difficulty 分层，也无 Ped↔Cyclist confusion matrix。把两者放在同一低代价组甚至可能容忍二者互相混淆：7 类 mAP 为正不能推出 Ped/Cyc macro 为正。
- Fig. 4 仅给定性示例，正文举例主要是 Truck↔Car、Bus↔Truck、Rider/Bike（BDD100K）；不是 KITTI Ped/Cyc 的定量机制证据。

**H. 失败、限制与不可比风险**

- 最大风险是协议：随机三折 7 类 mAP、未明 IoU/AP 采样、无 difficulty，与官方 KITTI 和我们 frozen Ped/Cyc Moderate AP_R40 完全不是一个 estimand。
- 手工 `W` 同时编码外观与“安全严重度”，但没有可重复的标注者协议或敏感性分析；观察到增益不能区分语义层级本身、额外正则化或手工超参调优。
- 无 per-fold numbers/CI、无训练 recipe/epoch/seed、无代码；复现门较弱。结论段还把实验数据写成 BDD100K 与 `Drone2019`，和正文的 KITTI 不一致，是额外的文本质量/可追溯风险（PDF p.6）。[5]
- 官方页摘要称 state-of-the-art，但原文 KITTI Table 2 只对 YOLOv3/SSD 自身对照，没有官方 leaderboard 或同协议 SOTA 表；该主张不能沿用。

**I. 与本项目逐项差异**

- 与 plain P3–P5/plain-P2：WLOD 不改尺度层、neck 或 P2；只有分类 loss 的类别关系正则。
- 与 fusion：没有可靠性门、没有特征噪声抑制或多尺度收益/伤害选择。
- 与 DCLI：若 DCLI 主要校准定位/回归，WLOD 则校准类别分布；两者可能共享分类语义但不是同一个机制。我们的 AB 负交互不能由此论文直接解释。
- 与 NWD/RFLA：NWD 把框视为 Gaussian、解决小框 IoU 敏感；WLOD 对类别 histogram 做 transport。名称相似不构成同构。

**J. 可借鉴与不可复制边界**

- 可借鉴：先做**错误严重度/类别相似性分解**，不要只看 macro AP；尤其统计 Ped→Cyclist、Cyclist→Ped、Person(sitting) 混淆与背景 FP，判断 DCLI/fusion 是否改变错误类型。
- 必须引用：任何预定义类别 ground matrix、CE/BCE+Wasserstein 分类正则、将相似误分类视为较低代价的做法均已有此先例。[5]
- 不可直接复用：未发现许可代码；手工安全距离矩阵也不能未经专家/数据验证照搬到我们的两类任务。

**K. 面向本项目的可证伪假设与最小实验**

- H1：若 DCLI/AB 跨 seed 不稳源于“定位改善但 Ped/Cyc 类别排序恶化”，则相对共同基线 C，B/AB 的 matched boxes IoU 会提高，而 Ped↔Cyclist confusion 或 class-score margin 会恶化；且该反向变化集中在 P2/small/far。H0：类别混淆/score margin 不随定位改善反向变化。
- 最小实验：固定已有 C/B/AB 三 seed predictions，按同一 GT matching 产生每类 TP/FP/FN、Ped↔Cyc confusion、matched IoU、true-vs-rival logit/score margin，并做 image-paired bootstrap CI；不训练、不改指标。
- 若 H1 不成立，则 WLOD 类语义层级不是负交互根因，停止该路线；若成立，再预注册一个仅作用于 P2 分类分支的小型语义距离正则，与等强度 label smoothing 负对照做 30-epoch screen。不能直接把 WLOD 换名为原创。

**L. 原始定位**

- 问题与贡献：PDF pp.1–2，Abstract/Fig. 1/Introduction。[5]
- 检测损失与 Wasserstein 化简：PDF pp.3–4，Sec. 3.1–3.2，Eq. 1–4；416×416 YOLOv3 结构：PDF p.4，Sec. 3.3。[5]
- KITTI protocol 与全部数值：PDF p.5，Sec. 4，Table 2；定性图：PDF p.6，Fig. 4。[5]
- 原文没有能够定位的 KITTI per-class/difficulty/IoU/AP11/AP40/效率/seed/CI 表，这些均记 `not reported`，不是遗漏抄录。

### K08 — Simon et al., Complexer-YOLO（CVPRW 2019）

**结论先行：这是 RGB+LiDAR 语义点云上的实时 3D/BEV 检测与跟踪系统，不是 RGB 2D YOLO，也不是 P2 改进。** 官方 test 表在 2019 年 AP40 切换前产生，故属于旧 **AP11**；其 Pedestrian/Cyclist 3D Moderate 分别只有 15.32/23.48，不能与本项目的 2D Pedestrian/Cyclist Moderate AP_R40 相减。论文最有价值的不是可搬用的“增益”，而是两个机制警告：语义输入有助于稀疏点云，但点数后过滤虽使平均结果约 +1.3，却使 Car Moderate BEV **-4.8 AP**，证明总体平均提升可以掩盖重要层的伤害。[7][9]

**A. 元数据、官方论文与代码**

- Martin Simon, Karl Amende, Andrea Kraus, Jens Honer, Timo Sämann, Hauke Kaulbersch, Stefan Milz, Horst-Michael Gross；CVPR Workshops 2019，IEEE DOI `10.1109/CVPRW.2019.00158`，正式论文 pp.1190–1199；CVF 原文与元数据见 [7][8]。
- 论文及 CVF 官方页未给代码链接；截至本次按完整标题、作者和机构检索，未核验到作者官方仓库。因此 `official code = not found`、`license = unknown`。同名第三方仓库不作为实现或许可证证据。

**B. 研究问题与真正 gap**

- 点云距离测量准确但远距离极稀疏，RGB 语义丰富但没有直接深度；问题是能否把图像语义逐点投影到 LiDAR，再在不牺牲实时性的前提下完成多类 3D 检测与在线多目标跟踪（PDF pp.1–2，Abstract/Introduction/Fig.1）。[7]
- 第二个 gap 是旋转 3D 框精确 IoU 计算慢、且仅用 3D IoU 不显式区分朝向相差 `π` 的物体；论文提出可调的 Scale–Rotation–Translation score（SRTs）作为训练相似度/损失近似（PDF pp.4–5，Sec.3.2，Eq.1–6）。这与本项目 P2 噪声、DCLI 定位校准不是同一个问题。

**C. 输入、网络流程、训练目标、匹配与后处理**

- Fig.1（PDF p.2）给出完整流程：同步 RGB+LiDAR → ENet 图像语义分割 → LiDAR 点反投影到语义图 → 语义点云体素化 → Complex-YOLO v3 风格单阶段 3D 多类检测 → Labeled Multi-Bernoulli（LMB）跟踪器。[7]
- ENet 先在 Cityscapes 预训练、再在 KITTI 微调；点云 ROI 为 `x∈[0,60] m, y∈[-40,40] m, z∈[-2.73,1.27] m`，体素网格 `768×1024×21`，约 `0.08×0.08×0.19 m`。体素内特征不是 RGB，而是归一化语义类别值（PDF p.4，Sec.3.1）。
- 检测网络沿用 Complex-YOLO 的鸟瞰输入和 anchor/grid 回归，使用 stride-2 convolution 代替 max-pooling、加入 residual connections，共 49 个卷积层；除位置、尺寸、置信和类别外增加高度 `h` 与地面 `z` 偏移回归（PDF pp.3–4，Fig.2/Sec.3.1）。普通多尺度 YOLO 与残差连接均是公共组件。
- SRTs 为 `p_t(αS_s+βS_t+γS_r)`；论文采用 `w_s=0.3,w_t=1,w_r=0.5`、`γ=0.4, α=β=0.3`。它把尺度、平移和旋转分量显式分开；官方 KITTI 表仍按官方 IoU AP 评估，不能把 SRTs 表当 KITTI AP（PDF pp.4–5，Eq.1–6）。
- 跟踪状态含 `[x,y,z,l,w,h,yaw]`，以 LMB 随机有限集与 Kalman/Unscented Kalman coordinated-turn 模型更新（PDF pp.5–6，Sec.3.3）。这是独立的时序后处理，不应归入检测 AP 增益。

**D. KITTI split、训练身份与独立性**

- 使用 KITTI object detection 的 7,481 labeled train images 与 7,518 hidden test images；论文说“following [4]”划分 training/validation 用于设置优化，随后用全部 training data 训练并提交官方 test，但没有在正文给出 split 的图像数、ID 或随机种子（PDF p.6，Sec.4.1）。不能仅凭引文替作者补写成 Chen split 或某个固定计数。
- 数据增强把每幅训练图扩大 4 倍：原图外随机取三个 `[-20°,20°]` 旋转，彼此至少相隔 8°，再随机沿 x 轴翻转；Darknet 扩展版从头训练 140k iterations，学习率在 20k/80k/120k 调整（PDF p.6，Sec.4.1）。
- validation 用于消融和设置选择，official test 用于一次 leaderboard 提交；论文没有独立 confirmation、多 seed 或 CI。ENet 又使用 Cityscapes 预训练与 KITTI 微调，因此并非仅依赖 KITTI 检测框监督。

**E. 任务、类别、IoU、AP11/AP40 与 difficulty**

- 任务分别是 orientation similarity、BEV detection、3D detection 和 multi-object tracking；绝非 RGB 2D AP。Table 1 报 Car/Pedestrian/Cyclist 的 Easy/Moderate/Hard（PDF p.7）。[7]
- KITTI 官方页明确：2019-10-08 才从 Pascal/VOC 的 11 recall positions 改为 40 positions。[9] 论文发表于 2019-06、其 official test 结果早于该切换，因此 Table 1 应标记 **official KITTI AP11**，而不是 AP_R40。
- 论文 Table 1 没有在表头重写 IoU 阈值；相关阈值只能按同日 KITTI official protocol 解读（Car 0.7、Pedestrian/Cyclist 0.5）。若需要逐值复算，仍须取回当时 evaluator/预测；本文没有开放预测文件。

**F. 绝对结果、消融、参数与速度**

- Table 1（official test，AP11，FPS 15.6）绝对值如下：

  | task | class | Easy | Moderate | Hard |
  |---|---|---:|---:|---:|
  | orientation | Car | 87.97 | 79.08 | 78.75 |
  | orientation | Pedestrian | 37.80 | 31.80 | 31.26 |
  | orientation | Cyclist | 64.51 | 56.32 | 56.23 |
  | BEV | Car | 74.23 | 66.07 | 65.70 |
  | BEV | Pedestrian | 22.00 | 20.88 | 20.81 |
  | BEV | Cyclist | 36.12 | 30.16 | 26.01 |
  | 3D | Car | 55.63 | 49.44 | 44.13 |
  | 3D | Pedestrian | 19.45 | 15.32 | 14.80 |
  | 3D | Cyclist | 28.36 | 23.48 | 22.85 |

  这些数与 F-PointNet、AVOD-FPN、VoxelNet、SECOND、BirdNet 等异构模型并列，不是以同一 backbone 做单变量 baseline/full 对照；Ped/Cyc 精度明显不等于“各类均达 SOTA”（PDF p.7，Table 1）。[7]
- Table 3（validation，3D mAP）比较输入表征，列为 IoU 0.7 / SRTs 0.7：RGB `28.64/30.02`，Occupancy `31.93/33.24`，Intensity `32.39/33.57`，Semantic `34.14/35.43`。Semantic 对 RGB 是 `+5.50/+5.41`，对 Intensity 是 `+1.75/+1.86`，对 Occupancy 是 `+2.21/+2.19`（PDF p.9，Table 3）。正文没有把该 mAP 的类别/难度组合完全展开，故不得推断为 Ped/Cyc Moderate。
- 21 与 51 个 z-height channels 结果接近，更多通道略慢；使用“点数少于 13 且距离小于 52 m”的检测后过滤，平均结果约 `+1.3`，但 **Car Moderate BEV -4.8**。SRTs 替换 IoU 训练约 `+1.3 mAP@IoU0.7`、训练时间减半，并使推理快 10–20%；rotation anchors 反而降低精度（PDF pp.8–9，Sec.4.3）。这些 delta 分属不同设置，不能相加。
- Table 2 的 tracking 结果：MOTA 75.70、MOTP 78.46、MT 58.00、ML 5.08，tracker 单独 100 FPS（PDF p.8，Table 2）。整条链在 GTX1080Ti/Titan 约 11.5 FPS：detector 15.6 FPS、ENet 90 FPS、tracker 100 FPS。论文未报参数量、FLOPs、显存或 per-class latency。

**G. Pedestrian/Cyclist、small/far/occlusion 证据**

- Table 1 确实有 Pedestrian/Cyclist 的 official AP11，但任务是 LiDAR+RGB 3D/BEV；不能把 3D Ped Moderate 15.32 或 Cyclist 23.48 解释为本项目 2D 小目标精度低。
- 作者明确写到：small pedestrian/cyclist 只有在与其他物体保持最小空间距离时才容易检测，近邻点云/框会互相干扰（PDF p.9，Discussion/Conclusion 附近）。这是“邻近实例与稀疏证据交互”的定性失败证据，没有 small/far 分桶 AP。
- 论文还指出 KITTI 3D/BEV evaluator 当时不对 DontCare 做同等过滤，且人工 2D pedestrian boxes 与 3D 框投影不完全吻合，可能把几何上合理的 3D 预测计为 FP（PDF pp.7–8，Table 1 后正文）。这提醒我们必须对 fixed predictions 做 official/internal 对账，而不能把 evaluator 差异解释成模型机制。

**H. 失败、限制与不可比风险**

- 最大不可比因素：LiDAR+RGB+Cityscapes/KITTI semantic supervision、BEV/3D/AP11、140k-iteration 全训练集 official test，对比本项目 RGB-only 2D、3341/371 development、Ped/Cyc Moderate AP_R40，输入、任务、split、evaluator 与 AP sampling 全不同。
- 语义表征的 +5.50 是对 RGB voxel feature，不是“给 YOLO 加一个语义模块就 +5.5”；Table 3 又未给 Ped/Cyc 分项、seed 或 CI。
- 后过滤平均 +1.3 却伤害 Car Moderate -4.8，说明 average-only 选择会制造 no-harm 失败；这与本项目必须同时报告 Ped/Cyc、small/far、FP/FN 的要求直接相关。
- 论文声称与 SOTA 相当，但 Table 1 的 Ped/Cyc 多项明显落后高精度两阶段方法；其主要优势更接近速度–精度折中。无预测、无 split IDs、无代码和统计区间，复现门较弱。

**I. 与本项目逐项差异**

- plain P3–P5/plain-P2：Complexer-YOLO 的“YOLO”是鸟瞰 LiDAR voxel 上的 3D detector；没有 RGB P2 检测头，也没有比较 P3–P5 与 P2。
- fusion：它是传感器级 semantic point fusion，不是项目中的多尺度可靠性门控、收益/伤害估计或 feature-level fusion。
- DCLI：SRTs 同时分解尺度/平移/旋转的 3D box similarity；DCLI 是本项目定位校准路径，监督对象和任务不同。
- C/B/AB 负交互：论文没有 2×2 factorial、梯度冲突或共享估计器分析，不能作为“语义分离修复负交互”的直接先例或证据。

**J. 可借鉴思想、必须引用与不可复制边界**

- 可借鉴而不复制结构：把“平均提升是否由一项硬过滤换来某个重要层伤害”设为固定诊断；对 P2-only boxes 按邻近实例距离、遮挡、框内语义纯度/边缘支持与置信分桶，分别报告 FP/FN/localization。
- 必须引用的公共思想：图像语义投影至点云、semantic voxel grid、SRTs、LMB 3D tracker，以及以点数/距离做检测后过滤均已有清楚先例。[7]
- 未找到许可明确的作者官方代码，不能复制第三方实现；本项目如采用“语义支持/邻近实例”只可独立实现新的 2D 诊断或门控，并应明确它与论文传感器融合的区别。

**K. 面向本项目的可证伪假设与最小实验**

- H1：若 plain-P2 退化或 AB 负交互来自高分辨率分支在“语义支持弱、邻近实例密集”的区域放大背景/重复框，那么相对 C，P2/AB 新增 FP 与定位错配应集中在低 foreground-support、低边缘一致性、高邻框 IoU/低中心距离 strata；关闭 P2 因素路径后，这些 strata 应优先恢复。H0：误差不随上述变量集中。
- 最小实验：只复用同一 371-development 的固定预测与 GT，不训练；计算每个新增/丢失 box 的最近 GT/预测距离、遮挡、GT height、局部语义代理（已有 backbone feature energy 或无训练边缘/纹理比），并对 C/B/AB 做 image-paired bootstrap。先确认关联，再设计一个预注册、硬 no-harm fallback 的 30-epoch gate；若关联不存在，否定该路线。
- 这只借鉴“稀疏证据 + 语义支持 + 后过滤伤害”的研究问题，不采用其 ENet/voxel/SRTs/LMB 结构，也不承诺 AP 为正。

**L. 原始定位**

- 总流程与问题：PDF pp.1–2，Abstract/Introduction/Fig.1；相关工作：pp.2–3。[7]
- Semantic point cloud、ROI/voxel、49-layer detector：PDF pp.3–4，Sec.3.1/Fig.2；SRTs：pp.4–5，Sec.3.2/Eq.1–6；tracker：pp.5–6，Sec.3.3。
- 数据、训练与增强：PDF p.6，Sec.4.1；official test 值：p.7，Table 1；tracking 与速度：p.8，Table 2；输入消融及过滤/SRTs trade-off：pp.8–9，Sec.4.3/Table 3。
- AP11 判定的外部协议锚：KITTI official object evaluation page 的 2019-10-08 Note 2 明确此前为 11 recall positions、此后为 40。[9]

### K09 — Brazil & Liu, M3D-RPN（ICCV 2019）

**结论先行：这是单目 RGB 的 3D/BEV 单阶段 RPN，不是 2D P2 检测器；它却是“道路图像纵向位置/深度先验 + 全局/局部双路 + 每个输出独立融合权重”的直接先例。** 论文原表形成于 KITTI AP40 切换前，是旧 AP11；当前 KITTI 官方详情页展示同一方法在现行协议下明显更低的结果。最关键的项目启示有两点：`b=1/4` 的额外路径反而退化，说明“多一条局部分支”不是天然有益；Cyclist Moderate 从 validation 的约 9–10 AP11 掉到 official test 的约 1 AP11，作者也承认 cyclist 不稳定，说明小类/稀疏类别必须独立确认。[9][10][13]

**A. 元数据、原文、官方代码与许可证**

- Garrick Brazil, Xiaoming Liu；ICCV 2019 Oral；CVF proceedings pp.9287–9296；DOI `10.1109/ICCV.2019.00938`；原论文见 [10]。
- 作者/实验室官方代码由论文链接至 [11]，PyTorch、Ubuntu 16.04/CUDA 8/GTX1080Ti；仓库提供 val1/val2 split、训练配置、恢复入口和发布模型。许可证是 MIT（Copyright 2020 Garrick Brazil），复用必须保留版权与许可文本 [12]。
- 本次核验的原始 PDF 已保存为 `E:\myyolo\kitti_project\research-notes\_src_m3drpn.pdf`，SHA256 `4BA3780D26FB2ECC7CEBAC80D03879943562EFB275B840ECDC9E29D92731DA5F`；此路径只是阅读副本，不是项目代码依赖。

**B. 研究问题与真正 gap**

- 单目图像缺少显式深度，既有 monocular 3D 系统常串接外部 2D detector、depth/point-cloud、semantic/instance segmentation 或多阶段 R-CNN；组件分离带来噪声上限、训练不一致和工程复杂度。作者的问题是能否把 2D/3D anchor、分类与定位放进一个 end-to-end 3D RPN（PDF pp.1–3，Fig.1–3）。[10]
- 固定车载相机下，图像纵向 row 与深度/尺度高度相关；普通 convolution 的空间平移不变性对高层 3D 特征未必合适。论文以 row-wise non-shared kernels 注入位置/深度意识（PDF p.5，Sec.3.2）。这是道路几何/透视感知卷积的明确先例，不能把“按图像纵向位置适配特征”本身写成我们的原创。

**C. 输入、网络、anchor、loss、assignment、NMS 与后处理**

- 输入是单幅 RGB 与已知相机投影矩阵 `P`；DenseNet-121 backbone 去掉最后 pooling，使 stride=16，并对最后 DenseBlock 采用 dilation=2。末端有 global regular-convolution 与 local depth-aware-convolution 两路，各自输出 512-channel proposal features（PDF p.5，Sec.3.3/Fig.3）。
- 每个 anchor 同时包含 `[w,h]2D`、投影深度 `zP` 和 `[w,h,l,θ]3D`；3D center 先在图像空间预测，再利用 `P^{-1}` 回投 camera coordinates。每个 2D anchor 对所有 IoU≥0.5 的训练 GT 统计其 `z,w,h,l,θ` 均值，作为 3D priors（PDF pp.3–4，Fig.4/Eq.1–3）。
- assignment：generated box 与 GT 的 2D IoU≥0.5 时取最佳 GT，否则为 background、忽略 box regression。loss 为 shared softmax classification `Lc`、2D `-log IoU` loss 与 3D SmoothL1，`L=Lc+λ1Lb2D+λ2Lb3D`，`λ1=λ2=1`（PDF p.4，Eq.4–7；p.6，Sec.3.5）。
- depth-aware convolution 把 feature map 沿 row 划为 `b` 个 bins，每个 bin 学独立 kernel；理论 convolution FLOPs 不变，但 kernel memory 约 `×b`，PyTorch unfold/group-convolution 实现实测有 2–20% overhead（PDF p.5，Sec.3.2；p.8）。
- global/local 每个输出 `i` 学独立 `α_i`，`O_i=α_i O_global+(1-α_i)O_local`；Table 6 显示 classification `c` 的 α=33%（最低、即最偏 local），其他 2D/3D 参数约 38–48%。因此“任务/输出专用融合权重”亦有直接先例（PDF pp.5,7，Eq.8/Table 6）。
- orientation post-optimization 迭代调整 `θ`，最小化 projected 3D corners 与 predicted 2D box 的 L1 consistency，约 8 iterations 收敛（PDF pp.5–6，Algorithm 1/Eq.9）。推理使用 2D NMS IoU=0.4、score≥0.75；训练用 random mirroring 与每 minibatch top 20% high-loss boxes 的 OHEM（PDF p.6，Sec.3.5）。

**D. KITTI split、训练身份与独立性**

- 数据为 KITTI 7,481 labeled train / 7,518 hidden test；论文同时使用 `val1`（3DOP [8]）与 `val2`（SubCNN [39]）以及 official test，并声明各 split 从不重叠 sequence 切分，评估 frame 及其邻帧未用于训练（PDF p.6，Sec.4.1）。[10]
- 论文正文没有列图像 IDs；作者代码 [11] 固化了两套 split。split1 官方仓库文件明确为 3,712 train / 3,769 val；split2 通过作者提供的 `kitti_ids_new.mat` 与 setup script 生成，但本证据卡不在没有逐 ID 解析前擅自写计数。
- global/warmup 与 depth-aware/main 两阶段各训练 50k iterations；local path 从已训练 global weights 初始化以减轻局部 bins 过拟合。LR=0.004、poly power=0.9、batch=2、weight decay=0.9；图像统一缩放至 height=512，`b=32`，36 anchors（12 scales × 3 ratios）（PDF p.6，Sec.3.5）。作者仓库明确支持 snapshot restore，但论文无 seed 数、CI 或独立 confirmation。

**E. 任务、类别、IoU、AP11/AP40 与 difficulty**

- 主任务是 BEV 与 camera-coordinate 3D detection；辅助报告 2D detection。类别为 Car/Pedestrian/Cyclist；Easy/Moderate/Hard；Car official IoU≥0.7，Pedestrian/Cyclist≥0.5（PDF p.6，Sec.4.1）。
- 原论文 v2 与 KITTI submission 均为 2019-08-11，早于官网 2019-10-08 AP40 切换 [9][13]；因此 Tables 1–4 的值属于 **旧 AP11**。论文没有写 `R40`，不得以后来引用表替换原表。
- 当前 KITTI method-detail [13] 在现行 benchmark 页面上另列该方法的当前结果；因官方当前协议已是 40 recall positions [9]，这些可记作官网现行 AP_R40 展示。两组数必须并列，不可混算。

**F. baseline、完整方法、消融、绝对值与效率**

- 论文 Table 1（Car BEV，AP11，IoU≥0.7）：M3D-RPN val1/val2/test Moderate `21.18/21.15/18.36`；最近的 Multi-Fusion 为 `13.63/12.17/9.62`，对应系统级差 `+7.55/+8.98/+8.74` AP 点。模型、输入训练与阶段均不同，这不是单组件公平消融。
- Table 2（Car 3D，AP11，IoU≥0.7）：M3D-RPN val1/val2/test Moderate `17.06/16.48/15.70`；Multi-Fusion `5.69/5.39/5.18`，差 `+11.37/+11.09/+10.52`。这同样是整系统跨方法比较，不能作为 P2 预期。
- Table 5（val1 Car Moderate）给出可控消融：

  | b / setting | post-opt | AP2D | AP3D | APBEV | runtime ms |
  |---|---|---:|---:|---:|---:|
  | baseline（global only） | no | 82.16 | 10.99 | 12.99 | 118 |
  | global only | yes | 82.16 | 15.08 | 17.47 | 128 |
  | 1（two global-like streams） | yes | 82.88 | 12.87 | 17.91 | 133 |
  | 4 | yes | 84.15 | 14.46 | 19.14 | 134 |
  | 8 | yes | 83.86 | 16.04 | 20.99 | 143 |
  | 16 | yes | 83.02 | 15.97 | 18.48 | 153 |
  | 32（full） | yes | 83.67 | 17.06 | 21.18 | 161 |

  post-opt 相对 baseline 为 AP3D `+4.09`、APBEV `+4.48`；在 post-opt 已开启时，b=32 相对 global-only 为 AP2D `+1.51`、AP3D `+1.98`、APBEV `+3.71`。但论文没有 `b=32/no-post`，所以不是完整 2×2，不能估计 depth-aware×post-opt interaction。b=1/4 的 AP3D 比 post-only 分别 `-2.21/-0.62`，直接否定“加局部分支必然提升”（PDF pp.7–8，Table 5/Sec.4.2）。
- full runtime 161 ms/image（约 6.2 FPS）vs 118 ms baseline；KITTI 官方页报 submission runtime 0.16 s/GPU@1.5GHz [13]。论文不报总参数/FLOPs/显存；只说明理论 conv FLOPs 不变、kernel memory×b、实际 reshape 2–20% overhead。

**G. Pedestrian/Cyclist、small/far/occlusion 证据**

- 论文 Table 3（single shared multi-class model，Moderate，AP11）为：

  | class | APBEV val1 / val2 / test | AP3D val1 / val2 / test |
  |---|---|---|
  | Car | 21.18 / 21.15 / 18.36 | 17.06 / 16.48 / 15.70 |
  | Pedestrian | 11.60 / 11.44 / 11.35 | 11.28 / 11.30 / 10.54 |
  | Cyclist | 10.13 / 9.09 / 1.29 | 10.01 / 9.09 / 1.03 |

  Pedestrian 从 validation 到 test 接近，Cyclist 却约跌 8–9 AP 点。作者明确称 cyclist “noticeably less stable”，建议 advanced sampling/data augmentation，并把它归因于训练数据偏向 Car/Pedestrian（PDF p.7，Table 3 后正文）。这是论文内的诚实负面结论，而非本证据卡推断。[10]
- 当前 KITTI 官方详情页 [13] 的现行结果进一步显示：Pedestrian 2D Moderate 41.46、3D 3.48、BEV 4.05；Cyclist 2D Moderate 41.54、3D 0.65、BEV 0.81。任务不同且 current AP_R40 与 paper AP11 不同；任何一个数字都不能与本项目 2D Ped/Cyc Moderate AP_R40 直接相减。
- 论文没有按 height/distance/occlusion 的独立 small/far AP，也没有 Ped/Cyc 的 depth-aware 或 post-opt 消融；因此不能证明 row-aware features 对远处行人/骑行者有效。

**H. 失败、限制与不可比风险**

- 论文自己说明 M3D-RPN 的 2D Car AP 低于使用外部强 2D detectors 的方法；它优化的是 3D localization，不是 2D detection SOTA（PDF p.7，Table 4 后正文）。
- Cyclist validation→test 崩塌说明 shared multi-class 与 validation 调参并未保证稀疏类别泛化；无 seed/CI、无类别样本分母，会让高方差看起来像机制收益。
- depth-aware convolution 依赖固定 camera geometry 和已知 intrinsics；换相机、裁剪、resize/letterbox、道路坡度或相机俯仰会破坏 row-depth 对应。作者也指出没有已知 intrinsics 时需预测相机参数/弱监督（PDF p.6，Sec.3.5）。
- AP11 与 current AP_R40 差异巨大。例如 Ped 3D Moderate 论文 Table 3 为 10.54，而当前官方页为 3.48；Cyclist 为 1.03 vs 0.65。这是同名方法也不能只凭名称相减的强例证。
- Table 5 只有 Car val1 Moderate，缺 Ped/Cyc、full factorial、CI 和参数量；full 比 baseline 慢约 36%，不能只报 AP。

**I. 与本项目逐项差异及碰撞**

- plain P3–P5/plain-P2：M3D-RPN stride=16 single proposal map，不添加 RGB stride-4 P2 head；目标为 3D box，不是本项目 2D small-object AP_R40。
- fusion：M3D-RPN 已有 global/local parallel paths 与每个输出独立学习 α；因此“双路特征、按任务参数自适应融合”不可称原创。本项目只有在“P2 噪声可靠性/收益伤害、硬预算、no-harm”具体问题上才可能形成差异。
- geometry：row-wise non-shared kernels 明确利用固定车载视角的 row-depth correspondence；“道路透视/纵向位置引导特征”必须引用，不能独创宣称。
- DCLI：post 3D→2D orientation consistency 是几何后优化，不等同本项目定位校准 loss；但它证明分类/2D/3D 参数对 local 分支的最优依赖不同，支持检查 DCLI/fusion 是否争用同一 shared representation。
- 负交互：b=1/4 的分支会伤 AP3D，但论文没有梯度诊断或 2×2 interaction estimand；它只提供“额外路径可负迁移”的现象先例，不解释本项目 AB 根因。

**J. 可借鉴思想、必须引用与代码边界**

- 可借鉴：把 road geometry 作为**显式、可关闭且需跨相机验证的 conditioning**，并让 classification/regression 拥有不同的小适配器/权重，而不是共享一个几何门；同时检查 row bin、distance proxy、camera calibration perturbation 下的 no-harm。
- 必须引用：row-wise depth-aware non-shared convolution、global/local parallel output fusion、2D/3D shared anchors、3D→2D orientation consistency post-optimization均已有 M3D-RPN 先例。[10]
- 官方代码为 MIT，可依法复用但必须保留许可证/版权和明确归属；本项目若实现 2D P2 可靠性机制，优先独立最小实现，不能复制其模块后改名。

**K. 可证伪项目假设与最小实验**

- H1：若 P2/fusion×DCLI 负交互来自**同一 row/depth prior 同时控制分类与定位**，则按 image-row/distance proxy 分层时，AB 相对 B 的分类 score margin 与 matched-IoU 变化应呈不同甚至相反方向；将几何 conditioning 仅用于候选/分类，定位 adapter stop-gradient 后，冲突层应下降且 AP/FP/FN 同向恢复。H0：误差与 row/depth 无关，或分离后机制指标与 AP 不共同改善。
- 最小诊断先不训练：在 fixed C/B/AB predictions 上以 normalized y-center、GT height、occlusion 为 predeclared bins，输出 Ped/Cyc TP/FP/FN、matched IoU、class margin 和 image-paired CI。只有观察到稳定的“分类受益/定位受损（或反之）”结构，才做 30-epoch `shared vs task-specific adapter` 两条件筛查。
- 停止门：若三 seed 的错误方向不一致，或 adapter 只改善某个挑选切片、不改善主 Ped/Cyc Moderate macro AP_R40，则否定该机制，不堆新模块。

**L. 原始定位**

- 问题、贡献、与外部多阶段系统差异：PDF pp.1–3，Fig.1–3。[10]
- anchor/3D projection/loss：pp.3–4，Sec.3.1，Fig.4，Eq.1–7；depth-aware/global-local fusion：p.5，Sec.3.2–3.3，Eq.8；post-opt：pp.5–6，Algorithm 1/Eq.9。
- 训练、NMS/OHEM、已知 intrinsics 限制：p.6，Sec.3.5；KITTI protocol：p.6，Sec.4.1；Car AP11 表：p.6，Tables 1–2；multi-class/2D/消融：p.7，Tables 3–6；解释与效率：pp.7–8，Sec.4.1–4.2。
- 当前 AP_R40 与运行时间：KITTI official method detail [13]；AP11→AP40 切换日期：KITTI official Note 2 [9]；split IDs/训练恢复/发布模型与许可证：作者官方仓库 [11][12]。

## 增量保存日志

- 2026-08-11：创建任务账本；K06–K10 尚未形成结论。
- 2026-08-11：K06 全文核验完成。判定其没有完整 KITTI 目标检测实验；保留否定证据，推荐 Mao et al. CVPR 2017 作为显式替换候选。下一篇 K07。
- 2026-08-12：K07 全文与官方元数据核验完成。确认为随机三折 7 类 custom mAP；最佳 YOLOv3 +1.1、SSD +1.9，但与官方/本项目 AP_R40 不可比。下一篇 K08。
- 2026-08-12：K08 原文 Methods/Experiments 与 KITTI 官方 AP 变更记录核验完成。确认为 RGB+LiDAR 3D/BEV official AP11；语义体素消融虽正向，但重要 strata 存在后过滤伤害，不能外推至本项目 2D P2。下一篇 K09。
- 2026-08-12：K09 原论文、作者官方代码/许可证、KITTI 官方 method detail 完成对账。严格拆分论文 AP11 与官网当前 AP_R40；确认 row-depth 双路融合是直接先例，b=1/4 负向、Cyclist validation→test 崩塌是关键限制。下一篇 K10。
