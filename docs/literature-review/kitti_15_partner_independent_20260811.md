# KITTI 15 篇论文独立深读：第二审稿人技术卡

> 状态：独立稿15/15完成；已在独立判断冻结后读取主稿K01–K11做第一轮差异核验（禁止用本文献数字改写项目既有负/零结果）  
> 研究日期 / AS_OF：2026-08-11  
> 模式：Standard deep research  
> 证据边界：仅原论文全文、作者官方代码仓库、KITTI 官方协议页；搜索结果摘要只用于定位，不作为数值证据。  
> 反方审查：优先寻找 2D/BEV/3D 任务混淆、AP11/AP40 混淆、Chen split/官方 test 混淆、单次消融与多随机种子稳定性混淆。

## 0. 研究问题、纳入标准与项目事实边界

主问题：KITTI 文献中哪些机制真正针对小/远/遮挡目标，哪些结果可与本项目的 plain P3–P5、plain-P2、DCLI、fusion 建立因果上可检验的联系？

纳入标准：指定论文必须包含 KITTI 目标检测实验或定义 KITTI 检测协议；若不满足，保留原论文证据并提出同主题且确实使用 KITTI 的替代论文。不同任务、类别、IoU、AP 采样规则、划分和难度不做横向数值排序。

本项目已冻结的事实边界（仅作对照，不由文献反向修改）：plain-P2 相对标准 P3–P5 的当前总结果为 **-1.5918 AP**；DCLI 三种子平均相对共同基线约 **+0.3257 AP**、联合版约 **+0.1710 AP**，置信区间跨 0；AB 相对 B 为负。本文只能形成后续可证伪假设，不能把论文中的正收益借给本项目。

## 1. 15 行 protocol matrix（完成中）

| # | 论文 | KITTI任务 | 输入 | split/test | 类别 | IoU | AP规则 | Easy/Mod/Hard | 直接可比性 |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | KITTI benchmark protocol | 2D detection / AOS；后扩展 BEV/3D | 左/右彩色图，可选时序、Velodyne | 7481 train / 7518 hidden test；序列隔离；无官方 Chen split | Car/Pedestrian/Cyclist（原始论文表1只汇总 Car/Ped） | 现行2D：Car 0.7，Ped/Cyc 0.5 | 2019-10-08前 AP11，之后 AP40 | 40/25/25 px；遮挡0/1/2；截断15/30/50% | 定义所有后续口径，不是可比模型 |
| 2 | MS-CNN | 2D detection / proposal | monocular RGB；多尺度训练/可输入上采样 | 7481内按3DOP划分做val；全train→官方test | Car/Ped/Cyc | 0.7/0.5/0.5 | AP11（2016） | E/M/H | 与本项目同为2D但架构、年代/AP规则不同 |
| 3 | SubCNN | 2D detection + AOS（附带seg/3D loc） | monocular RGB image pyramid | 3682/3799自定义val；全7481→official test | Car/Ped/Cyc | 0.7/0.5/0.5 | AP11（2016/17） | E/M/H | 同为2D，但两阶段/姿态额外监督/AP11 |
| 4 | RRC | 2D detection | monocular RGB，1272×375 | 自建图像相似度隔离 split，val 2741；全train→official test | Car/Ped/Cyc | 0.7/0.5/0.5 | AP11（2017） | E/M/H；Ped/Cyc test重点报Moderate | 同为2D，但SSD式anchor、循环多输出及自定义split |
| 5 | DeepParts | 2D pedestrian detection | monocular RGB；LDCF proposals+part ConvNets | **不在KITTI训练**；Caltech训练后直接 official KITTI test | Pedestrian | 0.5 | AP11（2015；现行详情页数值已漂移） | E/M/H | 跨数据集迁移且只评Ped，不能与KITTI训练YOLO公平相减 |
| 6 | **替换：What Can Help Pedestrian Detection? / HyperLearner**（原TLL+TFA排除） | 2D Pedestrian detection | RGB；训练期可用edge/seg等辅助监督；直接通道实验含stereo/temporal | 7481内按文献[5]拆train/val，精确索引未重列；无official test | Pedestrian（误差分析另辨Cyclist FP） | 0.5 | AP11（2017） | E/M/H | 同为2D Ped，但Faster R-CNN、多任务/外部监督、单split |
| 7 | Wasserstein Loss-Based Deep Object Detection | 非官方KITTI 2D multi-class mAP | RGB；YOLOv3/SSD | 7481随机三折2495/2493/2493 | 7细类并成3风险组，排Misc | 论文只称mAP，KITTI阈值细节未给 | 非官方，非AP11/AP40榜单 | 不分E/M/H | 与项目口径不兼容，仅loss思想可检验 |
| 8 | Complexer-YOLO | BEV/3D/AOS+MOT | LiDAR BEV + RGB语义分割标签投影 | official KITTI test；消融split未说明 | Car/Ped/Cyc | Car .7，Ped/Cyc .5 | 2019历史AP11；现行详情页有漂移 | E/M/H | 传感器、3D任务与项目2D完全不同 |
| 9 | M3D-RPN | monocular 2D/BEV/3D | 单目RGB | val1 3712/3769、val2 3682/3799、official test | Car/Ped/Cyc共享模型 | Car .7；Ped/Cyc .5 | 2019 AP11 | E/M/H；多类表重点Mod | 仅2D附表可接近；核心3D不可比 |
| 10 | D4LCN | monocular 2D/BEV/3D | RGB + 单目/可替换深度估计图 | split1 3712/3769、split2 3682/3799、official test | Car/Ped/Cyc | Car .7；Ped/Cyc .5 | split消融同时列AP11/AP40；跨方法表/官方test有历史口径混杂风险 | E/M/H | 2D附表勉强相关；深度输入、3D监督与项目不同 |
| 11 | End-to-End Pseudo-LiDAR | stereo image-based BEV/3D | 双目RGB；训练深度用LiDAR监督；相机标定 | KITTI train内标准train/val；全train→official test；文中未给序列隔离审计 | Car | .5与.7 | 表仅写AP、未显式标R11/R40，不与AP_R40直接相减 | E/M/H | 传感器、任务和监督均不同；仅梯度耦合机制可借鉴 |
| 12 | MonoRUn | monocular 3D（另报2D/BEV） | 单目RGB；可选稀疏LiDAR NOC监督 | 常用3712/3769 split；全train→official test | Car/Ped/Cyc | Car .7；Ped/Cyc .5 | AP40（正文明确40 recall positions） | E/M/H | 项目为2D；不确定性/训练耦合只能作机制对照 |
| 13 | GUPNet | monocular BEV/3D | 单目RGB+标定，无额外data | 3712/3769；全7481→official test | Car/Ped/Cyc | Car .7；Ped/Cyc .5 | AP40 | E/M/H | 3D几何层级任务；对项目仅优化稳定机制相关 |
| 14 | MonoDLE | monocular 2D/AOS/BEV/3D | 单目RGB+标定，无额外data | 3712/3769；全7481→official test | Car/Ped/Cyc | Car .7；Ped/Cyc .5 | AP40 | E/M/H | 2D附带输出但主消融是Car 3D，不能与项目AP横减 |
| 15 | **MonoDETR**（无需换OccupancyM3D） | monocular BEV/3D（补充多类） | 单目RGB+标定；仅object-wise depth标签 | 3712/3769；全7481→official test | Car；补充Ped/Cyc | Car .7；Ped/Cyc .5 | AP40 | E/M/H | DETR/3D任务不同；matching稳定性与远样本过滤可借鉴 |

## 2. Citation registry（边读边冻结）

### Approved sources

[1] KITTI Vision Benchmark Suite — Object Detection Evaluation 2012 | https://www.cvlibs.net/datasets/kitti/eval_object.php | Source-Type: official | Accessibility: public | As Of: 2026-08-11 | Authority: 10/10 | 用途：现行数据量、IoU、难度、AP40变更与提交政策。  
[2] Geiger, Lenz, Urtasun — *Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite* (CVPR 2012) | https://www.cvlibs.net/publications/Geiger2012CVPR.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2012 | Authority: 10/10 | 用途：采集、标注、序列隔离、原始检测/AOS定义。  
[3] KITTI Vision Benchmark Suite — dataset home | https://www.cvlibs.net/datasets/kitti/ | Source-Type: official | Accessibility: public | As Of: 2026-08-11 | Authority: 10/10 | 用途：官方引用关系与任务入口。
[4] Cai et al. — *A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection* (ECCV 2016) | https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Cai2016ECCV.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2016 | Authority: 10/10 | 用途：MS-CNN方法、消融、KITTI val/test。  
[5] Cai et al. — MS-CNN official code | https://github.com/zhaoweicai/mscnn | Source-Type: official-code | Accessibility: public | As Of: 2026-08-11 | Authority: 9/10 | 用途：作者论文明确给出的实现入口。
[6] Xiang et al. — *Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection* | https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Xiang2016ARXIV.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2016 | Authority: 10/10 | 用途：SubCNN方法、KITTI val/test和消融。  
[7] Xiang et al. — official author supplementary material | https://yuxng.github.io/Papers/2017/xiang_wacv17_tr.pdf | Source-Type: academic/primary-supplement | Accessibility: public | As Of: 2017 | Authority: 9/10 | 用途：运行时间补充；未发现作者公开的该版本代码入口。
[8] Ren et al. — *Accurate Single Stage Detector Using Recurrent Rolling Convolution* (CVPR 2017) | https://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_Accurate_Single_Stage_CVPR_2017_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2017 | Authority: 10/10 | 用途：RRC结构、训练、KITTI val消融与test结果。  
[9] Ren et al. — RRC official code | https://github.com/xiaohaoChen/rrc_detection | Source-Type: official-code | Accessibility: public | As Of: 2026-08-11 | Authority: 9/10 | 用途：论文明确给出的实现入口。  
[10] Tian et al. — *Deep Learning Strong Parts for Pedestrian Detection* (ICCV 2015) | https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tian_Deep_Learning_Strong_ICCV_2015_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2015 | Authority: 10/10 | 用途：DeepParts方法、Caltech消融与KITTI迁移实验。  
[11] KITTI official result detail — DeepParts | https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=c739a8a7ff3820af0c12d0496974ab41b843da69 | Source-Type: official | Accessibility: public | As Of: 2026-08-11 | Authority: 10/10 | 用途：当前服务器保存的DeepParts分数、运行时与提交说明。
[12] Mao et al. — *What Can Help Pedestrian Detection?* (CVPR 2017) | https://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2017 | Authority: 10/10 | 用途：替换K06；HyperLearner、KITTI Table 1–4与错误分析。  
[13] Song et al. — *Small-scale Pedestrian Detection Based on Topological Line Localization and Temporal Feature Aggregation* (ECCV 2018) | https://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2018 | Authority: 10/10 | 用途：K06排除核验；KITTI Fig.7与正文38.72%，Caltech/CityPersons主实验边界。
[14] Han et al. — *Wasserstein Loss-Based Deep Object Detection* (CVPRW 2020) | https://openaccess.thecvf.com/content_CVPRW_2020/papers/w60/Han_Wasserstein_Loss-Based_Deep_Object_Detection_CVPRW_2020_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2020 | Authority: 9/10 | 用途：WLOD损失、KITTI三折与Table 2。  
[15] Simon et al. — *Complexer YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds* (CVPRW 2019) | https://openaccess.thecvf.com/content_CVPRW_2019/papers/Autonomous%20Driving/Simon_Complexer_YOLO_Real-Time_3D_Object_Detection_and_Tracking_on_Semantic_CVPRW_2019_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2019 | Authority: 9/10 | 用途：多模态3D流程、KITTI test与消融。  
[16] KITTI official result detail — Complexer-YOLO | https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=3fa2d259db7b5018c7b2e494e01c5fd1f89d3347 | Source-Type: official | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10 | 用途：现行服务器Ped 3D/BEV分数与历史表漂移。  
[17] Brazil & Liu — *M3D-RPN: Monocular 3D Region Proposal Network for Object Detection* (ICCV 2019) | https://openaccess.thecvf.com/content_ICCV_2019/papers/Brazil_M3D-RPN_Monocular_3D_Region_Proposal_Network_for_Object_Detection_ICCV_2019_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2019 | Authority: 10/10 | 用途：方法、双val split、test、Table 1–6。  
[18] Brazil & Liu — M3D-RPN official code | https://github.com/garrickbrazil/M3D-RPN | Source-Type: official-code | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10 | 用途：split1/split2、warmup/main配置与复现入口。
[19] Ding et al. — *Learning Depth-Guided Convolutions for Monocular 3D Object Detection* (CVPR 2020) | https://openaccess.thecvf.com/content_CVPR_2020/papers/Ding_Learning_Depth-Guided_Convolutions_for_Monocular_3D_Object_Detection_CVPR_2020_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2020 | Authority: 10/10 | 用途：D4LCN方法、KITTI split/test与主消融。
[20] Ding et al. — D4LCN supplemental | https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Ding_Learning_Depth-Guided_Convolutions_CVPR_2020_supplemental.pdf | Source-Type: academic/primary-supplement | Accessibility: public | As Of: 2020 | Authority: 10/10 | 用途：卷积替代、AP11/AP40双口径与实现细节。
[21] Ding et al. — D4LCN official code | https://github.com/dingmyu/D4LCN | Source-Type: official-code | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10 | 用途：作者实现入口与配置核验。
[22] Qian et al. — *End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection* (CVPR 2020) | https://openaccess.thecvf.com/content_CVPR_2020/papers/Qian_End-to-End_Pseudo-LiDAR_for_Image-Based_3D_Object_Detection_CVPR_2020_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2020 | Authority: 10/10 | 用途：可微CoR、梯度平衡、KITTI消融与test。
[23] Qian et al. — End-to-End Pseudo-LiDAR supplemental | https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Qian_End-to-End_Pseudo-LiDAR_for_CVPR_2020_supplemental.pdf | Source-Type: academic/primary-supplement | Accessibility: public | As Of: 2020 | Authority: 10/10 | 用途：深度变化、边界解释与补充实验。
[24] Qian et al. — pseudo-LiDAR_e2e official code | https://github.com/mileyan/pseudo-LiDAR_e2e | Source-Type: official-code | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10 | 用途：作者给出的复现入口。
[25] Chen et al. — *MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation* (CVPR 2021) | https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_MonoRUn_Monocular_3D_Object_Detection_by_Reconstruction_and_Uncertainty_Propagation_CVPR_2021_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2021 | Authority: 10/10 | 用途：重建、自监督几何、不确定性、KITTI消融/test。
[26] Chen et al. — MonoRUn supplemental | https://openaccess.thecvf.com/content/CVPR2021/supplemental/Chen_MonoRUn_Monocular_3D_CVPR_2021_supplemental.pdf | Source-Type: academic/primary-supplement | Accessibility: public | As Of: 2021 | Authority: 10/10 | 用途：多类结果、训练与推理细节。
[27] Chen et al. — MonoRUn official code | https://github.com/tjiiv-cprg/MonoRUn | Source-Type: official-code | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10 | 用途：作者实现与配置入口。
[28] KITTI official result detail — MonoRUn | https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=4471a46f15dfbb89eaa608dca8266e60db2e80a6 | Source-Type: official | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10 | 用途：官方test提交身份与当前详情页。
[29] Lu et al. — *Geometry Uncertainty Projection Network for Monocular 3D Object Detection* (ICCV 2021) | https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2021 | Authority: 10/10 | 用途：GUP/HTL、AP40、多类与消融。
[30] Lu et al. — GUPNet official code | https://github.com/SuperMHP/GUPNet | Source-Type: official-code | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10 | 用途：作者实现、配置和补充入口。
[31] KITTI official result detail — GUPNet | https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=9d44d2a957398840c436e4501d76b1008060f563 | Source-Type: official | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10 | 用途：官方test提交身份。
[32] Ma et al. — *Delving Into Localization Errors for Monocular 3D Object Detection* (CVPR 2021) | https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Delving_Into_Localization_Errors_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2021 | Authority: 10/10 | 用途：MonoDLE诊断、远样本、中心与IoU消融。
[33] Ma et al. — MonoDLE official code | https://github.com/xinzhuma/monodle | Source-Type: official-code | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10 | 用途：作者实现和训练配置。
[34] KITTI official result detail — MonoDLE | https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=15205716520b230a261101283a9de63ea0bcda08 | Source-Type: official | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10 | 用途：官方test提交身份。
[35] Zhang et al. — *MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection* (ICCV 2023) | https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_MonoDETR_Depth-guided_Transformer_for_Monocular_3D_Object_Detection_ICCV_2023_paper.pdf | Source-Type: academic/primary | Accessibility: public | As Of: 2023 | Authority: 10/10 | 用途：query匹配、depth guidance、KITTI AP40与消融。
[36] Zhang et al. — MonoDETR supplemental | https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhang_MonoDETR_Depth-guided_Transformer_ICCV_2023_supplemental.pdf | Source-Type: academic/primary-supplement | Accessibility: public | As Of: 2023 | Authority: 10/10 | 用途：Ped/Cyc、matching collapse与附加消融。
[37] Zhang et al. — MonoDETR official code | https://github.com/ZrrSkywalker/MonoDETR | Source-Type: official-code | Accessibility: public | As Of: 2026-08-12 | Authority: 9/10 | 用途：作者实现入口。
[38] KITTI official result detail — MonoDETR | https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=22a0e176d4f7794e7c142c93f4f8891749aa738f | Source-Type: official | Accessibility: public | As Of: 2026-08-12 | Authority: 10/10 | 用途：官方test分数、0.04s与无NMS描述。

### Dropped / 仅定位未采用

[D1] Song et al. [13] — 从冻结15篇的“完整KITTI技术卡”纳入位排除。理由：原文KITTI仅Fig.7(a)定性示例及正文一处Moderate 38.72% AP，用于说明不规则框宽与固定0.41宽高比的annotation bias；主训练、协议化消融与表格结果在Caltech/CityPersons。没有可复现的KITTI baseline→full表、split细节或official-test提交证据。保留为反例来源，不把38.72解释成可迁移收益。

## 3. 逐篇 A–L 技术卡

> 每篇固定字段：A 元数据与链接；B gap；C 模块/损失/匹配/NMS；D split；E 任务/类别/IoU/AP/难度；F baseline/消融/delta/效率；G Ped/Cyc/small/far/遮挡；H 失败与不可比风险；I 与本项目四分支区别；J 借鉴与引用边界；K 可证伪假设与最小实验；L 原始页表路径。

### K01. Geiger et al. — KITTI benchmark protocol

**A. 元数据与官方链接。** Andreas Geiger, Philip Lenz, Raquel Urtasun, “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite,” CVPR 2012, pp. 3354–3361。[原论文][2]；[现行 object evaluation 页面][1]；[数据集主页][3]。用户清单写作“Vision Meets Robotics: The KITTI Dataset / KITTI benchmark protocol”；检测协议应以 CVPR 2012 与持续更新的官方 evaluation 页面为主，IJRR 2013 的 *Vision Meets Robotics* 更偏 raw dataset 描述。

**B. 真正 gap。** 论文要解决的不是网络结构，而是当时实验室数据集与真实自动驾驶之间的偏差：传感器同步/标定、真实城市场景、多任务统一数据、3D标签、隐藏测试集和在线服务器不足。作者明确希望用真实道路中的尺度、遮挡、反光、速度和多目标密度暴露算法假设失效（原文 pp.1–2）。

**C. 数据/评测流程、匹配与后处理。** 采集平台含两对 Flea2 相机、Velodyne HDL-64E、GPS/IMU，标定后在点云中标 3D tracklet，再投影为图像框；标注含 visible/semi-occluded/fully-occluded/truncated（pp.2–4）。检测按分数排序并以最大 IoU 逐个匹配 GT；一个 GT 的重复检测计 FP。原始 CVPR 协议给 2D AP 与 AOS：AOS 在 11 个 recall 点对匹配检测的角度余弦相似度积分（Eq.4–5, p.5）。模型训练目标、NMS 不属于 benchmark，故 N/A；但 `DontCare` 和低于最小高度的检测由评测脚本忽略，不能把它误写成模型 NMS。

**D. 划分。** 现行 object benchmark 为 7481 张公开标签 training、7518 张无标签 test；官方 test 需服务器评测。原论文在 benchmark selection 中明确保证同一 sequence 的图像不会同时进入 train/test（p.4），这是防止视频近邻泄漏的核心。**Chen split 不是 KITTI 官方划分**，而是后续单目3D研究常用的 3712/3769 训练内划分；不能追溯成官方协议。

**E. 任务、类别、IoU、AP与难度。** 现行 2D detection 对 Car 要求 IoU≥0.7，Pedestrian/Cyclist ≥0.5；Easy/Moderate/Hard 分别为最小框高 40/25/25 px、最大遮挡 fully visible/partly occluded/difficult to see、最大截断 15/30/50%。Moderate 用于榜单排序；Hard 中约 2% 框人类也未识别，官方提示其 recall 上限约 98%。2019-10-08 官方从 PASCAL 式 11 recall positions 改为 40 positions；因此早期论文表格是 AP11，今天榜单多为 AP40，**即使类别与 IoU相同也不可直接相减**。另一个版本漂移是：CVPR 2012 正文对其当时 2D 匹配统一写 50%（p.5），现行 Car 规则已为 70%；研究报告必须注明采用哪一版。

**F. baseline/消融/效率。** 这是 benchmark 论文而非检测器，没有“完整方法相对 baseline”的模型增益、参数、FLOPs 或 FPS。论文贡献是提供数据与评测；任何把某榜单方法与本文作模型消融都不成立。2017-04-25 官方还修复了检测结果最小框高过滤 bug；修复前后 Easy/小框 FP 处理不同，是另一个历史不可比点。

**G. Ped/Cyc/small/far/遮挡证据。** 官方难度只通过像素高度、遮挡、截断间接表征 small/far/occlusion，并没有 COCO `APs` 或按米距离分层。25 px 是 Moderate/Hard 的纳入下限，不等于“small AP”。原论文展示了 Pedestrian 的遮挡/截断分布和每帧目标数（Fig.2, p.4），但不提供专门 Ped/Cyc small/far AP。Cyclist 在现行任务中评价，原始论文表1的汇总说明只列两个主要类，反映协议后续演化。

**H. 失败、泄漏与不可比风险。** (1) 隐藏 test 无法做逐目标错误审计；(2) 小数据导致自定义 split 波动，视频连续帧随机划分会泄漏；(3) AP11→AP40 与 2017 脚本修复会改变数值；(4) 2D/AP、AOS、BEV AP、3D AP不可混；(5) Car 0.7 与 Ped/Cyc 0.5不可直接宏平均后宣称“统一定位能力”；(6) 只报告 Moderate 会掩盖近/易样本和 Hard 的相反变化。

**I. 与本项目逐项区别。** plain P3–P5、plain-P2、DCLI、fusion 都是模型/训练干预；KITTI protocol 只决定输入标签和评分。当前项目指标是 Ped/Cyc Moderate AP_R40 macro，且基线接近天花板；早期论文 AP11 的几十点数值不能说明我们的模块还有同等提升空间。P2 退化可能集中在 25–39 px、遮挡或定位误差，但协议本身不能解释机制；必须按框高/距离/遮挡分层取证。

**J. 可借鉴思想与引用边界。** 可借鉴：序列级隔离、严格声明 AP版本、把像素高度/遮挡/截断作为预注册 strata、保留 hidden test 最终一次确认。必须引用：KITTI 数据规模、IoU、难度阈值、AP40变更、`DontCare`处理。不可“复用”为创新：Easy/Moderate/Hard 定义和官方 devkit 是公共协议。

**K. 不抄袭、可证伪假设与最小实验。** 假设 K01-H：plain-P2 的 -1.5918 AP 主要不是“大目标语义损失”，而是 Moderate 中 25–39 px 或遮挡1样本的排序/重复框 FP 增加。最小实验：冻结同一三种子预测，在官方 AP_R40 匹配逻辑下按 `height=[25,40), [40,80), ≥80`、occlusion、truncation 分层，统计 TP/FP/FN、重复匹配 FP 和定位 IoU分布；若退化不集中于小/遮挡层，该假设被否证。该分析不得重训，也不得改变主结果。

**L. 原始页/表/路径。** [2] p.1摘要与采集；Table 1 p.3（PDF页2）数据集比较；Fig.2 p.4（PDF页3）类别/遮挡/尺寸统计；Sec.2.4 p.4（序列隔离）；Sec.2.5 Eq.4–5 p.5（AP/AOS匹配）。[1] lines 35–56（数据量、IoU、难度、2017 bugfix、2019 AP40）。置信度：**High**，因均为官方原始来源；历史版本差异已显式保留。

### K02. Cai et al. — MS-CNN (ECCV 2016)

**A. 元数据与官方链接。** Zhaowei Cai, Quanfu Fan, Rogerio S. Feris, Nuno Vasconcelos, “A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection,” ECCV 2016。[原论文][4]；[作者代码][5]。论文以 VGG-Net/Caffe 实现，源码链接直接出现在 Sec.5（论文 p.10）。

**B. 真正 gap。** Faster R-CNN 的单一深层 feature map 用固定感受野模板覆盖多尺度目标，造成小目标尺寸与感受野不匹配；直接上采样输入虽能改善小目标，却显著增加显存和计算（pp.1–2）。MS-CNN 的 gap 不是“缺一个P2层”，而是**目标尺度、检测分支感受野、训练样本分配三者不一致**。

**C. 网络流程、训练目标、匹配与后处理。** 流程是：VGG trunk → 在 stride 8/16/32/64 的多个中间层接 scale-specific proposal branches（Fig.3）→ 汇总各尺度 proposal → deconvolution 将 conv4-3 特征升采样 → RoI pooling 同时取目标区与更大 context 区 → FC 分类与框回归（Fig.4）。每个分支只接收对应尺度样本；总损失是分支加权和，单分支损失为 softmax cross-entropy + positive-only smooth-L1 box regression（Eq.1–3, pp.5–6；联合 proposal/detection 为 Eq.6, p.8）。正样本由 anchor 与 GT 的最大 IoU 决定，负池 `IoU<0.2`；负样本用 random、bootstrapping 或 mixture 采样，类别正负项再按比例重权（Eq.5）。低层分支前加 buffer convolution，阻断其梯度直接强扰 trunk，作者明确说否则训练不稳定（p.5）。论文正文未给 NMS 阈值，记为 **unknown**，不能凭通用 Faster R-CNN习惯补写。

**D. KITTI split。** 论文说明 7481 train/val、7518 hidden test；消融“follow [3DOP]”在公开 trainval 内拆 train/validation，但正文未重新列具体索引数量，故不武断称 Chen split。全方法 `h768-ctx-c` 用全训练数据后提交 official test。Car 单独训练一个模型，Ped/Cyc 共享另一个模型；还另训 Caltech pedestrian 模型（p.10）。

**E. 类别、任务、IoU、AP与难度。** KITTI 2D detection/proposal；Car/Ped/Cyc；proposal recall 与 test AP 的 IoU 为 0.7/0.5/0.5；Easy/Moderate/Hard 全报。论文发表于 2016，表3/4为当时 KITTI **AP11**，不可与当前 AP_R40 直接相减。它还报告 proposal recall-vs-IoU，不等于 detection AP。

**F. baseline、完整方法、消融、delta与效率。** 关键公平消融均在 val Table 3：`h384→h576`，Car Moderate 80.63→88.14（+7.51），Ped 68.37→70.77（+2.40），但时间约翻倍 0.11→0.22s / 0.09→0.19s；`h576→h576-2x`（feature deconv）Car 88.14→89.12（+0.98），Ped 70.77→72.49（+1.72），仅约 +0.01s；`h576→h576-ctx` Car +0.74、Ped +0.68，但报告参数 471M/217M→863M/357M；加 dimensionality-reduction 的 `h576-ctx-c` 参数降至297M/155M，Car/Ped Moderate 89.13/72.13。proposal-only h576 73.49/60.54，对 unified h576 88.14/70.77，说明二阶段精修贡献远大于单一小模块。Test Table 4：MS-CNN 为 Car 90.03/89.02/76.11、Ped 83.92/73.70/68.31、Cyc 84.06/75.46/66.07；相对 3DOP 的 Moderate 为 Car +0.38、Ped +6.23、Cyc +6.52，但 3DOP 是 stereo 且慢，非纯模块消融；相对 SDP+RPN 为 +0.17/+3.54/+1.72。test 报 0.4s（约2.5 FPS）；作者另称无输入上采样的 KITTI 可到10 FPS。FLOPs 未报告。

**G. Ped/Cyc/small/far/遮挡证据。** Table 2 直接按 Car 框高分层：25–50 px 时 det-8 recall 0.9180、combined 0.9360；50–100 px 由 det-16 最强；100–200 px 由 det-32 最强，实证支持“尺度路由”而非“所有尺度混在同一头”。100 proposals 时 Fig.6 的 recall 约为 Car 62.2、Ped 47.7、Cyc 50.2（曲线中同时考察高IoU区域）；作者明确说 Cyclist val 数量少导致方差大，因此消融只做 Car/Ped。论文没有以米定义 far，也没有 KITTI 遮挡子集 AP；其对 small/occlusion 的更细结论主要来自 Caltech，不能移植成 KITTI 数字。

**H. 失败、限制与不可比风险。** (1) Cyclist 小样本方差被作者直接承认；(2) 输入分辨率本身可带来多AP收益，若不等分辨率会把数据尺度效应错归模块；(3) `h768-ctx-c` 是多组件组合；(4) test SOTA比较混有 stereo、额外数据和不同系统；(5) 参数表数值极大且为作者实现定义，不能无换算与现代YOLO参数量并列；(6) 未报告多seed、FLOPs和 KITTI small/far专门指标；(7) AP11历史口径。

**I. 与本项目逐项区别。** plain P3–P5 是现代 anchor-free YOLO；MS-CNN 是 VGG 两阶段、anchor/proposal/RoI。plain-P2 是把更高分辨率头接入共同检测损失；MS-CNN 则给不同尺度分支**显式限定样本尺度**，并在低层分支前放 buffer conv 稳定梯度。DCLI 是退化条件干预，不是尺度路由；fusion 是特征融合，而 MS-CNN 的 context/deconv只是一部分，其增益依赖 proposal→RoI 精修和 hard-negative sampling。对当前 P2 退化最有因果价值的不是“加低层”，而是作者已观察到低层分支梯度会破坏 trunk、需 buffer，以及分支必须按尺度分配正样本。

**J. 可借鉴思想与引用边界。** 可借鉴思想：P2 只负责预注册的小尺度范围；对 P2→trunk 梯度做隔离/缩放；报告每尺度覆盖率与负样本比例；把 feature upsampling 与 input upsampling分开消融。必须引用 MS-CNN 对 scale-specific branches、buffer conv、bootstrapping 与 deconvolution 的具体设计；不能照搬其结构后换名，也不能把 test 表中的 +6/+7 当作单组件收益。

**K. 可证伪项目假设与最小实验。** K02-H：plain-P2 退化源于“无尺度约束的 P2 正负分配 + 低层梯度占优”，而不是 P2 分辨率无价值。最小实验：不增加新模块，冻结训练配置，只做两个干预的 2×2：P2 positive assignment 仅限框高25–50 px / 不限；P2→trunk gradient stop-or-0.1 scale / 原样。三seed比较总 AP_R40、25–50px recall、duplicate FP和各头梯度范数。若尺度限定和梯度保护都不能改善P2相对P3–P5，则该因果解释被否证。

**L. 原始页/表/路径。** [4] Fig.3与Eq.1–3 pp.5–6；Sec.3.3–3.4 pp.6–7（hard negative与稳定训练）；Fig.4/Eq.6 p.8；Table 1–2 p.10（anchor/按高度recall/split）；Fig.5–6 pp.11–12；Table 3 p.12（val消融、时间、参数）；Table 4 p.13（official test）；Sec.5.2 pp.11–13（作者限制解释）。置信度：**High**；NMS与具体split索引因原文未给，标 unknown。

### K03. Xiang et al. — SubCNN

**A. 元数据与官方链接。** Yu Xiang, Wongun Choi, Yuanqing Lin, Silvio Savarese, “Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection,” arXiv:1604.04693 / WACV 2017 版本。[原论文][6]；[作者补充材料][7]。未从论文或作者页确认可访问的官方代码仓库，故代码为 **unknown/not found**，不以第三方复现代替。

**B. 真正 gap。** 作者指出 KITTI 的尺度、遮挡、截断让 Selective Search/EdgeBoxes 等低层 proposal 成为瓶颈；普通 RPN 也难高效覆盖大尺度变化。第二个 gap 是常规2D检测不输出姿态/可见性等属性。其核心主张是用 pose/appearance subcategory 同时改进 proposal、检测和属性估计，而非泛化的“加注意力”。

**C. 输入与模块流程、损失、匹配、后处理。** 第一阶段：多尺度 image pyramid → CNN feature maps → bilinear feature-extrapolating layer补中间尺度 → 每个 subcategory 一个卷积滤波器输出 heatmaps → 跨subcategory max + threshold生成 RoI → RoI pooling → subcategory softmax与class-specific bbox regression（Fig.2, pp.4–6）。训练 RoI 时按与GT IoU筛正负，预算内取**最低分正样本**和**最高分负样本**做 hard mining；RPN损失为 subcategory CE + positive-only smooth-L1（Eq.1）。第二阶段：RoI映射到合适金字塔尺度，小RoI从更大图像尺度取特征 → 3个FC → subcategory、object class、bbox 三头；总损失为两个 CE + smooth-L1（Eq.2, pp.7–8）。KITTI image pyramid 的显式/外推尺度最终21层；每位置7种高宽比。正文只描述 heatmap threshold 和 box generation，未给最终 NMS阈值，记 unknown。

**D. KITTI划分。** train 3682张（Car 14898/Ped 3154/Cyc 916），val 3799张（13714/1333/711），合计7481（Table 1 p.8）；作者说 follow 3DVP 的 split，不称 Chen split。分析/消融在该val；official test 则用全7481训练并服务器提交（Sec.5.3）。

**E. 任务、类别、IoU、AP规则。** 主任务为 KITTI monocular 2D detection/AP 与 orientation/AOS，另对Car报告 ASA分割和 ALA 3D位置；Car/Ped/Cyc IoU 0.7/0.5/0.5，E/M/H全报。论文历史表格为 AP11。RPN训练有一处需细分：proposal mini-batch超参段写 KITTI/PASCAL统一 70% 正阈值（p.9），检测网络正样本则 Car 70%、其余50%；这与最终评价IoU不能混为同一个匹配规则。

**F. baseline、消融、delta与效率。** Table 4 固定 detection net 更换 proposal：Car Moderate `Faster-RPN(unshared) 82.58→ours 86.64`（+4.06）；Ped 69.32→69.95（+0.63）；Cyc 54.83→59.13（+4.30）。Table 5：同用 Fast R-CNN 检测头，`RPN[7]→our RPN` Moderate 为 Car 77.83→85.20（+7.37）、Ped 68.39→68.55（+0.16）、Cyc 46.36→55.88（+9.52）；这显示proposal收益强烈依类而异。`w/o Extra→Full` Moderate：Car 86.29→86.64（+0.35）、Ped 68.87→69.95（+1.08）、Cyc 55.56→59.13（+3.57），且 Ped AOS Moderate 59.06→58.91（-0.15），不是全指标单调提升。共享特征在 Table4 的 Faster-RPN条件下 Car/Ped下降、Cyc上升；作者进一步报告把自己结构强行共享时训练不收敛。Official test Table7：SubCNN Car 90.74/88.55/77.95；Ped 79.13/66.13/61.27；Cyc 74.40/61.98/54.75。它并非所有类最优：MS-CNN/3DOP在 Ped/Cyc更高。补充材料给 KITTI AlexNet 总时约2.3s（RPN1.5+det0.8），Faster R-CNN约2.0s；参数/FLOPs未报告。

**G. Ped/Cyc/small/far/遮挡证据。** Table3 proposal recall 显示同为2000 proposals，ours vs RPN：Ped Moderate 93.46 vs91.69（+1.77），Cyc 93.03 vs91.80（+1.23），但 Car Moderate反而96.28 vs97.37（-1.09）；“覆盖率高”并非每类都成立。作者把 KITTI 的尺度/遮挡/截断作为方法动机，却没有按像素高度、距离或遮挡级别拆 AP。Cyc训练样本只有916，表4显示共享特征反而有益；这是与本项目跨seed不稳定最直接的样本量证据，但仍是作者单split观察，不是随机种子统计。

**H. 失败、限制、泄漏与不可比风险。** (1) 3DVP subcategory使用 KITTI 3D pose/shape等附加标注，不能与只用2D框的YOLO称同监督；(2) 4倍 image pyramid带来大计算/显存；(3) 自己RPN与detection共享时不收敛，说明融合/共享会产生优化冲突；(4) Ped/Cyc test弱于若干同时代系统；(5) 无多seed、FLOPs和small/far分层；(6) AP11；(7) test SOTA表混合 monocular/stereo，作者虽明确指出3DOP为stereo，读者仍不可直接算公平delta。

**I. 与本项目逐项区别。** plain P3–P5/plain-P2 是YOLO单阶段共享neck/head；SubCNN是两阶段、多输入尺度、显式subcategory与RoI。DCLI根据退化因素处理样本，不等于pose subcategory；fusion试图共享/融合特征，而SubCNN提供了反证：足量Car/Ped时RPN与检测分支共享会下降，架构差异大时甚至不收敛；低样本Cyc共享反而有益。当前AB相对B跨seed为负，与“共享表示并不天然互补”的方向一致，但不能据此宣称已证明同一机制。

**J. 借鉴思想与边界。** 可借鉴：对P2候选做低置信正样本/高置信负样本审计；分开报告coverage与最终AP；把不同任务分支的梯度共享作为可开关变量；低样本类单独测共享正则效应。必须引用 SubCNN 的 subcategory filters、feature extrapolation、hard example selection。不可把 pose标签或3DVP supervision隐去后与plainYOLO作“同数据”比较。

**K. 可证伪假设与最小实验。** K03-H：fusion 导致 AB<B，是因为 DCLI与P2检测需要的梯度方向冲突；该冲突在样本多的Ped比样本少/近饱和的Cyc更明显。最小实验：在现有三seed checkpoint上只读统计 shared layers 中 `cos(g_DCLI,g_det)`，按类和框高分层，并与每类 AP delta关联；另做 stop-gradient fusion 的最小训练对照。若梯度余弦不负、stop-gradient不恢复AB-B，则该解释被否证。

**L. 原始页/表/路径。** [6] Fig.2与Eq.1 pp.4–6；Fig.3/Eq.2 pp.7–8；Table1与metric/split pp.8–9；Table3 p.10（proposal recall）；Table4–5 pp.11–12（固定头与组件消融）；Table7 pp.12–13（official test）；Sec.5.3 p.13。 [7] runtime表。置信度：**High**；代码与NMS阈值因原始来源未确认，明确 unknown。

### K04. Ren et al. — RRC (CVPR 2017)

**A. 元数据与官方链接。** Jimmy Ren, Xiaohui Zeng, Junyang Xu, Yongchao Pan, S. Kevin Zhou, “Accurate Single Stage Detector Using Recurrent Rolling Convolution,” CVPR 2017。[原论文][8]；[作者代码][9]。

**B. 真正 gap。** SSD式单阶段检测器虽然快，但各尺度层各自预测，浅层缺语义、深层缺空间细节，跨层信息只沿backbone单向流动；同时，单一共享框回归器难覆盖不同位置/尺度的回归分布。RRC因此针对的是**相邻尺度特征的反复双向上下文交换**与回归空间离散化，不是简单增加一个高分辨率头。

**C. 网络、循环、损失、匹配与后处理。** 以缩减的VGG-16/SSD为骨干，输入1272×375，预测层为 conv4_3、FC7、conv8_2、conv9_2、conv10_2。每次rolling：高分辨率层经1×1卷积/池化向下传，低分辨率层经1×1卷积/反卷积向上传；相邻信息通道归一化后与本层融合，循环间共享权重（Fig.2–3）。训练做5次rolling，连同初始状态形成6个检测输出，每个输出均施加SSD式 confidence classification 与 localization regression loss；论文没有在RRC正文重新给出SSD正负anchor阈值和精确损失公式，故只记“继承SSD”，不靠惯例补数。每个特征层配置5个独立bbox regressors以离散化回归空间。最终把第3–5次输出合并做NMS；`RRC*`合并第2–6次，说明输出选择本身也是方法的一部分。NMS阈值原文未在已核正文明确列出，记 unknown。

**D. KITTI split。** 作者不是Chen split，而是根据图像相似度构造train/val，使两部分差异尽量大，validation为2741张；val消融只做Car。最终用全公开train训练后提交KITTI hidden test。该自建split提高了“去近邻泄漏”的意识，但也使val数字不能直接对齐MS-CNN/SubCNN的划分。

**E. 任务、类别、IoU、AP规则。** monocular 2D detection，Car/Ped/Cyc，官方IoU分别0.7/0.5/0.5；历史2017表为AP11。Car val Table 2额外扫IoU 0.60–0.80；test对Car给Moderate/Hard，对Ped/Cyc主要列Moderate，不能把表中不同列伪装成统一全难度比较。

**F. baseline、消融、delta与效率。** Table 2同split、同输入下，SSD→RRC在IoU 0.60/0.65/0.70/0.75/0.80的Car AP为 `90.43→94.52`（+4.09）、`90.15→90.81`（+0.66）、`89.16→90.65`（+1.49）、`87.22→89.27`（+2.05）、`76.12→82.82`（+6.70）；RRC*为94.33/90.78/90.32/89.01/82.19，均略低于精选3–5输出的RRC。Official test：Car Moderate RRC(single) 89.85、ensemble 90.19；Table 5 Ped Moderate 75.33；Table 6 Cyc Moderate 76.47。与表内MS-CNN的89.02/71.33/75.46相比为+0.83/+4.00/+1.01，但这是跨方法、非统一训练消融，不能称RRC模块净收益。原文未报告参数、FLOPs或可核FPS，只强调1×1卷积的计算效率，故均记 unknown。

**G. Ped/Cyc/small/far/遮挡证据。** Ped/Cyc只有official-test Moderate的总体AP，没有按像素高度、距离或遮挡状态分层；val因Ped数据不足只做Car，不能据此声称RRC专治远距Ped/Cyc。高IoU 0.80增益明显只说明Car定位/排序更稳，不能等价为small-object证据。

**H. 失败、限制、泄漏与不可比风险。** (1) 第2–6次全合并的RRC*反而弱于3–5次，表明循环并非越深越好，晚期输出可能冗余或过拟合；(2) 无显式memory，作者把它列为未来方向；(3) 自建split、AP11、单次训练、只在Car做val消融；(4) test比较混合不同输入/系统；(5) 多输出合并会同时改变候选数量和NMS竞争，收益不能全归特征传播；(6) 没有参数/FLOPs/FPS与small/far证据。

**I. 与本项目逐项区别。** plain P3–P5/plain-P2为现代YOLO单次neck融合与anchor-free预测；RRC是SSD anchor、多层循环、六次深监督、多输出NMS。DCLI是样本退化干预，不是循环上下文；fusion与RRC都跨尺度交换信息，但RRC的关键反例是：更多迭代/更多输出并不单调更好。它可解释当前AB<B的一种可能——融合增加相关候选与梯度路径，若无选择门会恶化排序/NMS——但RRC没有多seed数据，不能为DCLI跨seed不稳定背书。

**J. 借鉴思想与边界。** 可借鉴：把fusion拆成单向、双向、一次、重复以及“只取中间输出”四个可控变量；审计各阶段候选的重合率与NMS互抑；在同参数预算内比较。必须引用RRC的rolling权重共享、六输出深监督和选择性合并；直接复刻rolling convolution或多regressor结构必须明确来源。

**K. 可证伪假设与最小实验。** K04-H：当前fusion的负delta主要来自高度相关的P2/P3候选在NMS前重复并互相改变排序，而不是融合特征本身无效。最小实验：不重训，分别用B与AB现有三seed预测记录pre-NMS候选，按GT/框高统计跨头同类IoU>0.7重复率、正确框被低质量高分框抑制率及NMS前后AP差；若AB没有增加重复/错误抑制，假设被否证。若后续训练，只允许一次融合与重复融合的最小对照，不引入其他组件。

**L. 原始页/表/路径。** [8] Fig.1–3与Sec.3 pp.2–5（双向rolling、权重共享、回归器）；Sec.4 p.6（输入、优化、split与输出合并）；Table 2 p.7（val IoU扫描）；Table 3–6 pp.7–8（official test Car/Ped/Cyc）；Sec.5 p.8（无memory限制）。置信度：**High**；SSD匹配阈值、NMS阈值、参数/FLOPs/FPS因原文未给而保持unknown。

### K05. Tian et al. — DeepParts (ICCV 2015)

**A. 元数据与官方链接。** Yonglong Tian, Ping Luo, Xiaogang Wang, Xiaoou Tang, “Deep Learning Strong Parts for Pedestrian Detection,” ICCV 2015；用户清单的“DeepParts: Part-based CNN for Pedestrian Detection”是方法俗称而非论文正式标题。[原论文][10]；[KITTI官方结果详情][11]。未在原论文确认作者公开代码入口，记 unknown。

**B. 真正 gap。** 整人模板在遮挡时会被不可见区域噪声污染，手工部件既不一定判别、也难覆盖未知遮挡模式；同时proposal偏移会让固定部件位置失配。DeepParts要从大候选池中学习“强部件”，并用局部平移容忍proposal misalignment。

**C. 输入、模块、训练目标、匹配与后处理。** 上游LDCF生成行人proposals；在2m×m网格（m=3）枚举宽/高至少2格、步长1的45个部件原型，每个部件独立训练ConvNet，再由linear SVM从池中选6个互补部件并融合分数。训练负样本与GT IoU<0.5；mini-batch 16正/48负，10k iterations、lr 0.001。遮挡可见性图以0.4阈值决定part可见。为抗proposal偏移，把网络转全卷积，对n=2的3×3邻域9个位置取“平移后得分减位移惩罚”的最大值，惩罚参数a,b用6-fold CV选。损失细式与LDCF内部NMS阈值没有在论文中完整重述，不能推定；最终是proposal打分/融合，不是YOLO式密集匹配。

**D. 划分。** 核心训练/消融用Caltech：set00–05训练、set06–10测试。**KITTI没有训练/验证划分**：作者把在Caltech训练的全部组件直接迁移到KITTI official test，明确说未用KITTI训练数据。因此它不是“在KITTI上学习部件”的证据，也不存在Chen split。

**E. 任务、类别、IoU、AP规则。** KITTI只评2D Pedestrian，IoU 0.5，Easy/Moderate/Hard，2015历史结果为AP11。论文报告70.49/58.67/52.78；当前KITTI详情页保存71.47/58.15/51.92。两组数值不能平均或择优：很可能涉及评测脚本/服务器历史重算，报告时必须同时保留版本与来源。

**F. baseline、消融、delta与效率。** 论文真正公平消融在Caltech、指标为log-average miss rate（越低越好），不是KITTI AP：Table 4 `LDCF 24.80→AlexNet 21.19`（下降3.61点），换GoogLeNet 17.52（再降3.67），image→box预训练16.43（降1.09），part ensemble13.12（降3.31），shifting11.89（降1.23）；6 parts为12.31，所谓top-6为15.28。Table 3 shifting：Upper part 26.02→23.93、Left 29.21→27.43、Full 16.43→15.41。KITTI论文值Moderate 58.67，作者称比R-CNN高8.54点、但低于Regionlets 61.15；这不是同训练集消融。官方详情现报约1s/image（Matlab/GPU，2.5GHz CPU字段），参数/FLOPs未报告；作者把模型压缩列未来工作。

**G. Ped/Cyc/small/far/遮挡证据。** 方法直接针对遮挡部件，论文在Caltech遮挡设置与可见图上提供证据，但KITTI只给Ped E/M/H总体分数，没有按occlusion/height/distance独立AP；Cyclist完全未评。跨数据集仍获可观分数是泛化证据，却不能证明对KITTI Moderate中的small/far专门有效。

**H. 失败、限制、泄漏与不可比风险。** (1) 45个独立部件ConvNet成本高，作者承认需压缩；(2) KITTI没有本域训练或消融，域差与外部Caltech监督是大混杂；(3) LDCF proposals、ConvNet parts、SVM与shift多阶段，无法把test差异归一模块；(4) 论文与现行官方详情数值不一致；(5) AP11、单次提交、无Ped小/远/遮挡分层；(6) 不评Cyc；(7) Caltech miss rate不能与KITTI AP做delta。

**I. 与本项目逐项区别。** plain P3–P5/plain-P2是端到端密集检测，DeepParts是外部proposal+多个part ConvNets+SVM。DCLI对退化条件干预可能与遮挡概念接近，但DeepParts依赖显式part可见性；fusion是共享特征，DeepParts反而是独立part专家再做分数融合。它对P2退化最相关的是“高分辨率候选若有小位移，固定部件/分配会失配”，不是“部件网络能提高多少AP”。

**J. 借鉴思想与边界。** 可借鉴：在现有预测上做有限局部平移敏感性测试；按可见区域/遮挡模式审核分类分数；将专家互补性与单专家强度分开报告。必须引用部件池、SVM选6部件、shift-max及位移惩罚；不可复制其part定义/选择流程后改名，也不可把Caltech消融称KITTI收益。

**K. 可证伪假设与最小实验。** K05-H：plain-P2的退化部分来自高分辨率头对1–4 px框偏移更敏感，导致Moderate Ped正样本匹配/分类分数不稳定。最小实验：对同一验证集GT与现有P3–P5/P2预测做±1/±2/±4 px局部jitter的只读敏感性曲线，按遮挡和25–50 px分层比较匹配IoU、置信度、TP保留率；若P2不比基线敏感，假设否证。该测试不引入DeepParts结构。

**L. 原始页/表/路径。** [10] Sec.3/Fig.2 pp.3–5（45部件池与选择）；Sec.4/Fig.3 pp.5–6（shifting）；Sec.5.1 pp.6–7（Caltech split/训练）；Table 3–4 pp.7–8（消融）；Sec.5.3/Fig.6 p.9（KITTI跨数据集值）；结论p.10（压缩限制）。[11] 当前官方E/M/H与runtime。置信度：方法/Caltech消融 **High**；KITTI历史分数存在原文—现行服务器漂移，已明确标注为版本不确定性。

### K06. Mao et al. — What Can Help Pedestrian Detection? / HyperLearner（替换TLL+TFA）

**A. 元数据、官方链接与替换理由。** Jiayuan Mao, Tete Xiao, Yuning Jiang, Zhimin Cao, “What Can Help Pedestrian Detection?,” CVPR 2017。[原论文][12]。原指定Song et al. [13]不满足“在KITTI做可追溯目标检测实验”的纳入门：其Sec.实验把Caltech与CityPersons作为可公平评价的紧框/统一宽高比数据，KITTI只在Fig.7(a)展示定性框，并在正文报告固定0.41宽高比框与不规则GT失配导致Moderate 38.72% AP，未给KITTI训练/完整表/消融。因此保留[D1]证据，替换为同样研究小尺度Ped、且在KITTI有Table 1–4的Mao et al.。未从原文或作者页确认官方代码，记 unknown。

**B. 真正 gap。** 低分辨率Ped易与路牌/柱子等背景混淆，需要高层语义；拥挤Ped边界被CNN卷积/池化模糊，又需要低层轮廓。过去额外channel在boosted forest有效，但“哪些外部特征帮助CNN、在哪种分辨率通过何种错误类型起作用”缺少控制实验。第二个gap是直接计算edge/seg/flow/disparity推理代价大，HyperLearner尝试把这些特征改成训练期监督。

**C. 输入、模块、损失、匹配与后处理。** 第一部分以改造Faster R-CNN为baseline：VGG-16去掉conv5以保留高分辨率，anchor由3尺度×3比例增至5尺度×7比例；额外特征走两层随机初始化side branch，与conv4_3拼接，测试ICF、HED edge、COCO预训练语义seg、heatmap、optical flow、stereo disparity。第二部分HyperLearner：取conv1_2/2_2/3_3/4_3，各接两层卷积并统一32通道，高层上采样后拼成aggregated activation map；该图同时送CFN、RPN和FRCNN。CFN对edge用类平衡二元CE、seg用多类CE、其他任务用MSE；总损失 `L_CFN + L_RPN-cls + L_RPN-box + L_FRCNN-cls + L_FRCNN-box`，四个检测权重均1。四阶段训练：CFN→RPN→FRCNN→全网联合；FRCNN优化时proposal坐标视常量。anchor匹配与NMS沿Faster R-CNN，但本文未重列精确阈值，记 inherited/unknown。

**D. KITTI split。** 7481张公开训练图内按参考文献[5]拆成独立train/validation；正文没有重新列索引或样本数，不能无证据写成Chen split或3682/3799。所有KITTI Table 1–4为该validation，不是hidden official test。1×为原约1250×375，2×为输入加倍；直接channel实验还使用邻帧/双目，因此不同channel具有不同输入可得性。

**E. 类别、任务、IoU、AP规则。** KITTI monocular/多输入2D Pedestrian detection，PASCAL IoU 0.5，Easy/Moderate/Hard；2017为AP11。论文说person含Pedestrian/Cyclist两子类，但结果表只报Pedestrian，Cyclist只在FP分类中作为误检来源，不能称评了Cyc AP。

**F. baseline、完整方法、消融、delta与效率。** Table 1（1×、seg channel）baseline 68.96/73.33/60.43；两层随机side branch 70.80/78.15/62.16，对Mod/Easy/Hard为+1.84/+4.82/+1.73；一层随机70.40/75.17/61.92，两层预训练69.92/77.33/61.65，作者观察预训练支路激活过大、过度依赖。Table 3同baseline的1×Moderate：Original image -0.33、ICF -0.56、Edge +0.53、Seg +1.84、Heatmap +1.37、Disparity +1.07、Flow +0.43；2×Moderate分别+0.12/+0.59/+1.13/+1.33/+0.18/+0.51/-0.08，说明并非所有融合都正向且尺度会翻转结论。Table 4 HyperNet→HyperLearner：1× Edge 69.72→71.25（+1.53 Mod），Seg→71.15（+1.43）；相对另一baseline69.80则+1.45/+1.35。2× HyperNet72.23→Edge72.51（+0.28）、Seg72.35（+0.12）；相对baseline71.73则+0.78/+0.62。KITTI未报参数/FLOPs/FPS；Cityscapes另报140/250ms，不能移植为KITTI效率。

**G. Ped/Cyc/small/far/遮挡证据。** Table 2在70% precision按Ped高度：≤80 px recall baseline21.3→seg35.6（+14.3点），80–160为87.6→88.2，>160均96.8；这是最直接的KITTI small证据，但不是AP、也没有多seed。Fig.4在2×约800张val子样本做FP归因，edge相对baseline降低约9点localization-error fraction；定义定位错为与GT有重叠但IoU<0.5。没有按米far或KITTI遮挡等级AP；Cyclist只是FP类别。

**H. 失败、限制、泄漏与不可比风险。** (1) 输入翻倍本身大幅改AP，channel收益随尺度变号；(2) original-image/ICF/flow在部分指标负向，融合不是天然互补；(3) 预训练side branch因激活失衡而退化，是明确优化反例；(4) seg来自MS-COCO、edge来自HED，直接通道实验含外部监督，flow/stereo还增输入；(5) HyperLearner四阶段训练与多任务损失共同变化；(6) 自定义split索引不透明、无official test、AP11、无seed/效率；(7) Table 3“平均提升”混合E/M/H，不应当作独立样本置信度。

**I. 与本项目逐项区别。** plain P3–P5/plain-P2是YOLO单阶段；Mao是Faster R-CNN proposal+RoI。plain-P2与其多层聚合都引入浅层细节，但论文表明浅层/辅助支路若尺度与激活不平衡会退化。DCLI以退化条件选/调样本，而HyperLearner用edge/seg辅助任务；fusion最接近side-branch/aggregated map。当前P2 -1.5918与AB<B在因果上更像其“分支过度依赖、2×时收益缩小/变号”的负证据，而不是Table 4正收益；DCLI跨seed不稳也不能由单split论文数值证明。

**J. 可借鉴思想与边界。** 可借鉴：用激活幅值/梯度占比检查P2或fusion支路是否压过主干；把辅助监督限定训练期；按分辨率与错误类型预注册交互项；用raw-image side branch作“只是多参数/多路径”的安慰剂。必须引用Mao et al. 的channel taxonomy、HyperLearner/CFN和四阶段训练；HED/seg网络与其标签来源也必须披露，不能称纯KITTI 2D框监督。

**K. 可证伪假设与最小实验。** K06-H：P2与fusion退化由浅层支路激活/梯度尺度失衡引起，尤其在已高分辨率输入时边际语义不足。最小实验：在现有三seed checkpoint上记录P2/P3主支路拼接前的RMS激活、归一化后梯度范数及门控权重，按25–80/80–160/>160 px分层关联AP delta；再做不增参数的固定unit-RMS归一化对照。若退化seed并无支路占优，或归一化不恢复，否证该解释。

**L. 原始页/表/路径。** [13] Sec.5开头与Fig.7(a)（论文pp.10–11，KITTI 38.72及排除证据），其Table 1–4为Caltech/CityPersons。[12] Sec.3.1 p.3（KITTI/split/baseline）；Table 1 p.5（支路深度/初始化）；Table 2与Table 3 pp.5–6（高度recall、channel×scale消融）；Fig.4 p.6（FP类型）；Fig.5、Sec.4.1 pp.6–7（HyperLearner、损失、四阶段）；Table 4 p.8（KITTI val）。置信度：Mao技术卡 **High**；精确split索引、NMS/匹配阈值、KITTI效率为unknown；Song排除 **High**。

### K07. Han et al. — Wasserstein Loss-Based Deep Object Detection (CVPRW 2020)

**A. 元数据与链接。** Yuzhuo Han et al., “Wasserstein Loss-Based Deep Object Detection,” CVPR Workshops 2020。[原论文][14]。未确认官方代码，记unknown。

**B. gap。** one-hot CE/BCE把所有错误类别等价处理；自动驾驶中把Car错成Truck与错成Pedestrian的风险/外观距离不同。WLOD希望保留真类判别，同时用类别ground matrix惩罚“严重错类”。它解决class relation，不解决小框定位。

**C. 流程/损失/匹配/NMS。** 不改YOLOv3(Darknet-53)或SSD骨干/框回归与NMS，只把分类目标扩展为原CE/BCE加λ倍Wasserstein距离；预测概率作为source distribution，真类质量集中为target distribution，预定义ground matrix编码组内/组间距离。λ取0.5或1；KITTI把Car/Van/Truck、Tram、Cyclist/Pedestrian/Person(sitting)分三组。框匹配、bbox loss、NMS沿基线，论文未重列阈值。

**D. split。** 排除Misc后把7481张有标签图**随机**三折2495/2493/2493交叉验证；不是Chen split、不是官方test，且未说明序列级隔离/随机种子，存在相邻帧泄漏和不可复现风险。

**E. 口径。** 自定义七细类2D mAP；论文没有报告官方Car/Ped/Cyc E/M/H，也没有清楚绑定KITTI官方0.7/0.5类别阈值或AP11/AP40。故Table 2不能与本项目Ped/Cyc Moderate AP_R40相减。

**F. 数值/效率。** Table 2：YOLOv3 68.3，W0.5-y 69.2（+0.9），W1-y 69.4（+1.1）；SSD 72.8，W0.5-s 74.7（+1.9），W1-s 74.4（+1.6）。没有每折方差、每类AP、参数/FLOPs/FPS；λ最优随backbone不同，已反驳“固定Wasserstein必稳增益”。

**G. Ped/Cyc/small/far。** Ped/Cyc/Person被放同风险组，定性图提及Rider/Bike漏检和同一目标多类输出，但KITTI无Ped/Cyc单类AP、尺寸/距离/遮挡分层，不能宣称改善小目标。

**H. 风险。** ground matrix人工设计；随机图像三折可能泄漏；任务口径不透明；仅两种λ、无seed/显著性；mAP提升可能来自标签合并/相似类容忍而非检测召回；NMS/训练细节不足。

**I. 项目区别。** plain/P2/fusion改特征，DCLI改退化条件；WLOD只改分类几何。它最接近项目中Ped↔Cyc混淆审计，但不能解释P2定位或DCLI种子波动。

**J. 边界。** 可借鉴“错类成本矩阵”作为诊断或预注册loss对照；必须引用WLOD与人工ground matrix，不得把三折mAP称官方KITTI AP提升。

**K. 假设。** K07-H：DCLI增益不稳主要来自少量Ped↔Cyc排名翻转。最小实验先不训练：按三seed统计混淆矩阵与交换标签后oracle AP；若不稳定delta不集中在互错类，否证。再决定是否做仅分类项的成本敏感loss。

**L. 原始路径。** [14] Fig.1–3、Sec.3 pp.2–4（损失/矩阵）；Sec.4与Table 2 pp.5–6（λ、三折、分组、mAP）。置信度：方法/表值High；官方协议对应性Low且已显式拒绝比较。

### K08. Simon et al. — Complexer-YOLO (CVPRW 2019)

**A. 元数据与链接。** Martin Simon et al., “Complexer YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds,” CVPRW 2019。[原论文][15]；[KITTI官方详情][16]。检索到的Complex-YOLOv3仓库为第三方，未作为官方代码。

**B. gap。** LiDAR BEV检测缺少图像语义，小类/稀疏目标困难；3D IoU用于训练匹配计算重；逐帧检测缺时序稳定。论文将RGB semantic segmentation投影到点云、引入快速SRT score、再接在线多目标跟踪。

**C. 流程/损失/NMS。** RGB经ENet式语义分割给每点语义标签；点云体素化为BEV，21个高度channel加反射强度/占用/语义，经Complex-YOLO式单阶段网络预测类别、3D中心/尺度和以复数形式回归yaw。SRTs把scale、rotation、translation相似度组合，用作更快匹配/训练评价替代昂贵3D IoU；后处理NMS，再用特征/运动的在线tracking。精确loss权重与NMS阈值应由论文实现表取；原文未提供可核官方代码，本文不补猜。

**D. split/test。** 论文把检测与MOT提交KITTI official test；Sec.4.3称固定training setup做消融，却没有在可核正文重新列train/val索引，故消融split unknown。不是Chen split证据。

**E. 口径。** KITTI AOS、BEV、3D detection与MOT，Car/Ped/Cyc，Car IoU0.7、Ped/Cyc0.5，E/M/H；2019主表是历史AP11。与本项目2D AP_R40完全不同。

**F. 数值/效率。** 原文Table 1 historical 3D：Car55.63/49.44/44.13，Ped19.45/15.32/14.80，Cyc28.36/23.48/22.85；BEV表另见论文。当前official详情[16]的Ped 3D为17.60/13.96/12.70、BEV21.42/18.26/17.06，显示服务器/AP版本漂移。FPS 15.6（约64ms）。消融文字：21高度channel与51相近，51更慢；SRTs推理最高快20%、训练近减半；但语义融合、SRT、tracking的完整逐项AP数表未充分报告，不能析出“语义净增益”。参数/FLOPs unknown。

**G. Ped/Cyc/small/far。** official test确报Ped/Cyc 3D/BEV，但绝对值远低Car，反映点云稀疏/小体积困难；没有按距离或点数分层。语义标签可能帮类别，却无公平“去语义”Ped/Cyc delta。

**H. 风险。** 依赖LiDAR和预训练图像分割；多系统组件；AP版本漂移；SRT既影响训练又影响速度；MOT按2D框处理DontCare/ignored，作者承认与3D目标不完全一致；无seed、split与完整消融不透明。

**I. 项目区别。** 本项目纯RGB 2D；Complexer是RGB语义+LiDAR 3D+tracking。其“fusion”跨传感器，不能为P2/DCLI加分。唯一因果关联是21 vs51高度bins相近：更细粒度输入未被网络利用，类比P2增加分辨率未必转化为有效信息。

**J. 边界。** 可借鉴信息利用率审计与按候选匹配开销分解；语义点云/SRT/复数yaw必须引用，不可称2D方法创新。

**K. 假设。** K08-H：P2退化因为额外高分辨率通道高度相关、有效秩未增加。最小实验在已有checkpoint估算P2融合前后特征谱有效秩/CKA与25–50px AP delta；若P2提供独立特征仍退化，否证“冗余”解释。

**L. 原始路径。** [15] Fig.1–3与Sec.3 pp.2–6；Table 1–2 pp.7–8（test detection/MOT）；Sec.4.3 pp.8–9（21/51 channel、SRT）；[16]现行Ped详情。置信度：方法High；消融净delta Medium-Low。

### K09. Brazil & Liu — M3D-RPN (ICCV 2019)

**A. 元数据与链接。** Garrick Brazil, Xiaoming Liu, “M3D-RPN: Monocular 3D Region Proposal Network for Object Detection,” ICCV 2019 Oral。[原论文][17]；[官方代码][18]。

**B. gap。** 单目3D缺深度，既有方法依赖外部depth/CAD/多阶段；普通卷积平移不变，却忽略道路相机中图像纵向位置与深度的强几何关系。目标是共享2D/3D anchors的一阶段proposal网络并使高层特征具位置/深度意识。

**C. 网络/损失/匹配/NMS。** DenseNet-121 backbone；每个2D anchor同时带从训练统计初始化的3D中心投影、深度、尺寸、朝向先验，预测类别、2D框及3D参数。depth-aware convolution将特征图按水平位置分b个bin，对各bin用独立卷积核，再与普通卷积分支融合；3D→2D post-optimization调整yaw，使预测3D框投影与2D框一致。多任务分类与smooth-L1式回归、anchor匹配/NMS在正文定义；这里只保留其共享anchor事实，未核出的阈值不补猜。

**D. split。** 同时报告val1=3712/3769（通常Chen split）、val2=3682/3799与全训练official test；官方代码含`kitti_split1/2`。这是少数显式展示split敏感性的论文。

**E. 口径。** monocular 2D、BEV、3D；Car/Ped/Cyc共享单模型；Car IoU0.7，Ped/Cyc0.5；E/M/H；2019论文AP11。其3D/BEV不可与项目2D AP_R40比较。

**F. 数值/消融/效率。** Table 5 Car Moderate：普通卷积、无post为AP2D/3D/BEV 82.16/10.99/12.99，加入post为82.16/15.08/17.47，即3D+4.09、BEV+4.48；b=4/8/16/32并含post为12.87/17.91、14.46/19.14、16.04/20.99、17.06/21.18（3D/BEV），非单调且b16→32 3D+1.02、BEV+0.19。runtime 118ms baseline，post 128ms，b32 161ms（1080Ti）；depth-aware开销2–20%，post约13ms/8次迭代。Table 1/2 val1 Car APBEV 25.94/21.18/17.90，AP3D20.27/17.06/15.21；test AP11为26.43/18.36/16.24与20.65/15.70/13.32。Table 3 Moderate multi-class test：Ped BEV/3D 11.35/10.54，Cyc 1.29/1.03，显露共享模型并不保证小类成功。参数/FLOPs未报。

**G. Ped/Cyc/small/far。** Fig.5给Ped/Cyc定性；Table3直接显示Ped/Cyc尤其Cyc test崩落，而val2 Cyc约9.09，强烈提示小样本/split/域偏移。没有距离/遮挡分层。Car depth-aware收益不能外推Ped/Cyc。

**H. 风险。** 利用固定相机/道路位置先验，跨相机泛化弱；需要3D标签；post依赖可靠2D框；b非单调且增计算；val1/val2/test差异大，Cyc尤其不稳；AP11、无seed；共享多类可能被Car支配。

**I. 项目区别。** plain/P2是2D特征层；M3D-RPN增加3D先验与按纵向位置变卷积。DCLI若显式处理雾/低照等退化也不同；fusion仅在“普通+depth-aware分支融合”形式相似。其Cyc跨split崩落与DCLI跨seed不稳有同类统计警示，但任务/监督不同，不能归同一机制。

**J. 边界。** 可借鉴：同时报告两split/多seed；把位置条件化作为可证伪变量；检查b的非单调性。必须引用共享2D/3D anchors、depth-aware bins和3D→2D优化。

**K. 假设。** K09-H：DCLI的seed不稳由Cyc有效样本过少、且退化条件与图像纵向位置耦合造成。最小实验按类×框高×y-center统计每seed有效TP数和delta，做固定分层bootstrap；若不稳定不集中于稀疏strata，否证。

**L. 原始路径。** [17] Fig.2–4/Sec.3 pp.3–6；Table1–3 pp.6–7；Table5 p.8；Table6 p.9；[18] split1/split2与warmup/main。置信度High；历史AP11与当前AP40严禁混用。

### K10. Ding et al. — D4LCN (CVPR 2020)

**A. 元数据与官方链接。** Mingyu Ding, Yuqi Huo, Hongwei Yi, Zhe Wang, Jianping Shi, Zhiwu Lu, Ping Luo, “Learning Depth-Guided Convolutions for Monocular 3D Object Detection,” CVPR 2020, pp.11672–11681。[原论文][19]；[补充材料][20]；[官方代码][21]。

**B. gap。** 单目3D检测缺深度；普通2D卷积对所有样本、位置和尺度复用全局核，难表达随距离变化的物体尺度/局部几何；pseudo-LiDAR又对深度误差敏感并丢失RGB高层语义。D4LCN的真正问题是“如何让RGB卷积被深度条件化”，不是一般小目标检测。

**C. 模块、损失、匹配与后处理。** RGB与预估depth两支并行；depth支生成sample-wise、position-wise、channel-wise的动态局部核及离散dilation，作用于RGB特征。depth-wise设计降计算，shift-pooling在不显式展开大局部核的情况下交换邻域信息；多层特征汇合到单阶段3D head，回归2D框、投影3D中心、深度、尺寸、allocentric朝向和3D corners。损失由分类与2D/3D/角点回归组成，3D anchors/分配和NMS沿论文基线实现；正文未给足以脱离代码复写的全部阈值，故不补猜。动态dilation并非“加P2”，它改变每位置的感受野与滤波器。

**D. split。** 正文同时报split1=3712/3769、split2=3682/3799及全7481训练后的hidden official test（Table 4）；深度网络来源/训练集必须单列，因为使用DORN、DispNet或PSMNet会改变监督与输入可得性。split1常被称Chen split，但这不是KITTI官方划分。

**E. 任务口径。** monocular 2D/BEV/3D；Car IoU 0.7，Ped/Cyc 0.5，Easy/Moderate/Hard。Table 2在同一split明确并列AP|R11和AP|R40；Table 4跨早期val与官方test的数值沿当时社区表述，正文未逐列重新标R11/R40，禁止把该表不同列当同一采样规则横减。项目2D Ped/Cyc AP_R40只能参考其2D附表，不能引用3D增益。

**F. baseline、消融、delta、效率。** Table 2（split1 Car 3D）：3DNet baseline AP40 E/M/H=16.72/12.13/9.46；+CL=17.10/12.09/9.47，Moderate **-0.04**；+DLCN=18.32/13.50/10.61；+SP=19.69/14.44/11.52；完整D4LCN=22.32/16.20/12.30。因此若以+CL作普通卷积对照，Moderate +4.11；若以SP-DLCN作最邻近对照，动态dilation +1.76。AP11同表baseline 17.94/14.61/12.74，完整26.97/21.71/18.22，Moderate +7.10，显示AP口径会显著改变“提升”。补充Table 1：Dynamic 17.47/12.18/9.53、Dynamic Local 21.09/13.93/11.31、Deformable 19.05/13.42/10.07、D4LCN 22.32/16.20/12.30（均AP40）；不是任意动态卷积都等价。正文Table 3换深度源，AP40 3D E/M/H：MonoDepth 16.82/13.18/10.87；DORN 22.32/16.20/12.30；DispNet 25.73/18.56/15.10；PSMNet 25.24/19.80/16.45，输入深度质量是强混杂。正文未完整报告端到端参数/FLOPs/FPS，尤其外部depth网络开销不能省略，记unknown。

**G. Ped/Cyc/small/far/遮挡证据。** Fig.5只给远车/行人的定性案例并宣称adaptive dilation覆盖尺度，没有距离/框高/遮挡分层AP。Table 4的3D结果（split1/split2/test）显示Ped Moderate 11.23/10.37/3.42、Cyc Moderate 4.41/6.54/1.67；跨split/test塌缩远大于常见单模块收益，不能用Car结果代替Ped/Cyc证据。完整2D split1 AP40为94.25/86.93/70.34（Car E/M/H），仍没有Ped/Cyc 2D消融。

**H. 失败、限制、泄漏与不可比。** (1) +CL Moderate略降，组件非单调；(2)深度源替换带来的变化可大于结构变化，且stereo深度不再是纯单目条件；(3) Table 4 Ped/Cyc val→test大幅坠落，提示少样本/split/排行榜泛化风险；(4)无多seed；(5)外部depth训练数据若与KITTI val重叠会产生公平性风险，论文没有完整的数据血缘审计；(6)AP11/AP40、2D/3D、Car/Ped/Cyc、不同IoU不能混；(7)动态局部卷积和全系统同时变化，无法把完整增益归给单一“高分辨率”。

**I. 与项目四分支区别。** plain P3–P5/plain-P2是纯RGB 2D YOLO尺度层；D4LCN是depth-conditioned 3D卷积与3D head。DCLI按退化条件调节，本方法按每像素深度调核；fusion虽同为双支，但D4LCN用乘性动态滤波而非普通特征拼接。与P2退化最相关的是：更细局部并不自动有效（+CL下降、局部变体非单调），必须由与尺度因果相关的条件信号控制；这不能反向证明项目应加深度。

**J. 可借鉴与引用边界。** 可借鉴“尺度条件化而非无条件加细节”、分解普通/局部/depth-wise/shift/dilation、同时报R11/R40的实验设计。若实现深度生成动态核、shift-pooling或其dilation离散化，必须引用D4LCN；若采用DORN/DispNet/PSMNet还需披露其监督和成本。不得写成“D4LCN在KITTI 2D Ped/Cyc提升4.11 AP”。

**K. 可证伪假设与最小实验。** K10-H：P2退化来自无条件高分辨率特征在远小目标与背景间共享同一滤波，而非分辨率本身。最小实验不引入depth：在既有checkpoint上按框高与y-center计算P2/P3门控、激活熵、梯度范数和AP delta；再以停止梯度的框尺度代理做两档条件门控，只训练门控参数。若退化不集中于门控失配层，或条件门控三seed不能一致改善，否证。

**L. 原始页/表/路径。** [19] Fig.2–4、Sec.3 pp.3–6（D4LCN/shift/dilation/head）；Table 1–4 pp.7–9（比较、逐组件、深度源、多split/test）；[20] Table 1 p.1及“D. Ablative Results” pp.1–2（动态/局部/可变形卷积与组件）；[21]作者代码入口。置信度：结构和Table 2/3 **High**；Table 4历史AP采样列 **Medium**，因此显式拒绝跨列delta；端到端效率unknown。

### K11. Qian et al. — End-to-End Pseudo-LiDAR (CVPR 2020)

**A. 元数据与官方链接。** Rui Qian et al., “End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection,” CVPR 2020, pp.5881–5890。[原论文][22]；[补充材料][23]；[官方代码][24]。

**B. gap。** 传统pseudo-LiDAR把stereo depth和LiDAR-style detector分开训练，深度的全图误差目标与最终物体检测不对齐。论文目标是通过可微change-of-representation（CoR）让检测损失反传至depth；它解决跨任务目标对齐，不是证明“融合必然正收益”。

**C. 流程、损失、匹配和后处理。** 双目图→stereo depth/disparity网络→用相机内外参与可微CoR把像素深度映射成3D pseudo-points→PointRCNN；另一实现把点软量化进BEV后接PIXOR。总损失 `L=λ_depth L_depth+λ_det L_det`；文中取λ_depth=1，PointRCNN λ_det=0.01、PIXOR=0.1，使两类梯度总量可比。PointRCNN的RPN/RCNN分配、box回归和NMS，以及PIXOR的BEV dense匹配/NMS沿原detector；论文贡献不在重新设计匹配。训练深度仍用KITTI同步LiDAR生成的监督，推理才只需stereo。

**D. split。** 在KITTI有标签train内做train/val并在全train训练后提交official test；论文沿用pseudo-LiDAR协议，但正文没有给序列级隔离审计，本卡不把它写成KITTI官方split。训练监督需要双目、标定和LiDAR深度；与本项目单目RGB 2D数据条件不同。

**E. 口径。** 只评Car，BEV/3D，IoU 0.5与0.7，E/M/H。Tables 2–5列为“AP”但没有显式标注11还是40 recall positions；因此即使数值来自KITTI，也不与项目Ped/Cyc AP_R40横减。没有Ped/Cyc结果。

**F. baseline、逐路径消融、delta与效率。** Table 2 val、IoU.7：PL++/PointRCNN BEV 82.0/64.0/57.3、3D 67.9/50.1/45.3；E2E为82.7/65.7/58.4与71.1/51.7/46.7，Moderate +1.7 BEV/+1.6 3D。PIXOR Moderate BEV 61.1→64.3，+3.2。Table 3 official test：PL++/PointRCNN BEV 78.3/58.0/51.3、3D61.1/42.4/37.0；E2E 79.6/58.8/52.1、64.8/43.9/38.1，Moderate +0.8/+1.5；PIXOR Moderate 48.3→51.7，+3.4。Table 4（val PointRCNN，IoU.7 Moderate BEV/3D）：冻结全部64.0/50.1；只训depth 64.5/50.5；只训RPN 63.9/50.0；只训RCNN 63.4/50.4；RPN+RCNN 64.0/51.2；depth+RPN 64.7/50.7；depth+RCNN65.5/51.0；三者全训65.7/51.7。单路径可以负向，完整联合才最佳。Table 1给像素梯度：depth loss约3%非零、均值1e-5、总量0.1；PointRCNN det约4%、1e-3、10；PIXOR约70%、1e-5、1，直接说明必须按梯度总量配λ。参数/FLOPs/完整stereo+detector FPS未统一报告，记unknown；不能只报detector耗时。

**G. Ped/Cyc/small/far/遮挡。** 无Ped/Cyc、框高/距离/遮挡AP。补充材料以近物体区域的深度均值/方差和边界可视化解释检测梯度可能锐化物体轮廓，但作者也指出detector前景只覆盖约10%像素，整体深度改善有限；这是机制观察，不是KITTI small AP证据。

**H. 失败与不可比。** (1) 单独解冻RPN/RCNN可负向，证明end-to-end不是单调收益；(2) λ相差10–100倍，未平衡时强分支会压制弱分支；(3) stereo、LiDAR depth supervision、标定与重型3D detector均是项目没有的输入/监督；(4)只Car；(5)AP采样规则未明；(6)无多seed、参数/FLOPs与统一总时延；(7)软量化PIXOR提升不能与PointRCNN混为同一组件；(8)论文没有训练图像地理/序列泄漏审计。

**I. 项目区别与因果关联。** plain/P2/DCLI/fusion均为单目RGB 2D；该文是stereo→3D表示→LiDAR detector。其最可转移证据不是AP数值，而是Table 1/4：支路梯度密度和总量极不平衡，部分联合训练会比冻结baseline差。项目AB<B与DCLI跨seed不稳定可能同样来自fusion/P2支路的梯度支配，但在实测梯度前只是候选机制。

**J. 借鉴与边界。** 可借鉴逐支路freeze/unfreeze矩阵、报告梯度非零密度/均值/总量、用权重平衡而非凭参数规模猜测。若使用可微pseudo-LiDAR CoR、soft quantization或其联合损失，必须引用该文并披露stereo/LiDAR监督；不得将Car 3D +1.5 AP写成KITTI 2D Ped/Cyc或“融合通常+1.5”。

**K. 假设与最小实验。** K11-H：项目fusion负向和DCLI种子波动由辅助支路的梯度总量/密度相对主干失衡造成。最小实验：不重跑先从三seed现有训练日志/checkpoint估计每损失对融合前P2/P3参数的非零梯度密度、均值、总L2范数与夹角；若失衡存在，再仅做baseline frozen、只主支、只辅支、全训四格微型训练，并用固定grad-norm比例。若负向seed没有更强失衡，或平衡后仍一致退化，否证。

**L. 原始页/表/路径。** [22] Fig.2与Sec.3 pp.3–5（CoR/soft quantization/联合损失）；Table 1 p.6（梯度尺度）；Tables 2–5 pp.7–9（val/test与freeze矩阵）；[23] depth统计、边界和补充训练细节；[24]作者代码。置信度：表值/梯度机制 **High**；精确train/val索引、AP11/AP40、总效率 **unknown/显式不可比**。

### K12. Chen et al. — MonoRUn (CVPR 2021)

**A. 元数据与官方链接。** Hansheng Chen, Yuyao Huang, Wei Tian, Zhong Gao, Lu Xiong, “MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation,” CVPR 2021, pp.10379–10388。[原论文][25]；[补充材料][26]；[官方代码][27]；[KITTI官方提交][28]。

**B. gap。** 基于dense 2D–3D correspondence+PnP的6DoF方法通常需要CAD/表面几何GT，真实道路很难取得；直接回归单目深度又高度不确定。MonoRUn用3D box即可自监督学习归一化物体坐标（NOC），再把aleatoric/epistemic不确定性传播到PnP、协方差和打分。

**C. 流程、损失、匹配、NMS。** FPN式2D detector产生RoI；regional reconstruction network解码3×28×28 NOC与2×28×28 aleatoric uncertainty，同时回归尺寸。预测NOC投影回图像，以Robust KL对重投影残差加权；推理用uncertainty-aware PnP估位姿/协方差，MC dropout估epistemic，score head学习3D质量。总目标含2D detection、projection、dimension、score与calibration（λ_calib=0.01）；可选稀疏LiDAR NOC监督。作者另测可微PnP端到端refine，但只追加15 epochs且强基线反而下降。3D NMS IoU阈值0.01；MC dropout 50次。匹配由两阶段2D detector/RoI流程决定，非YOLO anchor-free匹配。

**D. split。** 训练集内部用常见3712/3769 train/val，并以全7481训练提交hidden test；AdamW，batch 6、2×2080Ti，完整集32 epochs、split 50 epochs，端到端PnP再追加15 epochs（正文/补充实现节）。额外LiDAR版本必须和纯box自监督版本分开。

**E. 口径。** monocular 3D为主，另报BEV/2D；Car IoU0.7，Ped/Cyc0.5；E/M/H；正文明确使用40 recall positions，即AP40。Table 3的“mAP”是六个Car指标的内部平均，不是COCO mAP，更不是项目Ped/Cyc宏平均。

**F. baseline、消融、delta与效率。** Table 1 official test Car 3D AP40：无额外监督16.04/10.53/9.11；加LiDAR NOC 19.65/12.30/10.58，E/M/H +3.61/+1.77/+1.47，但监督升级不可称纯模块增益。val IoU.7：17.26/12.27/10.41→20.02/14.65/12.61；IoU.5：55.88/40.03/35.59→59.71/43.39/38.44。Table 3六指标均值：Robust-KL+self+LiDAR 31.21；self only 28.57；LiDAR only18.84；Smooth-L1 26.35；Laplace-KL29.47；mixed-KL30.05；Lap-KL+E2E29.73（+0.26）；Robust-KL+E2E31.09（**-0.12**）；Robust-KL+epistemic dimension31.47（+0.26）；full epistemic sampling31.16（**-0.05**）；去latent29.78（-1.43）。完整推理约0.070 s/图（Table 1）；正文未统一给参数/FLOPs，50次MC sampling的成本需包含。

**G. Ped/Cyc/small/far/遮挡。** Table 2 official test 3D AP40 Ped：10.88/6.78/5.83；Cyc：1.01/0.61/0.48。相对列出的MonoPair，Ped Moderate 6.68→6.78仅+0.10，而Cyc远低；作者直接归因于Cyclist训练样本不足。补充材料纯自监督多类结果同样显示Cyc极弱。没有框高、距离或遮挡分层AP；不能把uncertainty自动解释为small/far提升。

**H. 失败、限制、泄漏、不可比。** (1) robust强基线上E2E PnP -0.12、full epistemic -0.05，更多不确定性传播非单调；(2)Cyclist几乎失效；(3)LiDAR监督版本改变信息条件；(4)自监督几何只被3D box投影约束，形状可过拟合，LiDAR充当正则；(5)calibration在训练分布拟合，域移时可能过度自信；(6)作者指出DORN-based方法的depth训练数据可能与KITTI object val重叠，构成明确公平性/泄漏风险；(7)无多seed；(8)3D NMS、MC采样与项目2D后处理不同。

**I. 项目区别与因果关联。** plain/P2是2D尺度层，DCLI是退化条件，fusion是特征融合；MonoRUn是RoI级NOC重建+几何PnP+不确定性传播。相关警示是“不确定性机制与全链路传播可能在强baseline上负向”，与DCLI跨seed不稳方向一致但不能归同因。其Cyc失败支持先查有效样本/类别稀疏度，而不是用总AP掩盖小类。

**J. 借鉴与边界。** 可借鉴按aleatoric/epistemic拆分、校准误差与分层样本量一起报告，并设“局部使用/全链路传播”的消融。实现Robust KL、uncertainty-aware PnP/NOC reconstruction必须引用；额外LiDAR结果必须标注。不可把+1.77写成纯RGB 2D增益。

**K. 假设与最小实验。** K12-H：DCLI波动主要来自Cyc/Hard样本少时条件权重过度自信，且全链路门控放大校准误差。最小实验：对三seed按类/难度/框高计算ECE、Brier、有效TP数、门控权重方差；比较只在head末端使用DCLI与向P2–P5全链传播。预注册：若不稳seed没有更差校准/更少有效样本，或局部门控不降低方差，否证。

**L. 原始页/表/路径。** [25] Fig.2、Sec.3 pp.3–6（NOC、Robust KL、PnP、uncertainty）；Sec.4与Tables 1–3 pp.7–9（AP40、Ped/Cyc、消融、时延、泄漏讨论）；[26]多类纯自监督、训练/MC实现细节；[27]作者代码；[28]官方提交身份。置信度：方法/Table 1–3 **High**；参数/FLOPs unknown。

### K13. Lu et al. — GUPNet (ICCV 2021)

**A. 元数据与官方链接。** Yan Lu, Xinzhu Ma, Lei Yang, Tianzhu Zhang, Yating Liu, Qi Chu, Junjie Yan, Wanli Ouyang, “Geometry Uncertainty Projection Network for Monocular 3D Object Detection,” ICCV 2021, pp.3111–3121。[原论文][29]；[官方代码][30]；[KITTI官方提交][31]。

**B. gap。** 透视几何用 `depth=f·h3D/h2D` 引入强先验，但3D高度仅0.1m偏差在远处可放大成约4m深度偏差；训练早期的高度噪声还会经层级任务放大，令置信度失真和优化不稳。GUPNet分别处理推理可靠性（几何不确定性）与训练依赖（HTL），不是泛化的“加不确定性”。

**C. 模块、损失、匹配与后处理。** DLA-34+CenterNet式heatmap、2D offset/size产生框；RoIAlign后拼接归一化coordinate map，预测projected 3D center、角度和3D尺寸。GUP把3D高度和learned depth bias都建模为Laplace分布，经投影传播均值/方差，`σ_d=sqrt(σ_p²+σ_b²)`；深度以Laplace NLL训练，推理置信度 `p_depth=exp(-σ_d)`，最终分数 `p3D=p2D·p_depth`。HTL把2D→基础3D→depth设成三层先修关系，按近K=5 epoch损失导数的学习状态动态从0到1开启下游loss。CenterNet中心匹配/3×3 max-pool NMS沿基线；论文没有给独立3D IoU NMS的新贡献。

**D. split。** 明确3712 train/3769 val做消融；全7481训练后official test。输入380×1280、stride4；140 epochs、batch32、3×Titan XP、前5 epochs warm-up。没有额外depth/LiDAR数据。

**E. 口径。** monocular 3D与BEV，AP40；Car IoU0.7，Ped/Cyc0.5；E/M/H。正文Table 1 test三类，Table 2/3/4为Car val。其Ped/Cyc是3D AP40，不是项目2D AP40。

**F. baseline、完整方法、消融、delta和效率。** Table 3 Car val AP3D@.7 E/M/H：裸baseline(a)15.18/11.00/9.52；+coordinate map(b)16.39/12.44/11.01；+UnC(c)19.69/13.53/11.33；+GeP无UnC(d)17.27/12.79/10.51；GeP+直接UnC(e)18.23/13.57/11.22；+GeU(f)20.86/15.70/13.21；+HTL无GeU(g)21.00/15.63/12.98；完整(h)22.76/16.46/13.72。最邻近f→h，HTL Moderate +0.76；g→h，GeU +0.83；但c→e加入传统GeP使Easy -1.46、Hard -0.11，显示交互非单调。Table 4：GradNorm 16.19/10.49/9.04、task uncertainty18.95/13.94/12.18、HTL22.76/16.46/13.72；通用权重平衡在层级依赖下明显失效。Table 2完整val Car AP3D@.7=22.76/16.46/13.72。Table 1 official test Car20.11/14.20/11.77；Ped14.72/9.53/7.87；Cyc4.18/2.65/2.09。29.4 FPS/Titan XP；参数/FLOPs unknown。

**G. Ped/Cyc/small/far/遮挡。** 论文给trainval实例数Car/Ped/Cyc=14,357/2,207/734，直接把Cyclist弱归因于样本少；Cyc仍落后LiDAR-supervised方法。Fig.6给远距/遮挡坏例，其GeU较高，但没有按距离或框高AP。传统GeP在Table 3 Hard两组下降，作者明确联系heavy occlusion/truncation；这是失败证据，不是“远小目标已解决”。

**H. 失败、限制、泄漏、不可比。** (1)c→e存在负交互；(2)GradNorm/通用task uncertainty远低于HTL，说明错误的平衡器会恶化；(3)HTL根据早期K=5梯度趋势，单个split/seed下可能把噪声当学习状态；(4)Ped/Cyc少样本且无其消融；(5)confidence由模型内假定的Laplace方差映射，未做域外校准；(6)无参数/FLOPs、多seed、距离分层；(7)3D几何误差不能直接解释2D AP；(8)Table 1“vs None”是跨论文SOTA差，不是受控baseline delta。

**I. 项目区别与因果关联。** plain/P2是YOLO尺度层，DCLI按退化条件调节，fusion是特征支路；GUPNet是RoI级几何概率与有向任务图。与项目最紧密的是HTL反例：若P2/fusion依赖尚未稳定的主干语义，固定同时训练会放大误差；但GradNorm失败说明不能直接套通用平衡。DCLI跨seed波动可能来自用短窗口训练状态做门控时放大噪声，需要多seed验证。

**J. 可借鉴与引用边界。** 可借鉴先画任务依赖DAG，再做阶段开启；同时与固定权重、GradNorm、task-uncertainty比较；报告坏例置信度和实例数。若采用几何不确定性传播、UnC分数或HTL公式必须引用GUPNet。不可写“GUPNet让KITTI Ped提升4.7 AP”为单模型消融；4.7是对先前无额外数据方法的跨论文Easy差。

**K. 假设与最小实验。** K13-H：P2/fusion负向源于辅助分支在主干/分类尚未稳定时过早注入，DCLI seed差异由开启时刻不同放大。最小实验：在现有训练曲线上建立主干→P2/fusion→DCLI依赖图，按5-epoch loss-slope计算但不立刻改训练；检查坏seed是否更早/更陡开启。再做固定晚启与基于先修收敛的晚启两种短程对照。若坏seed与开启时刻无关，或延迟开启不能降低方差，否证。

**L. 原始页/表/路径。** [29] Fig.1–5、Eq.1–10、Sec.3 pp.3–7（GUP/UnC/HTL）；Table 1–4、Fig.6–7、Sec.4 pp.7–10（split/AP40/多类/消融/速度/坏例）；[30]作者代码；[31]official test身份。置信度High；参数/FLOPs和多seed unknown。

### K14. Ma et al. — MonoDLE (CVPR 2021)

**A. 元数据与官方链接。** Xinzhu Ma, Yinmin Zhang, Dan Xu, Dongzhan Zhou, Shuai Yi, Haojie Li, Wanli Ouyang, “Delving Into Localization Errors for Monocular 3D Object Detection,” CVPR 2021, pp.4721–4730。[原论文][32]；[官方代码][33]；[KITTI官方提交][34]。

**B. gap。** 论文先用oracle替换诊断单目3D误差，发现projected center+depth共同造成的3D localization是瓶颈；几像素中心误差随距离线性放大。方法不是单一模块，而是三个针对性措施：用projected 3D center监督coarse center、忽略极远训练样本、以对最终3D IoU敏感度重分配尺寸loss。

**C. 网络、损失、匹配、NMS。** DLA-34+anchor-free CenterNet，一阶段七个轻量head：heatmap/category、2D center offset/size、projected 3D center offset、depth、3D size、12-bin orientation。分类modified focal；2D/center L1；depth用aleatoric uncertainty loss；方向multi-bin；完整法把尺寸L1改成按1/h:1/w:1/l近似IoU导数重分配并做均值补偿。coarse heatmap由projected 3D center监督，但保留2D head作辅助任务。训练样本可在s=60m硬删除或用reverse-sigmoid软权重。推理以3×3 max-pool做CenterNet式局部NMS，confidence<0.2丢弃；无额外后处理。

**D. split。** 3712/3769做诊断/消融；全7481→official test。140 epochs，batch16、2×1080Ti，Adam lr1.25e-3，90/120 epoch降10倍，5 epoch warm-up；训练约9h。随机crop/scale仅用于2D，另用水平翻转。

**E. 口径。** 2D/3D/BEV/AOS；AP40；Car .7、Ped/Cyc .5；E/M/H。主消融是Car 3D/BEV；test另报Ped/Cyc 3D/BEV。项目2D Ped/Cyc AP_R40只在任务定义层相关。

**F. baseline、oracle、消融、delta、效率。** Error-analysis Table 1 Moderate AP3D：baseline11.12；GT projected center 23.90（正文叙述另一处写18.97，原文内部冲突，表值优先并标异常）；GT depth38.01；GT 3D location78.84；GT size11.96；GT orientation11.88，说明定位瓶颈但oracle不是可实现提升。累计Table 4（BEV/3D E/M/H）：baseline20.29/14.51、16.15/11.12、14.07/9.97；+projected center 23.10/15.78、18.15/12.65、16.11/10.62，Moderate 3D +1.53；+IoU 18.34/12.97，+0.32；+far-sample设计19.33/13.66，+0.69。Table 6训练样本：其局部baseline16.12/12.97/10.99；硬s=40降为14.25/11.25/9.63，硬s=60升17.45/13.66/11.68；软c=40,T1降14.50/11.74/9.95；软c=60,T1为17.50/13.54/11.32；T5为17.25/13.03/11.01，阈值/温度强非单调。official test Car AP3D17.23/12.26/10.29、BEV24.79/18.89/16.00、AOS93.46/90.23/80.11；40ms/GTX1080Ti=25FPS。参数/FLOPs未报。

**G. Ped/Cyc/small/far/遮挡。** Table 3 official test BEV/3D：Ped Moderate6.96/6.55，低于MonoPair7.04/6.68；Cyc3.28/2.66，高于MonoPair2.87/2.12，但作者警告少样本会波动。Table 2量化中心偏移：60m处(6,4)像素→0.61m误差，接近Car IoU.7约0.62m最大容忍；(8,6)→0.85m。训练集中>60m为1301/28742=4.5%，>65m为767=2.7%；删除它们提高总体AP却没有改善这些far样本，且牺牲尾部覆盖。没有Ped/Cyc框高/距离/遮挡分层。

**H. 失败、限制、泄漏、不可比。** (1)s=40/c=40明显退化，远样本策略对阈值敏感；(2)删远样本优化榜单平均而非远距安全；(3)Ped略低于对手、少样本波动；(4)oracle表/正文23.90 vs18.97内部不一致；(5)作者明确排除DORN类val比较，因为DORN训练集与KITTI object val重叠，这是直接泄漏证据；(6)单split/无seed；(7)累计设计同时改变监督、loss和数据分布；(8)Car 3D定位结论不能直接外推2D P2。

**I. 项目区别与因果关联。** plain/P2/DCLI/fusion是2D YOLO；MonoDLE有相机标定、3D head和距离标签。对P2最有价值的是“高分辨率仍受1–2 feature-pixel偏移控制”与s阈值非单调：P2增加细节若中心偏移/误匹配不降，反而增加难噪声。对DCLI不稳的关联是少量极难/远样本可主导梯度，但删除不是默认解，必须检查项目AP目标是否要求这些样本。

**J. 借鉴与边界。** 可借鉴oracle分解、框高/中心误差/距离三维诊断、硬/软阈值全曲线以及“总体AP与尾部召回同时报”。使用projected-center监督、距离样本权重或IoU-oriented size loss需引用。不得宣传“去掉远目标改善远目标检测”；它只改善总体Car 3D AP。

**K. 假设与最小实验。** K14-H：plain-P2 -1.5918主要来自25–40px远/难目标的中心匹配抖动和背景FP，而不是缺少分辨率。最小实验不训练：按框高、中心量化误差、遮挡统计P2相对P3–P5的TP→FP/FN迁移；做GT-center oracle重新匹配但不改分数。若oracle不能回收大部分损失，否证。若能，再做只对高中心不确定度样本降低P2训练权重的三seed实验，同时报告该层recall，禁止只报总体AP。

**L. 原始页/表/路径。** [32] Fig.1–4、Tables 1–2、Sec.3 pp.2–6（oracle、中心/距离误差、三策略、loss）；Tables 3–6、Sec.4 pp.6–9（split/AP40/test/Ped/Cyc/累计/远样本/速度）；[33]作者代码；[34]official test身份。置信度High；Table 1正文冲突已原样记录，不擅自消解。

### K15. Zhang et al. — MonoDETR (ICCV 2023；未替换为OccupancyM3D)

**A. 元数据与官方链接。** Renrui Zhang, Han Qiu, Tai Wang, Ziyu Guo, Ziteng Cui, Yu Qiao, Hongsheng Li, Peng Gao, “MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection,” ICCV 2023, pp.9155–9166。[原论文][35]；[补充][36]；[官方代码][37]；[KITTI官方提交][38]。指定论文明确在KITTI做Car val/test、补充报Ped/Cyc test且协议完整，故无需启用备选OccupancyM3D；替换会改变冻结同读清单并引入另一方法族。

**B. gap。** center-guided单目3D只从物体中心邻域取局部视觉特征，忽略全局场景几何和物体间深度关系。MonoDETR以稀疏object-wise depth标签生成foreground depth map，让query先跨全图聚合depth，再交互视觉语义；gap是非局部深度上下文与稳定set prediction，不是小目标专用网络。

**C. 模块、损失、匹配和后处理。** ResNet-50取1/8、1/16、1/32；多尺度对齐相加后两层3×3得到1/16 depth feature，预测80-bin、0–60m的LID foreground depth map，框内像素赋object depth、重叠取更近物体。1/32视觉特征经3层deformable visual encoder；depth经1层global self-attention；50 learnable queries在每个3层decoder按`Depth cross-attn→inter-query self-attn→visual cross-attn→FFN`。head预测class、2D size、projected 3D center、depth、3D size、orientation。Hungarian matching只用稳定的`L2D=class+center+2D box/GIoU`，不把早期不稳的`L3D=depth+size+orientation`放进cost；完整loss对匹配对累加L2D+L3D，再加depth-map focal。推理confidence<0.2，**无anchors、无NMS**。

**D. split。** 3712/3769；全7481→official test。195 epochs、batch16、单RTX3090、AdamW lr2e-4；为稳定删除depth<2m或>65m训练样本。该样本过滤必须视为方法的一部分而非普通数据清洗。

**E. 口径。** KITTI AP40，Car 3D/BEV IoU.7，补充Ped/Cyc 3D IoU.5，E/M/H。没有项目2D Ped/Cyc主指标的受控消融。

**F. baseline、消融、delta、效率。** Table 5 Car val AP3D：完整28.84/20.61/16.38；去整个depth-guided transformer19.69/15.15/13.93（Moderate -5.46）；仅depth predictor无transformer20.19/16.05/14.18；视觉transformer无depth guidance24.14/17.81/15.60，说明完整变化是depth+transformer交互，不能称单注意力delta。Table 6 depth encoder：Global SA20.61；Deformable18.91；2×3×3 conv18.36；无encoder18.38，局部卷积甚至略低于无encoder。Table 7 decoder顺序Moderate：D→I→V20.61，I→D→V19.28，I→V→D18.85，I→D+V18.41，融合顺序差2.20。Table 8 foreground LID20.61、dense LID19.85、foreground UD18.90、SID18.95；更多dense监督并非更好。补充Table 5 matching：只L2D为20.61；加入size3D为19.21、orientation18.63，加入depth或全部L3D直接**training collapse**。官方论文Table 2 test Car AP3D25.00/16.47/13.58；KITTI当前详情页[38]为24.52/16.26/13.93，存在提交/版本漂移。Table 3统一RTX3090：38ms、62.12 GFLOPs；参数未报。

**G. Ped/Cyc/small/far/遮挡。** 补充Table 3 official test AP3D：Ped12.54/7.89/6.65，Cyc7.33/4.18/2.92；作者认为少类缺少可学的尺寸先验，结果仍受样本量限制。1/16 depth feature利用多尺度输入但视觉只用1/32，论文没有框高/遮挡分层AP。LID扩大远距bin并删除>65m训练样本，不能宣称改善极远目标；attention图是定性机制证据。

**H. 失败、限制、泄漏、不可比。** (1)把3D loss放Hungarian cost会崩溃，是最强负证据；(2)decoder顺序/encoder类型/深度离散都非单调；(3)dense depth supervision低于object-wise，但监督/数据条件也不同；(4)删<2m/>65m掩盖尾部；(5)论文表与official detail分数漂移；(6)无seed，DETR在KITTI小数据上可能高方差；(7)Ped/Cyc只补充结果无逐组件；(8)Car 3D/BEV不能迁移为2D小目标提升；(9)论文限制明确未讨论multi-modal输入。

**I. 项目区别与因果关联。** plain P3–P5/P2是dense 2D prediction；MonoDETR是低分辨率query set prediction，且视觉只用1/32，反而反驳“必须加P2才有提升”。DCLI按退化条件，MonoDETR按depth交互；fusion最相似，但Table 7显示融合顺序可产生2.2 AP差，补充matching更说明不稳定信号不应过早决定正负样本。项目P2退化和DCLI不稳应先查matching/gradient路径，而非换成Transformer。

**J. 借鉴与边界。** 可借鉴：(i)把稳定2D项用于匹配，不让高方差辅助项改assignment；(ii)预注册融合顺序；(iii)foreground-only辅助监督；(iv)明确报告collapse。若用foreground depth、depth cross-attention或其matching设计必须引用。不得把Table 5的+5.46称为“depth模块单独提升”，因为baseline同时移除了Transformer；不得用Car 3D替项目2D AP背书。

**K. 假设与最小实验。** K15-H：DCLI/fusion跨seed不稳主要由退化/高分辨率辅助分支进入label assignment或置信排序太早，使少量query/anchor正负匹配翻转。最小实验：冻结同一预测，分别用baseline稳定cost与加入DCLI/P2高方差项的cost重放assignment，统计三seed每GT匹配Jaccard、Ped/Cyc翻转率和后续AP。若加入项不增加匹配翻转或翻转与负delta无关，否证；若相关，再做“辅助项只进loss、不进matching”的三seed最小训练。

**L. 原始页/表/路径。** [35] Fig.3–5、Eq.1–6、Sec.3 pp.4–8（depth map、双encoder、decoder、matching/loss）；Tables 2–9、Sec.4 pp.8–11（AP40、效率、消融、训练过滤）；[36] Tables 3–6（Ped/Cyc、matching collapse、depth预测/块数）；[37]作者代码；[38]official test版本。置信度High；论文/official分数漂移已并列，不选一方覆盖另一方。

## 4. 与主稿K01–K11的第一轮差异草案

> 顺序声明：本节是在本独立稿K01–K15、registry和protocol matrix全部完成后，才读取主稿K01–K11；因此以下是交换核验，不是用主稿回填独立判断。核验范围仅主稿现有K01–K11，不对尚未交换的K12–K15作一致性声明。

### 4.1 共同结论

1. K06均独立判定Song et al. TLL+TFA不满足“可复现KITTI检测实验”纳入门，并以Mao et al.替换；38.72不能当KITTI模块收益。
2. K10均确认AP11/AP40可改变消融方向，corner loss在AP40 Moderate为12.13→12.09（-0.04）；D4LCN 3D Car收益不能转写为项目2D Ped/Cyc收益。
3. K11均确认单独训练/解冻某一子网可负向，只有适当梯度量级和联合路径才得到完整收益；这只能提出项目fusion/DCLI梯度失衡假设，不能证明现有AB-B负值的根因。

### 4.2 前三项关键遗漏/差异与合并处理

| 优先级 | 位置 | 主稿相对独立稿新增的关键证据 | 独立稿原状态 | 最终合并建议 |
|---:|---|---|---|---|
| 1 | K08 Complexer-YOLO | 主稿记录硬过滤“平均约+1.3”同时令Car Moderate BEV **-4.8 AP**；这是总体正向掩盖关键类/难度伤害的直接no-harm反例 | 独立稿只记录输入/任务不可比和缺少公平去语义消融，漏掉该定量负例 | 把-4.8写入联合路线的强制no-harm依据：任何总体AP上升都必须同时通过Ped/Cyc、small/far、near/far分层门 |
| 2 | K10 D4LCN | 主稿核出摘要“Moderate相对提升9.1%”是相对AM3D 10.74的相对百分比，绝对仅 **+0.98 AP点** | 独立稿详细写同表+CL→full的+4.11和SP→full的+1.76，但未解释论文headline | 合并时同时列“跨方法headline +0.98绝对”和“内部系统消融+4.11”，严禁混为同一个组件收益 |
| 3 | K11 E2E Pseudo-LiDAR | 主稿补出Table 5：soft quantization本身未必优于hard，真正改善依赖联合训练；另给PointRCNN/PIXOR/SDN时延0.49/0.55/0.39s量级 | 独立稿记录freeze矩阵与梯度量级，但把统一总效率记unknown，且未突出soft-vs-hard的负/零证据 | 联合结论应改为“可微表示只是梯度通路，不是收益来源”；成本分别列组件时延，不拼成无证据的单一总FPS |

### 4.3 未发现的材料冲突

在K01–K11核心数值、任务口径和项目因果边界上，未发现需要二选一裁决的实质冲突；差异主要是主稿更完整的负例、许可证/运行成本和headline解释。本独立稿保留自己的原始卡，不用主稿数字倒改独立结论；上述三项只作为后续联合稿的增补清单。

## 5. 强制 counter-review（完成度核验版）

- 任何“KITTI已饱和，因此换COCO更容易提升”的结论都没有被这15篇支持：文献跨2D/BEV/3D、AP11/AP40、不同split和不同传感器，绝对AP高低不能代表可提升空间。
- 任何“加P2/多尺度/融合通常会+1–3 AP”的结论都被反例否定：RRC过多输出、Mao侧支路、D4LCN局部组件、MonoDLE错误阈值、MonoDETR matching均可为负或collapse。
- 任何把单split或单次消融当跨seed稳定性的结论都不成立；尤其Ped/Cyc样本少、split敏感和训练过滤会放大方差。
- 文献数字只用于形成可证伪实验，不用于修改项目冻结事实：plain-P2 -1.5918 AP；DCLI三seed约+0.3257；联合约+0.1710且CI跨0；AB相对B为负。

## 6. 完整性与可追溯性校验

- 技术卡：K01–K15共15篇，每篇均检出A–L十二字段；K06保留Song排除证据并完成Mao替换卡。
- Protocol matrix：恰好15行，逐行注明任务、输入、split/test、类别、IoU、AP规则、难度和直接可比性。
- Citation registry：38条Approved primary/official/official-code来源；另有[D1]排除记录。检索摘要未作为数值证据。
- Unknown纪律：未核出的NMS/匹配阈值、参数/FLOPs、seed、AP采样和统一总时延保持unknown；没有用发表年份替代协议证据。
- 口径纪律：2D/BEV/3D、Car/Ped/Cyc、IoU .7/.5、AP11/AP40、val/test、自定义mAP没有横向相减。
- 独立性纪律：独立稿完成后才读取主稿K01–K11；差异草案只列交换增补，不重写原独立判断。
