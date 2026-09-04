# 15篇KITTI论文：具体办法、同表提升与失败条件

- 核验日期：2026-08-12
- 数值原则：只写论文原表中可定位的绝对值与差值；“+x AP”均为绝对AP点，除非明确标注“相对百分比”。
- 禁止用途：本表不能用来预测或拼接本项目增益；2D/BEV/3D、AP11/AP40、validation/test不得横向相减。

| ID | 论文与具体办法 | 最可信的同表变化 | 负结果/条件 | 对本项目真正有用的学习 |
|---|---|---|---|---|
| K01 | KITTI协议：hidden test、IoU、难度、AOS/AP | 无模型提升 | AP11→AP40版本变化；E/M/H不是纯距离 | 先冻结协议与ID；371 development不能称confirmation |
| K02 | MS-CNN：共享多尺度feature maps，同时做proposal与detection；高分辨率输入、hard-negative bootstrapping | 2D val AP11：输入384→576时Car Moderate `80.63→88.14`（+7.51）、Ped `68.37→70.77`（+2.40）；在576上再用2×特征上采样为Car `88.14→89.12`（+0.98）、Ped `70.77→72.49`（+1.72）；576→768 Ped再+1.49 | Ped/Cyc模型时间约0.09→0.19→0.36s；随机负样本使Ped低4.86 AP | 高分辨率收益依赖负样本控制；先查P2噪声/FP，不能只加分辨率 |
| K03 | SubCNN：图像金字塔、subcategory filters、hard mining与姿态辅助 | 2D val AP11：feature extrapolation同表Car/Ped/Cyc Moderate约`+0.35/+1.08/+3.57`；作者proposal相对普通RPN为`+4.06/+0.63/+4.30`，但多因素 | 共享卷积使Car/Ped下降，结构差异更大时甚至不收敛 | “共享是否有效”依赖任务/类别；我们的fusion×DCLI冲突不是首次现象 |
| K04 | RRC：浅层/深层双向循环聚合，连续多轮监督，多回归器 | 2D Car val、历史AP11：SSD→RRC在IoU .7为`89.16→90.65`（+1.49），IoU .8为`76.12→82.82`（+6.70） | 第3轮后loss反升；合并过多输出不更好 | 高分辨率浅层需要语义补偿；严IoU改善可与宽IoU覆盖区分 |
| K05 | DeepParts：45个遮挡part ConvNet，学习互补part并集成 | Caltech miss rate 24.80→11.89；6个互补part 12.31优于“6个单独最强”15.28 | KITTI只做跨域Ped test；45网络成本高；最强组件相加不等于互补 | 选择互补错误模式，而不是挑单项最高分；不能把切片正数相加 |
| K06 | HyperLearner：检测主干联合edge/seg/heatmap等辅助语义，并用CFN融合 | 2D Pedestrian val、历史AP11：1× baseline69.80→edge71.25（+1.45）；2×71.73→72.51（+0.78） | optical flow在2×为-0.08；heatmap从1×+1.37缩到2×+0.18 | 更强基线会吞掉辅助收益；辅助分支可能压制主干，需逐路径归因 |
| K07 | WLOD：把分类代价改为类别间Wasserstein代价，强调危险类别关系 | YOLOv3 68.3→69.4（+1.1）；SSD72.8→74.7（+1.9） | 非官方随机三折mAP；无逐类AP、seed、CI、E/M/H | 损失替换可筛查，但不能把非官方mAP当KITTI AP_R40证据 |
| K08 | Complexer-YOLO：RGB语义投影到LiDAR点云，BEV 3D YOLO+跟踪 | validation semantic voxel相对RGB 3D mAP `28.64→34.14`（+5.50），多因素/3D；另一个SRTs替代IoU实验约+1.3 mAP | 与SRTs不同的硬过滤实验平均也约+1.3，却使Car Moderate BEV约-4.8；不同类别受损 | 所有系统增益都要逐类no-harm；平均正不能遮住关键类别负迁移 |
| K09 | M3D-RPN：global/local双路、depth-aware row-wise convolution、3D anchors与post optimization | Car 3D/BEV val、历史AP11：仅post 3D15.08；b=32+post17.06（+1.98），BEV17.47→21.18（+3.71） | b=1/4的3D分别12.87/14.46，低于post-only；Cyclist val约9–10而test约1 | 路径容量/深度bin不是单调；小类泛化需独立验证与分母 |
| K10 | D4LCN：预测深度引导动态局部卷积、shift pooling和一致性loss | split1 Car Moderate AP40：12.13→full16.20（累计+4.07）；论文“+9.1%”对AM3D其实绝对仅+0.98 | 普通+corner loss 12.13→12.09；DLCN13.50低于LCN13.93；深度源可把结果推到19.80 | 先区分模块、额外深度与组合收益；相对百分比不能写成AP点 |
| K11 | End-to-End Pseudo-LiDAR：可微深度→点云→3D检测联合训练；按梯度覆盖/量级重权 | Car 3D/BEV val（论文未声明该表AP11或AP40）：PointRCNN Moderate 3D`50.1→51.7`（+1.6）；PIXOR BEV@.5 `78.4→84.8`（+6.4，多因素） | 单独解冻检测阶段可负；soft quantization本身不优于hard，收益依赖联合训练 | 记录梯度support、norm sum与覆盖，不凭loss名推断协同；重权是公开先例 |
| K12 | MonoRUn：重建+Robust-KL不确定性，PnP与校准；pose detach限制校准梯度 | Table 3 Car六项AP_R40平均mAP：LapKL+E2E `29.47→29.73`（+0.26），不是单一Moderate AP | 更强Robust-KL+E2E `31.21→31.09`（-0.12）；full epistemic -0.05；Cyc Mod仅0.61 | 校准梯度隔离已有直接先例；更复杂不确定性可负，需校准与AP共同验证 |
| K13 | GUPNet：geometry uncertainty projection、uncertainty confidence与hierarchical task learning | Car 3D val AP40的跨方法同表比较：MonoPair→GUPNet在严格IoU .7 Moderate `12.30→16.46`（+4.16），宽IoU .5却`42.39→42.33`（-0.06）；不是GUPNet自身消融 | GeP加在UnC上仅+0.04且部分strata下降；GradNorm10.49远低于HTL16.46 | 延迟/层级调度是公开方案；先判定coverage还是localization，不能直接套headline |
| K14 | MonoDLE：GT替换定位瓶颈、projected center、dimension-aware IoU loss、距离采样 | Car 3D val AP_R40 Moderate累计`11.12→12.65→12.97→13.66`（总+2.54） | hard40m使12.97→11.25；默认60m删除far目标；Ped略低于MonoPair；论文数字有冲突 | 最有价值是误差替换与宽/严IoU诊断；删除far换总AP不符合本项目目标 |
| K15 | MonoDETR：foreground depth+DETR，depth-first decoder；仅稳定L2D参与Hungarian matching | Car 3D val AP_R40 Moderate：full20.61 vs去掉整个depth-guided transformer15.15（多因素+5.46）；去depth guidance17.81（+2.80） | matching加size降1.40、加orientation降1.98、加depth/完整L3D直接collapse；2个3×3 conv略低于无encoder | 不稳定定位目标不能控制共享assignment；语义所有权隔离是强先例，不是原创空白 |

## 跨论文可重复规律

1. **更高分辨率本身不保证提升**：MS-CNN需要hard-negative控制；RRC需要深层语义回流；plain-P2在本项目反而总体`-1.5918 AP`。
2. **单项有效不代表组合有效**：SubCNN、MonoRUn、GUPNet、MonoDETR都给出直接负交互；本项目三seed `AB-B`也全部为负。
3. **“提升”常集中于定位而非召回**：RRC、GUPNet、MonoDLE在严格IoU上收益更强；必须分开coverage/localization/ranking。
4. **小类别/远目标极易产生假稳定**：Cyclist在M3D-RPN从val约9–10掉到test约1；MonoDETR仓库自报约±1 AP波动；本项目Cyclist Moderate分母小，必须给CI。
5. **删难样本可抬总体AP但不解决难样本**：MonoDLE和MonoDETR都过滤远距离训练目标；本项目不得用这种方式包装far改善。
6. **已公开的不能声称原创**：普通P2/多尺度融合、动态卷积、uncertainty loss、延迟启用、stop-gradient、稳定assignment、crop/zoom都已有直接先例。
