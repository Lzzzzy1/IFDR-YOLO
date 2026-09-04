# 15篇KITTI论文评测协议矩阵

- 核验日期：2026-08-12
- 用途：防止把2D、BEV、3D、AP11、AP40、内部validation与官方hidden test混在一起比较。
- 本项目事实边界：当前主结果是3341 fit / 371 development、RGB 2D、Pedestrian/Cyclist Moderate macro、内部AP_R40；371已参与研发，不能称独立confirmation。

| ID | 论文 | 任务/输入 | 数据划分 | 类别与IoU | AP/evaluator | 主要训练与成本 | 能否与本项目直接相减 |
|---|---|---|---|---|---|---|---|
| K01 | KITTI benchmark protocol | 2D detection/AOS；后扩展BEV/3D；相机/激光多传感器数据 | 7,481 labeled train；7,518 hidden test；论文强调序列隔离；3712/3769不是官方split | 现行2D常用Car .7、Ped/Cyc .5；E/M/H由框高、遮挡、截断定义 | 2019-10-08前主要AP11，之后AP40；官方test隐藏标签 | 数据/评测协议，不是模型 | 否；它定义口径 |
| K02 | MS-CNN | 单目RGB 2D，多尺度feature/图像金字塔 | 7,481内按3DOP划分做val；全train提交test | Car/Ped/Cyc；.7/.5/.5 | 历史AP11，official test | Caffe多尺度网络；无seed/CI | 任务接近，但AP版本、split、架构不同 |
| K03 | SubCNN | 单目RGB 2D proposal+detection+AOS；额外姿态/子类监督 | 自定义3,682/3,799；全train提交test | Car/Ped/Cyc；.7/.5/.5 | 历史AP11 | 图像金字塔、数百子类；成本不完整 | 否；两阶段、额外监督、AP11 |
| K04 | RRC | 单目RGB 2D；SSD式多尺度循环聚合 | 图像相似度隔离的自定义split，val 2,741；全train提交test | Car/Ped/Cyc；.7/.5/.5 | 历史AP11 | 1272×375；约120k iterations/4 Titan X；无CI | 否；split/AP和anchor体系不同 |
| K05 | DeepParts | RGB 2D pedestrian parts | Caltech训练，直接迁移到KITTI official test | Pedestrian，IoU .5 | 历史AP11，现行详情页可能漂移 | LDCF proposals+多个part ConvNet | 否；跨数据集且仅Pedestrian |
| K06 | Mao/HyperLearner | RGB 2D pedestrian；边缘/分割等训练期辅助监督 | 7,481内按旧文献拆train/val，未重列精确ID；无official test | Pedestrian，IoU .5 | 历史AP11 | Faster R-CNN式多任务；单split | 否；只能借鉴负交互与辅助监督诊断 |
| K07 | Wasserstein Loss-Based Detection | RGB 2D；YOLOv3/SSD损失替换 | 7,481随机三折约2,495/2,493/2,493 | 7细类合并3风险组；IoU细节未充分报告 | 非官方mAP；非KITTI AP11/AP40榜单 | 自定义loss；无官方devkit对账/CI | 否；仅损失思想可作公开基线 |
| K08 | Complexer-YOLO | RGB语义+LiDAR BEV，3D/BEV/AOS与跟踪 | official KITTI test；消融split未清楚报告 | Car/Ped/Cyc；.7/.5/.5 | 2019历史AP11；官方详情后来有AP40漂移 | 多模态3D系统 | 否；输入和任务均不同 |
| K09 | M3D-RPN | 单目RGB，2D/BEV/3D；depth-aware convolution | val1 3712/3769、val2 3682/3799、official test | Car/Ped/Cyc；.7/.5/.5 | 2019历史AP11；现行官方详情另有AP40 | 3D anchors+几何卷积；单seed | 核心3D不可比；2D附表也非同协议 |
| K10 | D4LCN | RGB+预测深度，单目2D/BEV/3D | 3712/3769、3682/3799、official test | Car/Ped/Cyc；.7/.5/.5 | 文中混列AP11/AP40，必须逐表辨认 | depth-guided dynamic conv；无CI | 否；深度输入/3D监督/协议不同 |
| K11 | End-to-End Pseudo-LiDAR | 双目RGB→深度→点云→BEV/3D；LiDAR监督深度 | KITTI内部train/val与official test；序列identity未给 | 主要Car；IoU .5/.7 | 表只写AP，未明确每表R11/R40 | 双目+PointRCNN/PIXOR，0.49/0.55s级；无CI | 否；只可借鉴梯度覆盖与量级平衡 |
| K12 | MonoRUn | 单目RGB 3D；可选稀疏LiDAR NOC监督 | 3712/3769；全train提交official test | Car/Ped/Cyc；.7/.5/.5 | 正文明示AP_R40 | 50 epochs，2×2080Ti，推理约0.070s | 否；不确定性/校准隔离只作机制先例 |
| K13 | GUPNet | 单目RGB+标定，BEV/3D | 3712/3769；全train提交official test | Car/Ped/Cyc；.7/.5/.5 | AP40 | 380×1280，140 epochs，3×Titan XP，29.4 FPS | 否；可借鉴层级调度与宽/严IoU诊断 |
| K14 | MonoDLE | 单目RGB+标定，2D/AOS/BEV/3D | 3712/3769；全train提交official test | Car/Ped/Cyc；.7/.5/.5 | AP40 | 140 epochs，2×1080Ti，约9h，40ms | 否；主消融是Car 3D，且含far样本删除 |
| K15 | MonoDETR | 单目RGB+标定，DETR式BEV/3D | 3712/3769；全train提交official test | Car；补充Ped/Cyc；.7/.5 | AP40 | 195 epochs，RTX3090，38ms，62.12 GFLOPs | 否；matching稳定性可借鉴，3D AP不能搬运 |

## 强制解释规则

1. 只有同一图像清单、同一类别、同一difficulty、同一IoU、同一AP版本、同一evaluator与同一权重角色的结果才允许相减。
2. 论文中的“%”常实际表示绝对AP点；只有原文明确relative improvement时才写相对百分比。
3. Easy/Moderate/Hard不是纯距离分层；本项目small/far必须作为另一个预注册切片单独报告。
4. 官方hidden test与内部development不是同一种证据。当前371只支持研发判断，不能证明最终泛化。
5. 3D/BEV AP数值低于本项目2D AP并不代表模型更差，而是任务空间与IoU要求更难。
