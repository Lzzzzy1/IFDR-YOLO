# KITTI目标检测15篇原始论文深读（YOLO agent独立稿）

- 检索与核验日期：2026-08-11
- 状态：逐篇增量保存；先独立判断，后与第二独立稿交换合并。
- 来源门：原始论文全文、CVF/出版社页面、作者官方代码与KITTI官方页面。博客、论文解说和排行榜转载不作为数值证据。
- 项目边界：本项目现有P3–P5、plain-P2、DCLI、fusion结果不因文献而改变；不同KITTI任务、split、AP11/AP40、IoU阈值、2D/BEV/3D数字禁止相减。
- 目标：学习可复现的方法与评测设计，不复制论文表述、结构图或未许可代码；任何公共思想必须引用并独立实现或遵守许可证。

## 统一清单与状态

| ID | 论文 | 任务 | 深读状态 |
|---|---|---|---|
| K01 | Geiger et al., KITTI benchmark / Vision Meets Robotics | 数据集与协议 | 完成 |
| K02 | Cai et al., MS-CNN | 2D多尺度检测 | 完成 |
| K03 | Xiang et al., SubCNN | 2D proposal/姿态/遮挡 | 完成 |
| K04 | Ren et al., RRC | 2D单阶段多尺度上下文 | 完成 |
| K05 | Tian et al., DeepParts | 行人部件/遮挡 | 完成 |
| K06 | Mao et al., What Can Help Pedestrian Detection?（替换Song TLL） | 辅助语义/边缘与错误分解 | 完成 |
| K07 | Han et al., WLOD | 分类代价/损失 | 完成 |
| K08 | Simon et al., Complexer-YOLO | LiDAR+语义3D YOLO/跟踪 | 完成 |
| K09 | Brazil & Liu, M3D-RPN | 单目3D/几何/深度感知卷积 | 完成 |
| K10 | Ding et al., D4LCN | 深度引导动态卷积 | 完成 |
| K11 | Qian et al., End-to-End Pseudo-LiDAR | 立体深度到3D检测联合训练 | 完成 |
| K12 | Chen et al., MonoRUn | 重建与不确定性传播 | 完成 |
| K13 | Lu et al., GUPNet | 几何投影不确定性/分层任务学习 | 完成 |
| K14 | Ma et al., MonoDLE | 3D定位错误诊断 | 完成 |
| K15 | Zhang et al., MonoDETR | 深度引导Transformer | 完成 |

> 元数据纠正：冻结清单把GUPNet写成“Simonelli et al.”，原论文作者实际为Yan Lu等；本稿按原论文更正并保留此差异记录。

## K01 — KITTI benchmark protocol / Vision Meets Robotics

- **A 元数据**：Geiger, Lenz, Stiller, Urtasun；KITTI Vision Benchmark（CVPR 2012）与数据集扩展（IJRR 2013）。官方入口：https://www.cvlibs.net/datasets/kitti/ 。CVPR DOI `10.1109/CVPR.2012.6248074`。数据许可证为CC BY-NC-SA 3.0，商业复用受限。
- **B 研究问题/gap**：当时自动驾驶视觉研究缺少同时具有标定双目图像、Velodyne、GPS/IMU、真实城市道路、多任务统一基准的数据。KITTI的贡献是数据与隐藏测试评测，不是一个检测网络。
- **C 数据流程**：同步多传感器采集并标定；object benchmark提供彩色/灰度双目图、点云、相机标定及训练标签。测试标签隐藏，提交预测由服务器计算；这使官方test与本地development本质不同。
- **D split**：object detection官方为7,481张带标签training与7,518张无公开标签test。论文后续常把7,481再切成约3,712/3,769的Chen split，但那不是KITTI官方独立test；必须记录具体文件清单和序列去重。
- **E 任务/指标**：2D detection、BEV和3D任务彼此不同。Car通常IoU 0.7，Pedestrian/Cyclist通常IoU 0.5。Easy/Moderate/Hard由框高、遮挡和截断筛选；AP历史上曾用11点采样，后来主流/官方更新为40个recall positions。AP11与AP40不得混用。
- **F 数值/效率**：数据集论文不提供可供我们认领的“模型提升”。排行榜数字还会随评测版本变化，引用时必须绑定提交日期、任务、类别、难度和AP版本。
- **G Ped/Cyc与远目标**：Moderate/Hard不是简单“距离标签”，而由可见框高、occlusion、truncation决定；把框高或z距离定义为small/far属于项目自定义分层，不能冒充KITTI官方难度。
- **H 风险**：KITTI规模小、固定城市/相机/天气；训练帧可能来自相邻序列。反复在公开training内的同一val上选模型会产生开发集过拟合。当前本项目371张已经用于路线选择，只能称development。
- **I 与本项目区别**：我们当前96.7687是371 development、两类Moderate、内部AP_R40，不是官方7,518 test结果。官方devkit尚未对账；因此不能与排行榜或论文test AP直接比较。
- **J 可借鉴边界**：可借鉴官方提交格式、难度筛选、DontCare处理和隐藏测试思想；数据和devkit须按许可引用，不能把内部实现称为官方等价。
- **K 可证伪假设/最小实验**：H0为内部evaluator与官方devkit在同一预测上的GT计数、PR采样和AP不一致；最小实验是对固定预测运行两套评测，逐类逐难度对账。任一差异未解释前，内部AP只能用于研发。
- **L 原始位置**：KITTI官网Copyright/Citation/Changelog；Geiger et al. CVPR 2012；IJRR 2013。官方页面明确训练/test规模、引用与许可；后续AP版本必须以object benchmark/devkit说明为准。

## K03 — Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection

- **A 元数据与来源**：Yu Xiang, Wongun Choi, Yuanqing Lin, Silvio Savarese；WACV 2017，pp. 924–933；DOI `10.1109/WACV.2017.108`；arXiv `1604.04693`。作者原文：https://yuxng.github.io/Papers/2017/xiang_wacv17.pdf 。本次未核验到作者发布的对应实现/许可证，因此不复制代码。
- **B 研究问题/gap**：Faster R-CNN式proposal面对KITTI显著尺度变化、遮挡和截断时，通用objectness成为瓶颈，而且只给2D类别，不能同时解释姿态、遮挡与3D位置。论文用“子类别”把外观/姿态结构带入proposal和检测。
- **C 流程、训练、匹配与后处理**：输入先形成多尺度图像金字塔；卷积后用双线性feature extrapolation补出未显式计算的中间尺度，降低密集图像金字塔成本。每个subcategory有专用卷积filter，RoI层进行hard positive/negative mining，随后检测网同时做subcategory softmax与bbox regression。Car子类别来自3D voxel pattern，Ped/Cyc按朝向各分24类；最终输出类别、框与姿态。proposal训练示例以IoU>0.5为正/低于阈值为负，KITTI检测RoI以Car 0.7、其余0.5为正；测试仍依赖候选框排序和检测后处理，并非YOLO式一阶段头。
- **D split与训练配置**：官方KITTI为7,481 train/7,518 hidden test；内部分析实际采用3,682 train/3,799 val，而非常见3712/3769，更不是我们的3341/371。内部用173个子类别（Car 125，Ped/Cyc各24），官方test全量训练用275个。proposal输入尺度0.25/0.5/1/2/3并插值成21级特征；检测输入1/2/3/4再插值。Caffe，ImageNet初始化，比较AlexNet/VGG16/GoogleNet；未报告seed或CI。
- **E 任务/指标**：KITTI 2D detection AP与orientation AOS，另对Car报告ASA/ALP；Car IoU 0.7，Ped/Cyc 0.5，Easy/Moderate/Hard。论文发表于AP40更新前，内部与test表均属历史AP11时代，不能与本项目内部AP_R40相减。
- **F 同表结果与效率**：内部val Table 2在相同检测网下，普通RPN→作者RPN的Moderate AP：Car 82.58→86.64（+4.06），Ped 69.32→69.95（+0.63），Cyc 54.83→59.13（+4.30）；这些变化同时包含多尺度、子类别与hard mining。Table 3中feature extrapolation的同模型比较`w/o Extra`→`Full`：Car 86.29→86.64（+0.35）、Ped 68.87→69.95（+1.08）、Cyc 55.56→59.13（+3.57）。官方test Table 5 GoogleNet版Ped Moderate 71.33、Cyc 71.06；VGG16为66.13/61.98。论文未给参数、FLOPs、完整端到端FPS，故记为unknown，不能借旧SubCat的0.7秒代替。
- **G Ped/Cyc/尺度证据**：Table 1在2k proposal下，作者方法对Moderate的recall为Ped 93.46、Cyc 93.03；普通RPN为91.69/91.80。更关键的是作者明确指出Cyclist样本少：共享卷积对Car/Ped有害、对Cyclist反而有利，说明“共享是否好”受数据量和任务表征需求影响，而不是统一规律。
- **H 失败、限制与不可比风险**：图像金字塔最高放大4倍、21级特征及数百子类别成本高，论文却缺参数/FLOPs/FPS。子类别依赖额外3D/姿态元数据。最重要的负结果是：RPN与检测网共享卷积使Car Moderate 82.58→82.21、Ped 69.32→68.28，却使Cyc 54.83→56.69；当作者尝试让自己结构差异更大的RPN与检测网共享时，训练甚至不能收敛（第5页）。因此“共享语义必然协同”被论文直接否定，但该现象没有多seed或梯度因果诊断。
- **I 与本项目的对应关系**：多尺度金字塔、尺度插值、hard-example mining、子类别/姿态辅助任务都有直接先例，不能冒充原创。它与我们最接近的不是P2本身，而是**共享特征的负交互**：本项目fusion与DCLI共同作用于共享可靠性/语义路径，AB相对B三种子均为负；SubCNN已证明一般性的“两个任务共享会退化”不是新发现。我们仍可能贡献的只是：在反事实退化监督驱动的融合与定位校准中，定位具体语义所有权冲突，并以预注册干预同时改善冲突指标和AP。
- **J 可借鉴与边界**：可借鉴按数据充足度决定共享/分离、hard-example分母控制、在同检测网下替换proposal做因果对照。必须引用SubCNN；不能复制其图像金字塔、subcategory filter或声称“任务专用适配器/共享冲突”概念首创。没有确认许可证前不复用代码。
- **K 可证伪假设/最小实验**：H0：P2层fusion与DCLI的共享节点不是AB-B退化来源；H1：只在P2的共享可靠性张量上做语义所有权分离、保持P3–P5和所有loss权重不变，可使AB-B由负转为不低于0，同时降低该节点预注册冲突频率。最小实验先用现有S1/S2/S3开关做互斥30轮screen；只有“冲突指标下降且AB-B/AP同时恢复”才支持机制，单独AP或cosine改善均不够。若三种开关都无恢复，停止“共享冲突”故事，转查assignment/覆盖。
- **L 页表定位**：网络流程与feature extrapolation见第1–3页Fig. 1–2；KITTI split、尺度和IoU见第4–5页；proposal recall见第5页Table 1；共享/非共享和相同检测网对照见第5页Table 2；完整检测网消融见第6页Table 3；官方test见第6页Table 5。共享导致退化及不收敛原文在第5页Table 2后分析段。

## K04 — Accurate Single Stage Detector Using Recurrent Rolling Convolution（RRC）

- **A 元数据与来源**：Jimmy Ren等；CVPR 2017，pp. 5420–5428；arXiv `1704.05776`。CVF入口：https://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Accurate_Single_Stage_CVPR_2017_paper.html ，原文：https://arxiv.org/pdf/1704.05776 ，官方代码：https://github.com/xiaohaoChen/rrc_detection 。仓库LICENSE为继承Caffe/SSD的BSD式条款并要求保留版权与免责声明；任何复用仍需区分上游与作者新增代码并归属。
- **B 研究问题/gap**：SSD一类单阶段检测器虽快，但浅层高分辨率特征缺高层抽象和上下文，在高IoU评价下小目标、遮挡和重叠目标定位差。论文提出的问题非常接近我们plain-P2退化：高分辨率层有细节，却未必“语义足够深”。
- **C 网络、损失、匹配与后处理**：以reduced VGG16+SSD多尺度层为基线，统一各层到256通道；每轮对相邻尺度做双向rolling：深层经1×1卷积+反卷积下传，浅层经1×1卷积+池化上传，拼接后降回原通道。相同聚合权重跨5轮共享，每一轮都接分类/回归监督，形成6次预测。分类为cross-entropy，框为Smooth L1；因上下文使回归空间更复杂，每层再离散为5个回归器。最终不是取最后一轮，而对第3–5次预测做NMS。
- **D split与训练**：KITTI 7,481 trainval/7,518 hidden test，无额外数据；作者用图像相似度构造2,741张val，试图降低相邻帧相似性，但未发布等同我们清单的identity。输入1272×375；SGD momentum 0.9、weight decay 0.0005、初始LR 0.0005、每40k iteration降10倍；官方仓库说明约60k iteration、4×Titan X。未报告seed/CI。
- **E 任务/指标**：KITTI 2D detection；Car/Ped/Cyc；Car IoU 0.7，Ped/Cyc 0.5；Easy/Moderate/Hard。2017表格属于历史AP11时代。内部Table 2只对Car、跨IoU 0.6–0.8；作者明确因Ped数据稀少难以可靠拆分，未做Ped内部消融。
- **F 绝对结果与消融**：同一reduced-VGG16内部val Table 2：SSD→RRC的Car AP在IoU 0.7为89.16→90.65（+1.49），IoU 0.8为76.12→82.82（+6.70）；这说明收益主要是框质量而非普通IoU下的分类。Table 1显示迭代并非越多越好：val loss从输出1的1.461降到输出3的1.357，随后输出4/5/6为1.361/1.366/1.375，后期反而退化。官方test Moderate：Car single 89.85、ensemble 90.19；Ped 75.33；Cyc 76.47。论文未给端到端FPS、参数或FLOPs，故效率只能依据结构/硬件描述，不能量化为项目收益。
- **G 小目标/Ped/Cyc证据**：论文Fig. 1把低IoU失败定位到小目标和遮挡；理论分析指出浅层分辨率高但抽象浅、上下文弱。Ped/Cyc只提供官方test点值，没有内部消融、分层或置信区间；作者明确承认Ped训练/验证拆分困难。因此其Ped/Cyc排名不能证明该机制对小类稳定有效。
- **H 失败与不可比风险**：论文自身最有价值的负结果是循环到后期会退化，作者归因缺少有效memory，并用开发结果挑第3–5次输出做NMS；这属于开发集驱动的输出选择。复杂聚合会让框回归更难，故又增加5个专用regressor。官方test、AP11、Car主消融、不同split和4GPU训练均不与我们的两类AP_R40同协议。把其IoU 0.8的+6.70转述成“整体+6.7 AP”是错误的。
- **I 与本项目的对应**：plain P2直接暴露浅层细节但语义不足的问题已有RRC明确分析；双向多尺度融合、反卷积、跨层上下文、迭代监督、回归器分工也都是直接先例，不能声称原创。RRC没有DCLI/factor supervision、可靠性门控或我们三种子AB-B检验。它提示当前P2的-1.5918可能源于“细节增加而语义/回归条件没对齐”，但这只是文献支持的候选解释，不是本项目实证结论。
- **J 可借鉴与边界**：可借鉴“浅层细节与语义深度分开诊断”“每次聚合必须有独立可观察进展”“过度迭代会伤害”“定位质量用更高IoU切片验证”。不得再把GFPN/双向融合/多次细化命名成原创P2；不应照搬其5轮RRC这种大模块。官方代码有BSD式上游许可，但若实际复用必须保留版权、列明来源和修改。
- **K 可证伪假设/最小实验**：H0：plain-P2退化不是浅层语义不足，而是assignment/样本分母；H1：冻结P3–P5，只给P2做一次、零新增检测分支的深层语义残差对齐，可在不提高P2候选数的前提下降低P2定位误差并改善macro AP。最小判别应先离线量化P2 vs P3匹配框的分类margin、IoU误差和FP来源；只有“P2独有TP存在但定位/语义分数显著差”才GO一次残差对齐screen，否则转向assignment/覆盖路线。该实验避免先堆RRC式架构。
- **L 页表定位**：失败分析与形式化条件见第2–3页Eq. 1–4；RRC结构见第3–4页Fig. 2–3/Eq. 5；损失、回归空间离散和KITTI设置见第4–5页；迭代退化见第5–6页Table 1；SSD同骨干对照见第6页Table 2；官方test见第6页Tables 3–6。代码训练说明与60k iteration/4 Titan X见官方README。

## K05 — Deep Learning Strong Parts for Pedestrian Detection（DeepParts）

- **A 元数据与来源**：Yonglong Tian, Ping Luo, Xiaogang Wang, Xiaoou Tang；ICCV 2015，pp. 1904–1912。CVF入口：https://openaccess.thecvf.com/content_iccv_2015/html/Tian_Deep_Learning_Strong_ICCV_2015_paper.html ，原文：https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tian_Deep_Learning_Strong_ICCV_2015_paper.pdf 。论文未给官方代码入口，本次未核验许可证。
- **B 研究问题/gap**：整人ConvNet在复杂遮挡下失效；手工指定少数遮挡模式又不能覆盖不同场景。论文要从只含行人框的弱标注中学习大量候选人体部件，再数据驱动选择互补部件，不需要人工part标签。
- **C 方法流程、训练与融合**：将人体划为`2m×m`网格，枚举45个矩形part prototype；每个part独立微调一个ConvNet（AlexNet/Clarifai/GoogLeNet），只把GT可见区域完整覆盖该part的样本作正例。测试时对proposal内邻近位置打分并用位移惩罚取最大值，处理proposal与GT错位。45个part分数再用线性SVM学习互补权重，部署时保留权重最高的6个。它依赖外部LDCF proposal并以部件得分重排/级联，不是端到端YOLO、anchor assignment或DCLI。
- **D 数据与split**：核心训练和全部受控消融在Caltech：set00–05训练、set06–10测试，并用每帧/每3、5、10、30帧采样研究数据量。KITTI实验完全不使用KITTI训练集，所有组件均在Caltech训练后直接提交/测试，因此它是跨数据集泛化证据，不是KITTI同协议训练消融，更不对应我们的3341/371。
- **E 任务/指标**：Caltech主指标为9点log-average miss rate（FPPI 10^-2到10^0），Reasonable/Partial/Heavy Occlusion；KITTI仅Pedestrian 2D Easy/Moderate/Hard AP，2015历史AP11时代、IoU 0.5。Caltech miss rate越低越好，KITTI AP越高越好，两者不能相加或互换。
- **F 绝对值、消融与成本**：Caltech Table 4从LDCF 24.80% miss rate开始，依次到AlexNet 21.19、GoogLeNet 17.52、object-level预训练16.43、45-part ensemble13.12、加shifting handling 11.89；这些是miss-rate绝对点下降。6个已学习互补part为12.31，而“挑6个单独最强part”反而15.28，较12.31差2.97点；6个full-body模型ensemble为15.50，说明数量相加不是互补。KITTI跨域结果Easy/Moderate/Hard为70.49/58.67/52.78；Regionlets Moderate 61.15因同时用Cyclist监督，作者自己说明高2.48。参数/FLOPs/FPS未报告；45个独立ConvNet成本很高，作者把模型压缩列为未来工作。
- **G Ped/small/遮挡证据**：Caltech对可见比例和高度有显式分层，DeepParts在Partial/Heavy Occlusion上相对当时方法有较大miss-rate改善；KITTI Moderate要求行人不低于25像素，但没有far距离或Cyclist独立结果。该论文说明可见部件互补能救遮挡，不说明普通远距离纹理噪声可由part模型解决。
- **H 失败与不可比风险**：45个模型、跨数据集、无seed/CI、Caltech研发后再看KITTI；KITTI结果无内部同协议ablation。最关键的反例是“6个单独最强模块直接合并”比学习到的互补组合更差，证明单模块点值不能相加；但其互补选择本身也在开发数据上学习。论文的8.54点“超过R-CNN”是跨方法系统比较，不是part模块的同协议增益。
- **I 与本项目对应**：部件模型、遮挡处理、位移搜索、线性收益选择和“互补而非堆叠”均有明确先例。它与我们A/B/AB最相关的证据是：单独表现好的分支直接组合会退化，只有真正互补的选择才有意义；因此我们不能把历史fusion-only与DCLI-only数字相加，也不能因AB-C为正就保留fusion。DeepParts没有P2、DCLI或梯度诊断，不能直接证明我们的AB负交互根因。
- **J 可借鉴与边界**：可借鉴“按样本互补性而非单模块平均收益选分支”“专门记录proposal错位”“遮挡分层”。必须引用DeepParts；部件池、位移惩罚、top-6 SVM选择都不能改名冒充原创。无官方代码许可证证据，不复用实现。
- **K 可证伪假设/最小实验**：H0：fusion与DCLI在371张development上救回的是互补样本；H1：二者的TP rescue集合显著重叠而harm集合不同，导致AB没有可加收益。最小实验直接复用冻结C/A/B/AB预测做逐目标rescue/harm/Jaccard和AP配对bootstrap，不训练新模块；若rescue高度重叠且AB新增FP集中于fusion独有样本，则只保留B并停止联合路线。若互补但AB仍退化，才允许查共享梯度/语义所有权。这一诊断已与项目冻结estimand一致。
- **L 页表定位**：part pool/训练见第2–3页Fig. 3–4；位移处理见第3–4页Eq. 3–5；数据量和位移消融见第5页Tables 2–3；互补组合与“best parts反而更差”见第6–7页Table 4；KITTI跨域结果与“不用KITTI训练”见第7页Sec. 4.5/Fig. 12。

## K06排除记录 — Song et al., TLL + Temporal Feature Aggregation

冻结清单原K06经原文全文核验后不满足“在KITTI做可复现目标检测实验”的纳入条件。论文主训练、Tables 1–2和所有消融均在Caltech/CityPersons；KITTI只在Fig. 7给定性图和一个Pedestrian Moderate AP 38.72，却没有train/val ID、是否official test、IoU、AP11/AP40、Cyclist或KITTI消融。原文：https://arxiv.org/pdf/1807.01438 。因此保留为“标注表示/小尺度边界噪声”的旁证，不把38.72纳入KITTI比较矩阵；按冻结规则显式替换为下文真正有KITTI受控实验的Mao et al.。

## K06 — What Can Help Pedestrian Detection?（Mao et al.，显式替换）

- **A 元数据与来源**：Jiayuan Mao, Tete Xiao, Yuning Jiang, Zhimin Cao；CVPR 2017。CVF页面：https://openaccess.thecvf.com/content_cvpr_2017/html/Mao_What_Can_Help_CVPR_2017_paper.html ，原文：https://arxiv.org/pdf/1705.02757 。本次未在原文/官方页核验到作者官方代码和许可证，故不复制实现。
- **B 研究问题/gap**：低分辨率行人与交通牌/立柱等hard negative外观相似，需要高层语义；高分辨率拥挤行人边界又被深层卷积模糊，需要低层edge。论文不是问“哪个模块万能”，而是系统测量不同辅助channel在不同输入分辨率下解决哪类错误。
- **C 网络、目标与训练**：共同基线是VGG16 Faster R-CNN，增加anchor到5尺度×7比例并移除conv5以保持高分辨率。第一组实验把ICF、edge、segmentation、heatmap、optical flow或stereo disparity经浅侧支路拼到conv4_3。HyperLearner则汇聚conv1_2/2_2/3_3/4_3到统一尺度，CFN预测辅助channel；RPN/FRCNN共享该聚合特征。总损失为CFN像素loss+RPN分类/回归+FRCNN分类/回归，各权重1；采用CFN→RPN→FRCNN→联合微调四阶段，且FRCNN阶段不穿过proposal坐标反传。
- **D split与额外监督**：KITTI 7,481 trainval/7,518 hidden test，按引用方案拆成train/validation；论文未给逐图清单，因此与我们的3341/371不同。报告1×和2×输入。Edge来自HED，segmentation网络在MS-COCO训练；optical flow用相邻帧，disparity用stereo/DispNet，故不同channel包含不同额外数据与传感器成本。没有seed/CI。
- **E 任务/指标**：KITTI 2D Pedestrian AP，Easy/Moderate/Hard，历史AP11时代；文中将Cyclist作为Pedestrian检测的FP错误类型，但没有报告Cyclist AP。另报告70% precision下按高度的recall和FP来源；这两个诊断数不能加到AP。
- **F 同表消融、绝对值与成本**：Table 3共同baseline Moderate：1× 68.96，加入edge 69.49（+0.53）、segmentation 70.80（+1.84）、heatmap 70.33（+1.37）；2× baseline 71.21，edge 72.34（+1.13）、segmentation72.54（+1.33），heatmap仅71.39（+0.18），optical flow 71.13（-0.08）。Table 4 HyperLearner共同baseline为1×69.80/2×71.73；edge监督71.25/72.51（+1.45/+0.78），segmentation监督71.15/72.35（+1.35/+0.62）；无CFN的HyperNet为69.72/72.23。KITTI未给速度；Cityscapes同结构Table 5给baseline 130/240ms、joint segmentation 140/250ms，不能当KITTI延迟，只能说明CFN推理开销约10ms于该硬件设置。
- **G small/Ped/Cyc与错误分层**：Table 2在70% precision下，height≤80px recall由21.3%升到35.6%，全尺度70.0→74.0；这是recall点而非AP。2×误差分析把FP分为localization/background/Cyclist/annotation；edge相对baseline使localization error比例明显下降，而纯heatmap缺低层边界，在高分辨率几乎无净AP。论文支持“低分辨率更需语义，高分辨率更需边界”，不支持统一辅助分支。
- **H 失败、偏差与不可比风险**：作者发现预训练侧支路会因激活值过大而支配VGG主流，造成性能退化，最终改用随机初始化；这是与我们fusion/DCLI范数支配非常接近的直接先例。不同channel使用不同外部数据，1×/2×收益明显缩水甚至翻负；只单seed val、旧AP、仅Pedestrian。其FP分类由约800张抽样/Top-200构成，不是全量配对CI。
- **I 与本项目对应**：多层汇聚、edge/seg辅助监督、四阶段训练、侧路与主路平衡均为直接先例；不能称为原创P2或原创语义保护。对我们最关键的是两条可检验关系：高分辨率P2若主要是localization error，应优先边界/定位而非再加纯语义；若background hard negative占主导，才支持语义可靠性。预训练分支支配还提供了“梯度/激活范数主导”这一竞争解释，不能预设余弦冲突。
- **J 可借鉴与边界**：可借鉴全量FP taxonomy、按分辨率选择语义或边界证据、显式监测分支激活/梯度范数比、先分阶段再联合。必须引用HyperLearner；不能复制其CFN/HyperNet结构或把multi-task辅助监督包装为首创。没有许可证证据不得复用代码。
- **K 可证伪假设/最小实验**：先不加模块。H0：plain-P2相对P3–P5的-1.5918在small/far中没有特定错误类型集中；H1a：若新增FP主要为background/Cyclist混淆，则P2语义可靠性不足；H1b：若TP数近似但IoU<0.5/duplicate上升，则是边界/assignment问题。最小实验复用371固定预测，逐类输出background、localization、duplicate、cross-class、annotation proxy及高度/距离分层和配对CI。只允许根据预注册占比选择一条30轮路线：语义可靠性或定位/assignment，禁止同时堆叠。
- **L 页表定位**：问题与低分辨率/拥挤观察见第1页Fig. 1；KITTI、基线和anchor修改见第2页Sec. 3.1；侧支路与small recall见第3页Tables 1–2；不同channel、2×负结果和FP分解见第4页Table 3/Fig. 4；HyperLearner结构/损失/四阶段训练见第5–6页Fig. 5与Eq. 1–2；KITTI HyperLearner对照见第6页Table 4。

## 增量日志

## K02 — MS-CNN：A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection

- **A 元数据与来源**：Zhaowei Cai, Quanfu Fan, Rogerio S. Feris, Nuno Vasconcelos；ECCV 2016；DOI `10.1007/978-3-319-46493-0_22`。原文PDF：https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Cai2016ECCV.pdf ，作者代码：https://github.com/zhaoweicai/mscnn 。截至本次核验未在论文或仓库证据中确认可复用许可证，因此代码只作结构核对，不能直接复制。
- **B 研究问题/gap**：普通候选框网络在KITTI中面对跨多个octave的尺度分布时，小行人和骑行者容易因下采样丢失；把整图无限放大又带来显存与速度成本。论文问题是：怎样让不同感受野负责匹配尺度，并在共享卷积上同时生成高召回proposal和精确检测。
- **C 网络、损失与后处理**：VGG主干上设置`det-8/16/32/64`多个输出层，各层使用与其感受野匹配的anchor；proposal子网联合分类与框回归，多层候选再合并。检测子网从proposal取RoI，浅层特征用固定双线性反卷积上采样后与深层语义对齐，并可编码上下文。正anchor要求其为某GT的最大IoU且IoU不低于0.5（论文第5页）；训练先随机负样本和小多任务权重，再切换hard-negative bootstrapping，缓解多任务初期不稳定（第6页）。论文使用常规框回归和最终检测后处理，并没有本项目的DCLI、不确定性门控或收益—伤害选择器。
- **D KITTI split/训练**：使用KITTI 7,481张带标签training与7,518张隐藏test；内部分析按引用方案将training再分train/val。论文分别训练Car模型与Pedestrian/Cyclist共享模型；并在Caltech另训行人模型。其内部split并非我们的3341/371清单，也不是独立confirmation。输入高度消融为384/576/768；论文原始实现为Caffe/C++、Titan GPU。
- **E 任务与指标**：KITTI 2D detection；Car/Pedestrian/Cyclist；proposal recall用Car IoU 0.7、Ped/Cyc IoU 0.5，最终报告Easy/Moderate/Hard AP。论文发表于2016且早于KITTI 2019年的40-recall更新，表中只写“AP”，因此属于历史AP11时代证据，不能与本项目内部AP_R40直接相减。
- **F 绝对结果、消融与效率**：内部val Table 3：`h576`的Car/Ped Moderate为88.14/70.77；`h576-2x`为89.12/72.49，即同表内+0.98/+1.72 AP；`h576-ctx-c`为89.13/72.13，即+0.99/+1.36。随机负样本`h576-random`为87.50/65.91，相对bootstrapping低0.64/4.86 AP，且作者明确观察到更多行人FP。将输入384升到576时Ped Moderate从68.37到70.77（+2.40），再到768为72.26（再+1.49），代价由约0.09秒升至0.19、0.36秒（Ped/Cyc模型）。上下文不降维版本参数由217M增至357M；降维版本为155M且Ped Moderate 72.13。官方test Table 4中MS-CNN为Car 90.03/89.02/76.11、Ped 83.92/73.70/68.31、Cyc 84.06/75.46/66.07，速度约0.4秒；这些是历史test/AP11数字，不是可移植增益。
- **G 小目标/Ped/Cyc证据**：Table 2按高度分层显示不同检测层分别擅长匹配尺度，组合层提高跨尺度召回。Table 3显示输入放大、浅层特征上采样和hard-negative mining对Pedestrian更敏感。作者因Cyclist val样本少、方差大而将Cyclist排除在消融之外（第10页），这与我们当前Cyclist Moderate分母小、CI宽的问题直接同构：单点Cyclist AP不能作为机制成功证据。
- **H 限制与不可比风险**：多尺度、输入放大、反卷积和上下文同时存在，官方test优势不是单组件因果效应；参数和速度成本明显。内部val未给多seed或CI，Cyclist消融被省略。Table 4和我们的split、AP版本、训练清单、初始化、模型族均不同。反卷积提升浅层语义不等于“P2必然有效”，因为浅层同时会带入纹理噪声和更多负候选。
- **I 与本项目逐项区别**：MS-CNN的scale-specific heads与浅层上采样是plain-P2/多尺度检测的直接先例，不能声称原创；其hard-negative bootstrapping对应我们应检查的P2负样本分配，但它没有可靠性门控、DCLI校准、C/B/AB交互或节点梯度诊断。我们的clean P2相对P3–P5为-1.5918 AP，恰好说明只增加高分辨率路径并不自动复现MS-CNN在另一协议下的收益。
- **J 可借鉴思想与边界**：可借鉴“按目标尺度分配检测层”“低成本语义上采样”“显式hard-negative采样”“输入尺寸—精度—速度成组报告”；必须引用MS-CNN及其先例，不复制其网络图或未确认许可的Caffe代码。可独立实现一个针对P2候选分母的诊断，不把多尺度头本身包装成原创。
- **K 可证伪假设/最小实验**：H0：P2退化与P2层过量易负样本无关；H1：在不改变主干、assigner和损失的前提下，仅将P2负样本预算/难度分布匹配P3，能降低P2 FP并提高Ped/Cyc Moderate macro至少预注册的最小效应。最小实验只做`plain-P2`与“P2负样本分母匹配”二条件、同一3341/371、30轮screen，报告AP、P2/P3 TP/FP/FN、small/far、延迟；若AP方向不一致或FP下降伴随FN显著上升则NO-GO。这是从论文观察导出的新检验，不是复制MS-CNN。
- **L 页表定位**：方法总览第1–8页；KITTI设置与多层anchor见第9页Table 1；proposal尺度/输入实验见第9–11页Table 2与Fig. 5–6；检测消融、速度和参数见第11页Table 3；官方test见第12页Table 4/Fig. 7；hard-negative与两阶段训练见第5–6页。所有delta均由同一Table 3行内独立相减，未把Table 4系统比较冒充模块增益。

## 增量日志

- 2026-08-11：冻结15篇清单并完成K01协议卡。
- 2026-08-11：完成K02 MS-CNN A–L卡；确认“尺度专用头/浅层上采样/上下文”均有直接先例，提取P2负样本分母匹配这一可证伪诊断方向。下一篇K03 SubCNN。
- 2026-08-11：完成K03 SubCNN A–L卡；发现与本项目高度相关的直接先例：共享卷积可使Car/Ped退化、Cyc改善，结构差异过大时甚至不收敛。一般“共享任务负交互”不能主张原创；下一篇K04 RRC。
- 2026-08-11：完成K04 RRC A–L卡；确认“P2细节充分但语义/上下文不足”已有直接理论与实验先例，且过度聚合会再次退化。下一篇K05 DeepParts。
- 2026-08-11：完成K05 DeepParts A–L卡；确认“单独最强模块直接组合可能更差、互补集合才有意义”已有清楚实证。K01–K05首批独立卡完成；K06原清单协议不合格，进入替换核验。
- 2026-08-11：K06 Song TLL被原文证据排除，并按冻结规则替换、完成Mao et al. A–L卡。最关键先例是高分辨率edge有效、纯语义收益缩水，以及预训练侧路可因范数支配拖累主路。下一篇K07 WLOD。

## K07 — Wasserstein Loss-Based Deep Object Detection（WLOD）

- **A 元数据与来源**：Yuzhuo Han, Xiaofeng Liu, Zhenfei Sheng, Yutao Ren, Xu Han, Jane You, Risheng Liu, Zhongxuan Luo；CVPR Workshops 2020，pp. 4299–4305；DOI `10.1109/CVPRW50498.2020.00507`。CVF原始页面：https://openaccess.thecvf.com/content_CVPRW_2020/html/w60/Han_Wasserstein_Loss-Based_Deep_Object_Detection_CVPRW_2020_paper.html ，原始PDF：https://openaccess.thecvf.com/content_CVPRW_2020/papers/w60/Han_Wasserstein_Loss-Based_Deep_Object_Detection_CVPRW_2020_paper.pdf 。本地核验PDF为7页、8,428,364 bytes、SHA256 `87EDA545987F16899A20FAAF320E1FDF4D36401B0073E302BF09FE6B94F13E24`。截至2026-08-12，对精确标题、作者及`WLOD KITTI`的定向检索未找到作者官方代码仓库或许可证，因此只学习论文思想，不复制第三方实现。
- **B 研究问题与真正gap**：标准CE/BCE只奖励真类概率，没有区分“错成相似类别”与“错成风险差异很大的类别”。论文把检测分类视为带语义/风险距离的属性感知问题：在真类概率相同的情况下，把Bike错成Motor应比错成Car付出较小代价。它解决的是**跨类别误分类代价**，不是小目标覆盖、P2纹理噪声、框定位或assignment。
- **C 网络流程、损失、匹配与后处理**：主干保持YOLOv3/Darknet-53或SSD；YOLO输入缩放到416×416，在13/26/52三个特征图上每格预测3个anchor。原定位损失、置信度损失、分类CE/BCE均保留，仅增加`λ·L_WD`。预定义类别ground-distance矩阵`W`按外观相似与误分类严重程度分组；在预测分布与one-hot真值之间，Wasserstein项可化为预测误差与真类对应矩阵行的内积，等价于对所有类别概率施加软权重。KITTI分组为`{Car, Van, Truck}`、`{Tram}`、`{Cyclist, Pedestrian, Person}`。论文没有改变YOLO/SSD的anchor matching、框回归和NMS，也没有给出针对P2的专用路径。
- **D KITTI split与训练设置**：作者使用KITTI全部7,481张有标签图像做随机三折交叉验证，三折大小为2,495/2,493/2,493；排除Misc，其余7类参与。论文未公布每折image ID、随机种子、每折训练/验证角色、epoch、优化器或三折方差，无法复建同一划分。原文把KITTI test写成7,512张，与KITTI官方7,518张不一致，作为论文元数据错误保留。该协议既不是Chen 3,712/3,769 split，也不是本项目3,341/371 clean protocol。
- **E 任务、类别与指标**：任务是多类2D检测。Table 2只写`mAP`，没有说明KITTI官方Easy/Moderate/Hard、Car与Ped/Cyc各自IoU阈值、AP11/AP40、PR采样方式或官方devkit；正文对BDD100K明确提到AP50，却没有把KITTI Table 2的阈值和evaluator闭合。因此这些值只能称“论文内部三折mAP”，不能称KITTI官方Moderate AP，也不能与本项目Pedestrian/Cyclist Moderate macro AP_R40相减。
- **F 绝对结果、消融与效率**：Table 2同一YOLOv3基线为68.3，`W0.5-y`为69.2（+0.9），`W1-y`为69.4（+1.1）；同一SSD基线72.8，`W0.5-s`为74.7（+1.9），`W1-s`为74.4（+1.6）。这些是不同λ下的同表差值，而不是Ped/Cyc Moderate增益。论文没有逐类AP、折间均值±方差、seed、CI、参数增量、FLOPs、FPS或延迟；除λ=0.5/1外也没有拆开“类别分组设计”和“Wasserstein项”的机制消融。BDD100K Table 1的25.8→28.7等数字属于另一数据集，禁止带入KITTI结论。
- **G Pedestrian/Cyclist、small/far/遮挡证据**：Pedestrian与Cyclist被放入同一语义组，说明作者预设二者错分代价较近；但Table 2只有7类mAP，没有Pedestrian/Cyclist单类AP、GT分母、small/far、框高、距离、遮挡或截断分层。因此论文不能证明该损失改善远处行人/骑行者，也不能证明对本项目Cyclist小分母稳定。
- **H 失败、限制与不可比风险**：ground matrix由人工先验定义，类别距离改变就可能改变结论；随机三折身份、方差和评测器未公开；只报告聚合mAP，可能掩盖某类受益、某类受损。论文没有定位/重复框/背景FP分解，也没有成本。更关键的是，KITTI表未闭合IoU与AP版本，且test数量有笔误；因此其+0.9至+1.9只能作为该论文内部点估计，不能当成可移植收益或论文级稳定性证据。
- **I 与本项目逐项区别**：WLOD是分类语义代价；plain-P2是新增高分辨率检测层；fusion调节多尺度特征；DCLI调节定位不确定性/校准。WLOD既不处理P2产生的额外背景候选，也不做框质量校准或门控，更没有C/B/AB三种子交互。它提醒我们先区分P2退化究竟来自cross-class confusion还是background/localization；若后两者占主导，照搬此类分类损失没有因果依据。
- **J 可借鉴思想与不可复用边界**：可借鉴“错误严重程度不应只用0/1正确性表示”和“先建立逐类混淆/代价矩阵再设计损失”；Wasserstein分类损失、人工ground matrix与相似类别分组均已有直接先例，必须引用，不能把它们改名为原创DCLI/P2。没有官方代码/许可证时禁止复制搜索到的非官方实现；若未来独立实现，也只作为公共基线。
- **K 可证伪假设与最小实验**：H0：clean P2相对P3–P5的-1.5918 AP与Pedestrian↔Cyclist或相邻类别错分无关；H1：P2新增错误主要集中于cross-class confusion，且其占新增FP/FN的预注册多数。最小实验先复用371张固定预测，不训练：逐匹配统计`background / localization / duplicate / Ped↔Cyc cross-class`，按small/far分层并给配对CI。只有cross-class占主导时，才允许做一个“类别代价损失 vs plain-P2”的30轮互斥screen；若background或localization占主导，本路线立即NO-GO。即使screen为正，也必须报告两类no-harm、成本和三seed，不能把WLOD表中+1.9当先验承诺。
- **L 原始页表定位**：问题与Bike/Motor/Car例子见PDF pp.1–2 Fig.1；检测多任务损失与CE/BCE局限见p.3 Eq.(1)；Wasserstein定义、ground matrix与总损失见p.4 Eq.(2)–(4)/Fig.3；416输入和13/26/52输出见p.4 Sec.3.3；KITTI随机三折、类别分组与Table 2绝对值见p.5。论文没有可定位的Ped/Cyc逐类表、官方difficulty/AP版本、seed/CI或效率表，因此一律记为unknown。

- 2026-08-12：完成K07 WLOD A–L卡。确认它是“类别语义代价”先例而非NWD式框几何损失；KITTI仅给未闭合评测协议的三折聚合mAP，不能拿其+0.9至+1.9与本项目AP_R40相减。下一篇K08 Complexer-YOLO。

## K08 — Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds

- **A 元数据与来源**：Martin Simon, Karl Amende, Andrea Kraus, Jens Honer, Timo Sämann, Hauke Kaulbersch, Stefan Milz, Horst-Michael Gross；CVPR Workshops 2019，pp.1190–1199；DOI `10.1109/CVPRW.2019.00158`；arXiv `1904.07537`。CVF原始页：https://openaccess.thecvf.com/content_CVPRW_2019/html/Autonomous_Driving/Simon_Complexer_YOLO_Real-Time_3D_Object_Detection_and_Tracking_on_Semantic_CVPRW_2019_paper.html 。本地核验arXiv PDF共10页、2,858,902 bytes、SHA256 `D053CBCD6794377733233D632713381BE5074E8A6AA13DD7FD48D8B040D95DBD`。论文/CVF页未链接作者官方代码，定向检索也未找到带可核许可证的作者仓库，故不复制同名第三方实现。
- **B 研究问题与真正gap**：LiDAR有准确3D位置但远处点云稀疏，RGB有语义却缺直接深度；论文试图把相机分割语义逐点投影到LiDAR，兼顾多类3D检测、跟踪与实时性。另一问题是旋转3D框IoU计算慢且对相差π的朝向不敏感，作者另提Scale–Rotation–Translation score（SRTs）。两者都不是RGB 2D P2改进。
- **C 输入、网络、损失、匹配与后处理**：流程为RGB经ENet分割；用标定把LiDAR点投影至语义图；将`x∈[0,60]m, y∈[-40,40]m, z∈[-2.73,1.27]m`体素化为`768×1024×21`；体素填归一化语义类别；再由Complex-YOLO式49层BEV单阶段网络回归中心、长宽高、地面z偏移、yaw、objectness与类别。作者用stride-2卷积替代池化并加残差。SRTs把scale、translation、rotation拆分，以`ws=.3, wt=1, wr=.5, α=.3, β=.3, γ=.4`组合；训练相似度可替代旋转3D IoU。检测后接Labeled Multi-Bernoulli多目标跟踪与Kalman/UKF运动模型；这部分不能归入单帧检测AP。
- **D KITTI split与训练身份**：使用7,481 labeled training和7,518 hidden test；先按引用方案在training内切train/validation调设置，文中未给ID和数量，之后以全training训练并提交官方test。训练集通过每图额外3个`[-20°,20°]`、相隔至少8°的旋转和随机x翻转扩为4倍；Darknet扩展版从头训练140k iterations，学习率在20k/80k/120k调整。语义分割器先在Cityscapes预训练，再于KITTI微调；因此输入监督并非只含KITTI 3D框。无seed/CI/独立confirmation。
- **E 任务、类别与指标口径**：任务包含orientation similarity、BEV detection、3D detection和tracking；类别为Car/Pedestrian/Cyclist，报告Easy/Moderate/Hard与FPS。论文于2019-06发表，而KITTI官方到2019-10-08才把11个recall positions改为40，因此Table 1属于官方test **AP11**，不是AP_R40。BEV/3D和本项目RGB 2D AP属于不同任务，绝对值不得相减。
- **F 绝对结果、消融与效率**：official test Table 1中，Complexer-YOLO在15.6 FPS下：orientation Moderate为Car 79.08、Ped 31.80、Cyc 56.32；BEV Moderate为66.07/20.88/30.16；3D Moderate为49.44/15.32/23.48。它们是系统结果而非单模块增益。validation Table 3的3D mAP（IoU.7 / SRTs.7）为RGB `28.64/30.02`、Occupancy `31.93/33.24`、Intensity `32.39/33.57`、Semantic `34.14/35.43`；Semantic相对RGB为+5.50/+5.41，但表未展开类别和难度。21与51个高度通道接近，21更快；SRTs替代IoU约+1.3 mAP、训练时间减半、推理快10–20%；rotation anchors反而降精度。全链约11.5 FPS（detector15.6、ENet90、tracker100，GTX1080Ti/Titan）；无参数/FLOPs/显存。
- **G Ped/Cyc、small/far和遮挡证据**：Table 1确有Ped/Cyc官方AP11，但属于3D/BEV。作者定性指出小行人/骑行者只有与其他目标保持一定空间距离时才容易被检测，说明稀疏点和邻近实例会相互干扰；没有small/far分桶AP或配对统计。作者还指出2D人工Ped框与3D框重投影不完全一致、DontCare在BEV/3D处理会带来额外FP，这属于评测/标注问题而非模型提升。
- **H 失败、局限与no-harm警告**：按“框内LiDAR点少于13且距离小于52m”过滤，平均检测结果约+1.3，却令Car Moderate BEV下降4.8 AP，说明总体平均正向可以掩盖关键层伤害。语义体素+5.50还混合额外传感器与语义监督，且没有逐类、seed、CI；不能转写成“语义模块对KITTI普遍+5.5”。RGB+LiDAR、3D/AP11、全训练集test提交与本项目RGB-only 2D/371 development/AP_R40完全不可比。
- **I 与本项目逐项区别**：这里的YOLO在BEV语义体素上做3D框，不含RGB P2头；fusion是相机语义到点云的传感器融合，不是我们的feature reliability gate；SRTs是3D尺度/平移/旋转相似度，不是DCLI的2D定位校准；论文没有C/B/AB factorial、共享梯度或三seed交互。因此它不能证明或修复当前AB-B退化。
- **J 可借鉴思想与不可复用边界**：可借鉴两点：其一，远距离稀疏证据需要语义支持；其二，任何硬过滤必须同时报告每类/难度no-harm，不能用平均数掩盖损失。语义点云、SRTs、LMB跟踪和点数过滤均已有直接先例，必须引用；无官方许可代码则不得复制。项目若采用“局部语义支持度”也只能独立实现为2D诊断/门控，并清楚区分传感器融合。
- **K 可证伪假设与最小实验**：H0：plain-P2新增FP与局部语义支持、邻近实例密度无关；H1：P2坏例集中在低前景/边缘支持、高邻框IoU或低中心距离区域。最小实验先复用371固定预测，按GT高度、遮挡、最近实例距离、边缘/纹理支持分桶，对P3–P5与P2的新增FP、丢失TP、duplicate/localization做paired bootstrap。只有关联方向和CI支持时，才允许做单一“P2语义可靠性门”30轮screen；必须设置Ped/Cyc、near/far no-harm，若总体上升但关键层下降则NO-GO。
- **L 原始页表定位**：总流程与贡献见PDF pp.1–2/Fig.1；语义点云ROI/体素和49层检测器见pp.4–5 Sec.3.1–3.2/Fig.2；SRTs见pp.4–5 Eq.(10)–(15)；跟踪见p.5 Sec.3.3；数据增强/140k训练和官方结果见pp.6–7 Sec.4.1–4.2/Table 1；过滤负结果、SRTs与输入表征消融见pp.7–8 Sec.4.3/Table 3；速度汇总见p.8 Fig.5。

- 2026-08-12：完成K08 Complexer-YOLO A–L卡。它是LiDAR+RGB 3D/BEV AP11系统，不能支撑RGB 2D P2增益；最有用的证据是“平均+1.3但关键Car Moderate -4.8”的no-harm反例。下一篇K09 M3D-RPN。

## K09 — M3D-RPN: Monocular 3D Region Proposal Network for Object Detection

- **A 元数据、原文与代码**：Garrick Brazil, Xiaoming Liu；ICCV 2019 Oral，pp.9287–9296；arXiv `1907.06038`。CVF原始页：https://openaccess.thecvf.com/content_ICCV_2019/html/Brazil_M3D-RPN_Monocular_3D_Region_Proposal_Network_for_Object_Detection_ICCV_2019_paper.html 。作者官方代码：https://github.com/brazgardian/M3D-RPN ，仓库明确MIT License并提供val1/val2/test模型、split setup与resume入口。核验PDF共10页、2,396,634 bytes、SHA256 `4BA3780D26FB2ECC7CEBAC80D03879943562EFB275B840ECDC9E29D92731DA5F`。即使许可证允许复用，任何使用仍需保留MIT声明并引用论文；本项目当前不需要复制其3D实现。
- **B 研究问题与真正gap**：单目RGB缺直接深度，先前单目3D系统依赖外部2D detector、深度、分割或点云子网，误差级联且系统复杂。作者把2D/3D proposal统一到一个RPN：共享类别与图像平面anchor，同时用固定车载相机下“图像纵向位置/2D尺度与深度相关”的几何先验解决3D定位。该gap是单目3D定位，不是2D P2覆盖。
- **C 网络、损失、匹配与后处理**：每个anchor同时含2D宽高、投影深度`zP`、3D长宽高与观察角；其3D先验由训练集中与2D anchor IoU≥0.5的GT统计得到。输出共享分类、2D box、投影3D中心、3D尺寸和角度。正匹配为与某GT IoU≥0.5，分类用softmax，2D框用`-log IoU`，3D变换用SmoothL1，总损失`Lc+λ1Lb2D+λ2Lb3D`。DenseNet-121去末池化、stride保持16，最后分成global普通卷积与local depth-aware卷积；local把特征图纵向切成`b`个bin，各bin独立kernel，12类输出分别学习`α`在global/local之间加权。推理后用3D→2D投影一致性迭代只修正yaw，平均约8次；2D NMS IoU=.4，score<.75过滤。
- **D 数据、split与训练**：KITTI 7,481 labeled/7,518 hidden test；作者使用两个sequence-disjoint validation split（val1、val2）和official test，强调评估帧及邻帧不进入训练，但不是本项目3341/371。输入高512；36 anchors（12尺度×3比例），`b=32`；随机镜像与每minibatch最高loss 20% hard-negative mining。先global warmup再local主阶段，各50k iterations；LR .004 poly decay、batch2。需要3D框和相机投影矩阵，无seed/CI/独立confirmation。
- **E 任务、类别与AP版本**：核心是单目RGB的BEV与3D detection，Car按IoU .7、Ped/Cyc按.5，Easy/Moderate/Hard；另报2D Car。论文/arXiv结果在KITTI 2019-10-08切换40 recall positions之前形成，Tables 1–6属于历史AP11口径；当前排行榜重算值不能与论文表混写。2D、BEV、3D以及IoU .5/.7必须分别解释，不能与项目2D Ped/Cyc Moderate AP_R40相减。
- **F 绝对结果、消融与成本**：Table 1 test Car BEV IoU.7为26.43/18.36/16.24（E/M/H）；Table 2 test Car 3D为20.65/15.70/13.32。Table 3共享多类模型在Moderate的`val1/val2/test`：Car BEV `21.18/21.15/18.36`、3D `17.06/16.48/15.70`；Ped BEV `11.60/11.44/11.35`、3D `11.28/11.30/10.54`；Cyc BEV `10.13/9.09/1.29`、3D `10.01/9.09/1.03`。Cyclist从val约9–10跌至test约1，是强烈泛化警告。Table 5 Car Moderate val1：无local/无post为AP2D/3D/BEV `82.16/10.99/12.99`、118ms；仅post为`82.16/15.08/17.47`、128ms；`b=32+post`为`83.67/17.06/21.18`、161ms，即相对仅post +1.51/+1.98/+3.71。`b=1`的3D仅12.87，`b=4`仅14.46，说明加局部分支/分bin并非单调有效。depth-aware带2–20% reshape开销；无FLOPs/多seed。
- **G Ped/Cyc、small/far与饱和证据**：共享模型确实报Ped/Cyc，但无Easy/Hard逐类消融、small/far、遮挡或GT分母。Cyclist official test崩塌说明少类/小分母和split漂移可压倒validation上的正结果；作者也明确认为Cyclist不稳定，需要采样/增强。其二维Car检测Moderate只有83.67 val1/83.78 test，低于同表旧2D方法约87–89，说明优化3D定位不等于2D AP同步提高。
- **H 失败、限制与替代解释**：固定相机行位置先验对相机俯仰、道路坡度和域迁移敏感；local kernel内存扩大约`×b`，收益随b非单调；post yaw优化贡献了3D/BEV约+4但不改2D，完整系统收益不能全归给depth-aware卷积。val/test的Cyclist巨大落差、单seed、AP11和仅Car消融都限制主张。它证明“几何先验+局部分支可能有用”，也证明“任意加分支会伤害、开发集正向未必泛化”。
- **I 与本项目逐项区别和撞车**：row-wise depth-aware kernels、固定道路相机几何、global/local learned fusion、2D尺度→3D深度先验均是**直接先例**；我们不能把“道路几何条件化P2”笼统称原创。M3D-RPN没有P2头噪声、fusion可靠性估计、DCLI、收益–伤害fallback或C/B/AB交互。其local/global融合按输出学静态α，而项目候选贡献若存在，只能落在基于实际检测残差/不确定性的候选级价值与no-harm，而不是行分bin本身。
- **J 可借鉴与许可证边界**：可借鉴“低层共享、只在高层引入位置特异性”“每个输出参数分别学习global/local权重”“同时报告定位严阈值与宽阈值”“固定相机几何的敏感性”。MIT代码可在保留声明、归属和引用条件下复用，但直接拷贝depth-aware实现不构成原创；更合适的是独立做小规模诊断或把其作为公开基线。
- **K 可证伪假设与最小实验**：H0：plain-P2退化在图像纵向位置/框高/估计距离上均匀；H1：退化集中于少数道路几何带，说明静态P2对不同深度分配了错误的候选/语义预算。先复用371固定预测，画`y_center×height×distance`分层的TP/FP/FN、duplicate、matched IoU与paired CI；若没有稳定位置依赖，geometry路线NO-GO。若成立，只允许一个“P2几何条件化分配/校准”30轮screen，对照plain-P2，固定其他组件，并在不同相机姿态代理层做no-harm。它若只是row bins或static α，必须标为M3D-RPN类公共基线，不能写原创。
- **L 原始页表定位**：统一anchor与动机见PDF pp.1–3/Figs.1–3；anchor统计、分类/2D/3D损失见pp.3–4 Eq.(1)–(7)/Fig.4；depth-aware global/local融合、post yaw算法见p.5 Eq.(8)–(9)/Algorithm 1；训练/NMS/OHEM与split见p.6 Sec.3.5–4.1；BEV/3D比较见pp.6–7 Tables 1–4；`b`、post、延迟和非单调负结果见pp.7–8 Tables 5–6。

- 2026-08-12：完成K09 M3D-RPN A–L卡。它直接占用了“固定道路相机几何+位置特异卷积+global/local融合”的大部分宽泛原创说法；更重要的是Cyclist从val约10 AP跌到test约1，说明开发集正向与独立泛化必须分开。下一篇K10 D4LCN。

## K10 — Learning Depth-Guided Convolutions for Monocular 3D Object Detection（D4LCN）

- **A 元数据、原文与代码**：Mingyu Ding, Yuqi Huo, Hongwei Yi, Zhe Wang, Jianping Shi, Zhiwu Lu, Ping Luo；CVPR 2020，pp.11672–11681；arXiv `1912.04799`。CVF原始页：https://openaccess.thecvf.com/content_CVPR_2020/html/Ding_Learning_Depth-Guided_Convolutions_for_Monocular_3D_Object_Detection_CVPR_2020_paper.html 。作者官方代码：https://github.com/dingmyu/D4LCN ，MIT License。核验PDF含主文与附录共12页、4,143,608 bytes、SHA256 `C810FA87F3ED587AA7C2296BE678A32EE3EEDF4C00CB2E1AEDE0AC376957A7E3`。代码可以按MIT保留声明/归属后复用作基线，但动态卷积思想本身不能作为本项目原创。
- **B 研究问题与真正gap**：单目3D检测既受尺度/透视变化影响，又缺物理深度；普通2D卷积对所有位置用同一核，难区分不同深度的目标和背景；pseudo-LiDAR又高度依赖深度图坐标精度并丢失RGB语义。作者用深度图**指导RGB卷积核与感受野**，而不是把深度直接变成点云。该问题是单目3D表征，不是2D P2覆盖或DCLI交互。
- **C 网络、损失、匹配与后处理**：ResNet-50 RGB分支提取特征，前三个block另有深度filter-generation分支；每个位置/样本/通道的动态depthwise local kernel由估计深度生成，并用shift+elementwise product减少计算。shift-pooling在通道间交换信息；每通道再从dilation 1/2/3学习softmax权重，适配尺度。检测头沿用2D–3D priors：36 anchors、2D框、投影3D中心、深度/尺寸/yaw、8个3D角点。总损失为focal调制`(1-s_t)^γ`乘分类CE、2D SmoothL1、3D SmoothL1和corner loss，`γ=.5`。2D NMS=.4，yaw再经hill-climbing优化。深度引导动态/局部/depthwise/dilated四部分均有明确消融。
- **D 数据、split、训练与额外输入**：KITTI 7,481 train/7,518 test；split1=3,712/3,769、split2=3,682/3,799。输入512×1760；DORN产生单目深度图，另比较MonoDepth、DispNet、PSMNet；ImageNet预训练ResNet-50。40k iterations，SGD momentum .9、weight decay .0005、batch8/4×V100、只水平翻转。训练与测试都需相机内参和预先提取深度；无seed、CI或独立confirmation，且不是本项目3341/371。
- **E 任务、类别、IoU与AP口径**：2D、BEV和3D任务分开；主表是Car IoU .7，multi-class另含Ped/Cyc（通常IoU .5）。论文非常重要地同时给AP_R11与AP_R40：official test Table 1只用AP_R40；split1/2比较旧方法时仍用AP_R11，Table 2才在同一split1同时列两种口径。故任何`split1/split2/test`三元组本身混合AP_R11/AP_R40，不能把下降全解释成模型泛化，也不能与本项目2D两类Moderate macro相减。
- **F 绝对结果、消融与成本**：Table 1 official test Car 3D AP_R40为16.65/11.72/9.51（E/M/H）。文中“Moderate相对提升9.1%”指相对AM3D 10.74的**相对百分比**，绝对只+0.98 AP点。Table 2 split1 AP_R40 Car Moderate：3DNet 3D 12.13、+corner loss 12.09（-0.04）、+DLCN 13.50、+shift-pooling 14.44、full 16.20；BEV 16.67→17.75→18.68→20.07→22.58；2D 84.45→84.93→86.71→86.52→86.93。Table 7更细：ConvNet 12.09、depth-guided CN 12.18、LCN 13.93、DLCN 13.50（相对LCN -0.43）、SP-DLCN 14.44、full 16.20；说明组件并非单调增益。Table 3不同深度图的AP_R40 Moderate为MonoDepth13.18、DORN16.20、DispNet18.56、PSMNet19.80，额外/更准深度可贡献数点，不能全归D4LCN。论文未报参数/FLOPs/端到端延迟或多seed；动态实现的实际成本不能记作零。
- **G Ped/Cyc、small/far与分母**：Table 5按`split1 AP11 / split2 AP11 / test AP40`混合口径列三类3D：Ped Moderate `11.23/10.37/3.42`，Cyc `4.41/6.54/1.67`；不能直接把三者下降量当泛化gap。作者解释人/骑行者非刚体、深度难估，远/近与尺度需要不同dilation；但没有GT分母、small/far、遮挡分层、seed或CI。Ped/Cyc absolute很低且协议混合，不能支持我们2D P2会有大幅提升。
- **H 失败、局限与替代解释**：`+corner loss`在AP11 Moderate由14.61→15.57看似+0.96，但AP_R40由12.13→12.09反而-0.04，直接证明评测采样改变可以翻转结论；DLCN相对LCN也为负。full提升混合深度先验、local/dynamic/depthwise、shift pooling、adaptive dilation、corner loss与不同depth estimator。固定深度图质量、相机标定、3D标签和大输入带来额外成本；只Car消融、无CI、mixed AP，使其不能作为本项目收益承诺。
- **I 与本项目逐项区别和原创撞车**：深度引导动态局部卷积、逐位置/逐样本filter、逐通道自适应感受野、道路透视尺度条件化都已有直接先例；普通“动态P2卷积/几何自适应感受野”不能再声称原创。D4LCN没有plain-P2对照、可靠性门控、DCLI、收益–伤害估计或AB-B检验。其任务是3D Car主导，不能解释当前2D Ped/Cyc P2 -1.5918。
- **J 可借鉴与许可证边界**：可借鉴“几何先验更适合用来调制局部表征而非替代RGB”“感受野应按尺度适配”“同一消融必须同时报AP11/AP40/各类”。官方MIT代码可作公开基线，但若复制D4LCN模块就只能是引用复现，绝非原创P2。项目更应先验证是否真有几何依赖，再决定是否独立实现一个更小、不同estimand的门控。
- **K 可证伪假设与最小实验**：H0：P2新增背景FP/定位错配与预测深度/纵向位置/所需感受野无关；H1：坏例集中在远距离小框，且P2局部纹理的最佳上下文尺度随深度系统变化。先在固定371预测上按距离、框高、纵向位置统计FP/FN/matched IoU/duplicate并做paired CI；若无稳定交互，动态几何路线NO-GO。若成立，30轮只比较plain-P2与单一“深度条件的P2上下文尺度”版本，深度估计与延迟计入成本，并以固定filter/随机条件为负对照；不得同时叠DCLI/fusion。
- **L 原始页表定位**：问题与四种动态属性见PDF pp.1–3/Figs.1–2；RGB/深度双分支和filter module见pp.3–5/Figs.3–4/Eq.(1)–(2)；2D–3D anchor和损失见pp.5–6 Eq.(3)–(7)；数据/AP11/AP40/训练见pp.6–7 Sec.4.1；official test和主消融见p.7 Tables 1–2；depth/卷积替代/multi-class见p.8 Tables 3–5；完整组件分解见附录p.10 Table 7。

- 2026-08-12：完成K10 D4LCN A–L卡。最关键核验是论文所谓9.1%为相对百分比、只等于+0.98绝对AP；并且corner loss在AP11为正、AP40为-0.04，说明必须冻结评测口径。下一篇K11 End-to-End Pseudo-LiDAR。

## K11 — End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection

- **A 元数据、原文与代码**：Rui Qian, Divyansh Garg, Yan Wang, Yurong You, Serge Belongie, Bharath Hariharan, Mark Campbell, Kilian Q. Weinberger, Wei-Lun Chao；CVPR 2020，pp.5881–5890；DOI `10.1109/CVPR42600.2020.00592`；arXiv `2004.03080`。CVF原始页：https://openaccess.thecvf.com/content_CVPR_2020/html/Qian_End-to-End_Pseudo-LiDAR_for_Image-Based_3D_Object_Detection_CVPR_2020_paper.html 。作者官方代码：https://github.com/mileyan/pseudo-LiDAR_e2e ，仓库根目录MIT License（Copyright 2020 Yan Wang）。本地核验PDF 10页、952,028 bytes、SHA256 `B361A681E3EDADF298F6F6BA674A0AC54F614AEB57AE2C05F57662D76378EC7F`。代码虽可按MIT复用，但必须保留许可和归属；CoR、soft quantization及检测损失反向塑造深度均已有直接先例。
- **B 研究问题与真正gap**：pseudo-LiDAR先训练通用双目深度，再冻结深度网训练3D detector；像素级深度误差与最终检测目标不对齐。约90%像素是背景，车及人/骑行者合计约10%，其中人不足1%；普通深度损失会偏向背景、近景和有稠密监督处，而远物体只有少量点，检测/定位最易失败。真正gap是让检测误差穿过表示变换回到真正影响检测的深度像素，同时保持pseudo-LiDAR模块化，而不是加P2或提高输入分辨率。
- **C 输入、流程、损失、匹配与后处理**：输入为左右双目；SDN产生dense depth，经Eq.(1)相机模型投成3D点。PointRCNN路线对点云做高度过滤和按球坐标角度bin稀疏化，被保留点直接传梯度；PIXOR路线原本硬体素化不可导，作者用每点影响自身及26邻近bin的RBF soft quantization，固定`σ²=.01`、`700×800×35`体素，使空/邻bin的检测梯度能推拉点。训练三阶段：Scene Flow预训练并在KITTI微调SDN；冻结SDN从头训detector；最后联合微调。总损失`λ_det L_det + λ_depth L_depth`，其中检测仍用各自分类/3D框回归，深度用有LiDAR投影GT像素上的SmoothL1。论文复用PointRCNN/PIXOR自身assignment与后处理，没有提出YOLO 2D matching/NMS。
- **D KITTI split、监督和独立性**：采用3,712 train / 3,769 validation / 7,518 official test；论文给出数目但没有在该段证明sequence-level independence。训练还使用Scene Flow、双目右图、相机标定与投影LiDAR稀疏深度GT，监督强于本项目单幅RGB 2D检测。作者既在3,769 val选方案，也提交隐藏test；但无多seed/CI，不能把val当独立confirmation。本项目的3341/371与其split完全不同。
- **E 任务、类别、IoU、难度与AP定义**：只评Car；任务为BEV与3D AP，不是2D AP。validation在IoU .5和.7分别报Easy/Moderate/Hard，official test报IoU .7。论文表头只写AP，正文没有明确AP11或AP40 recall采样，因此本卡将其记为`unknown`，不凭发表年份补写。Ped/Cyc只出现在像素分布动机，未给检测AP，绝不能与我们的Ped/Cyc Moderate macro AP_R40相减。
- **F 绝对结果、消融、梯度与成本**：Table 2同协议PointRCNN val IoU .7 Moderate，PL++→E2E的BEV `64.0→65.7`（+1.7 AP点）、3D `50.1→51.7`（+1.6）；Table 3 official test Moderate为BEV `58.0→58.8`（+0.8）、3D `42.4→43.9`（+1.5），Hard为`51.3/37.0→52.1/38.1`（+0.8/+1.1）。PIXOR val Moderate BEV IoU .5/.7为`78.4/61.1→84.8/64.3`（+6.4/+3.2），test Moderate BEV `48.3→51.7`（+3.4），但这里同时改变可导量化与联合更新，不能拆成两个独立模块增益。Table 1显示depth loss、PointRCNN loss、PIXOR loss在深度图上的非零梯度覆盖约`3%/4%/70%`，梯度均值`1e-5/1e-3/1e-5`、梯度和`0.1/10/1`；作者据此用PointRCNN路线`λ_depth=1, λ_det=.01`，PIXOR路线`1/.1`平衡开局梯度和。速度为单GPU PointRCNN 0.49 s/frame、PIXOR 0.55 s/frame，SDN本身约0.39 s/frame；无统一FLOPs/参数/多seed。
- **G Ped/Cyc、small/far/遮挡证据**：没有Ped/Cyc AP、GT分母、small/far数值分层或CI。Figure 6只定性展示联合版减少远车漏检、改善远车定位以及一个场景的FP；结论还承认KITTI远车标注较少。这个观察可产生远目标假设，却不是far AP证据，更不能支持我们两类2D P2会提升。
- **H 失败、局限和替代解释**：Table 4说明继续训练单个子网并不自动变好：在IoU .7 Moderate，冻结基线`64.0/50.1`（BEV/3D），只训Depth `64.5/50.5`、只训RPN `63.9/50.0`、只训RCNN `63.4/50.4`；全联合才为`65.7/51.7`，但某些列的最佳仍来自非完整组合。Table 5也显示soft quantization本身未必优于hard版本，真正改善依赖联合训练。性能还混合Scene Flow、双目、LiDAR深度监督、量化方式和loss配权；val/test有差距，无seed/CI/AP采样声明。作者只说明梯度和要平衡，不能排除学习率、额外训练或表示变化等解释。
- **I 与本项目逐项区别和因果关联**：论文中的`PL`是pseudo-LiDAR，绝非P2特征层；它没有YOLOv8、Ped/Cyc 2D head、fusion gate、DCLI或C/B/AB factorial。可迁移的是研究方法：合并两条监督路径前，要量化共享参数上梯度是否覆盖正确样本、方向是否冲突、范数是否支配。它使“多任务梯度量级会破坏联合优化”成为明确先例，因此我们不能声称梯度冲突/配权本身原创；同时它没有证明当前AB-B三seed为负的原因就是梯度，因为我们的正式日志尚未把诊断变化与AP/FP/FN因果连起来。
- **J 可借鉴思想与许可边界**：可借鉴`非零梯度覆盖率 + 梯度和/范数 + 分层性能`三联证据、先冻结两段式强基线、再让唯一联合路径接受检测目标监督。必须引用differentiable CoR、soft quantization、检测引导深度、按梯度量级配loss。MIT代码若复用须保留声明；本项目只需独立的小型诊断hook，不应复制其3D结构、图或表述，也不能把通用loss reweight/stop-gradient改名为原创P2。
- **K 可证伪项目假设与最小实验**：H0：DCLI启用后，P2共享参数上的fusion/DCLI梯度覆盖、范数比和cosine与`AB-B<0`无稳定对应；H1：退化在DCLI ramp后集中出现，且仅P2节点的某一监督对共享可靠性语义形成持续范数支配或负cosine，P3–P5不出现同等变化。最小实验只复用已保存C/B/AB梯度日志和预测，按epoch与P2–P5节点联结梯度coverage/norm/cosine、TP/FP/FN/matched-IoU变化；只有时序、节点和错误类型三者对应，才允许一个与诊断唯一对应的30轮隔离/配权screen。若只有cosine异常或只有AP变化，则因果声明NO-GO。
- **L 原始页表定位**：问题与像素不平衡见PDF pp.1–3/Figs.1–3；soft quantization与subsampling见pp.3–4 Eqs.(2)–(4)/Fig.4；联合损失、split和梯度统计见p.5 Eq.(5)/Table 1/Sec.4.1–4.2；val与test绝对值见p.6 Tables 2–3；子网与量化消融见p.7 Tables 4–5；远车/FP定性和延迟见p.8 Fig.6/Sec.4.5–4.6。官方代码仓库根LICENSE明确MIT。

- 2026-08-12：完成K11 End-to-End Pseudo-LiDAR A–L卡。它最直接地说明联合优化必须审计梯度覆盖与量级，但任务、输入和AP口径均与本项目不同；其结果不能搬成我们的增益或根因证明。下一篇K12 MonoRUn。

## K12 — MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation

- **A 元数据、原文与代码**：Hansheng Chen, Yuyao Huang, Wei Tian, Zhong Gao, Lu Xiong；CVPR 2021，pp.10379–10388；DOI `10.1109/CVPR46437.2021.01024`；arXiv `2103.12605`。CVF原始页：https://openaccess.thecvf.com/content/CVPR2021/html/Chen_MonoRUn_Monocular_3D_Object_Detection_by_Reconstruction_and_Uncertainty_Propagation_CVPR_2021_paper.html 。作者官方代码：https://github.com/tjiiv-cprg/MonoRUn ，MIT License；仓库说明基于MMDetection/MMDetection3D，PnP C++由PVNet修改，复用时还必须追踪这些上游许可。本地PDF 10页、1,752,017 bytes、SHA256 `FDF30AAB182104185871A307BB5F662EC99670B56BACA045CCB180493C11DA23`。
- **B 研究问题与真正gap**：单目3D定位可通过dense 2D–3D correspondence和PnP求解，但道路实景没有逐像素真实物体几何；简单重投影又会被背景点和遮挡异常值破坏。作者要在只有3D框标签时自监督学习形状/对应关系，并把aleatoric/epistemic uncertainty一直传播到PnP pose covariance与最终3D排序。这个gap是单目3D重建和概率定位，不是P2覆盖。
- **C 网络、损失、匹配与后处理**：预训练Faster R-CNN+ResNet-101和六层FPN给2D box/RoI。global extractor从7×7 RoI预测3D尺寸和16维全局latent；NOC decoder把14×14局部特征与latent相加、CARAFE上采样到28×28，预测normalized object coordinates以及重投影二维标准差。物体坐标为NOC与尺寸逐元素乘积，经GT pose投回图像做自监督。Robust-KL由Gaussian/Laplacian KL改为Huber式mixed KL，并用EMA估计`E[1/σ]`归一化，避免σ下降造成多任务梯度持续放大。推理用不确定性加权PnP求pose与近似协方差，再由MLP融合pose uncertainty和网络特征预测3D localization score；3D NMS IoU=.01。默认总损失含2D detection、重投影、dimension、score、covariance calibration，另可加稀疏LiDAR NOC监督。
- **D 数据、split与训练**：KITTI 7,481 labeled/7,518 hidden test；使用3,712/3,769 Chen split，并提交官方test。完整trainval模型与split模型均由官方仓库提供。AdamW、weight decay .01、batch6/2×RTX2080Ti、base LR .0002 cosine；split训练50 epochs，full trainval 32 epochs；端到端PnP仅在自监督预训练后追加15 epochs、LR .00003。数据增强为水平翻转与photometric distortion；默认论文模型使用LiDAR supervision，另报告无额外监督版本。无多seed、CI或独立confirmation。
- **E 任务、类别、IoU与AP**：论文明确使用官方40-point interpolated AP，即AP_R40。Car 3D按IoU .7，另在val报IoU .5；Pedestrian/Cyclist 3D按IoU .5；均分Easy/Moderate/Hard。不是2D AP，不能与项目Ped/Cyc 2D Moderate macro直接相减。Table 1还主动标记DORN预训练深度与KITTI-object val重叠导致的过拟合数字，说明额外数据身份是协议核心。
- **F 绝对结果、消融和效率**：official test Car 3D AP_R40，Ours无额外监督为`16.04/10.53/9.11`，+LiDAR为`19.65/12.30/10.58`；val Car IoU .7为无额外监督`17.26/12.27/10.41`，+LiDAR `20.02/14.65/12.61`。official test Ped为`10.88/6.78/5.83`，Cyclist仅`1.01/0.61/0.48`，作者归因训练样本不足。Table 3的Car六项平均mAP：self+LiDAR Robust-KL baseline `31.21`；仅self `28.57`；仅LiDAR `18.84`；SmoothL1 `26.35`、LapKL `29.47`、mixed KL `30.05`、Robust-KL `31.21`。LapKL+E2E `29.47→29.73`（+0.26），但更强Robust-KL+E2E `31.21→31.09`（-0.12）；dimension-only epistemic `31.47`（+0.26），full epistemic `31.16`（-0.05）；删latent `29.78`（-1.43）。推理0.070 s，含50次MC dropout和PnP；无参数/FLOPs/多seed。
- **G Ped/Cyc、small/far、遮挡与不确定性证据**：Ped Moderate 6.78但Cyclist Moderate仅0.61，直接暴露小类/少样本失败；没有GT分母或CI。latent被设计编码occlusion/truncation/shape，Table 4显示将latent置零对无aleatoric模型几乎无影响`26.35→26.33`，对Robust-KL则`31.21→29.23`（-1.98），说明其作用主要经不确定性分支。Figure 6按距离比较预测与真实定位协方差：训练集校准较好，validation普遍过度自信，仍需经验缩放；这是比只报平均AP更诚实的far/泛化风险证据。
- **H 失败、局限、泄漏与替代解释**：强结果默认用LiDAR NOC监督；纯self几何可能学到不真实形状并过拟合，Figure 5明确展示LiDAR的shape regularization。uncertainty更复杂并不单调改善：full epistemic和Robust-KL上的E2E都略降；Cyclist几乎失效；val covariance overconfident。3D NMS、50次MC、LiDAR监督、Faster R-CNN及3D标签均与项目预算不同。消融是单split/单seed的六指标平均，不能确认-0.12/+0.26是否稳定。
- **I 与本项目的直接关系和撞车**：Robust-KL的异方差定位、weight normalization、uncertainty-driven scoring、校准与stop-gradient均是直接先例；我们不能把DCLI异方差思想、uncertainty分支或“校准梯度隔离”本身称原创。尤其Eq.(12)明确将`p*` detach、只训练4维calibration vector，说明语义所有权隔离不是空白。它不含YOLO P2、可靠性融合或2D Ped/Cyc C/B/AB；只能支持一个诊断假设：更强基础目标下继续端到端回传可能负迁移，隔离或受控校准要用AP+校准误差共同验证。
- **J 可借鉴思想与许可边界**：可借鉴三点：用EMA归一化异方差权重避免范数膨胀；只对真正的校准参数回传校准损失；同时报告预测不确定性与实际误差随距离的可靠性。必须引用Robust-KL、PnP covariance calibration、pose detach、MLP uncertainty score。MIT代码可在保留许可且核对MMDetection/MMDetection3D/PVNet上游后复用作对照；项目最好独立实现最小标量/梯度隔离，不复制3D PnP系统。
- **K 可证伪项目假设与最小实验**：H0：DCLI calibration的梯度是否进入P2共享语义参数，不影响B或AB；H1：当校准梯度进入共享P2/fusion语义时会改变类别/可靠性表示，隔离后AB-B不再持续为负，同时定位校准误差不恶化。最小30轮screen只比较原AB与“calibration stop-gradient ownership”单一干预，固定seed/split/schedule；必须同时报`AB-B`、Ped/Cyc macro、matched-IoU、FP/FN、confidence calibration、small/far及梯度范数。若只AP回升而校准/梯度机制不对应，不能宣称因果；若三seed仍非正则NO-GO。
- **L 原始页表定位**：总流程见PDF pp.1–3/Figs.1–2；global/NOC网络和aleatoric/epistemic见pp.3–4/Figs.3–4/Eqs.(1)–(2)；Robust-KL与EMA归一见pp.4–5 Eqs.(3)–(8)；uncertainty PnP、pose detach校准和score见p.5 Eqs.(9)–(15)；训练/split/AP_R40见p.6 Sec.3.6–4.2；official结果、Ped/Cyc和负/正消融见p.7 Tables 1–3；latent及距离校准见p.8 Table 4/Figs.5–6。

- 2026-08-12：完成K12 MonoRUn A–L卡。它直接先例化了异方差定位、权重归一与校准梯度隔离，并展示更强目标上的E2E与full epistemic可为负；因此我们的学术价值不能是“首次stop-gradient”，只能是对特定P2共享语义所有权的可证伪诊断与验证。下一篇K13 GUPNet。

## K13 — Geometry Uncertainty Projection Network for Monocular 3D Object Detection（GUPNet）

- **A 元数据、原文与代码**：Yan Lu, Xinzhu Ma, Lei Yang, Tianzhu Zhang, Yating Liu, Qi Chu, Junjie Yan, Wanli Ouyang；ICCV 2021，pp.3111–3121；DOI `10.1109/ICCV48922.2021.00310`；arXiv `2107.13774`。CVF原始页：https://openaccess.thecvf.com/content/ICCV2021/html/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.html 。官方代码：https://github.com/SuperMHP/GUPNet ，MIT License（Copyright 2021 Yan Lu）。本地PDF 11页、1,942,233 bytes、SHA256 `6E9CF4BFC453FBEA7B6D9D5F03820A328C0BF6334FE95A951E9293692D456CD0`。仓库还明确提示实现含论文未描述tricks、ROIAlign有不可完全复现的随机性，released checkpoint与论文数值有jitter；复现时必须记录版本而不能只引用论文表。
- **B 研究问题与真正gap**：单目3D常用`depth=focal×3D height/2D height`几何投影，但2D/3D高度的小误差会被杠杆放大成深度大误差；训练早期前置任务未收敛时，深度分支又反向污染整个多任务网络。作者同时解决推理可靠性和训练层级依赖：把高度与bias的不确定性显式传播为depth uncertainty，并让后置任务只在前置任务学稳后逐渐增权。
- **C 网络、损失、匹配与推理**：DLA-34+CenterNet先预测heatmap、2D offset/size；RoIAlign后拼normalized coordinate maps，再预测3D offset、angle、3D size。3D height与learned depth bias均设Laplace分布，投影均值`f μ_h/h_2d`，投影方差与bias方差按独立项合成，得到Geometry Uncertainty（GeU）；depth用Laplace NLL，`exp(-σ_depth)`作Uncertainty-Confidence并与2D score相乘。Hierarchical Task Learning（HTL）把任务分为2D detection→basic 3D heads→depth三层；每项权重用polynomial scheduler，指数由前置loss最近K epoch相对最初K epoch的导数趋势决定，所有前置趋稳后后置权重才升高。论文沿用CenterNet式中心匹配/解码，不是YOLO assignment。
- **D 数据、split、训练与额外数据**：KITTI 7,481 train/7,518 test；3,712/3,769 split做消融，trainval提交official test。输入380×1280、stride4、batch32/3×Titan XP、140 epochs，LR `1.25e-3`在90/120降10倍，前5 epochs linear warmup，HTL窗口K=5。方法标记`None`额外数据；无depth/LiDAR预训练。没有seed、CI或独立confirmation，本项目3341/371不可与其直接比较。
- **E 类别、任务、IoU和AP**：论文明确使用AP40以避免AP11偏差。official test给Car IoU .7、Ped/Cyc IoU .5的3D AP40，Easy/Moderate/Hard；val还给Car 3D/BEV的IoU .5与.7。不是2D AP。标题/摘要所谓“+3.74% Car、+4.7% Ped”实为**绝对AP点**且都是Easy相对`None`组最好先例，不是Moderate，更不是本项目提升。
- **F 绝对结果、消融与成本**：official test GUPNet Car `20.11/14.20/11.77`，相对无额外数据最好方法为`+3.74/+3.19/+2.25`；Ped `14.72/9.53/7.87`，相对None为`+4.70/+2.85/+2.34`；Cyc `4.18/2.65/2.09`，相对None仅`+0.39/+0.53/+0.26`，且相对LiDAR-supervised CaDNN为`-4.52/-2.09/-2.09`。val Car对MonoPair：严格IoU .7 Moderate 3D `12.30→16.46`（+4.16），但IoU .5为`42.39→42.33`（-0.06），说明主要改善高精度定位/排序，不是宽松覆盖。Table 3 Moderate 3D按a→h为`11.00,12.44,13.53,12.79,13.57,15.70,15.63,16.46`；GeP在已有UnC时只有+0.04且Easy/Hard下降，GeU/HTL才恢复。Table 4 GradNorm/Task-Uncertainty/HTL Moderate为`10.49/13.94/16.46`。单Titan XP 29.4 FPS；无FLOPs/参数/多seed。
- **G Ped/Cyc、small/far、遮挡与分母**：作者给出trainval实例总数Car/Ped/Cyc=`14,357/2,207/734`，用少样本解释Cyclist弱；这是珍贵分母证据，但无每难度GT或CI。Figure 6展示远距离、遮挡坏例有更高GeU，只是定性。HTL的depth权重约第11 epoch才明显升高，说明依赖链早期开启有风险；但没有按far/small AP或no-harm数值，因此不能把“远目标改善”当事实。
- **H 失败、局限与替代解释**：所有组件非加性：GeP与UnC局部负交互；GeU与HTL的边际增益随另一项存在而明显缩小；普通GradNorm在层级任务中比baseline还低。宽IoU不升而严IoU大升，可能是定位校准而非召回。代码承认未描述tricks与ROIAlign jitter；论文无seed/CI，headline选Easy，Cyclist相对强额外监督基线为负。固定相机内参/投影对域迁移敏感，HTL含观察窗口和训练时长假设。
- **I 与本项目逐项区别和原创撞车**：几何投影不确定性、uncertainty→confidence、按依赖关系延迟/渐增后置任务、基于loss趋势动态权重都已有直接先例；“延迟DCLI”“几何不确定性门控”“层级任务调度”不能笼统声称原创。GUPNet是单目3D CenterNet，不含YOLO P2/fusion/DCLI；但其消融与我们现象高度相关：两个单独合理组件可能局部负交互，且强收益集中于严格定位IoU而非宽松覆盖。它提示先分解coverage/localization/ranking，再选择唯一干预。
- **J 可借鉴思想与许可边界**：可借鉴显式任务依赖图、后置loss延迟到前置收敛、同时看IoU .5/.7以判断覆盖还是定位、报告小类分母。必须引用GeU、UnC和HTL；PCGrad/GradNorm/uncertainty weighting也都是公共基线。MIT代码可依许可复现，但仓库自述有未公开tricks，故更适合将论文公式独立实现为对照，而非直接复制并宣称原创。
- **K 可证伪项目假设与最小实验**：H0：DCLI ramp时机与fusion/P2前置任务稳定性无关；H1：DCLI在P2可靠性/类别分配尚快速变化时启动，导致AB-B持续为负，待前置loss/梯度稳定后启用可消除破坏。最小screen只测试一个公开基线式干预：固定epoch延迟或以预注册前置指标触发DCLI，其他代码不变；对照B、旧AB，报三类诊断（低IoU coverage、高IoU localization、confidence ranking）与Ped/Cyc、small/far、FP/FN、延迟。若仅30轮单seed正或AB-B仍≤0，NO-GO；即使成功，原创候选也只能是“在P2可靠性×定位校准特定冲突上的实证机制”，不是HTL本身。
- **L 原始页表定位**：问题与geometry leverage见PDF pp.1–2/Figs.1–2；CenterNet/RoI/basic heads见pp.3–4/Fig.3；GeU/UnC公式见p.4 Eqs.(1)–(6)；HTL依赖与loss趋势见p.5 Figs.4–5/Eqs.(7)–(10)；split/AP40/训练与test三类绝对值见p.6 Table 1；val、完整消融、GradNorm对照、实例分母和FPS见p.7 Tables 2–4；远/遮挡定性与depth约epoch11启权见p.8 Figs.6–7。

- 2026-08-12：完成K13 GUPNet A–L卡。它既给出层级延迟训练的直接先例，也给出GeP×UnC负交互、GradNorm失败和宽/严IoU结论翻转；对本项目最有用的是先区分coverage、localization、ranking，再决定是否只测试一个延迟DCLI基线。下一篇K14 MonoDLE。

## K14 — Delving Into Localization Errors for Monocular 3D Object Detection（MonoDLE）

- **A 元数据、原文与代码**：Xinzhu Ma, Zhihui Wang, Haojie Li, Pengbo Zhang, Wanli Ouyang；CVPR 2021，pp.4721–4730；DOI `10.1109/CVPR46437.2021.00469`；arXiv `2103.16237`。CVF原始页：https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Delving_Into_Localization_Errors_for_Monocular_3D_Object_Detection_CVPR_2021_paper.html 。作者官方代码：https://github.com/xinzhuma/monodle ，MIT License。本地PDF `research-notes/source-pdfs/Ma_2021_MonoDLE.pdf`，10页、15,477,018 bytes、SHA256 `8C7C3F3005463E9539680454D2D26B3596343FB0AC01F02266B72D60A37BB0FF`。
- **B 研究问题与真正gap**：作者没有先堆新模块，而是把单目3D检测的预测量逐个替换成GT，回答“到底是哪一类定位误差限制AP”。诊断显示投影3D中心与深度是主要瓶颈，尤其远处目标的几像素中心偏移会被相机投影放大成米级误差。真正贡献是误差定位、针对唯一误差源设计训练目标，并揭示直接联合优化会崩溃；不是普通2D P2检测。
- **C 网络、损失、匹配与后处理**：以DLA-34/CenterNet为基础，同时预测2D中心热图、2D中心偏移、2D框大小、投影3D中心偏移、深度、3D尺寸与方向。2D中心用focal loss，其余常规分支用L1/MultiBin；深度采用带不确定性的回归。作者保留2D辅助任务帮助3D表征，并把3D尺寸损失改为dimension-aware、IoU-oriented的归一化误差：只优化尺寸可控制的IoU部分，再动态补偿尺度使其平均量级接近普通L1。作者明确报告，直接用完整3D IoU loss会使困难定位项压倒尺寸项并导致训练崩溃。
- **D 数据、split与训练**：KITTI 7,481 labeled trainval / 7,518 hidden test；采用3,712/3,769 Chen split做开发消融，并提交official test。140 epochs，2×GTX1080Ti，batch16，Adam，初始LR `1.25e-3`，第90/120轮衰减，前5轮warmup，约9小时。训练样本方案会按深度阈值丢弃目标，默认hard `s=60m`；推理采用中心热图局部max-pool NMS，置信度低于0.2删除。无多seed、CI或独立confirmation。
- **E 类别、任务、IoU和AP**：论文使用AP_R40；Car 3D/BEV的严格IoU为0.7，Pedestrian/Cyclist为0.5，均报告Easy/Moderate/Hard。它是单目3D/BEV，不是本项目的RGB 2D Pedestrian/Cyclist Moderate macro AP_R40；即使都写AP40，也禁止跨任务相减。
- **F 绝对结果、逐项消融与效率**：official test Car AP3D为`17.23/12.26/10.29`，APBEV为`24.79/18.89/16.00`，AOS为`93.46/90.23/80.11`，单帧40ms。val严格IoU .7的Car AP3D为`17.45/13.66/11.68`，对MonoPair的同表差为`+1.17/+1.36/+1.26`；但宽松IoU .5时为`55.41/43.42/37.81`，相对MonoPair仅`+0.03/+1.03/-0.18`，APBEV@.5甚至`-0.33/-0.76/-0.03`。累计消融的Moderate AP3D为：baseline `11.12`，加投影3D中心`12.65`（+1.53），再加IoU-oriented尺寸损失`12.97`（+0.32），再加距离采样`13.66`（+0.69），总计+2.54；这是3D累计改造，不是可搬到2D的单模块收益。论文报告25 FPS；未给参数、FLOPs、seed或CI。
- **G Ped/Cyc、small/far与分母证据**：official test Pedestrian Moderate APBEV/AP3D为`6.96/6.55`，略低于MonoPair的`7.04/6.68`；Cyclist为`3.28/2.66`，高于MonoPair的`2.87/2.12`，但作者明确提示两类样本少、波动大。作者没有给2D small AP或独立far AP。28,742个训练目标中，距离大于60m只有1,301个（4.5%），大于65m只有767个（2.7%）；默认策略直接丢弃>60m目标，因此总体AP提高不等于远目标检测改善。
- **H 失败、局限与替代解释**：Table 8显示距离阈值高度敏感：baseline Easy/Moderate/Hard=`16.12/12.97/10.99`；hard 40m变为`14.25/11.25/9.63`，全面退化；hard 60m才到`17.45/13.66/11.68`。soft `(40,1)`也只有`14.50/11.74/9.95`，而`(60,1)`为`17.50/13.54/11.32`。这说明作者在同一val上选到了有利阈值，且没有报告被删除far目标的no-harm。Table 1、正文和Fig.1还存在基线`11.12`与`9.97`、投影中心替换`23.90`与`18.97`等内部数字不一致；本笔记保留冲突，不替作者暗自选值。
- **I 与本项目逐项区别和原创撞车**：误差替换诊断、定位瓶颈分解、难任务loss可能压倒共享表征、针对单一可控误差做隔离目标、保留上游2D辅助任务，均已有直接先例。“多任务定位损失负交互”“梯度隔离”“IoU-oriented回归”不能声称首创。它不含YOLO P2、fusion或DCLI，不能证明本项目根因；但其宽/严IoU结论提示我们先判断plain-P2的`-1.5918 AP`来自coverage、localization还是ranking，再决定是否碰回归损失。
- **J 可借鉴思想与边界**：可借鉴GT替换只作oracle诊断、同时报告宽松/严格IoU、把损失限制在唯一可控变量、保留2D语义辅助、对far删除做反事实no-harm。必须引用MonoDLE的dimension-aware loss、距离采样和定位误差分析；不复制其完整3D CenterNet实现。尤其不能通过删除far样本“做出总体正数”，因为这与本项目改善far/small的目标相反。
- **K 可证伪项目假设与最小实验**：H0：plain-P2相对P3–P5在低IoU覆盖上也没有增加，退化主要是无效/噪声候选；H1：P2确实增加small/far的低IoU coverage，但定位或置信排序把新增覆盖抵消。最小实验只复用固定371 development预测，按Ped/Cyc×small/far报告max-recall@IoU .3/.5/.7、matched center/size误差、FP/FN、score calibration和image-paired CI。只有“低IoU覆盖正、严格IoU或排序负”成立，才允许一个定位/可靠性screen；否则停止DCLI式修补并转查assignment/receptive field。
- **L 原始页表定位**：问题与oracle替换见pp.1–3/Fig.1/Table 1；像素中心偏差到米级误差见p.4/Table 2；投影中心、dimension-aware loss和距离采样见pp.4–6；official test和val多IoU见pp.6–7/Tables 3–5；累计消融、2D辅助与距离阈值见pp.7–8/Tables 6–8。**证据结论**：MonoDLE最可借鉴的是“先定位误差，再做唯一干预”；其+2.54是3D累计消融，且部分来自删除远样本，绝不能当成本项目可预期提升。

- 2026-08-12：完成K14 MonoDLE A–L卡。它同时提供定位/覆盖分解模板和一个重要反例：总体AP可因忽略far样本而变高，因此本项目必须保留far/small与no-harm门，不能只追总体正数。

## K15 — MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection

- **A 元数据、原文与代码**：Renrui Zhang, Han Qiu, Tai Wang, Ziyu Guo, Ziteng Cui, Yu Qiao, Hongsheng Li, Peng Gao；ICCV 2023，pp.9155–9166；arXiv `2203.13310`。CVF正式页：https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_MonoDETR_Depth-guided_Transformer_for_Monocular_3D_Object_Detection_ICCV_2023_paper.html ，补充材料：https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhang_MonoDETR_Depth-guided_Transformer_ICCV_2023_supplemental.pdf 。官方代码：https://github.com/ZrrSkywalker/MonoDETR ，MIT License。本地主文`research-notes/source-pdfs/Zhang_2023_MonoDETR.pdf`，1,647,847 bytes，SHA256 `1669F27AB9CB72DE6736E6691FF5786828C8D51E0C84947B05202493110788C4`；supplement 1,149,550 bytes，SHA256 `42D8C1C5EF21EDD15B42F52EAB4E66442230C5554EC55F7BB7CC84DBA6EEF03E`。
- **B 研究问题与真正gap**：center-guided单目3D检测只在目标中心邻域回归3D属性，缺少场景级几何与长距离对象间深度关系。MonoDETR让query先与全局depth embedding交互，再做query间和视觉交互。对本项目更关键的次级问题是：训练早期3D尺寸、方向、深度不稳定时，是否应允许这些目标参与Hungarian assignment并决定“哪个query属于哪个GT”。
- **C 输入、网络、损失、匹配与后处理**：ResNet-50提取stride 8/16/32特征；多尺度特征统一到stride16后相加，预测foreground object-wise depth map，并用一个全局self-attention depth encoder。50个query在3-block decoder内按`depth cross-attention→self-attention→visual cross-attention→FFN`更新。深度综合direct regression、几何深度与depth-map采样，并用Laplacian aleatoric loss。最关键的隔离是Hungarian matching只用稳定的`L2D=2Lclass+10Lxy3D+5Llrtb+2LGIoU`；尺寸、方向、深度组成的`L3D`只在匹配完成后优化。推理阈值0.2，无anchor、无NMS。
- **D 数据、split与训练**：KITTI 7,481 trainval / 7,518 hidden test；3,712/3,769 Chen split用于消融，最终提交official test。单RTX3090，195 epochs，batch16，AdamW，LR `2e-4`，第125/165轮衰减；3个visual encoder blocks、1个depth encoder block、3个decoder blocks、256维、50 queries。训练删除深度<2m或>65m的目标，而depth map范围又有限；无far no-harm、多seed、paired CI或独立confirmation。
- **E 类别、任务、IoU和AP**：主指标是40 recall positions的AP3D/APBEV，Car通常IoU .7；supplement给Pedestrian/Cyclist official test AP3D@.5。它不是本项目RGB 2D AP_R40。相同“AP40”标签不代表同任务、同IoU、同分母或同evaluator，禁止与本项目96/95数字相减。
- **F 绝对结果、消融、负交互与成本**：official test Car AP3D=`25.00/16.47/13.58`，APBEV=`33.60/22.11/18.60`；val AP3D=`28.84/20.61/16.38`。这些相对second-best的差是跨论文排行榜，不是同代码因果比较。完整20.61对去掉整个depth-guided transformer的15.15为+5.46，但同时删除深度预测、transformer和全局聚合，不能称单模块提升；去掉depth guidance为17.81（+2.80）。depth encoder用global attention为20.61，两个3×3卷积18.36，甚至略低于完全无encoder的18.38；相同模块仅改变decoder顺序，Moderate从20.61降到19.28/18.85/18.41，最大差2.20。最关键的supplement Table 5：matching仅`L2D`为20.61；加3D size降到19.21（-1.40），加orientation降到18.63（-1.98），加入depth或完整`L3D`直接训练崩溃。RTX3090 batch1为38ms、62.12 GFLOPs；参数量unknown。
- **G Ped/Cyc、small/far和波动**：official test AP3D@.5：Pedestrian=`12.54/7.89/6.65`，Cyclist=`7.33/4.18/2.92`。这是final system的跨方法结果，不是own-baseline Ped/Cyc消融。论文声称多尺度/深度有助小和远目标，却没有small/far AP、距离分层CI或no-harm；反而训练排除>65m目标。官方仓库明确警告KITTI单目训练约有`±1 AP3D`随机波动，并公开Moderate 20.83/20.64/20.14三次结果，因此0.6–0.9 AP级单点差不能自动视为稳定。
- **H 失败、局限与替代解释**：更多结构不一定更好：4-block visual encoder、4-block decoder和更宽FFN均低于轻量配置；foreground LID比dense LID只高0.76，小于仓库自报波动；训练数据少是作者给出的解释。所有结构选择在同一3,769 val完成，无独立confirmation。训练剔除近/远样本、主结果集中Car、版本从camera-ready到后续stable repo有漂移；论文也未给assignment错误率或多seed机制证据。
- **I 与本项目逐项区别和原创撞车**：MonoDETR没有YOLO P2、PAN/FPN gate、DCLI或局部复检。它的多尺度相加、foreground reliability/depth cue、uncertainty和任务顺序均为公开先例。最直接的撞车是**不稳定定位目标不能取得共享assignment的语义所有权**：将其加入matching会降1.40–1.98 AP甚至崩溃。因此stop-gradient、稳定匹配、task-specific adapter或“先分配后校准”本身不能声称原创。我们只可能研究特定的P2 fusion×DCLI共享可靠性路径，并用自己的时序/梯度/样本分配证据证明。
- **J 可借鉴思想与边界**：可借鉴“稳定分支决定assignment/gate，不稳定分支只学习已分配样本”、记录assignment churn、比较同组件不同顺序、在小数据集上优先轻量结构。必须引用MonoDETR的foreground depth、LID、depth-first decoder、Laplacian深度、仅L2D匹配等思想。不要移植完整DETR/深度transformer：它同时改变backbone、head、assignment和195轮训练，无法回答当前唯一因果问题。
- **K 可证伪项目假设与最小实验**：H0：DCLI calibration是否回流到共享P2/fusion语义参数与`AB-B<0`无关；H1：DCLI改变了fusion gate或共享anchor的样本所有权，造成assignment churn/负梯度，隔离后两项机制指标与`AB-B`应同时改善。最小screen仅比较现有AB与一个`calibration stop-gradient + tiny task adapter`的AB-isolated；固定3341/371、初始化、seed、30 epochs、预算，不加attention或新neck。训练期记录P2–P5 assignment Jaccard/churn和三路gradient cosine/norm；结果报告AB-B、Ped/Cyc macro、small/far、FP/FN、宽/严IoU与延迟。只有机制指标和AP同时按预注册方向改善才进入三seed；单独AP正或cosine好都不构成因果结论。
- **L 原始页表定位**：架构与depth transformer见正式论文pp.9157–9160；KITTI设置与主结果见pp.9160–9161/Tables 2–3；完整深度组件、encoder、decoder顺序、depth representation与位置编码见pp.9161–9163/Tables 5–9。Ped/Cyc见supplement p.4/Table 3；三路depth、matching负交互和容量消融见supplement pp.4–5/Tables 4–6；depth error见supplement p.7/Table 7。**证据结论**：MonoDETR不能给本项目带来可相加的+5.46；它提供的是一个强因果约束——不稳定定位信号若控制共享分配会伤害或崩溃，故只允许测试一个最小“语义所有权隔离”假设。

- 2026-08-12：完成K15 MonoDETR A–L卡，并独立核对主文Tables 2–9、supplement Tables 2–7及官方仓库的约±1 AP警告。15篇主清单现已逐篇完成；下一步只做15行协议矩阵、引用注册表与双稿差异合并，不再追加无边界文献。
