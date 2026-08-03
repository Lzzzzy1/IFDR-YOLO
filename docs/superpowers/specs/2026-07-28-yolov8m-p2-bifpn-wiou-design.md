# YOLOv8m-P2-BiFPN-WIoU 科研改进设计

## 1. 项目背景

本项目以 Ultralytics YOLOv8m 为基线，在 KITTI 二维目标检测数据集上检测 `Car`、`Pedestrian` 和 `Cyclist`。当前数据集包含 6732 张训练图像和 749 张验证图像，基线模型已经完成 300 个 epoch 的训练。

当前 PyCharm 项目环境为：

- Python：`D:\ana\envs\yolo\python.exe`
- Ultralytics：8.4.98
- PyTorch：2.5.1+cu121
- 基线权重：`kitti_project/yolov8m.pt`
- 数据配置：`kitti_project/kitti.yaml`

KITTI 中的远距离行人和骑行者在图像中占用像素较少。YOLOv8 默认的 P3、P4、P5 检测尺度经过至少 8 倍下采样，可能丢失浅层的边缘、轮廓和位置信息。本设计从检测尺度、特征融合和边界框回归损失三个方面改进基线。

## 2. 目标与验收标准

项目需要实现以下功能：

1. 将检测输出从 P3、P4、P5 扩展为 P2、P3、P4、P5，检测步长为 `[4, 8, 16, 32]`。
2. 使用带可学习归一化权重的双向多尺度特征融合结构。
3. 使用 WIoU v3 替换 YOLOv8 边界框回归中的 CIoU 部分，同时保留 DFL、分类损失和 Task-Aligned Assigner。
4. 保留独立开关，使 P2、双向融合和 WIoU 能够分别进行消融实验。
5. 不直接修改 Conda 环境中的 Ultralytics 安装文件。
6. 在当前 PyCharm 环境中通过单元测试、模型构建、虚拟输入前向传播、损失反向传播和最小 KITTI 冒烟训练。

模型构建验收条件：

- Detect 层接收四个尺度的特征图。
- Detect stride 等于 `[4, 8, 16, 32]`。
- 所有自适应融合节点包含可训练参数。
- 每个融合节点的有效权重均为非负数且总和接近 1。
- WIoU v3 对合法预测框输出有限值，并能产生有限梯度。
- 训练模式会更新 WIoU 的 IoU 损失运行均值，验证模式不会更新。

## 3. 方案选择

### 3.1 采用方案：项目内扩展

所有科研模块、模型 YAML、训练入口和测试均保存在 `kitti_project` 中。训练入口在构建模型前，将项目内的自适应融合类和 WIoU 边界框损失注册到 Ultralytics 8.4.98 的运行时解析器中。

该方案具有以下优点：

- 不污染 `site-packages`。
- 重新安装环境时不会丢失科研代码。
- 改进模块与第三方源码边界清楚，便于解释和复现。
- 可以通过一个命令切换消融实验组合。

该方案依赖 Ultralytics 8.4.98 的模型解析和损失接口，因此训练入口启动时必须检查版本；版本不匹配时给出明确错误，防止静默使用不兼容实现。

### 3.2 未采用方案

- 复制完整 Ultralytics 源码：便于任意修改，但项目体积大，第三方代码和科研代码不易区分。
- 直接修改 `site-packages`：实现速度快，但不可复现，换环境后修改会丢失。

## 4. 模型结构

### 4.1 Backbone

保留 YOLOv8m 原始 Backbone，使用以下四个输出：

- P2：4 倍下采样，保留较丰富的空间细节。
- P3：8 倍下采样。
- P4：16 倍下采样。
- P5：32 倍下采样，包含较强的高级语义。

P2 的加入只改变 Neck 和 Detect，不修改 YOLOv8m Backbone 的 C2f 与 SPPF 结构。

### 4.2 通道对齐

P2、P3、P4、P5 首先经过独立的 `1×1 Conv` 对齐到相同的基础通道数。模型 YAML 继续使用 YOLOv8 的 `m` 复合缩放系数，因此基础通道会按照 YOLOv8m 的宽度系数自动缩放。

对齐操作放在融合节点外部，使自适应融合模块只承担“对相同形状特征进行加权求和”的职责，便于测试和复用。

### 4.3 自顶向下路径

自顶向下路径依次执行：

1. P5 上采样到 P4 分辨率，与 P4 横向特征进行自适应融合。
2. 融合后的 P4 上采样到 P3 分辨率，与 P3 横向特征进行自适应融合。
3. 融合后的 P3 上采样到 P2 分辨率，与 P2 横向特征进行自适应融合。

每次融合后使用 C2f 进行局部特征提取。

### 4.4 自底向上路径

自底向上路径依次执行：

1. 自顶向下得到的 P2 通过步长为 2 的卷积下采样，与 P3 横向特征和自顶向下 P3 特征融合。
2. 新 P3 下采样后，与 P4 横向特征和自顶向下 P4 特征融合。
3. 新 P4 下采样后，与 P5 横向特征融合。

最终的 P2、P3、P4、P5 特征共同传入一个四尺度 Detect 层。这里不是两个互相独立的检测头，而是一个接收四个输入尺度的 Detect 模块。

### 4.5 自适应加权融合

对于同一分辨率、同一通道数的输入特征 `F_i`，融合输出为：

\[
\hat w_i = \operatorname{ReLU}(w_i)
\]

\[
F_{out} =
\frac{\sum_i \hat w_i F_i}
{\epsilon + \sum_i \hat w_i}
\]

其中 `w_i` 为可学习标量，初始化为 1，`epsilon` 默认为 `1e-4`。该形式使用快速归一化融合，保持权重非负并避免分母为零。

融合模块会验证：

- 至少包含两个输入特征。
- 所有输入形状完全一致。
- 实际输入数量与初始化时声明的数量一致。

如果验证失败，模块抛出包含实际形状和期望输入数的 `ValueError`，避免错误广播造成静默训练问题。

## 5. WIoU v3

### 5.1 基础定义

设：

\[
L_{IoU}=1-IoU
\]

预测框和目标框中心距离的平方为 \(\rho^2\)，最小外接框对角线平方为 \(c^2\)。WIoU v1 的距离注意项为：

\[
R_{WIoU} =
\exp\left(\frac{\rho^2}{(c^2)^*+\epsilon}\right)
\]

其中上标 `*` 表示分母从计算图中分离。WIoU v1 为：

\[
L_{WIoU-v1}=R_{WIoU}L_{IoU}
\]

### 5.2 动态非单调聚焦

维护 IoU 损失的指数运行均值 \(\overline{L_{IoU}}\)，初始值为 1。训练模式下使用作者实现中的动量 `0.01` 更新：

\[
\overline{L_{IoU}} \leftarrow
0.99\overline{L_{IoU}}+0.01\operatorname{mean}(L_{IoU})
\]

离群程度为：

\[
\beta =
\frac{L_{IoU}^{*}}
{\overline{L_{IoU}}+\epsilon}
\]

WIoU v3 的非单调聚焦增益为：

\[
r =
\frac{\beta}
{\delta\alpha^{\beta-\delta}}
\]

使用作者实现的 `alpha=1.7`、`delta=2.7`。最终损失为：

\[
L_{WIoU-v3}=rL_{WIoU-v1}
\]

实现中会对宽高、并集面积、最小外接框对角线和运行均值使用 `epsilon` 保护，避免退化框产生 NaN 或 Inf。

### 5.3 与 YOLOv8 损失的关系

Ultralytics 8.4.98 的检测回归损失由 IoU 损失和 DFL 组成。本项目只将：

\[
1-CIoU
\]

替换为：

\[
L_{WIoU-v3}
\]

以下部分保持不变：

- DFL 的计算方法和权重。
- 分类 BCE 损失。
- Task-Aligned Assigner。
- 前景样本的 target score 加权方式。
- `box`、`cls` 和 `dfl` 三项超参数增益。

WIoU 的运行均值注册为模型 buffer，使其可以随 checkpoint 的 `state_dict` 保存和恢复。

## 6. 项目文件与职责

新增文件：

```text
kitti_project/
├── models/
│   ├── yolov8m-p2.yaml
│   └── yolov8m-p2-bifpn.yaml
├── research_yolo/
│   ├── __init__.py
│   ├── fusion.py
│   ├── wiou.py
│   ├── integration.py
│   └── model_factory.py
├── tests/
│   ├── test_fusion.py
│   ├── test_wiou.py
│   ├── test_integration.py
│   └── test_model_build.py
├── train_research.py
└── verify_research.py
```

各组件职责：

- `fusion.py`：实现归一化的自适应特征融合。
- `wiou.py`：实现 WIoU v3 核心计算和兼容 YOLOv8 的 BboxLoss。
- `integration.py`：执行版本检查，并将本地类注册到 Ultralytics 运行时。
- `model_factory.py`：根据实验配置构建模型，安全加载 YOLOv8m Backbone 预训练权重。
- `models/*.yaml`：定义 P2-P5 PAN-FPN 和 P2-P5 加权双向融合结构。
- `train_research.py`：科研训练命令行入口。
- `verify_research.py`：无需完整训练即可执行模型结构和反向传播检查。
- `tests/`：使用 Python 标准库 `unittest`，避免增加测试依赖。

## 7. 预训练权重策略

自定义模型从 YAML 构建后，只迁移官方 `yolov8m.pt` 中 Backbone 第 0 至第 9 层且名称、形状均匹配的参数。

不直接调用无筛选的整模型加载，原因是新 Neck 的层编号和语义已经改变；无筛选加载可能把原 Neck 权重误载入形状恰好相同、但功能不同的新层。新的通道对齐层、融合层、Neck 和四尺度 Detect 层使用框架默认初始化。

训练入口会输出：

- 成功迁移的 Backbone 张量数量。
- 因名称或形状不匹配而未迁移的张量数量。
- 当前模型配置和 IoU 损失类型。

## 8. 训练和消融接口

训练入口将模型结构与损失类型分开配置：

```text
--variant baseline | p2 | p2_bifpn
--iou-loss ciou | wiou
```

默认参数为：

```text
--variant p2_bifpn
--iou-loss wiou
--epochs 300
--imgsz 640
--batch 4
--optimizer SGD
--lr0 0.01
--device 0
```

典型消融组合：

| 实验 | variant | iou-loss | P2 | 自适应双向融合 | WIoU v3 |
|---|---|---|---:|---:|---:|
| 基线 | baseline | ciou | × | × | × |
| P2 | p2 | ciou | ✓ | × | × |
| P2-BiFPN | p2_bifpn | ciou | ✓ | ✓ | × |
| P2-WIoU | p2 | wiou | ✓ | × | ✓ |
| 完整模型 | p2_bifpn | wiou | ✓ | ✓ | ✓ |

数据配置会在内存中解析，并将数据根目录规范化为 `kitti_project/kitti` 的绝对路径，避免 PyCharm 工作目录不同导致找不到数据。

输出目录采用：

```text
kitti_project/runs/research/<实验名称>
```

实验名称包含结构、损失、图像尺寸和随机种子，减少结果被意外覆盖的风险。默认 `exist_ok=False`。

## 9. 测试设计

### 9.1 融合模块

1. 两个相同特征、初始权重相同时，输出应等于两者近似平均值。
2. 有效权重必须非负且总和接近 1。
3. 反向传播后，每个融合权重均应获得有限梯度。
4. 输入数量错误或形状不一致时必须抛出 `ValueError`。

### 9.2 WIoU v3

1. 完全重合框的损失应接近 0。
2. 偏移框的损失应为正且有限。
3. 损失反向传播后预测框梯度应为有限值。
4. 训练模式会更新 `iou_mean`。
5. 验证模式保持 `iou_mean` 不变。
6. 退化尺寸和不相交框不得产生 NaN 或 Inf。
7. 兼容 BboxLoss 应返回标量 IoU 损失和 DFL 损失。

### 9.3 运行时接入

1. Ultralytics 版本不是 8.4.98 时拒绝注册并给出明确说明。
2. 选择 `ciou` 时不替换原始 BboxLoss。
3. 选择 `wiou` 时，新建 criterion 使用项目内 WIoU BboxLoss。
4. 重复注册不会重复修改或造成状态冲突。

### 9.4 模型结构

1. P2 模型能够从 YAML 成功构建。
2. P2-BiFPN 模型能够从 YAML 成功构建。
3. 四尺度 Detect stride 为 `[4, 8, 16, 32]`。
4. 虚拟输入前向传播输出四个尺度。
5. 合成标签上的完整检测损失能够反向传播。
6. Backbone 迁移只加载第 0 至第 9 层。

### 9.5 冒烟训练

全部单元测试通过后，在 KITTI 上运行 1 个 epoch 的小比例训练：

- `imgsz=320`
- `batch=1`
- `workers=0`
- 固定随机种子
- 使用少量训练样本

冒烟训练只验证数据读取、模型、损失、优化器、验证和 checkpoint 保存的完整链路，不作为科研结果。

## 10. 错误处理

- 找不到 `yolov8m.pt`、`kitti.yaml` 或数据目录时，在训练开始前终止并输出绝对路径。
- CUDA 不可用但指定 `device=0` 时，明确报错，不自动切换 CPU，以免用户误以为正在使用 GPU。
- 模型 YAML 与融合模块输入数量不一致时，在第一次前向传播立即终止。
- WIoU 出现非有限值时，验证脚本报告输入框和中间量；正式训练依靠数值保护避免该情况。
- 加载 checkpoint 进行预测或恢复训练时，入口必须先注册本地科研组件。

## 11. 非目标

本轮不包含以下内容：

- 不修改 KITTI 数据划分和类别映射。
- 不设计新的 Backbone。
- 不同时实现 Inner-IoU 和 Powerful-IoU。
- 不自动运行完整 300 epoch 训练。
- 不声称改进模型一定优于基线；性能结论必须由同一数据划分和超参数下的消融实验决定。

## 12. 参考资料

- Tan, M., Pang, R., & Le, Q. V. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.  
  https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html
- Tong, Z., Chen, Y., Xu, Z., & Yu, R. Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism. arXiv:2301.10051.  
  https://arxiv.org/abs/2301.10051
- Wise-IoU 作者实现。  
  https://github.com/Instinct323/Wise-IoU/blob/v2/iou.py

