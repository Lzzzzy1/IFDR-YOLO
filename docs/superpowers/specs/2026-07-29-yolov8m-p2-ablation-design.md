# YOLOv8m-P2 单变量消融设计

日期：2026-07-29  
状态：已选择方案 A，待书面评审  
适用版本：Ultralytics 8.4.98

## 1. 目的与研究定位

本阶段在已验收的 YOLOv8m 基线上增加 P2/4 检测尺度，使 Detect 从
P3/8、P4/16、P5/32 扩展为 P2/4、P3/8、P4/16、P5/32。目标是建立一个
可复现、可单独消融的标准 P2 对照，为后续自适应双向融合和改进 IoU 损失
提供共同结构基础。

Ultralytics 8.4.98 已提供标准 `yolov8-p2.yaml`。因此，本阶段不把“增加
P2”表述为论文方法创新，只把它作为老师方向中的基础工程改造和实验因子。
论文创新性必须来自后续经过先验检索、退化诊断和严格消融支持的机制。

本设计替代
`docs/superpowers/specs/2026-07-28-yolov8m-p2-bifpn-wiou-design.md`
中关于 P2 的旧数据、旧划分和旧训练入口描述。可信数据与评测口径继续以
Phase 1、Phase 2A 的已验收结果为准。

## 2. 已确认事实

本机安装的 Ultralytics 8.4.98 官方 P2 配置实测结果：

| 项目 | 结果 |
|---|---|
| Detect 输入数 | 4 |
| stride | `[4.0, 8.0, 16.0, 32.0]` |
| KITTI `nc=3` 参数量 | 25,052,620 |
| state-dict 项数 | 581 |
| 官方无约束 `.load(yolov8m.pt)` 匹配项 | 319 |

无约束加载匹配到模型层 0、1、2、3、4、5、6、7、8、9、12、15、18、
19、21。其中 18、19、21 在基线和 P2 模型中语义不同，尽管匹配项主要是
形状相同的标量状态，也不能依赖这种偶然匹配。因此本项目不直接使用
Ultralytics 的全模型名称/形状交集作为科研初始化策略。

## 3. 备选方案与选择

### 3.1 方案 A：官方 P2-P5 拓扑，语义安全迁移

项目内保存一份与 Ultralytics 8.4.98 官方 `yolov8-p2.yaml` 的 `m` 尺度
计算图一致的展开版模型 YAML。只迁移基线中语义保持一致的模型层 0–15，
P2 扩展和新的自底向上路径使用框架默认初始化。

优点：

- 对应公开、可复现的标准 P2 实现；
- Detect 四尺度和 PAN 路径完整；
- 初始化边界可审计，不会因层编号碰巧相同而误加载；
- 适合作为后续融合模块的结构基座。

该方案已由用户选择。

### 3.2 未采用：旁路式 P2

在原 P3-P5 Head 旁增加 P2 输出，不让 P2 回流到 P3-P5。它更接近“只多
一个输出”，但不是标准 P2 PAN-FPN，且不适合作为后续双向融合的共同基座。

### 3.3 未采用：动态 P2

在本阶段同时加入门控、注意力或动态路由。该方案会把“尺度增加”和“融合
机制”混为一个变量，无法形成可信消融，因此留到后续独立设计。

## 4. 模型结构

Backbone 完全保持 YOLOv8m 原结构，模型层 0–9 不变：

- P2：Backbone 第 2 层输出，4 倍下采样；
- P3：Backbone 第 4 层输出，8 倍下采样；
- P4：Backbone 第 6 层输出，16 倍下采样；
- P5：Backbone 第 9 层输出，32 倍下采样。

Head 采用官方 P2 计算图：

1. P5 自顶向下融合 P4；
2. 新 P4 自顶向下融合 P3；
3. 新 P3 自顶向下融合 Backbone P2；
4. P2 依次下采样并经过 P3、P4、P5 的 PAN 融合；
5. 一个 Detect 模块同时接收 P2、P3、P4、P5。

项目模型文件命名为 `models/kitti-p2-m.yaml`。它不依赖 Ultralytics 根据
文件名猜测尺度，而是把 `scales.m=[0.67, 0.75, 768]` 展开成实际重复次数
和通道数。这样可以避免 `yaml_model_load()` 把带 `yolov8m` 的本地文件名
规范化到环境内同名通用 YAML，保证实际加载的是项目内已校验文件。

展开只消除复合缩放，不改变计算图：Backbone 实际通道为
48、96、192、384、576，C2f 实际重复次数为 2、4、4、2；Head 中 C2f
重复次数均为 2。项目 YAML 直接固定 `nc=3`，使初始化、结构验收和训练使用
同一个 KITTI 计算图，不依赖训练器二次覆盖分类头。YAML 同时记录官方
AGPL-3.0 来源、上游版本和上游文件 SHA256：

运行时必须把完成 306 项迁移的同一个底层 `torch.nn.Module` 直接交给
版本锁定的 Ultralytics trainer。不能调用 YAML handle 的高层
`YOLO.train()`，因为该路径会在 trainer 内重建模型并丢失内存中刚迁移的
权重。

```text
ba54363e9f283e8f60b0fa0843eb37115093338d6f9273ccd4f74f60639754f4
```

项目配置另外记录展开版本地文件的实际 SHA256。

## 5. 初始化策略

P2 实验使用与基线相同的官方 `yolov8m.pt`：

```text
SHA256 =
5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5
```

允许迁移的目标键必须同时满足：

1. 键名符合 `model.<layer_index>.*`；
2. `0 <= layer_index <= 15`；
3. 源张量和目标张量形状完全一致；
4. 目标键真实存在；
5. Ultralytics 版本严格等于 8.4.98。

层 0–9 是完全相同的 Backbone；层 10–15 包含无参数的上采样/拼接和语义
相同的 P4、P3 自顶向下 C2f，其中实际可迁移参数位于 12、15。按当前锁定
版本，预期迁移 306 个 state-dict 项。数量、键集合或形状任一不符时必须在
训练开始前失败，不能退回无约束加载。

运行目录保存 `initialization.json`，至少记录：

- 架构文件和 SHA256；
- 预训练权重和 SHA256；
- 迁移策略名称与最大层号；
- 已迁移键的有序列表、数量和形状；
- 未迁移项数量；
- Ultralytics 版本。

## 6. 实验配置和现有流水线接入

新增正式配置：

```text
configs/experiments/kitti_yolov8m_p2_s17.yaml
```

配置继续使用当前严格解析器，并增加可选的 `initialization` 节：

```yaml
experiment:
  dataset: kitti
  model: yolov8m
  variant: p2
  seed: 17

paths:
  model: models/kitti-p2-m.yaml
  # 其余数据路径与 baseline 完全一致

initialization:
  pretrained: yolov8m.pt
  pretrained_sha256: 5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5
  strategy: semantic_prefix
  max_layer: 15
  expected_items: 306
```

`paths.model_sha256` 仍是必填字段；模型 YAML 随实现提交后，使用其实际
SHA256 固定配置值，不在设计阶段预先虚构哈希。

基线配置不包含 `initialization`，保持现有 `.pt` 直接加载行为。解析器要求
`initialization` 要么完整出现，要么完全缺失，不接受部分字段或未知字段。

现有 `dry-run / smoke / full` 入口保持不变：

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_p2_s17.yaml `
  --mode <dry-run|smoke|full>
```

虽然脚本沿用历史名称 `train_baseline.py`，其职责是受控实验编排，不在本阶段
进行无收益重命名。运行 ID 中的 variant 为 `p2`，因此不会与 baseline 冲突。

## 7. 训练与评测控制

P2 和 baseline 必须使用相同的：

- Chen common split：3712 train / 3769 val；
- 三类映射与真实尺寸 YOLO 标签；
- 300 epoch、`imgsz=640`、SGD、学习率和数据增强默认值；
- 正式有效 batch；
- seed 17，确认实验再使用 seed 42、3407；
- 推理阈值和 KITTI AP40 实现。

本地 smoke 固定为 16 train / 16 val、1 epoch、`imgsz=320`、`batch=2`、
`workers=0`、`amp=False`，只验证链路，不作为论文结果。

正式报告同时给出参数量、GFLOPs、峰值显存和推理延迟。官方 P2 Head 的参数
分配与基线不同，因此不能把 P2 实验描述为“只增加参数”；论文只声明计算图
增加了 P2 尺度，并如实报告整体复杂度变化。

## 8. 组件边界

计划新增或修改：

```text
models/
└── kitti-p2-m.yaml              # 固定的官方 m 尺度 P2-P5 展开计算图
ifdr_yolo/models/
├── __init__.py
└── initialization.py            # 语义安全权重筛选与初始化清单
ifdr_yolo/experiments/
├── config.py                    # 可选 initialization 严格配置
├── provenance.py                # 两组文件哈希预检
├── ultralytics_runtime.py       # 构建 YAML 模型并应用初始化
└── baseline.py                  # 保存 initialization.json
configs/experiments/
└── kitti_yolov8m_p2_s17.yaml
tests/
├── test_model_initialization.py
├── test_p2_model.py
└── 现有实验测试的增量用例
```

`initialization.py` 只负责选择、验证和加载张量，不负责训练。Ultralytics
适配器负责延迟构建模型，实验编排器只传递配置并保存清单。模型 YAML 不包含
项目 Python 自定义模块，因此 P2 权重可由锁定版本的原生 Ultralytics 加载。

## 9. 错误处理

以下情况在创建正式运行或进入训练前失败：

- 模型 YAML、预训练权重或数据不存在；
- 任一记录的 SHA256 不匹配；
- Ultralytics 不是 8.4.98；
- 模型不是四输入 Detect；
- stride 不是 `[4, 8, 16, 32]`；
- `nc=3` 模型参数量不是 25,052,620；
- 迁移项不等于 306；
- 发现迁移键来自 15 之后的层；
- CUDA 正式训练请求无法满足；
- 运行目录已存在。

失败继续使用现有状态机保留阶段、异常类型、消息和已经生成的产物。

## 10. 测试与验收

严格执行 RED–GREEN–REFACTOR，至少覆盖：

1. `initialization` 完整配置可解析；
2. 缺字段、未知字段、非法哈希和非法层号被拒绝；
3. 基线旧配置继续可解析；
4. 纯函数只选择允许层且要求形状一致；
5. 语义不同但键名/形状偶然一致的层不会迁移；
6. 锁定版本真实模型迁移项恰为 306；
7. P2 模型 Detect 输入数为 4；
8. stride 为 `[4, 8, 16, 32]`；
9. `nc=3` 参数量为 25,052,620；
10. 320×320 虚拟输入前向传播产生四尺度训练输出；
11. dry-run 同时校验 YAML 和预训练权重哈希；
12. 真实 KITTI smoke 产生 `best.pt`、16 个预测文件、AP40 和
    `initialization.json`；
13. 现有 101 项测试继续通过。

P2 阶段通过条件：

- 所有自动化测试为绿色；
- 真实 GPU smoke 状态为 `complete`；
- 正式数据目录无新缓存和内容变化；
- smoke 指标明确标记为非科研结果；
- 代码只做本地提交，按用户要求暂不再次推送 GitHub。

## 11. 非目标

本阶段不实现：

- 自适应双向特征融合；
- WIoU、Inner-IoU 或其他回归损失；
- 新 Backbone、注意力模块或动态路由；
- 300 epoch 正式训练；
- 论文性能提升结论；
- 删除或修改 Phase 1、Phase 2A 的可信数据和基线产物。
