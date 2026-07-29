# Phase 2A 可信 YOLOv8m 基线设计

日期：2026-07-29
状态：已批准实施
上游验收：`docs/reports/phase1-acceptance.md`

## 1. 目标

本阶段只建立可复现、可审计、可自动评测的 YOLOv8m 可信基线，不加入 P2、BiFPN、退化后验或新 IoU 损失。基线必须在 Phase 1 重建的数据、固定 common split 和 KITTI AP40 上重新训练，成为后续所有结构与损失消融的共同对照。

本阶段完成后，应能用同一入口执行：

1. `dry-run`：只验证配置、环境、数据、权重和输出合同；
2. `smoke`：在固定小样本上完成训练、验证、预测和 AP40 全链路；
3. `full`：在 AutoDL 上执行 300 epoch 正式训练和 common val 评测。

## 2. 方案选择

### 2.1 采用：项目内受控封装 Ultralytics

使用项目内配置、预检、实验登记和训练后评测代码调用已安装的 Ultralytics 8.4.98。训练语义仍由原生 YOLOv8m 提供，项目代码负责冻结输入、记录运行状态、保存溯源信息并将预测接入 KITTI AP40。

优点：

- 不修改 Conda `site-packages`；
- 不复制整个 Ultralytics 源码；
- 基线保持接近官方实现；
- 后续模型变体可以复用同一实验合同；
- 环境或数据不一致时在消耗 GPU 前失败。

### 2.2 不采用：直接继续使用历史 `train.py`

历史入口没有固定 common split、数据哈希、运行目录合同、失败状态和 AP40 后处理，无法作为论文基线。

### 2.3 不采用：维护 Ultralytics 源码分叉

源码分叉对后续自定义模型可能有价值，但基线阶段会引入不必要的第三方代码维护和语义偏差。只有项目内扩展接口无法实现模型创新时，才另行设计最小、可追踪的源码镜像。

## 3. 训练口径

第一条正式基线采用：

| 项目 | 固定值 |
|---|---|
| 模型 | 官方 `yolov8m.pt` |
| 权重 SHA256 | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` |
| Ultralytics | `8.4.98` |
| 数据 | `data/processed/kitti_yolo_v2` |
| 划分 | 3712 train / 3769 val Chen common split |
| 图像尺寸 | 640 |
| epoch | 300 |
| batch | 16 |
| optimizer | SGD |
| lr0 / lrf | 0.01 / 0.01 |
| momentum | 0.937 |
| weight decay | 0.0005 |
| warmup epoch | 3.0 |
| patience | 0，不提前停止 |
| AMP | 开启 |
| deterministic | 开启 |
| seed | 首轮 17；确认实验 42、3407 |
| device | 正式训练为 CUDA 0 |
| cache | 关闭 |
| workers | 8 |

不自动降低 batch 或切换 CPU。若服务器发生 OOM，先记录失败，再创建新的、明确命名的 batch 配置；所有论文对照必须使用同一有效 batch 设置重新运行。

Ultralytics 仍会生成自身的 `args.yaml` 和 `results.csv`。项目同时保存输入配置和运行时环境，避免只依赖框架内部输出。

## 4. 预测与评测口径

训练结束后默认使用 `best.pt` 对固定 common val 推理：

- `conf=0.001`；
- `iou=0.7`；
- `max_det=300`；
- `augment=False`；
- 配置保留 `half=False`，调用框架时省略该已弃用参数，由 Ultralytics 默认使用 FP32；
- 输出 Ultralytics YOLO 文本，包含置信度；
- 每个 val ID 都必须有对应文件，允许内容为空；
- 禁止用训练 YOLO 标签替代原始 KITTI GT。

预测完成后调用 Phase 1 的 `ifdr_yolo.eval`：

1. 读取每张图真实尺寸；
2. 将归一化预测恢复为像素坐标；
3. 从原始 KITTI 标签读取 GT、ignore 与 DontCare；
4. 输出 Car/Pedestrian/Cyclist × Easy/Moderate/Hard AP40；
5. 将结果保存为 `metrics_ap40.json`。

Ultralytics mAP50 和 mAP50–95 作为辅助指标保存在框架产物中，不替代 KITTI AP40。

## 5. 实验配置

正式配置保存在：

```text
configs/experiments/kitti_yolov8m_baseline_s17.yaml
```

配置顶层字段固定为：

```yaml
schema_version: 1
experiment:
  dataset: kitti
  model: yolov8m
  variant: baseline
  seed: 17
paths:
  model: yolov8m.pt
  model_sha256: 5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5
  data: configs/data/kitti_v2.yaml
  generated_data: data/processed/kitti_yolo_v2
  raw_images: kitti_raw/training/image_2/training/image_2
  raw_labels: kitti_raw/training/label_2/training/label_2
  train_ids: configs/splits/kitti_train.txt
  val_ids: configs/splits/kitti_val.txt
training:
  epochs: 300
  imgsz: 640
  batch: 16
  workers: 8
  device: 0
  optimizer: SGD
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3.0
  patience: 0
  amp: true
  deterministic: true
  cache: false
prediction:
  conf: 0.001
  iou: 0.7
  max_det: 300
  half: false
```

路径在预检阶段相对仓库根目录解析为绝对路径，解析结果写入运行目录。未知字段、缺失字段、错误类型和非法范围均拒绝，不静默使用拼写错误的参数。

## 6. 运行标识与不可覆盖规则

运行 ID 为：

```text
<UTC时间>-<dataset>-<model>-<variant>-s<seed>-<git短SHA>
```

例如：

```text
20260729T120000Z-kitti-yolov8m-baseline-s17-034aee2
```

运行目录：

```text
runs/<experiment_id>/
├── config.input.yaml
├── config.resolved.yaml
├── data_manifest.json
├── environment.json
├── git_commit.txt
├── status.json
├── train.log
├── args.yaml
├── results.csv
├── predictions/
│   └── labels/
├── metrics_ap40.json
└── weights/
    ├── best.pt
    └── last.pt
```

同名目录存在时必须失败，禁止自动添加 `2`、`3` 后缀或覆盖旧结果。正式 `full` 模式拒绝有已跟踪未提交修改的仓库；未跟踪的本地数据和历史结果只记录名称，不作为失败条件。`smoke` 允许工作树有当前开发修改。

## 7. 预检

预检在导入 Ultralytics 和创建正式运行前完成以下检查：

1. 找到仓库根目录；
2. 在导入 Ultralytics 前设置 `YOLO_CONFIG_DIR=<repo>/tmp/yolo-config`；
3. Python、PyTorch、CUDA 和 Ultralytics 版本可读取；
4. 正式模式要求 Ultralytics 精确为 8.4.98；
5. 正式模式要求 CUDA 可用且指定设备存在；
6. `yolov8m.pt` SHA256 与配置一致；
7. 数据配置、派生数据清单和 train/val split 存在；
8. split 数量、SHA256、无交集和全覆盖通过；
9. 派生数据的 7481 个图像、标签和清单通过 Phase 1 审计；
10. 输出目录不存在；
11. 正式模式 Git 已跟踪文件无修改。

`dry-run` 在全部检查通过后停止，不训练、不创建权重。

Split 保留两种不同用途的哈希，不允许直接互比：

- `configs/splits/source.json` 的 `train_sha256`、`val_sha256` 是下载文件的原始字节哈希；
- 派生数据 `manifest.json` 的 `train_split_sha256`、`val_split_sha256` 是将每个 ID 规范化为 `<id>\n` 后计算的序列哈希。

预检分别用对应算法验证两组值，并额外比较解析后的 ID 元组。当前上游 split 文件末尾没有换行，因此原始字节哈希与规范化序列哈希不同是预期行为。

## 8. Smoke 数据视图

Smoke 不改变正式数据和 split。它从固定 split 头部选择 16 个 train ID 和 16 个 val ID，在 `tmp/smoke-kitti/` 建立隔离视图：

- 选中图像和标签的临时副本；
- 绝对图像路径列表；
- 临时数据 YAML；
- 选择 ID 与源 split SHA256。

临时视图继续保持 `images/<split>` 与 `labels/<split>` 对称结构。这样 Ultralytics 自动生成的标签缓存只会写入 `tmp`，不会污染 Phase 1 派生数据。预测使用临时验证图像目录作为 source，避免 Python 图片列表被框架重命名为 `image0` 等非 KITTI ID。临时副本不得回写正式数据。Smoke 参数固定为：

- 1 epoch；
- `imgsz=320`；
- `batch=2`；
- `workers=0`；
- `amp=False`，避免仅用于链路验收的本地 smoke 触发联网模型自检；正式训练仍开启 AMP；
- seed 17；
- 本机优先 CUDA，显式要求 CPU 时使用 CPU；
- 对 16 个 val ID 完成预测和 AP40。

Smoke 结果只验证链路，不进入论文表格。

## 9. 状态机与失败处理

`status.json` 使用原子替换写入，状态为：

```text
prepared -> running -> trained -> evaluating -> complete
```

任何异常写入：

```text
failed
```

失败记录至少包含当前阶段、异常类型、异常消息和 UTC 时间。已生成日志和 checkpoint 保留，不自动删除。只有 `complete` 且存在 `metrics_ap40.json` 的运行才允许进入结果汇总。

断点恢复不覆盖原目录；恢复命令显式接收已有 `last.pt`，并在原实验 ID 后添加新的 attempt 标识，同时记录父运行目录。

## 10. 模块边界

计划新增：

```text
ifdr_yolo/experiments/
├── __init__.py
├── config.py          # 严格解析、类型与范围验证
├── provenance.py      # Git、环境、权重与数据摘要
├── run_store.py       # 运行 ID、目录和原子状态
├── smoke_data.py      # 固定小样本数据视图
└── baseline.py        # 预检、训练、预测和 AP40 编排
scripts/
└── train_baseline.py  # CLI
tests/
├── test_experiment_config.py
├── test_provenance.py
├── test_run_store.py
├── test_smoke_data.py
└── test_baseline_cli.py
```

`config.py` 不导入 Ultralytics。`provenance.py` 只做只读采集。`run_store.py` 不理解训练框架。`baseline.py` 通过延迟导入调用 Ultralytics，使纯配置和元数据测试不依赖 GPU。

## 11. 测试与验收

必须先写失败测试，再写实现。至少覆盖：

1. 合法配置能解析为不可变对象；
2. 未知字段、缺失字段、布尔值伪装整数和非法范围被拒绝；
3. 相对路径稳定解析到仓库根目录；
4. 权重 SHA 和 split SHA 不符时预检失败；
5. 正式模式拒绝已跟踪修改；
6. 运行 ID 可重复测试且生产生成不冲突；
7. 已存在运行目录不被覆盖；
8. 状态转换合法，非法跳转被拒绝；
9. 失败状态保留异常摘要；
10. smoke ID 固定、无交集、来源可追溯；
11. `YOLO_CONFIG_DIR` 在 Ultralytics 导入前设置；
12. `dry-run` 不调用训练；
13. 预测缺文件或缺置信度时评测失败；
14. 完整 smoke 生成 `best.pt`、逐图预测和 `metrics_ap40.json`；
15. 现有 64 项 Phase 1 测试继续通过。

阶段验收命令最终应提供：

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode dry-run
```

以及：

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode smoke
```

## 12. 非目标与阶段门

本阶段不实现：

- P2 检测层；
- BiFPN 或动态融合；
- 退化干预生成器；
- WIoU、Inner-IoU 或期望 IoU；
- 三随机种子正式确认实验；
- 第二数据集。

只有以下条件全部满足，才进入退化诊断设计：

1. dry-run 与 smoke 全链路通过；
2. AutoDL 正式 baseline 完成；
3. common val 的 KITTI AP40、Ultralytics 指标、权重和逐图预测齐全；
4. 运行配置、环境、Git SHA 和数据哈希可追溯；
5. 相同提交和配置能够恢复或重新启动实验。
