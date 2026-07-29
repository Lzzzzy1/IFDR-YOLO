# IFDR-YOLO Phase 2A 可信基线基础设施验收报告

日期：2026-07-29  
状态：**ACCEPTED（本地训练全链路已通过，可进入 AutoDL 正式基线训练）**

## 1. 验收结论

Phase 2A 已建立受控的 YOLOv8m 基线实验入口，能够在同一配置体系下执行
`dry-run`、固定样本 `smoke` 和正式 `full` 训练。最终验收确认：

- 严格解析实验配置，未知字段、非法范围和类型错误会立即失败；
- 训练前全量复核 7481 对源图像/标签哈希、common split、派生数据和预训练权重；
- 实验目录防覆盖，状态文件使用原子写入，失败运行保留现场；
- Git 提交、环境、数据摘要、输入配置和实际生效配置均随运行保存；
- smoke 数据复制到隔离的 `tmp` 视图，Ultralytics 缓存不写入正式派生数据；
- 训练、验证、逐图预测和 KITTI AP40 评估已在真实 GPU 上完整跑通；
- 101 项自动化测试全部通过。

正式论文结果尚未产生。本报告只验收实验基础设施，不声明模型精度提升。

## 2. 代码与执行入口

验收提交：

```text
d53c8914965800fcabc49a2acae0d2e662eee879
```

分支：

```text
feature/phase2a-trusted-baseline
```

正式配置：

```text
configs/experiments/kitti_yolov8m_baseline_s17.yaml
```

真实 smoke 命令：

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode smoke `
  --device 0
```

运行 ID：

```text
20260729T064331Z-kitti-yolov8m-baseline-s17-d53c891
```

终端状态为 `complete`，本次命令墙钟时间约 86.7 秒。

## 3. 验收环境

| 项目 | 记录值 |
|---|---|
| OS | Windows 10 `10.0.26200` |
| Python | 3.11.15 |
| Python 可执行文件 | `D:\ana\envs\yolo\python.exe` |
| PyTorch | 2.5.1+cu121 |
| CUDA runtime | 12.1 |
| Ultralytics | 8.4.98 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6144 MiB |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |

最终 smoke 的实际训练参数为 1 epoch、`imgsz=320`、`batch=2`、
`workers=0`、seed 17、确定性模式、CUDA 0 和 `amp=False`。关闭 smoke
AMP 只为避免本地链路验收触发 Ultralytics 的联网模型自检；正式 `full`
配置仍为 300 epoch、`imgsz=640`、`batch=16`、`workers=8`、AMP 开启。

## 4. 数据与权重溯源

| 项目 | 数量或 SHA256 |
|---|---|
| 原始图像 | 7481 |
| 原始标签 | 7481 |
| 派生 YOLO 行 | 34856 |
| 全量复核源文件对 | 7481 |
| common train | 3712 |
| common val | 3769 |
| train 原始字节 SHA256 | `b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb` |
| val 原始字节 SHA256 | `657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86` |
| train 规范化 ID SHA256 | `e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c` |
| val 规范化 ID SHA256 | `6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8` |
| `yolov8m.pt` SHA256 | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` |

验收时 Git 记录为 `tracked_clean=true`。smoke 产生的 `train.cache` 和
`val.cache` 只存在于 `tmp/smoke-kitti/labels/`；正式
`data/processed/kitti_yolo_v2/labels/` 中不存在这两个缓存文件。

## 5. 自动化测试

执行：

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

结果：

```text
Ran 101 tests
OK
```

覆盖内容包括配置校验、数据重建、全量审计、KITTI 解析、AP40 规则与参考
实现对照、实验状态机、溯源、隔离 smoke 视图、Ultralytics 适配器、训练
编排和 CLI。

## 6. Smoke 产物审计

运行目录：

```text
runs/20260729T064331Z-kitti-yolov8m-baseline-s17-d53c891
```

关键产物全部存在：

- `status.json`，最终状态 `complete`；
- `config.input.yaml` 与 `config.resolved.yaml`；
- `data_manifest.json`、`environment.json`、`git_status.json` 和 `git_commit.txt`；
- `weights/best.pt` 与 `weights/last.pt`，各 51,975,947 字节；
- 16 个验证图预测文件，文件名与固定 smoke val ID 集合完全一致；
- `metrics_ap40.json`，评估器为 `ifdr_yolo.kitti_ap40`，`split_count=16`；
- Ultralytics 的 `args.yaml`、`results.csv`、曲线图和批次可视化。

曾有一次受控失败运行
`20260729T062612Z-kitti-yolov8m-baseline-s17-ed0590e`：Ultralytics 将 Python
图片列表重命名为 `image0` 等名称，项目在评估阶段拒绝了错误 ID，并把状态
记录为 `failed`。回归测试和修复已将预测 source 改为隔离目录，最终运行的
16 个 KITTI ID 全部正确。该失败现场保留用于证明失败可追溯机制有效。

## 7. Smoke-only AP40

以下数值来自 16 train / 16 val、1 epoch、320 像素的链路测试，**不是论文
结果，不得与正式基线或改进模型比较，也不得放入论文主结果表**。

| 类别 | Easy AP40 | Moderate AP40 | Hard AP40 |
|---|---:|---:|---:|
| Car | 94.6212 | 79.1087 | 74.1087 |
| Pedestrian | 2.8488 | 2.0278 | 2.0278 |
| Cyclist | 0.0000 | 0.0000 | 0.0000 |

这些数值仅证明预测坐标、置信度、逐图文件补全、原始 KITTI GT 读取和 AP40
计算链路可以协同工作。

## 8. 正式训练前约束

1. 在 AutoDL 上使用相同提交、配置、数据哈希和 Ultralytics 版本。
2. 先执行 `dry-run`，再执行 `full`；正式训练必须使用 CUDA，不允许静默回退 CPU。
3. 正式训练如因显存不足调整 batch，必须创建新配置，并让基线和所有消融实验
   使用同一有效 batch 重新运行。
4. 正式结果至少保留 `best.pt`、全部 3769 个 val 预测文件、KITTI AP40、
   Ultralytics 指标、配置、环境、Git SHA 和数据摘要。
5. 只有正式 YOLOv8m 基线完成后，才进入 P2、动态双向融合和改进 IoU 的退化
   诊断与消融实验；smoke 指标不能用于选择论文模块。

## 9. 阶段判定

Phase 2A 本地基础设施验收通过。下一里程碑是将本分支合并到 `master`、
再次运行 101 项测试并同步 GitHub，随后在 AutoDL 执行 300 epoch 的可信
YOLOv8m 正式基线。
