# IFDR-YOLO

> 2026-09-04 archive status: this public repository is the safe, reproducible
> project archive for the PLAIN_P2/DCLI study. It contains source, frozen
> configurations, split identities, raw metric JSON/CSV, audit receipts,
> teacher-review documents, and manuscript deliverables. Licensed datasets,
> model weights, checkpoints, full prediction dumps, caches, remote-access
> material, and local environments are intentionally not distributed.

## Frozen scientific result

The formal experiment uses a disjoint KITTI-derived `fit3341 / development371`
split, paired seeds `0-4`, 30 epochs, batch 16, 640-pixel input, and the final
`last.pt` endpoint. The primary quantity is Moderate Pedestrian/Cyclist macro
AP_R40 on the internal development subset.

| Method | Five-seed mean | Standard deviation |
|---|---:|---:|
| PLAIN_P2 | 95.3886 | 0.5779 |
| DCLI | 94.3702 | 0.8473 |

The paired PLAIN_P2-minus-DCLI mean is `+1.0184` AP, with a 95% t interval of
`[-0.4055, 2.4423]`; PLAIN_P2 is higher in four of five paired seeds. Therefore
PLAIN_P2 is retained as the tested development choice, while DCLI is reported
as an exploratory negative result rather than a stable improvement. These are
internal development results, not official KITTI test-server scores. A causal
P2-versus-three-scale gain is also not claimed because a matched three-scale
control is absent.

## Repository map

- `ifdr_yolo/`, `scripts/`, `configs/`, `models/`, `tests/`: the versioned
  research implementation and infrastructure.
- `reproducibility/formal-runtime/`: the frozen formal runtime source closure,
  exact seed configurations, and supporting measurement code used by the
  five-seed evidence package.
- `evidence/formal-five-seed-20260825/`: ten canonical terminal runs, raw AP40
  metrics, epoch receipts, binding receipts, paired statistics, and explicit
  verified/not-verified boundaries. No weights or image data are included.
- `evidence/directional-screen-v213e/`: a separately labelled 15-epoch/batch-2
  local directional screen. Its frozen decision is `NO_GO`; it is not formal
  30-epoch/batch-16 evidence and cannot alter the formal conclusion.
- `paper/AIAC2026-review/`: current seven-page, anonymous, venue-specific review
  manuscript plus its format/evidence audit.
- `paper/venue-neutral-ieee/`: venue-neutral IEEE-style editable DOCX/PDF and
  audit; this is an intermediate manuscript, not a camera-ready claim.
- `paper/chapter3/` and `paper/teacher-review/`: the English chapter and the
  teacher-facing decision reports with compact evidence receipts.
- `docs/literature-review/`: project-authored literature and protocol notes;
  third-party source PDFs are not redistributed.

See `ARCHIVE_SCOPE.md` for the exact inclusion/exclusion policy and
`WEIGHTS_AND_DATA.md` for lawful data and weight reconstruction boundaries.

IFDR-YOLO 是“基于干预监督与因子化退化可靠性学习的道路小目标检测”研究项目。当前主线不是简单叠加 P2、BiFPN 和 IoU 损失，而是学习具有明确退化语义的后验，并用同一后验控制 P2–P5 尺度路由与不确定性定位。

## 当前状态

Phase 1 数据与评测基础设施已于 2026-07-29 验收通过：

- KITTI 7481 张图像；
- Chen common split：3712 train / 3769 val；
- 按真实图像尺寸重建 YOLO 标签；
- KITTI 2D AP40；
- 64 个测试；
- AP40 参考实现一致性最大差为 0。

详见 `docs/reports/phase1-acceptance.md`。

## Phase 2A 可信 YOLOv8m 基线

正式配置：

```text
configs/experiments/kitti_yolov8m_baseline_s17.yaml
```

训练前全量预检数据、split、权重、环境和 Git：

```powershell
python scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode dry-run
```

本地固定 16 train / 16 val 冒烟训练：

```powershell
python scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode smoke `
  --device 0
```

AutoDL 正式 300 epoch 训练：

```powershell
python scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_baseline_s17.yaml `
  --mode full
```

正式模式使用官方 `yolov8m.pt`、640 输入、SGD、固定 300 epoch 和
seed 17，不允许提前停止或自动降低 batch。运行目录不可覆盖，并至少保存：

```text
runs/<experiment_id>/
├── config.input.yaml
├── config.resolved.yaml
├── data_manifest.json
├── environment.json
├── git_commit.txt
├── git_status.json
├── status.json
├── predictions/labels/
├── metrics_ap40.json
└── weights/
```

只有 `status.json` 为 `complete` 且存在 `metrics_ap40.json` 的运行可以进入结果汇总。KITTI AP40 是论文主指标；Ultralytics mAP50 和 mAP50–95 仅作为工程辅助指标。

## Phase 2B P2–P5 严格消融

正式配置：

```text
configs/experiments/kitti_yolov8m_p2_s17.yaml
```

依次运行模型与数据预检、本地 16 train / 16 val 冒烟训练、AutoDL
正式 300 epoch 训练：

```powershell
python scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_p2_s17.yaml `
  --mode dry-run `
  --device 0

python scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_p2_s17.yaml `
  --mode smoke `
  --device 0

python scripts/train_baseline.py `
  --config configs/experiments/kitti_yolov8m_p2_s17.yaml `
  --mode full
```

该实验仅把检测尺度从 P3–P5 改为 P2–P5，其训练与预测配置和
seed 17 基线完全一致。模型结构、预训练权重及 306 项语义安全迁移均有
固定哈希或清单审计。P2 是来自锁定上游拓扑的消融因素，不作为论文原创
贡献；后续原创方法必须在相同实验协议下同时对比 baseline 与 P2。

## 环境

本机已验证：

```text
Python 3.11.15
PyTorch 2.5.1+cu121
Ultralytics 8.4.98
NumPy 2.4.6
Pillow 12.3.0
PyYAML 6.0.3
```

解释器：使用已激活、版本匹配的 Python 环境；下面所有命令均以
`python` 表示该解释器。

```text
python
```

## 数据约定

原始 KITTI 保持只读：

```text
<KITTI_ROOT>\training\image_2\training\image_2
<KITTI_ROOT>\training\label_2\training\label_2
```

派生数据写入：

```text
data\processed\kitti_yolo_v2
```

原始数据、派生数据、模型权重和训练运行目录均不提交到 Git。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 固定划分

划分文件已经版本化：

```text
configs/splits/kitti_train.txt
configs/splits/kitti_val.txt
configs/splits/source.json
```

重新导入下载好的上游文件：

```powershell
python scripts/import_kitti_split.py `
  --train-source 'tmp\kitti_splits\train.txt' `
  --val-source 'tmp\kitti_splits\val.txt' `
  --image-dir "$env:KITTI_ROOT\training\image_2\training\image_2" `
  --label-dir "$env:KITTI_ROOT\training\label_2\training\label_2"
```

## 重建 KITTI YOLO 数据

```powershell
python scripts/rebuild_kitti.py `
  --image-dir "$env:KITTI_ROOT\training\image_2\training\image_2" `
  --label-dir "$env:KITTI_ROOT\training\label_2\training\label_2" `
  --train-ids 'configs\splits\kitti_train.txt' `
  --val-ids 'configs\splits\kitti_val.txt' `
  --output-dir 'data\processed\kitti_yolo_v2'
```

若派生目录已经存在，只有明确确认重建时才添加：

```text
--overwrite-generated
```

## AP40 评测

预测目录使用 Ultralytics 文本格式：

```text
class_id x_center y_center width height confidence
```

运行：

```powershell
python scripts/evaluate_kitti.py `
  --prediction-dir 'path\to\predictions' `
  --label-dir "$env:KITTI_ROOT\training\label_2\training\label_2" `
  --image-dir "$env:KITTI_ROOT\training\image_2\training\image_2" `
  --split 'configs\splits\kitti_val.txt' `
  --output 'runs\example\metrics_ap40.json'
```

训练用 YOLO 标签不能作为评测 GT，因为训练标签不含 Van、Person_sitting 和 DontCare。评测器始终读取原始 KITTI 标签。

## AP40 参考核对

下载 OpenPCDet 的 `eval.py` 和 `rotate_iou.py` 后运行：

```powershell
python scripts/check_ap40_reference.py `
  --reference-dir 'tmp\reference\openpcdet_eval' `
  --output 'docs\reports\ap40-reference-check.json'
```

## Phase 1 一键验收

```powershell
python scripts/audit_phase1.py `
  --image-dir "$env:KITTI_ROOT\training\image_2\training\image_2" `
  --label-dir "$env:KITTI_ROOT\training\label_2\training\label_2" `
  --generated-dir 'data\processed\kitti_yolo_v2'
```

验收成功必须以以下文本结束：

```text
PHASE 1 ACCEPTED
```

## 研究约束

- 不直接修改 Conda 环境中的 `site-packages`。
- 不覆盖原始 KITTI 数据或历史实验目录。
- 不把 P2、BiFPN、WIoU 等已有模块单独声明为原创。
- 所有正式实验保存配置、Git 提交、split 哈希、环境、随机种子、日志和逐项指标。
- 数据与 AP40 验收失败时，不进入模型结构改造。
