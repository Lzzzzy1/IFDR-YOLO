# IFDR-YOLO

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

解释器：

```text
D:\ana\envs\yolo\python.exe
```

## 数据约定

原始 KITTI 保持只读：

```text
E:\myyolo\kitti_project\kitti_raw\training\image_2\training\image_2
E:\myyolo\kitti_project\kitti_raw\training\label_2\training\label_2
```

派生数据写入：

```text
E:\myyolo\kitti_project\data\processed\kitti_yolo_v2
```

原始数据、派生数据、模型权重和训练运行目录均不提交到 Git。

## 测试

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
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
& 'D:\ana\envs\yolo\python.exe' scripts/import_kitti_split.py `
  --train-source 'tmp\kitti_splits\train.txt' `
  --val-source 'tmp\kitti_splits\val.txt' `
  --image-dir 'E:\myyolo\kitti_project\kitti_raw\training\image_2\training\image_2' `
  --label-dir 'E:\myyolo\kitti_project\kitti_raw\training\label_2\training\label_2'
```

## 重建 KITTI YOLO 数据

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/rebuild_kitti.py `
  --image-dir 'E:\myyolo\kitti_project\kitti_raw\training\image_2\training\image_2' `
  --label-dir 'E:\myyolo\kitti_project\kitti_raw\training\label_2\training\label_2' `
  --train-ids 'configs\splits\kitti_train.txt' `
  --val-ids 'configs\splits\kitti_val.txt' `
  --output-dir 'E:\myyolo\kitti_project\data\processed\kitti_yolo_v2'
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
& 'D:\ana\envs\yolo\python.exe' scripts/evaluate_kitti.py `
  --prediction-dir 'path\to\predictions' `
  --label-dir 'E:\myyolo\kitti_project\kitti_raw\training\label_2\training\label_2' `
  --image-dir 'E:\myyolo\kitti_project\kitti_raw\training\image_2\training\image_2' `
  --split 'configs\splits\kitti_val.txt' `
  --output 'runs\example\metrics_ap40.json'
```

训练用 YOLO 标签不能作为评测 GT，因为训练标签不含 Van、Person_sitting 和 DontCare。评测器始终读取原始 KITTI 标签。

## AP40 参考核对

下载 OpenPCDet 的 `eval.py` 和 `rotate_iou.py` 后运行：

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/check_ap40_reference.py `
  --reference-dir 'tmp\reference\openpcdet_eval' `
  --output 'docs\reports\ap40-reference-check.json'
```

## Phase 1 一键验收

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/audit_phase1.py `
  --image-dir 'E:\myyolo\kitti_project\kitti_raw\training\image_2\training\image_2' `
  --label-dir 'E:\myyolo\kitti_project\kitti_raw\training\label_2\training\label_2' `
  --generated-dir 'E:\myyolo\kitti_project\data\processed\kitti_yolo_v2'
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
