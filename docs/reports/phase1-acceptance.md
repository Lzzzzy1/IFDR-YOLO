# IFDR-YOLO Phase 1 验收报告

日期：2026-07-29
状态：**ACCEPTED，可以进入 Phase 2**

## 1. 验收结论

Phase 1 已建立可信的 KITTI 数据与 2D AP40 评测基础设施：

- 原始 7481 张图像与 7481 个标签文件一一对应；
- common split 为 3712 张 train、3769 张 val，无交集且覆盖全部数据；
- 按每张图真实宽高重新生成 7481 个 YOLO 标签文件；
- 34856 个 Car/Pedestrian/Cyclist 训练框全部合法；
- 51865 个原始 KITTI 对象完整保留在元数据中；
- `invalid_box_count=0`，`clipped_box_count=0`；
- 7481 对源图像与源标签 SHA256 全量复核通过；
- KITTI AP40 与公开参考移植版的 45 个受控对照全部一致；
- 63 个单元与 CLI 测试全部通过。

一键验收终端结果：

```text
PHASE 1 ACCEPTED
images=7481 train=3712 val=3769
labels=7481 yolo_rows=34856
metadata_images=7481 metadata_objects=51865
verified_source_hashes=7481
unit_tests=passed
split_integrity=passed
yolo_coordinates=passed
ap40_reference_check=passed
```

## 2. 数据与固定划分

原始数据保持只读：

- 图像：`E:\myyolo\kitti_project\kitti_raw\training\image_2\training\image_2`
- 标签：`E:\myyolo\kitti_project\kitti_raw\training\label_2\training\label_2`

派生数据：

- `E:\myyolo\kitti_project\data\processed\kitti_yolo_v2`

主实验采用 Chen common split，不称为 KITTI official split：

| 文件 | 数量 | SHA256 |
|---|---:|---|
| `configs/splits/kitti_train.txt` | 3712 | `b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb` |
| `configs/splits/kitti_val.txt` | 3769 | `657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86` |

来源：

- `traveller59/second.pytorch/second/data/ImageSets/train.txt`
- `traveller59/second.pytorch/second/data/ImageSets/val.txt`

## 3. 固定尺寸标签错误量化

旧转换脚本把所有图像按 `1242×375` 归一化。真实尺寸分布为：

| 图像尺寸 | 图像数 | 主类目标数 | 受影响目标数 | 平均归一化误差 | 最大归一化误差 |
|---|---:|---:|---:|---:|---:|
| 1242×375 | 6057 | 28078 | 0 | 0 | 0 |
| 1224×370 | 770 | 2991 | 2991 | 0.0048036560 | 0.0144802619 |
| 1238×374 | 358 | 1880 | 1880 | 0.0009648124 | 0.0031936217 |
| 1241×376 | 296 | 1907 | 1907 | 0.0005701132 | 0.0021672695 |

总计：

- 训练主类目标：34856；
- 受旧固定尺寸归一化影响：6778；
- 受影响比例：19.45%；
- 全体坐标分量平均绝对归一化误差：0.0004954323；
- 最大绝对归一化误差：0.0144802619。

因此，现有 300 epoch 模型的结果不能直接作为论文可信基线。Phase 2 必须在重建标签与固定划分上重新训练。

完整机器可读数据：

- `docs/reports/kitti_label_rebuild_audit.json`
- `docs/reports/kitti_label_rebuild_audit.md`

## 4. 原始对象统计

| 类别 | 数量 |
|---|---:|
| Car | 28742 |
| Pedestrian | 4487 |
| Cyclist | 1627 |
| Van | 2914 |
| Truck | 1094 |
| Tram | 511 |
| Person_sitting | 222 |
| Misc | 973 |
| DontCare | 11295 |

训练标签只导出 Car、Pedestrian、Cyclist。Van、Person_sitting 和 DontCare 未被静默合并，而是保留在原始对象元数据和评测 ignore 逻辑中。

## 5. AP40 评测口径

实现包含：

- Car IoU 严格大于 0.70；
- Pedestrian/Cyclist IoU 严格大于 0.50；
- Easy/Moderate/Hard 的框高、遮挡和截断过滤；
- GT 框高严格大于 40/25/25 px；
- Van 对 Car、Person_sitting 对 Pedestrian 的 ignore；
- DontCare 的 intersection-over-detection-area 抑制；
- 小于难度最小框高的检测不计 FP；
- 重复检测、空预测、无 GT、分数排序和阈值边界；
- 41 点 precision 曲线中排除 recall=0 后的 AP_R40。

KITTI 官方 object devkit：

- URL：`https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip`
- SHA256：`ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1`

本机没有 `g++`、`clang++`、MSVC、CMake，也没有可用 WSL Linux 发行版，因此无法直接编译官方 C++ 程序。按照计划，使用 OpenPCDet 的 KITTI 官方协议 Python 移植版进行第二核对：

- OpenPCDet commit：`233f849829b6ac19afb8af8837a0246890908755`
- `eval.py` SHA256：`d9260a41f68083d685fc43fecac1c417b14a1bc53df4915a46b2924f3a6a3c64`
- `rotate_iou.py` SHA256：`d9b2820b5be5d15ee7233147643134c5d73eff129c773ae4bf59ce73a8ecd7f4`
- 受控图像数：50；
- 场景：perfect、duplicate、high_fp、half_missed、ignore；
- 类别 × 难度 × 场景：45 组；
- 最大 AP40 绝对差：0。

完整结果：`docs/reports/ap40-reference-check.json`

## 6. 历史 YOLOv8m 结果处理

历史目录包含：

- `train/weights/best.pt`
- `train/weights/last.pt`
- `train/results.csv`
- 聚合曲线与可视化图片。

该目录没有逐图像、带置信度的预测文本。虽然可以从 `best.pt` 重新推理，但旧模型使用 6732/749 随机划分和错误的固定尺寸标签；common val 中会包含旧训练样本，重新推理存在数据泄漏。因此：

- 不从聚合 CSV 反推 AP40；
- 不把旧模型结果放入论文主表；
- 旧权重只保留为工程诊断和训练时长参考；
- Phase 2 在新标签/common split 上重新训练 YOLOv8m。

## 7. 已知限制

1. 当前实现覆盖 KITTI 2D bbox AP40，不覆盖 AOS、BEV 或 3D AP。
2. 参考一致性采用受控合成套件；正式 baseline 产生逐图预测后，还需保存完整 AP40 输出。
3. 派生图像可能使用同卷硬链接以节省空间，训练代码不得原地修改图像文件。
4. common split 来自广泛采用的外部划分，不是 KITTI 官方验证划分，论文必须明确说明。
5. 正式 baseline、退化诊断和模型创新尚未开始，不能用 Phase 1 结果声明检测性能提升。

## 8. Phase 2 入口条件

以下条件全部满足：

- [x] 数据一一对应与全覆盖；
- [x] common split 来源和哈希固定；
- [x] 真实尺寸坐标转换；
- [x] 原始对象与 ignore 语义保留；
- [x] 全部 YOLO 坐标合法；
- [x] AP40 边界、ignore 和 DontCare 测试；
- [x] AP40 参考一致性；
- [x] 全量源文件哈希验证；
- [x] 63 个测试通过；
- [x] 历史结果的适用边界已记录。

**决策：批准进入 Phase 2“可信基线与退化诊断”。**
