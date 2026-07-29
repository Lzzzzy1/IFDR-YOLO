# IFDR-YOLO Phase 2B P2 消融基础设施验收报告

日期：2026-07-29  
状态：**ACCEPTED（P2–P5 严格消融已在本地 GPU 全链路通过）**

## 1. 验收结论

Phase 2B 已建立与 YOLOv8m 基线严格同协议的 P2–P5 消融实验。最终验收确认：

- 项目自有、固定展开的 m 规模 P2 YAML 不依赖文件名推断模型尺度；
- 模型输出 stride 固定为 4、8、16、32，共四个 Detect 输入；
- 模型参数量固定为 25,052,620，`state_dict` 固定为 581 项；
- 官方 `yolov8m.pt` 仅向语义不变的层迁移权重，精确迁移 306 项；
- 模型 YAML、官方权重、Ultralytics 版本、迁移键与形状均写入初始化清单；
- 完成迁移的同一个底层 `torch.nn.Module` 被直接交给 trainer，不会在训练
  前被高层 API 重建；
- 真实 KITTI dry-run、1 epoch GPU 训练、验证、预测和 AP40 已完整跑通；
- 114 项自动化测试全部通过。

P2 是严格消融基座，不是论文原创贡献。本报告只证明实验实现可信，不声明
精度提升。

## 2. 代码与执行入口

验收提交：

```text
32bdd95c3fd0239309a6673dbe809d08249ddc40
```

分支：

```text
feature/phase2b-p2-ablation
```

正式配置：

```text
configs/experiments/kitti_yolov8m_p2_s17.yaml
```

模型定义：

```text
models/kitti-p2-m.yaml
```

最终 smoke 运行 ID：

```text
20260729T080819Z-kitti-yolov8m-p2-s17-32bdd95
```

## 3. 模型结构与初始化锁定

| 项目 | 验收值 |
|---|---|
| 检测尺度 | P2、P3、P4、P5 |
| stride | 4、8、16、32 |
| Detect 输入数 | 4 |
| 参数量 | 25,052,620 |
| `state_dict` 项数 | 581 |
| 项目 YAML SHA256 | `0d2cbd9215dc62c4c9920127964ce0ab61b5d10581861d8e83c33fc168aa302a` |
| 上游 P2 YAML SHA256 | `ba54363e9f283e8f60b0fa0843eb37115093338d6f9273ccd4f74f60639754f4` |
| `yolov8m.pt` SHA256 | `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5` |
| 来源状态项 | 475 |
| 目标状态项 | 581 |
| 迁移状态项 | 306 |
| 未迁移目标项 | 275 |
| 迁移层 | 0–9、12、15 |
| seed | 17 |
| 确定性模式 | true |

迁移必须同时满足层号上限、目标键存在和张量形状一致三个条件，并强制实际
数量等于 306。任何模型版本或结构漂移都会使初始化失败，不允许静默加载。

## 4. 真实数据与环境验收

真实 KITTI dry-run 输出：

```text
BASELINE PREFLIGHT PASSED
```

它全量复核 7481 对源图像与标签、3712/3769 common split、派生数据、两个
模型 SHA256、Ultralytics 8.4.98 和 CUDA 设备，并实际构建模型和执行迁移。

最终 smoke 环境：

| 项目 | 记录值 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.1+cu121 |
| Ultralytics | 8.4.98 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6144 MiB |
| 训练显存峰值 | 约 0.918 GiB |
| smoke 墙钟时间 | 约 54.2 秒 |

实际训练参数为 1 epoch、`imgsz=320`、`batch=2`、`workers=0`、
`amp=False`、CUDA 0、seed 17、确定性模式，并显式记录
`pretrained=False`。这里的 `pretrained=False` 表示 trainer 不再自行解析
预训练来源；306 项迁移已在 trainer 创建前由项目代码完成。

## 5. 自动化测试

执行：

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

结果：

```text
Ran 114 tests
OK
```

新增覆盖包括 P2 YAML 哈希与结构、四尺度前向输出、纯语义键选择、真实
306 项迁移、确定性模型准备、同一底层模型训练交接、初始化清单、哈希预检
及初始化失败证据保留。

## 6. 最终 smoke 产物审计

运行目录：

```text
runs/20260729T080819Z-kitti-yolov8m-p2-s17-32bdd95
```

审计结果：

- `status.json` 为 `complete`；
- `git_commit.txt` 为完整验收 SHA；
- `initialization.json` 记录 306 个迁移键及其形状；
- `weights/best.pt` 存在，大小 50,426,233 字节；
- 固定 smoke 验证集为 16 个 ID，预测标签文件也精确为 16 个；
- `metrics_ap40.json` 使用 `ifdr_yolo.kitti_ap40`，`split_count=16`；
- `train.cache` 与 `val.cache` 只位于 `tmp/smoke-kitti/labels/`；
- 正式 `data/processed/` 下不存在 Ultralytics 缓存。

## 7. 受控诊断运行

第一次运行
`20260729T075752Z-kitti-yolov8m-p2-s17-01bcff6` 完成了文件级流程，但源码
核查发现高层 `YOLO.train()` 会为 YAML handle 在 trainer 内重建模型，
因此不能证明 306 项迁移进入优化器。该目录已加入
`INVALIDATED_DO_NOT_USE.md`，不得用于研究。

第二次运行
`20260729T080530Z-kitti-yolov8m-p2-s17-e45d759` 在 trainer 创建阶段因缺少
模型引用而失败，`status.json` 将现场记录为 `training` 阶段
`TypeError`。最终实现既保留锁定 YAML 引用，又把已初始化模块直接附加给
trainer，第三次运行通过。

## 8. Smoke-only 指标边界

最终 smoke 的各类 Easy、Moderate、Hard AP40 均为 0。16/16 图像、1 epoch
和 320 输入只用于验证工程链路，不能用于判断 P2 是否有效，不得写入论文
主结果表或用于选择后续模块。

## 9. 正式实验约束

1. AutoDL 上必须使用相同代码提交、配置、数据哈希和 Ultralytics 版本。
2. 正式 YOLOv8m baseline 与 P2 必须都运行 300 epoch，其他配置完全一致。
3. 如显存要求改变 batch，必须新建配置并同时重跑 baseline 与 P2。
4. 只有 `status=complete` 且包含 3769 个验证预测、AP40 和全部溯源文件的
   运行才能进入论文表格。
5. 后续自适应融合和 IoU 方法必须分别对 baseline、P2 进行逐因素消融，P2
   本身不得表述为原创。

## 10. 阶段判定

Phase 2B 本地 P2 消融基础设施验收通过。下一阶段可以在该受控基座上实现
自适应双向特征融合，但正式精度结论仍需 AutoDL 完成 baseline 与 P2 的
300 epoch 对照实验后才能给出。
