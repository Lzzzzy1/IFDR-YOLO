# PLAIN_P2 / DCLI 五种子正式实验：完整代码与证据交付

请发送同目录最终生成的 `PLAIN_P2-DCLI-Teacher-Complete-20260903.zip`。本目录是非覆盖式新交付，原项目 `E:\myyolo` 未被修改。

## 一句话结论

正式五种子结果支持将 **PLAIN_P2 作为论文主线**；DCLI 只能作为探索性负结果和诊断贡献，不能表述为稳定提升。

## 冻结结果

- 协议：KITTI `fit3341 / development371`，seeds `0–4`，`30 epoch`，`batch 16`，`imgsz 640`，Moderate Pedestrian/Cyclist macro AP_R40。
- PLAIN_P2 五种子均值：`95.3886399233`。
- DCLI 五种子均值：`94.3702399983`。
- 配对差值 `PLAIN_P2 − DCLI`：`+1.0183999250` AP。
- 配对 95% t 区间：`[-0.4054911520, 2.4422910020]`；区间跨 0。
- PLAIN_P2 在 `4/5` 个 seed 上更高；DCLI 在 `1/5` 个 seed 上更高。
- 科学表述：DCLI 的稳定增益不成立；PLAIN_P2 是稳定的推理结构与论文主方法。

## 目录说明

- `01_FORMAL_RUNTIME_SOURCE/`：服务器正式运行时源码闭包。它从冻结运行时包恢复，并用服务器身份收据所绑定的四个精确源码版本覆盖；包含 P2 模型 YAML、PLAIN_P2/DCLI 的 10 份 seed 配置，以及正式统计测试所需的 v124/v128/v129 冻结支持源码与只读证据镜像，不包含权重。
- `02_FORMAL_FIVE_SEED_EVIDENCE/`：十个正式 run 的 `results.csv`、`metrics_ap40.json`、terminal receipt、300 份逐 epoch publication receipt、五种子绑定收据和统计输出。
- `03_TEACHER_REPORT/`：已验收的五种子教师决策报告 DOCX/PDF、材料清单和证据收据。
- `04_CHAPTER3_AND_PAPER/`：老师认可的英文 Chapter 3，以及与冻结结果一致的 venue-neutral 8 页英文稿。该稿不是 AIAC 或其他具体会议的官方模板稿。
- `05_FORMAL_RUN_BUILDERS/`：十个正式 ordinal 的本地构造/启动证据包，用于追溯正式服务器 generation。
- `06_HISTORICAL_SOURCE_SNAPSHOT/`：2026-08-27 的历史“295 文件源码快照”。它保留用于对照，但漏装了配置引用的 `models/kitti-p2-m.yaml`，不要把它单独当作开箱可运行包。
- `00_README_AND_AUDIT/`：自动审计收据、人工一致性审计、完整哈希清单和复核脚本。

## 代码、实验与论文如何对应

- P2 四尺度结构：`01_FORMAL_RUNTIME_SOURCE/code/models/kitti-p2-m.yaml`，Detect 输入固定为层 `18, 21, 24, 27`，对应 stride `4, 8, 16, 32`。
- PLAIN_P2 入口：`01_FORMAL_RUNTIME_SOURCE/code/scripts/run_p2_fit_reference.py`。
- DCLI 入口：`01_FORMAL_RUNTIME_SOURCE/code/scripts/run_p2_interaction_s0.py`。
- DCLI 损失与不确定性：`01_FORMAL_RUNTIME_SOURCE/code/ifdr_yolo/losses/dcli.py` 和 `ifdr_detection.py`。
- 冻结训练调度：DCLI 配置中 `5` 个 frozen epoch、`10` 个 ramp epoch；`beta=0.5`、校准增益 `0.1`、因子监督增益 `0.2`；fusion gate 与 counterfactual consistency 均关闭。
- 五种子结果：`02_FORMAL_FIVE_SEED_EVIDENCE/outputs-v1/per_seed_results.csv` 和 `paired_statistics.json`。
- Chapter 3：`04_CHAPTER3_AND_PAPER/P2-YOLOv8_Chapter_3_English_Draft.docx`。

## 复现边界

本包不包含 KITTI 图像/标签、`yolov8m.pt`、checkpoint、训练缓存、虚拟环境或 Ultralytics/PyTorch 第三方源码。原正式环境记录为 Ultralytics `8.4.98`、PyTorch `2.8.0+cu128`、CUDA `12.8`；预训练权重的冻结 SHA-256 为 `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`。真正重训前仍需由项目持有人合法提供数据与该权重，并恢复等价环境。

本轮没有启动新训练，也没有修改数据、seed、划分、评价器或正式结果。本地 `15 epoch / batch 2` 方向筛选不属于正式五种子证据，未混入本交付的正式结论。

## 验收摘要

- 正式 run：`10/10 PASS`。
- 逐 epoch 发布收据：`300/300`。
- 正式运行时基线源码哈希：`279` 个文件通过；服务器精确覆盖源码：`4/4` 通过。
- PLAIN_P2/DCLI seed 配置：`10/10` 与身份快照或绑定收据一致。
- P2 模型 YAML：已补齐并通过配置声明的 SHA-256。
- 原始证据文件：`355`；统计输出文件：`26`。
- Python 源文件语法解析：`246/246 PASS`（兼容 UTF-8 BOM）。
- 五种子统计/曲线/成本契约相关回归测试：在交付目录内直接执行，`11/11 PASS`。
- Chapter 3 机制描述与冻结代码：PASS。
- 8 页英文稿正式数值与负结论边界：PASS。
- 权重/checkpoint 混入：`0`；凭据模式命中：`0`。

完整细节见 `00_README_AND_AUDIT/代码-实验-论文一致性审计.md` 与 `AUDIT_RECEIPT.json`。
