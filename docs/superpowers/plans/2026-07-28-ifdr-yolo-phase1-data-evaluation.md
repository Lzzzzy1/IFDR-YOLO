# IFDR-YOLO Phase 1 Data and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用真实图像尺寸重建 KITTI 训练标签和固定划分，并交付经过合成用例验证的 KITTI 2D AP40 评测器，为后续所有模型结论建立可信基础。

**Architecture:** 原始 KITTI 保持只读；解析器先将标签转换为不可变类型，再由划分校验器、YOLO 导出器和审计器消费。评测器直接消费原始 KITTI 语义和统一的检测类型，独立于 Ultralytics；Ultralytics 预测通过显式适配层进入评测器。所有派生文件写入被 Git 忽略的 `data/processed/kitti_yolo_v2`，可复现元数据和固定 split 文件进入版本控制。

**Tech Stack:** Python 3.11、标准库 `dataclasses/enum/json/hashlib/unittest`、NumPy、Pillow、PyYAML、Ultralytics 8.4.98、Git。

---

## 执行约束

- 工作目录：`E:\myyolo\kitti_project`
- Python：`D:\ana\envs\yolo\python.exe`
- 原始图像：`E:\myyolo\kitti_project\kitti_raw\training\image_2\training\image_2`
- 原始标签：`E:\myyolo\kitti_project\kitti_raw\training\label_2\training\label_2`
- 派生数据：`E:\myyolo\kitti_project\data\processed\kitti_yolo_v2`
- Ultralytics 配置缓存：`E:\myyolo\kitti_project\tmp\yolo-config`（仅在需要导入 Ultralytics 时设置 `YOLO_CONFIG_DIR`）
- 不覆盖现有 `kitti/`、`kitti.yaml`、`train.py`、`runs/` 和 `KITTI_YOLOv8m_Results/`。
- 每个任务先运行失败测试，再写最小实现，再运行全量测试。
- 本阶段禁止修改模型结构或损失函数。

## 类型与口径冻结

核心类型名称在本阶段保持一致：

```python
@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class KittiObject:
    kind: str
    truncated: float
    occluded: int
    alpha: float
    bbox: BoundingBox
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float
    score: float | None = None


class Difficulty(Enum):
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"


@dataclass(frozen=True)
class Detection:
    image_id: str
    kind: str
    score: float
    bbox: BoundingBox
```

训练类映射固定为：

```python
TRAIN_CLASS_TO_ID = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}
```

Van 对 Car、Person_sitting 对 Pedestrian 只作为评测 ignore；DontCare 只用于抑制未匹配检测，不导出为训练框。

---

### Task 1: 建立最小包结构、入口与数据保护

**Files:**

- Create: `ifdr_yolo/__init__.py`
- Create: `ifdr_yolo/data/__init__.py`
- Create: `ifdr_yolo/eval/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_repository_safety.py`
- Modify: `.gitignore`

- [ ] **Step 1: 写数据保护失败测试**

在 `tests/test_repository_safety.py` 写入：

```python
from pathlib import Path
import unittest


class RepositorySafetyTest(unittest.TestCase):
    def test_generated_data_is_git_ignored(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ignore = (root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/processed/", ignore)

    def test_raw_dataset_is_not_under_generated_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = (root / "kitti_raw").resolve()
        generated = (root / "data" / "processed").resolve()
        self.assertNotEqual(raw, generated)
        self.assertNotIn(raw, generated.parents)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_repository_safety -v
```

Expected: `test_generated_data_is_git_ignored` fails because `.gitignore` 尚未包含 `data/processed/`。

- [ ] **Step 3: 创建包文件并更新忽略规则**

在 `.gitignore` 追加：

```gitignore
# Generated datasets and evaluation caches
data/processed/
data/cache/
```

所有 `__init__.py` 初始为空。

- [ ] **Step 4: 运行测试**

Run:

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_repository_safety -v
```

Expected: 2 tests pass。

- [ ] **Step 5: 提交**

```powershell
git add .gitignore ifdr_yolo scripts tests
git commit -m "chore: scaffold IFDR-YOLO data pipeline"
```

---

### Task 2: 实现严格的 KITTI 标签类型与解析器

**Files:**

- Create: `ifdr_yolo/data/kitti_types.py`
- Create: `ifdr_yolo/data/kitti_parser.py`
- Create: `tests/test_kitti_parser.py`

- [ ] **Step 1: 写解析器失败测试**

`tests/test_kitti_parser.py`：

```python
import unittest

from ifdr_yolo.data.kitti_parser import parse_kitti_line


class KittiParserTest(unittest.TestCase):
    def test_parses_ground_truth_line(self) -> None:
        line = (
            "Pedestrian 0.25 1 -0.20 10.0 20.0 30.0 80.0 "
            "1.70 0.60 0.80 1.0 2.0 15.0 0.10"
        )
        obj = parse_kitti_line(line)
        self.assertEqual(obj.kind, "Pedestrian")
        self.assertEqual(obj.occluded, 1)
        self.assertEqual(obj.bbox.as_xyxy(), (10.0, 20.0, 30.0, 80.0))
        self.assertEqual(obj.dimensions_hwl, (1.70, 0.60, 0.80))
        self.assertIsNone(obj.score)

    def test_parses_detection_score(self) -> None:
        line = (
            "Car 0 0 0 1 2 11 22 1 2 3 4 5 6 0.5 0.91"
        )
        self.assertAlmostEqual(parse_kitti_line(line).score or 0.0, 0.91)

    def test_rejects_wrong_field_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "15 or 16"):
            parse_kitti_line("Car 0 0")

    def test_rejects_inverted_box(self) -> None:
        line = "Car 0 0 0 20 10 5 30 1 1 1 0 0 0 0"
        with self.assertRaisesRegex(ValueError, "invalid bounding box"):
            parse_kitti_line(line)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 确认测试因模块缺失而失败**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_kitti_parser -v
```

Expected: import error。

- [ ] **Step 3: 实现不可变类型**

`ifdr_yolo/data/kitti_types.py` 必须包含：

- `BoundingBox`；
- `BoundingBox.as_xyxy()`；
- `BoundingBox.width`、`height`、`area`；
- `KittiObject`；
- `Difficulty`；
- `Detection`；
- `TRAIN_CLASS_TO_ID`；
- `EVAL_CLASSES = ("Car", "Pedestrian", "Cyclist")`。

`BoundingBox` 构造后验证所有数值有限且 `x2 >= x1`、`y2 >= y1`。

- [ ] **Step 4: 实现严格解析**

`parse_kitti_line(line: str) -> KittiObject`：

- 只接受 15 字段 GT 或 16 字段 detection；
- 将数值字段转换为 `float/int`；
- 保留原始类别字符串；
- 验证 `truncated` 有限、`occluded` 为整数；
- 错误信息包含字段数或非法框原因。

另实现 `parse_kitti_file(path: Path) -> tuple[KittiObject, ...]`。空文件返回空元组；非空行的错误附带文件和行号。

- [ ] **Step 5: 运行局部与全量测试**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_kitti_parser -v
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass。

- [ ] **Step 6: 提交**

```powershell
git add ifdr_yolo/data tests/test_kitti_parser.py
git commit -m "feat: add strict KITTI label parser"
```

---

### Task 3: 导入并锁定固定划分

**Files:**

- Create: `configs/splits/kitti_train.txt`
- Create: `configs/splits/kitti_val.txt`
- Create: `configs/splits/source.json`
- Create: `ifdr_yolo/data/splits.py`
- Create: `scripts/import_kitti_split.py`
- Create: `tests/test_kitti_splits.py`

- [ ] **Step 1: 写划分校验测试**

`tests/test_kitti_splits.py`：

```python
from pathlib import Path
import tempfile
import unittest

from ifdr_yolo.data.splits import load_ids, validate_split


class KittiSplitTest(unittest.TestCase):
    def test_rejects_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split(("000001",), ("000001",), {"000001"})

    def test_rejects_missing_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_split(("000001",), (), {"000001", "000002"})

    def test_load_ids_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("000001\n000001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_ids(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 确认测试失败**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_kitti_splits -v
```

Expected: import error。

- [ ] **Step 3: 实现划分读取与验证**

`ifdr_yolo/data/splits.py` 提供 `load_ids(path: Path) -> tuple[str, ...]`、`discover_ids(image_dir: Path, label_dir: Path) -> set[str]`、`validate_split(train_ids, val_ids, available_ids) -> None` 和 `sha256_file(path: Path) -> str`。

验证六位数字格式、重复、交集、图像/标签一一对应和全集覆盖。

- [ ] **Step 4: 下载上游 common split 到临时目录**

```powershell
New-Item -ItemType Directory -Force -Path 'tmp\kitti_splits' | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt' -OutFile 'tmp\kitti_splits\train.txt'
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt' -OutFile 'tmp\kitti_splits\val.txt'
```

若网络或上游路径失败，停止本任务并核对 OpenPCDet/SECOND 官方仓库，禁止自行随机生成 3712/3769 后冒充 common split。

- [ ] **Step 5: 校验数量与本地全集**

```powershell
& 'D:\ana\envs\yolo\python.exe' -c "from pathlib import Path; from ifdr_yolo.data.splits import load_ids, discover_ids, validate_split; tr=load_ids(Path('tmp/kitti_splits/train.txt')); va=load_ids(Path('tmp/kitti_splits/val.txt')); ids=discover_ids(Path('kitti_raw/training/image_2/training/image_2'), Path('kitti_raw/training/label_2/training/label_2')); validate_split(tr,va,ids); print(len(tr),len(va),len(ids))"
```

Expected exactly:

```text
3712 3769 7481
```

- [ ] **Step 6: 固化文件和来源**

实现 `scripts/import_kitti_split.py`，它先调用 `load_ids` 和 `validate_split`，再将临时文件复制为：

- `configs/splits/kitti_train.txt`
- `configs/splits/kitti_val.txt`

脚本随后以 `json.dumps(..., indent=2, sort_keys=True)` 写入 `configs/splits/source.json`，字段固定为：

- `name`：`Chen common KITTI train/val split`；
- `train_count`：由 `len(train_ids)` 生成，且程序断言为 3712；
- `val_count`：由 `len(val_ids)` 生成，且程序断言为 3769；
- `train_url` 和 `val_url`：使用 Step 4 的两个完整 URL；
- `train_sha256` 和 `val_sha256`：分别由复制后的文件调用 `sha256_file` 生成。

运行：

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/import_kitti_split.py --train-source 'tmp\kitti_splits\train.txt' --val-source 'tmp\kitti_splits\val.txt'
```

Expected: 终端打印 `train=3712 val=3769` 和两个 64 位 SHA256；JSON 内直接保存实际值。

- [ ] **Step 7: 增加仓库内真实划分测试**

在 `tests/test_kitti_splits.py` 增加：

```python
def test_repository_split_counts(self) -> None:
    root = Path(__file__).resolve().parents[1]
    train_ids = load_ids(root / "configs/splits/kitti_train.txt")
    val_ids = load_ids(root / "configs/splits/kitti_val.txt")
    self.assertEqual(len(train_ids), 3712)
    self.assertEqual(len(val_ids), 3769)
```

- [ ] **Step 8: 运行全量测试并提交**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
git add configs/splits ifdr_yolo/data/splits.py tests/test_kitti_splits.py
git commit -m "data: lock common KITTI train validation split"
```

Expected: all tests pass；提交中包含两个 split 文件和来源哈希。

---

### Task 4: 实现真实尺寸 YOLO 转换

**Files:**

- Create: `ifdr_yolo/data/yolo_export.py`
- Create: `tests/test_yolo_export.py`

- [ ] **Step 1: 写多尺寸和类别策略失败测试**

`tests/test_yolo_export.py`：

```python
import unittest

from ifdr_yolo.data.kitti_types import BoundingBox, KittiObject
from ifdr_yolo.data.yolo_export import object_to_yolo


def make_object(kind: str, bbox: BoundingBox) -> KittiObject:
    return KittiObject(
        kind=kind,
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox=bbox,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 0.0),
        rotation_y=0.0,
    )


class YoloExportTest(unittest.TestCase):
    def test_uses_actual_image_size(self) -> None:
        obj = make_object("Pedestrian", BoundingBox(0.0, 0.0, 122.4, 37.0))
        row = object_to_yolo(obj, image_width=1224, image_height=370)
        self.assertEqual(row.class_id, 1)
        self.assertAlmostEqual(row.x_center, 0.05)
        self.assertAlmostEqual(row.y_center, 0.05)
        self.assertAlmostEqual(row.width, 0.10)
        self.assertAlmostEqual(row.height, 0.10)

    def test_non_training_class_is_not_exported(self) -> None:
        obj = make_object("Van", BoundingBox(0.0, 0.0, 100.0, 100.0))
        self.assertIsNone(object_to_yolo(obj, 1242, 375))

    def test_clips_box_to_image(self) -> None:
        obj = make_object("Car", BoundingBox(-10.0, 10.0, 1250.0, 400.0))
        row = object_to_yolo(obj, 1242, 375)
        self.assertIsNotNone(row)
        assert row is not None
        for value in row.as_tuple()[1:]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_drops_zero_area_after_clipping(self) -> None:
        obj = make_object("Cyclist", BoundingBox(1300.0, 10.0, 1400.0, 20.0))
        self.assertIsNone(object_to_yolo(obj, 1242, 375))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 确认测试失败**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_yolo_export -v
```

Expected: import error。

- [ ] **Step 3: 实现转换**

`ifdr_yolo/data/yolo_export.py` 提供不可变 `YoloRow`：

```python
@dataclass(frozen=True)
class YoloRow:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_tuple(self) -> tuple[int, float, float, float, float]:
        return (
            self.class_id,
            self.x_center,
            self.y_center,
            self.width,
            self.height,
        )

    def serialize(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )
```

另提供 `object_to_yolo(obj: KittiObject, image_width: int, image_height: int) -> YoloRow | None`。

实现顺序：

1. 非训练主类返回 `None`；
2. 将框裁剪到 `[0,width] × [0,height]`；
3. 裁剪后零面积返回 `None` 并由调用者计入审计；
4. 使用真实宽高归一化；
5. 验证全部值有限且处于 `[0,1]`。

- [ ] **Step 4: 运行测试并提交**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_yolo_export -v
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
git add ifdr_yolo/data/yolo_export.py tests/test_yolo_export.py
git commit -m "feat: convert KITTI boxes with true image dimensions"
```

Expected: all tests pass。

---

### Task 5: 构建可复现数据集和审计清单

**Files:**

- Create: `ifdr_yolo/data/build_dataset.py`
- Create: `scripts/rebuild_kitti.py`
- Create: `tests/test_dataset_builder.py`
- Create: `configs/data/kitti_v2.yaml`

- [ ] **Step 1: 写端到端小样本失败测试**

`tests/test_dataset_builder.py` 使用 `tempfile.TemporaryDirectory` 创建：

- 两张不同尺寸 PNG：`1242×375` 和 `1224×370`；
- 两个 KITTI 标签文件；
- train/val 各一个 ID。

测试：

```python
result = build_dataset(
    image_dir=image_dir,
    label_dir=label_dir,
    train_ids=("000001",),
    val_ids=("000002",),
    output_dir=output_dir,
)
self.assertEqual(result.image_count, 2)
self.assertTrue((output_dir / "labels/train/000001.txt").exists())
self.assertTrue((output_dir / "metadata/objects.jsonl").exists())
self.assertEqual(result.invalid_box_count, 0)
```

同时断言第二张图中的 122.4 px 宽框被归一化为 `0.1`，证明未使用固定宽度。

- [ ] **Step 2: 确认测试失败**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_dataset_builder -v
```

Expected: import error。

- [ ] **Step 3: 实现构建器**

`build_dataset(image_dir, label_dir, train_ids, val_ids, output_dir) -> BuildSummary` 必须：

- 校验输出目录不等于或包含原始目录；
- 默认拒绝非空输出目录，只有显式 `--overwrite-generated` 才重建；
- 读取每张 PNG 真实尺寸；
- 使用硬链接或复制图像，失败时回退复制；
- 按 split 写入 `images/{train,val}` 和 `labels/{train,val}`；
- 写入 `metadata/objects.jsonl`，保留每个原始对象的完整字段；
- 写入 `metadata/images.jsonl`，记录 ID、split、宽高和源文件 SHA256；
- 写入 `manifest.json`，记录 split 哈希、构建时间、代码提交和汇总；
- 统计类别、图像尺寸、框高、遮挡、截断、裁剪和丢弃数量；
- 不修改原始图像与标签。

使用临时输出目录完成构建后再原子重命名，避免中断后留下半成品。

- [ ] **Step 4: 实现 CLI**

`scripts/rebuild_kitti.py` 参数：

```text
--image-dir
--label-dir
--train-ids
--val-ids
--output-dir
--overwrite-generated
```

`--image-dir` 默认指向 `kitti_raw/training/image_2/training/image_2`，`--label-dir` 默认指向 `kitti_raw/training/label_2/training/label_2`，其余参数使用本计划中的仓库相对路径。运行时打印全部解析后的绝对路径。成功退出码 0；数据错误退出码非 0。若服务器数据目录结构不同，必须显式传参，不做递归猜测。

- [ ] **Step 5: 写数据 YAML**

`configs/data/kitti_v2.yaml`：

```yaml
path: data/processed/kitti_yolo_v2
train: images/train
val: images/val
names:
  0: Car
  1: Pedestrian
  2: Cyclist
```

- [ ] **Step 6: 运行小样本测试**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_dataset_builder -v
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass。

- [ ] **Step 7: 运行 7481 张正式构建**

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/rebuild_kitti.py
```

Expected summary contains:

```text
images=7481
train=3712
val=3769
```

并且 `invalid_box_count=0` 或每个异常均列入审计报告、人工确认后有明确处理。不得静默跳过。

- [ ] **Step 8: 检查旧固定尺寸错误**

构建器额外输出 `docs/reports/kitti_label_rebuild_audit.json` 和 `.md`，对每种图像尺寸报告：

- 图像数量；
- 旧固定 `1242×375` 与真实尺寸归一化的平均/最大坐标差；
- 受影响框数量；
- 越界或裁剪数量。

- [ ] **Step 9: 提交代码和报告，不提交派生数据**

```powershell
git status --short
git add configs/data ifdr_yolo/data/build_dataset.py scripts/rebuild_kitti.py tests/test_dataset_builder.py docs/reports/kitti_label_rebuild_audit.json docs/reports/kitti_label_rebuild_audit.md
git commit -m "data: rebuild KITTI labels and audit fixed-size error"
```

Expected: `data/processed/` 不出现在 staged files 中。

---

### Task 6: 实现 KITTI 2D AP40 基础匹配

**Files:**

- Create: `ifdr_yolo/eval/kitti_ap40.py`
- Create: `tests/test_kitti_ap40.py`

- [ ] **Step 1: 写几何、难度和完美检测失败测试**

`tests/test_kitti_ap40.py` 至少包含：

```python
import unittest

from ifdr_yolo.data.kitti_types import (
    BoundingBox,
    Detection,
    Difficulty,
    KittiObject,
)
from ifdr_yolo.eval.kitti_ap40 import (
    box_iou,
    evaluate_class,
    is_valid_ground_truth,
)


def ground_truth(
    kind: str,
    box: BoundingBox,
    truncated: float = 0.0,
    occluded: int = 0,
) -> KittiObject:
    return KittiObject(
        kind=kind,
        truncated=truncated,
        occluded=occluded,
        alpha=0.0,
        bbox=box,
        dimensions_hwl=(1.0, 1.0, 1.0),
        location_xyz=(0.0, 0.0, 0.0),
        rotation_y=0.0,
    )


class KittiAP40Test(unittest.TestCase):
    def test_iou_identity_and_disjoint(self) -> None:
        box = BoundingBox(0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(box_iou(box, box), 1.0)
        self.assertAlmostEqual(
            box_iou(box, BoundingBox(20.0, 20.0, 30.0, 30.0)),
            0.0,
        )

    def test_difficulty_height_threshold(self) -> None:
        short = ground_truth("Pedestrian", BoundingBox(0, 0, 20, 24))
        tall = ground_truth("Pedestrian", BoundingBox(0, 0, 20, 25))
        self.assertFalse(
            is_valid_ground_truth(short, "Pedestrian", Difficulty.MODERATE)
        )
        self.assertTrue(
            is_valid_ground_truth(tall, "Pedestrian", Difficulty.MODERATE)
        )

    def test_perfect_detection_has_one_hundred_ap(self) -> None:
        box = BoundingBox(10, 10, 50, 80)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", box),)},
            detections_by_image={
                "000001": (Detection("000001", "Pedestrian", 0.9, box),)
            },
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertAlmostEqual(result.ap40, 100.0)

    def test_no_detection_has_zero_ap(self) -> None:
        box = BoundingBox(10, 10, 50, 80)
        result = evaluate_class(
            gt_by_image={"000001": (ground_truth("Pedestrian", box),)},
            detections_by_image={"000001": ()},
            class_name="Pedestrian",
            difficulty=Difficulty.MODERATE,
        )
        self.assertAlmostEqual(result.ap40, 0.0)
```

- [ ] **Step 2: 确认测试失败**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_kitti_ap40 -v
```

Expected: import error。

- [ ] **Step 3: 实现难度规则与 IoU**

固定规则：

| Difficulty | min height | max occlusion | max truncation |
|---|---:|---:|---:|
| Easy | 40 px | 0 | 0.15 |
| Moderate | 25 px | 1 | 0.30 |
| Hard | 25 px | 2 | 0.50 |

类别 IoU：

- Car：0.70；
- Pedestrian：0.50；
- Cyclist：0.50。

实现 `box_iou` 时显式处理零面积和无交集，禁止 NaN/Inf。

- [ ] **Step 4: 实现匹配和 AP40**

接口为 `evaluate_class(gt_by_image, detections_by_image, class_name, difficulty) -> ClassMetrics`。两个映射的值分别是 `tuple[KittiObject, ...]` 和 `tuple[Detection, ...]`。

`ClassMetrics` 至少包含 `ap40`、`precision`、`recall`、`scores`、`num_valid_gt`。

AP40 使用 40 个召回采样点；精度先做从右到左的单调包络。所有分数以百分制输出，并在 docstring 中说明。

- [ ] **Step 5: 运行测试**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_kitti_ap40 -v
```

Expected: current tests pass。

---

### Task 7: 补齐 ignore、DontCare 和边界用例

**Files:**

- Modify: `ifdr_yolo/eval/kitti_ap40.py`
- Modify: `tests/test_kitti_ap40.py`

- [ ] **Step 1: 写 ignore 行为失败测试**

新增用例：

1. Car 检测命中 Van，不计 TP 也不计 FP；
2. Pedestrian 检测命中 Person_sitting，不计 TP 也不计 FP；
3. 未匹配检测与 DontCare 重叠达到规则时不计 FP；
4. 检测命中因难度被忽略的同类 GT，不计 FP；
5. 无 GT 且存在普通检测时 AP 为 0、FP 数正确；
6. 同一 GT 的第二个重复检测计 FP；
7. 低分正确检测和高分错误检测的排序正确；
8. 未知类别请求抛出 `ValueError`。

DontCare 测试示例：

```python
def test_detection_inside_dontcare_is_ignored(self) -> None:
    dontcare = ground_truth("DontCare", BoundingBox(0, 0, 100, 100))
    detection = Detection(
        "000001",
        "Pedestrian",
        0.9,
        BoundingBox(10, 10, 50, 80),
    )
    result = evaluate_class(
        gt_by_image={"000001": (dontcare,)},
        detections_by_image={"000001": (detection,)},
        class_name="Pedestrian",
        difficulty=Difficulty.MODERATE,
    )
    self.assertEqual(result.false_positives, 0)
```

- [ ] **Step 2: 确认新增测试失败**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_kitti_ap40 -v
```

Expected: ignore 相关用例失败。

- [ ] **Step 3: 实现三级 GT 状态**

每个 GT 对当前 class/difficulty 分类为：

- `VALID`：可成为 TP；
- `IGNORED`：匹配后抑制 FP；
- `IRRELEVANT`：不参与该类别匹配。

类关系：

```python
IGNORE_KIND = {
    "Car": {"Van"},
    "Pedestrian": {"Person_sitting"},
    "Cyclist": set(),
}
```

DontCare 在正常/ignored GT 匹配后处理。2D DontCare 重叠按“交集面积 / 检测框面积”计算，并使用当前评测类的最小重叠阈值；达到抑制条件的未匹配检测不计 FP。匹配顺序和分数阈值选择遵循 KITTI 2D 评测语义，并在模块 docstring 中记录与官方开发包的对应关系。

- [ ] **Step 4: 增加数值与输入验证**

- 检测分数必须有限；
- 所有 image ID 的检测按 score 降序稳定排序；
- 同分数保持输入顺序；
- 重复 image ID 不丢失；
- 空字典返回结构化结果而不是异常；
- AP 不得超出 `[0,100]`。

- [ ] **Step 5: 运行全量测试并提交**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
git add ifdr_yolo/eval/kitti_ap40.py tests/test_kitti_ap40.py
git commit -m "feat: implement tested KITTI AP40 evaluation"
```

Expected: all tests pass。

---

### Task 8: 连接 Ultralytics 预测和评测 CLI

**Files:**

- Create: `ifdr_yolo/eval/prediction_io.py`
- Create: `scripts/evaluate_kitti.py`
- Create: `tests/test_prediction_io.py`

- [ ] **Step 1: 写预测转换失败测试**

测试至少覆盖：

- Ultralytics normalized `class x_center y_center width height confidence` 到像素 `xyxy`；
- 使用每张图真实尺寸；
- 类别 ID 只允许 0/1/2；
- 无置信度字段报错；
- 越界框裁剪；
- 非有限置信度报错。

- [ ] **Step 2: 确认失败**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest tests.test_prediction_io -v
```

Expected: import error。

- [ ] **Step 3: 实现预测 I/O**

提供 `load_yolo_predictions(prediction_dir, image_sizes) -> dict[str, tuple[Detection, ...]]` 和 `load_kitti_ground_truth(label_dir, image_ids) -> dict[str, tuple[KittiObject, ...]]`。

评测不依赖训练标签，因为训练标签已经丢弃 ignore 类；GT 必须从原始 `label_2` 加载。

- [ ] **Step 4: 实现 CLI**

`scripts/evaluate_kitti.py` 参数：

```text
--prediction-dir
--label-dir
--image-dir
--split
--output
```

输出 JSON 顶层固定包含：

- `evaluator`，值为 `ifdr_yolo.kitti_ap40`；
- `split_sha256`，运行时通过 `sha256_file(split_path)` 生成；
- `classes`，依次包含 Car、Pedestrian、Cyclist，每类包含 easy、moderate、hard 的完整 `ClassMetrics` 数值。

同时在终端打印紧凑表格。

- [ ] **Step 5: 运行测试与空预测冒烟**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
New-Item -ItemType Directory -Force -Path 'tmp\empty_predictions' | Out-Null
& 'D:\ana\envs\yolo\python.exe' scripts/evaluate_kitti.py --prediction-dir 'tmp\empty_predictions' --output 'tmp\empty_metrics.json'
```

Expected:

- tests pass；
- CLI 完成且所有 AP 为 0；
- JSON 中 split SHA256 与 `configs/splits/kitti_val.txt` 一致。

- [ ] **Step 6: 提交**

```powershell
git add ifdr_yolo/eval/prediction_io.py scripts/evaluate_kitti.py tests/test_prediction_io.py
git commit -m "feat: evaluate YOLO predictions with KITTI AP40"
```

---

### Task 9: 参考一致性检查与 Phase 1 验收

**Files:**

- Create: `scripts/audit_phase1.py`
- Create: `docs/reports/phase1-acceptance.md`
- Modify: `README.md`（若不存在则创建）

- [ ] **Step 1: 实现一键验收脚本**

`scripts/audit_phase1.py` 依次执行只读检查：

1. 原始图像/标签数量与 ID 一致；
2. split 数量、交集、全集覆盖和 SHA256；
3. 派生图片/标签/元数据数量；
4. 所有 YOLO 行字段数、类别、有限性和 `[0,1]` 范围；
5. manifest 与实际文件一致；
6. AP40 合成测试；
7. 全量 `unittest`。

任何一项失败返回非零退出码。

- [ ] **Step 2: 对少量样本做双实现核对**

从 val 固定选取前 20 个 ID，构造：

- 完美预测；
- 空预测；
- 每个 GT 平移 5 px 的预测；
- 每个 GT 重复一次的预测。

将本实现首先与 KITTI 官方 object development kit 核对 TP/FP 和 AP；只有官方 C++ 工具在当前环境无法构建时，才使用经过注明的公开移植版做第二核对。官方来源固定为 `https://www.cvlibs.net/datasets/kitti/eval_object.php`。记录下载文件 SHA256、参考实现 commit（若有）和执行命令。允许浮点误差 `1e-6`；若差异超过误差，停止验收并定位语义差异。

- [ ] **Step 3: 运行完整验收**

```powershell
& 'D:\ana\envs\yolo\python.exe' scripts/audit_phase1.py
```

Expected terminal ending:

```text
PHASE 1 ACCEPTED
images=7481 train=3712 val=3769
unit_tests=passed
split_integrity=passed
yolo_coordinates=passed
ap40_reference_check=passed
```

- [ ] **Step 4: 尝试评测历史 baseline**

历史结果使用旧标签和旧随机划分，因此只做诊断：

- 若存在逐图像带置信度预测，适配到 common val 后运行 AP40；
- 若只有聚合 CSV，没有逐图预测，不伪造 AP40，在报告中写明“无法从聚合指标恢复”；
- 任何历史结果不得进入最终主表。

- [ ] **Step 5: 写验收报告**

`docs/reports/phase1-acceptance.md` 必须包含：

- 原始数据摘要；
- common split 来源与哈希；
- 固定尺寸错误量化；
- 转换后的类别/难度/尺寸统计；
- AP40 测试和参考一致性；
- 已知限制；
- 进入 Phase 2 的明确批准或阻止原因。

- [ ] **Step 6: 更新 README**

README 增加：

- 数据目录约定；
- 重建命令；
- 测试命令；
- AP40 评测命令；
- 不提交原始/派生数据的说明。

- [ ] **Step 7: 最终验证**

```powershell
& 'D:\ana\envs\yolo\python.exe' -m unittest discover -s tests -v
& 'D:\ana\envs\yolo\python.exe' scripts/audit_phase1.py
git status --short
```

Expected:

- all tests pass；
- `PHASE 1 ACCEPTED`；
- 无派生数据被 Git 跟踪；
- 既有用户文件未被修改或删除。

- [ ] **Step 8: 提交**

```powershell
git add scripts/audit_phase1.py docs/reports/phase1-acceptance.md README.md
git commit -m "docs: accept reproducible KITTI data and AP40 pipeline"
```

---

## Phase 1 完成定义

仅当以下条件全部满足，才生成 Phase 2 的详细实施计划：

- [ ] 7481 张图像与标签一一对应；
- [ ] 3712/3769 split 来源、哈希、交集和覆盖检查通过；
- [ ] 真实图像尺寸转换完成；
- [ ] 原始数据未修改，派生数据未进入 Git；
- [ ] 训练类和评测 ignore 语义分离；
- [ ] AP40 的完美、空预测、重复检测、难度、Van、Person_sitting 和 DontCare 测试通过；
- [ ] 小样本参考一致性通过；
- [ ] 数据审计和 Phase 1 验收报告已提交；
- [ ] 全量单元测试通过；
- [ ] 当前工作树中的既有用户文件得到保留。

Phase 2 计划必须以 Phase 1 的真实类别、框高、难度和旧标签误差统计为输入，不能在这些统计产生前凭空冻结干预强度范围。
