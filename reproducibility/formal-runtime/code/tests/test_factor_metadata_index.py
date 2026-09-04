from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml

from ifdr_yolo.data.development_split import build_development_split
from ifdr_yolo.data.metadata_index import (
    KittiLabelCandidate,
    KittiMetadataObject,
    build_metadata_index,
    compute_sampling_score,
    compute_visibility_score,
    match_metadata_object,
)
from ifdr_yolo.data.natural_degradation import (
    compute_sampling_score as registered_sampling_score,
    compute_visibility_score as registered_visibility_score,
)


SOURCE_SHA256 = "b" * 64
SPLIT_SHA256 = "a" * 64
LABEL_SOURCE_SHA256 = "c" * 64
FACTOR_METADATA_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_factor_metadata.py"
)
REGISTERED_SEED = 20260805
REGISTERED_FRACTION = 0.10


def metadata_object(
    *,
    image_id: str = "000001",
    object_index: int = 2,
    class_id: int = 2,
    class_name: str = "Cyclist",
    bbox_xyxy: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 60.0),
    depth_m: float | None = 40.0,
    occlusion: int = 2,
    truncation: float = 0.25,
) -> KittiMetadataObject:
    return KittiMetadataObject(
        image_id=image_id,
        object_index=object_index,
        class_id=class_id,
        class_name=class_name,
        bbox_xyxy=bbox_xyxy,
        depth_m=depth_m,
        occlusion=occlusion,
        truncation=truncation,
    )


def labels_for(objects: list[KittiMetadataObject]) -> dict[str, list[object]]:
    labels: dict[str, list[object]] = {}
    for source in objects:
        labels.setdefault(source.image_id, []).append(source.as_label())
    return labels


def _full_split_rows(count: int = 40) -> list[dict[str, object]]:
    """Build four deterministic strata for full-mode split derivation."""
    if count != 40:
        raise ValueError("the full-mode fixture is locked to 40 images")
    rows: list[dict[str, object]] = []
    for index in range(10):
        rows.append(
            {
                "image_id": f"image_{index:04d}",
                "cyclist": False,
                "cyclist_joint": 0.0,
            }
        )
    for index in range(10, 40):
        rank = index - 10
        if rank < 10:
            joint = 1.0 / 6.0
        elif rank < 20:
            joint = 0.75
        else:
            joint = 1.0
        rows.append(
            {
                "image_id": f"image_{index:04d}",
                "cyclist": True,
                "cyclist_joint": joint,
            }
        )
    return rows


def _full_images_rows(
    label_hashes: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    rows = [
        {"image_id": f"image_{index:04d}", "split": "train"}
        for index in range(40)
    ]
    if label_hashes is not None:
        for row in rows:
            row["source_label_sha256"] = label_hashes[row["image_id"]]
    rows.extend(
        [
            {"image_id": "heldout_0001", "split": "val"},
            {"image_id": "heldout_0002", "split": "val"},
        ]
    )
    return rows


def _full_raw_label_line(index: int) -> str:
    if index < 10:
        kind, truncated, occluded = "Car", 0.0, 0
        y2, depth = 64.0, 15.0
    elif index < 20:
        kind, truncated, occluded = "Cyclist", 0.0, 0
        y2, depth = 54.0, 15.0
    elif index < 30:
        kind, truncated, occluded = "Cyclist", 0.0, 0
        y2, depth = 34.0, 37.5
    else:
        kind, truncated, occluded = "Cyclist", 1.0, 3
        y2, depth = 4.0, 60.0
    return (
        f"{kind} {truncated} {occluded} -1.0 0.0 0.0 10.0 {y2} "
        f"1.0 1.0 1.0 0.0 0.0 {depth} 0.0\n"
    )


def _full_metadata_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(40):
        image_id = f"image_{index:04d}"
        if index < 10:
            rows.append(
                {
                    "image_id": image_id,
                    "kind": "Car",
                    "truncated": 0.0,
                    "occluded": 0,
                    "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 64.0},
                    "location_xyz": [0.0, 0.0, 15.0],
                }
            )
            continue

        rank = index - 10
        if rank < 10:
            bbox = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 54.0}
            location_xyz = [0.0, 0.0, 15.0]
            occluded, truncated = 0, 0.0
        elif rank < 20:
            bbox = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 34.0}
            location_xyz = [0.0, 0.0, 37.5]
            occluded, truncated = 0, 0.0
        else:
            bbox = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 4.0}
            location_xyz = [0.0, 0.0, 60.0]
            occluded, truncated = 3, 1.0
        rows.append(
            {
                "image_id": image_id,
                "kind": "Cyclist",
                "truncated": truncated,
                "occluded": occluded,
                "bbox": bbox,
                "location_xyz": location_xyz,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_full_fixture(root: Path, *, checkpoint_bytes: bytes = b"init-checkpoint\n") -> dict[str, Path]:
    metadata_jsonl = root / "objects.jsonl"
    images_jsonl = root / "images.jsonl"
    checkpoint = root / "initialization.pt"
    raw_label_dir = root / "raw_labels"
    raw_label_dir.mkdir()
    label_hashes: dict[str, str] = {}
    for index in range(40):
        image_id = f"image_{index:04d}"
        label_path = raw_label_dir / f"{image_id}.txt"
        label_path.write_text(_full_raw_label_line(index), encoding="utf-8")
        label_hashes[image_id] = _sha256_file(label_path)
    _write_jsonl(metadata_jsonl, _full_metadata_rows())
    _write_jsonl(images_jsonl, _full_images_rows(label_hashes))
    checkpoint.write_bytes(checkpoint_bytes)
    return {
        "metadata": metadata_jsonl,
        "images": images_jsonl,
        "checkpoint": checkpoint,
        "raw_labels": raw_label_dir,
    }


def _refresh_label_hash(fixture: dict[str, Path], image_id: str) -> None:
    rows = json.loads("[" + ",".join(
        line
        for line in fixture["images"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ) + "]")
    label_hash = _sha256_file(fixture["raw_labels"] / f"{image_id}.txt")
    for row in rows:
        if row["image_id"] == image_id:
            row["source_label_sha256"] = label_hash
    _write_jsonl(fixture["images"], rows)


def _raw_labels_sha256(raw_label_dir: Path, image_ids: Sequence[str]) -> str:
    entries = [
        {
            "image_id": image_id,
            "sha256": _sha256_file(raw_label_dir / f"{image_id}.txt"),
        }
        for image_id in sorted(image_ids)
    ]
    payload = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_full_mode(
    fixture: dict[str, Path],
    root: Path,
    *,
    output_root: Path | None = None,
    extra_args: tuple[str, ...] = (),
    output_dir: Path | None = None,
    config_output: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    output_dir = output_dir or (root / "factor-artifacts")
    config_output = config_output or (
        root / "kitti_ifdr_factor_repair_dev_s17.yaml"
    )
    args = [
        sys.executable,
        str(FACTOR_METADATA_SCRIPT),
        "--metadata-jsonl",
        str(fixture["metadata"]),
        "--images-jsonl",
        str(fixture["images"]),
        "--raw-label-dir",
        str(fixture["raw_labels"]),
        "--initialization-checkpoint",
        str(fixture["checkpoint"]),
        "--output-dir",
        str(output_dir),
        "--config-output",
        str(config_output),
    ]
    if output_root is not None:
        args.extend(("--output-root", str(output_root)))
    args.extend(extra_args)
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )


class FactorMetadataIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = metadata_object()

    def build(
        self,
        objects: list[KittiMetadataObject],
        *,
        labels: dict[str, list[object]] | None = None,
        source_sha256: str | None = SOURCE_SHA256,
        split_sha256: str | None = SPLIT_SHA256,
        label_source_sha256: str | None = LABEL_SOURCE_SHA256,
    ):
        return build_metadata_index(
            objects,
            labels=labels if labels is not None else labels_for(objects),
            source_sha256=source_sha256,
            split_sha256=split_sha256,
            label_source_sha256=label_source_sha256,
        )

    def test_metadata_index_binds_exact_label_and_scores(self) -> None:
        index = self.build([self.source])

        record = index.by_image["000001"][0]
        self.assertEqual(record.object_id, "000001:000002")
        self.assertEqual(record.class_id, 2)
        self.assertEqual(record.class_name, "Cyclist")
        self.assertEqual(record.bbox_xyxy, self.source.bbox_xyxy)
        self.assertEqual(record.height, 40.0)
        self.assertEqual(
            record.sampling,
            registered_sampling_score(record.height, record.depth_m),
        )
        self.assertEqual(
            record.visibility,
            registered_visibility_score(record.occlusion, record.truncation),
        )
        self.assertEqual(record.sampling, compute_sampling_score(40.0, 40.0))
        self.assertEqual(record.visibility, compute_visibility_score(2, 0.25))
        self.assertEqual(
            record.joint,
            1.0 - (1.0 - record.sampling) * (1.0 - record.visibility),
        )
        self.assertTrue(record.sampling_valid)
        self.assertTrue(record.visibility_valid)

    def test_label_candidate_carries_object_index(self) -> None:
        candidate = self.source.as_label()
        self.assertIsInstance(candidate, KittiLabelCandidate)
        self.assertEqual(candidate.object_index, 2)

    def test_object_id_uses_exact_image_and_zero_padded_object_index(self) -> None:
        source = metadata_object(image_id="000017", object_index=0)
        record = self.build([source]).by_image["000017"][0]
        self.assertEqual(record.object_id, "000017:000000")

    def test_registered_score_helpers_keep_natural_boundaries(self) -> None:
        for height, depth in ((64.0, 15.0), (34.0, 37.5), (4.0, 60.0)):
            with self.subTest(height=height, depth=depth):
                self.assertEqual(
                    compute_sampling_score(height, depth),
                    registered_sampling_score(height, depth),
                )
        for occlusion, truncation in ((0, 0.0), (1, 0.0), (3, 1.0)):
            with self.subTest(occlusion=occlusion, truncation=truncation):
                self.assertEqual(
                    compute_visibility_score(occlusion, truncation),
                    registered_visibility_score(occlusion, truncation),
                )

    def test_iou_at_099_is_accepted(self) -> None:
        candidate = metadata_object(bbox_xyxy=(10.0, 20.0, 29.8, 60.0))
        index = self.build(
            [self.source],
            labels={"000001": [candidate.as_label()]},
        )
        self.assertEqual(index.by_image["000001"][0].bbox_xyxy, self.source.bbox_xyxy)

    def test_iou_below_099_fails(self) -> None:
        candidate = metadata_object(bbox_xyxy=(10.0, 20.0, 29.78, 60.0))
        with self.assertRaisesRegex(ValueError, "metadata match"):
            self.build(
                [self.source],
                labels={"000001": [candidate.as_label()]},
            )

    def test_class_mismatch_fails(self) -> None:
        candidate = metadata_object(class_id=1, class_name="Pedestrian")
        with self.assertRaisesRegex(ValueError, "metadata match"):
            self.build(
                [self.source],
                labels={"000001": [candidate.as_label()]},
            )

    def test_duplicate_object_identity_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.build([self.source, self.source])

    def test_nonfinite_box_fails(self) -> None:
        with self.assertRaises(ValueError):
            invalid = metadata_object(
                bbox_xyxy=(math.nan, 20.0, 30.0, 60.0),
            )
            self.build([invalid])

    def test_reversed_box_fails(self) -> None:
        with self.assertRaises(ValueError):
            invalid = metadata_object(
                bbox_xyxy=(30.0, 60.0, 10.0, 20.0),
            )
            self.build([invalid])

    def test_zero_area_box_fails(self) -> None:
        with self.assertRaises(ValueError):
            invalid = metadata_object(
                bbox_xyxy=(10.0, 20.0, 10.0, 60.0),
            )
            self.build([invalid])

    def test_invalid_positive_depth_masks_only_depth(self) -> None:
        invalid_depth = metadata_object(
            bbox_xyxy=(10.0, 20.0, 30.0, 54.0),
            depth_m=0.0,
        )
        index = self.build([invalid_depth])
        record = index.by_image["000001"][0]

        self.assertIsNone(record.depth_m)
        self.assertTrue(record.sampling_valid)
        self.assertEqual(record.sampling, registered_sampling_score(34.0, None))
        self.assertEqual(index.invalid_depth_count, 1)

    def test_invalid_nonfinite_depth_fails_closed(self) -> None:
        for image_number, invalid_depth in enumerate(
            (math.nan, math.inf, -math.inf),
            start=1,
        ):
            with self.subTest(invalid_depth=invalid_depth):
                with self.assertRaises(ValueError):
                    source = metadata_object(
                        image_id=f"{image_number:06d}",
                        depth_m=invalid_depth,
                        bbox_xyxy=(10.0, 20.0, 30.0, 54.0),
                    )
                    self.build([source])

    def test_missing_depth_masks_only_depth(self) -> None:
        source = metadata_object(
            depth_m=None,
            bbox_xyxy=(10.0, 20.0, 30.0, 54.0),
        )
        index = self.build([source])
        record = index.by_image["000001"][0]
        self.assertIsNone(record.depth_m)
        self.assertTrue(record.sampling_valid)
        self.assertEqual(record.sampling, registered_sampling_score(34.0, None))
        self.assertEqual(index.invalid_depth_count, 1)

    def test_invalid_occlusion_or_truncation_fails(self) -> None:
        for kwargs in (
            {"occlusion": 4},
            {"truncation": 1.1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    invalid = metadata_object(**kwargs)
                    self.build([invalid])

    def test_missing_source_or_split_hash_fails(self) -> None:
        for kwargs in (
            {"source_sha256": None},
            {"split_sha256": None},
            {"source_sha256": ""},
            {"split_sha256": ""},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.build([self.source], **kwargs)

    def test_invalid_source_or_split_hash_fails(self) -> None:
        for kwargs in (
            {"source_sha256": "b" * 63},
            {"split_sha256": "a" * 63},
            {"source_sha256": "g" * 64},
            {"split_sha256": "g" * 64},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.build([self.source], **kwargs)

    def test_serialization_order_is_stable(self) -> None:
        first = metadata_object(image_id="000001", object_index=2)
        second = metadata_object(image_id="000002", object_index=1)
        forward = self.build([first, second])
        reversed_index = self.build([second, first])

        self.assertEqual(forward.to_json_bytes(), reversed_index.to_json_bytes())
        self.assertEqual(forward.sha256, reversed_index.sha256)
        self.assertEqual(forward.by_image, reversed_index.by_image)

    def test_index_digest_matches_canonical_payload_without_embedded_digest(self) -> None:
        index = self.build([self.source])
        payload = json.loads(index.to_json_bytes().decode("utf-8"))
        embedded_digest = payload.pop("sha256")
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        recomputed = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(embedded_digest, index.sha256)
        self.assertEqual(recomputed, index.sha256)

    def test_index_digest_is_sensitive_to_identity_and_provenance_fields(self) -> None:
        baseline = self.build([self.source])
        variants = (
            self.build(
                [metadata_object(image_id="000002")],
            ),
            self.build(
                [metadata_object(bbox_xyxy=(10.0, 20.0, 31.0, 60.0))],
            ),
            self.build([metadata_object(depth_m=20.0)]),
            self.build([metadata_object(depth_m=None)]),
            self.build([self.source], source_sha256="d" * 64),
            self.build([self.source], split_sha256="e" * 64),
            self.build([self.source], label_source_sha256="f" * 64),
        )
        for variant in variants:
            with self.subTest(variant=variant.sha256):
                self.assertNotEqual(variant.sha256, baseline.sha256)

    def test_match_metadata_object_freezes_registered_iou_threshold(self) -> None:
        candidate = self.source.as_label()
        for minimum_iou in (
            0.98,
            float("nan"),
            float("inf"),
            True,
            0,
            "0.99",
            Decimal("0.99"),
        ):
            with self.subTest(minimum_iou=minimum_iou):
                with self.assertRaises(ValueError):
                    match_metadata_object(
                        self.source,
                        [candidate],
                        minimum_iou=minimum_iou,
                    )

    def test_zero_multiple_and_ambiguous_matches_fail_closed(self) -> None:
        cases = (
            ("zero", []),
            (
                "multiple",
                [
                    self.source.as_label(),
                    metadata_object(bbox_xyxy=(10.0, 20.0, 29.999, 60.0)).as_label(),
                ],
            ),
            ("ambiguous", [self.source.as_label(), self.source.as_label()]),
        )
        for name, candidates in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "metadata match"):
                    self.build(
                        [self.source],
                        labels={"000001": candidates},
                    )

    def test_metadata_index_is_deeply_immutable(self) -> None:
        index = self.build([self.source])

        with self.assertRaises(TypeError):
            index.by_image["000001"] = ()  # type: ignore[index]
        with self.assertRaises(AttributeError):
            index.by_image["000001"].append(index.by_image["000001"][0])  # type: ignore[attr-defined]
        with self.assertRaises((FrozenInstanceError, AttributeError, TypeError)):
            index.by_image["000001"][0].sampling = 0.0  # type: ignore[misc]

    def test_exact_id_whitespace_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            source_with_trailing_space = metadata_object(image_id="000001 ")
            self.build(
                [source_with_trailing_space],
                labels={"000001": [self.source.as_label()]},
            )

        with self.assertRaises(ValueError):
            self.build(
                [self.source],
                labels={" 000001": [self.source.as_label()]},
            )


class FactorMetadataBuildCliTest(unittest.TestCase):
    def _artifact_paths(self, root: Path) -> tuple[Path, ...]:
        output_dir = root / "factor-artifacts"
        return (
            output_dir / "fit_ids.txt",
            output_dir / "development_ids.txt",
            output_dir / "metadata_index.json",
            output_dir / "manifest.json",
            root / "kitti_ifdr_factor_repair_dev_s17.yaml",
        )

    def test_existing_input_jsonl_split_only_mode_remains_compatible(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_jsonl = root / "split_rows.jsonl"
            _write_jsonl(input_jsonl, _full_split_rows())
            output_dir = root / "split-only"
            result = subprocess.run(
                [
                    sys.executable,
                    str(FACTOR_METADATA_SCRIPT),
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_dir / "fit_ids.txt").is_file())
            self.assertTrue((output_dir / "development_ids.txt").is_file())
            self.assertTrue((output_dir / "development_split.json").is_file())

    def test_full_mode_writes_hash_bound_artifacts_and_canonical_yaml(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            result = _run_full_mode(fixture, root)
            self.assertEqual(result.returncode, 0, result.stderr)

            artifact_paths = self._artifact_paths(root)
            for path in artifact_paths:
                self.assertTrue(path.is_file(), path)
            output_dir = root / "factor-artifacts"
            fit_ids = set(
                (output_dir / "fit_ids.txt").read_text(encoding="utf-8").splitlines()
            )
            development_ids = set(
                (output_dir / "development_ids.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertTrue(fit_ids.isdisjoint(development_ids))
            self.assertEqual(
                fit_ids | development_ids,
                {row["image_id"] for row in _full_images_rows() if row["split"] == "train"},
            )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            metadata_index = json.loads(
                (output_dir / "metadata_index.json").read_text(encoding="utf-8")
            )
            self.assertRegex(metadata_index["sha256"], r"^[0-9a-f]{64}$")
            expected_split = build_development_split(
                _full_split_rows(),
                seed=REGISTERED_SEED,
                fraction=REGISTERED_FRACTION,
            )
            train_image_ids = [
                row["image_id"]
                for row in _full_images_rows()
                if row["split"] == "train"
            ]
            expected_hashes = {
                "source_metadata_sha256": _sha256_file(fixture["metadata"]),
                "images_metadata_sha256": _sha256_file(fixture["images"]),
                "raw_labels_sha256": _raw_labels_sha256(
                    fixture["raw_labels"], train_image_ids
                ),
                "split_sha256": expected_split.sha256,
                "metadata_index_sha256": metadata_index["sha256"],
                "metadata_index_file_sha256": _sha256_file(
                    output_dir / "metadata_index.json"
                ),
                "initialization_checkpoint_sha256": _sha256_file(fixture["checkpoint"]),
                "fit_ids_sha256": _sha256_file(output_dir / "fit_ids.txt"),
                "development_ids_sha256": _sha256_file(output_dir / "development_ids.txt"),
            }
            for key, expected in expected_hashes.items():
                self.assertEqual(manifest[key], expected)
                self.assertRegex(manifest[key], r"^[0-9a-f]{64}$")
            self.assertEqual(manifest["fit_count"], 36)
            self.assertEqual(manifest["development_count"], 4)

            config = yaml.safe_load(
                (root / "kitti_ifdr_factor_repair_dev_s17.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["schema_version"], 1)
            self.assertEqual(config["identity"], {
                "source_metadata_sha256": expected_hashes["source_metadata_sha256"],
                "images_metadata_sha256": expected_hashes["images_metadata_sha256"],
                "raw_labels_sha256": expected_hashes["raw_labels_sha256"],
                "split_sha256": expected_hashes["split_sha256"],
                "metadata_sha256": expected_hashes["metadata_index_sha256"],
                "initialization_checkpoint_sha256": expected_hashes[
                    "initialization_checkpoint_sha256"
                ],
                "fit_ids_sha256": expected_hashes["fit_ids_sha256"],
                "development_ids_sha256": expected_hashes["development_ids_sha256"],
            })
            self.assertEqual(
                config["development"],
                {"seed": REGISTERED_SEED, "fraction": REGISTERED_FRACTION},
            )
            expected_conditions = {
                **{
                    name: {"track": "metadata", "epochs": 60}
                    for name in ("M1", "M2", "M3")
                },
                **{
                    name: {"track": "factor", "epochs": 30}
                    for name in ("F0", "F1", "F2", "F3")
                },
            }
            self.assertEqual(config["conditions"], expected_conditions)
            self.assertEqual(config["task_adaptation_epochs"], 60)
            self.assertEqual(config["max_selected_factor_repairs"], 1)
            self.assertFalse(config["early_stopping"])
            self.assertEqual(config["training"], {"imgsz": 640})
            self.assertEqual(config["factor_loss"], {
                "natural_gain": 1.0,
                "specificity_gain": 0.5,
                "specificity_margin": 0.05,
                "factor_weights": [1.0, 1.0],
            })
            self.assertEqual(config["model"], {
                "nodes": [11, 14, 17, 20, 23, 26],
                "primary_nodes": [17, 20, 23, 26],
            })
            self.assertEqual(
                config["paths"],
                {
                    "metadata_jsonl": fixture["metadata"].resolve().as_posix(),
                    "images_jsonl": fixture["images"].resolve().as_posix(),
                    "raw_label_dir": fixture["raw_labels"].resolve().as_posix(),
                    "initialization_checkpoint": fixture["checkpoint"].resolve().as_posix(),
                    "output_root": "runs/factor-repair",
                },
            )
            self.assertEqual(config["schedule"], {
                "replay": {
                    "eta_peak": 0.30,
                    "ramp_epochs": 5,
                    "focus_end_epoch": 40,
                    "recovery_start_epoch": 41,
                    "total_epochs": 60,
                    "priority_clip_quantile": 0.95,
                    "eligible_floor": 0.05,
                    "replacement": True,
                    "draws_per_epoch": "fit_count",
                },
                "factor_calibration": {
                    "epochs": 30,
                    "views_per_sample": 3,
                    "fusion_schedule": 0.0,
                    "dcli_schedule": 0.0,
                },
                "task_adaptation": {"epochs": 60},
            })
            self.assertEqual(config["checkpoint_policy"], {
                "primary": "last.pt",
                "diagnostic": "best.pt",
                "early_stopping": False,
            })
            self.assertEqual(config["metadata_replay"], {
                "M1": "original",
                "M2": "cyclist_uniform",
                "M3": "joint_score",
            })
            self.assertEqual(config["factor_gate"], {
                "seed17_min_positive_primary_directions": 3,
                "formal_min_positive_seed_node_directions": 10,
                "formal_total_seed_node_directions": 12,
                "minimum_severity_ordering": 0.8,
                "diagnostic_reverse_abs_rho": 0.1,
                "selection_tie_tolerance": 1e-12,
                "require_paired_delta_ci_lower_positive": True,
                "require_zero_malformed": True,
            })
            self.assertNotIn("bootstrap_seed", config["factor_gate"])
            self.assertNotIn("bootstrap_replicates", config["factor_gate"])

    def test_full_mode_is_byte_idempotent_on_identical_rerun(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            first = _run_full_mode(fixture, root)
            self.assertEqual(first.returncode, 0, first.stderr)
            first_bytes = {path: path.read_bytes() for path in self._artifact_paths(root)}
            second = _run_full_mode(fixture, root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                first_bytes,
                {path: path.read_bytes() for path in self._artifact_paths(root)},
            )

    def test_explicit_output_root_serializes_portable_posix_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            explicit_output_root = Path("runs") / "factor-repair-explicit"
            result = _run_full_mode(
                fixture,
                root,
                output_root=explicit_output_root,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = yaml.safe_load(
                (root / "kitti_ifdr_factor_repair_dev_s17.yaml").read_text(
                    encoding="utf-8"
                )
            )
            serialized = config["paths"]["output_root"]
            self.assertEqual(serialized, explicit_output_root.as_posix())
            self.assertNotIn("\\", serialized)

    def test_tampered_metadata_index_fails_before_any_write(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            first = _run_full_mode(fixture, root)
            self.assertEqual(first.returncode, 0, first.stderr)
            artifact_paths = self._artifact_paths(root)
            before = {path: path.read_bytes() for path in artifact_paths}
            tampered_path = root / "factor-artifacts" / "metadata_index.json"
            tampered_bytes = b"tampered metadata index\n"
            tampered_path.write_bytes(tampered_bytes)

            rerun = _run_full_mode(fixture, root)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertRegex(
                rerun.stdout + rerun.stderr,
                r"metadata_index|tamper|identity|existing",
            )
            self.assertEqual(tampered_path.read_bytes(), tampered_bytes)
            for path, bytes_before in before.items():
                if path != tampered_path:
                    self.assertEqual(path.read_bytes(), bytes_before)

    def test_tampered_yaml_fails_before_any_write(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            first = _run_full_mode(fixture, root)
            self.assertEqual(first.returncode, 0, first.stderr)
            artifact_paths = self._artifact_paths(root)
            before = {path: path.read_bytes() for path in artifact_paths}
            config_path = root / "kitti_ifdr_factor_repair_dev_s17.yaml"
            tampered_bytes = b"tampered config\n"
            config_path.write_bytes(tampered_bytes)

            rerun = _run_full_mode(fixture, root)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertRegex(
                rerun.stdout + rerun.stderr,
                r"config|yaml|tamper|identity|existing",
            )
            self.assertEqual(config_path.read_bytes(), tampered_bytes)
            for path, bytes_before in before.items():
                if path != config_path:
                    self.assertEqual(path.read_bytes(), bytes_before)

    def test_independent_raw_labels_fail_closed_before_any_write(self) -> None:
        cases = ("missing", "extra", "class_drift", "box_drift")
        for case in cases:
            with self.subTest(case=case):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    fixture = _write_full_fixture(root)
                    label_path = fixture["raw_labels"] / "image_0010.txt"
                    if case == "missing":
                        label_path.unlink()
                    elif case == "extra":
                        label_path.write_text(
                            label_path.read_text(encoding="utf-8")
                            + _full_raw_label_line(10),
                            encoding="utf-8",
                        )
                        _refresh_label_hash(fixture, "image_0010")
                    elif case == "class_drift":
                        label_path.write_text(
                            label_path.read_text(encoding="utf-8").replace(
                                "Cyclist ", "Car ", 1
                            ),
                            encoding="utf-8",
                        )
                        _refresh_label_hash(fixture, "image_0010")
                    else:
                        label_path.write_text(
                            label_path.read_text(encoding="utf-8").replace(
                                "10.0 54.0", "11.0 54.0", 1
                            ),
                            encoding="utf-8",
                        )
                        _refresh_label_hash(fixture, "image_0010")

                    result = _run_full_mode(fixture, root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertRegex(
                        result.stdout + result.stderr,
                        r"label|metadata match|candidate|unused|raw",
                    )
                    self.assertFalse((root / "factor-artifacts").exists())
                    self.assertFalse(
                        (root / "kitti_ifdr_factor_repair_dev_s17.yaml").exists()
                    )

    def test_known_val_raw_label_is_ignored_by_train_index(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            (fixture["raw_labels"] / "heldout_0001.txt").write_text(
                _full_raw_label_line(0),
                encoding="utf-8",
            )
            result = _run_full_mode(fixture, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata_index = json.loads(
                (root / "factor-artifacts" / "metadata_index.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("heldout_0001", metadata_index["by_image"])

    def test_unknown_raw_label_is_rejected_before_any_write(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            (fixture["raw_labels"] / "unknown_id.txt").write_text(
                _full_raw_label_line(0),
                encoding="utf-8",
            )
            result = _run_full_mode(fixture, root)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stdout + result.stderr, r"raw label|image row|unknown")
            self.assertFalse((root / "factor-artifacts").exists())
            self.assertFalse(
                (root / "kitti_ifdr_factor_repair_dev_s17.yaml").exists()
            )

    def test_symlink_raw_label_is_rejected_before_any_write(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            symlink_path = fixture["raw_labels"] / "unknown_id.txt"
            try:
                os.symlink(
                    fixture["raw_labels"] / "image_0000.txt",
                    symlink_path,
                )
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            result = _run_full_mode(fixture, root)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stdout + result.stderr, r"symlink|raw label|unknown")
            self.assertFalse((root / "factor-artifacts").exists())
            self.assertFalse(
                (root / "kitti_ifdr_factor_repair_dev_s17.yaml").exists()
            )

    def test_full_mode_rejects_canonical_path_collisions_before_writes(self) -> None:
        output_dir_name = "factor-artifacts"
        collision_names = (
            "output-dir",
            "fit_ids.txt",
            "development_ids.txt",
            "metadata_index.json",
            "manifest.json",
        )
        for collision_name in collision_names:
            with self.subTest(collision_name=collision_name):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    fixture = _write_full_fixture(root)
                    output_dir = root / output_dir_name
                    config_output = (
                        output_dir
                        if collision_name == "output-dir"
                        else output_dir / collision_name
                    )
                    result = _run_full_mode(
                        fixture,
                        root,
                        config_output=config_output,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertRegex(
                        result.stdout + result.stderr,
                        r"collision|distinct|same path",
                    )
                    self.assertFalse(output_dir.exists())
                    self.assertFalse(
                        (root / "kitti_ifdr_factor_repair_dev_s17.yaml").exists()
                    )

    def test_images_jsonl_rejects_split_typo_before_writes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            rows = json.loads("[" + ",".join(
                line
                for line in fixture["images"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ) + "]")
            rows[0]["split"] = "trian"
            _write_jsonl(fixture["images"], rows)
            result = _run_full_mode(fixture, root)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stdout + result.stderr, r"split|train")
            self.assertFalse((root / "factor-artifacts").exists())
            self.assertFalse(
                (root / "kitti_ifdr_factor_repair_dev_s17.yaml").exists()
            )

    def test_completed_bundle_missing_artifact_fails_closed(self) -> None:
        for missing_name in ("metadata_index.json", "__config__"):
            with self.subTest(missing_name=missing_name):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    fixture = _write_full_fixture(root)
                    first = _run_full_mode(fixture, root)
                    self.assertEqual(first.returncode, 0, first.stderr)
                    artifact_paths = self._artifact_paths(root)
                    before = {path: path.read_bytes() for path in artifact_paths}
                    missing_path = (
                        root / "kitti_ifdr_factor_repair_dev_s17.yaml"
                        if missing_name == "__config__"
                        else root / "factor-artifacts" / missing_name
                    )
                    missing_path.unlink()

                    rerun = _run_full_mode(fixture, root)
                    self.assertNotEqual(rerun.returncode, 0)
                    self.assertRegex(
                        rerun.stdout + rerun.stderr,
                        r"completed bundle is incomplete/corrupt",
                    )
                    self.assertFalse(missing_path.exists())
                    for path, bytes_before in before.items():
                        if path != missing_path:
                            self.assertEqual(path.read_bytes(), bytes_before)

    def test_empty_checkpoint_fails_before_writing_outputs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root, checkpoint_bytes=b"")
            result = _run_full_mode(fixture, root)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stdout + result.stderr, r"checkpoint|empty|non-empty")
            self.assertFalse((root / "factor-artifacts").exists())
            self.assertFalse((root / "kitti_ifdr_factor_repair_dev_s17.yaml").exists())

    def test_full_mode_rejects_argument_conflict_and_partial_arguments(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _write_full_fixture(root)
            split_input = root / "split_rows.jsonl"
            _write_jsonl(split_input, _full_split_rows())
            conflict = _run_full_mode(
                fixture,
                root,
                extra_args=("--input-jsonl", str(split_input)),
            )
            self.assertNotEqual(conflict.returncode, 0)
            self.assertRegex(
                conflict.stdout + conflict.stderr,
                r"conflict|mutually exclusive|cannot combine",
            )

            partial = subprocess.run(
                [
                    sys.executable,
                    str(FACTOR_METADATA_SCRIPT),
                    "--metadata-jsonl",
                    str(fixture["metadata"]),
                    "--images-jsonl",
                    str(fixture["images"]),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(partial.returncode, 0)
            self.assertRegex(
                partial.stdout + partial.stderr,
                r"required|checkpoint|output-dir|config-output|full mode",
            )


if __name__ == "__main__":
    unittest.main()
