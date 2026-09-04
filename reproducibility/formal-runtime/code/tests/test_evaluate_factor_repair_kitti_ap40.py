from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_factor_repair_kitti_ap40 import (
    evaluate_factor_repair_kitti,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(split_sha256: str = "0" * 64) -> dict[str, object]:
    classes = {
        name: {
            difficulty: {
                "ap40": 0.0,
                "num_valid_gt": 0,
                "true_positives": 0,
                "false_positives": 0,
                "ignored_detections": 0,
            }
            for difficulty in ("easy", "moderate", "hard")
        }
        for name in ("Car", "Pedestrian", "Cyclist")
    }
    return {
        "evaluator": "ifdr_yolo.kitti_ap40",
        "split_sha256": split_sha256,
        "split_count": 2,
        "classes": classes,
    }


class EvaluateFactorRepairKittiTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path, str]:
        run_dir = root / "run" / "F0"
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        checkpoint = weights / "last.pt"
        checkpoint.write_bytes(b"primary-checkpoint")
        (run_dir / "status.json").write_text(
            json.dumps({"state": "complete"}), encoding="utf-8"
        )
        (run_dir / "checkpoint_roles.json").write_text(
            json.dumps(
                {
                    "primary_checkpoint": {
                        "path": "last.pt",
                        "role": "primary",
                        "sha256": _sha256(checkpoint),
                    }
                }
            ),
            encoding="utf-8",
        )
        development = root / "development_ids.txt"
        development.write_text("000001\n000002\n", encoding="utf-8", newline="\n")
        image_dir = root / "images"
        image_dir.mkdir()
        for image_id in ("000001", "000002"):
            (image_dir / f"{image_id}.png").write_bytes(f"{image_id}".encode())
        label_dir = root / "labels"
        label_dir.mkdir()
        for image_id in ("000001", "000002"):
            (label_dir / f"{image_id}.txt").write_text("", encoding="utf-8")
        return run_dir, development, image_dir, label_dir, _sha256(checkpoint)

    def test_stages_exact_hardlinks_and_publishes_kitti_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, development, image_dir, label_dir, checkpoint_sha = self._fixture(root)
            calls: list[Path] = []

            def predict(**kwargs: object) -> Path:
                output_dir = Path(kwargs["output_dir"])
                labels = output_dir / "labels"
                labels.mkdir(parents=True)
                calls.append(labels)
                return labels

            def evaluate(**_: object) -> dict[str, object]:
                return _payload(_sha256(development))

            with patch.multiple(
                "scripts.evaluate_factor_repair_kitti_ap40",
                REGISTERED_DEVELOPMENT_COUNT=2,
                REGISTERED_DEVELOPMENT_IDS_SHA256=_sha256(development),
            ):
                result = evaluate_factor_repair_kitti(
                    run_dir=run_dir,
                    condition="F0",
                    development_ids=development,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    output_dir=root / "ap40" / "F0",
                    checkpoint_sha256=checkpoint_sha,
                    predictor=predict,
                    evaluator=evaluate,
                )

            staging = root / "ap40" / "F0" / "development-images"
            self.assertEqual(
                sorted(path.name for path in staging.glob("*.png")),
                ["000001.png", "000002.png"],
            )
            self.assertTrue(
                (staging / "000001.png").samefile(image_dir / "000001.png")
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["evaluator"], "ifdr_yolo.kitti_ap40")
            self.assertTrue((root / "ap40" / "F0" / "kitti_ap40.json").is_file())
            provenance = json.loads(
                (root / "ap40" / "F0" / "provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                provenance["primary_checkpoint"]["resolved_semantic_role"],
                "calibration_last",
            )
            self.assertTrue(provenance["primary_checkpoint"]["role_field_missing"])

    def test_resume_is_explicit_and_does_not_repredict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, development, image_dir, label_dir, checkpoint_sha = self._fixture(root)
            calls = 0

            def predict(**kwargs: object) -> Path:
                nonlocal calls
                calls += 1
                labels = Path(kwargs["output_dir"]) / "labels"
                labels.mkdir(parents=True)
                return labels

            with patch.multiple(
                "scripts.evaluate_factor_repair_kitti_ap40",
                REGISTERED_DEVELOPMENT_COUNT=2,
                REGISTERED_DEVELOPMENT_IDS_SHA256=_sha256(development),
            ):
                kwargs = {
                    "run_dir": run_dir,
                    "condition": "F0",
                    "development_ids": development,
                    "image_dir": image_dir,
                    "label_dir": label_dir,
                    "output_dir": root / "ap40" / "F0",
                    "checkpoint_sha256": checkpoint_sha,
                    "predictor": predict,
                    "evaluator": lambda **_: _payload(_sha256(development)),
                }
                evaluate_factor_repair_kitti(**kwargs)
                with self.assertRaises(FileExistsError):
                    evaluate_factor_repair_kitti(**kwargs)
                evaluate_factor_repair_kitti(**kwargs, resume=True)

            self.assertEqual(calls, 1)

    def test_existing_staging_with_extra_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, development, image_dir, label_dir, checkpoint_sha = self._fixture(root)
            output_dir = root / "ap40" / "F0"
            staging = output_dir / "development-images"
            staging.mkdir(parents=True)
            (staging / "000001.png").write_bytes(b"extra")
            (staging / "000002.png").write_bytes(b"extra")
            (staging / "999999.png").write_bytes(b"extra")
            with patch.multiple(
                "scripts.evaluate_factor_repair_kitti_ap40",
                REGISTERED_DEVELOPMENT_COUNT=2,
                REGISTERED_DEVELOPMENT_IDS_SHA256=_sha256(development),
            ):
                with self.assertRaises(ValueError):
                    evaluate_factor_repair_kitti(
                        run_dir=run_dir,
                        condition="F0",
                        development_ids=development,
                        image_dir=image_dir,
                        label_dir=label_dir,
                        output_dir=output_dir,
                        checkpoint_sha256=checkpoint_sha,
                        predictor=lambda **_: output_dir / "predictions" / "labels",
                        evaluator=lambda **_: _payload(_sha256(development)),
                        resume=True,
                    )

    def test_checkpoint_role_or_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, development, image_dir, label_dir, checkpoint_sha = self._fixture(root)
            roles_path = run_dir / "checkpoint_roles.json"
            roles = json.loads(roles_path.read_text(encoding="utf-8"))
            roles["primary_checkpoint"]["sha256"] = "0" * 64
            roles_path.write_text(json.dumps(roles), encoding="utf-8")
            with patch.multiple(
                "scripts.evaluate_factor_repair_kitti_ap40",
                REGISTERED_DEVELOPMENT_COUNT=2,
                REGISTERED_DEVELOPMENT_IDS_SHA256=_sha256(development),
            ):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    evaluate_factor_repair_kitti(
                        run_dir=run_dir,
                        condition="F0",
                        development_ids=development,
                        image_dir=image_dir,
                        label_dir=label_dir,
                        output_dir=root / "ap40" / "F0",
                        checkpoint_sha256=checkpoint_sha,
                        predictor=lambda **_: root / "predictions" / "labels",
                        evaluator=lambda **_: _payload(_sha256(development)),
                    )


if __name__ == "__main__":
    unittest.main()
