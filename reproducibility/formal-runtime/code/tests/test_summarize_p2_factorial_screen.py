from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


def _write_fixture(root: Path) -> dict[str, Path]:
    image_dir = root / "images"
    label_dir = root / "labels"
    split = root / "development.txt"
    image_dir.mkdir()
    label_dir.mkdir()
    prediction_dirs: dict[str, Path] = {}
    ids = ("000001", "000002", "000003")
    for image_id in ids:
        Image.new("RGB", (100, 100)).save(image_dir / f"{image_id}.png")
        (label_dir / f"{image_id}.txt").write_text(
            "Pedestrian 0 0 0 10 10 40 80 1 1 1 0 0 0 0\n"
            "Cyclist 0 0 0 50 10 80 80 1 1 1 0 0 0 0\n",
            encoding="utf-8",
        )
    split.write_text("\n".join(ids) + "\n", encoding="utf-8")
    for condition in ("C", "A", "B", "AB"):
        directory = root / condition
        directory.mkdir()
        prediction_dirs[condition] = directory
        for image_id in ids:
            lines: list[str] = []
            if condition in ("C", "B"):
                lines.append("1 0.25 0.45 0.30 0.70 0.90")
            if condition in ("C", "A"):
                lines.append("2 0.65 0.45 0.30 0.70 0.90")
            (directory / f"{image_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
    return {
        "image_dir": image_dir,
        "label_dir": label_dir,
        "split": split,
        "prediction_dirs": prediction_dirs,
    }


class SummarizeP2FactorialScreenTest(unittest.TestCase):
    def test_common_draw_records_four_ap_values_and_factorial_estimands(self) -> None:
        from scripts.summarize_p2_factorial_screen import run_bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_fixture(root)
            output_dir = root / "out"
            mirror_dir = root / "mirror"
            result = run_bootstrap(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                prediction_dirs=fixture["prediction_dirs"],
                output_dir=output_dir,
                mirror_dir=mirror_dir,
                replicates=7,
                seed=17,
                checkpoint_interval=2,
            )

            rows = result["replicates"]
            self.assertEqual(len(rows), 7)
            self.assertEqual(result["config"]["replicates"], 7)
            self.assertEqual(result["config"]["seed"], 17)
            self.assertTrue(all(len(row["draw_sha256"]) == 64 for row in rows))
            for row in rows:
                self.assertEqual(set(row["ap40"]), {"C", "A", "B", "AB"})
                for condition in ("C", "A", "B", "AB"):
                    self.assertEqual(
                        set(row["ap40"][condition]),
                        {"Pedestrian", "Cyclist", "macro"},
                    )
                macro = {condition: row["ap40"][condition]["macro"] for condition in row["ap40"]}
                effects = row["effects"]
                self.assertAlmostEqual(effects["A_minus_C"], macro["A"] - macro["C"])
                self.assertAlmostEqual(effects["B_minus_C"], macro["B"] - macro["C"])
                self.assertAlmostEqual(effects["AB_minus_C"], macro["AB"] - macro["C"])
                self.assertAlmostEqual(
                    effects["AB_minus_max_A_B"],
                    macro["AB"] - max(macro["A"], macro["B"]),
                )
                self.assertAlmostEqual(
                    effects["interaction"],
                    macro["AB"] - macro["A"] - macro["B"] + macro["C"],
                )

            output = json.loads((output_dir / "factorial_bootstrap.json").read_text(encoding="utf-8"))
            self.assertEqual(output["state"], "complete")
            self.assertEqual(json.loads((mirror_dir / "factorial_bootstrap.json").read_text(encoding="utf-8")), output)
            with (output_dir / "factorial_bootstrap.csv").open(newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 7)
            self.assertIn("interaction", csv_rows[0])
            observed = result["observed"]
            self.assertEqual(observed["ap40"]["C"]["Pedestrian"], 100.0)
            self.assertEqual(observed["ap40"]["C"]["Cyclist"], 100.0)
            self.assertEqual(observed["ap40"]["C"]["macro"], 100.0)
            self.assertEqual(observed["ap40"]["A"]["macro"], 50.0)
            self.assertEqual(observed["ap40"]["B"]["macro"], 50.0)
            self.assertEqual(observed["ap40"]["AB"]["macro"], 0.0)
            self.assertEqual(observed["effects"]["A_minus_C"], -50.0)
            self.assertEqual(observed["effects"]["B_minus_C"], -50.0)
            self.assertEqual(observed["effects"]["AB_minus_C"], -100.0)
            self.assertEqual(observed["effects"]["AB_minus_max_A_B"], -50.0)
            self.assertEqual(observed["effects"]["interaction"], 0.0)
            implementation = result["identity"]["implementation_sha256"]
            self.assertIn("ifdr_yolo/eval/kitti_ap40.py", implementation)
            self.assertIn("ifdr_yolo/eval/prediction_io.py", implementation)
            self.assertIn("ifdr_yolo/data/kitti_types.py", implementation)

    def test_interruption_resume_matches_uninterrupted_and_rejects_identity_change(self) -> None:
        from scripts.summarize_p2_factorial_screen import run_bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_fixture(root)
            interrupted = root / "interrupted"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_bootstrap(
                    split=fixture["split"],
                    label_dir=fixture["label_dir"],
                    image_dir=fixture["image_dir"],
                    prediction_dirs=fixture["prediction_dirs"],
                    output_dir=interrupted,
                    mirror_dir=root / "interrupted-mirror",
                    replicates=9,
                    seed=17,
                    checkpoint_interval=2,
                    stop_after=3,
                )
            checkpoint = json.loads((interrupted / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["state"], "running")
            self.assertEqual(checkpoint["next_replicate"], 3)
            self.assertEqual(checkpoint["owner"]["hostname"].lower(), (os.environ.get("COMPUTERNAME") or checkpoint["owner"]["hostname"]).lower())
            progress_lines = (interrupted / "progress.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(progress_lines[0])["completed"], 0)
            first_progress = json.loads(progress_lines[0])
            self.assertEqual(first_progress["total_replicates"], 9)
            self.assertEqual(first_progress["remaining"], 9)
            self.assertIsNone(first_progress["eta_seconds"])

            uninterrupted = run_bootstrap(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                prediction_dirs=fixture["prediction_dirs"],
                output_dir=root / "clean",
                mirror_dir=root / "clean-mirror",
                replicates=9,
                seed=17,
                checkpoint_interval=2,
            )
            resumed = run_bootstrap(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                prediction_dirs=fixture["prediction_dirs"],
                output_dir=interrupted,
                mirror_dir=root / "interrupted-mirror",
                replicates=9,
                seed=17,
                checkpoint_interval=2,
                resume=True,
            )
            self.assertEqual(resumed["replicates"], uninterrupted["replicates"])

            changed = fixture["prediction_dirs"]["A"] / "000001.txt"
            changed.write_text("1 0.25 0.45 0.30 0.70 0.95\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity|mismatch"):
                run_bootstrap(
                    split=fixture["split"],
                    label_dir=fixture["label_dir"],
                    image_dir=fixture["image_dir"],
                    prediction_dirs=fixture["prediction_dirs"],
                    output_dir=interrupted,
                    mirror_dir=root / "interrupted-mirror",
                    replicates=9,
                    seed=17,
                    checkpoint_interval=2,
                    resume=True,
                )

    def test_resume_rejects_live_other_owner_and_mirror_overlap(self) -> None:
        from scripts import summarize_p2_factorial_screen as module
        from scripts.summarize_p2_factorial_screen import run_bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_fixture(root)
            output = root / "out"
            mirror = root / "mirror"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_bootstrap(
                    split=fixture["split"],
                    label_dir=fixture["label_dir"],
                    image_dir=fixture["image_dir"],
                    prediction_dirs=fixture["prediction_dirs"],
                    output_dir=output,
                    mirror_dir=mirror,
                    replicates=3,
                    checkpoint_interval=2,
                    stop_after=1,
                )
            for checkpoint_path in (output / "checkpoint.json", mirror / "checkpoint.json"):
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                checkpoint["owner"]["pid"] = 123456
                checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            with patch.object(module, "_pid_alive", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "owner|active"):
                    run_bootstrap(
                        split=fixture["split"],
                        label_dir=fixture["label_dir"],
                        image_dir=fixture["image_dir"],
                        prediction_dirs=fixture["prediction_dirs"],
                        output_dir=output,
                        mirror_dir=mirror,
                        replicates=3,
                        checkpoint_interval=2,
                        resume=True,
                    )
            with self.assertRaisesRegex(ValueError, "overlap|distinct"):
                run_bootstrap(
                    split=fixture["split"],
                    label_dir=fixture["label_dir"],
                    image_dir=fixture["image_dir"],
                    prediction_dirs=fixture["prediction_dirs"],
                    output_dir=root / "nested",
                    mirror_dir=root / "nested" / "mirror",
                    replicates=1,
                )

    def test_atomic_write_fsyncs_and_resample_builds_ground_truth_once(self) -> None:
        from scripts import summarize_p2_factorial_screen as module
        from scripts.summarize_p2_factorial_screen import run_bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_fixture(root)
            with patch.object(module.os, "fsync", wraps=module.os.fsync) as fsync:
                run_bootstrap(
                    split=fixture["split"],
                    label_dir=fixture["label_dir"],
                    image_dir=fixture["image_dir"],
                    prediction_dirs=fixture["prediction_dirs"],
                    output_dir=root / "out",
                    mirror_dir=root / "mirror",
                    replicates=1,
                )
                self.assertGreater(fsync.call_count, 0)
            with patch.object(module, "_resample_ground_truth", wraps=module._resample_ground_truth) as ground_truth_resample, patch.object(module, "_resample_predictions", wraps=module._resample_predictions) as prediction_resample:
                run_bootstrap(
                    split=fixture["split"],
                    label_dir=fixture["label_dir"],
                    image_dir=fixture["image_dir"],
                    prediction_dirs=fixture["prediction_dirs"],
                    output_dir=root / "out2",
                    mirror_dir=root / "mirror2",
                    replicates=1,
                )
                self.assertEqual(ground_truth_resample.call_count, 1)
                self.assertEqual(prediction_resample.call_count, 4)

    def test_completed_resume_verifies_primary_mirror_artifact_hashes(self) -> None:
        from scripts.summarize_p2_factorial_screen import run_bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_fixture(root)
            output = root / "out"
            mirror = root / "mirror"
            expected = run_bootstrap(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                prediction_dirs=fixture["prediction_dirs"],
                output_dir=output,
                mirror_dir=mirror,
                replicates=2,
            )
            mirror_json = mirror / "factorial_bootstrap.json"
            mirror_json.write_text(mirror_json.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            recovered = run_bootstrap(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                prediction_dirs=fixture["prediction_dirs"],
                output_dir=output,
                mirror_dir=mirror,
                replicates=2,
                resume=True,
            )
            self.assertEqual(recovered["replicates"], expected["replicates"])

    def test_finalization_damage_is_republished_from_complete_checkpoint(self) -> None:
        from scripts.summarize_p2_factorial_screen import run_bootstrap

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _write_fixture(root)
            output = root / "out"
            mirror = root / "mirror"
            expected = run_bootstrap(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                prediction_dirs=fixture["prediction_dirs"],
                output_dir=output,
                mirror_dir=mirror,
                replicates=2,
            )
            (mirror / "manifest.json").unlink()
            (output / "factorial_bootstrap.csv").write_text("corrupt\n", encoding="utf-8")
            recovered = run_bootstrap(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                prediction_dirs=fixture["prediction_dirs"],
                output_dir=output,
                mirror_dir=mirror,
                replicates=2,
                resume=True,
            )
            self.assertEqual(recovered, expected)
            self.assertTrue((mirror / "manifest.json").is_file())
            self.assertEqual(
                (output / "factorial_bootstrap.csv").read_bytes(),
                (mirror / "factorial_bootstrap.csv").read_bytes(),
            )

    def test_formal_replicates_are_fixed_unless_benchmark_mode_is_explicit(self) -> None:
        from scripts.summarize_p2_factorial_screen import resolve_replicates

        self.assertEqual(resolve_replicates(benchmark=False, requested=None), 10000)
        self.assertEqual(resolve_replicates(benchmark=True, requested=11), 11)
        with self.assertRaisesRegex(ValueError, "formal|10000"):
            resolve_replicates(benchmark=False, requested=11)
