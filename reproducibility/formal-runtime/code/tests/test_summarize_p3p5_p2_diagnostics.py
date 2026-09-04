from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


def _fixture(root: Path) -> dict[str, Path]:
    image_dir = root / "images"
    label_dir = root / "labels"
    reference_dir = root / "reference"
    candidate_dir = root / "candidate"
    for directory in (image_dir, label_dir, reference_dir, candidate_dir):
        directory.mkdir()
    ids = tuple(f"{index:06d}" for index in range(1, 7))
    for image_id in ids:
        Image.new("RGB", (100, 100)).save(image_dir / f"{image_id}.png")
        (label_dir / f"{image_id}.txt").write_text(
            "Pedestrian 0 0 0 10 10 30 50 1 1 1 0 0 10 0\n"
            "Cyclist 0 0 0 50 10 70 50 1 1 1 0 0 10 0\n",
            encoding="utf-8",
        )
        (reference_dir / f"{image_id}.txt").write_text(
            "1 0.20 0.30 0.20 0.40 0.90\n"
            "2 0.60 0.30 0.20 0.40 0.90\n",
            encoding="utf-8",
        )
        (candidate_dir / f"{image_id}.txt").write_text(
            "1 0.20 0.30 0.20 0.40 0.90\n",
            encoding="utf-8",
        )
    split = root / "development.txt"
    split.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return {
        "split": split,
        "image_dir": image_dir,
        "label_dir": label_dir,
        "reference_dir": reference_dir,
        "candidate_dir": candidate_dir,
    }


class P3P5P2DiagnosticsTest(unittest.TestCase):
    def test_observed_strata_and_macro_are_bound_to_two_prediction_manifests(self) -> None:
        from scripts.summarize_p3p5_p2_diagnostics import run_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root)
            result = run_diagnostics(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                reference_dir=fixture["reference_dir"],
                candidate_dir=fixture["candidate_dir"],
                output_dir=root / "out",
                mirror_dir=root / "mirror",
                replicates=5,
                seed=17,
                mode="benchmark",
                strict_registered_identities=False,
            )
            observed = result["observed"]
            self.assertEqual(observed["moderate"]["reference"]["Pedestrian"]["ap40"], 100.0)
            self.assertEqual(observed["moderate"]["candidate"]["Cyclist"]["ap40"], 0.0)
            self.assertAlmostEqual(observed["moderate"]["delta"]["macro"], -50.0)
            self.assertIn("small_25_40", observed["strata"]["height"])
            self.assertIn("far_gt_40m", observed["strata"]["depth"])
            self.assertIn("tp", observed["moderate"]["reference"]["Pedestrian"])
            self.assertIn("fn", observed["moderate"]["candidate"]["Pedestrian"])
            self.assertIn("localization_error", observed["moderate"]["reference"]["Pedestrian"])
            self.assertEqual(len(result["replicates"]), 5)
            self.assertTrue(all("moderate_macro_delta" in row for row in result["replicates"]))
            self.assertNotEqual(
                result["identity"]["prediction_manifest_sha256"]["reference"],
                result["identity"]["prediction_manifest_sha256"]["candidate"],
            )
            self.assertTrue((root / "out" / "diagnostics.json").is_file())
            self.assertTrue((root / "out" / "diagnostics.csv").is_file())
            self.assertTrue((root / "out" / "manifest.json").is_file())
            manifest = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))
            record_artifact = manifest["artifacts"]["records.jsonl"]
            self.assertEqual(record_artifact["primary_sha256"], record_artifact["mirror_sha256"])

    def test_interruption_resume_is_byte_equivalent_and_identity_mismatch_fails_closed(self) -> None:
        from scripts.summarize_p3p5_p2_diagnostics import run_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root)
            kwargs = dict(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                reference_dir=fixture["reference_dir"],
                candidate_dir=fixture["candidate_dir"],
                replicates=7,
                seed=17,
                mode="benchmark",
                strict_registered_identities=False,
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_diagnostics(
                    **kwargs,
                    output_dir=root / "interrupted",
                    mirror_dir=root / "interrupted-mirror",
                    stop_after=3,
                )
            clean = run_diagnostics(
                **kwargs,
                output_dir=root / "clean",
                mirror_dir=root / "clean-mirror",
            )
            resumed = run_diagnostics(
                **kwargs,
                output_dir=root / "interrupted",
                mirror_dir=root / "interrupted-mirror",
                resume=True,
            )
            self.assertEqual(resumed["replicates"], clean["replicates"])
            self.assertEqual(
                (root / "clean" / "diagnostics.json").read_bytes(),
                (root / "interrupted" / "diagnostics.json").read_bytes(),
            )
            (fixture["candidate_dir"] / "000001.txt").write_text(
                "1 0.20 0.30 0.20 0.40 0.80\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "identity|mismatch"):
                run_diagnostics(
                    **kwargs,
                    output_dir=root / "interrupted",
                    mirror_dir=root / "interrupted-mirror",
                    resume=True,
                )

    def test_resume_rejects_mirror_journal_tamper(self) -> None:
        from scripts.summarize_p3p5_p2_diagnostics import run_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root)
            kwargs = dict(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                reference_dir=fixture["reference_dir"],
                candidate_dir=fixture["candidate_dir"],
                output_dir=root / "out",
                mirror_dir=root / "mirror",
                replicates=5,
                seed=17,
                mode="benchmark",
                strict_registered_identities=False,
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_diagnostics(**kwargs, stop_after=2)
            mirror_records = root / "mirror" / "records.jsonl"
            lines = mirror_records.read_text(encoding="utf-8").splitlines()
            record = json.loads(lines[1])
            record["draw_sha256"] = "0" * 64
            lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
            mirror_records.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint prefixes|differ"):
                run_diagnostics(**kwargs, resume=True)

    def test_resume_trims_one_sided_legal_tail_after_checkpoint(self) -> None:
        from scripts.summarize_p3p5_p2_diagnostics import run_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root)
            interrupted = root / "interrupted"
            interrupted_mirror = root / "interrupted-mirror"
            kwargs = dict(
                split=fixture["split"],
                label_dir=fixture["label_dir"],
                image_dir=fixture["image_dir"],
                reference_dir=fixture["reference_dir"],
                candidate_dir=fixture["candidate_dir"],
                replicates=5,
                seed=17,
                mode="benchmark",
                strict_registered_identities=False,
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_diagnostics(
                    **kwargs,
                    output_dir=interrupted,
                    mirror_dir=interrupted_mirror,
                    stop_after=2,
                )
            continuous = run_diagnostics(
                **kwargs,
                output_dir=root / "continuous",
                mirror_dir=root / "continuous-mirror",
            )
            continuous_lines = (root / "continuous" / "records.jsonl").read_text(encoding="utf-8").splitlines()
            with (interrupted / "records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(continuous_lines[3] + "\n")
            resumed = run_diagnostics(
                **kwargs,
                output_dir=interrupted,
                mirror_dir=interrupted_mirror,
                resume=True,
            )
            self.assertEqual(resumed["replicates"], continuous["replicates"])
            self.assertEqual(
                (root / "continuous" / "diagnostics.json").read_bytes(),
                (interrupted / "diagnostics.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
