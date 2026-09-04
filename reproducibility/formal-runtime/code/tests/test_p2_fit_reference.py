from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

from ifdr_yolo.experiments.config import load_baseline_config
from ifdr_yolo.data.splits import sha256_file
from ifdr_yolo.experiments.p2_fit_reference import (
    ActiveReferenceRunError,
    REGISTERED_DEVELOPMENT_COUNT,
    REGISTERED_FIT_COUNT,
    P2ReferenceIdentityError,
    P2ReferenceServices,
    build_reference_identity,
    prepare_p2_fit_reference,
    run_p2_fit_reference,
    _hash_code,
    validate_fit_development_split,
    validate_primary_checkpoint,
    validate_plain_p2_model,
    extract_primary_metric,
    prepare_p3p5_fit_reference,
    run_p3p5_fit_reference,
    validate_plain_p3p5_model,
    STAGE9_REFERENCE_EXECUTION_PURPOSE,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_baseline_config(
    ROOT / "configs/experiments/kitti_yolov8m_p2_s17.yaml",
    repository_root=ROOT,
)
P3P5_CONFIG = load_baseline_config(
    ROOT / "configs/experiments/kitti_yolov8m_baseline_s17.yaml",
    repository_root=ROOT,
)


class AuditAdapter:
    """Tiny runtime that emits the Ultralytics provenance files we audit."""

    def __init__(self, *, cache_ids: tuple[str, ...] = ("000000",), write_cache: bool = True, data_override: str | None = None) -> None:
        self.cache_ids = cache_ids
        self.write_cache = write_cache
        self.data_override = data_override
        self.train_args: list[dict[str, object]] = []

    def prepare_model(self, **_: object) -> object:
        return object()

    def train(self, *, run_dir: Path, data_path: Path, args: dict[str, object] | None = None, **_: object) -> Path:
        self.train_args.append(dict(args or {}))
        weights = run_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        (run_dir / "results.csv").write_text("epoch,loss\n1,1.0\n", encoding="utf-8")
        args_data = self.data_override if self.data_override is not None else str(data_path)
        (run_dir / "args.yaml").write_text(f"data: {args_data}\n", encoding="utf-8")
        if self.write_cache:
            labels = [
                {"im_file": str(run_dir / "view" / "images" / "train" / f"{image_id}.png")}
                for image_id in self.cache_ids
            ]
            (run_dir / "train.cache").write_text(json.dumps({"labels": labels}), encoding="utf-8")
        return weights / "last.pt"

    def predict(self, *, output_dir: Path, image_paths: tuple[Path, ...], **_: object) -> Path:
        labels = output_dir / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        for image in image_paths:
            (labels / f"{image.stem}.txt").write_text("", encoding="utf-8")
        return labels


class P2FitReferenceTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[object, Path, Path, Path, Path, object]:
        generated = root / "generated"
        for split in ("train",):
            (generated / "images" / split).mkdir(parents=True)
            (generated / "labels" / split).mkdir(parents=True)
        for image_id in ("000000", "000001"):
            (generated / "images" / "train" / f"{image_id}.png").write_bytes(b"png")
            (generated / "labels" / "train" / f"{image_id}.txt").write_text("", encoding="utf-8")
        full = root / "full.txt"
        fit = root / "fit.txt"
        dev = root / "dev.txt"
        full.write_text("000000\n000001\n", encoding="utf-8")
        fit.write_text("000000\n", encoding="utf-8")
        dev.write_text("000001\n", encoding="utf-8")
        raw_images = root / "raw-images"
        raw_labels = root / "raw-labels"
        raw_images.mkdir()
        raw_labels.mkdir()
        config = replace(
            CONFIG,
            paths=replace(
                CONFIG.paths,
                generated_data=generated,
                train_ids=full,
                raw_images=raw_images,
                raw_labels=raw_labels,
            ),
            training=replace(CONFIG.training, epochs=3),
        )
        identity = build_reference_identity(
            model_sha256=sha256_file(CONFIG.paths.model),
            pretrained_sha256=sha256_file(CONFIG.initialization.pretrained),
            fit_ids_sha256=sha256_file(fit),
            development_ids_sha256=sha256_file(dev),
            config_sha256=sha256_file(CONFIG.source_path),
            code_sha256=_hash_code(ROOT),
        )
        return config, fit, dev, identity, generated, full

    def _p3_fixture(self, root: Path) -> tuple[object, Path, Path]:
        generated = root / "generated"
        (generated / "images" / "train").mkdir(parents=True)
        (generated / "labels" / "train").mkdir(parents=True)
        for image_id in ("000000", "000001"):
            (generated / "images" / "train" / f"{image_id}.png").write_bytes(b"png")
            (generated / "labels" / "train" / f"{image_id}.txt").write_text("", encoding="utf-8")
        full = root / "full.txt"
        fit = root / "fit.txt"
        dev = root / "dev.txt"
        full.write_text("000000\n000001\n", encoding="utf-8")
        fit.write_text("000000\n", encoding="utf-8")
        dev.write_text("000001\n", encoding="utf-8")
        raw_images = root / "raw-images"
        raw_labels = root / "raw-labels"
        raw_images.mkdir()
        raw_labels.mkdir()
        config = replace(
            P3P5_CONFIG,
            paths=replace(
                P3P5_CONFIG.paths,
                generated_data=generated,
                train_ids=full,
                raw_images=raw_images,
                raw_labels=raw_labels,
            ),
        )
        return config, fit, dev

    @staticmethod
    def _metrics(split_path: Path) -> dict[str, object]:
        return {
            "evaluator": "ifdr_yolo.kitti_ap40",
            "split_sha256": sha256_file(split_path),
            "split_count": 1,
            "classes": {
                "Car": {"moderate": {"ap40": 1.0}},
                "Pedestrian": {"moderate": {"ap40": 80.0}},
                "Cyclist": {"moderate": {"ap40": 90.0}},
            },
        }

    def test_rejects_original_3712_image_training_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full.txt"
            full.write_text("".join(f"{i:06d}\n" for i in range(3712)), encoding="utf-8")
            fit = root / "fit.txt"
            dev = root / "dev.txt"
            fit.write_text(full.read_text(encoding="utf-8"), encoding="utf-8")
            dev.write_text("003712\n", encoding="utf-8")
            config = replace(CONFIG, paths=replace(CONFIG.paths, train_ids=full))
            with self.assertRaisesRegex(ValueError, "fit split count"):
                validate_fit_development_split(config, fit, dev)

    def test_rejects_fit_development_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full.txt"
            full.write_text("000000\n000001\n", encoding="utf-8")
            fit = root / "fit.txt"
            dev = root / "dev.txt"
            fit.write_text("000000\n", encoding="utf-8")
            dev.write_text("000000\n", encoding="utf-8")
            config = replace(CONFIG, paths=replace(CONFIG.paths, train_ids=full))
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_fit_development_split(
                    config,
                    fit,
                    dev,
                    expected_fit_count=1,
                    expected_development_count=1,
                )

    def test_rejects_wrong_registered_count_or_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit = root / "fit.txt"
            dev = root / "dev.txt"
            fit.write_text("000000\n", encoding="utf-8")
            dev.write_text("000001\n", encoding="utf-8")
            full = root / "full.txt"
            full.write_text("000000\n000001\n", encoding="utf-8")
            config = replace(CONFIG, paths=replace(CONFIG.paths, train_ids=full))
            with self.assertRaisesRegex(ValueError, "full training split count"):
                validate_fit_development_split(config, fit, dev)

    def test_rejects_non_plain_p2_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "ifdr.yaml"
            model.write_text("# ifdr\nnc: 3\n", encoding="utf-8")
            config = replace(CONFIG, paths=replace(CONFIG.paths, model=model))
            with self.assertRaisesRegex(ValueError, "plain P2"):
                validate_plain_p2_model(config)

    def test_rejects_best_as_primary_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "last.pt"):
                validate_primary_checkpoint(checkpoint)

    def test_live_owner_rejected_and_stale_identity_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = build_reference_identity(
                model_sha256="a" * 64,
                pretrained_sha256="b" * 64,
                fit_ids_sha256="c" * 64,
                development_ids_sha256="d" * 64,
                config_sha256="e" * 64,
                code_sha256="f" * 64,
            )
            mirror = root / "mirror"
            # A tiny synthetic split is accepted when the expected counts and
            # hashes are supplied explicitly; the registered guard remains the
            # default for the production entry point.
            full = root / "full.txt"
            full.write_text("000000\n000001\n", encoding="utf-8")
            fit = root / "fit.txt"
            dev = root / "dev.txt"
            fit.write_text("000000\n", encoding="utf-8")
            dev.write_text("000001\n", encoding="utf-8")
            config = replace(CONFIG, paths=replace(CONFIG.paths, train_ids=full))
            job = root / "job"
            job.mkdir()
            (job / "reference_identity.json").write_text(
                json.dumps({"identity": {"wrong": True}, "identity_sha256": "0" * 64}),
                encoding="utf-8",
            )
            with self.assertRaises(P2ReferenceIdentityError):
                prepare_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=job,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                )

    def test_stale_job_is_resumable_and_same_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev, identity, _, _ = self._fixture(root)
            job = root / "job"
            mirror = root / "mirror"
            first = prepare_p2_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=job,
                mirror_dir=mirror,
                fit_ids=fit,
                development_ids=dev,
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
                identity=identity,
            )
            self.assertTrue(
                os.path.samefile(
                    root / "generated" / "images" / "train" / "000000.png",
                    job / "view" / "images" / "train" / "000000.png",
                )
            )
            self.assertTrue(
                os.path.samefile(
                    root / "generated" / "images" / "train" / "000001.png",
                    job / "view" / "images" / "val" / "000001.png",
                )
            )
            (job / "weights").mkdir()
            (job / "weights" / "last.pt").write_bytes(b"last")
            (job / "results.csv").write_text("epoch,loss\n1,1.0\n", encoding="utf-8")
            status = json.loads((job / "status.json").read_text(encoding="utf-8"))
            status.update({"state": "failed", "pid": 999999, "hostname": socket.gethostname()})
            (job / "status.json").write_text(json.dumps(status), encoding="utf-8")
            resumed = prepare_p2_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=job,
                mirror_dir=mirror,
                fit_ids=fit,
                development_ids=dev,
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
                identity=identity,
                pid_alive=lambda _: False,
            )
            self.assertTrue(resumed.resumable)
            self.assertEqual(resumed.output_dir, first.output_dir)

            bad_identity = build_reference_identity(
                model_sha256=identity.model_sha256,
                pretrained_sha256=identity.pretrained_sha256,
                fit_ids_sha256=identity.fit_ids_sha256,
                development_ids_sha256=identity.development_ids_sha256,
                config_sha256=identity.config_sha256,
                code_sha256="e" * 64,
            )
            with self.assertRaises(P2ReferenceIdentityError):
                prepare_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=job,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=bad_identity,
                    pid_alive=lambda _: False,
                )

            status.update({"state": "running", "pid": os.getpid()})
            (job / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaises(ActiveReferenceRunError):
                prepare_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=job,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                    pid_alive=lambda _: True,
                )

            status.update({"state": "running", "pid": 999999, "hostname": "other-host"})
            (job / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaises(P2ReferenceIdentityError):
                prepare_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=job,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                    pid_alive=lambda _: False,
                )

    def test_interrupt_resume_is_equivalent_and_publishes_progress(self) -> None:
        class FakeAdapter:
            def __init__(self, interrupt: bool) -> None:
                self.interrupt = interrupt

            def prepare_model(self, **_: object) -> object:
                return object()

            def train(self, *, run_dir: Path, **kwargs: object) -> Path:
                weights = run_dir / "weights"
                weights.mkdir(parents=True, exist_ok=True)
                (weights / "best.pt").write_bytes(b"best")
                (weights / "last.pt").write_bytes(b"last-1")
                (run_dir / "results.csv").write_text("epoch,loss\n1,1.0\n", encoding="utf-8")
                data_path = kwargs.get("data_path")
                (run_dir / "args.yaml").write_text(f"data: {data_path}\n", encoding="utf-8")
                (run_dir / "train.cache").write_text(
                    json.dumps({"labels": [{"im_file": str(run_dir / "view" / "images" / "train" / "000000.png")}] }),
                    encoding="utf-8",
                )
                if self.interrupt:
                    time.sleep(5.5)
                    raise RuntimeError("intentional interruption")
                for epoch in (2, 3):
                    with (run_dir / "results.csv").open("a", encoding="utf-8") as output:
                        output.write(f"{epoch},1.0\n")
                    (weights / "last.pt").write_bytes(f"last-{epoch}".encode())
                return weights / "best.pt"

            def predict(self, *, output_dir: Path, image_paths: tuple[Path, ...], **_: object) -> Path:
                labels = output_dir / "labels"
                labels.mkdir(parents=True, exist_ok=True)
                for image in image_paths:
                    (labels / f"{image.stem}.txt").write_text("", encoding="utf-8")
                return labels

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev, identity, _, _ = self._fixture(root)
            output = root / "job"
            mirror = root / "mirror"
            interrupted = P2ReferenceServices(
                adapter=FakeAdapter(True),
                evaluate=lambda **_: {"classes": {}, "split_count": 1},
                pid_alive=lambda _: False,
            )
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                run_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=output,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    mode="smoke",
                    services=interrupted,
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                )
            with self.assertRaisesRegex(ValueError, "requires --resume"):
                run_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=output,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    mode="smoke",
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                    services=interrupted,
                )
            failed_status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(failed_status["current_epoch"], 1)
            self.assertEqual(json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))["files"][-1]["name"], "status.json")

            def resume(checkpoint: Path, job: object, **_: object) -> Path:
                weights = checkpoint.parent
                with (Path(job.output_dir) / "results.csv").open("a", encoding="utf-8") as output_file:
                    for epoch in (2, 3):
                        output_file.write(f"{epoch},1.0\n")
                (weights / "last.pt").write_bytes(b"last-3")
                return weights / "last.pt"

            resumed = P2ReferenceServices(
                adapter=FakeAdapter(False),
                evaluate=lambda **_: {"classes": {}, "split_count": 1},
                pid_alive=lambda _: False,
                resume_training=resume,
            )
            final = run_p2_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=output,
                mirror_dir=mirror,
                fit_ids=fit,
                development_ids=dev,
                mode="smoke",
                resume=True,
                services=resumed,
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
                identity=identity,
            )
            self.assertEqual(final.output_dir, output.resolve())
            self.assertEqual(json.loads((output / "status.json").read_text(encoding="utf-8"))["state"], "complete")
            self.assertEqual((output / "results.csv").read_text(encoding="utf-8"), "epoch,loss\n1,1.0\n2,1.0\n3,1.0\n")
            with self.assertRaisesRegex(ValueError, "already complete"):
                run_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=output,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    mode="smoke",
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                    services=resumed,
                )

            clean_output = root / "clean-job"
            clean_mirror = root / "clean-mirror"
            clean = run_p2_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=clean_output,
                mirror_dir=clean_mirror,
                fit_ids=fit,
                development_ids=dev,
                mode="smoke",
                services=P2ReferenceServices(
                    adapter=FakeAdapter(False),
                    evaluate=lambda **_: {"classes": {}, "split_count": 1},
                    pid_alive=lambda _: False,
                ),
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
                identity=identity,
            )
            self.assertEqual((clean.output_dir / "results.csv").read_bytes(), (output / "results.csv").read_bytes())

    def test_training_audit_requires_args_and_nonempty_train_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev, identity, _, _ = self._fixture(root)
            with self.assertRaisesRegex((FileNotFoundError, ValueError), "cache"):
                run_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=root / "missing-cache",
                    mirror_dir=root / "missing-cache-mirror",
                    fit_ids=fit,
                    development_ids=dev,
                    mode="smoke",
                    services=P2ReferenceServices(
                        adapter=AuditAdapter(write_cache=False),
                        evaluate=lambda **_: self.fail("evaluation must not run without train cache"),
                        pid_alive=lambda _: False,
                    ),
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                )

    def test_training_audit_rejects_development_in_train_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev, identity, _, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "development|overlap|train cache"):
                run_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=root / "leaky-cache",
                    mirror_dir=root / "leaky-cache-mirror",
                    fit_ids=fit,
                    development_ids=dev,
                    mode="smoke",
                    services=P2ReferenceServices(
                        adapter=AuditAdapter(cache_ids=("000000", "000001")),
                        evaluate=lambda **_: self.fail("evaluation must not run for a leaky train cache"),
                        pid_alive=lambda _: False,
                    ),
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                )

    def test_training_audit_rejects_args_data_path_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev, identity, _, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "data path"):
                run_p2_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=root / "wrong-args",
                    mirror_dir=root / "wrong-args-mirror",
                    fit_ids=fit,
                    development_ids=dev,
                    mode="smoke",
                    services=P2ReferenceServices(
                        adapter=AuditAdapter(data_override=str(root / "wrong.yaml")),
                        evaluate=lambda **_: self.fail("evaluation must not run for wrong args data path"),
                        pid_alive=lambda _: False,
                    ),
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    identity=identity,
                )

    def test_training_audit_publishes_observed_ids_and_checkpoint_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev, identity, _, _ = self._fixture(root)
            output = root / "audited"
            mirror = root / "audited-mirror"
            run_p2_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=output,
                mirror_dir=mirror,
                fit_ids=fit,
                development_ids=dev,
                mode="smoke",
                services=P2ReferenceServices(
                    adapter=AuditAdapter(),
                    evaluate=lambda **_: {"classes": {}, "split_count": 1},
                    pid_alive=lambda _: False,
                ),
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
                identity=identity,
            )
            self.assertEqual((output / "observed_train_ids.txt").read_text(encoding="utf-8"), "000000\n")
            audit = json.loads((output / "post_training_leakage_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["observed_train_count"], 1)
            self.assertEqual(audit["intersection_count"], 0)
            provenance = json.loads((output / "checkpoint_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["checkpoint_role"], "last.pt")
            self.assertEqual(provenance["post_training_audit_sha256"], sha256_file(output / "post_training_leakage_audit.json"))
            mirrored = json.loads((mirror / "manifest.json").read_text(encoding="utf-8"))
            names = {entry["name"] for entry in mirrored["files"]}
            self.assertTrue({"observed_train_ids.txt", "post_training_leakage_audit.json", "checkpoint_provenance.json"} <= names)

    def test_p3p5_registration_requires_plain_baseline_contract(self) -> None:
        validate_plain_p3p5_model(P3P5_CONFIG)
        with self.assertRaisesRegex(ValueError, "variant=baseline"):
            validate_plain_p3p5_model(CONFIG)
        with self.assertRaisesRegex(ValueError, "extra initialization"):
            validate_plain_p3p5_model(replace(P3P5_CONFIG, initialization=CONFIG.initialization))
        with self.assertRaisesRegex(ValueError, "epochs=300"):
            validate_plain_p3p5_model(replace(P3P5_CONFIG, training=replace(P3P5_CONFIG.training, epochs=1)))

    def test_p3p5_seed0_benchmark_context_allows_only_registered_30_epoch_protocol(self) -> None:
        seed0 = replace(P3P5_CONFIG, experiment=replace(P3P5_CONFIG.experiment, seed=0),
                        training=replace(P3P5_CONFIG.training, epochs=30))
        context = SimpleNamespace(identity={"arm": "P3P5_CONTROL", "seed": 0, "formal_epochs": 30})
        validate_plain_p3p5_model(seed0, benchmark_context=context)
        with self.assertRaisesRegex(ValueError, "seed=17"):
            validate_plain_p3p5_model(seed0)
        with self.assertRaisesRegex(ValueError, "benchmark identity"):
            validate_plain_p3p5_model(seed0, benchmark_context=SimpleNamespace(identity={"arm": "DCLI", "seed": 0, "formal_epochs": 30}))

    def test_stage9_seed0_reference_purpose_is_narrow_and_identity_bound(self) -> None:
        p3p5_seed0 = replace(
            P3P5_CONFIG,
            experiment=replace(P3P5_CONFIG.experiment, seed=0),
            training=replace(P3P5_CONFIG.training, epochs=30),
        )
        validate_plain_p3p5_model(
            p3p5_seed0,
            execution_purpose=STAGE9_REFERENCE_EXECUTION_PURPOSE,
        )
        with self.assertRaisesRegex(ValueError, "Stage9 reference"):
            validate_plain_p3p5_model(
                P3P5_CONFIG,
                execution_purpose=STAGE9_REFERENCE_EXECUTION_PURPOSE,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p2_config, fit, dev, _, _, _ = self._fixture(root)
            p2_seed0 = replace(
                p2_config,
                experiment=replace(p2_config.experiment, seed=0),
                training=replace(
                    p2_config.training,
                    epochs=30,
                    imgsz=640,
                    batch=16,
                    workers=8,
                    deterministic=True,
                ),
            )
            job = prepare_p2_fit_reference(
                p2_seed0,
                repository_root=ROOT,
                output_dir=root / "stage9-p2",
                mirror_dir=root / "stage9-p2-mirror",
                fit_ids=fit,
                development_ids=dev,
                run_mode="full",
                execution_purpose=STAGE9_REFERENCE_EXECUTION_PURPOSE,
                pid_alive=lambda _: False,
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
            )
            self.assertEqual(job.identity.execution_purpose, STAGE9_REFERENCE_EXECUTION_PURPOSE)
            self.assertEqual(job.identity.as_dict()["execution_purpose"], STAGE9_REFERENCE_EXECUTION_PURPOSE)

            invalid = replace(p2_seed0, training=replace(p2_seed0.training, workers=0))
            with self.assertRaisesRegex(ValueError, "Stage9 reference"):
                prepare_p2_fit_reference(
                    invalid,
                    repository_root=ROOT,
                    output_dir=root / "invalid-stage9-p2",
                    mirror_dir=root / "invalid-stage9-p2-mirror",
                    fit_ids=fit,
                    development_ids=dev,
                    run_mode="full",
                    execution_purpose=STAGE9_REFERENCE_EXECUTION_PURPOSE,
                    pid_alive=lambda _: False,
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                )

    def test_p3p5_identity_binds_role_and_run_mode_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev = self._p3_fixture(root)
            output = root / "job"
            mirror = root / "mirror"
            smoke = prepare_p3p5_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=output,
                mirror_dir=mirror,
                fit_ids=fit,
                development_ids=dev,
                run_mode="smoke",
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
                pid_alive=lambda _: False,
            )
            self.assertEqual(smoke.identity.model_role, "plain_p3p5")
            self.assertEqual(smoke.identity.run_mode, "smoke")
            with self.assertRaisesRegex(P2ReferenceIdentityError, "identity"):
                prepare_p3p5_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=output,
                    mirror_dir=mirror,
                    fit_ids=fit,
                    development_ids=dev,
                    run_mode="full",
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                    pid_alive=lambda _: False,
                )

    def test_p3p5_smoke_uses_640_and_full_preserves_300_640_batch16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev = self._p3_fixture(root)

            def evaluate(**kwargs: object) -> dict[str, object]:
                return self._metrics(Path(kwargs["split_path"]))

            smoke_adapter = AuditAdapter()
            run_p3p5_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=root / "smoke",
                mirror_dir=root / "smoke-mirror",
                fit_ids=fit,
                development_ids=dev,
                mode="smoke",
                services=P2ReferenceServices(
                    adapter=smoke_adapter,
                    evaluate=evaluate,
                    pid_alive=lambda _: False,
                ),
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
            )
            self.assertEqual(smoke_adapter.train_args[0]["epochs"], 1)
            self.assertEqual(smoke_adapter.train_args[0]["imgsz"], 640)
            self.assertEqual(smoke_adapter.train_args[0]["batch"], 2)
            self.assertEqual(smoke_adapter.train_args[0]["workers"], 0)
            self.assertTrue(smoke_adapter.train_args[0]["amp"])
            self.assertEqual(smoke_adapter.train_args[0]["save_period"], -1)

            full_adapter = AuditAdapter()
            run_p3p5_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=root / "full",
                mirror_dir=root / "full-mirror",
                fit_ids=fit,
                development_ids=dev,
                mode="full",
                services=P2ReferenceServices(
                    adapter=full_adapter,
                    evaluate=evaluate,
                    preflight=lambda **_: None,
                    pid_alive=lambda _: False,
                ),
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
            )
            self.assertEqual(full_adapter.train_args[0]["epochs"], 300)
            self.assertEqual(full_adapter.train_args[0]["imgsz"], 640)
            self.assertEqual(full_adapter.train_args[0]["batch"], 16)

    def test_p3p5_interrupt_resume_same_directory_matches_clean_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev = self._p3_fixture(root)

            class InterruptAdapter(AuditAdapter):
                def __init__(self) -> None:
                    super().__init__()
                    self.interrupt = True

                def train(self, **kwargs: object) -> Path:
                    checkpoint = super().train(**kwargs)
                    if self.interrupt:
                        raise RuntimeError("intentional P3-P5 interruption")
                    return checkpoint

            def evaluate(**kwargs: object) -> dict[str, object]:
                return self._metrics(Path(kwargs["split_path"]))

            interrupted = InterruptAdapter()
            interrupted_services = P2ReferenceServices(
                adapter=interrupted,
                evaluate=evaluate,
                pid_alive=lambda _: False,
            )
            with self.assertRaisesRegex(RuntimeError, "intentional P3-P5"):
                run_p3p5_fit_reference(
                    config,
                    repository_root=ROOT,
                    output_dir=root / "interrupted",
                    mirror_dir=root / "interrupted-mirror",
                    fit_ids=fit,
                    development_ids=dev,
                    mode="smoke",
                    services=interrupted_services,
                    expected_fit_count=1,
                    expected_development_count=1,
                    expected_fit_sha256=sha256_file(fit),
                    expected_development_sha256=sha256_file(dev),
                )
            interrupted.interrupt = False
            interrupted_services = replace(
                interrupted_services,
                resume_training=lambda checkpoint, job, **_: checkpoint,
            )
            run_p3p5_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=root / "interrupted",
                mirror_dir=root / "interrupted-mirror",
                fit_ids=fit,
                development_ids=dev,
                mode="smoke",
                resume=True,
                services=interrupted_services,
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
            )
            run_p3p5_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=root / "clean",
                mirror_dir=root / "clean-mirror",
                fit_ids=fit,
                development_ids=dev,
                mode="smoke",
                services=P2ReferenceServices(
                    adapter=AuditAdapter(),
                    evaluate=evaluate,
                    pid_alive=lambda _: False,
                ),
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
            )
            for name in ("results.csv", "metrics_ap40.json", "primary_metric.json"):
                self.assertEqual(
                    (root / "interrupted" / name).read_bytes(),
                    (root / "clean" / name).read_bytes(),
                )

    def test_primary_metric_extracts_two_class_macro(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "development_ids.txt"
            split.write_text("000001\n", encoding="utf-8")
            metric = extract_primary_metric(
                self._metrics(split),
                split_sha256=sha256_file(split),
                checkpoint_sha256="a" * 64,
                split_count=1,
                identity_sha256="b" * 64,
            )
            self.assertEqual(metric["pedestrian_moderate_ap_r40"], 80.0)
            self.assertEqual(metric["cyclist_moderate_ap_r40"], 90.0)
            self.assertEqual(metric["macro_ap_r40"], 85.0)
            self.assertEqual(metric["last_pt_sha256"], "a" * 64)

    def test_p3p5_mirrors_primary_metric_and_raw_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, fit, dev = self._p3_fixture(root)

            def evaluate(**kwargs: object) -> dict[str, object]:
                return self._metrics(Path(kwargs["split_path"]))

            run_p3p5_fit_reference(
                config,
                repository_root=ROOT,
                output_dir=root / "job",
                mirror_dir=root / "mirror",
                fit_ids=fit,
                development_ids=dev,
                mode="smoke",
                services=P2ReferenceServices(
                    adapter=AuditAdapter(),
                    evaluate=evaluate,
                    pid_alive=lambda _: False,
                ),
                expected_fit_count=1,
                expected_development_count=1,
                expected_fit_sha256=sha256_file(fit),
                expected_development_sha256=sha256_file(dev),
            )
            for name in ("metrics_ap40.json", "primary_metric.json", "checkpoint_provenance.json", "status.json"):
                primary = root / "job" / name
                mirror = root / "mirror" / name
                self.assertEqual(primary.read_bytes(), mirror.read_bytes())
            manifest = json.loads((root / "mirror" / "manifest.json").read_text(encoding="utf-8"))
            names = {entry["name"] for entry in manifest["files"]}
            self.assertTrue({"metrics_ap40.json", "primary_metric.json", "checkpoint_provenance.json", "status.json"} <= names)
            self.assertNotIn("P2", (root / "job" / "status.json").read_text(encoding="utf-8"))

    def test_cli_exposes_fixed_output_and_resume_flags(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/run_p2_fit_reference.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--fit-ids", completed.stdout)
        self.assertIn("--mirror-dir", completed.stdout)
        self.assertIn("--resume", completed.stdout)
        self.assertIn("--execution-purpose", completed.stdout)

        p3p5 = subprocess.run(
            [sys.executable, "scripts/run_p3p5_fit_reference.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(p3p5.returncode, 0, p3p5.stderr)
        self.assertIn("--execution-purpose", p3p5.stdout)


if __name__ == "__main__":
    unittest.main()
