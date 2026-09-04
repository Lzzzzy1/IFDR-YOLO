"""Measure repaired Stage11 slices for ten runs and matched cost for seed0."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import socket
import stat
import subprocess
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import TypedDict


EXPECTED_HOST = "autodl-container-b0b141a97e-0bf7a394"
EXPECTED_GPU_UUID = "GPU-1cad8e35-3692-fb69-b736-d79060a3276b"
EXPECTED_PACKAGE_MANIFEST_SHA256 = "5eff8f3a18f9dd2b579aea323dee9f0e8407b45334cc7df264b707c8e9b771b3"
EXPECTED_DEVELOPMENT_IDS_SHA256 = "b1b6b6ee7e5398e93868fab407a2e8a86a53c753667002ef9b8381734ef2cda8"
EXPECTED_FIT_IDS_SHA256 = "50a1f8d72b747d1e2f460a4e0a355469484d33dbf1e9928b64ba6594c1873362"
EXPECTED_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CODE_ROOT = Path(
    "/root/autodl-tmp/jobs/stage11-v179-ordinal1-e8bb21cc47a945f78b1aac38ce70231a/runtime/code"
)
RAW_IMAGE_ROOT = Path("/root/autodl-tmp/kitti_project/kitti_raw/training/image_2/training/image_2")
RAW_LABEL_ROOT = Path("/root/autodl-tmp/kitti_project/kitti_raw/training/label_2/training/label_2")


class RunSpec(TypedDict):
    ordinal: int
    seed: int
    candidate: str
    directory: str
    output_name: str
    root: str
    metrics_sha256: str
    results_sha256: str
    terminal_sha256: str


RUN_SPECS: tuple[RunSpec, ...] = (
    {
        "ordinal": 1,
        "seed": 0,
        "candidate": "PLAIN_P2",
        "directory": "seed0-plain",
        "output_name": "seed0-plain-p2",
        "root": "/root/autodl-tmp/jobs/stage11-v179-ordinal1-e8bb21cc47a945f78b1aac38ce70231a",
        "metrics_sha256": "c85f31bf44d13319590376d492123c6ad39f7ff9593549ac2774bc93531d329d",
        "results_sha256": "2542cdc16b51f4bc2e243e2d04110b42be5be80dabeced79e76af57c72667034",
        "terminal_sha256": "da7871766f84d0d4753e50bf246f643e4dd7c4fa6e71988af6ba66c11b9fbfcd",
    },
    {
        "ordinal": 2,
        "seed": 0,
        "candidate": "DCLI",
        "directory": "seed0-dcli",
        "output_name": "seed0-dcli",
        "root": "/root/autodl-tmp/jobs/stage11-v186-ordinal2-0e45b8675b904a468ae4fff3b9cddffb",
        "metrics_sha256": "b8379e4ba6548477518d094270fbf5b3eb64690471a88b0f4dcef047d898abe5",
        "results_sha256": "6d04a048b6595442a5eebbe7ff7487bf6c9dcf5bb8ec6b32a6464807e96ba38c",
        "terminal_sha256": "38e4160bcef65c4e28e16cb0508f075074287d80305845dc1fe6fcec03852346",
    },
    {
        "ordinal": 3,
        "seed": 1,
        "candidate": "DCLI",
        "directory": "seed1-dcli",
        "output_name": "seed1-dcli",
        "root": "/root/autodl-tmp/jobs/stage11-v188-ordinal3-40a36fc43a554d2c9c0b1042a8e46545",
        "metrics_sha256": "f0a64df4fa96650202dbfe5673de161ea3665dc913f5d1cc9a14bef4ac933b03",
        "results_sha256": "c950dba019bef755b51dff26d8bc6e08a236708309dcef2ab87a57a46c449725",
        "terminal_sha256": "d0472ca96824586d690326bcabca926a5ed32ad11cbd094ec659803084b28db0",
    },
    {
        "ordinal": 4,
        "seed": 1,
        "candidate": "PLAIN_P2",
        "directory": "seed1-plain",
        "output_name": "seed1-plain-p2",
        "root": "/root/autodl-tmp/jobs/stage11-v189-ordinal4-544e0c31dd394a3e83ea54516d247869",
        "metrics_sha256": "2b28de53165342f6b6024f5a0065d103754c350e78fe779cf12029e8e6ac013e",
        "results_sha256": "63310c287bed0ea84953c0853eb222ca06fd3fa6ab72bfaaa75ec7b61343e2e7",
        "terminal_sha256": "ece2745d74e75632bb7a3bbe28cea8e3d3823d3c298148b164cb40110ae2aa16",
    },
    {
        "ordinal": 5,
        "seed": 2,
        "candidate": "PLAIN_P2",
        "directory": "seed2-plain",
        "output_name": "seed2-plain-p2",
        "root": "/root/autodl-tmp/jobs/stage11-v192-ordinal5-a12030247f844ea6b5f94418151700fc",
        "metrics_sha256": "2e835e1689a4ddc1a514af0967dc986fa11e917b3b9616e3bfa50d0f8f54c242",
        "results_sha256": "98f396ee2d445c442bfac0fb5e03d46497be1bd43dd46dcc9e6c4adb7c240150",
        "terminal_sha256": "66ac987c60355c0a907d3c2be17f4193bad040453aeceb975525a937b1222064",
    },
    {
        "ordinal": 6,
        "seed": 2,
        "candidate": "DCLI",
        "directory": "seed2-dcli",
        "output_name": "seed2-dcli",
        "root": "/root/autodl-tmp/jobs/stage11-v193-ordinal6-1198e3345fe8457da04b55cd5a776d71",
        "metrics_sha256": "363125196a8261968e974e96c1f4cb0769f231dc0dda9478a94e5eab62489714",
        "results_sha256": "081a8dd389f5866bd83e70fb220d3ff394145a30d83e6f360696db4eff801fb3",
        "terminal_sha256": "1d4f531a332ffc3fd7ecd6ad404f3d9b1cbf4ee036cfb3d779d65b152e5cd06d",
    },
    {
        "ordinal": 7,
        "seed": 3,
        "candidate": "DCLI",
        "directory": "seed3-dcli",
        "output_name": "seed3-dcli",
        "root": "/root/autodl-tmp/jobs/stage11-v196-ordinal7-651d4a7909814012b954a2f5f59e2489",
        "metrics_sha256": "4a3d069ade5e6465791478f2d1c08937665c35c099b049e89a0b3ce1e5a68b16",
        "results_sha256": "cfcfbece59c86a894c9684480e4b6925f94fafccb2adbc9a741895c663f09335",
        "terminal_sha256": "4bc504c4cbe531c3239b227668f1020e8bb20af6f408ad83617840e4dc375b31",
    },
    {
        "ordinal": 8,
        "seed": 3,
        "candidate": "PLAIN_P2",
        "directory": "seed3-plain",
        "output_name": "seed3-plain-p2",
        "root": "/root/autodl-tmp/jobs/stage11-v197-ordinal8-6832c87b3e364e86aa3be25c21e4f9c1",
        "metrics_sha256": "8520d596312c3caccddf41d5ba02e3f5887d56ad29206cfe82b4bf7ecbfe754d",
        "results_sha256": "81f7aee785b6a9612034b4b9e7ea4a29ef93e0bf8ec2a0c08bc166eb07a7b7a3",
        "terminal_sha256": "7d59b0d075966dcd9fe4e64cfb7f702b9834f71845d9b0570d80e639d5e441ed",
    },
    {
        "ordinal": 9,
        "seed": 4,
        "candidate": "PLAIN_P2",
        "directory": "seed4-plain",
        "output_name": "seed4-plain-p2",
        "root": "/root/autodl-tmp/jobs/stage11-v198-ordinal9-129d7b0105854ebd9194b91d1399c8b2",
        "metrics_sha256": "d7ce83eefa869a3d5d57d0ba03d45aebffc1d41a1d54c6fc0ce5df32aa7b768c",
        "results_sha256": "df3eaf564e0c322e456f76a8c9dfb20d3cba2cc72aebf711935745b06149a0e7",
        "terminal_sha256": "97f25a2b894a31b5a0d8b4cb2589088c91f342d370987ac7f39a0ec5e4bb0009",
    },
    {
        "ordinal": 10,
        "seed": 4,
        "candidate": "DCLI",
        "directory": "seed4-dcli",
        "output_name": "seed4-dcli",
        "root": "/root/autodl-tmp/jobs/stage11-v199-ordinal10-5048b518e6c74eaab23eb25e14874201",
        "metrics_sha256": "df0dff89324fdf344420640618bbcd5525cf395b7e21898062f142a75b053ee2",
        "results_sha256": "892da32e708f837edef6eb911620e8c447a43d54081ee593cd95e400201647b7",
        "terminal_sha256": "a25d6542bec6b619f8689254c94ceb8938483c1a800eeed043ddb0b454789d5e",
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    return value


def build_cost_identity(
    runtime_observation: Mapping[str, object],
    profiler_observation: Mapping[str, object],
    model_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Bind cost evidence to the exact frozen v124 prediction protocol."""

    from stage11_v124_measurement_contract import cost_measurement_contract

    contract = _mapping(cost_measurement_contract(), "cost measurement contract")
    prediction_protocol = _mapping(contract.get("prediction_protocol"), "cost prediction protocol")
    return {
        "protocol": "KITTI_FIT3341_DEV371",
        "development_ids_ordered_sha256": EXPECTED_DEVELOPMENT_IDS_SHA256,
        "hardware_runtime_sha256": canonical_sha256(runtime_observation),
        "prediction_protocol_sha256": canonical_sha256(prediction_protocol),
        "profiler_identity_sha256": canonical_sha256(profiler_observation),
        "model_sha256": model_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _positive(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{role} must be positive and finite")
    return number


def validate_run_specs(specs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if len(specs) != 10:
        raise ValueError("slice/cost measurement requires exactly ten formal runs")
    validated = [dict(spec) for spec in specs]
    if [item.get("ordinal") for item in validated] != list(range(1, 11)):
        raise ValueError("slice/cost measurement requires exact ordinals 1 through 10")
    seeds = sorted({item.get("seed") for item in validated})
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"slice/cost measurement requires exact seeds {list(EXPECTED_SEEDS)}")
    for seed in EXPECTED_SEEDS:
        candidates = {item.get("candidate") for item in validated if item.get("seed") == seed}
        if candidates != {"PLAIN_P2", "DCLI"}:
            raise ValueError(f"seed {seed} must contain exact PLAIN_P2/DCLI pair")
    return validated


def format_yolo_detections(
    rows: Sequence[tuple[int, float, float, float, float, float]],
) -> bytes:
    lines = [
        f"{class_id} {center_x:.9f} {center_y:.9f} {width:.9f} {height:.9f} {confidence:.9f}\n"
        for class_id, confidence, center_x, center_y, width, height in rows
    ]
    return "".join(lines).encode("ascii")


def _run_checked(command: Sequence[str], role: str) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{role} failed: command={list(command)}, returncode={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    return completed.stdout


def _validate_sha_file(path: Path, expected_sha256: str, role: str) -> None:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"{role} SHA256 mismatch: path={path}, expected={expected_sha256}, observed={observed}")


def _load_ids(path: Path) -> tuple[str, ...]:
    ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(ids) != 371 or len(set(ids)) != 371:
        raise ValueError("development ID sequence must contain 371 unique entries")
    observed = hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()
    if observed != EXPECTED_DEVELOPMENT_IDS_SHA256:
        raise ValueError(f"development ID SHA256 mismatch: observed={observed}")
    return ids


def _one_image(image_id: str) -> Path:
    candidates = (RAW_IMAGE_ROOT / f"{image_id}.png", RAW_IMAGE_ROOT / f"{image_id}.jpg")
    existing = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(existing) != 1:
        raise ValueError(f"image identity differs for {image_id}: observed={existing}")
    return existing[0]


def _validate_source_run(spec: RunSpec) -> dict[str, object]:
    root = Path(spec["root"])
    output_root = root / "outputs" / spec["output_name"]
    metrics_path = output_root / "metrics_ap40.json"
    results_path = output_root / "results.csv"
    terminal_path = root / "terminal-receipt.json"
    provenance_path = output_root / "checkpoint_provenance.json"
    _validate_sha_file(metrics_path, spec["metrics_sha256"], "metrics")
    _validate_sha_file(results_path, spec["results_sha256"], "results")
    _validate_sha_file(terminal_path, spec["terminal_sha256"], "terminal")
    terminal = _mapping(json.loads(terminal_path.read_text(encoding="utf-8")), "terminal")
    if (
        terminal.get("state") != "PASS"
        or terminal.get("ordinal") != spec["ordinal"]
        or terminal.get("published_epochs") != 30
        or terminal.get("error") is not None
    ):
        raise ValueError(f"ordinal {spec['ordinal']} terminal is not canonical PASS")
    provenance = _mapping(json.loads(provenance_path.read_text(encoding="utf-8")), "checkpoint provenance")
    weight_path = output_root / "weights" / "last.pt"
    if provenance.get("checkpoint_path") != str(weight_path) or provenance.get("checkpoint_role") != "last.pt":
        raise ValueError(f"ordinal {spec['ordinal']} checkpoint provenance path/role differs")
    if provenance.get("identity_sha256") != terminal.get("execution_identity_sha256"):
        raise ValueError(f"ordinal {spec['ordinal']} checkpoint execution identity differs")
    expected_weight_sha = provenance.get("checkpoint_sha256")
    if not isinstance(expected_weight_sha, str) or len(expected_weight_sha) != 64:
        raise ValueError(f"ordinal {spec['ordinal']} checkpoint SHA identity differs")
    _validate_sha_file(weight_path, expected_weight_sha, "formal last.pt")
    with results_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30 or [int(float(row["epoch"])) for row in rows] != list(range(1, 31)):
        raise ValueError(f"ordinal {spec['ordinal']} results epoch sequence differs")
    return {
        "root": str(root),
        "output_root": str(output_root),
        "weight_path": str(weight_path),
        "weight_sha256": expected_weight_sha,
        "provenance_path": str(provenance_path),
        "provenance_sha256": file_sha256(provenance_path),
        "training_runtime_seconds": _positive(float(rows[-1]["time"]), "training runtime"),
        "execution_identity_sha256": terminal.get("execution_identity_sha256"),
        "task_sha256": terminal.get("task_sha256"),
    }


def _gpu_preflight() -> dict[str, object]:
    if socket.gethostname() != EXPECTED_HOST:
        raise RuntimeError(f"hostname identity differs: observed={socket.gethostname()}")
    rows = _run_checked(
        (
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        "GPU identity observation",
    ).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU row: observed={rows}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4 or fields[0] != EXPECTED_GPU_UUID or fields[1] != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"GPU identity differs: observed={fields}")
    compute_processes = _run_checked(
        (
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ),
        "GPU process observation",
    ).strip()
    if compute_processes:
        raise RuntimeError(f"GPU is not exclusive before measurement: compute_pids={compute_processes}")
    if int(fields[2]) > 16 or int(fields[3]) != 0:
        raise RuntimeError(f"GPU is not idle before measurement: memory_mib={fields[2]}, utilization={fields[3]}")
    return {
        "hostname": socket.gethostname(),
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "memory_used_mib": int(fields[2]),
        "utilization_percent": int(fields[3]),
    }


def _predict_one(model: object, image_path: Path) -> object:
    results = model.predict(
        source=str(image_path),
        stream=False,
        imgsz=640,
        batch=1,
        device=0,
        rect=True,
        augment=False,
        conf=0.001,
        iou=0.7,
        max_det=300,
        verbose=False,
        save=False,
    )
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("prediction result cardinality differs")
    return results[0]


def _prediction_values(result: object) -> list[tuple[int, float, float, float, float, float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    coordinates = boxes.xywhn.detach().cpu().tolist()
    if not (len(classes) == len(confidences) == len(coordinates)):
        raise RuntimeError("prediction tensor lengths differ")
    return [
        (int(class_id), float(confidence), float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for class_id, confidence, box in zip(classes, confidences, coordinates, strict=True)
    ]


def _profile_flops(network: object, torch_module: object) -> float:
    from torch.profiler import ProfilerActivity, profile

    device = torch_module.device("cuda:0")
    dummy = torch_module.zeros((1, 3, 640, 640), dtype=torch_module.float32, device=device)
    with torch_module.inference_mode():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            network(dummy)
        torch_module.cuda.synchronize(device)
    flops = sum(float(event.flops or 0.0) for event in profiler.key_averages())
    if flops <= 0.0:
        raise RuntimeError("FLOP profiler returned no positive FLOPs")
    return flops


def _create_prediction_tar(prediction_root: Path, ids: Sequence[str], output_path: Path) -> dict[str, object]:
    with output_path.open("xb") as raw_stream:
        with tarfile.open(fileobj=raw_stream, mode="w") as archive:
            for image_id in ids:
                source = prediction_root / f"{image_id}.txt"
                content = source.read_bytes()
                info = tarfile.TarInfo(name=f"predictions/{image_id}.txt")
                info.size = len(content)
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                import io

                archive.addfile(info, io.BytesIO(content))
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
    return {
        "path": str(output_path),
        "size": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "file_count": len(ids),
    }


def _write_create_only(path: Path, payload: object) -> dict[str, object]:
    content = canonical_bytes(payload)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _copy_create_only(source: Path, destination: Path) -> dict[str, object]:
    source_sha = file_sha256(source)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        for block in iter(lambda: input_stream.read(1024 * 1024), b""):
            output_stream.write(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    destination_sha = file_sha256(destination)
    if destination_sha != source_sha:
        raise RuntimeError(f"published file differs: source={source}, destination={destination}")
    return {"path": str(destination), "size": destination.stat().st_size, "sha256": destination_sha}


def _measure_run(
    spec: RunSpec,
    source: Mapping[str, object],
    ids: Sequence[str],
    image_paths: Sequence[Path],
    ground_truth: Mapping[str, object],
    primary_root: Path,
    mirror_root: Path,
) -> dict[str, object]:
    import torch
    from ultralytics import YOLO
    from ifdr_yolo.eval.prediction_io import load_yolo_predictions
    from stage11_v124_measurement_contract import evaluate_moderate_slices, validate_slice_measurement
    from stage11_v128_measurement_executor import build_cost_measurement

    run_primary = primary_root / spec["directory"]
    run_mirror = mirror_root / spec["directory"]
    run_primary.mkdir(mode=0o700)
    run_mirror.mkdir(mode=0o700)
    prediction_root = run_primary / "predictions"
    prediction_root.mkdir(mode=0o700)
    model = YOLO(str(source["weight_path"]), task="detect")
    network = model.model.to("cuda:0").float().eval()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    formal_cost = spec["seed"] == 0
    flops: float | None = None
    per_pass_latency_ms: list[list[float]] = []
    peak_vram_bytes: int | None = None
    image_sizes: dict[str, tuple[int, int]] = {}
    with torch.inference_mode():
        if formal_cost:
            flops = _profile_flops(network, torch)
            _predict_one(model, image_paths[0])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for image_path in image_paths[1:50]:
                _predict_one(model, image_path)
            torch.cuda.synchronize()
            for pass_index in range(5):
                pass_latency: list[float] = []
                for image_id, image_path in zip(ids, image_paths, strict=True):
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    result = _predict_one(model, image_path)
                    torch.cuda.synchronize()
                    pass_latency.append((time.perf_counter_ns() - started) / 1_000_000.0)
                    if pass_index == 0:
                        height, width = result.orig_shape
                        image_sizes[image_id] = (int(width), int(height))
                        (prediction_root / f"{image_id}.txt").write_bytes(
                            format_yolo_detections(_prediction_values(result))
                        )
                per_pass_latency_ms.append(pass_latency)
            peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        else:
            for image_id, image_path in zip(ids, image_paths, strict=True):
                result = _predict_one(model, image_path)
                height, width = result.orig_shape
                image_sizes[image_id] = (int(width), int(height))
                (prediction_root / f"{image_id}.txt").write_bytes(
                    format_yolo_detections(_prediction_values(result))
                )
            torch.cuda.synchronize()
    detections = load_yolo_predictions(prediction_root, image_sizes)
    slice_measurement = validate_slice_measurement(evaluate_moderate_slices(dict(ground_truth), detections))
    prediction_tar = run_primary / "predictions.tar"
    prediction_identity = _create_prediction_tar(prediction_root, ids, prediction_tar)
    mirror_prediction_identity = _copy_create_only(prediction_tar, run_mirror / "predictions.tar")
    slice_identity = _write_create_only(run_primary / "slice-measurement.json", slice_measurement)
    mirror_slice_identity = _copy_create_only(
        run_primary / "slice-measurement.json",
        run_mirror / "slice-measurement.json",
    )
    cost_measurement: Mapping[str, object] | None = None
    cost_identities: dict[str, object] | None = None
    if formal_cost:
        if flops is None or peak_vram_bytes is None or len(per_pass_latency_ms) != 5:
            raise RuntimeError("formal seed0 cost observations are incomplete")
        runtime_observation = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
        cost_identity = build_cost_identity(
            runtime_observation,
            {"tool": "torch.profiler", "torch": torch.__version__, "input": "fp32_1x3x640x640"},
            str(source["execution_identity_sha256"]),
            str(source["weight_sha256"]),
        )
        cost_measurement = build_cost_measurement(
            cost_identity,
            parameter_count,
            flops,
            per_pass_latency_ms,
            peak_vram_bytes,
            source["training_runtime_seconds"],
        )
        primary_cost = _write_create_only(run_primary / "cost-measurement.json", cost_measurement)
        mirror_cost = _copy_create_only(run_primary / "cost-measurement.json", run_mirror / "cost-measurement.json")
        cost_identities = {"primary": primary_cost, "mirror": mirror_cost}
    run_receipt = {
        "schema": "stage11-v200-repaired-run-slice-cost-receipt-v1",
        "state": "PASS",
        "ordinal": spec["ordinal"],
        "seed": spec["seed"],
        "candidate": spec["candidate"],
        "source": dict(source),
        "prediction_primary": prediction_identity,
        "prediction_mirror": mirror_prediction_identity,
        "slice_primary": slice_identity,
        "slice_mirror": mirror_slice_identity,
        "slice_measurement_sha256": canonical_sha256(slice_measurement),
        "cost_measurement_sha256": canonical_sha256(cost_measurement) if cost_measurement is not None else None,
        "cost_publication": cost_identities,
    }
    primary_receipt = _write_create_only(run_primary / "run-receipt.json", run_receipt)
    mirror_receipt = _copy_create_only(run_primary / "run-receipt.json", run_mirror / "run-receipt.json")
    del model
    del network
    torch.cuda.empty_cache()
    print(
        canonical_bytes(
            {
                "event": "RUN_MEASUREMENT_PASS",
                "ordinal": spec["ordinal"],
                "seed": spec["seed"],
                "candidate": spec["candidate"],
                "slice_sha256": slice_identity["sha256"],
            }
        ).decode("ascii"),
        flush=True,
    )
    return {
        "ordinal": spec["ordinal"],
        "seed": spec["seed"],
        "candidate": spec["candidate"],
        "primary_receipt": primary_receipt,
        "mirror_receipt": mirror_receipt,
    }


def run_measurement(primary_root: Path, mirror_root: Path) -> dict[str, object]:
    validated = validate_run_specs(RUN_SPECS)
    if primary_root.exists() or mirror_root.exists():
        raise FileExistsError(f"measurement output root already exists: primary={primary_root}, mirror={mirror_root}")
    primary_parent = primary_root.parent.resolve(strict=True)
    mirror_parent = mirror_root.parent.resolve(strict=True)
    if os.stat(primary_parent).st_dev == os.stat(mirror_parent).st_dev:
        raise ValueError("measurement primary and mirror parents must be on independent filesystems")
    gpu = _gpu_preflight()
    manifest_path = Path(RUN_SPECS[0]["root"]) / "runtime" / "package-manifest.json"
    _validate_sha_file(manifest_path, EXPECTED_PACKAGE_MANIFEST_SHA256, "runtime package manifest")
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    sources = [_validate_source_run(spec) for spec in RUN_SPECS]
    development_ids_path = Path(RUN_SPECS[0]["root"]) / "inputs" / "development_ids.txt"
    fit_ids_path = Path(RUN_SPECS[0]["root"]) / "inputs" / "fit_ids.txt"
    _validate_sha_file(fit_ids_path, EXPECTED_FIT_IDS_SHA256, "fit IDs")
    ids = _load_ids(development_ids_path)
    image_paths = [_one_image(image_id) for image_id in ids]
    from ifdr_yolo.eval.prediction_io import load_kitti_ground_truth

    ground_truth = load_kitti_ground_truth(RAW_LABEL_ROOT, ids)
    primary_root.mkdir(mode=0o700)
    mirror_root.mkdir(mode=0o700)
    if not stat.S_ISDIR(os.lstat(primary_root).st_mode) or not stat.S_ISDIR(os.lstat(mirror_root).st_mode):
        raise RuntimeError("measurement output root identity differs")
    results: list[dict[str, object]] = []
    try:
        for raw_spec, source in zip(validated, sources, strict=True):
            spec: RunSpec = {
                "ordinal": int(raw_spec["ordinal"]),
                "seed": int(raw_spec["seed"]),
                "candidate": str(raw_spec["candidate"]),
                "directory": str(raw_spec["directory"]),
                "output_name": str(raw_spec["output_name"]),
                "root": str(raw_spec["root"]),
                "metrics_sha256": str(raw_spec["metrics_sha256"]),
                "results_sha256": str(raw_spec["results_sha256"]),
                "terminal_sha256": str(raw_spec["terminal_sha256"]),
            }
            results.append(
                _measure_run(
                    spec,
                    source,
                    ids,
                    image_paths,
                    ground_truth,
                    primary_root,
                    mirror_root,
                )
            )
        terminal = {
            "schema": "stage11-v200-repaired-five-seed-slice-cost-terminal-v1",
            "state": "PASS",
            "host_gpu": gpu,
            "run_count": len(results),
            "formal_cost_runs": ["seed0-plain", "seed0-dcli"],
            "runs": results,
        }
        primary_terminal = _write_create_only(primary_root / "terminal-receipt.json", terminal)
        mirror_terminal = _copy_create_only(primary_root / "terminal-receipt.json", mirror_root / "terminal-receipt.json")
        return {**terminal, "primary_terminal": primary_terminal, "mirror_terminal": mirror_terminal}
    except Exception as error:
        failure = {
            "schema": "stage11-v200-repaired-five-seed-slice-cost-failure-v1",
            "state": "FAIL",
            "completed_runs": results,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_create_only(primary_root / "failure-receipt.json", failure)
        _write_create_only(mirror_root / "failure-receipt.json", failure)
        raise


def main(primary_root: Path, mirror_root: Path) -> None:
    result = run_measurement(primary_root, mirror_root)
    print(canonical_bytes(result).decode("ascii"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: stage11_v200_slice_cost_measurement.py PRIMARY_ROOT MIRROR_ROOT")
    main(Path(sys.argv[1]), Path(sys.argv[2]))


__all__: tuple[str, ...] = (
    "RUN_SPECS",
    "format_yolo_detections",
    "validate_run_specs",
)
