from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


FROZEN_IDENTITY = "b5047a2d097b1bb4679f58136aaf557484059bc131cf878cc47c4334d3ef78c3"
FROZEN_DEVELOPMENT_COUNT = 371
FROZEN_PREDICTION_ARGS = {
    "device": "0",
    "imgsz": 640,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "augment": False,
    "verbose": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_hashes(
    prediction_dir: Path,
    expected_ids: tuple[str, ...],
) -> tuple[dict[str, str], list[str], list[str]]:
    expected = set(expected_ids)
    paths = {
        path.stem: path
        for path in prediction_dir.glob("*.txt")
        if path.is_file()
    }
    non_txt = [
        path.name
        for path in prediction_dir.iterdir()
        if path.is_file() and path.suffix != ".txt"
    ]
    hashes = {image_id: sha256_file(path) for image_id, path in sorted(paths.items())}
    id_errors = sorted((expected - set(paths)) | (set(paths) - expected))
    return hashes, id_errors, sorted(non_txt)


def _aggregate_hash(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for image_id, file_hash in sorted(hashes.items()):
        digest.update(f"{image_id}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def compare_prediction_dirs(
    *,
    reference_dir: Path,
    replay_dir: Path,
    expected_ids: tuple[str, ...],
) -> dict[str, object]:
    reference_hashes, reference_id_errors, reference_non_txt = _prediction_hashes(
        Path(reference_dir),
        expected_ids,
    )
    replay_hashes, replay_id_errors, replay_non_txt = _prediction_hashes(
        Path(replay_dir),
        expected_ids,
    )
    mismatches = sorted(
        image_id
        for image_id in set(reference_hashes) | set(replay_hashes)
        if reference_hashes.get(image_id) != replay_hashes.get(image_id)
    )
    exact = (
        not reference_id_errors
        and not replay_id_errors
        and not reference_non_txt
        and not replay_non_txt
        and not mismatches
        and len(reference_hashes) == len(expected_ids)
        and len(replay_hashes) == len(expected_ids)
    )
    return {
        "exact_match": exact,
        "expected_count": len(expected_ids),
        "reference_count": len(reference_hashes),
        "replay_count": len(replay_hashes),
        "reference_id_errors": reference_id_errors,
        "replay_id_errors": replay_id_errors,
        "reference_non_txt": reference_non_txt,
        "replay_non_txt": replay_non_txt,
        "mismatches": mismatches,
        "reference_aggregate_sha256": _aggregate_hash(reference_hashes),
        "replay_aggregate_sha256": _aggregate_hash(replay_hashes),
        "reference_file_sha256": reference_hashes,
        "replay_file_sha256": replay_hashes,
    }


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_ids(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(values) != FROZEN_DEVELOPMENT_COUNT or len(set(values)) != len(values):
        raise ValueError("prediction replay requires unique frozen DEV371 IDs")
    return values


def run_replay(
    *,
    source_root: Path,
    config_path: Path,
    run_root: Path,
    development_ids_path: Path,
    output_dir: Path,
    expected_identity: str = FROZEN_IDENTITY,
) -> dict[str, object]:
    source_root = source_root.resolve()
    config_path = config_path.resolve()
    run_root = run_root.resolve()
    development_ids_path = development_ids_path.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir.parent / "prediction_replay_manifest.json"
    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError("prediction replay output or manifest already exists")

    status = _load_json(run_root / "status.json")
    screen = _load_json(run_root / "screen_manifest.json")
    if (
        status.get("state") != "complete"
        or int(status.get("epoch", -1)) != 15
        or status.get("identity_sha256") != expected_identity
        or screen.get("identity_sha256") != expected_identity
        or int(screen.get("epochs", -1)) != 15
    ):
        raise ValueError("prediction replay requires frozen complete v213e identity")
    checkpoint = run_root / "weights" / "last.pt"
    checkpoint_sha = sha256_file(checkpoint)
    if status.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("prediction replay checkpoint SHA does not match terminal status")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from ifdr_yolo.experiments.config import load_ifdr_config
    from ifdr_yolo.experiments.ifdr_runtime import IFDRRuntimeAdapter

    config = load_ifdr_config(config_path, repository_root=source_root)
    if (
        str(config.training.device) != FROZEN_PREDICTION_ARGS["device"]
        or int(config.training.imgsz) != FROZEN_PREDICTION_ARGS["imgsz"]
        or not math.isclose(float(config.prediction.conf), 0.001, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(float(config.prediction.iou), 0.7, rel_tol=0.0, abs_tol=1e-15)
        or int(config.prediction.max_det) != 300
    ):
        raise ValueError("prediction replay config does not match frozen inference args")
    development_ids = _load_ids(development_ids_path)
    image_root = run_root / "view" / "images" / "val"
    image_paths = tuple(image_root / f"{image_id}.png" for image_id in development_ids)
    actual_images = {path.stem for path in image_root.glob("*.png") if path.is_file()}
    if actual_images != set(development_ids):
        raise ValueError("prediction replay image view does not exactly match DEV371")

    runtime = IFDRRuntimeAdapter(config)
    labels = runtime.predict(
        weights=checkpoint,
        image_paths=image_paths,
        output_dir=output_dir,
        args=FROZEN_PREDICTION_ARGS,
    )
    comparison = compare_prediction_dirs(
        reference_dir=run_root / "predictions" / "labels",
        replay_dir=labels,
        expected_ids=development_ids,
    )
    report = {
        "schema_version": 1,
        "role": "engineering_replay_not_result_selection",
        "exact_match": comparison["exact_match"],
        "identity_sha256": expected_identity,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "development_ids_path": str(development_ids_path),
        "development_ids_sha256": sha256_file(development_ids_path),
        "prediction_args": FROZEN_PREDICTION_ARGS,
        "reference_prediction_dir": str((run_root / "predictions" / "labels").resolve()),
        "replay_prediction_dir": str(labels.resolve()),
        "comparison": comparison,
        "replay_tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report["manifest_path"] = str(manifest_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay and hash-compare frozen v213e terminal prediction.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", default=FROZEN_IDENTITY)
    args = parser.parse_args(argv)
    report = run_replay(
        source_root=args.source_root,
        config_path=args.config,
        run_root=args.run_root,
        development_ids_path=args.development_ids,
        output_dir=args.output_dir,
        expected_identity=args.expected_identity,
    )
    print(f"exact_match={report['exact_match']}")
    print(f"prediction_replay_manifest={report['manifest_path']}")
    return 0 if report["exact_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
