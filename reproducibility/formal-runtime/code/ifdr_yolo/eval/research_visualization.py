from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSES = ("Car", "Pedestrian", "Cyclist")
DIFFICULTIES = ("easy", "moderate", "hard")
PALETTE = ("#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#BBBBBB")
DISPLAY_NAMES = {
    "baseline": "YOLOv8m",
    "p2": "YOLOv8m-P2",
    "ifdr_full": "Initial IFDR",
    "factor_control": "Factor control",
    "fusion_only": "Reliability fusion",
    "dcli_only": "DCLI only",
    "full_control": "Joint control",
    "protected_only": "Protected only",
    "counterfactual_only": "Counterfactual only",
    "protected_counterfactual": "Protected + CF",
    "joint_forward": "Joint-forward fix",
}
OVERVIEW_ORDER = (
    "baseline",
    "p2",
    "ifdr_full",
    "factor_control",
    "fusion_only",
    "dcli_only",
    "joint_forward",
)
MECHANISM_ORDER = (
    "full_control",
    "protected_only",
    "counterfactual_only",
    "protected_counterfactual",
    "joint_forward",
)


@dataclass(frozen=True)
class RunRecord:
    directory: Path
    experiment: str
    seed: int
    max_epoch: int
    ap40: dict[str, dict[str, float]]
    results: tuple[dict[str, float], ...]

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES[self.experiment]

    @property
    def moderate_mean(self) -> float:
        return mean(self.ap40[class_name]["moderate"] for class_name in CLASSES)


def _experiment_from_name(name: str) -> str | None:
    checks = (
        ("protected-counterfactual-joint", "joint_forward"),
        ("protected-counterfactual-e90", "protected_counterfactual"),
        ("counterfactual-only", "counterfactual_only"),
        ("protected-only", "protected_only"),
        ("full-control", "full_control"),
        ("factor-control", "factor_control"),
        ("fusion-only", "fusion_only"),
        ("dcli-only", "dcli_only"),
        ("yolov8m-p2", "p2"),
        ("yolov8m-baseline", "baseline"),
    )
    for marker, experiment in checks:
        if marker in name:
            return experiment
    if "yolov8m-ifdr" in name:
        return "ifdr_full"
    return None


def _read_results(path: Path) -> tuple[dict[str, float], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, float] = {}
            for raw_key, raw_value in source.items():
                if raw_key is None or raw_value is None:
                    continue
                key = raw_key.strip()
                try:
                    row[key] = float(raw_value.strip())
                except ValueError:
                    continue
            if row:
                rows.append(row)
    return tuple(rows)


def _max_epoch(results: Sequence[dict[str, float]]) -> int:
    if not results:
        return -1
    epochs = [row.get("epoch", -1.0) for row in results]
    return int(max(epochs))


def _read_ap40(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    classes = raw["classes"]
    return {
        class_name: {
            difficulty: float(classes[class_name][difficulty]["ap40"])
            for difficulty in DIFFICULTIES
        }
        for class_name in CLASSES
    }


def discover_canonical_runs(runs_root: Path | str) -> list[RunRecord]:
    root = Path(runs_root)
    selected: dict[tuple[str, int], RunRecord] = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "metrics_ap40.json").exists():
            continue
        experiment = _experiment_from_name(directory.name)
        seed_match = re.search(r"-s(\d+)(?:-|$)", directory.name)
        if experiment is None or seed_match is None:
            continue
        results = _read_results(directory / "results.csv")
        record = RunRecord(
            directory=directory,
            experiment=experiment,
            seed=int(seed_match.group(1)),
            max_epoch=_max_epoch(results),
            ap40=_read_ap40(directory / "metrics_ap40.json"),
            results=results,
        )
        key = (record.experiment, record.seed)
        current = selected.get(key)
        if current is None or (record.max_epoch, record.directory.name) > (
            current.max_epoch,
            current.directory.name,
        ):
            selected[key] = record
    return sorted(selected.values(), key=lambda item: (OVERVIEW_ORDER.index(item.experiment) if item.experiment in OVERVIEW_ORDER else 99, item.experiment, item.seed))


def load_gradient_diagnostics(path: Path | str) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _gradient_group_payload(row: dict, name: str) -> dict:
    groups = row.get("parameter_groups")
    if isinstance(groups, dict):
        payload = groups.get(name)
        return payload if isinstance(payload, dict) else {}
    return row if name == "semantic_anchor" else {}


def _summarize_gradient_payloads(payloads: Sequence[dict]) -> dict:
    pairs: dict[str, dict[str, object]] = {}
    norms: dict[str, list[float]] = {}
    for payload in payloads:
        for loss_name, value in payload.get("gradient_norms", {}).items():
            norms.setdefault(loss_name, []).append(float(value))
        for pair_name, values in payload.get("pairs", {}).items():
            pair = pairs.setdefault(pair_name, {"observations": 0, "cosines": [], "conflicts": 0})
            pair["observations"] = int(pair["observations"]) + 1
            cosine = values.get("cosine")
            if cosine is not None:
                pair["cosines"].append(float(cosine))
                pair["conflicts"] = int(pair["conflicts"]) + int(bool(values.get("conflict")))
    compact_pairs: dict[str, dict[str, float | int | None]] = {}
    for pair_name, values in pairs.items():
        cosines = list(values["cosines"])
        valid = len(cosines)
        conflicts = int(values["conflicts"])
        compact_pairs[pair_name] = {
            "observations": int(values["observations"]),
            "valid": valid,
            "conflicts": conflicts,
            "conflict_rate": conflicts / valid if valid else None,
            "mean_cosine": mean(cosines) if cosines else None,
            "min_cosine": min(cosines) if cosines else None,
            "max_cosine": max(cosines) if cosines else None,
        }
    compact_norms = {
        loss_name: {
            "observations": len(values),
            "nonzero": sum(value > 0.0 for value in values),
            "mean": mean(values),
            "max": max(values),
        }
        for loss_name, values in norms.items()
    }
    return {
        "records": len(payloads),
        "gradient_norms": compact_norms,
        "pairs": compact_pairs,
    }


def summarize_gradient_diagnostics(rows: Iterable[dict]) -> dict:
    materialized = list(rows)
    group_names = {"semantic_anchor"}
    for row in materialized:
        groups = row.get("parameter_groups")
        if isinstance(groups, dict):
            group_names.update(groups)
    group_summaries = {
        name: _summarize_gradient_payloads(
            [
                payload
                for row in materialized
                if (payload := _gradient_group_payload(row, name))
            ]
        )
        for name in sorted(group_names)
    }
    anchor = group_summaries["semantic_anchor"]
    anchor_detection = anchor["gradient_norms"].get("detection", {})
    anchor_blocked = bool(materialized) and anchor_detection.get("nonzero", 0) == 0
    fusion_detection = group_summaries.get("fusion_adapters", {}).get(
        "gradient_norms", {}
    ).get("detection", {})
    localization_detection = group_summaries.get(
        "localization_adapter", {}
    ).get("gradient_norms", {}).get("detection", {})
    protection_verified = (
        anchor_blocked
        and fusion_detection.get("nonzero", 0) > 0
        and localization_detection.get("nonzero", 0) > 0
    )
    return {
        "records": len(materialized),
        "detection_gradient_missing": anchor_blocked,
        "semantic_anchor_detection_blocked": anchor_blocked,
        "protection_path_verified": protection_verified,
        "pairs": anchor["pairs"],
        "parameter_groups": group_summaries,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> list[Path]:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _s17(records: Sequence[RunRecord], order: Sequence[str]) -> list[RunRecord]:
    by_experiment = {(record.experiment, record.seed): record for record in records}
    return [by_experiment[(experiment, 17)] for experiment in order if (experiment, 17) in by_experiment]


def _overview(records: Sequence[RunRecord], output: Path) -> list[Path]:
    chosen = _s17(records, OVERVIEW_ORDER)
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    values = [record.moderate_mean for record in chosen]
    bars = axis.bar(np.arange(len(chosen)), values, color=PALETTE[: len(chosen)], edgecolor="white")
    axis.set_xticks(np.arange(len(chosen)), [record.display_name for record in chosen], rotation=24, ha="right")
    axis.set_ylabel("Mean Moderate AP40 (%)")
    axis.set_title("KITTI Moderate AP40: Main Development Path (seed 17)")
    axis.set_ylim(max(0.0, min(values, default=0.0) - 8.0), min(100.0, max(values, default=100.0) + 5.0))
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    return _save_figure(figure, output, "01_method_ap40_overview")


def _classwise(records: Sequence[RunRecord], output: Path) -> list[Path]:
    chosen = _s17(records, OVERVIEW_ORDER)
    matrix = np.array([[record.ap40[class_name]["moderate"] for class_name in CLASSES] for record in chosen])
    figure, axis = plt.subplots(figsize=(7.4, max(4.2, len(chosen) * 0.58)))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=max(0.0, float(matrix.min()) - 5.0), vmax=min(100.0, float(matrix.max()) + 2.0), aspect="auto")
    axis.set_xticks(np.arange(len(CLASSES)), CLASSES)
    axis.set_yticks(np.arange(len(chosen)), [record.display_name for record in chosen])
    axis.set_title("Class-wise KITTI Moderate AP40 (seed 17)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color="black", fontsize=9)
    figure.colorbar(image, ax=axis, label="AP40 (%)", fraction=0.035, pad=0.03)
    return _save_figure(figure, output, "02_classwise_moderate_ap40")


def _confidence_interval(values: Sequence[float]) -> tuple[float, float]:
    center = mean(values)
    if len(values) < 2:
        return center, 0.0
    multiplier = 4.303 if len(values) == 3 else 1.96
    return center, multiplier * stdev(values) / math.sqrt(len(values))


def _multiseed(records: Sequence[RunRecord], output: Path) -> list[Path]:
    experiments = ("baseline", "p2", "fusion_only")
    grouped = {
        experiment: [record.moderate_mean for record in records if record.experiment == experiment]
        for experiment in experiments
    }
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    positions = np.arange(len(experiments))
    centers_errors = [_confidence_interval(grouped[experiment]) for experiment in experiments]
    centers = [item[0] for item in centers_errors]
    errors = [item[1] for item in centers_errors]
    axis.errorbar(positions, centers, yerr=errors, fmt="o", capsize=7, color="#222222", linewidth=1.7, label="mean ± 95% t-CI")
    for index, experiment in enumerate(experiments):
        values = grouped[experiment]
        offsets = np.linspace(-0.08, 0.08, len(values)) if values else []
        axis.scatter(index + offsets, values, color=PALETTE[index], s=52, zorder=3, label="individual seeds" if index == 0 else None)
    axis.set_xticks(positions, [DISPLAY_NAMES[experiment] for experiment in experiments])
    axis.set_ylabel("Mean Moderate AP40 (%)")
    axis.set_title("Three-seed Stability (17, 29, 41)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="best")
    return _save_figure(figure, output, "03_multiseed_stability")


def _mechanism(records: Sequence[RunRecord], output: Path) -> list[Path]:
    chosen = _s17(records, MECHANISM_ORDER)
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    if not chosen:
        axis.text(0.5, 0.5, "No formal mechanism runs available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        x = np.arange(len(chosen))
        width = 0.23
        for index, class_name in enumerate(CLASSES):
            values = [record.ap40[class_name]["moderate"] for record in chosen]
            axis.bar(x + (index - 1) * width, values, width, label=class_name, color=PALETTE[index])
        axis.set_xticks(x, [record.display_name for record in chosen], rotation=20, ha="right")
        axis.set_ylabel("Moderate AP40 (%)")
        axis.set_title("Mechanism Ablation and Joint-forward Correction")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=3)
    return _save_figure(figure, output, "04_mechanism_ablation")


def _column(rows: Sequence[dict[str, float]], token: str) -> tuple[list[float], list[float]]:
    if not rows:
        return [], []
    key = next((name for name in rows[0] if token.lower() in name.lower()), None)
    if key is None:
        return [], []
    return [row.get("epoch", float(index)) for index, row in enumerate(rows)], [row.get(key, float("nan")) for row in rows]


def _training(records: Sequence[RunRecord], output: Path) -> list[Path]:
    chosen = _s17(records, ("baseline", "p2", "fusion_only", "joint_forward"))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for index, record in enumerate(chosen):
        epochs, box_loss = _column(record.results, "train/box_loss")
        map_epochs, map_values = _column(record.results, "mAP50-95")
        if epochs:
            axes[0].plot(epochs, box_loss, color=PALETTE[index], label=record.display_name, linewidth=1.5)
        if map_epochs:
            axes[1].plot(map_epochs, map_values, color=PALETTE[index], label=record.display_name, linewidth=1.5)
    axes[0].set(title="Training Box Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation mAP50-95", xlabel="Epoch", ylabel="mAP50-95")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Convergence of Key Configurations", y=1.02)
    return _save_figure(figure, output, "05_training_curves")


def _gradient(records: Sequence[RunRecord], output: Path, rows: Sequence[dict], summary: dict) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    cosines = [
        float(_gradient_group_payload(row, "semantic_anchor")["pairs"]["counterfactual::factor"]["cosine"])
        for row in rows
        if _gradient_group_payload(row, "semantic_anchor").get("pairs", {}).get("counterfactual::factor", {}).get("cosine") is not None
    ]
    steps = [
        float(row.get("step", index))
        for index, row in enumerate(rows)
        if _gradient_group_payload(row, "semantic_anchor").get("pairs", {}).get("counterfactual::factor", {}).get("cosine") is not None
    ]
    if cosines:
        axes[0].hist(cosines, bins=28, color=PALETTE[4], alpha=0.85, edgecolor="white")
        axes[0].axvline(0.0, color="#222222", linestyle="--", linewidth=1.2)
        axes[1].scatter(steps, cosines, s=10, alpha=0.45, color=PALETTE[0])
        axes[1].axhline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No valid cosine records", ha="center", va="center", transform=axis.transAxes)
    pair = summary.get("pairs", {}).get("counterfactual::factor", {})
    rate = pair.get("conflict_rate")
    axes[0].set(title=f"Auxiliary Gradient Cosine\nconflict rate: {rate:.1%}" if rate is not None else "Auxiliary Gradient Cosine", xlabel="Cosine similarity", ylabel="Count")
    axes[1].set(title="Conflict Evolution", xlabel="Training step", ylabel="Cosine similarity")
    title = (
        "Anchor Auxiliary Conflict — Detection Routed to Task Adapters"
        if summary.get("protection_path_verified")
        else "Factor vs Counterfactual Gradients — Adapter Routing Unverified"
    )
    figure.suptitle(title, color="#AA3377", y=1.04)
    return _save_figure(figure, output, "06_auxiliary_gradient_conflict")


def _dashboard(records: Sequence[RunRecord], output: Path, gradient_summary: dict) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    chosen = _s17(records, OVERVIEW_ORDER)
    axes[0, 0].barh([record.display_name for record in chosen], [record.moderate_mean for record in chosen], color=PALETTE[: len(chosen)])
    axes[0, 0].set(title="Moderate AP40", xlabel="Mean AP40 (%)")
    joint = next((record for record in records if record.experiment == "joint_forward" and record.seed == 17), None)
    if joint:
        axes[0, 1].bar(CLASSES, [joint.ap40[class_name]["moderate"] for class_name in CLASSES], color=PALETTE[:3])
        axes[0, 1].set(title="Joint-forward Class AP40", ylabel="AP40 (%)")
    grouped = [[record.moderate_mean for record in records if record.experiment == experiment] for experiment in ("baseline", "p2", "fusion_only")]
    axes[1, 0].boxplot(grouped, tick_labels=["Base", "P2", "Fusion"], showmeans=True)
    axes[1, 0].set(title="Three-seed Stability", ylabel="Mean Moderate AP40 (%)")
    pair = gradient_summary.get("pairs", {}).get("counterfactual::factor", {})
    if gradient_summary.get("protection_path_verified"):
        text = (
            f"Canonical formal runs: {len(records)}\n"
            f"Anchor auxiliary conflict: {pair.get('conflicts', 0)}/{pair.get('valid', 0)}\n"
            "Detection → semantic anchor: blocked\n"
            "Detection → fusion adapter: present\n"
            "Detection → localization adapter: present\n\n"
            "Protected gradient routing verified end-to-end."
        )
    else:
        text = (
            f"Canonical formal runs: {len(records)}\n"
            f"Auxiliary conflict: {pair.get('conflicts', 0)}/{pair.get('valid', 0)}\n"
            "Task-adapter detection routing: unverified\n\n"
            "Current evidence supports auxiliary-task conflict only."
        )
    axes[1, 1].text(0.02, 0.96, text, ha="left", va="top", transform=axes[1, 1].transAxes, fontsize=11)
    axes[1, 1].set_axis_off()
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.16)
    figure.suptitle("IFDR-YOLO Research Evidence Dashboard", fontsize=15, y=1.01)
    figure.tight_layout()
    return _save_figure(figure, output, "07_research_evidence_dashboard")


def _write_summary(records: Sequence[RunRecord], output: Path) -> Path:
    path = output / "summary_metrics.csv"
    fields = ["experiment", "display_name", "seed", "max_epoch", "run_directory"]
    fields += [f"{difficulty}_{class_name.lower()}_ap40" for difficulty in DIFFICULTIES for class_name in CLASSES]
    fields += [f"{difficulty}_mean_ap40" for difficulty in DIFFICULTIES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "experiment": record.experiment,
                "display_name": record.display_name,
                "seed": record.seed,
                "max_epoch": record.max_epoch,
                "run_directory": record.directory.name,
            }
            for difficulty in DIFFICULTIES:
                values = []
                for class_name in CLASSES:
                    value = record.ap40[class_name][difficulty]
                    row[f"{difficulty}_{class_name.lower()}_ap40"] = value
                    values.append(value)
                row[f"{difficulty}_mean_ap40"] = mean(values)
            writer.writerow(row)
    return path


def generate_research_figures(runs_root: Path | str, output_root: Path | str) -> list[Path]:
    _style()
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    records = discover_canonical_runs(runs_root)
    if not records:
        raise ValueError(f"no canonical AP40 runs found under {runs_root}")
    joint = next((record for record in records if record.experiment == "joint_forward" and record.seed == 17), None)
    gradient_rows = load_gradient_diagnostics(joint.directory / "gradient_diagnostics.jsonl") if joint else []
    gradient_summary = summarize_gradient_diagnostics(gradient_rows)
    generated: list[Path] = []
    generated.extend(_overview(records, output))
    generated.extend(_classwise(records, output))
    generated.extend(_multiseed(records, output))
    generated.extend(_mechanism(records, output))
    generated.extend(_training(records, output))
    generated.extend(_gradient(records, output, gradient_rows, gradient_summary))
    generated.extend(_dashboard(records, output, gradient_summary))
    generated.append(_write_summary(records, output))
    gradient_path = output / "gradient_summary.json"
    gradient_path.write_text(json.dumps(gradient_summary, indent=2, sort_keys=True), encoding="utf-8")
    generated.append(gradient_path)
    manifest_path = output / "visualization_manifest.json"
    manifest = {
        "schema_version": 1,
        "canonical_runs": [record.directory.name for record in records],
        "generated_files": sorted(path.name for path in generated),
        "caveats": [
            "KITTI AP40 values use the project evaluator and locked validation split.",
            "Three-seed intervals use a t interval with n=3 and are descriptive, not definitive.",
            "Detection gradients were zero in the current diagnostic records; only factor-counterfactual conflict is supported.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    generated.append(manifest_path)
    return generated
