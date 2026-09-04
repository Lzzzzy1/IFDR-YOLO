from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SLICE_LABELS = {
    "small_25_40": "Small (25-40 px)",
    "far_gt_40m": "Far (>40 m)",
    "occlusion_2": "Heavy occlusion",
}


def generate_bootstrap_forest_plots(
    summary_path: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            payload["schema_version"] != 1
            or payload["metric"]
            != "KITTI_PAIRED_BOOTSTRAP_CROSS_SEED_SUMMARY"
            or not payload["groups"]
        ):
            raise ValueError("unsupported or empty bootstrap summary")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid bootstrap summary") from error

    groups_by_class: dict[str, list[dict[str, object]]] = {}
    for group in payload["groups"]:
        groups_by_class.setdefault(group["class_name"], []).append(group)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for class_name in sorted(groups_by_class):
        groups = sorted(
            groups_by_class[class_name],
            key=lambda group: (
                group["reference"],
                group["candidate"],
                group["slice_name"],
            ),
        )
        seeds = sorted(
            {
                int(result["seed"])
                for group in groups
                for result in group["seed_summary"]["seed_results"]
            }
        )
        colors = {
            seed: plt.get_cmap("tab10")(index % 10)
            for index, seed in enumerate(seeds)
        }
        figure, axis = plt.subplots(
            figsize=(10.5, max(4.0, 1.05 * len(groups) + 1.8))
        )
        for group_index, group in enumerate(groups):
            results = sorted(
                group["seed_summary"]["seed_results"],
                key=lambda result: int(result["seed"]),
            )
            center = len(groups) - group_index - 1
            if len(results) == 1:
                offsets = (0.0,)
            else:
                step = 0.36 / (len(results) - 1)
                offsets = tuple(-0.18 + step * index for index in range(len(results)))
            for result, offset in zip(results, offsets, strict=True):
                seed = int(result["seed"])
                difference = float(result["difference_ap40"])
                lower = float(result["ci_lower"])
                upper = float(result["ci_upper"])
                axis.errorbar(
                    difference,
                    center + offset,
                    xerr=[[difference - lower], [upper - difference]],
                    fmt="o",
                    markersize=5.5,
                    capsize=3,
                    color=colors[seed],
                    label=f"seed {seed}" if group_index == 0 else None,
                )
        axis.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
        axis.set_yticks(range(len(groups)))
        axis.set_yticklabels(
            [
                (
                    f"{group['reference']} → {group['candidate']} | "
                    f"{SLICE_LABELS.get(group['slice_name'], group['slice_name'])}"
                )
                for group in reversed(groups)
            ]
        )
        axis.set_xlabel("Conditional AP40 difference (candidate - reference)")
        axis.set_title(f"{class_name}: paired image-bootstrap effects")
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)
        axis.legend(frameon=False, ncol=max(1, min(3, len(seeds))))
        figure.tight_layout()
        output = output_dir / (
            f"{class_name.lower().replace(' ', '_')}_paired_bootstrap_forest.png"
        )
        temporary = output.with_suffix(".png.tmp")
        figure.savefig(temporary, format="png", dpi=220, bbox_inches="tight")
        plt.close(figure)
        temporary.replace(output)
        outputs.append(output)
    return tuple(outputs)
