"""Summarize all scaling seeds, fixed endpoints, failures, and measured GPU cost."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .config import load_config


def read(file_name: Path, default: dict | None = None) -> dict | None:
    return (
        json.loads(file_name.read_text(encoding="utf-8"))
        if file_name.is_file()
        else default
    )


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def valid_candidate(row: dict) -> bool:
    return (
        row.get("failed_clips") == 0
        and row.get("trajectory_count") == 24
        and row.get("clip_count") == 72
        and finite(row.get("score"))
    )


def write_csv(file_name: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with file_name.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(cohort: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan_root = Path(__file__).resolve().parents[1] / "configs" / "scaling_32gpu"
    plan = read(plan_root / "plan.json")
    runs, curves, warnings = [], [], []
    artifact_ids = set()
    for task in plan["tasks"]:
        run = cohort / "runs" / task["id"]
        status = read(run / "status.json", {})
        row = {
            "task": task["id"],
            "model_id": task["model_id"],
            "seed": task["seed"],
            "parameters": task["parameter_count"],
            "state": status.get("state", "pending"),
            "update": status.get("update", 0),
            "error": status.get("error"),
            "training_gpu_hours": status.get("training_gpu_hours"),
            "allocated_gpu_hours": status.get("allocated_gpu_hours"),
            "endpoint_uv_validation24": None,
            "endpoint_uv_validation100": None,
            "endpoint_weights": None,
        }
        if not (run / "config.json").exists():
            runs.append(row)
            continue
        config = load_config(run / "config.json")
        expected = load_config(plan_root / task["config"])
        if config != expected:
            raise ValueError(f"{task['id']}: run differs from the fixed scaling config")
        candidates = [
            read(item) for item in sorted((run / "evaluation_records").glob("*.json"))
        ]
        groups = {}
        for candidate in candidates:
            if (
                candidate.get("config") != config
                or candidate.get("training_seed") != task["seed"]
                or candidate.get("debug") is not False
                or candidate.get("examples_seen") != candidate["update"] * 32
            ):
                raise ValueError(
                    f"{task['id']}: candidate evidence differs from the run"
                )
            # Ignore outputs beyond the recovery cursor while a run is incomplete.
            if candidate["update"] > row["update"]:
                continue
            artifact_ids.add(candidate["artifact_id"])
            groups.setdefault(candidate["update"], []).append(candidate)
        for update, group in sorted(groups.items()):
            if {item["weights"] for item in group} != set(
                config["validation"]["weights"]
            ):
                warnings.append(
                    f"{task['id']} update {update}: incomplete raw/EMA evaluation"
                )
                continue
            valid = [item for item in group if valid_candidate(item)]
            if not valid:
                warnings.append(
                    f"{task['id']} update {update}: no finite complete candidate"
                )
                continue
            best = min(
                valid,
                key=lambda item: (
                    item["score"],
                    config["validation"]["weights"].index(item["weights"]),
                ),
            )
            curves.append(
                {
                    "task": task["id"],
                    "model_id": task["model_id"],
                    "seed": task["seed"],
                    "update": update,
                    "windows": update * 32,
                    "weights": best["weights"],
                    "uv_validation24": best["score"],
                    "training_gpu_hours": best.get("training_gpu_hours"),
                }
            )
        endpoint = read(run / "selection_endpoint.json", {})
        if status.get("state") == "complete" and row["update"] == 125000:
            endpoint_group = [
                item for item in groups.get(125000, []) if valid_candidate(item)
            ]
            endpoint_weights = {item["weights"] for item in groups.get(125000, [])}
            if endpoint_group and endpoint_weights == set(
                config["validation"]["weights"]
            ):
                expected_endpoint = min(
                    endpoint_group,
                    key=lambda item: (
                        item["score"],
                        config["validation"]["weights"].index(item["weights"]),
                    ),
                )
                if any(
                    endpoint.get(key) != expected_endpoint[key]
                    for key in ("weights", "score", "checkpoint", "update")
                ) or not endpoint.get("complete_stage"):
                    raise ValueError(
                        f"{task['id']}: endpoint selection disagrees with evidence"
                    )
                row.update(
                    endpoint_uv_validation24=endpoint["score"],
                    endpoint_weights=endpoint["weights"],
                )
        validation_dir = cohort / "validation" / task["id"] / "endpoint"
        validation = read(validation_dir / "summary.json", {})
        if read(validation_dir / "status.json", {}).get("state") == "complete":
            provenance = validation.get("provenance", {})
            if (
                not finite(row["endpoint_uv_validation24"])
                or provenance.get("config") != config
                or provenance.get("weights") != endpoint.get("weights")
                or provenance.get("update") != 125000
                or provenance.get("selection") != "endpoint"
                or provenance.get("artifact_id") != endpoint.get("artifact_id")
                or provenance.get("checkpoint_id") != endpoint.get("checkpoint_id")
                or validation.get("failed_clips") != 0
                or validation.get("trajectory_count") != 100
                or validation.get("clip_count") != 100
                or validation.get("sampling_seeds") != [0]
                or provenance.get("sampling_steps") != 6
                or provenance.get("ensemble_size") != 8
                or provenance.get("aggregation") != "physical_uvp_mean"
                or not finite(validation.get("selection_uv_relative_rmse"))
            ):
                raise ValueError(
                    f"{task['id']}: full Validation evidence is mismatched or incomplete"
                )
            row["endpoint_uv_validation100"] = validation["selection_uv_relative_rmse"]
        runs.append(row)
    if len(artifact_ids) > 1:
        raise ValueError("scaling runs use different prepared artifacts")
    output.mkdir(parents=True, exist_ok=False)
    sizes = list(dict.fromkeys(task["model_id"] for task in plan["tasks"]))
    summary = []
    for model_id in sizes:
        group = [row for row in runs if row["model_id"] == model_id]
        aggregate = {"model_id": model_id, "parameters": group[0]["parameters"]}
        for scope in (24, 100):
            key = f"endpoint_uv_validation{scope}"
            values = [row[key] for row in group if finite(row[key])]
            aggregate[f"n_validation{scope}"] = len(values)
            aggregate[f"mean_validation{scope}"] = mean(values) if values else None
            aggregate[f"std_validation{scope}"] = (
                stdev(values) if len(values) > 1 else None
            )
        summary.append(aggregate)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "curves_validation24.csv", curves)
    write_csv(output / "endpoints.csv", summary)
    complete = all(row["n_validation100"] == 3 for row in summary)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "format": "graph_dit.scaling32.summary.v1",
                "complete": complete,
                "scope": "Validation only; Test sealed",
                "runs": runs,
                "models": summary,
                "warnings": warnings,
                "artifact_ids": sorted(artifact_ids),
                "uncertainty": "sample standard deviation across independent training seeds",
                "cost": "measured training GPU-hours including activation recomputation; not FLOPs",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    colors = dict(zip(sizes, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for x_key, xlabel, file_name in (
        ("update", "Optimizer updates (global batch 32)", "quality_vs_updates"),
        ("training_gpu_hours", "Measured training GPU-hours", "quality_vs_gpu_hours"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for model_id in sizes:
            for seed in (0, 1, 2):
                points = sorted(
                    (
                        row
                        for row in curves
                        if row["model_id"] == model_id
                        and row["seed"] == seed
                        and finite(row[x_key])
                    ),
                    key=lambda row: row["update"],
                )
                if points:
                    axis.plot(
                        [row[x_key] for row in points],
                        [row["uv_validation24"] for row in points],
                        color=colors[model_id],
                        alpha=0.7,
                        label=f"{model_id}, seed {seed}",
                    )
        axis.set(
            xlabel=xlabel,
            ylabel="Validation24 UV relative RMSE",
            title="Common raw/EMA selection at each evaluation point",
        )
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{file_name}.png", dpi=180)
        fig.savefig(output / f"{file_name}.pdf")
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for scope, marker in ((24, "o"), (100, "s")):
        for row in summary:
            score = row[f"mean_validation{scope}"]
            if score is None:
                continue
            x_value = row["parameters"] / 1e6
            axis.errorbar(
                x_value,
                score,
                yerr=row[f"std_validation{scope}"],
                marker=marker,
                color="C0" if scope == 24 else "C1",
                capsize=4,
            )
            axis.annotate(
                f"V{scope}, n={row[f'n_validation{scope}']}",
                (x_value, score),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set(
        xlabel="Trainable DiT parameters (millions)",
        ylabel="UV relative RMSE (mean ± training-seed SD)",
        title="125k updates / 4M windows per run"
        + ("" if complete else " — incomplete cohort"),
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "quality_vs_parameters.png", dpi=180)
    fig.savefig(output / "quality_vs_parameters.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.cohort_dir, args.output_dir)


if __name__ == "__main__":
    main()
