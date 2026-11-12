"""Build comparable tables, training curves and shared-scale physical previews."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_json(file: Path) -> dict:
    return json.loads(file.read_text(encoding="utf-8"))


def flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif not isinstance(item, list):
            result[name] = item
    return result


def write_csv(file: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(base: Path, value: str) -> Path:
    file = Path(value)
    return file if file.is_absolute() else base / file


def preview(case: dict, base: Path, destination: Path) -> None:
    """Use the identical mesh, truth, time and color range for every method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    records = []
    for item in case["fields"]:
        with np.load(resolve(base, item["file"]), allow_pickle=False) as archive:
            records.append(
                (
                    item["method"],
                    {
                        key: archive[key]
                        for key in (
                            "prediction",
                            "target",
                            "points",
                            "cells",
                            "physical_time",
                            "trajectory_index",
                        )
                    },
                )
            )
    reference = records[0][1]
    for _, record in records:
        for key in ("target", "points", "cells", "physical_time", "trajectory_index"):
            if not np.array_equal(record[key], reference[key]):
                raise ValueError(f"shared preview inputs differ: {key}")
    fields = [("Truth", reference["target"])] + [
        (name, record["prediction"]) for name, record in records
    ]
    mesh = tri.Triangulation(
        reference["points"][:, 0], reference["points"][:, 1], reference["cells"]
    )
    frames = case.get("frames", [1, 24, 48, 64])
    if any(frame < 1 or frame > 64 for frame in frames):
        raise ValueError("preview frames must be in 1..64")
    for component in ("speed", "pressure"):
        values = [
            (
                name,
                np.linalg.norm(value[..., :2], axis=-1)
                if component == "speed"
                else value[..., 2],
            )
            for name, value in fields
        ]
        if any(not np.isfinite(value[frames]).all() for _, value in values):
            raise ValueError(
                "failed predictions remain in evidence; cannot render finite field comparison"
            )
        low = min(float(value[frames].min()) for _, value in values)
        high = max(float(value[frames].max()) for _, value in values)
        fig, axes = plt.subplots(
            len(values),
            len(frames),
            figsize=(4 * len(frames), 2.5 * len(values)),
            squeeze=False,
            constrained_layout=True,
        )
        for row, (name, value) in enumerate(values):
            for col, frame in enumerate(frames):
                artist = axes[row, col].tripcolor(
                    mesh,
                    value[frame],
                    shading="gouraud",
                    vmin=low,
                    vmax=high,
                    cmap="viridis",
                )
                axes[row, col].set_aspect("equal")
                axes[row, col].set_title(
                    f"{name} · t={reference['physical_time'][frame]:g} s"
                )
                axes[row, col].set_xlabel("x")
                axes[row, col].set_ylabel("y")
        fig.colorbar(
            artist, ax=axes.ravel().tolist(), label=component + " (physical units)"
        )
        fig.savefig(
            destination.with_name(destination.name + f"_{component}.png"), dpi=180
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    base = args.manifest.resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, missing, costs, losses = [], [], [], []
    evaluations = list(manifest.get("evaluations", []))
    for group in manifest.get("evaluation_roots", []):
        directory = resolve(base, group["root"])
        if group.get("kind") == "curves":
            for split in ("validation", "test"):
                for update in range(12500, 250001, 12500):
                    expected = (
                        directory / split / f"update_{update:09d}" / "summary.json"
                    )
                    if not expected.is_file():
                        missing.append(
                            {
                                **group,
                                "summary": str(expected),
                                "reason": "missing curve milestone",
                            }
                        )
        summaries = sorted(directory.glob("**/summary.json"))
        if not summaries:
            missing.append({**group, "reason": "no evaluation summaries"})
        evaluations.extend(
            {**group, "summary": str(file.resolve())} for file in summaries
        )
    for item in evaluations:
        file = resolve(base, item["summary"])
        if not file.is_file():
            missing.append({**item, "reason": "missing summary"})
            continue
        summary = read_json(file)
        provenance = summary.get("provenance", {})
        config = provenance.get("config", provenance.get("configuration", {}))
        row = {
            "dataset": item["dataset"],
            "method": item["method"],
            "summary_file": str(file),
            "split": provenance.get(
                "split", provenance.get("scope", item.get("split"))
            ),
            "update": provenance.get("update", provenance.get("checkpoint_update")),
            "training_seed": provenance.get("training_seed", config.get("seed")),
            "weights": provenance.get("weights"),
            "sampling_steps": provenance.get("sampling_steps"),
            "ensemble_size": provenance.get("ensemble_size"),
            "selection_uv_relative_rmse": summary.get("selection_uv_relative_rmse"),
            "failed_clips": summary.get("failed_clips"),
            "trajectory_count": summary.get("trajectory_count"),
            **{
                f"metric.{key}": value
                for key, value in flatten(summary).items()
                if not key.startswith(("provenance.", "environment.", "settings."))
            },
        }
        rows.append(row)
    for item in manifest.get("runs", []):
        run = resolve(base, item["run"])
        status_file = run / "status.json"
        if not status_file.is_file():
            missing.append({**item, "reason": "missing training status"})
            continue
        status = read_json(status_file)
        config = read_json(run / "config.json")
        world = config.get("distributed", {}).get("world_size", 1)
        seconds = status.get("elapsed_seconds")
        training_seconds = status.get("train_update_seconds")
        updates = status.get("update", 0)
        devices = []
        for launch in sorted(run.glob("attempt_*/launch.json")):
            devices.append(read_json(launch).get("devices"))
        costs.append(
            {
                "dataset": item["dataset"],
                "method": item["method"],
                "run": str(run),
                "training_seed": config["seed"],
                "updates": updates,
                "world_size": world,
                "elapsed_seconds": seconds,
                "dynamics_gpu_hours": seconds * world / 3600 if seconds else None,
                "training_updates_per_second": updates / training_seconds
                if training_seconds
                else None,
                "end_to_end_updates_per_second": updates / seconds if seconds else None,
                "validation_seconds": status.get("validation_seconds"),
                "checkpoint_io_seconds": status.get("checkpoint_io_seconds"),
                "devices": json.dumps(devices),
                "cost_scope": "dynamics; AE and preparation excluded",
            }
        )
        by_update = {}
        for log in sorted(run.glob("attempt_*/training.jsonl")):
            for line in log.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("event") == "update" and entry["update"] <= updates:
                    by_update[entry["update"]] = entry
        for entry in by_update.values():
            losses.append(
                {
                    "dataset": item["dataset"],
                    "method": item["method"],
                    "training_seed": config["seed"],
                    "update": entry["update"],
                    "loss": entry["loss"],
                    "objective": "latent MSE"
                    if config.get("model_type") == "deterministic"
                    else "noise MSE",
                }
            )
    for item in manifest.get("cost_records", []):
        file = resolve(base, item["summary"])
        if not file.is_file():
            missing.append({**item, "reason": "missing stage cost record"})
            continue
        record = read_json(file)
        seconds = record.get("elapsed_seconds")
        training_seconds = record.get(
            "train_seconds", record.get("train_update_seconds")
        )
        updates = record.get("updates", record.get("update"))
        costs.append(
            {
                "dataset": item["dataset"],
                "method": item["method"],
                "stage": item["stage"],
                "summary_file": str(file),
                "world_size": item["world_size"],
                "hardware": item["hardware"],
                "elapsed_seconds": seconds,
                "updates": updates,
                "allocated_gpu_hours": seconds * item["world_size"] / 3600
                if seconds is not None
                else None,
                "training_updates_per_second": updates / training_seconds
                if updates and training_seconds
                else None,
                "end_to_end_updates_per_second": updates / seconds
                if updates and seconds
                else None,
                "exit_code": record.get("exit_code"),
                "state": record.get("state"),
            }
        )
    write_csv(args.output_dir / "results.csv", rows)
    write_csv(args.output_dir / "training_cost.csv", costs)
    write_csv(args.output_dir / "training_loss.csv", losses)
    (args.output_dir / "missing.json").write_text(json.dumps(missing, indent=2) + "\n")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ordinal, dataset in enumerate(sorted({row["dataset"] for row in rows})):
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = {}
        for row in rows:
            if (
                row["dataset"] != dataset
                or row["update"] is None
                or row["selection_uv_relative_rmse"] is None
            ):
                continue
            key = (
                row["method"],
                row["split"],
                row["training_seed"],
                row["weights"],
                row["sampling_steps"],
                row["ensemble_size"],
            )
            groups.setdefault(key, []).append(row)
        for key, values in groups.items():
            values.sort(key=lambda row: row["update"])
            ax.plot(
                [row["update"] for row in values],
                [row["selection_uv_relative_rmse"] for row in values],
                marker=".",
                label=" / ".join(map(str, key)),
            )
        ax.set(xlabel="Optimizer updates", ylabel="UV relative RMSE", title=dataset)
        if groups:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"dataset_{ordinal}_curves.png", dpi=180)
        plt.close(fig)
    for ordinal, dataset in enumerate(sorted({row["dataset"] for row in losses})):
        objectives = sorted(
            {row["objective"] for row in losses if row["dataset"] == dataset}
        )
        fig, axes = plt.subplots(
            1, len(objectives), figsize=(6 * len(objectives), 4), squeeze=False
        )
        for axis, objective in zip(axes[0], objectives):
            groups = {}
            for row in losses:
                if row["dataset"] == dataset and row["objective"] == objective:
                    groups.setdefault((row["method"], row["training_seed"]), []).append(
                        row
                    )
            for key, values in groups.items():
                values.sort(key=lambda row: row["update"])
                axis.plot(
                    [r["update"] for r in values],
                    [r["loss"] for r in values],
                    label=str(key),
                    linewidth=0.8,
                )
            axis.set(xlabel="Optimizer updates", ylabel=objective, title=dataset)
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"dataset_{ordinal}_training_loss.png", dpi=180)
        plt.close(fig)
    for ordinal, case in enumerate(manifest.get("comparisons", [])):
        preview(case, base, args.output_dir / f"case_{ordinal}")
    lines = [
        "# Experiment report",
        "",
        "| Dataset | Method | Split | S/K | UV relative RMSE | Failed clips |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['split']} | {row['sampling_steps']}/{row['ensemble_size']} | {row['selection_uv_relative_rmse']} | {row['failed_clips']} |"
        )
    lines.extend(
        [
            "",
            f"Missing evidence records: {len(missing)}. See missing.json.",
            "Training loss logs retain each objective; sampled log means are not full-update loss averages.",
            "Training seeds, sampling members, splits and weight-selection policies remain separate in results.csv.",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
