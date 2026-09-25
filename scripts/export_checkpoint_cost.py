"""Export existing Validation24 checkpoint scores and measured training costs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

TIME_SCOPE = (
    "Cumulative trainer elapsed time at the completed optimizer update, before "
    "that update's checkpoint save and monitor; includes earlier in-training "
    "validation and checkpoint I/O. Excludes preparation, downtime between "
    "attempts, and offline evaluation. Restored counters follow the trainer's "
    "recovery lineage; discarded work after a recovery point is excluded."
)


def read_json(file_name: Path) -> dict:
    return json.loads(file_name.read_text(encoding="utf-8"))


def write_json(file_name: Path, payload: Any) -> None:
    file_name.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def write_csv(file_name: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with file_name.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def controls(config: dict) -> dict:
    """Retain scientific settings while allowing the declared attention change."""
    value = copy.deepcopy(config)
    for key in ("seed", "ablation", "protocol"):
        value.pop(key, None)
    for key in ("attention_mode", "graph_hop_limit"):
        value["model"].pop(key, None)
    return value


def execution_signature(launch: dict) -> str:
    """Compare recorded runtime and topology without machine names or rank IDs."""
    fields = ("gpu", "torch", "cuda", "backend", "local_world_size")
    devices = [
        {key: device.get(key) for key in fields} for device in launch.get("devices", [])
    ]
    return json.dumps(
        {"devices": devices, "environment": launch.get("environment")},
        sort_keys=True,
    )


def collect_run(run: Path, label: str) -> dict:
    """Read JSON evidence without importing model code or loading checkpoints."""
    config = read_json(run / "config.json")
    status = read_json(run / "status.json")
    world = config["distributed"]["world_size"]
    weight_order = config["validation"]["weights"]
    if not isinstance(world, int) or world < 1:
        raise ValueError(f"{label}: invalid configured world size")
    logs: dict[int, list[dict]] = {}
    launches, issues = [], []
    for attempt in sorted(run.glob("attempt_*")):
        log_file = attempt / "training.jsonl"
        if not log_file.is_file():
            continue
        launch = read_json(attempt / "launch.json")
        launches.append({"source": str(attempt / "launch.json"), **launch})
        with log_file.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{log_file}:{line_number}: incomplete JSON; export a "
                        "stable copy of the log and retry"
                    ) from error
                if row.get("event") != "update":
                    continue
                row = {**row, "source": f"{log_file}:{line_number}"}
                row["launch"] = launch
                logs.setdefault(row["update"], []).append(row)
    candidates: dict[int, list[dict]] = {}
    for record_file in sorted((run / "evaluation_records").glob("*.json")):
        row = read_json(record_file)
        if row["update"] > status.get("update", -1):
            issues.append(f"{record_file.name}: beyond current run cursor")
            continue
        candidates.setdefault(row["update"], []).append(
            {**row, "source": str(record_file)}
        )
    points = []
    for update, group in sorted(candidates.items()):
        errors = []
        identities = {
            (row.get("checkpoint_id"), row.get("artifact_id")) for row in group
        }
        if len(identities) != 1 or any(None in identity for identity in identities):
            errors.append(
                "candidate checkpoint/cache identities disagree or are missing"
            )
        if len(group) != len(weight_order) or {row["weights"] for row in group} != set(
            weight_order
        ):
            errors.append("incomplete raw/EMA candidate set")
        for row in group:
            if (
                row.get("config") != config
                or row.get("training_seed") != config["seed"]
                or row.get("scope") != "validation24_monitor"
                or row.get("debug") is not False
                or row.get("examples_seen") != update * world
                or row.get("failed_clips") != 0
                or row.get("trajectory_count") != 24
                or row.get("clip_count")
                != 24 * len(config["validation"]["sampling_seeds"])
                or not finite(row.get("score"))
            ):
                errors.append(
                    f"{row['weights']}: incomplete or mismatched Validation24 evidence"
                )
        timing = logs.get(update, [])
        elapsed = train_hours = None
        timing_source = None
        hardware = None
        runtime = None
        timing_error = None
        if len(timing) != 1:
            timing_error = "missing or ambiguous exact-update training log"
        else:
            timer = timing[0]
            launch = timer["launch"]
            devices = launch.get("devices", [])
            models = {item.get("gpu") for item in devices}
            valid_launch = (
                launch.get("config") == config
                and launch.get("world_size") == world
                and launch.get("debug") is False
                and len(devices) == world
                and {item.get("rank") for item in devices} == set(range(world))
                and len(models) == 1
                and None not in models
                and all(
                    row.get("checkpoint_id") == f"{launch.get('run_id')}:{update}"
                    and row.get("artifact_id") == launch.get("artifact_id")
                    for row in group
                )
                and timer.get("world_size") == world
                and timer.get("examples_seen") == update * world
                and all(
                    previous.get("world_size") == world
                    and execution_signature(previous) == execution_signature(launch)
                    for previous in launches
                )
            )
            seconds = timer.get("elapsed_seconds")
            train_seconds = timer.get("train_update_seconds")
            if not valid_launch or not finite(seconds) or seconds <= 0:
                timing_error = "timing/launch identity or elapsed time is invalid"
            elif not finite(train_seconds) or not 0 <= train_seconds <= seconds:
                timing_error = "training-update timer is invalid"
            elif any(
                "training_gpu_hours" in row
                and (
                    not finite(row["training_gpu_hours"])
                    or not math.isclose(
                        row["training_gpu_hours"],
                        train_seconds * world / 3600,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                )
                for row in group
            ):
                timing_error = "candidate training timer differs from exact-update log"
            else:
                elapsed = seconds / 3600
                train_hours = train_seconds / 3600
                timing_source = timer["source"]
                hardware = next(iter(models))
                runtime = execution_signature(launch)
        best = (
            min(
                group,
                key=lambda row: (row["score"], weight_order.index(row["weights"])),
            )
            if not errors
            else None
        )
        points.append(
            {
                "label": label,
                "training_seed": config["seed"],
                "update": update,
                "examples_seen": update * world,
                "world_size": world,
                "hardware": hardware,
                "recorded_runtime": runtime,
                "elapsed_hours": elapsed,
                "elapsed_gpu_hours": elapsed * world if elapsed is not None else None,
                "train_update_hours": train_hours,
                "train_update_gpu_hours": train_hours * world
                if train_hours is not None
                else None,
                "selected_weights": best["weights"] if best else None,
                "validation24_uv_relative_rmse": best["score"] if best else None,
                "checkpoint_id": best["checkpoint_id"] if best else None,
                "artifact_id": best["artifact_id"] if best else None,
                "timing_source": timing_source,
                "timing_issue": timing_error,
                "quality_issues": "; ".join(errors),
                "run_state": status.get("state"),
                **{
                    f"score_{row['weights']}": row["score"]
                    if finite(row.get("score"))
                    else None
                    for row in group
                },
                "candidate_sources": "; ".join(row["source"] for row in group),
            }
        )
        issues.extend(f"update {update}: {error}" for error in errors)
        if timing_error:
            issues.append(f"update {update}: {timing_error}")
    if not points:
        issues.append("no retained Validation24 checkpoints at the current cursor")
    return {
        "label": label,
        "run": str(run),
        "config": config,
        "status": status,
        "launches": launches,
        "points": points,
        "issues": issues,
    }


def plot_curves(runs: list[dict], dataset: str, output: Path) -> list[str]:
    """Keep hardware and training seeds in separate matched-control panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple, list[tuple[dict, dict]]] = {}
    for run in runs:
        for point in run["points"]:
            if (
                point["elapsed_hours"] is not None
                and point["validation24_uv_relative_rmse"] is not None
            ):
                key = (point["hardware"], point["world_size"], point["training_seed"])
                groups.setdefault(key, []).append((run, point))
    issues = []
    for index, (key, entries) in enumerate(sorted(groups.items())):
        signatures = {
            json.dumps(controls(run["config"]), sort_keys=True) for run, _ in entries
        }
        artifacts = {point["artifact_id"] for _, point in entries}
        runtimes = {point["recorded_runtime"] for _, point in entries}
        if len(signatures) != 1 or len(artifacts) != 1 or len(runtimes) != 1:
            issues.append(
                f"{key}: plot withheld; controlled config/cache/runtime differs; inspect report.json"
            )
            continue
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for label in dict.fromkeys(point["label"] for _, point in entries):
            values = sorted(
                (point for _, point in entries if point["label"] == label),
                key=lambda row: row["update"],
            )
            axis.plot(
                [row["elapsed_hours"] for row in values],
                [row["validation24_uv_relative_rmse"] for row in values],
                "o-",
                label=label,
            )
        axis.set(
            xlabel="Cumulative trainer elapsed time (hours)",
            ylabel="Validation24 UV relative RMSE",
            title=f"{dataset}: {key[1]} x {key[0]}, training seed {key[2]}",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(output / f"attention_time_{index}.{extension}", dpi=200)
        plt.close(fig)
    return issues


def checkpoint_cost(run: dict, update: int, weights: str, ae_file: Path) -> dict:
    points = [row for row in run["points"] if row["update"] == update]
    if len(points) != 1:
        raise ValueError("requested checkpoint has no unique retained monitor record")
    point = points[0]
    if point["quality_issues"] or point["elapsed_hours"] is None:
        raise ValueError(
            "requested checkpoint lacks consistent quality/timing evidence"
        )
    if not finite(point.get(f"score_{weights}")):
        raise ValueError("requested raw/EMA weights lack a complete monitor score")
    ae = read_json(ae_file)
    required = (
        "artifact_id",
        "checkpoint",
        "hardware",
        "gpu_count",
        "elapsed_hours",
        "time_scope",
        "evidence",
    )
    if (
        any(key not in ae for key in required)
        or ae["artifact_id"] != point["artifact_id"]
    ):
        raise ValueError(
            "autoencoder cost record must identify the same prepared artifact"
        )
    if (
        ae.get("measurement") != "measured"
        or not finite(ae["elapsed_hours"])
        or ae["elapsed_hours"] <= 0
        or not isinstance(ae["gpu_count"], int)
        or ae["gpu_count"] < 1
        or any(
            not ae[key] for key in ("checkpoint", "hardware", "time_scope", "evidence")
        )
    ):
        raise ValueError(
            "autoencoder cost needs measured elapsed time and source evidence"
        )
    same_hardware = ae["hardware"] == point["hardware"]
    return {
        "checkpoint_id": point["checkpoint_id"],
        "update": update,
        "weights": weights,
        "artifact_id": point["artifact_id"],
        "autoencoder": {
            **ae,
            "elapsed_gpu_hours": ae["elapsed_hours"] * ae["gpu_count"],
        },
        "autoencoder_cost_source": str(ae_file),
        "denoiser": point,
        "denoiser_time_scope": TIME_SCOPE,
        "stage_elapsed_hours_sum": ae["elapsed_hours"] + point["elapsed_hours"],
        "same_hardware": same_hardware,
        "total_gpu_hours": ae["elapsed_hours"] * ae["gpu_count"]
        + point["elapsed_gpu_hours"]
        if same_hardware
        else None,
        "accounting": "Per-model cost includes the shared autoencoder once. Across seeds, count a shared autoencoder once for the campaign. Stage hours sum sequential costs; hardware-specific GPU-hours remain separate when devices differ.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    curves = commands.add_parser("curves", help="export retained intermediate monitors")
    curves.add_argument(
        "--run", nargs=2, action="append", metavar=("LABEL", "DIRECTORY"), required=True
    )
    curves.add_argument("--dataset", choices=("CylinderFlow", "Airfoil"), required=True)
    curves.add_argument("--plot", action="store_true")
    curves.add_argument("--output-dir", type=Path, required=True)
    cost = commands.add_parser(
        "cost", help="combine measured AE and exact DiT checkpoint cost"
    )
    cost.add_argument("--run", type=Path, required=True)
    cost.add_argument("--update", type=int, required=True)
    cost.add_argument("--weights", required=True)
    cost.add_argument("--autoencoder-cost", type=Path, required=True)
    cost.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "curves":
        labels = [label for label, _ in args.run]
        if len(labels) != len(set(labels)):
            parser.error("each run needs a unique label")
        runs = [
            collect_run(Path(folder).resolve(), label) for label, folder in args.run
        ]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_csv(
            args.output_dir / "checkpoints.csv",
            [point for run in runs for point in run["points"]],
        )
        plot_issues = (
            plot_curves(runs, args.dataset, args.output_dir) if args.plot else []
        )
        write_json(
            args.output_dir / "report.json",
            {
                "dataset": args.dataset,
                "time_scope": TIME_SCOPE,
                "quality_scope": "Validation24 monitor, best raw/EMA at each update; independent scores averaged within trajectory, then trajectories averaged. Every measured point is retained, including unfinished runs.",
                "runs": runs,
                "plot_issues": plot_issues,
            },
        )
    else:
        run = collect_run(args.run.resolve(), args.run.name)
        result = checkpoint_cost(
            run, args.update, args.weights, args.autoencoder_cost.resolve()
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "checkpoint_cost.json", result)


if __name__ == "__main__":
    main()
