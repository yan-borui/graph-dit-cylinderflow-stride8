"""Report all nine tasks, endpoint/best scores, paired seed effects and failures."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .ablation_contract import TASKS, VARIANTS
from .runtime import read_jsonl, write_csv, write_json

METRICS = (
    "uv_relative_rmse",
    "uv_rmse",
    "pressure_gauge_free_rmse",
    "pressure_raw_rmse",
    "vorticity_rmse",
    "divergence_rmse",
    "energy_relative_rmse",
    "enstrophy_relative_rmse",
    "temporal_velocity_spectrum_relative_l2",
    "boundary_uv_rmse_pre_writeback",
    "boundary_uv_rmse_post_writeback",
)


def read(file_name: Path, default: Any = None) -> Any:
    return (
        json.loads(file_name.read_text(encoding="utf-8"))
        if file_name.exists()
        else default
    )


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def number(value: Any) -> str:
    return f"{value:.6g}" if finite(value) else "—"


def evaluation_folder(cohort: Path, task: str, selection: str) -> Path:
    folder = cohort / "evaluation" / task / selection
    reference = read(folder / "reference.json")
    if reference is not None:
        if selection != "best" or reference.get("target") != "../endpoint":
            raise ValueError("unexpected evaluation reuse reference")
        folder = folder.parent / "endpoint"
    return folder


def report(cohort: Path, output: Path, *, movies: bool = False) -> None:
    """Write complete and incomplete task evidence without ranking surviving runs."""
    from .ablation_evaluate import locked_selections
    from .config import load_config

    output.mkdir(parents=True, exist_ok=True)
    rows, issues, summaries, curves, training_rows = [], [], {}, [], []
    artifact_ids, environments = set(), set()
    for task in TASKS:
        run = cohort / "runs" / task
        variant, seed_text = task.split("_seed")
        status = read(run / "status.json", {"state": "not_started"})
        launch = read(cohort / "launcher" / f"{task}_train_latest.json", {})
        if status["state"] == "not_started" and launch.get("state") == "failed":
            status = {**status, "state": "launch_failed"}
        configs = read(run / "config.json")
        evaluation_status = read(
            cohort / "evaluation" / task / "status.json", {"state": "not_started"}
        )
        evaluation_launch = read(
            cohort / "launcher" / f"{task}_evaluate_latest.json", {}
        )
        if evaluation_launch.get("state") == "failed":
            evaluation_status = {**evaluation_status, "state": "failed"}
        final_costs = [
            read(item)
            for item in sorted(
                (cohort / "evaluation" / task / "attempts").glob("*.json")
            )
        ]
        selected = {}
        if status.get("state") == "complete":
            try:
                config = load_config(run / "config.json")
                if (config.get("ablation", {}).get("variant"), config["seed"]) != (
                    variant,
                    int(seed_text),
                ):
                    raise ValueError(
                        "task name and saved attention/training seed disagree"
                    )
                selected = locked_selections(run, config)
                environments.add(json.dumps(status.get("environment"), sort_keys=True))
            except (ValueError, KeyError, OSError) as error:
                issues.append(f"{task}: {error}")
        records = [
            row
            for attempt in sorted(run.glob("attempt_*"))
            for row in read_jsonl(attempt / "training.jsonl")
        ]
        training_rows.extend(
            {"task": task, **row}
            for row in {row["update"]: row for row in records}.values()
        )
        candidates = read_jsonl(run / "candidates.jsonl")
        for candidate in candidates:
            if candidate.get("update", 0) <= status.get("update", 0):
                curves.append(
                    {
                        "task": task,
                        "variant": variant,
                        "training_seed": int(seed_text),
                        "update": candidate["update"],
                        "windows": candidate["update"] * 4,
                        "weights": candidate["weights"],
                        "score": candidate["score"],
                        "failed_clips": candidate["failed_clips"],
                    }
                )
        for selection in ("endpoint", "best"):
            chosen = selected.get(selection, {})
            row = {
                "task": task,
                "variant": variant,
                "training_seed": int(seed_text),
                "selection": selection,
                "state": status["state"],
                "evaluation_state": evaluation_status["state"],
                "evaluation_error": evaluation_status.get("error"),
                "final_evaluation_allocated_gpu_hours": sum(
                    item.get("allocated_gpu_hours", 0) for item in final_costs
                ),
                "updates": status.get("update", 0),
                "selected_update": chosen.get("update"),
                "weights": chosen.get("weights"),
                "validation24_uv": chosen.get("score"),
                "validation100_complete": False,
                "failed_clips": None,
                "training_gpu_hours": status.get("training_gpu_hours"),
                "validation_gpu_hours": status.get("validation_gpu_hours"),
                "training_windows_per_second": status.get(
                    "training_windows_per_second"
                ),
                "training_peak_allocated_gib": max(
                    (
                        item.get("peak_allocated_gib", 0)
                        for item in status.get("rank_metrics", [])
                    ),
                    default=None,
                ),
                **{metric: None for metric in METRICS},
            }
            folder = evaluation_folder(cohort, task, selection)
            summary = read(folder / "summary.json")
            if summary and chosen:
                provenance = summary.get("provenance", {})
                if (
                    any(
                        provenance.get(key) != chosen.get(key)
                        for key in ("checkpoint_id", "artifact_id", "weights", "update")
                    )
                    or provenance.get("config") != configs
                ):
                    issues.append(
                        f"{task}/{selection}: evaluation identity differs from locked selection"
                    )
                else:
                    artifact_ids.add(provenance["artifact_id"])
                    row["failed_clips"] = summary["failed_clips"]
                    complete = (
                        summary.get("indices") == list(range(1000, 1100))
                        and summary.get("sampling_seeds") == [0, 1, 2]
                        and summary.get("clip_count") == 300
                        and summary.get("trajectory_count") == 100
                        and summary.get("failed_clips") == 0
                        and finite(summary.get("selection_uv_relative_rmse"))
                    )
                    row["validation100_complete"] = complete
                    if complete:
                        summaries[(task, selection)] = summary
                        for metric in METRICS:
                            row[metric] = summary.get(metric, {}).get("mean")
                        row["uv_relative_rmse"] = summary["selection_uv_relative_rmse"]
                    else:
                        issues.append(
                            f"{task}/{selection}: incomplete or failed Validation100; aggregate withheld"
                        )
            performance = read(folder / "performance/summary.json", {})
            if performance:
                row.update(
                    inference_state=performance.get("state"),
                    inference_latency_seconds=performance.get(
                        "latency_seconds", {}
                    ).get("mean"),
                    inference_peak_allocated_gib=(
                        performance.get("peak_allocated_bytes", 0) / 2**30
                    ),
                    inference_failed_measurements=performance.get(
                        "failed_measurements"
                    ),
                    inference_failed_warmups=performance.get("failed_warmups"),
                )
            rows.append(row)
    if len(artifact_ids) > 1 or len(environments) > 1:
        issues.append(
            "cohort uses different caches or execution environments; cross-task aggregates withheld"
        )
    comparable = len(artifact_ids) <= 1 and len(environments) <= 1
    aggregates, paired = [], []
    for selection in ("endpoint", "best"):
        for variant in VARIANTS:
            group = [
                row
                for row in rows
                if row["variant"] == variant and row["selection"] == selection
            ]
            for metric in METRICS:
                values = [row[metric] for row in group if finite(row[metric])]
                complete = comparable and len(values) == 3
                aggregates.append(
                    {
                        "selection": selection,
                        "variant": variant,
                        "metric": metric,
                        "training_seeds_complete": len(values),
                        "mean": mean(values) if complete else None,
                        "std": stdev(values) if complete else None,
                    }
                )
        for control in ("h2", "full"):
            for seed in range(3):
                left = next(
                    row
                    for row in rows
                    if row["task"] == f"h1_seed{seed}" and row["selection"] == selection
                )
                right = next(
                    row
                    for row in rows
                    if row["task"] == f"{control}_seed{seed}"
                    and row["selection"] == selection
                )
                paired.append(
                    {
                        "selection": selection,
                        "pair": f"H1-{control.upper()}",
                        "training_seed": seed,
                        **{
                            metric: left[metric] - right[metric]
                            if comparable
                            and finite(left[metric])
                            and finite(right[metric])
                            else None
                            for metric in METRICS
                        },
                    }
                )
    paired_summary = []
    for selection in ("endpoint", "best"):
        for pair in ("H1-H2", "H1-FULL"):
            for metric in METRICS:
                values = [
                    row[metric]
                    for row in paired
                    if row["selection"] == selection
                    and row["pair"] == pair
                    and finite(row[metric])
                ]
                paired_summary.append(
                    {
                        "selection": selection,
                        "pair": pair,
                        "metric": metric,
                        "n": len(values),
                        "mean": mean(values) if len(values) == 3 else None,
                        "std": stdev(values) if len(values) == 3 else None,
                    }
                )
    for name, values in (
        ("runs", rows),
        ("aggregate", aggregates),
        ("paired_seeds", paired),
        ("paired_summary", paired_summary),
        ("monitor_curves", curves),
    ):
        write_csv(output / f"{name}.csv", values)
    complete = (
        len(summaries) == 18
        and comparable
        and not issues
        and all(row["evaluation_state"] == "complete" for row in rows)
        and all(
            row.get("inference_state") == "complete"
            for row in rows
            if row["selection"] == "endpoint"
        )
    )
    write_json(
        output / "report.json",
        {
            "complete": complete,
            "runs": rows,
            "aggregate": aggregates,
            "paired": paired,
            "issues": issues,
            "uncertainty": "sample SD across three training seeds; sampling draws are averaged within trajectory",
            "scope": "CylinderFlow stride8, 250000 windows, four GPUs; Validation only",
        },
    )
    lines = [
        "# CylinderFlow 四卡注意力消融",
        "",
        f"完成状态：{'完整' if complete else '尚未完整'}；预算为每任务250,000窗口。",
        "",
        "主表为62,500更新终点；best为补充。三个采样先在轨迹内平均，再对100条轨迹等权平均。均值±标准差来自三个训练seed。",
        "",
        "| 选择 | 配置 | 已完成seed | UV relative RMSE（均值 ± SD） |",
        "|---|---|---:|---:|",
    ]
    for row in aggregates:
        if row["metric"] == "uv_relative_rmse":
            lines.append(
                f"| {row['selection']} | {row['variant'].upper()} | {row['training_seeds_complete']}/3 | {number(row['mean'])} ± {number(row['std'])} |"
            )
    lines.extend(
        [
            "",
            "| 任务 | 训练／评价状态 | 更新数 | 终点Validation100 UV | best Validation100 UV |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        group = [row for row in rows if row["task"] == task]
        lines.append(
            f"| {task} | {group[0]['state']} / {group[0]['evaluation_state']} | {group[0]['updates']} | {number(group[0]['uv_relative_rmse'])} | {number(group[1]['uv_relative_rmse'])} |"
        )
    lines.extend(
        [
            "",
            "`paired_summary.csv`中H1−对照的误差差值为负表示H1较低。缺失或失败任务保留，三seed完整前不发布总体均值。",
            "",
            "训练/评价成本与失败数见`runs.csv`；训练曲线见`training_curves.png`。原始checkpoint、逐样例指标和预测保留在cohort目录。",
        ]
    )
    if issues:
        lines.extend(["", "需核对的记录：", "", *[f"- {issue}" for issue in issues]])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_curves(training_rows, curves, output)
    if movies and len(summaries) == 18 and comparable:
        render_cases(cohort, output, summaries)


def plot_curves(training: list[dict], curves: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for seed in range(3):
        for variant in VARIANTS:
            task = f"{variant}_seed{seed}"
            records = sorted(
                (row for row in training if row["task"] == task),
                key=lambda row: row["update"],
            )
            axes[0, seed].plot(
                [row["examples_seen"] for row in records],
                [row["loss"] for row in records],
                label=variant.upper(),
            )
            records = [
                row
                for row in curves
                if row["task"] == task
                and row["failed_clips"] == 0
                and finite(row["score"])
            ]
            steps = sorted({row["windows"] for row in records})
            axes[1, seed].plot(
                steps,
                [
                    min(row["score"] for row in records if row["windows"] == step)
                    for step in steps
                ],
                "o-",
                label=variant.upper(),
            )
        for index, ylabel in enumerate(
            ("Training epsilon MSE", "Validation24 UV relRMSE (best weight at step)")
        ):
            axes[index, seed].set(
                xlabel="Global windows", ylabel=ylabel, title=f"Training seed {seed}"
            )
            axes[index, seed].grid(alpha=0.2)
            axes[index, seed].legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def render_cases(cohort: Path, output: Path, summaries: dict) -> None:
    from .media import render

    errors = {
        task: {
            row["trajectory_index"]: row["uv_relative_rmse"]
            for row in summaries[(task, "endpoint")]["trajectory_metrics"]
        }
        for task in TASKS
    }
    ordered = sorted(
        range(1000, 1100),
        key=lambda index: (mean(errors[task][index] for task in TASKS), index),
    )
    regression = max(
        range(1000, 1100),
        key=lambda index: mean(
            errors[f"h1_seed{seed}"][index]
            - min(errors[f"h2_seed{seed}"][index], errors[f"full_seed{seed}"][index])
            for seed in range(3)
        ),
    )
    cases = {
        "median": ordered[50],
        "p90": ordered[89],
        "worst": ordered[-1],
        "largest_h1_minus_control": regression,
    }
    write_json(
        output / "visual_cases.json",
        {
            "cases": cases,
            "selection": "endpoint",
            "training_seed": 0,
            "sampling_label": 0,
            "rule": "difficulty ranked by mean error across all nine runs; largest H1-minus-better-control averaged over training seeds",
        },
    )
    for label, index in cases.items():
        destination = output / "movies" / label
        if (destination / "render.json").exists():
            continue
        if destination.exists():
            destination = destination.with_name(
                destination.name + "_retry_" + uuid.uuid4().hex[:8]
            )
        inputs = [
            evaluation_folder(cohort, f"{variant}_seed0", "endpoint")
            / "predictions"
            / f"trajectory_{index:04d}_seed0.npz"
            for variant in VARIANTS
        ]
        render(inputs, destination, labels=[variant.upper() for variant in VARIANTS])
