"""Make compact, readable review pages while retaining full artifacts on the cluster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .campaign import leaderboard
from .runtime import read_jsonl, write_json
from .media import render


def report_run(run: Path, output: Path, *, movies: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((run / "config.json").read_text())
    status = json.loads((run / "status.json").read_text())
    launches = sorted(run.glob("attempt_*/launch.json"))
    debug = bool(launches and json.loads(launches[-1].read_text()).get("debug"))
    scope_label = "Synthetic acceptance" if debug else "Validation-24"
    all_rows = [
        row
        for file_name in sorted(run.glob("attempt_*/training.jsonl"))
        for row in read_jsonl(file_name)
    ]
    # Latest attempt owns each replayed update; original attempt logs remain on disk.
    rows = sorted(
        {row["update"]: row for row in all_rows}.values(), key=lambda row: row["update"]
    )
    candidates = read_jsonl(run / "candidates.jsonl")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, name, title in zip(
        axes.flat[:3],
        ("loss", "gradient_norm_pre_clip", "learning_rate"),
        (
            "Epsilon training MSE",
            "Gradient norm before clipping",
            "Actual learning rate",
        ),
    ):
        axis.plot(
            [row["update"] for row in rows], [row[name] for row in rows], linewidth=0.9
        )
        axis.set(title=title, xlabel="Optimizer updates = examples (B1)")
        axis.grid(alpha=0.2)
        if name != "loss":
            axis.set_yscale("log")
    axis = axes.flat[3]
    for weights in config["validation"]["weights"]:
        selected = [
            row
            for row in candidates
            if row["weights"] == weights and row["score"] is not None
        ]
        axis.plot(
            [row["update"] for row in selected],
            [row["score"] for row in selected],
            "o-",
            label=weights,
        )
    axis.set(
        title=f"{scope_label} trajectory UV relative RMSE", xlabel="Optimizer updates"
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.suptitle(f"{run.name} | {status['state']} | {scope_label}", fontsize=10)
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=150)
    plt.close(figure)
    selection = (
        json.loads((run / "selection.json").read_text())
        if (run / "selection.json").exists()
        else None
    )
    lines = [
        f"Graph DiT | H1 | effective batch 1 | training seed {config['seed']} | {scope_label}",
        f"width={config['model']['width']}, blocks={config['model']['blocks']}, heads=8",
        f"LR={config['training']['learning_rate']:g}, schedule={config['training']['schedule']}, precision={config['training']['precision']}",
        f"schedule endpoint={config['training']['schedule_total_updates']:,}; allocated endpoint={status.get('stage_end_updates')}",
        f"Status: {status['state']}; completed updates={status.get('update')}",
        "Train 75-frame VGAE / fixed first-65-frame DiT; observed frame 0 -> future 1..64",
        "dt=0.08; common physical evaluator; Test sealed",
        f"Selected: {selection['weights']} at update {selection['update']}, score={selection['score']}, failed clips={selection['failed_clips']}"
        if selection
        else "No completed checkpoint selection",
        f"Error: {status.get('error', 'none recorded')}",
    ]
    figure = plt.figure(figsize=(12, 4))
    figure.text(0.04, 0.93, "\n\n".join(lines), va="top", fontsize=10, wrap=True)
    figure.savefig(output / "run_card.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    write_json(
        output / "review_manifest.json",
        {
            "run": str(run),
            "status": status,
            "selection": selection,
            "synthetic_debug": debug,
            "retention": "Checkpoints, all predictions, candidates, failed attempts and source remain with the experiment owner.",
        },
    )
    if selection:
        summary_file = run / selection["summary"]
        summary = json.loads(summary_file.read_text())
        ordered = sorted(
            [row for row in summary["trajectory_metrics"] if row["finite"]],
            key=lambda row: row["uv_relative_rmse"],
        )
        if movies and ordered:
            positions = {
                "median": int(round((len(ordered) - 1) * 0.5)),
                "p90": int(round((len(ordered) - 1) * 0.9)),
                "worst": len(ordered) - 1,
            }
            for label, ordinal in positions.items():
                index = ordered[ordinal]["trajectory_index"]
                source = (
                    summary_file.parent
                    / "predictions"
                    / f"trajectory_{index:04d}_seed0.npz"
                )
                render([source], output / label, labels=[selection["weights"]])


def report_campaign(plan: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rows = leaderboard(plan)
    write_json(output / "all_candidates.json", {"candidates": rows})
    cells = []
    for row in rows:
        chosen, status = row["selected"], row["status"]
        score = chosen.get("score")
        cells.append(
            [
                row["id"],
                status.get("state"),
                str(status.get("update", "")),
                chosen.get("weights", ""),
                "" if score is None else f"{score:.8f}",
                str(chosen.get("failed_clips", "")),
            ]
        )
    for start in range(0, len(cells), 18):
        figure, axis = plt.subplots(figsize=(16, 7))
        axis.axis("off")
        table = axis.table(
            cellText=cells[start : start + 18],
            colLabels=[
                "Candidate",
                "State",
                "Updates",
                "Weights",
                "Val24 UV relRMSE",
                "Failures",
            ],
            cellLoc="left",
            loc="center",
            colWidths=[0.48, 0.09, 0.09, 0.12, 0.13, 0.07],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        figure.suptitle(
            "All planned candidates; incomplete and failed tasks retained. Validation monitor only."
        )
        figure.savefig(
            output / f"candidates_{start // 18 + 1:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)
    payload = json.loads(plan.read_text())
    if payload.get("phase") == "confirmation":
        complete = [
            row["selected"].get("score")
            for row in rows
            if row["status"].get("state") == "complete"
            and row["selected"].get("failed_clips") == 0
        ]
        finite = [
            value for value in complete if value is not None and np.isfinite(value)
        ]
        write_json(
            output / "confirmation.json",
            {
                "planned_training_seeds": len(rows),
                "complete_finite_seeds": len(finite),
                "per_seed_monitor_scores": complete,
                "monitor_mean": float(np.mean(finite)) if finite else None,
                "monitor_sample_std": float(np.std(finite, ddof=1))
                if len(finite) > 1
                else None,
                "full_validation": "Evaluate each selected run on Validation-100 and report all seeds; sampling seeds are not training replications.",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--run", type=Path)
    sources.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--movies", action="store_true")
    args = parser.parse_args()
    if args.run:
        report_run(args.run, args.output_dir, movies=args.movies)
    else:
        report_campaign(args.plan, args.output_dir)


if __name__ == "__main__":
    main()
