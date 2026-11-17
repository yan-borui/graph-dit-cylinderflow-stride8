"""Fixed-checkpoint evaluation, collection and progress for the GNN width ladder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

WIDTHS = (256, 512, 1024)
PARAMETERS = {128: 2480640, 256: 9417472, 512: 36857088, 1024: 146000128}


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate the recorded run's fixed endpoint without reading selection.json."""
    from graph_dit.matched_k1 import evaluate as evaluate_k1

    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    status = json.loads((args.run / "status.json").read_text(encoding="utf-8"))
    if config["model"]["width"] != args.width or config["dataset"] != args.dataset:
        raise ValueError("run differs from the requested width or dataset")
    if status["update"] != 250000 or status["state"] != "complete":
        raise ValueError("the fixed 250000-update training endpoint is not complete")
    splits = ("validation", "test") if args.split == "both" else (args.split,)
    if "test" in splits and args.test_data_dir is None:
        raise ValueError("Test evaluation requires --test-data-dir")
    for split in splits:
        if (args.output_dir / split).exists():
            raise FileExistsError(
                f"use a fresh output directory: {args.output_dir / split}"
            )
    for split in splits:
        evaluate_k1(
            argparse.Namespace(
                model="gnn",
                dataset=args.dataset,
                split=split,
                run=args.run,
                run_id=status["run_id"],
                checkpoint=Path("checkpoints/update_000250000.pt"),
                weights="ema_0.999",
                expected_gnn_width=args.width,
                data_dir=args.data_dir if split == "validation" else args.test_data_dir,
                cache_dir=args.cache_dir,
                artifacts=None,
                test_data_dir=None,
                output_dir=args.output_dir / split,
            )
        )


def collect(args: argparse.Namespace) -> None:
    """Require all twelve ladder cells; optional references remain separately named."""
    from graph_dit.matched_k1 import read_complete_summary
    from graph_latent_diffusion.model import GraphLatentDiffusion

    rows, populations, runs, references = {}, {}, {}, set()
    for reference, filenames in ((False, args.summaries), (True, args.references)):
        for filename in filenames:
            result = read_complete_summary(filename)
            dataset, split, model = (result[k] for k in ("dataset", "split", "model"))
            if model == "gnn":
                architecture = result.get("model_config", {})
                width = architecture.get("width")
                expected = GraphLatentDiffusion.architecture(width=width)
                if any(
                    architecture.get(key) != value for key, value in expected.items()
                ):
                    raise ValueError(f"unexpected GNN architecture: {filename}")
                if result.get("parameter_count") != PARAMETERS[width]:
                    raise ValueError(
                        f"parameter count differs from the width: {filename}"
                    )
                if (reference and width != 128) or (
                    not reference and width not in WIDTHS
                ):
                    raise ValueError(f"unexpected ladder/reference width: {filename}")
                variant = f"gnn_w{width:04d}"
            elif reference and model == "gladit":
                width, variant = None, "gladit"
            else:
                raise ValueError(
                    f"expected ladder GNN or original GNN/GLaDiT reference: {filename}"
                )
            key = (dataset, split, variant)
            if key in rows:
                raise ValueError(f"duplicate ladder cell: {key}")
            population = (
                result["indices"],
                result["artifact_id"],
                result["representation_id"],
            )
            if populations.setdefault(
                (dataset, split), population
            ) != population or not all(population[1:]):
                raise ValueError("ladder populations or representations differ")
            if (
                runs.setdefault((dataset, variant), result["run_id"])
                != result["run_id"]
            ):
                raise ValueError("Validation and Test must share the same run")
            if reference:
                references.add(variant)
            rows[key] = {
                "dataset": dataset,
                "split": split,
                "model": variant,
                "width": width,
                "parameter_count": result.get("parameter_count"),
                "uv_relative_rmse_percent": 100 * result["metrics"]["uv_relative_rmse"],
                "run_id": result["run_id"],
                "summary": str(filename),
            }
    variants = {f"gnn_w{width:04d}" for width in WIDTHS} | references
    required = {
        (dataset, split, variant)
        for dataset in ("cylinderflow", "airfoil")
        for split in ("validation", "test")
        for variant in variants
    }
    if set(rows) != required:
        raise ValueError(
            f"incomplete ladder; missing {sorted(required - set(rows))}, extra {sorted(set(rows) - required)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ordered = [rows[key] for key in sorted(rows)]
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    lines = [
        "# GNN width ladder: 250K / EMA 0.999 / S6",
        "",
        "Three independent K1 scores averaged within each trajectory, then equally over 100 trajectories.",
        "",
        "| Dataset | Split | Model | Parameters | UV error (%) |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['dataset']} | {row['split']} | {row['model']} | {row['parameter_count'] or ''} | {row['uv_relative_rmse_percent']:.4f} |"
        for row in ordered
    )
    (args.output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def progress(args: argparse.Namespace) -> None:
    """Read progress only; estimates never control or terminate a job."""
    status = json.loads((args.run / "status.json").read_text(encoding="utf-8"))
    attempts = sorted(args.run.glob("attempt_*/training.jsonl"))
    result = {"run": str(args.run), "status": status}
    if attempts:
        records = []
        for line in attempts[-1].read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A live writer may not have completed its last line.
        if len(records) >= 2:
            last = records[-1]
            recent = [row for row in records if row["update"] >= last["update"] - 12500]
            first = recent[0]
            span = last["update"] - first["update"]
            if span > 0:
                seconds = (last["elapsed_seconds"] - first["elapsed_seconds"]) / span
                result.update(
                    seconds_per_update=seconds,
                    measured_updates=span,
                    provisional=span < 12500,
                    remaining_training_hours_estimate=max(0, 250000 - status["update"])
                    * seconds
                    / 3600,
                    estimate_scope="recent training cycle; excludes final Validation100/Test100 evaluation and future queueing",
                )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument(
        "--dataset", choices=("cylinderflow", "airfoil"), required=True
    )
    evaluation.add_argument("--width", type=int, choices=WIDTHS, required=True)
    evaluation.add_argument(
        "--split", choices=("validation", "test", "both"), default="both"
    )
    for name in ("run", "data-dir", "cache-dir", "output-dir"):
        evaluation.add_argument("--" + name, type=Path, required=True)
    evaluation.add_argument("--test-data-dir", type=Path)
    summary = sub.add_parser("collect")
    summary.add_argument("--summaries", type=Path, nargs="+", required=True)
    summary.add_argument("--references", type=Path, nargs="*", default=[])
    summary.add_argument("--output-dir", type=Path, required=True)
    status = sub.add_parser("status")
    status.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    {"evaluate": evaluate, "collect": collect, "status": progress}[args.command](args)


if __name__ == "__main__":
    main()
