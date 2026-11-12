"""Explicit Ellipse experiment dispatch; each invocation runs only its requested action."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("h1", "h2", "full", "distance", "deterministic", "gnn_diffusion")
STEPS = (2, 3, 4, 6, 8, 10, 20, 30, 40, 50, 75, 100)
MEMBERS = (1, 2, 4, 8, 16)


def launch(command: list[str], env: dict, receipt: Path) -> None:
    """Save the exact invocation and exit status, stopping on a failed experiment."""
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise FileExistsError(f"choose a new evaluation root: {receipt}")
    record = {"command": command, "state": "running"}
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    code = subprocess.call(command, cwd=ROOT, env=env)
    record.update(state="complete" if code == 0 else "failed", exit_code=code)
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if code:
        raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "train",
            "resume",
            "validate",
            "test",
            "scan",
            "curves",
            "benchmark",
            "benchmark-scan",
        ),
    )
    parser.add_argument("--variant", choices=VARIANTS, default="h1")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="existing run for evaluation/resume")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("DATA_DIR"))
    parser.add_argument(
        "--artifacts", type=Path, default=os.environ.get("ARTIFACTS_DIR")
    )
    parser.add_argument("--sampling-steps", type=int, default=6)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument("--sampling-labels", type=int, nargs="+", default=[0])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("raw", "ema_0.999", "ema_0.9999"))
    parser.add_argument("--stage-end-updates", type=int, default=250000)
    args = parser.parse_args()
    if args.action in ("benchmark", "benchmark-scan") and args.split != "validation":
        parser.error("benchmark uses the fixed Validation24 registry")
    env = dict(os.environ)
    run = (args.run or args.cohort / f"{args.variant}_seed{args.seed}").resolve()
    if args.action in ("prepare", "train", "resume"):
        if not 1 <= args.stage_end_updates <= 250000:
            parser.error("stage endpoint must lie in 1..250000")
        config = json.loads(
            (ROOT / "configs" / f"ellipse_{args.variant}_4gpu.json").read_text()
        )
        config["seed"] = args.seed
        config_file = (
            args.cohort.resolve() / "configs" / f"{args.variant}_seed{args.seed}.json"
        )
        config_file.parent.mkdir(parents=True, exist_ok=True)
        if config_file.exists() and json.loads(config_file.read_text()) != config:
            raise ValueError(
                "cohort configuration already exists with different settings"
            )
        config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        env.update(
            ELLIPSE_CONFIG=str(config_file),
            RESULT_ROOT=str(run),
            STAGE_END_UPDATES=str(args.stage_end_updates),
        )
        if args.data_dir:
            env["DATA_DIR"] = str(args.data_dir.resolve())
        if args.artifacts:
            env["ARTIFACTS_DIR"] = str(args.artifacts.resolve())
        command = ["bash", "scripts/ellipse_4gpu.sh", args.action]
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))
    if args.output_dir is None or args.data_dir is None or args.artifacts is None:
        parser.error("evaluation requires --output-dir, --data-dir and --artifacts")
    config = json.loads((run / "config.json").read_text())
    if config.get("experiment", "h1") != args.variant or config["seed"] != args.seed:
        raise ValueError("run variant/seed differs from the requested experiment")
    deterministic = config.get("model_type") == "deterministic"
    members = (
        args.ensemble_size
        if args.ensemble_size is not None
        else (1 if deterministic else 8)
    )
    if args.action in ("scan", "benchmark-scan") and deterministic:
        parser.error("sampling scans apply to generative dynamics")
    executable = env.get("PYTHON", sys.executable)
    common = [
        "--run",
        str(run),
        "--artifacts",
        str(args.artifacts.resolve()),
        "--data-dir",
        str(args.data_dir.resolve()),
    ]
    explicit = []
    if args.checkpoint:
        if not args.weights:
            parser.error("--checkpoint requires --weights")
        explicit = [
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--weights",
            args.weights,
        ]
    elif args.weights:
        explicit = ["--weights", args.weights]
    if args.action in ("scan", "benchmark-scan") and args.checkpoint is None:
        selected = json.loads((run / "selection.json").read_text())
        if not selected.get("complete_stage"):
            raise ValueError("finish the stage before scanning its selected weights")
        explicit = [
            "--checkpoint",
            str(run / selected["checkpoint"]),
            "--weights",
            args.weights or selected["weights"],
        ]
    jobs = []
    if args.action == "curves":
        if explicit:
            parser.error("curves fixes 20 checkpoint milestones and EMA 0.999")
        # Validate the entire inventory before dispatching expensive evaluation.
        for update in range(12500, 250001, 12500):
            checkpoint = run / "checkpoints" / f"update_{update:09d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for split in ("validation", "test"):
                jobs.append(
                    (
                        f"{split}/update_{update:09d}",
                        split,
                        6,
                        1 if deterministic else 8,
                        [
                            "--checkpoint",
                            str(checkpoint),
                            "--weights",
                            "ema_0.999",
                            "--skip-members",
                        ],
                    )
                )
    elif args.action in ("scan", "benchmark-scan"):
        settings = sorted(
            {
                *((step, 1) for step in STEPS),
                *((step, k) for step in (6, 20) for k in MEMBERS),
            }
        )
        for step, k in settings:
            jobs.append((f"{args.split}/s{step}_k{k}", args.split, step, k, explicit))
    else:
        split = args.action if args.action in ("validate", "test") else args.split
        split = "validation" if split == "validate" else split
        jobs.append(
            (
                f"{split}/s{args.sampling_steps}_k{members}",
                split,
                args.sampling_steps,
                members,
                explicit,
            )
        )
    for name, split, step, k, extra in jobs:
        destination = args.output_dir.resolve() / name
        if args.action in ("benchmark", "benchmark-scan"):
            command = [
                executable,
                "-m",
                "graph_dit.benchmark",
                *common,
                "--output-dir",
                str(destination),
                "--sampling-steps",
                str(step),
                "--ensemble-size",
                str(k),
                *extra,
            ]
        else:
            command = [
                executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=4",
                "--module",
                "graph_dit.evaluate_ellipse",
                *common,
                "--split",
                split,
                "--output-dir",
                str(destination),
                "--sampling-steps",
                str(step),
                "--ensemble-size",
                str(k),
                "--seeds",
                *map(str, args.sampling_labels),
                *extra,
            ]
        launch(command, env, destination.with_suffix(".launch.json"))


if __name__ == "__main__":
    main()
