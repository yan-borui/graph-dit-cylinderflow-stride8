"""Prepare, launch and collect the fixed four-GPU attention campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid

from .ablation_contract import AE_BYTES, AE_REPOSITORY, AE_REVISION, TASKS, UPDATES

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/attention_ablation_4gpu"


def config_file(task: str) -> Path:
    if task not in TASKS:
        raise ValueError(f"unknown ablation task: {task}")
    return CONFIGS / f"{task}.json"


def prepare_inputs(data_dir: Path, artifacts: Path, autoencoder: Path) -> None:
    """Publish one complete cache under a shared lock; retain failed attempts."""
    import fcntl
    import torch

    from .config import load_config
    from .data import DATA_REVISION
    from .fetch import fetch_data
    from .representation import (
        checkpoint_features,
        load_artifacts,
        prepare,
        validate_representation_config,
    )
    from .runtime import write_json

    config = load_config(config_file("h1_seed0"))
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with data_dir.with_name(data_dir.name + ".download.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        data = fetch_data(data_dir)
    autoencoder.parent.mkdir(parents=True, exist_ok=True)
    with autoencoder.with_name(autoencoder.name + ".lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        downloaded = not autoencoder.exists()
        candidate = autoencoder
        if downloaded:
            attempt = Path(
                tempfile.mkdtemp(
                    prefix=autoencoder.name + ".download_", dir=autoencoder.parent
                )
            )
            candidate = attempt / "dit_autoencoder.pt"
            url = f"https://huggingface.co/{AE_REPOSITORY}/resolve/{AE_REVISION}/dit_autoencoder.pt"
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                candidate.open("xb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        if candidate.stat().st_size != AE_BYTES:
            raise ValueError("epoch1180 export length differs from the pinned release")
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        features = checkpoint_features(payload)
        validate_representation_config(
            config,
            {
                **features,
                "representation_id": payload.get("representation_id"),
                "autoencoder_format": payload.get("format"),
                "vgae_config_id": payload.get("config_id"),
                "autoencoder_seed": payload.get("seed"),
                "autoencoder_mode": payload.get("mode"),
            },
        )
        if (
            payload.get("dataset_revision") != DATA_REVISION
            or payload.get("normalization") != data.manifest["train_only_normalization"]
        ):
            raise ValueError(
                "epoch1180 export and Train normalization/data revision differ"
            )
        del payload
        if downloaded:
            os.replace(candidate, autoencoder)
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    with artifacts.with_name(artifacts.name + ".lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not artifacts.exists():
            attempt = Path(
                tempfile.mkdtemp(
                    prefix=artifacts.name + ".prepare_", dir=artifacts.parent
                )
            )
            try:
                prepare(
                    data_dir,
                    autoencoder,
                    attempt / "artifacts",
                    "cuda:0",
                    config=config,
                )
                load_artifacts(
                    attempt / "artifacts", data, config=config, autoencoder=autoencoder
                )
                os.replace(attempt / "artifacts", artifacts)
                write_json(
                    attempt / "status.json",
                    {"state": "complete", "artifacts": str(artifacts)},
                )
            except BaseException as error:
                write_json(
                    attempt / "status.json", {"state": "failed", "error": str(error)}
                )
                raise
        identity = load_artifacts(
            artifacts, data, config=config, autoencoder=autoencoder
        )
    print(
        json.dumps(
            {
                "state": "prepared",
                "artifact_id": identity["artifact_id"],
                "representation_id": identity["representation_id"],
            }
        ),
        flush=True,
    )


def launch(
    action: str, task: str, cohort: Path, data_dir: Path, artifacts: Path
) -> int:
    """Start exactly four allocated ranks, recording each attempt and exit code."""
    from .config import load_config
    from .runtime import write_json
    from .train import freeze_source

    config = load_config(config_file(task))
    run = cohort / ("preflight" if action == "preflight" else "runs") / task
    if action in ("train", "preflight") and run.exists():
        raise ValueError(f"fresh {action} requires a new directory: {run}")
    if action in ("resume", "evaluate") and not run.is_dir():
        raise ValueError(f"missing run: {run}")
    if action == "train":
        acceptance = cohort / "preflight" / f"{config['ablation']['variant']}_seed0"
        receipt = json.loads((acceptance / "acceptance.json").read_text())
        expected = load_config(config_file(f"{config['ablation']['variant']}_seed0"))
        expected["preflight_max_graph"] = True
        if receipt.get("state") != "passed" or receipt.get("config") != expected:
            raise ValueError(
                "run this variant's seed0 four-card preflight before formal training"
            )
        freeze_source(acceptance, resume=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = cohort / "launcher" / f"{task}_{action}_{stamp}_{uuid.uuid4().hex[:8]}"
    log_dir.mkdir(parents=True, exist_ok=False)
    module = (
        "graph_dit.preflight"
        if action == "preflight"
        else "graph_dit.ablation_evaluate"
        if action == "evaluate"
        else "graph_dit.train"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=4",
        "--max-restarts=0",
        "--log-dir",
        str(log_dir / "ranks"),
        "--tee",
        "3",
        "--module",
        module,
        "--data-dir",
        str(data_dir),
        "--artifacts",
        str(artifacts),
    ]
    if action == "evaluate":
        command.extend(
            ["--run", str(run), "--output-dir", str(cohort / "evaluation" / task)]
        )
    else:
        command.extend(
            [
                "--config",
                str(config_file(task)),
                "--output-dir",
                str(run),
                "--device",
                "cuda:0",
            ]
        )
        if action == "preflight":
            command.extend(["--updates", "8"])
        else:
            command.extend(["--stage-end-updates", str(UPDATES)])
            if action == "resume":
                command.append("--resume")
    write_json(
        log_dir / "launch.json",
        {
            "task": task,
            "action": action,
            "command": command,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    for name, probe in (
        ("gpu.txt", ["nvidia-smi"]),
        ("topology.txt", ["nvidia-smi", "topo", "-m"]),
        ("packages.txt", [sys.executable, "-m", "pip", "freeze"]),
    ):
        with (log_dir / name).open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                probe, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
            stream.write(f"\nprobe_exit_code={result.returncode}\n")
    with (log_dir / "launcher.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
        )
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = process.wait()
    (log_dir / "exit_code.txt").write_text(f"{code}\n", encoding="utf-8")
    write_json(
        cohort / "launcher" / f"{task}_{action}_latest.json",
        {
            "state": "complete" if code == 0 else "failed",
            "exit_code": code,
            "task": task,
            "action": action,
            "log_dir": str(log_dir),
        },
    )
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "preflight", "train", "resume", "evaluate", "report"),
    )
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--autoencoder", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--movies", action="store_true")
    args = parser.parse_args()
    cohort = args.cohort.resolve()
    if args.action == "report":
        from .ablation_report import report

        report(
            cohort, (args.output_dir or cohort / "report").resolve(), movies=args.movies
        )
        return
    if args.data_dir is None or args.artifacts is None:
        parser.error("this action requires --data-dir and --artifacts")
    data_dir, artifacts = args.data_dir.resolve(), args.artifacts.resolve()
    if args.action == "prepare":
        autoencoder = (
            args.autoencoder or artifacts.parent / "epoch1180/dit_autoencoder.pt"
        ).resolve()
        prepare_inputs(data_dir, artifacts, autoencoder)
        return
    tasks = TASKS if args.task == "all" else (args.task,)
    if args.action == "preflight" and args.task == "all":
        tasks = tuple(task for task in TASKS if task.endswith("seed0"))
    codes = []
    for task in tasks:
        if args.action == "resume":
            status_file = cohort / "runs" / task / "status.json"
            if status_file.exists():
                status = json.loads(status_file.read_text())
                if (
                    status.get("state") == "complete"
                    and status.get("update") == UPDATES
                ):
                    print(f"{task}: already complete", flush=True)
                    continue
        codes.append(launch(args.action, task, cohort, data_dir, artifacts))
    raise SystemExit(int(any(codes)))


if __name__ == "__main__":
    main()
