"""Materialize, dispatch, promote and freeze a finite parallel experiment plan."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

from .config import load_config, validate_config, resolve_window_config

ROOT = Path(__file__).resolve().parents[1]


def read(file_name: Path) -> dict:
    return json.loads(file_name.read_text(encoding="utf-8"))


def write(file_name: Path, payload: dict) -> None:
    file_name.parent.mkdir(parents=True, exist_ok=True)
    file_name.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def make_plan(base_file: Path, search_file: Path, output: Path) -> dict:
    base, search = load_config(base_file), read(search_file)
    if "candidates" in search:
        return make_window_plan(base, search, output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = []
    schedules = []
    for schedule in search["schedules"]:
        starts = (
            search.get("decay_start_updates", [base["training"]["decay_start_updates"]])
            if schedule == "late_decay"
            else [None]
        )
        schedules.extend((schedule, start) for start in starts)
    for lr, (schedule, start), width, depth in itertools.product(
        search["learning_rates"],
        schedules,
        search["widths"],
        search["depths"],
    ):
        config = deepcopy(base)
        config["seed"] = search["screen_seed"]
        config["model"].update(width=width, blocks=depth)
        config["training"].update(learning_rate=lr, schedule=schedule)
        if start is not None:
            config["training"]["decay_start_updates"] = start
        validate_config(config)
        timing = (
            f"_drop{start}"
            if start is not None and "decay_start_updates" in search
            else ""
        )
        name = f"h1_w{width}_d{depth}_lr{lr:g}_{schedule}{timing}_seed{config['seed']}"
        write(output / "configs" / (name + ".json"), config)
        tasks.append(
            {
                "id": name,
                "config": f"configs/{name}.json",
                "run": f"runs/{name}",
                "stage_end_updates": search["stage_end_updates"],
                "resume": False,
            }
        )
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate search candidates")
    plan = {
        "format": "graph_dit.campaign.v1",
        "phase": "screen",
        "tasks": tasks,
        "search": search,
        "requested_optimizer_updates": sum(task["stage_end_updates"] for task in tasks),
        "parallelism": "one independent GPU/process per task; effective batch one",
    }
    write(output / "plan.json", plan)
    return plan


def make_window_plan(base: dict, search: dict, output: Path) -> dict:
    """Materialize only explicitly listed candidates, with both training clocks."""
    tasks, configs = [], []
    world = search["world_size"]
    for candidate in search["candidates"]:
        config = deepcopy(base)
        config["seed"] = search["screen_seed"]
        config["distributed"] = {"world_size": world}
        config["model"].update(width=candidate["width"], blocks=candidate["depth"])
        config["training"].update(
            learning_rate=candidate["learning_rate"],
            min_learning_rate=search["min_learning_rate"],
            schedule="cosine",
            precision="fp32",
            microbatch=1,
            gradient_accumulation=1,
            ema_decay_unit="window",
        )
        config["validation"]["early_stopping"] = {
            "enabled": False,
            "min_updates": 500000,
            "patience_evaluations": 20,
        }
        config["window_schedule"] = {
            "budget_windows": search["budget_windows"],
            "total_windows": candidate["schedule_total_windows"],
            **{
                key: search[key]
                for key in (
                    "warmup_windows",
                    "checkpoint_every_windows",
                    "recovery_every_windows",
                    "log_every_windows",
                    "validation_every_windows",
                )
            },
        }
        config = resolve_window_config(config)
        validate_config(config)
        name = (
            f"h1_w{candidate['width']}_d{candidate['depth']}_lr{candidate['learning_rate']:g}"
            f"_cosine{candidate['schedule_total_windows']}_b{world}_seed{config['seed']}"
        )
        tasks.append(
            {
                "id": name,
                "config": f"configs/{name}.json",
                "run": f"runs/{name}",
                "stage_end_updates": config["training"]["budget_updates"],
                "stage_end_windows": search["budget_windows"],
                "world_size": world,
                "resume": False,
            }
        )
        configs.append(config)
    if not tasks or len({item["id"] for item in tasks}) != len(tasks):
        raise ValueError("empty or duplicate explicit search candidates")
    output.mkdir(parents=True, exist_ok=False)
    for task, config in zip(tasks, configs):
        write(output / task["config"], config)
    plan = {
        "format": "graph_dit.campaign.v2",
        "phase": "screen",
        "tasks": tasks,
        "search": search,
        "requested_optimizer_updates": sum(t["stage_end_updates"] for t in tasks),
        "requested_training_windows": sum(t["stage_end_windows"] for t in tasks),
        "parallelism": f"{world} GPUs per task; one window per rank",
        "automatic_promotion": False,
    }
    write(output / "plan.json", plan)
    return plan


def task_paths(plan_file: Path, task: dict) -> tuple[Path, Path]:
    return (
        (plan_file.parent / task["config"]).resolve(),
        (plan_file.parent / task["run"]).resolve(),
    )


def worker(
    plan_file: Path,
    index: int,
    data_dir: Path,
    artifacts: Path,
    device: str,
    resume_incomplete: bool = False,
) -> int:
    plan = read(plan_file)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("array index is outside the declared plan")
    task = plan["tasks"][index]
    config_file, run = task_paths(plan_file, task)
    config = load_config(config_file)
    status = read(run / "status.json") if (run / "status.json").exists() else {}
    if (
        status.get("state") == "complete"
        and status.get("update", 0) >= task["stage_end_updates"]
    ):
        if read(run / "config.json") != config:
            raise ValueError("completed output belongs to a different configuration")
        print(f"Complete already: {task['id']}", flush=True)
        return 0
    resume = bool(
        task["resume"] or (resume_incomplete and (run / "recovery_latest.pt").exists())
    )
    command = [
        sys.executable,
        "-m",
        "graph_dit.train",
        "--config",
        str(config_file),
        "--artifacts",
        str(artifacts.resolve()),
        "--data-dir",
        str(data_dir.resolve()),
        "--output-dir",
        str(run),
        "--stage-end-updates",
        str(task["stage_end_updates"]),
        "--device",
        device,
    ]
    if resume:
        command.append("--resume")
    world = config.get("distributed", {}).get("world_size", 1)
    if world > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world}",
            "--max-restarts=0",
            "--module",
            *command[2:],
        ]
    logs = plan_file.parent / "launcher_logs"
    logs.mkdir(exist_ok=True)
    attempt = 1
    while (logs / f"{index:03d}_attempt{attempt:03d}.log").exists():
        attempt += 1
    record = logs / f"{index:03d}_attempt{attempt:03d}"
    write(
        record.with_suffix(".json"),
        {
            "task": task,
            "command": command,
            "started_unix": time.time(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    with record.with_suffix(".log").open("x", encoding="utf-8") as log:
        rc = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
        ).returncode
    record.with_suffix(".exit").write_text(f"{rc}\n")
    if rc and run.is_dir():
        current = read(run / "status.json") if (run / "status.json").exists() else {}
        write(
            run / "status.json",
            {
                **current,
                "state": "failed",
                "launcher_exit_code": rc,
                "launcher_log": str(record.with_suffix(".log")),
            },
        )
    write(
        logs / f"{index:03d}_latest_status.json",
        {
            "state": "failed" if rc else "complete",
            "exit_code": rc,
            "log": str(record.with_suffix(".log")),
            "task_id": task["id"],
        },
    )
    print(f"{task['id']}: exit {rc}; log={record.with_suffix('.log')}", flush=True)
    return rc


def run_local(
    plan_file: Path, data_dir: Path, artifacts: Path, gpus: list[str], resume: bool
) -> int:
    if (
        not gpus
        or any(not item.strip() for item in gpus)
        or len(gpus) != len(set(gpus))
    ):
        raise ValueError("provide unique GPU identifiers allocated to this campaign")
    worlds = {task.get("world_size", 1) for task in read(plan_file)["tasks"]}
    if len(worlds) != 1:
        raise ValueError("run-local requires a uniform per-task GPU allocation")
    world = worlds.pop()
    if len(gpus) % world:
        raise ValueError("allocated GPU identifiers must form complete task groups")
    groups = [
        ",".join(gpus[start : start + world]) for start in range(0, len(gpus), world)
    ]
    pending = queue.Queue()
    for index in range(len(read(plan_file)["tasks"])):
        pending.put(index)

    def gpu_worker(gpu: str) -> list[int]:
        codes = []
        while True:
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return codes
            command = [
                sys.executable,
                "-m",
                "graph_dit.campaign",
                "worker",
                "--plan",
                str(plan_file.resolve()),
                "--index",
                str(index),
                "--data-dir",
                str(data_dir.resolve()),
                "--artifacts",
                str(artifacts.resolve()),
            ]
            if resume:
                command.append("--resume-incomplete")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
            }
            codes.append(subprocess.run(command, env=env, cwd=ROOT).returncode)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        results = list(pool.map(gpu_worker, groups))
    return int(any(code for group in results for code in group))


def leaderboard(plan_file: Path, *, require_finished: bool = False) -> list[dict]:
    rows = []
    for index, task in enumerate(read(plan_file)["tasks"]):
        _, run = task_paths(plan_file, task)
        status = (
            read(run / "status.json")
            if (run / "status.json").exists()
            else {"state": "not_started"}
        )
        launcher_file = (
            plan_file.parent / "launcher_logs" / f"{index:03d}_latest_status.json"
        )
        if status["state"] == "not_started" and launcher_file.exists():
            status = read(launcher_file)
        selected = (
            read(run / "selection.json") if (run / "selection.json").exists() else {}
        )
        finished = status.get("state") == "failed" or (
            status.get("state") == "complete"
            and status.get("update", 0) >= task["stage_end_updates"]
        )
        if require_finished and not finished:
            raise ValueError(
                f"finish the declared allocation before ranking: {task['id']} is {status.get('state')}"
            )
        row = {
            "id": task["id"],
            "run": str(run),
            "status": status,
            "selected": selected,
        }
        rows.append(row)
    return rows


def eligible(rows: list[dict]) -> list[dict]:
    dependencies = {
        row["selected"]["artifact_id"]
        for row in rows
        if row["selected"].get("artifact_id")
    }
    if len(dependencies) > 1:
        raise ValueError(
            "candidates used different representation/cache IDs; a joint ranking is invalid"
        )
    return sorted(
        [
            row
            for row in rows
            if row["status"].get("state") == "complete"
            and row["selected"].get("complete_stage")
            and row["selected"].get("failed_clips") == 0
            and row["selected"].get("score") is not None
            and math.isfinite(row["selected"]["score"])
        ],
        key=lambda row: (
            row["selected"]["score"],
            row["selected"]["update"],
            row["id"],
        ),
    )


def promote(plan_file: Path, output: Path, top_k: int, stage_end: int) -> dict:
    if top_k < 1:
        raise ValueError("top-k must be positive")
    source = read(plan_file)
    if source["phase"] == "confirmation":
        raise ValueError(
            "independent confirmation seeds are reported together and cannot be selected"
        )
    rows = leaderboard(plan_file, require_finished=True)
    winners = eligible(rows)[:top_k]
    if not winners:
        raise ValueError("no complete candidate with zero failed monitor clips")
    output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for winner in winners:
        old = next(task for task in source["tasks"] if task["id"] == winner["id"])
        config_file, run = task_paths(plan_file, old)
        config = load_config(config_file)
        if (
            not old["stage_end_updates"]
            < stage_end
            <= config["training"]["schedule_total_updates"]
        ):
            raise ValueError("promotion must extend the same fixed schedule")
        tasks.append(
            {
                **old,
                "config": os.path.relpath(config_file, output),
                "run": os.path.relpath(run, output),
                "stage_end_updates": stage_end,
                "resume": True,
            }
        )
    result = {
        "format": source["format"],
        "phase": "extended",
        "tasks": tasks,
        "source_plan": os.path.relpath(plan_file.resolve(), output),
        "selection_evidence": rows,
        "selection_rule": "complete allocation; zero failures; monitor score then update then ID",
        "requested_additional_updates": sum(
            stage_end
            - next(
                row["status"]["update"] for row in winners if row["id"] == task["id"]
            )
            for task in tasks
        ),
    }
    write(output / "plan.json", result)
    return result


def freeze(plan_file: Path, output: Path, seeds: list[int]) -> dict:
    source = read(plan_file)
    if source["phase"] != "extended":
        raise ValueError(
            "freeze the extended search; confirmation seeds are never ranked against each other"
        )
    rows = leaderboard(plan_file, require_finished=True)
    ranked = eligible(rows)
    if not ranked:
        raise ValueError("no candidate can be frozen")
    winner = ranked[0]
    task = next(task for task in source["tasks"] if task["id"] == winner["id"])
    config = load_config(task_paths(plan_file, task)[0])
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or config["seed"] in seeds
        or min(seeds) < 0
    ):
        raise ValueError(
            "confirmation seeds must be unique, nonnegative and distinct from the screening seed"
        )
    output.mkdir(parents=True, exist_ok=False)
    config["validation"]["weights"] = [winner["selected"]["weights"]]
    tasks = []
    for seed in seeds:
        independent = deepcopy(config)
        independent["seed"] = seed
        name = f"confirmation_seed{seed}"
        write(output / "configs" / f"{name}.json", independent)
        tasks.append(
            {
                "id": name,
                "config": f"configs/{name}.json",
                "run": f"runs/{name}",
                "stage_end_updates": config["training"]["schedule_total_updates"],
                "resume": False,
            }
        )
    locked = {
        "format": source["format"],
        "phase": "confirmation",
        "tasks": tasks,
        "selected_screening_candidate": winner,
        "selection_evidence": rows,
        "fixed_ema_variant": config["validation"]["weights"][0],
        "claim": "best observed within this finite Validation-selected search; independent seeds all reported",
    }
    write(output / "locked_recipe.json", config)
    write(output / "plan.json", locked)
    return locked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--base", type=Path, default=ROOT / "configs/base.json")
    plan.add_argument("--search", type=Path, default=ROOT / "configs/search.json")
    plan.add_argument("--output-dir", type=Path, required=True)
    for name in ("worker", "run-local", "leaderboard", "promote", "freeze"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", type=Path, required=True)
        if name in ("worker", "run-local"):
            sub.add_argument("--data-dir", type=Path, required=True)
            sub.add_argument("--artifacts", type=Path, required=True)
            sub.add_argument("--resume-incomplete", action="store_true")
        if name == "worker":
            sub.add_argument("--index", type=int, required=True)
            sub.add_argument("--device", default="cuda:0")
        if name == "run-local":
            sub.add_argument(
                "--gpus",
                required=True,
                help="allocated CUDA IDs/UUIDs, grouped by the plan's GPUs per task",
            )
        if name in ("promote", "freeze"):
            sub.add_argument("--output-dir", type=Path, required=True)
        if name == "promote":
            sub.add_argument("--top-k", type=int, default=6)
            sub.add_argument("--stage-end-updates", type=int, default=1000000)
        if name == "freeze":
            sub.add_argument("--seeds", default="101,102,103")
    args = parser.parse_args()
    if args.command == "plan":
        result = make_plan(args.base, args.search, args.output_dir)
        print(
            f"Wrote {len(result['tasks'])} tasks; allocated updates={result['requested_optimizer_updates']}"
        )
    elif args.command == "worker":
        raise SystemExit(
            worker(
                args.plan,
                args.index,
                args.data_dir,
                args.artifacts,
                args.device,
                args.resume_incomplete,
            )
        )
    elif args.command == "run-local":
        raise SystemExit(
            run_local(
                args.plan,
                args.data_dir,
                args.artifacts,
                args.gpus.split(","),
                args.resume_incomplete,
            )
        )
    elif args.command == "leaderboard":
        print(json.dumps(leaderboard(args.plan), indent=2))
    elif args.command == "promote":
        print(
            json.dumps(
                promote(args.plan, args.output_dir, args.top_k, args.stage_end_updates),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                freeze(
                    args.plan,
                    args.output_dir,
                    [int(seed) for seed in args.seeds.split(",")],
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
