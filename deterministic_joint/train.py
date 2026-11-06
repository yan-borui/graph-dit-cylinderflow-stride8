"""Four-GPU trajectory-weighted training and Validation24 model selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from graph_dit.config import learning_rate
from graph_dit.distributed import Context, capture_rng, restore_rng
from graph_dit.runtime import (
    ROOT,
    acquire_run_lock,
    append_json,
    code_identity,
    load_checkpoint,
    monitor_indices,
    peak_memory,
    save_checkpoint,
    seed_everything,
    write_json,
)
from graph_dit.train import EMA, configure_runtime, cpu_state, freeze_source
from .data import EvaluationData, TrainCache
from .evaluation import validation_rows
from .model import JointGNN


CHECKPOINT_FORMAT = "deterministic_joint.training.v1"
WORLD_SIZE = 4


def recipe(dataset: str, verification_updates: int | None) -> dict:
    """Fixed optimizer-window budget; a reduced verification run has its own identity."""
    if verification_updates is not None and not 1 <= verification_updates < 12500:
        raise ValueError("verification-updates must lie between 1 and 12499")
    return {
        "dataset": dataset,
        "protocol": CHECKPOINT_FORMAT,
        "seed": 0,
        "model": {
            "width": 128,
            "depths": [1, 2, 2],
            "latent_channels": 4,
            "future_frames": 64,
            "condition_channels": 512,
        },
        "world_size": WORLD_SIZE,
        "microbatch": 1,
        "accumulation": 1,
        "precision": "fp32",
        "effective_batch": 4,
        "ema_decay_unit": "window",
        "ema_decays": [0.999, 0.9999],
        "verification_updates": verification_updates,
        "validation_every": verification_updates or 12500,
        "budget_updates": verification_updates or 250000,
        "recovery_every": min(1250, verification_updates or 1250),
        "training": {
            "learning_rate": 1e-4,
            "min_learning_rate": 1e-7,
            "schedule": "cosine",
            "warmup_updates": 1000,
            "schedule_total_updates": 250000,
            "weight_decay": 1e-6,
            "gradient_clip": 1.0,
        },
    }


def snapshot_sources(run: Path, resume: bool) -> None:
    freeze_source(run, resume)
    files = [
        *ROOT.glob("deterministic_joint/*.py"),
        *ROOT.glob("airfoil_data/*.py"),
        ROOT / "scripts/deterministic_4gpu.sh",
    ]
    for source in files:
        destination = run / "source" / source.relative_to(ROOT)
        if resume:
            if (
                not destination.is_file()
                or destination.read_bytes() != source.read_bytes()
            ):
                raise ValueError(f"resume source mismatch: {source.relative_to(ROOT)}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def select(run: Path, run_id: str, update: int, weight_names: tuple[str, ...]) -> dict:
    """Only complete, finite 24-trajectory records can select a candidate."""
    candidates = []
    for file_name in sorted((run / "validation").glob("update_*.json")):
        record = json.loads(file_name.read_text(encoding="utf-8"))
        if record["run_id"] != run_id:
            raise ValueError("validation belongs to a different run")
        if record["update"] <= update:
            candidates.extend(record["candidates"])
    eligible = [row for row in candidates if row["score"] is not None]
    best = (
        min(
            eligible,
            key=lambda row: (
                row["score"],
                row["update"],
                weight_names.index(row["weights"]),
            ),
        )
        if eligible
        else {
            "score": None,
            "update": None,
            "checkpoint": None,
            "weights": None,
        }
    )
    write_json(run / "selection.json", {**best, "run_id": run_id})
    return best


def validate(
    ctx: Context,
    model: JointGNN,
    ema: EMA,
    data: EvaluationData,
    indices: tuple[int, ...],
    run: Path,
    run_id: str,
    update: int,
    relative: str,
) -> None:
    destination = run / "validation" / f"update_{update:09d}.json"
    if ctx.broadcast(destination.exists() if ctx.primary else None):
        return
    raw = cpu_state(model.state_dict())
    rng = capture_rng(ctx.device)
    candidates, predictions = [], {}
    try:
        model.eval()
        for weights in ("raw", *ema.states):
            if weights != "raw":
                model.load_state_dict(ema.states[weights])
            parts = ctx.all_call(
                lambda: validation_rows(
                    model, data, indices[ctx.rank :: ctx.world], ctx.device
                )
            )
            rows = sorted(
                (row for part in parts for row in part), key=lambda row: row["index"]
            )
            if [row["index"] for row in rows] != sorted(indices) or len(rows) != 24:
                raise ValueError(
                    "Validation24 trajectory IDs differ from the fixed population"
                )
            valid = [
                row
                for row in rows
                if row["finite"]
                and row["uv_relative"] is not None
                and math.isfinite(row["uv_relative"])
            ]
            candidates.append(
                {
                    "score": sum(row["uv_relative"] for row in valid) / 24
                    if len(valid) == 24
                    else None,
                    "failed_clips": 24 - len(valid),
                    "weights": weights,
                    "update": update,
                    "checkpoint": relative,
                }
            )
            predictions[weights] = rows
    finally:
        model.load_state_dict(raw)
        model.train()
        restore_rng(rng, ctx.device)
    ctx.primary_call(
        lambda: write_json(
            destination,
            {
                "run_id": run_id,
                "update": update,
                "indices": list(indices),
                "candidates": candidates,
                "trajectories": predictions,
            },
        )
    )


def train(args: argparse.Namespace) -> None:
    config = recipe(args.dataset, args.verification_updates)
    run = args.result_root
    ctx = Context(WORLD_SIZE, "cuda")
    lock = None
    try:
        configure_runtime(ctx.device, "fp32", allow_distributed=True)
        cache = TrainCache(args.cache_dir)
        data = EvaluationData(
            args.dataset, "validation", args.data_dir, args.cache_dir, ctx.device
        )
        if cache.identity["dataset"] != args.dataset:
            raise ValueError("training cache belongs to another dataset")
        identity = {
            key: cache.identity[key] for key in ("artifact_id", "representation_id")
        }
        validation = monitor_indices(tuple(data.indices))
        if len(validation) != 24 or len(set(validation)) != 24:
            raise ValueError(
                "formal training requires 24 distinct Validation trajectories"
            )

        def initialize() -> str:
            nonlocal lock
            if not args.resume:
                run.mkdir(parents=True, exist_ok=False)
            elif not (run / "recovery_latest.pt").is_file():
                raise FileNotFoundError(f"no recovery checkpoint in {run}")
            lock = acquire_run_lock(run)
            snapshot_sources(run, args.resume)
            if args.resume:
                if config != json.loads(
                    (run / "config.json").read_text(encoding="utf-8")
                ):
                    raise ValueError("resume configuration differs")
            else:
                write_json(run / "config.json", config)
                (run / "checkpoints").mkdir()
                (run / "validation").mkdir()
            attempt = run / f"attempt_{len(list(run.glob('attempt_*'))) + 1:03d}"
            attempt.mkdir()
            return str(attempt)

        attempt = Path(ctx.primary_call(initialize))
        seed_everything(config["seed"])
        model = JointGNN().to(ctx.device)
        ddp = DistributedDataParallel(
            model, device_ids=[ctx.device.index], broadcast_buffers=False
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
        ema = EMA(model, config["ema_decays"], windows_per_update=ctx.world)
        weight_names = ("raw", *ema.states)
        seed_everything(config["seed"] + ctx.rank * 1000003)
        run_id = ctx.broadcast(str(uuid.uuid4()) if ctx.primary else None)
        update, elapsed_before = 0, 0.0
        if args.resume:
            saved = load_checkpoint(run / "recovery_latest.pt")
            if (
                saved["format"] != CHECKPOINT_FORMAT
                or saved["config"] != config
                or any(saved[key] != value for key, value in identity.items())
                or len(saved["rank_rng"]) != ctx.world
            ):
                raise ValueError("resume model, representation, cache or world differs")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            ema.restore(saved["ema"])
            restore_rng(saved["rank_rng"][ctx.rank], ctx.device)
            run_id, update, elapsed_before = (
                saved["run_id"],
                saved["update"],
                saved["elapsed_seconds"],
            )
            if (
                saved["windows_seen"] != update * ctx.world
                or not 0 <= update <= config["budget_updates"]
            ):
                raise ValueError("invalid recovery cursor")
            del saved
        started = time.monotonic()
        metadata = {
            **identity,
            "dataset": args.dataset,
            "run_id": run_id,
            "world_size": ctx.world,
            "microbatch": 1,
            "accumulation": 1,
            "effective_batch": ctx.world,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "ema_decay_unit": "window",
            "ema_update_decays": ema.decays,
        }
        devices = ctx.gather(
            {
                "rank": ctx.rank,
                "gpu": torch.cuda.get_device_name(ctx.device),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
            }
        )
        ctx.primary_call(
            lambda: write_json(
                attempt / "launch.json",
                {
                    **metadata,
                    "resume_update": update,
                    "source": code_identity(),
                    "devices": devices,
                },
            )
        )

        def checkpoint(destinations: tuple[Path, ...]) -> None:
            rank_rng = ctx.gather(capture_rng(ctx.device))

            def write() -> None:
                payload = {
                    **metadata,
                    "format": CHECKPOINT_FORMAT,
                    "config": config,
                    "update": update,
                    "windows_seen": update * ctx.world,
                    "rank_rng": rank_rng,
                    "model": cpu_state(model.state_dict()),
                    "ema": {key: cpu_state(value) for key, value in ema.states.items()},
                    "optimizer": optimizer.state_dict(),
                    "elapsed_seconds": elapsed_before + time.monotonic() - started,
                }
                for destination in destinations:
                    save_checkpoint(destination, payload)

            ctx.primary_call(write)

        latest = run / "recovery_latest.pt"

        def checkpoint_and_select() -> None:
            relative = f"checkpoints/update_{update:09d}.pt"
            checkpoint((run / relative, latest))
            validate(ctx, model, ema, data, validation, run, run_id, update, relative)
            best = ctx.primary_call(lambda: select(run, run_id, update, weight_names))
            ctx.primary_call(
                lambda: write_json(
                    run / "status.json",
                    {
                        **metadata,
                        "state": "validated",
                        "update": update,
                        "windows_seen": update * ctx.world,
                        "selection": best,
                    },
                )
            )

        if update == 0:
            checkpoint((latest,))
        elif update % config["validation_every"] == 0:
            # A previous process may have completed training but not the validation transaction.
            checkpoint_and_select()

        permutation_epoch, order = -1, None
        train_indices = tuple(cache.indices)
        if not train_indices:
            raise ValueError("no Train windows")
        while update < config["budget_updates"]:
            epoch, offset = divmod(update * ctx.world + ctx.rank, len(train_indices))
            if permutation_epoch != epoch:
                order = torch.randperm(
                    len(train_indices),
                    generator=torch.Generator().manual_seed(
                        config["seed"] + epoch * 1000003
                    ),
                ).tolist()
                permutation_epoch = epoch
            index = train_indices[order[offset]]
            z0, context, graph, target = cache.window(index, ctx.device)
            lr = learning_rate(config, update + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            future = ddp(z0, context, graph)
            if future.shape != target.shape:
                raise ValueError(
                    f"predicted {tuple(future.shape)} vs target {tuple(target.shape)}"
                )
            # One equally weighted trajectory per rank. DDP averages these gradients over ranks.
            loss = F.mse_loss(future, target)
            mean_loss = float(ctx.mean(loss))
            if not math.isfinite(mean_loss):
                raise FloatingPointError("nonfinite trajectory loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            ema.update(model)
            update += 1
            if update == 1 or update % 25 == 0:
                hardware = ctx.gather(
                    {
                        "rank": ctx.rank,
                        "trajectory_index": index,
                        **peak_memory(ctx.device),
                    }
                )
                row = {
                    "update": update,
                    "windows_seen": update * ctx.world,
                    "loss": mean_loss,
                    "gradient_norm": float(grad_norm),
                    "lr": lr,
                    "devices": hardware,
                    "elapsed_seconds": elapsed_before + time.monotonic() - started,
                }

                def progress() -> None:
                    append_json(attempt / "training.jsonl", row)
                    write_json(
                        run / "status.json", {**row, **metadata, "state": "training"}
                    )
                    print(json.dumps(row), flush=True)

                ctx.primary_call(progress)
            if update % config["validation_every"] == 0:
                checkpoint_and_select()
            elif update % config["recovery_every"] == 0:
                checkpoint((latest,))
        best = ctx.primary_call(lambda: select(run, run_id, update, weight_names))
        state = (
            "complete" if best["checkpoint"] else "complete_without_finite_validation"
        )
        ctx.primary_call(
            lambda: write_json(
                run / "status.json",
                {
                    **metadata,
                    "state": state,
                    "update": update,
                    "windows_seen": update * ctx.world,
                    "elapsed_seconds": elapsed_before + time.monotonic() - started,
                    "selection": best,
                },
            )
        )
        if not best["checkpoint"]:
            raise RuntimeError(
                "No candidate had a finite score on every Validation24 trajectory"
            )
    finally:
        if lock is not None:
            lock.close()
        ctx.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["airfoil", "cylinderflow"])
    for name in ("data-dir", "cache-dir", "result-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verification-updates",
        type=int,
        help="shorten workload for a separately named four-GPU verification run",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
