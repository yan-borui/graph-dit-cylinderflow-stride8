"""Train one H1/B1 candidate with resumable schedules, raw weights and two EMAs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid

import h5py
import numpy as np
import torch

from dgn4cfd.nn.diffusion.models.graph_video_dit import GraphVideoDiT
from .config import load_config, validate_config, learning_rate
from .representation import load_artifacts
from .metrics import selection_key
from .runtime import (
    ROOT,
    acquire_run_lock,
    append_json,
    autocast,
    code_identity,
    load_checkpoint,
    monitor_indices,
    peak_memory,
    read_jsonl,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
    synchronize,
    write_json,
)

CHECKPOINT_FORMAT = "graph_dit.h1_b1.training.v1"


def configure_runtime(device: torch.device, precision: str) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this recipe needs one independent process per GPU, not DDP")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this device does not support BF16")
        # H1 uses an additive dense mask. Record actual runtime; do not claim sparse speedup.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.cuda.reset_peak_memory_stats(device)


class EMA:
    """FP32 device-resident averages, updated once after each optimizer step."""

    def __init__(self, model: GraphVideoDiT, decays: list[float]):
        self.decays = {f"ema_{decay:g}": decay for decay in decays}
        self.states = {
            name: {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            for name in self.decays
        }
        self.parameter_names = set(dict(model.named_parameters()))

    @torch.no_grad()
    def update(self, model: GraphVideoDiT) -> None:
        for name, decay in self.decays.items():
            for key, value in model.state_dict().items():
                if key in self.parameter_names:
                    self.states[name][key].mul_(decay).add_(
                        value.detach(), alpha=1 - decay
                    )
                else:
                    self.states[name][key].copy_(value)

    def restore(self, states: dict) -> None:
        if set(states) != set(self.states):
            raise ValueError("EMA bank differs from the frozen recipe")
        for name in states:
            if set(states[name]) != set(self.states[name]):
                raise ValueError("EMA parameter/buffer names differ")
            for key, value in states[name].items():
                self.states[name][key].copy_(value)


def cpu_state(state: dict) -> dict:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def window_at(cache: h5py.File, index: int, device: torch.device) -> dict:
    group = cache[f"sim_{index:05d}"]
    # Exactly one fixed first-65-frame window per trajectory, independent of allocation length.
    latent = torch.from_numpy(np.asarray(group["latents"][:65], dtype=np.float32)).to(
        device
    )[None]
    return {
        "clean_z0_raw": latent[:, :1],
        "clean_future_raw": latent[:, 1:],
        "node_context": torch.from_numpy(
            np.asarray(group["node_context"], dtype=np.float32)
        ).to(device)[None],
        "positions": torch.from_numpy(
            np.asarray(group["positions"], dtype=np.float32)
        ).to(device)[None],
        "graph_hops": torch.from_numpy(
            np.asarray(group["graph_hops"], dtype=np.int64)
        ).to(device)[None],
    }


def freeze_source(run: Path, resume: bool) -> None:
    source = run / "source"
    files = sorted(ROOT.glob("graph_dit/*.py")) + sorted(ROOT.glob("dgn4cfd/**/*.py"))
    for original in files:
        destination = source / original.relative_to(ROOT)
        if resume:
            if (
                not destination.is_file()
                or destination.read_bytes() != original.read_bytes()
            ):
                raise ValueError(
                    f"source changed since this run: {original.relative_to(ROOT)}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)


def train_run(
    config: dict,
    artifacts: Path,
    data_dir: Path,
    run: Path,
    stage_end: int,
    device_name: str,
    *,
    resume: bool = False,
    debug: bool = False,
) -> dict:
    validate_config(config)
    training = config["training"]
    if not 1 <= stage_end <= training["schedule_total_updates"]:
        raise ValueError("stage endpoint must lie within the immutable LR plan")
    device = torch.device(device_name)
    configure_runtime(device, training["precision"])
    identity = load_artifacts(artifacts)
    if identity["debug"] != debug:
        raise ValueError("synthetic artifacts and formal runs cannot be mixed")
    if resume:
        if not (run / "recovery_latest.pt").is_file():
            raise FileNotFoundError("resume needs this run's recovery_latest.pt")
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError(
                "resume cannot change architecture, LR plan, EMA, seeds, or evaluation"
            )
    else:
        run.mkdir(parents=True, exist_ok=False)
    lock = acquire_run_lock(run)
    try:
        freeze_source(run, resume)
        return _train_locked(
            config, artifacts, data_dir, run, stage_end, device, identity, resume, debug
        )
    finally:
        lock.close()


def _train_locked(
    config: dict,
    artifacts: Path,
    data_dir: Path,
    run: Path,
    stage_end: int,
    device: torch.device,
    identity: dict,
    resume: bool,
    debug: bool,
) -> dict:
    from .evaluate import Predictor, evaluate_model

    training = config["training"]
    write_json(run / "config.json", config)
    seed_everything(config["seed"])
    model = GraphVideoDiT(**config["model"]).to(device)
    model.set_latent_statistics(identity["latent_mean"], identity["latent_std"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    ema = EMA(model, training["ema_decays"])
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 17011)
    predictor = Predictor(
        model, artifacts, data_dir, device, training["precision"], debug=debug
    )
    predictor.model = model
    update, elapsed_before, run_id = 0, 0.0, str(uuid.uuid4())
    costs = {
        "train_update_seconds": 0.0,
        "validation_seconds": 0.0,
        "checkpoint_io_seconds": 0.0,
    }
    if resume:
        saved = load_checkpoint(run / "recovery_latest.pt")
        if (
            saved["format"] != CHECKPOINT_FORMAT
            or saved["config"] != config
            or saved["artifact_id"] != identity["artifact_id"]
        ):
            raise ValueError("resume checkpoint/config/representation mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        ema.restore(saved["ema"])
        generator.set_state(saved["training_generator"].cpu())
        restore_rng(saved["rng"])
        update, elapsed_before, run_id = (
            saved["update"],
            saved["elapsed_seconds"],
            saved["run_id"],
        )
        costs.update(saved.get("costs", {}))
        if stage_end < update:
            raise ValueError("stage endpoint precedes the restored sample cursor")
    (run / "checkpoints").mkdir(exist_ok=True)
    attempt = len(list(run.glob("attempt_*"))) + 1
    attempt_dir = run / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(exist_ok=False)
    write_json(
        attempt_dir / "launch.json",
        {
            "stage_end_updates": stage_end,
            "resume_update": update,
            "config": config,
            "code": code_identity(),
            "artifact_id": identity["artifact_id"],
            "device": str(device),
            "precision": training["precision"],
            "parameter_dtype": str(next(model.parameters()).dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "torch": str(torch.__version__),
            "debug": debug,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "run_id": run_id,
        },
    )
    write_json(
        run / "status.json",
        {"state": "running", "stage_end_updates": stage_end, "update": update},
    )
    selected_file = run / "selection.json"
    if selected_file.exists():
        prior = json.loads(selected_file.read_text())
        prior["complete_stage"] = False
        write_json(selected_file, prior)
    started = time.perf_counter()

    def checkpoint(destination: Path) -> dict:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "checkpoint_id": str(uuid.uuid4()),
            "run_id": run_id,
            "model": cpu_state(model.state_dict()),
            "ema": {name: cpu_state(state) for name, state in ema.states.items()},
            "optimizer": optimizer.state_dict(),
            "config": deepcopy(config),
            "update": update,
            "sample_cursor": update,
            "artifact_id": identity["artifact_id"],
            "debug": debug,
            "representation_id": identity["representation_id"],
            "rng": rng_state(),
            "training_generator": generator.get_state().cpu(),
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "costs": dict(costs),
        }
        saving_started = time.perf_counter()
        save_checkpoint(destination, payload)
        costs["checkpoint_io_seconds"] += time.perf_counter() - saving_started
        return payload

    def validate_current() -> None:
        checkpoint_relative = f"checkpoints/update_{update:09d}.pt"
        checkpoint_file = run / checkpoint_relative
        if not checkpoint_file.exists():
            snapshot = checkpoint(checkpoint_file)
        else:
            snapshot = load_checkpoint(checkpoint_file)
        candidates = read_jsonl(run / "candidates.jsonl")
        raw = cpu_state(model.state_dict())
        saved_rng = rng_state()
        try:
            for weights in config["validation"]["weights"]:
                if any(
                    row["checkpoint_id"] == snapshot["checkpoint_id"]
                    and row["weights"] == weights
                    for row in candidates
                ):
                    continue
                model.load_state_dict(raw if weights == "raw" else ema.states[weights])
                destination = (
                    run
                    / "monitor"
                    / f"update_{update:09d}_{weights}_attempt{attempt:03d}"
                )
                provenance = {
                    "checkpoint_id": snapshot["checkpoint_id"],
                    "artifact_id": identity["artifact_id"],
                    "weights": weights,
                    "update": update,
                    "training_seed": config["seed"],
                    "config": config,
                    "scope": "validation24_monitor",
                    "debug": debug,
                }
                validation_started = time.perf_counter()
                summary = evaluate_model(
                    predictor,
                    monitor_indices(predictor.data.splits["validation"]),
                    config["validation"]["sampling_seeds"],
                    destination,
                    provenance,
                )
                costs["validation_seconds"] += time.perf_counter() - validation_started
                row = {
                    **provenance,
                    "checkpoint": checkpoint_relative,
                    "summary": str((destination / "summary.json").relative_to(run)),
                    "failed_clips": summary["failed_clips"],
                    "score": summary["selection_uv_relative_rmse"],
                    "selection_key": list(selection_key(summary, update)),
                    "complete_stage": False,
                }
                append_json(run / "candidates.jsonl", row)
                candidates.append(row)
                print(
                    json.dumps(
                        {
                            "event": "validation",
                            "update": update,
                            "weights": weights,
                            "score": row["score"],
                            "failed_clips": row["failed_clips"],
                        }
                    ),
                    flush=True,
                )
        finally:
            model.load_state_dict(raw)
            model.train()
            restore_rng(saved_rng)
        if candidates:
            best = min(
                candidates,
                key=lambda row: (
                    row["failed_clips"],
                    float("inf") if row["score"] is None else row["score"],
                    row["update"],
                    config["validation"]["weights"].index(row["weights"]),
                ),
            )
            write_json(selected_file, best)

    try:
        if not resume:
            checkpoint(run / "recovery_latest.pt")
        with h5py.File(artifacts / "train_latents.h5", "r") as cache:
            indices = [int(value) for value in cache["sim_indices"]]
            if update and (
                update % config["validation"]["every_updates"] == 0
                or update == stage_end
            ):
                validate_current()
            current_epoch, permutation = -1, None
            while update < stage_end:
                step_started = time.perf_counter()
                epoch, offset = divmod(update, len(indices))
                if epoch != current_epoch:
                    permutation = torch.randperm(
                        len(indices),
                        generator=torch.Generator().manual_seed(
                            config["seed"] + epoch * 1000003
                        ),
                    )
                    current_epoch = epoch
                index = indices[int(permutation[offset])]
                sample = window_at(cache, index, device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(config, update + 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with autocast(device, training["precision"]):
                    loss = model.training_loss(**sample, generator=generator)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite epsilon training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training["gradient_clip"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                ema.update(model)
                synchronize(device)
                costs["train_update_seconds"] += time.perf_counter() - step_started
                update += 1
                if (
                    update % training["log_every_updates"] == 0
                    or update == 1
                    or update == stage_end
                ):
                    synchronize(device)
                    row = {
                        "event": "update",
                        "update": update,
                        "examples_seen": update,
                        "epoch": update / len(indices),
                        "loss": float(loss.detach()),
                        "gradient_norm_pre_clip": float(norm),
                        "learning_rate": lr,
                        "trajectory_index": index,
                        "elapsed_seconds": elapsed_before
                        + time.perf_counter()
                        - started,
                        "attempt": attempt,
                        **costs,
                        **peak_memory(device),
                    }
                    append_json(attempt_dir / "training.jsonl", row)
                    print(json.dumps(row), flush=True)
                    write_json(
                        run / "status.json",
                        {**row, "state": "running", "stage_end_updates": stage_end},
                    )
                if update % training["checkpoint_every_updates"] == 0:
                    destination = run / "checkpoints" / f"update_{update:09d}.pt"
                    if not destination.exists():
                        checkpoint(destination)
                if (
                    update % training["recovery_every_updates"] == 0
                    or update == stage_end
                ):
                    checkpoint(run / "recovery_latest.pt")
                if (
                    update % config["validation"]["every_updates"] == 0
                    or update == stage_end
                ):
                    validate_current()
            # Includes restored RNG and all EMA states after an interrupted monitor finishes.
            checkpoint(run / "recovery_latest.pt")
        selected = json.loads(selected_file.read_text())
        selected.update(complete_stage=True, stage_end_updates=stage_end)
        write_json(selected_file, selected)
        result = {
            "state": "complete",
            "update": update,
            "stage_end_updates": stage_end,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "selected_weights": selected["weights"],
            "selected_update": selected["update"],
            "score": selected["score"],
            "failed_clips": selected["failed_clips"],
            **costs,
            "training_gpu_hours": costs["train_update_seconds"] / 3600
            if device.type == "cuda"
            else None,
            "allocated_gpu_hours": (elapsed_before + time.perf_counter() - started)
            / 3600
            if device.type == "cuda"
            else None,
            **peak_memory(device),
        }
        write_json(run / "status.json", result)
        (attempt_dir / "exit_code").write_text("0\n")
        return result
    except BaseException as failure:
        (attempt_dir / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        try:
            checkpoint(attempt_dir / "failure_state.pt")
        except Exception:
            pass
        write_json(
            run / "status.json",
            {
                "state": "failed",
                "update": update,
                "stage_end_updates": stage_end,
                "error": f"{type(failure).__name__}: {failure}",
                "attempt": attempt,
            },
        )
        (attempt_dir / "exit_code").write_text("1\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-end-updates", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_run(
        load_config(args.config),
        args.artifacts,
        args.data_dir,
        args.output_dir,
        args.stage_end_updates,
        args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
