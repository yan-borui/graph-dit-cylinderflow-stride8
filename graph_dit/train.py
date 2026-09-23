"""Train one H1/B1 candidate with resumable schedules, raw weights and two EMAs."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
from types import FrameType
import uuid

import h5py
import numpy as np
import torch

from dgn4cfd.nn.diffusion.models.graph_video_dit import GraphVideoDiT
from .config import load_config, validate_config, learning_rate
from .representation import load_artifacts
from .metrics import selection_key
from .physical_monitor import PhysicalMonitor
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


def configure_runtime(
    device: torch.device, precision: str, *, allow_distributed: bool = False
) -> None:
    if not allow_distributed and int(os.environ.get("WORLD_SIZE", "1")) != 1:
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
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this device does not support BF16")
        # Keep FP32 SDPA backends fixed for dense and exact H1/H2 neighborhoods.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.cuda.reset_peak_memory_stats(device)


class EMA:
    """FP32 device-resident averages, updated once after each optimizer step."""

    def __init__(
        self, model: GraphVideoDiT, decays: list[float], *, windows_per_update: int = 1
    ):
        self.decays = {f"ema_{decay:g}": decay**windows_per_update for decay in decays}
        self.states = {
            name: {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            for name in self.decays
        }
        self.parameter_names = set(dict(model.named_parameters()))

    @torch.no_grad()
    def update(self, model: GraphVideoDiT) -> None:
        current = model.state_dict()
        parameter_keys = [key for key in current if key in self.parameter_names]
        parameters = [current[key] for key in parameter_keys]
        for name, decay in self.decays.items():
            averaged = [self.states[name][key] for key in parameter_keys]
            # Keep the original multiply-then-add rounding and per-update decay.
            torch._foreach_mul_(averaged, decay)
            torch._foreach_add_(averaged, parameters, alpha=1 - decay)
            for key, value in current.items():
                if key not in self.parameter_names:
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


@contextmanager
def graceful_pause_signals() -> Iterator[dict[str, str | None]]:
    """Defer INT/TERM until a completed optimizer/EMA update can be saved."""
    request = {"signal": None}

    def request_pause(number: int, _frame: FrameType | None) -> None:
        request["signal"] = signal.Signals(number).name

    previous = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for number in previous:
            signal.signal(number, request_pause)
        yield request
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


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
    acceptance: dict | None = None,
) -> dict:
    validate_config(config)
    if config.get("distributed", {}).get("world_size", 1) > 1:
        if acceptance is not None:
            raise ValueError("this acceptance contract requires one GPU")
        from .ddp_train import train_distributed

        return train_distributed(
            config,
            artifacts,
            data_dir,
            run,
            stage_end,
            device_name,
            resume=resume,
            debug=debug,
        )
    training = config["training"]
    if not 1 <= stage_end <= training["schedule_total_updates"]:
        raise ValueError("stage endpoint must lie within the immutable LR plan")
    if acceptance is not None and (
        acceptance.get("format") != "graph_dit.real_data_acceptance.v1"
        or acceptance.get("updates") != 8
        or acceptance.get("resume_update") != 4
        or stage_end not in (4, 8)
        or len(acceptance.get("validation_indices", [])) != 1
    ):
        raise ValueError("invalid bounded real-data acceptance contract")
    device = torch.device(device_name)
    configure_runtime(device, training["precision"])
    identity = load_artifacts(artifacts, config=config)
    if identity["debug"] != debug:
        raise ValueError("synthetic artifacts and formal runs cannot be mixed")
    if resume:
        if not (run / "recovery_latest.pt").is_file():
            raise FileNotFoundError("resume needs this run's recovery_latest.pt")
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError(
                "resume cannot change architecture, LR plan, EMA, seeds, or evaluation"
            )
        saved_acceptance = (
            json.loads((run / "acceptance.json").read_text())
            if (run / "acceptance.json").exists()
            else None
        )
        if saved_acceptance != acceptance:
            raise ValueError("acceptance and formal runs cannot be mixed")
    else:
        run.mkdir(parents=True, exist_ok=False)
        if acceptance is not None:
            write_json(run / "acceptance.json", acceptance)
    lock = acquire_run_lock(run)
    try:
        freeze_source(run, resume)
        with graceful_pause_signals() as pause_request:
            return _train_locked(
                config,
                artifacts,
                data_dir,
                run,
                stage_end,
                device,
                identity,
                resume,
                debug,
                acceptance,
                pause_request,
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
    acceptance: dict | None,
    pause_request: dict,
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
    if acceptance is not None and not set(acceptance["validation_indices"]).issubset(
        predictor.data.splits["validation"]
    ):
        raise ValueError("acceptance must evaluate a real Validation trajectory")
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
            or saved.get("acceptance") != acceptance
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
    physical_monitor = None
    strict_selection = (
        config["validation"].get("selection")
        == "validation24_complete_strict_uv_raw_ema"
    )
    if strict_selection or config["validation"].get("early_stopping", {}).get(
        "enabled", False
    ):
        physical_monitor = PhysicalMonitor(run, config["validation"], update)
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
            "acceptance": acceptance,
        },
    )
    if resume and acceptance is not None:
        restored = {
            "update": update,
            "raw_equal": all(
                torch.equal(value.detach().cpu(), saved["model"][key])
                for key, value in model.state_dict().items()
            ),
            "ema_equal": all(
                torch.equal(value.detach().cpu(), saved["ema"][name][key])
                for name, state in ema.states.items()
                for key, value in state.items()
            ),
            "generator_equal": torch.equal(
                generator.get_state().cpu(), saved["training_generator"]
            ),
            "optimizer_steps": sorted(
                {int(state["step"]) for state in optimizer.state.values()}
            ),
        }
        write_json(attempt_dir / "restored.json", restored)
        if (
            update != acceptance["resume_update"]
            or not all(
                restored[key] for key in ("raw_equal", "ema_equal", "generator_equal")
            )
            or restored["optimizer_steps"] != [update]
        ):
            raise ValueError("acceptance checkpoint restoration failed")
    if resume:
        del saved
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
            "checkpoint_id": f"{run_id}:{update}"
            if physical_monitor
            else str(uuid.uuid4()),
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
            "acceptance": acceptance,
        }
        if acceptance is not None:
            payload["parameter_names"] = sorted(ema.parameter_names)
        if physical_monitor is not None:
            payload["physical_monitor_state"] = deepcopy(physical_monitor.state)
        saving_started = time.perf_counter()
        save_checkpoint(destination, payload)
        costs["checkpoint_io_seconds"] += time.perf_counter() - saving_started
        return payload

    def pause_if_requested() -> dict | None:
        pause_file = run / "PAUSE"
        if pause_request["signal"] is None and not pause_file.exists():
            return None
        checkpoint(run / "recovery_latest.pt")
        result = {
            "state": "paused",
            "termination_reason": "user_request",
            "signal": pause_request["signal"],
            "update": update,
            "sample_cursor": update,
            "stage_end_updates": stage_end,
            "run_id": run_id,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            **costs,
            **peak_memory(device),
        }
        write_json(run / "status.json", result)
        write_json(attempt_dir / "paused.json", result)
        (attempt_dir / "exit_code").write_text("0\n")
        return result

    def validate_current() -> None:
        checkpoint_relative = f"checkpoints/update_{update:09d}.pt"
        checkpoint_file = run / checkpoint_relative
        if not checkpoint_file.exists():
            snapshot = checkpoint(checkpoint_file)
        else:
            snapshot = load_checkpoint(checkpoint_file)
        candidates = (
            physical_monitor.candidates(update)
            if physical_monitor is not None
            else read_jsonl(run / "candidates.jsonl")
        )
        raw = cpu_state(model.state_dict())
        saved_rng = rng_state()
        saved_generator = generator.get_state()
        saved_mode = model.training
        indices = (
            tuple(acceptance["validation_indices"])
            if acceptance is not None
            else monitor_indices(predictor.data.splits["validation"])
        )
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
                    "scope": "acceptance_validation1"
                    if acceptance is not None
                    else "validation24_monitor",
                    "debug": debug,
                }
                validation_started = time.perf_counter()
                summary = evaluate_model(
                    predictor,
                    indices,
                    config["validation"]["sampling_seeds"],
                    destination,
                    provenance,
                    fail_on_runtime_error=physical_monitor is not None,
                )
                costs["validation_seconds"] += time.perf_counter() - validation_started
                row = {
                    **provenance,
                    "checkpoint": checkpoint_relative,
                    "summary": str((destination / "summary.json").relative_to(run)),
                    "failed_clips": summary["failed_clips"],
                    "clip_count": summary["clip_count"],
                    "trajectory_count": summary["trajectory_count"],
                    "score": summary["selection_uv_relative_rmse"],
                    "selection_key": list(selection_key(summary, update)),
                    "complete_stage": False,
                }
                if physical_monitor is not None:
                    physical_monitor.store_candidate(row)
                else:
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
            if physical_monitor is not None:
                physical_monitor.complete(
                    update,
                    snapshot["checkpoint_id"],
                    len(indices) * len(config["validation"]["sampling_seeds"]),
                    len(indices),
                )
        except BaseException as failure:
            if physical_monitor is not None:
                physical_monitor.error(update, attempt, failure)
            raise
        finally:
            model.load_state_dict(raw)
            model.train(saved_mode)
            restore_rng(saved_rng)
            generator.set_state(saved_generator)
        if physical_monitor is not None:
            best = physical_monitor.state["best"]
            write_json(
                selected_file,
                best
                or {
                    "weights": None,
                    "update": None,
                    "checkpoint": None,
                    "score": None,
                    "failed_clips": None,
                },
            )
            checkpoint(run / "recovery_latest.pt")
            print(
                json.dumps(
                    {
                        "event": "physical_round",
                        "update": update,
                        "physical_monitor_state": physical_monitor.state,
                    }
                ),
                flush=True,
            )
        elif candidates:
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
            if acceptance is not None:
                shutil.copyfile(
                    run / "recovery_latest.pt",
                    run / "checkpoints" / "update_000000000.pt",
                )
        with h5py.File(artifacts / "train_latents.h5", "r") as cache:
            indices = [int(value) for value in cache["sim_indices"]]
            if acceptance is not None:
                if acceptance["train_index"] not in indices:
                    raise ValueError("acceptance graph is outside Train")
                indices = [acceptance["train_index"]]
            if update and (
                update % config["validation"]["every_updates"] == 0
                or update == stage_end
            ):
                validate_current()
            current_epoch, permutation = -1, None
            while update < stage_end and not (
                physical_monitor is not None
                and physical_monitor.state["stop_requested"]
            ):
                paused = pause_if_requested()
                if paused is not None:
                    return paused
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
                attention_norm = None
                if acceptance is not None:
                    attention_norm = float(
                        model.blocks[0].attention.in_proj_weight.grad.norm()
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
                    or acceptance is not None
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
                    if acceptance is not None:
                        row["attention_gradient_norm"] = attention_norm
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
                if (
                    acceptance is not None
                    and acceptance.get("pause_at_update") == update
                ):
                    (run / "PAUSE").write_text("bounded acceptance pause boundary\n")
                paused = pause_if_requested()
                if paused is not None:
                    return paused
            # Includes restored RNG and all EMA states after an interrupted monitor finishes.
            checkpoint(run / "recovery_latest.pt")
        selected = json.loads(selected_file.read_text())
        selected.update(complete_stage=True, stage_end_updates=stage_end)
        write_json(selected_file, selected)
        stopped = (
            physical_monitor is not None
            and physical_monitor.state["stop_requested"]
            and update < stage_end
        )
        result = {
            "state": "early_stopped" if stopped else "complete",
            "termination_reason": "physical_validation_plateau"
            if stopped
            else "allocated_budget",
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
        if physical_monitor is not None:
            result["physical_monitor_state"] = physical_monitor.state
            write_json(
                run / "checkpoint_inventory.json",
                {
                    "updates": sorted(
                        int(item.stem.split("_")[-1])
                        for item in (run / "checkpoints").glob("update_*.pt")
                        if int(item.stem.split("_")[-1]) <= update
                    ),
                    "last_checkpoint": f"checkpoints/update_{update:09d}.pt",
                    "best_physical_checkpoint": selected.get("checkpoint"),
                    "best_physical_weights": selected.get("weights"),
                    "best_physical_score": selected.get("score"),
                    "termination_reason": result["termination_reason"],
                },
            )
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
