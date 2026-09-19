"""Fixed-world H1 training with a global window clock and resumable physical evaluation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid

import h5py
import torch
from torch.nn.parallel import DistributedDataParallel

from dgn4cfd.nn.diffusion.models.graph_video_dit import GraphVideoDiT
from .config import learning_rate, validate_config
from .ablation_contract import CHECKPOINT_FORMAT as ABLATION_CHECKPOINT, is_ablation
from .distributed import Context, capture_rng, describe_device, restore_rng
from .evaluate import Predictor, evaluate_model
from .metrics import summarize_trajectories
from .representation import load_artifacts
from .runtime import (
    acquire_run_lock,
    append_json,
    autocast,
    clean_json,
    code_identity,
    load_checkpoint,
    monitor_indices,
    peak_memory,
    save_checkpoint,
    seed_everything,
    synchronize,
    write_csv,
    write_json,
)
from .train import EMA, configure_runtime, cpu_state, freeze_source, window_at

CHECKPOINT_FORMAT = "graph_dit.h1_ddp.training.v2"


class LossForward(torch.nn.Module):
    """Ensure the stochastic training forward is owned by DDP."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, generator):
        return self.model.training_loss(**sample, generator=generator)


def atomic_rows(destination: Path, rows: list[dict]) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(clean_json(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def merge_evaluation(
    output: Path, indices, seeds, provenance: dict, world: int
) -> dict:
    rows = []
    for rank in range(world):
        shard = output / f"rank_{rank:03d}"
        rows.extend(json.loads((shard / "completed_cases.json").read_text()))
        for row in json.loads((shard / "completed_cases.json").read_text()):
            source = shard / row["prediction_file"]
            destination = output / row["prediction_file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copyfile(source, destination)
    rows.sort(key=lambda row: (row["trajectory_index"], row["seed"]))
    expected = {(index, seed) for index in indices for seed in seeds}
    actual = [(row["trajectory_index"], row["seed"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("distributed validation has missing or duplicate clips")
    summary = summarize_trajectories(rows)
    summary.update(indices=list(indices), sampling_seeds=seeds, provenance=provenance)
    atomic_rows(output / "case_metrics.jsonl", rows)
    write_csv(output / "case_metrics.csv", rows)
    write_csv(output / "trajectory_metrics.csv", summary["trajectory_metrics"])
    write_json(
        output / "failures.json", {"failures": [r for r in rows if not r["finite"]]}
    )
    write_json(
        output / "identity.json",
        {**provenance, "indices": list(indices), "sampling_seeds": seeds},
    )
    write_json(output / "summary.json", summary)
    return summary


def train_distributed(
    config,
    artifacts,
    data_dir,
    run,
    stage_end,
    device_name,
    *,
    resume=False,
    debug=False,
):
    validate_config(config)
    training = config["training"]
    ablation = is_ablation(config)
    checkpoint_format = ABLATION_CHECKPOINT if ablation else CHECKPOINT_FORMAT
    if not 1 <= stage_end <= training["schedule_total_updates"]:
        raise ValueError("stage endpoint lies outside the immutable LR schedule")
    if (
        "scaling" in config
        and not config.get("preflight_max_graph")
        and stage_end != training["budget_updates"]
    ):
        raise ValueError("formal scaling runs use the common 125k update endpoint")
    ctx = Context(config["distributed"]["world_size"], device_name)
    device, lock, attempt_dir = ctx.device, None, None
    update = 0
    try:
        configure_runtime(device, training["precision"], allow_distributed=True)
        devices = ctx.all_call(lambda: describe_device(config, device))
        environment = None
        if ablation:
            from .ablation_runtime import bind_environment

            if (
                not config.get("preflight_max_graph")
                and stage_end != training["budget_updates"]
            ):
                raise ValueError(
                    "formal ablation runs use the fixed 62,500-update endpoint"
                )
            environment = bind_environment(config, devices, run, ctx)
        identity = load_artifacts(artifacts, config=config)
        if ablation:
            from .ablation_runtime import bind_inputs

            bind_inputs(identity, run, ctx)
        if identity["debug"] != debug:
            raise ValueError("formal/synthetic representation mismatch")
        identities = ctx.gather(
            (identity["artifact_id"], identity["representation_id"], config)
        )
        if any(item != identities[0] for item in identities):
            raise ValueError(
                "all ranks must use the same config and prepared representation"
            )

        def prepare():
            nonlocal lock
            if resume:
                if json.loads((run / "config.json").read_text()) != config:
                    raise ValueError("resume cannot change the frozen configuration")
            else:
                run.mkdir(parents=True, exist_ok=False)
            lock = acquire_run_lock(run)
            freeze_source(run, resume)
            write_json(run / "config.json", config)
            (run / "checkpoints").mkdir(exist_ok=True)
            attempt = len(list(run.glob("attempt_*"))) + 1
            folder = run / f"attempt_{attempt:03d}"
            folder.mkdir()
            return str(folder)

        attempt_dir = Path(ctx.primary_call(prepare))
        seed_everything(config["seed"])
        model = GraphVideoDiT(**config["model"]).to(device)
        model.activation_checkpointing = training.get("activation_checkpointing", False)
        model.set_latent_statistics(identity["latent_mean"], identity["latent_std"])
        ddp = DistributedDataParallel(
            LossForward(model),
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=config["distributed"].get(
                "gradient_as_bucket_view", False
            ),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        ema = EMA(
            model,
            training["ema_decays"],
            windows_per_update=ctx.world
            if training.get("ema_decay_unit") == "window"
            else 1,
        )
        initial_parameters = (
            cpu_state(model.state_dict()) if config.get("preflight_max_graph") else None
        )
        # Model initialization is shared; training noise/dropout streams belong to each rank.
        seed_everything(config["seed"] + ctx.rank * 1000003)
        generator = torch.Generator(device=device).manual_seed(
            config["seed"] + 17011 + ctx.rank * 1000003
        )
        predictor = Predictor(
            model, artifacts, data_dir, device, training["precision"], debug=debug
        )
        run_id = ctx.broadcast(str(uuid.uuid4()) if ctx.primary else None)
        elapsed_before = 0.0
        costs = {
            "train_update_seconds": 0.0,
            "validation_seconds": 0.0,
            "checkpoint_io_seconds": 0.0,
        }
        if resume:
            saved = load_checkpoint(run / "recovery_latest.pt")
            if (
                saved["format"] != checkpoint_format
                or saved["config"] != config
                or saved["artifact_id"] != identity["artifact_id"]
                or len(saved["rank_states"]) != ctx.world
            ):
                raise ValueError("checkpoint/config/world/representation mismatch")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            ema.restore(saved["ema"])
            local = saved["rank_states"][ctx.rank]
            generator.set_state(local["training_generator"].cpu())
            restore_rng(local["rng"], device)
            update = saved["update"]
            if saved["sample_cursor"] != update * ctx.world or update > stage_end:
                raise ValueError("invalid restored global sample cursor")
            elapsed_before, run_id = saved["elapsed_seconds"], saved["run_id"]
            costs.update(saved["costs"])
            if ablation:
                if saved.get("environment") != environment:
                    raise ValueError(
                        "resume environment differs from the saved four-card runtime"
                    )
                if config.get("preflight_max_graph"):
                    from .ablation_runtime import state_equal

                    def verify_restoration():
                        checks = {
                            "model": state_equal(
                                cpu_state(model.state_dict()), saved["model"]
                            ),
                            "ema": state_equal(ema.states, saved["ema"]),
                            "optimizer": state_equal(
                                optimizer.state_dict(), saved["optimizer"]
                            ),
                            "rng": state_equal(capture_rng(device), local["rng"]),
                            "training_generator": torch.equal(
                                generator.get_state().cpu(), local["training_generator"]
                            ),
                        }
                        if not all(checks.values()) or update != 4:
                            raise RuntimeError(
                                f"acceptance restoration mismatch: {checks}"
                            )
                        return {"rank": ctx.rank, "update": update, "checks": checks}

                    restored = ctx.all_call(verify_restoration)
                    ctx.primary_call(
                        lambda: write_json(
                            attempt_dir / "restored.json", {"ranks": restored}
                        )
                    )
            del saved
        started = time.perf_counter()
        metadata = {
            "world_size": ctx.world,
            "effective_batch": ctx.world,
            "ema_decay_unit": training.get("ema_decay_unit", "update"),
            "ema_update_decays": ema.decays,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "activation_checkpointing": model.activation_checkpointing,
            "gradient_as_bucket_view": ddp.gradient_as_bucket_view,
            "environment": environment,
        }
        if (
            "scaling" in config
            and metadata["parameter_count"] != config["scaling"]["parameter_count"]
        ):
            raise ValueError(
                "instantiated parameter count differs from the scaling plan"
            )
        if (
            ablation
            and metadata["parameter_count"] != config["ablation"]["parameter_count"]
        ):
            raise ValueError(
                "instantiated parameter count differs from the ablation plan"
            )
        ctx.primary_call(
            lambda: write_json(
                attempt_dir / "launch.json",
                {
                    **metadata,
                    "devices": devices,
                    "stage_end_updates": stage_end,
                    "resume_update": update,
                    "config": config,
                    "code": code_identity(),
                    "artifact_id": identity["artifact_id"],
                    "run_id": run_id,
                    "debug": debug,
                    "tf32": False,
                },
            )
        )
        candidates_dir = run / "evaluation_records"
        ctx.primary_call(lambda: candidates_dir.mkdir(exist_ok=True))

        def records():
            return [
                json.loads(item.read_text())
                for item in sorted(candidates_dir.glob("*.json"))
                if int(item.name.split("_")[0]) <= update
            ]

        def selection(complete=False, *, endpoint=False):
            candidates = records()
            if endpoint:
                candidates = [row for row in candidates if row["update"] == stage_end]
            valid = [
                row
                for row in candidates
                if row["failed_clips"] == 0
                and row["score"] is not None
                and math.isfinite(row["score"])
                and (
                    not ablation
                    or (
                        row["trajectory_count"]
                        == (4 if config.get("preflight_max_graph") else 24)
                        and row["clip_count"]
                        == (12 if config.get("preflight_max_graph") else 72)
                    )
                )
            ]
            best = (
                min(
                    valid,
                    key=lambda row: (
                        row["score"],
                        row["update"],
                        config["validation"]["weights"].index(row["weights"]),
                    ),
                )
                if valid
                else {
                    "weights": None,
                    "update": None,
                    "checkpoint": None,
                    "score": None,
                    "failed_clips": None,
                }
            )
            best = {**best, "complete_stage": complete, "stage_end_updates": stage_end}
            file_name = "selection_endpoint.json" if endpoint else "selection.json"
            write_json(run / file_name, best)
            if not endpoint:
                atomic_rows(run / "candidates.jsonl", candidates)
            return best

        ctx.primary_call(selection)

        def checkpoint(destination):
            rank_states = ctx.gather(
                {
                    "rng": capture_rng(device),
                    "training_generator": generator.get_state().cpu(),
                    "memory": peak_memory(device),
                }
            )
            begin = time.perf_counter()

            def save():
                save_checkpoint(
                    destination,
                    {
                        "format": checkpoint_format,
                        "checkpoint_id": f"{run_id}:{update}",
                        "run_id": run_id,
                        "model": cpu_state(model.state_dict()),
                        "ema": {
                            name: cpu_state(state) for name, state in ema.states.items()
                        },
                        "optimizer": optimizer.state_dict(),
                        "config": deepcopy(config),
                        "update": update,
                        "sample_cursor": update * ctx.world,
                        "rank_states": rank_states,
                        "artifact_id": identity["artifact_id"],
                        "representation_id": identity["representation_id"],
                        "debug": debug,
                        **metadata,
                        "training_gpu_hours": ctx.world
                        * costs["train_update_seconds"]
                        / 3600,
                        "elapsed_seconds": elapsed_before
                        + time.perf_counter()
                        - started,
                        "costs": dict(costs),
                        **metadata,
                    },
                )

            ctx.primary_call(save)
            costs["checkpoint_io_seconds"] += time.perf_counter() - begin

        def validate_current():
            relative = f"checkpoints/update_{update:09d}.pt"
            if not ctx.broadcast((run / relative).exists() if ctx.primary else None):
                checkpoint(run / relative)
            # Recovery must point to the state being evaluated even if this evaluation fails.
            checkpoint(run / "recovery_latest.pt")
            indices = monitor_indices(predictor.data.splits["validation"])
            if ablation and config.get("preflight_max_graph"):
                indices = indices[:4]
            seeds = config["validation"]["sampling_seeds"]
            raw, mode = cpu_state(model.state_dict()), model.training
            rng, noise = capture_rng(device), generator.get_state().cpu()
            begin = time.perf_counter()
            try:
                for weights in config["validation"]["weights"]:
                    record_file = candidates_dir / f"{update:09d}_{weights}.json"
                    if ctx.broadcast(record_file.exists() if ctx.primary else None):
                        continue
                    model.load_state_dict(
                        raw if weights == "raw" else ema.states[weights], strict=True
                    )
                    model.eval()
                    output = run / "monitor" / f"update_{update:09d}_{weights}"
                    provenance = {
                        "checkpoint_id": f"{run_id}:{update}",
                        "artifact_id": identity["artifact_id"],
                        "weights": weights,
                        "update": update,
                        "examples_seen": update * ctx.world,
                        "training_seed": config["seed"],
                        "config": config,
                        "scope": "validation4_acceptance"
                        if ablation and config.get("preflight_max_graph")
                        else "validation24_monitor",
                        "debug": debug,
                        **metadata,
                        "training_gpu_hours": (
                            ctx.world * costs["train_update_seconds"] / 3600
                        ),
                    }
                    ctx.all_call(
                        lambda: evaluate_model(
                            predictor,
                            indices[ctx.rank :: ctx.world],
                            seeds,
                            output / f"rank_{ctx.rank:03d}",
                            provenance,
                            fail_on_runtime_error=True,
                            resume=True,
                        )
                    )

                    def finish_weight():
                        summary = merge_evaluation(
                            output, indices, seeds, provenance, ctx.world
                        )
                        row = {
                            **provenance,
                            "checkpoint": relative,
                            "summary": str((output / "summary.json").relative_to(run)),
                            "failed_clips": summary["failed_clips"],
                            "clip_count": summary["clip_count"],
                            "trajectory_count": summary["trajectory_count"],
                            "score": summary["selection_uv_relative_rmse"],
                            "complete_stage": False,
                        }
                        write_json(record_file, row)
                        selection()
                        print(
                            json.dumps(
                                clean_json(
                                    {
                                        "event": "validation",
                                        **{
                                            key: row[key]
                                            for key in (
                                                "update",
                                                "examples_seen",
                                                "weights",
                                                "score",
                                                "failed_clips",
                                                "clip_count",
                                            )
                                        },
                                    }
                                )
                            ),
                            flush=True,
                        )

                    ctx.primary_call(finish_weight)
            finally:
                model.load_state_dict(raw, strict=True)
                model.train(mode)
                restore_rng(rng, device)
                generator.set_state(noise)
                costs["validation_seconds"] += time.perf_counter() - begin
            checkpoint(run / "recovery_latest.pt")

        if not resume:
            checkpoint(run / "recovery_latest.pt")
        with h5py.File(artifacts / "train_latents.h5", "r") as cache:
            indices = [int(item) for item in cache["sim_indices"]]
            max_index = max(
                indices, key=lambda i: cache[f"sim_{i:05d}"]["latents"].shape[1]
            )
            if update and (
                update % config["validation"]["every_updates"] == 0
                or update == stage_end
            ):
                validate_current()
            while update < stage_end:
                synchronize(device)
                begin = time.perf_counter()
                cursor = update * ctx.world + ctx.rank
                epoch, offset = divmod(cursor, len(indices))
                permutation = torch.randperm(
                    len(indices),
                    generator=torch.Generator().manual_seed(
                        config["seed"] + epoch * 1000003
                    ),
                )
                index = (
                    max_index
                    if config.get("preflight_max_graph")
                    else indices[int(permutation[offset])]
                )
                sample = window_at(cache, index, device)
                lr = learning_rate(config, update + 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, training["precision"]):
                    loss = ddp(sample, generator)
                if not all(ctx.gather(bool(torch.isfinite(loss)))):
                    raise FloatingPointError("nonfinite loss on at least one rank")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training["gradient_clip"],
                    error_if_nonfinite=True,
                )
                attention_norm = float(
                    model.blocks[0].attention.in_proj_weight.grad.norm()
                )
                optimizer.step()
                ema.update(model)
                update += 1
                synchronize(device)
                costs["train_update_seconds"] += time.perf_counter() - begin
                if (
                    update == 1
                    or (ablation and config.get("preflight_max_graph"))
                    or update % training["log_every_updates"] == 0
                    or update == stage_end
                ):
                    loss_mean = float(ctx.mean(loss))
                    ranks = ctx.gather(
                        {
                            "rank": ctx.rank,
                            "trajectory_index": index,
                            **peak_memory(device),
                        }
                    )
                    row = {
                        "event": "update",
                        "update": update,
                        "examples_seen": update * ctx.world,
                        "epoch": update * ctx.world / len(indices),
                        "loss": loss_mean,
                        "gradient_norm_pre_clip": float(norm),
                        "learning_rate": lr,
                        "attention_gradient_norm": attention_norm,
                        "elapsed_seconds": elapsed_before
                        + time.perf_counter()
                        - started,
                        "rank_metrics": ranks,
                        **costs,
                        **metadata,
                        **peak_memory(device),
                    }

                    def log():
                        append_json(attempt_dir / "training.jsonl", row)
                        write_json(
                            run / "status.json",
                            {**row, "state": "running", "stage_end_updates": stage_end},
                        )
                        print(json.dumps(clean_json(row)), flush=True)

                    ctx.primary_call(log)
                if update == stage_end and initial_parameters is not None:

                    def check_preflight():
                        changes = {}
                        for name, state in {
                            "raw": model.state_dict(),
                            **ema.states,
                        }.items():
                            values = [
                                float(
                                    (value.detach().cpu() - initial_parameters[key])
                                    .abs()
                                    .max()
                                )
                                for key, value in state.items()
                                if value.is_floating_point()
                            ]
                            if (
                                not all(math.isfinite(value) for value in values)
                                or max(values) <= 0
                            ):
                                raise RuntimeError(
                                    f"preflight parameters did not update finitely: {name}"
                                )
                            changes[name] = max(values)
                        if attention_norm <= 0:
                            raise RuntimeError(
                                "preflight did not reach a nonzero attention gradient"
                            )
                        saved_rng, saved_mode = capture_rng(device), model.training
                        try:
                            model.eval()
                            prediction, _ = predictor.predict(
                                predictor.load_case(max_index), 17
                            )
                            if prediction.shape[0] != 65 or not math.isfinite(
                                float(prediction.sum())
                            ):
                                raise RuntimeError(
                                    "maximum Train graph joint64 prediction failed"
                                )
                        finally:
                            restore_rng(saved_rng, device)
                            model.train(saved_mode)
                        return {
                            "rank": ctx.rank,
                            "max_train_index": max_index,
                            "parameter_changes": changes,
                            "attention_gradient_norm": attention_norm,
                            **peak_memory(device),
                        }

                    preflight = ctx.all_call(check_preflight)
                    ctx.primary_call(
                        lambda: write_json(run / "preflight.json", {"ranks": preflight})
                    )
                if (
                    update % training["checkpoint_every_updates"] == 0
                    or update == stage_end
                ):
                    checkpoint(run / "checkpoints" / f"update_{update:09d}.pt")
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
        checkpoint(run / "recovery_latest.pt")

        final_rank_memory = ctx.gather({"rank": ctx.rank, **peak_memory(device)})

        def finish():
            best = selection(True)
            endpoint = selection(True, endpoint=True)
            elapsed = elapsed_before + time.perf_counter() - started
            result = {
                "state": "complete",
                "termination_reason": "allocated_budget",
                "update": update,
                "examples_seen": update * ctx.world,
                "stage_end_updates": stage_end,
                "elapsed_seconds": elapsed,
                "rank_metrics": final_rank_memory,
                "validation_gpu_hours": ctx.world * costs["validation_seconds"] / 3600,
                "training_windows_per_second": (
                    update * ctx.world / costs["train_update_seconds"]
                    if costs["train_update_seconds"] > 0
                    else None
                ),
                "selected_weights": best["weights"],
                "selected_update": best["update"],
                "endpoint_selected_weights": endpoint["weights"],
                "endpoint_score": endpoint["score"],
                "score": best["score"],
                "failed_clips": best["failed_clips"],
                **costs,
                **metadata,
                "schedule_complete": update == training["schedule_total_updates"],
                "schedule_total_updates": training["schedule_total_updates"],
                "schedule_total_windows": training["schedule_total_updates"]
                * ctx.world,
                "endpoint_scores": {
                    row["weights"]: row["score"]
                    for row in records()
                    if row["update"] == update
                },
                "training_gpu_hours": ctx.world * costs["train_update_seconds"] / 3600
                if device.type == "cuda"
                else None,
                "allocated_gpu_hours": ctx.world * elapsed / 3600
                if device.type == "cuda"
                else None,
                **peak_memory(device),
            }
            write_json(
                run / "checkpoint_inventory.json",
                {
                    "last_checkpoint": f"checkpoints/update_{update:09d}.pt",
                    "best_physical_checkpoint": best["checkpoint"],
                    "best_physical_weights": best["weights"],
                    "best_physical_score": best["score"],
                    "termination_reason": "allocated_budget",
                },
            )
            write_json(run / "status.json", result)
            (attempt_dir / "exit_code").write_text("0\n")
            return result

        return ctx.primary_call(finish)
    except BaseException as failure:
        if attempt_dir is not None:
            write_json(
                attempt_dir / f"rank_{ctx.rank:03d}_failure.json",
                {
                    "rank": ctx.rank,
                    "update": update,
                    "error": str(failure),
                    "traceback": traceback.format_exc(),
                },
            )
            if ctx.primary:
                write_json(
                    run / "status.json",
                    {
                        "state": "failed",
                        "update": update,
                        "stage_end_updates": stage_end,
                        "error": str(failure),
                    },
                )
                (attempt_dir / "exit_code").write_text("1\n")
        # Do not enter a collective checkpoint from an asymmetric failure path.
        raise
    finally:
        if lock is not None:
            lock.close()
        ctx.close()
