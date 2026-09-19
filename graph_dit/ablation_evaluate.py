"""Locked endpoint/best Validation100 and matched four-rank inference timing."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
import uuid

import torch

from .ablation_contract import UPDATES, WEIGHTS, is_ablation, variant_name
from .ablation_runtime import bind_environment
from .config import load_config
from .ddp_train import merge_evaluation
from .distributed import Context, describe_device
from .evaluate import Predictor, evaluate_model, load_selected
from .performance import benchmark, distribution
from .runtime import acquire_run_lock, monitor_indices, read_jsonl, write_json
from .train import configure_runtime, freeze_source


def locked_selections(run: Path, config: dict) -> dict:
    """Reconstruct both choices from complete monitor evidence before evaluation."""
    status = json.loads((run / "status.json").read_text())
    if status.get("state") != "complete" or status.get("update") != UPDATES:
        raise ValueError("complete the declared 62,500 updates before final evaluation")
    rows = read_jsonl(run / "candidates.jsonl")
    expected = {
        (update, weight)
        for update in range(12500, UPDATES + 1, 12500)
        for weight in WEIGHTS
    }
    if (
        len(rows) != len(expected)
        or {(row["update"], row["weights"]) for row in rows} != expected
    ):
        raise ValueError("the five monitor rounds must contain every raw/EMA candidate")
    if any(row.get("config") != config for row in rows):
        raise ValueError("monitor evidence belongs to a different configuration")
    valid = [
        row
        for row in rows
        if row["failed_clips"] == 0
        and row["trajectory_count"] == 24
        and row["clip_count"] == 72
        and row["score"] is not None
        and math.isfinite(row["score"])
    ]
    selections = {}
    for name in ("endpoint", "best"):
        pool = [row for row in valid if name == "best" or row["update"] == UPDATES]
        if not pool:
            raise ValueError(f"no complete finite {name} candidate")
        expected_row = min(
            pool,
            key=lambda row: (
                row["score"],
                row["update"],
                WEIGHTS.index(row["weights"]),
            ),
        )
        file_name = (
            "selection_endpoint.json" if name == "endpoint" else "selection.json"
        )
        selected = json.loads((run / file_name).read_text())
        keys = (
            "checkpoint_id",
            "artifact_id",
            "checkpoint",
            "weights",
            "update",
            "score",
            "summary",
        )
        if not selected.get("complete_stage") or any(
            selected.get(key) != expected_row[key] for key in keys
        ):
            raise ValueError(
                f"{name} selection disagrees with the saved monitor evidence"
            )
        selections[name] = selected
    return selections


def benchmark_four_ranks(
    model, predictor, ctx, output: Path, provenance: dict, load_seconds: float
) -> None:
    complete_file = output / "performance/summary.json"

    def already_done():
        if not complete_file.exists():
            return False
        result = json.loads(complete_file.read_text())
        if result.get("provenance") != provenance:
            raise ValueError(
                "existing performance result has a different checkpoint identity"
            )
        return result.get("state") == "complete"

    if ctx.primary_call(already_done):
        return
    attempt_id = ctx.broadcast(uuid.uuid4().hex if ctx.primary else None)
    destination = output / "performance" / f"attempt_{attempt_id}"
    indices = monitor_indices(predictor.data.splits["validation"])
    local_indices = indices[ctx.rank :: ctx.world]

    def measure():
        torch.set_num_threads(2)
        return benchmark(
            method=f"graph_dit_{variant_name(provenance['config']).lower()}",
            indices=local_indices,
            registry_indices=local_indices,
            load_case=predictor.load_case,
            predict=lambda sample: predictor.predict(sample, diagnostics=False)[0],
            device=ctx.device,
            output_dir=destination / f"rank_{ctx.rank:03d}",
            data_identity=predictor.data.identity(),
            provenance=provenance,
            models=[model, predictor.codec.autoencoder],
            model_load_seconds=load_seconds,
        )

    results = ctx.all_call(measure)

    def merge():
        trajectories = sorted(
            (row for result in results for row in result["trajectory_metrics"]),
            key=lambda row: row["trajectory_index"],
        )
        if [row["trajectory_index"] for row in trajectories] != list(indices):
            raise ValueError("benchmark shards do not cover Validation24 exactly once")
        complete = all(row["complete"] for row in trajectories)
        write_json(
            complete_file,
            {
                "state": "complete" if complete else "completed_with_failures",
                "provenance": provenance,
                "world_size": 4,
                "latency_seconds": distribution(
                    [
                        row["latency_seconds"]["mean"]
                        for row in trajectories
                        if row["complete"]
                    ]
                ),
                "failed_measurements": sum(
                    result["failed_measurements"] for result in results
                ),
                "failed_warmups": sum(result["failed_warmups"] for result in results),
                "measured_samples": sum(
                    result["measured_samples"] for result in results
                ),
                "warmup_samples": sum(result["warmup_samples"] for result in results),
                "peak_allocated_bytes": max(
                    result["cuda_peak_allocated_bytes"] for result in results
                ),
                "peak_reserved_bytes": max(
                    result["cuda_peak_reserved_bytes"] for result in results
                ),
                "measured_inference_gpu_hours": sum(
                    result["measured_inference_gpu_hours"] for result in results
                ),
                "rank_summaries": [
                    str(
                        (destination / f"rank_{rank:03d}/summary.json").relative_to(
                            output
                        )
                    )
                    for rank in range(4)
                ],
                "trajectory_metrics": trajectories,
                "timing_scope": "CPU initial state -> encode -> joint64 DDIM20 -> decode/writeback -> CPU UVP; excludes IO, quality metrics and warmup",
                "aggregation": "three repeats per trajectory, then equal trajectory weight; four concurrent independent ranks",
            },
        )

    ctx.primary_call(merge)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "artifacts", "data-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.run / "config.json")
    if not is_ablation(config) or config.get("preflight_max_graph"):
        raise ValueError("final evaluation requires a formal attention ablation run")
    ctx = Context(4, "cuda:0")
    lock = None
    attempt_id = None
    started = time.perf_counter()
    try:
        configure_runtime(ctx.device, "fp32", allow_distributed=True)
        devices = ctx.all_call(lambda: describe_device(config, ctx.device))
        environment = bind_environment(config, devices, args.run, ctx)
        selections = ctx.primary_call(lambda: locked_selections(args.run, config))

        def prepare():
            nonlocal lock
            freeze_source(args.run, resume=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            lock = acquire_run_lock(args.output_dir)
            receipt = {
                "config": config,
                "selections": selections,
                "environment": environment,
            }
            destination = args.output_dir / "locked_selections.json"
            if destination.exists() and json.loads(destination.read_text()) != receipt:
                raise ValueError(
                    "final evaluation selection is already locked differently"
                )
            write_json(destination, receipt)
            write_json(args.output_dir / "devices.json", {"devices": devices})

        ctx.primary_call(prepare)
        attempt_id = ctx.broadcast(uuid.uuid4().hex if ctx.primary else None)
        for name in ("endpoint", "best"):
            selected = selections[name]
            if name == "best" and all(
                selected[key] == selections["endpoint"][key]
                for key in ("checkpoint_id", "weights")
            ):
                ctx.primary_call(
                    lambda: write_json(
                        args.output_dir / "best/reference.json",
                        {
                            "target": "../endpoint",
                            "checkpoint_id": selected["checkpoint_id"],
                            "weights": selected["weights"],
                        },
                    )
                )
                continue
            begin = time.perf_counter()
            model, saved = load_selected(
                args.run / selected["checkpoint"], args.artifacts, selected["weights"]
            )
            if (
                saved["config"] != config
                or saved["checkpoint_id"] != selected["checkpoint_id"]
                or saved.get("environment") != environment
            ):
                raise ValueError("selected checkpoint config/runtime/identity differs")
            provenance = {
                "checkpoint_id": saved["checkpoint_id"],
                "artifact_id": saved["artifact_id"],
                "weights": selected["weights"],
                "update": selected["update"],
                "config": config,
                "training_seed": config["seed"],
                "scope": "validation100_attention_ablation",
                "selection": name,
            }
            del saved
            model.to(ctx.device)
            predictor = Predictor(
                model, args.artifacts, args.data_dir, ctx.device, "fp32"
            )
            load_seconds = time.perf_counter() - begin
            identities = ctx.gather(provenance)
            if any(item != identities[0] for item in identities):
                raise ValueError("evaluation ranks selected different inputs")
            output = args.output_dir / name
            indices, seeds = (
                predictor.data.splits["validation"],
                config["validation"]["sampling_seeds"],
            )
            ctx.all_call(
                lambda predictor=predictor: evaluate_model(
                    predictor,
                    indices[ctx.rank :: ctx.world],
                    seeds,
                    output / f"rank_{ctx.rank:03d}",
                    provenance,
                    fail_on_runtime_error=True,
                    resume=True,
                )
            )
            ctx.primary_call(
                lambda: merge_evaluation(output, indices, seeds, provenance, ctx.world)
            )
            if name == "endpoint":
                benchmark_four_ranks(
                    model, predictor, ctx, output, provenance, load_seconds
                )
            del predictor, model
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        ctx.primary_call(
            lambda: write_json(
                args.output_dir / "attempts" / f"{attempt_id}.json",
                {
                    "state": "complete",
                    "elapsed_seconds": elapsed,
                    "allocated_gpu_hours": 4 * elapsed / 3600,
                },
            )
        )
        ctx.primary_call(
            lambda: write_json(
                args.output_dir / "status.json",
                {
                    "state": "complete",
                    "world_size": 4,
                    "elapsed_seconds_this_attempt": elapsed,
                    "allocated_gpu_hours_this_attempt": 4 * elapsed / 3600,
                },
            )
        )
    except BaseException as error:
        if lock is not None:
            if attempt_id is not None:
                elapsed = time.perf_counter() - started
                write_json(
                    args.output_dir / "attempts" / f"{attempt_id}.json",
                    {
                        "state": "failed",
                        "error": str(error),
                        "elapsed_seconds": elapsed,
                        "allocated_gpu_hours": 4 * elapsed / 3600,
                    },
                )
            write_json(
                args.output_dir / "status.json",
                {"state": "failed", "error": str(error)},
            )
        raise
    finally:
        if lock is not None:
            lock.close()
        ctx.close()


if __name__ == "__main__":
    main()
