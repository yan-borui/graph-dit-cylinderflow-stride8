"""Bounded real-data acceptance through the production save/resume/evaluate path."""

import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch

from .config import load_config
from .data import Dataset
from .representation import load_artifacts, paths
from .runtime import load_checkpoint, read_jsonl, write_json
from .train import train_run


def compare_reference(output: Path, reference: Path, device: str) -> dict:
    """Compare artifacts from two executions of this same production acceptance."""
    current_summary = json.loads((output / "preflight.json").read_text())
    baseline_summary = json.loads((reference / "preflight.json").read_text())
    if (
        baseline_summary["state"] != "passed"
        or current_summary["config"] != baseline_summary["config"]
        or current_summary["artifact_id"] != baseline_summary["artifact_id"]
        or current_summary["gpu"] != baseline_summary["gpu"]
        or current_summary["torch"] != baseline_summary["torch"]
    ):
        raise ValueError(
            "reference must use the same successful recipe, data and GPU runtime"
        )
    current = load_checkpoint(output / "training/recovery_latest.pt")
    baseline = load_checkpoint(reference / "training/recovery_latest.pt")
    drift = {}
    for name, state in {"raw": current["model"], **current["ema"]}.items():
        prior = baseline["model"] if name == "raw" else baseline["ema"][name]
        if set(state) != set(prior):
            raise ValueError("optimization changed checkpoint tensor names")
        maximum, squared_error, squared_reference = 0.0, 0.0, 0.0
        for key in current["parameter_names"]:
            if (
                state[key].shape != prior[key].shape
                or state[key].dtype != prior[key].dtype
            ):
                raise ValueError(f"optimization changed tensor schema: {key}")
            left = state[key].to(device=device, dtype=torch.float64)
            right = prior[key].to(device=device, dtype=torch.float64)
            difference = left - right
            maximum = max(maximum, float(difference.abs().max()))
            squared_error += float(difference.square().sum())
            squared_reference += float(right.square().sum())
        drift[name] = {
            "max_absolute_difference": maximum,
            "relative_l2_difference": (squared_error / max(squared_reference, 1e-300))
            ** 0.5,
        }
    same_generator = torch.equal(
        current["training_generator"], baseline["training_generator"]
    )
    if not same_generator:
        raise ValueError("optimization changed the training random-number cursor")
    del current, baseline

    def steady_seconds(summary):
        previous, timings = 0.0, []
        for row in summary["updates"]:
            elapsed = row["train_update_seconds"] - previous
            previous = row["train_update_seconds"]
            if row["update"] not in (1, 5):
                timings.append(elapsed)
        return float(np.mean(timings))

    before = steady_seconds(baseline_summary)
    after = steady_seconds(current_summary)
    loss_difference = max(
        abs(left["loss"] - right["loss"])
        for left, right in zip(
            current_summary["updates"], baseline_summary["updates"], strict=True
        )
    )
    current_candidates = read_jsonl(output / "training/candidates.jsonl")
    baseline_candidates = {
        (row["update"], row["weights"]): row
        for row in read_jsonl(reference / "training/candidates.jsonl")
    }
    score_difference = max(
        abs(
            row["score"] - baseline_candidates[(row["update"], row["weights"])]["score"]
        )
        for row in current_candidates
    )
    result = {
        "reference": str(reference),
        "comparison_device": device,
        "same_training_generator": same_generator,
        "final_weight_differences": drift,
        "max_training_loss_absolute_difference": loss_difference,
        "max_validation_score_absolute_difference": score_difference,
        "baseline_steady_seconds_per_update": before,
        "optimized_steady_seconds_per_update": after,
        "steady_speedup": before / after,
        "claim": "same eight-update acceptance and artifact comparison, not full-training quality equivalence",
    }
    write_json(output / "comparison.json", result)
    return result


def accept(
    config: dict,
    artifacts: Path,
    data_dir: Path,
    output: Path,
    device: str,
    reference: Path | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        identity = load_artifacts(artifacts, config=config)
        data = Dataset(*paths(data_dir))
        with h5py.File(artifacts / "train_latents.h5", "r") as cache:
            index = max(
                (int(value) for value in cache["sim_indices"]),
                key=lambda i: cache[f"sim_{i:05d}/latents"].shape[1],
            )
            nodes = int(cache[f"sim_{index:05d}/latents"].shape[1])
        contract = {
            "format": "graph_dit.real_data_acceptance.v1",
            "train_index": index,
            "latent_nodes": nodes,
            "validation_indices": [data.splits["validation"][0]],
            "updates": 8,
            "resume_update": 4,
            "pause_at_update": 4,
        }
        write_json(output / "acceptance.json", contract)
        run = output / "training"
        first = train_run(
            config, artifacts, data_dir, run, 4, device, acceptance=contract
        )
        if first["state"] != "paused" or first["update"] != 4:
            raise RuntimeError(
                "acceptance did not save at the requested pause boundary"
            )
        (run / "PAUSE").unlink()
        gc.collect()
        torch.cuda.empty_cache()
        train_run(
            config,
            artifacts,
            data_dir,
            run,
            8,
            device,
            resume=True,
            acceptance=contract,
        )
        records = [
            row
            for attempt in sorted(run.glob("attempt_*"))
            for row in read_jsonl(attempt / "training.jsonl")
        ]
        if [row["update"] for row in records] != list(range(1, 9)):
            raise RuntimeError(
                "acceptance did not resume from update 4 through update 8"
            )
        if not any(row["attention_gradient_norm"] > 0 for row in records[2:]):
            raise RuntimeError(
                "attention gradient stayed zero after AdaLN-Zero initialization"
            )
        initial = load_checkpoint(run / "checkpoints" / "update_000000000.pt")
        final = load_checkpoint(run / "recovery_latest.pt")
        if final["update"] != 8:
            raise RuntimeError("final recovery checkpoint has the wrong update")
        changes = {}
        for name, state in {"raw": final["model"], **final["ema"]}.items():
            differences = [
                float((state[key] - initial["model"][key]).abs().max())
                for key in final["parameter_names"]
            ]
            dtypes = sorted({str(state[key].dtype) for key in final["parameter_names"]})
            if not all(np.isfinite(differences)) or max(differences) <= 0:
                raise RuntimeError(f"nonfinite or unchanged {name} parameters")
            if config["training"]["precision"] == "fp32" and dtypes != [
                "torch.float32"
            ]:
                raise RuntimeError(f"unexpected {name} precision: {dtypes}")
            changes[name] = {"max_absolute_change": max(differences), "dtypes": dtypes}
        del initial, final
        candidates = read_jsonl(run / "candidates.jsonl")
        expected = {
            (update, weights)
            for update in (4, 8)
            for weights in config["validation"]["weights"]
        }
        if {(row["update"], row["weights"]) for row in candidates} != expected:
            raise RuntimeError("missing production evaluation candidates")
        for row in candidates:
            if (
                row["failed_clips"]
                or row["trajectory_count"] != 1
                or row["clip_count"] != len(config["validation"]["sampling_seeds"])
                or row["score"] is None
                or not np.isfinite(row["score"])
            ):
                raise RuntimeError("incomplete real Validation acceptance")
        launch = json.loads((run / "attempt_001" / "launch.json").read_text())
        write_json(
            output / "preflight.json",
            {
                "state": "passed",
                "contract": contract,
                "config": config,
                "artifact_id": identity["artifact_id"],
                "representation_id": identity["representation_id"],
                "updates": records,
                "parameter_changes": changes,
                "restored": json.loads(
                    (run / "attempt_002" / "restored.json").read_text()
                ),
                "gpu": launch["gpu"],
                "torch": launch["torch"],
                "tf32_matmul": launch["tf32_matmul"],
                "tf32_cudnn": launch["tf32_cudnn"],
                "elapsed_seconds": time.perf_counter() - started,
                "claim": "bounded save/resume/physical-evaluation acceptance; not convergence evidence",
            },
        )
        if reference is not None:
            gc.collect()
            torch.cuda.empty_cache()
            compare_reference(output, reference, device)
    except BaseException as failure:
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            output / "preflight.json",
            {
                "state": "failed",
                "error": f"{type(failure).__name__}: {failure}",
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        (output / "exit_code").write_text("1\n")
        raise
    (output / "exit_code").write_text("0\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("distributed", {}).get("world_size", 1) > 1:
        if args.updates < 4:
            raise ValueError("at least four updates are required")
        config["preflight_max_graph"] = True
        train_run(
            config,
            args.artifacts,
            args.data_dir,
            args.output_dir,
            args.updates,
            args.device,
        )
        return
    if args.updates != 8 or torch.device(args.device).type != "cuda":
        raise ValueError(
            "single-GPU acceptance requires eight updates on the target CUDA GPU"
        )
    accept(
        config,
        args.artifacts,
        args.data_dir,
        args.output_dir,
        args.device,
        args.reference,
    )


if __name__ == "__main__":
    main()
