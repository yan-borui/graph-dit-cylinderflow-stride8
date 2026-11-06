"""Physical evaluation and matched timing for the deterministic graph Transformer."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from graph_dit.metrics import aggregate_rows, compute_metrics, node_area_weights
from graph_dit.performance import benchmark, configure
from graph_dit.runtime import load_checkpoint, monitor_indices, write_json

from .data import EvaluationData
from .model import JointGraphTransformer
from .source import DT


METHOD = "deterministic_graph_transformer"
CHECKPOINT_FORMAT = f"{METHOD}.training.v1"


@torch.inference_mode()
def predict(
    model: JointGraphTransformer, data: EvaluationData, sample: dict
) -> np.ndarray:
    """Produce the entire forecast from a CPU initial condition and static mesh."""
    z0, context, positions, hops, decoder = data.encode(sample)
    latents = model(z0, context, positions, hops)
    return data.decode(latents, sample, decoder)


def _forecast_case(
    model: JointGraphTransformer, data: EvaluationData, index: int
) -> tuple[dict, np.ndarray | None, dict]:
    sample = data.initial(index)
    if int(sample["trajectory_index"]) != index:
        raise ValueError(f"trajectory registry mismatch for case {index}")
    row = {"trajectory_index": index, "finite": False}
    try:
        field = predict(model, data, sample)
        if field.shape != (65, len(sample["points"]), 3):
            raise ValueError(f"unexpected prediction shape {field.shape}")
        if not np.array_equal(field[0], sample["initial"]):
            raise ValueError("prediction altered the input frame")
        if not np.isfinite(field).all():
            raise FloatingPointError("nonfinite decoded prediction")
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
        row["failure"] = f"{type(error).__name__}: {error}"
        return sample, None, row
    return sample, field, row


@torch.inference_mode()
def validation_rows(
    model: JointGraphTransformer,
    data: EvaluationData,
    indices: Iterable[int],
    device: torch.device,
) -> list[dict]:
    """Return per-trajectory scores for Validation selection on this rank."""
    result = []
    was_training = model.training
    model.eval()
    try:
        for index in indices:
            sample, field, row = _forecast_case(model, data, int(index))
            score = None
            if field is not None:
                truth = data.truth(index)
                weights = node_area_weights(sample["points"], sample["cells"])
                target = np.asarray(truth[1:, :, :2], dtype=np.float64)
                delta = field[1:, :, :2] - target
                numerator = np.einsum("tni,n,tni->", delta, weights, delta)
                denominator = np.einsum("tni,n,tni->", target, weights, target)
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError(f"undefined reference velocity norm at {index}")
                score = float(np.sqrt(numerator / denominator))
            eligible = score is not None and math.isfinite(score)
            result.append(
                {
                    "index": int(index),
                    "finite": eligible,
                    "uv_relative": score if eligible else None,
                    "failure": row.get("failure"),
                }
            )
    finally:
        model.train(was_training)
    return result


def _load_selected(
    run: Path, dataset: str, identity: dict, device: torch.device
) -> tuple[JointGraphTransformer, dict]:
    choice = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    name = choice.get("checkpoint")
    if not name:
        raise ValueError("Validation has not selected a checkpoint in this run")
    file = (run / name).resolve()
    if not file.is_relative_to(run.resolve()):
        raise ValueError("selected checkpoint is outside the requested run")
    state = load_checkpoint(file)
    if (
        state.get("format") != CHECKPOINT_FORMAT
        or state.get("method") != METHOD
        or state.get("config", {}).get("model") != JointGraphTransformer.architecture()
        or state.get("dataset") != dataset
        or state.get("artifact_id") != identity["artifact_id"]
        or state.get("representation_id") != identity["representation_id"]
        or state.get("run_id") != choice.get("run_id")
        or state["update"] != choice["update"]
    ):
        raise ValueError(
            "selected checkpoint differs from the graph Transformer architecture, dataset, cache or update"
        )
    label = choice["weights"]
    if label != "raw" and label not in state["ema"]:
        raise ValueError("selected weights are missing from the checkpoint")
    parameters = state["model"] if label == "raw" else state["ema"][label]
    model = JointGraphTransformer()
    model.load_state_dict(parameters, strict=True)
    return model.to(device).eval(), {
        "checkpoint": name,
        "update": state["update"],
        "weights": label,
        "training_run": state.get("run_id"),
        "artifact_id": state["artifact_id"],
        "representation_id": state["representation_id"],
        "model": METHOD,
        "architecture": state["config"]["model"],
        "future_input": "zero_queries",
        "modulation": "fixed_anchor_embedding",
        "target": "standardized_absolute_future_latents",
        "seed": state["config"]["seed"],
        "model_calls": 1,
        "ensemble_size": 1,
        "future_frames": 64,
    }


@torch.inference_mode()
def evaluate(
    model: JointGraphTransformer,
    data: EvaluationData,
    dataset: str,
    split: str,
    output: Path,
    provenance: dict,
) -> dict:
    """Write every official case, including failed predictions, and the full summary."""
    output.mkdir(parents=True, exist_ok=False)
    registry = tuple(int(i) for i in data.indices)
    if len(registry) != 100 or len(set(registry)) != 100:
        raise ValueError(f"expected 100 distinct official {split} trajectories")
    write_json(
        output / "identity.json",
        {
            **provenance,
            "dataset": dataset,
            "split": split,
            "indices": registry,
            "dt": DT[dataset],
            "source": data.source_identity,
            "stored_frames": [0, 64],
        },
    )
    rows = []
    for index in registry:
        sample, field, row = _forecast_case(model, data, index)
        if field is not None:
            truth = data.truth(index)
            row.update(
                compute_metrics(
                    field,
                    truth,
                    sample["points"],
                    sample["cells"],
                    sample["node_type"],
                    DT[dataset],
                )
            )
            if dataset == "cylinderflow":
                row.pop("airfoil_surface_uv_rmse", None)
                wall = np.asarray(sample["node_type"]).reshape(-1) == 6
                if wall.any():
                    delta = field[1:, wall, :2].astype(np.float64) - truth[1:, wall, :2]
                    row["wall_uv_rmse"] = float(np.sqrt(np.mean(delta**2)))
        write_json(output / f"case_{index:05d}.json", row)
        rows.append(row)
    summary = aggregate_rows(rows)
    summary.update(
        {
            **provenance,
            "dataset": dataset,
            "split": split,
            "trajectory_count": len(rows),
            "aggregation": "area-weighted 64-frame metrics per trajectory, then equal trajectory weight",
            "complete": summary["failure_count"] == 0,
        }
    )
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eval", "benchmark"))
    parser.add_argument("--dataset", required=True, choices=tuple(DT))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    for name in ("data-dir", "cache-dir", "result-root", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "benchmark" and args.split != "validation":
        parser.error("the published timing protocol uses fixed Validation24")
    device = torch.device("cuda:0")
    configure(device)
    started = time.perf_counter()
    data = EvaluationData(
        args.dataset, args.split, args.data_dir, args.cache_dir, device
    )
    model, provenance = _load_selected(
        args.result_root, args.dataset, data.identity, device
    )
    if args.mode == "benchmark":
        report = benchmark(
            method=METHOD,
            indices=monitor_indices(tuple(data.indices)),
            load_case=data.initial,
            predict=lambda sample: predict(model, data, sample),
            device=device,
            output_dir=args.output_dir,
            data_identity=data.source_identity,
            provenance=provenance,
            models=(model, data.codec.autoencoder),
            model_load_seconds=time.perf_counter() - started,
            physical_dt=DT[args.dataset],
        )
        complete = report["status"] == "complete"
    else:
        complete = evaluate(
            model,
            data,
            args.dataset,
            args.split,
            args.output_dir,
            provenance,
        )["complete"]
    if not complete:
        raise SystemExit("incomplete population; failures were retained in the report")


if __name__ == "__main__":
    main()
