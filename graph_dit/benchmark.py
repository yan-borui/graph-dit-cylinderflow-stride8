"""Run the same end-to-end FP32 cost protocol as the four published baselines."""

import argparse
import json
from pathlib import Path
import time

import torch
import numpy as np
import os

from .evaluate import Predictor, load_selected
from .performance import benchmark, configure, seed_draw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sampling-steps", type=int, default=6)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("raw", "ema_0.999", "ema_0.9999"))
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not args.device.startswith(
        "cuda"
    ):
        parser.error("benchmark uses a standalone production CUDA device")
    if args.checkpoint:
        if args.weights is None:
            parser.error("explicit checkpoint requires --weights")
        selected = {"checkpoint": str(args.checkpoint), "weights": args.weights}
    else:
        selected = json.loads((args.run / "selection.json").read_text())
        if not selected.get("complete_stage"):
            raise ValueError("benchmark requires a selected, completed stage")
        if args.weights:
            selected["weights"] = args.weights
    device = torch.device(args.device)
    configure(device)
    begin = time.perf_counter()
    model, checkpoint = load_selected(
        args.run / selected["checkpoint"], args.artifacts, selected["weights"]
    )
    model.to(device)
    deterministic = checkpoint["config"].get("model_type") == "deterministic"
    members = (
        args.ensemble_size
        if args.ensemble_size is not None
        else (1 if deterministic else 8)
    )
    if members < 1 or (deterministic and members != 1):
        raise ValueError("invalid ensemble size")
    predictor = Predictor(
        model,
        args.artifacts,
        args.data_dir,
        device,
        "fp32",
        sampling_steps=args.sampling_steps,
    )
    load_seconds = time.perf_counter() - begin
    result = benchmark(
        method=checkpoint["config"].get("experiment", "h1"),
        indices=tuple(
            predictor.data.splits["validation"][i]
            for i in np.rint(np.linspace(0, 99, 24)).astype(int)
        ),
        load_case=predictor.load_case,
        predict=lambda sample: predictor.predict_ensemble(
            sample,
            [
                seed_draw(sample["trajectory_index"], member)
                for member in range(members)
            ],
        )[0],
        device=device,
        output_dir=args.output_dir,
        data_identity=predictor.data.identity(),
        provenance={
            "checkpoint_id": checkpoint["checkpoint_id"],
            "weights": selected["weights"],
            "artifact_id": checkpoint["artifact_id"],
            "training_seed": checkpoint["config"]["seed"],
            "update": checkpoint["update"],
            "training_precision": checkpoint["config"]["training"]["precision"],
            "sampling_steps": 1 if deterministic else args.sampling_steps,
            "ensemble_size": members,
            "aggregation": "physical_uvp_mean",
            "member_label_rule": "ensemble_label * ensemble_size + member_index",
        },
        models=[model, predictor.codec.autoencoder],
        model_load_seconds=load_seconds,
        validation_count=24,
    )
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
