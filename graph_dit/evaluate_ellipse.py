"""Four-GPU Ellipse Validation/Test forecasting with fixed S6/K8 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ddp_train import merge_evaluation
from .distributed import Context
from .evaluate import Predictor, evaluate_model, load_selected
from .train import configure_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "artifacts", "data-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    context = Context(4, "cuda")
    try:
        selected = json.loads((args.run / "selection.json").read_text())
        if not selected.get("complete_stage"):
            raise ValueError(
                "complete training before evaluating its selected checkpoint"
            )
        model, checkpoint = load_selected(
            args.run / selected["checkpoint"], args.artifacts, selected["weights"]
        )
        precision = checkpoint["config"]["training"]["precision"]
        configure_runtime(context.device, precision)
        model.to(context.device)
        predictor = Predictor(
            model,
            args.artifacts,
            args.data_dir,
            context.device,
            precision,
            sampling_steps=6,
        )
        indices = predictor.data.splits[args.split]
        provenance = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "weights": selected["weights"],
            "update": checkpoint["update"],
            "config": checkpoint["config"],
            "artifact_id": checkpoint["artifact_id"],
            "scope": args.split,
            "test_accessed": args.split == "test",
            "world_size": 4,
            "data_identity": predictor.data.identity(),
            "sampling_steps": 6,
            "ensemble_size": 8,
            "aggregation": "physical_uvp_mean",
            "member_label_rule": "ensemble_label * ensemble_size + member_index",
        }
        context.primary_call(
            lambda: args.output_dir.mkdir(parents=True, exist_ok=False)
        )
        context.all_call(
            lambda: evaluate_model(
                predictor,
                indices[context.rank :: context.world],
                [0],
                args.output_dir / f"rank_{context.rank:03d}",
                provenance,
                resume=True,
                ensemble_size=8,
            )
        )
        summary = context.primary_call(
            lambda: merge_evaluation(
                args.output_dir, indices, [0], provenance, context.world
            )
        )
        if summary["failed_clips"]:
            raise SystemExit(1)
    finally:
        context.close()


if __name__ == "__main__":
    main()
