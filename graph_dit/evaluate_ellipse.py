"""Four-GPU Ellipse forecasts from selected or explicitly identified weights."""

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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("raw", "ema_0.999", "ema_0.9999"))
    parser.add_argument("--sampling-steps", type=int, default=6)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--skip-members", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("sampling labels must be unique nonnegative integers")
    context = Context(4, "cuda")
    try:
        if args.checkpoint is None:
            selected = json.loads((args.run / "selection.json").read_text())
            if not selected.get("complete_stage"):
                raise ValueError(
                    "complete training before using its selected checkpoint"
                )
            checkpoint_file = args.run / selected["checkpoint"]
            weights = args.weights or selected["weights"]
        else:
            if args.weights is None:
                parser.error("explicit checkpoint evaluation requires --weights")
            checkpoint_file = args.checkpoint
            weights = args.weights
        model, checkpoint = load_selected(checkpoint_file, args.artifacts, weights)
        deterministic = checkpoint["config"].get("model_type") == "deterministic"
        members = (
            args.ensemble_size
            if args.ensemble_size is not None
            else (1 if deterministic else 8)
        )
        if members < 1 or (deterministic and (members != 1 or args.seeds != [0])):
            raise ValueError("deterministic evaluation uses one forward and label 0")
        precision = checkpoint["config"]["training"]["precision"]
        configure_runtime(context.device, precision, allow_distributed=True)
        model.to(context.device)
        predictor = Predictor(
            model,
            args.artifacts,
            args.data_dir,
            context.device,
            precision,
            sampling_steps=args.sampling_steps,
        )
        indices = predictor.data.splits[args.split]
        provenance = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "weights": weights,
            "checkpoint_file": str(checkpoint_file.resolve()),
            "selection": "explicit" if args.checkpoint else "validation_selected",
            "model_type": checkpoint["config"].get("model_type", "diffusion"),
            "training_seed": checkpoint["config"]["seed"],
            "update": checkpoint["update"],
            "config": checkpoint["config"],
            "artifact_id": checkpoint["artifact_id"],
            "scope": args.split,
            "test_accessed": args.split == "test",
            "world_size": 4,
            "data_identity": predictor.data.identity(),
            "sampling_steps": 1 if deterministic else args.sampling_steps,
            "ensemble_size": members,
            "aggregation": "physical_uvp_mean",
            "member_label_rule": "ensemble_label * member_label_stride + member_index",
            "member_label_stride": 100000,
        }
        context.primary_call(
            lambda: args.output_dir.mkdir(parents=True, exist_ok=False)
        )
        context.all_call(
            lambda: evaluate_model(
                predictor,
                indices[context.rank :: context.world],
                args.seeds,
                args.output_dir / f"rank_{context.rank:03d}",
                provenance,
                resume=True,
                ensemble_size=members,
                retain_members=not args.skip_members,
                member_label_stride=100000,
            )
        )
        summary = context.primary_call(
            lambda: merge_evaluation(
                args.output_dir, indices, args.seeds, provenance, context.world
            )
        )
        if summary["failed_clips"]:
            raise SystemExit(1)
    finally:
        context.close()


if __name__ == "__main__":
    main()
