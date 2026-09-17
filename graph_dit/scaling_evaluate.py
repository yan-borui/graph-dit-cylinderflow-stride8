"""Evaluate a fixed scaling endpoint on all 32 ranks, using Validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .ddp_train import merge_evaluation
from .distributed import Context, describe_device
from .evaluate import Predictor, evaluate_model, load_selected
from .runtime import acquire_run_lock, write_json
from .train import configure_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", choices=("endpoint", "best"), default="endpoint")
    args = parser.parse_args()
    config = load_config(args.run / "config.json")
    if "scaling" not in config or config.get("preflight_max_graph"):
        raise ValueError("full scaling evaluation requires a formal scaling run")
    ctx = Context(32, "cuda:0")
    lock = None
    try:
        configure_runtime(ctx.device, "fp32", allow_distributed=True)
        devices = ctx.all_call(lambda: describe_device(config, ctx.device))
        selected_file = (
            "selection_endpoint.json"
            if args.selection == "endpoint"
            else "selection.json"
        )
        selected = json.loads((args.run / selected_file).read_text(encoding="utf-8"))
        status = json.loads((args.run / "status.json").read_text(encoding="utf-8"))
        if (
            status.get("state") != "complete"
            or status.get("update") != 125000
            or not selected.get("complete_stage")
            or selected.get("weights") is None
            or selected.get("failed_clips") != 0
        ):
            raise ValueError(
                "complete all 125k updates and a valid Validation24 selection first"
            )
        if args.selection == "endpoint" and selected["update"] != 125000:
            raise ValueError("the main scaling comparison uses the fixed 125k endpoint")
        model, saved = load_selected(
            args.run / selected["checkpoint"], args.artifacts, selected["weights"]
        )
        if saved["config"] != config or saved["update"] != selected["update"]:
            raise ValueError("checkpoint and selected scaling identity differ")
        provenance = {
            "checkpoint_id": saved["checkpoint_id"],
            "artifact_id": saved["artifact_id"],
            "weights": selected["weights"],
            "update": saved["update"],
            "training_seed": config["seed"],
            "config": config,
            "scope": "validation100_scaling",
            "selection": args.selection,
        }
        identities = ctx.gather(provenance)
        if any(item != identities[0] for item in identities):
            raise ValueError(
                "evaluation ranks selected different checkpoints or data identities"
            )
        del saved
        model.to(ctx.device)
        predictor = Predictor(model, args.artifacts, args.data_dir, ctx.device, "fp32")
        output = args.output_dir / args.selection

        def prepare() -> None:
            nonlocal lock
            output.mkdir(parents=True, exist_ok=True)
            lock = acquire_run_lock(output)
            receipt = {**provenance, "world_size": ctx.world}
            receipt_file = output / "evaluation.json"
            if (
                receipt_file.exists()
                and json.loads(receipt_file.read_text()) != receipt
            ):
                raise ValueError(
                    "evaluation directory belongs to another selected checkpoint"
                )
            write_json(receipt_file, receipt)
            write_json(output / "devices.json", {"devices": devices})

        ctx.primary_call(prepare)
        indices = predictor.data.splits["validation"]
        seeds = config["validation"]["sampling_seeds"]
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
        ctx.primary_call(
            lambda: merge_evaluation(output, indices, seeds, provenance, ctx.world)
        )
        ctx.primary_call(
            lambda: write_json(output / "status.json", {"state": "complete"})
        )
    except BaseException as failure:
        if lock is not None:
            write_json(
                args.output_dir / args.selection / "status.json",
                {
                    "state": "failed",
                    "error": str(failure),
                },
            )
        raise
    finally:
        if lock is not None:
            lock.close()
        ctx.close()


if __name__ == "__main__":
    main()
