"""Measure S/K configurations and evaluate their physical ensemble means."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import DATA_REPOSITORY, DATA_REVISION
from .evaluate import Predictor, evaluate_model, load_selected
from .performance import benchmark, configure, runtime_identity, sync, write_json
from .runtime import monitor_indices


class EnsemblePredictor:
    """Serial complete predictions; nested label prefixes, physical-space means."""

    def __init__(self, predictor: Predictor, count: int):
        self.base = predictor
        self.count = count
        self.data = predictor.data
        self.device = predictor.device
        self.load_case = predictor.load_case

    def predict(
        self,
        sample: dict,
        sampling_seed: int | None = None,
        *,
        diagnostics: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # K is independent of timing repeats. Each repetition measures the same
        # ensemble, using explicit generators rather than global RNG reseeding.
        total, raw_total = None, None
        for label in range(self.count):
            seed = ((label + 1) * 1_000_003 + sample["trajectory_index"] * 9_176) % (
                2**31 - 1
            )
            prediction, raw = self.base.predict(sample, seed, diagnostics=diagnostics)
            if total is None:
                total = prediction.copy()
                raw_total = raw.copy() if diagnostics else None
            else:
                total += prediction
                if diagnostics:
                    raw_total += raw
        total /= self.count
        total[0] = sample["initial"]
        boundary = np.isin(np.asarray(sample["node_type"]).reshape(-1), (4, 6))
        total[1:, boundary, :2] = sample["initial"][None, boundary, :2]
        if diagnostics:
            raw_total /= self.count
            raw_total[0] = sample["initial"]
        return total, raw_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights")
    parser.add_argument(
        "--run", type=Path, help="read selection.json, including paused runs"
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign-id", required=True, help="same fresh ID in all five repos"
    )
    parser.add_argument("--sampling-steps", nargs="+", type=int, default=[6, 20])
    parser.add_argument(
        "--ensemble-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--mode", choices=["all", "timing", "quality"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.run and args.checkpoint is None:
        selected = json.loads((args.run / "selection.json").read_text())
        args.checkpoint = args.run / selected["checkpoint"]
        args.weights = args.weights or selected["weights"]
    if args.checkpoint is None or args.weights is None:
        parser.error("provide --checkpoint and --weights, or --run")
    for values in (args.sampling_steps, args.ensemble_sizes):
        if len(set(values)) != len(values) or any(value < 1 for value in values):
            parser.error("grid values must be distinct positive integers")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("formal handoff requires one CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configure(device, args.threads)
    started = time.perf_counter()
    model, checkpoint = load_selected(
        args.checkpoint, args.artifacts, args.weights, inference_only=True
    )
    model.to(device).eval().float()
    predictor = Predictor(
        model, args.artifacts, args.data_dir, device, inference_only=True
    )
    sync(device)
    load_seconds = time.perf_counter() - started
    environment = runtime_identity(device)
    data_identity = {
        "repository": DATA_REPOSITORY,
        "revision": DATA_REVISION,
        "split": "validation",
        "phase_offset": 0,
        "raw_frame_indices": list(range(0, 513, 8)),
    }
    provenance = {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "weights": args.weights,
        "artifact_id": checkpoint["artifact_id"],
        "representation_id": predictor.identity["representation_id"],
        "training_seed": checkpoint["config"]["seed"],
        "update": checkpoint["update"],
        "campaign_id": args.campaign_id,
        "sampler": "ddim_eta0_uniform_rounded_timesteps",
        "ensemble_semantics": "mean_of_physical_uvp_predictions",
        "training_precision": checkpoint["config"]["training"]["precision"],
        "seed_rule": "((label+1)*1000003 + trajectory_index*9176) mod (2**31-1)",
    }
    del checkpoint
    failed = False
    for steps in args.sampling_steps:
        if not 2 <= steps <= model.diffusion_steps:
            raise ValueError("sampling steps outside model diffusion schedule")
        predictor.sampling_steps = steps
        for count in args.ensemble_sizes:
            name = f"s{steps}_k{count}"
            output = args.output_dir / name
            output.mkdir()
            point = {
                **provenance,
                "sampling_steps": steps,
                "ensemble_size": count,
                "sampling_labels": list(range(count)),
            }
            wrapper = EnsemblePredictor(predictor, count)
            try:
                if args.mode in ("all", "timing"):
                    report = benchmark(
                        method="graph_dit_h1",
                        indices=monitor_indices(predictor.data.splits["validation"]),
                        load_case=wrapper.load_case,
                        predict=lambda sample: wrapper.predict(
                            sample, diagnostics=False
                        )[0],
                        device=device,
                        output_dir=output / "timing",
                        data_identity=data_identity,
                        provenance=point,
                        models=[model, predictor.codec.autoencoder],
                        model_load_seconds=load_seconds,
                    )
                    if report["status"] != "complete":
                        raise RuntimeError(
                            "timing completed with failures; see timing/samples.jsonl"
                        )
                if args.mode in ("all", "quality"):
                    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                        report = evaluate_model(
                            wrapper,
                            predictor.data.splits["validation"],
                            [0],
                            output / "quality",
                            point,
                            fail_on_runtime_error=True,
                        )
                    write_json(
                        output / "quality" / "execution.json",
                        {
                            "campaign_id": args.campaign_id,
                            "environment": environment,
                            "data_identity": data_identity,
                            "provenance": point,
                            "quality_semantics": "physical_ensemble_mean",
                            "test_accessed": False,
                        },
                    )
                    if (
                        report.get("failed_clips", 0)
                        or report.get("trajectory_count") != 100
                    ):
                        raise RuntimeError("quality evaluation incomplete")
                if runtime_identity(device) != environment:
                    raise ValueError("execution settings changed during the point")
                write_json(output / "exit.json", {"exit_code": 0})
                if args.mode == "all":
                    write_json(
                        output / "point.json",
                        {
                            "schema": "cylinderflow.pareto_point.v1",
                            "method": "graph_dit_h1",
                            "campaign_id": args.campaign_id,
                            "environment": environment,
                            "data_identity": data_identity,
                            "provenance": point,
                            "sampling_steps": steps,
                            "ensemble_size": count,
                            "quality_semantics": "physical_ensemble_mean",
                            "timing_file": "timing/summary.json",
                            "quality_file": "quality/summary.json",
                            "evaluator": "cylinderflow.physical_mesh.v1",
                            "test_accessed": False,
                        },
                    )
                print(f"Completed {name}", flush=True)
            except Exception as error:
                failed = True
                write_json(
                    output / "exit.json",
                    {"exit_code": 1, "error": f"{type(error).__name__}: {error}"},
                )
                print(f"Failed {name}: {error}", flush=True)
    write_json(args.output_dir / "exit.json", {"exit_code": int(failed)})
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
