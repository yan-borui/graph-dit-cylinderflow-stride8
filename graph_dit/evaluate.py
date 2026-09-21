"""Initial-frame-only joint64 prediction, shared physical scoring and selection."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import time
from pathlib import Path

import numpy as np
import torch

from dgn4cfd.cylinderflow_data import (
    CylinderFlowH5Dataset,
    CylinderFlowNormalization,
    build_cylinderflow_transform,
)
from dgn4cfd.graph_distance import unweighted_shortest_path_hops
from dgn4cfd.nn.diffusion.graph_window_codec import FrozenUVPLatentCodec
from dgn4cfd.nn.diffusion.models.graph_video_dit import GraphVideoDiT
from . import EVALUATOR_VERSION
from .data import Dataset
from .metrics import compute_metrics, summarize_trajectories
from .predictions import save_prediction, writeback_velocity, boundary_metrics
from .representation import paths, load_artifacts, validate_model_representation
from .runtime import (
    append_json,
    autocast,
    load_checkpoint,
    monitor_indices,
    synchronize,
    write_csv,
    write_json,
)
from .performance import seed_draw


class Predictor:
    """Keep only the initial state and static mesh available to the neural predictor."""

    def __init__(
        self,
        model: GraphVideoDiT,
        artifacts: Path,
        data_dir: Path,
        device: torch.device,
        precision: str = "fp32",
        *,
        debug: bool = False,
        sampling_steps: int = 20,
    ):
        if not 2 <= sampling_steps <= model.diffusion_steps:
            raise ValueError("sampling steps must be between 2 and diffusion_steps")
        self.sampling_steps = sampling_steps
        self.model, self.device, self.precision = model, device, precision
        self.data = Dataset(*paths(data_dir), debug=debug)
        self.identity = load_artifacts(artifacts, self.data)
        validate_model_representation(model.architecture(), self.identity)
        self.codec = FrozenUVPLatentCodec(
            str(artifacts / "autoencoder.pt"), device=device
        )
        self.normalization = CylinderFlowNormalization.from_manifest(paths(data_dir)[1])
        self.graphs = CylinderFlowH5Dataset(
            paths(data_dir)[0], manifest_path=paths(data_dir)[1]
        )
        self.transforms = build_cylinderflow_transform(paths(data_dir)[1]).transforms
        self.static_graphs = {}
        self.static_hops = {}

    def load_case(self, index: int) -> dict:
        initial = self.data.initial(index)
        if index not in self.static_graphs:
            graph = self.graphs.get_sequence(index, n_in=1)
            # Static edge scaling, boundary labels, mesh hierarchy; normalize UVP inside predict.
            for ordinal in (0, 2, 3):
                graph = self.transforms[ordinal](graph)
            self.static_graphs[index] = graph
            self.static_hops[index] = torch.from_numpy(
                unweighted_shortest_path_hops(
                    graph.edge_index_3, num_nodes=int(graph.pos_3.shape[0])
                )
            )
        return {
            **initial,
            "trajectory_index": int(index),
            "static_graph": self.static_graphs[index],
            "static_hops": self.static_hops[index],
        }

    @torch.no_grad()
    def predict(
        self,
        sample: dict,
        sampling_seed: int | None = None,
        *,
        diagnostics: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if "target" in sample or "field" in sample:
            raise ValueError(
                "predict accepts only physical initial state and static mesh"
            )
        graph = deepcopy(sample["static_graph"])
        graph.target = torch.from_numpy(
            np.asarray(sample["initial"], dtype=np.float32).copy()
        )
        graph = self.transforms[1](graph)
        graph = self.transforms[4](graph)
        graph.to(self.device)
        # The frozen VGAE retains FP32 arithmetic for both quality and matched timing.
        first, context = self.codec.encode_context(graph, graph.field)
        hops = sample["static_hops"].to(self.device).unsqueeze(0)
        generator = (
            None
            if sampling_seed is None
            else torch.Generator(device=self.device).manual_seed(sampling_seed)
        )
        with autocast(self.device, self.precision):
            future = self.model.sample(
                first[None, None],
                context.node_context[None],
                context.positions[None],
                graph_hops=hops,
                generator=generator,
                sampling_steps=self.sampling_steps,
            )
        fields = [np.asarray(sample["initial"], dtype=np.float32)]
        for raw_latent in future[0]:
            decoded = self.codec.autoencoder.decode(
                deepcopy(context.physical_graph),
                raw_latent.float(),
                [value.clone() for value in context.c_latent_list],
                [value.clone() for value in context.e_latent_list],
                [value.clone() for value in context.edge_index_list],
                [value.clone() for value in context.batch_list],
                None,
                None,
            )
            fields.append(self.normalization.denormalize(decoded).float().cpu().numpy())
        pre_boundary = np.stack(fields)
        if not diagnostics:
            return pre_boundary, None
        prediction = writeback_velocity(
            pre_boundary, sample["initial"], sample["node_type"]
        )
        return prediction, pre_boundary

    def predict_ensemble(
        self, sample: dict, member_seeds: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average decoded physical UVP fields while preserving the initial frame."""
        if not member_seeds:
            raise ValueError("an ensemble requires at least one member")
        mean = None
        for member_seed in member_seeds:
            member, _ = self.predict(sample, member_seed, diagnostics=False)
            if mean is None:
                mean = member.astype(np.float64)
            else:
                mean += member
        mean /= len(member_seeds)
        mean[0] = sample["initial"]
        return writeback_velocity(mean, sample["initial"], sample["node_type"]), mean


def evaluate_model(
    predictor: Predictor,
    indices: tuple[int, ...],
    seeds: list[int],
    output: Path,
    provenance: dict,
    *,
    fail_on_runtime_error: bool = False,
    resume: bool = False,
    ensemble_size: int = 1,
) -> dict:
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")
    if ensemble_size != 1 or predictor.sampling_steps != 20:
        provenance = {
            **provenance,
            "sampling_steps": predictor.sampling_steps,
            "ensemble_size": ensemble_size,
            "aggregation": "physical_uvp_mean",
            "member_label_rule": "ensemble_label * ensemble_size + member_index",
        }
    output.mkdir(parents=True, exist_ok=resume)
    expected_identity = {
        **provenance,
        "indices": list(indices),
        "sampling_seeds": seeds,
    }
    identity_file = output / "identity.json"
    if (
        identity_file.exists()
        and json.loads(identity_file.read_text()) != expected_identity
    ):
        raise ValueError("evaluation resume identity mismatch")
    write_json(
        output / "identity.json",
        {**provenance, "indices": list(indices), "sampling_seeds": seeds},
    )
    completed_file = output / "completed_cases.json"
    rows = (
        json.loads(completed_file.read_text())
        if resume and completed_file.exists()
        else []
    )
    completed = {(row["trajectory_index"], row["seed"]) for row in rows}
    if len(completed) != len(rows) or any(
        not (output / row["prediction_file"]).is_file() for row in rows
    ):
        raise ValueError("invalid completed evaluation records")
    started = time.perf_counter()
    for index in indices:
        sample = predictor.load_case(index)
        for label in seeds:
            if (index, label) in completed:
                continue
            member_seeds = [
                seed_draw(index, label * ensemble_size + member)
                for member in range(ensemble_size)
            ]
            actual_seed = member_seeds[0]
            error = None
            synchronize(predictor.device)
            begin = time.perf_counter()
            try:
                if ensemble_size == 1:
                    prediction, pre = predictor.predict(sample, actual_seed)
                else:
                    prediction, pre = predictor.predict_ensemble(sample, member_seeds)
            except (RuntimeError, FloatingPointError) as failure:
                if fail_on_runtime_error and isinstance(failure, RuntimeError):
                    write_json(
                        output / "evaluation_failure.json",
                        {
                            **provenance,
                            "trajectory_index": index,
                            "sampling_label": label,
                            "error": f"{type(failure).__name__}: {failure}",
                        },
                    )
                    raise
                error = f"{type(failure).__name__}: {failure}"
                prediction = np.full(
                    (65, len(sample["points"]), 3), np.nan, dtype=np.float32
                )
                prediction[0] = sample["initial"]
                pre = prediction.copy()
                if predictor.device.type == "cuda":
                    torch.cuda.empty_cache()
            synchronize(predictor.device)
            seconds = time.perf_counter() - begin
            # Read future reference only after inference has finished.
            target = predictor.data.evaluation(index)["field"]
            metrics = compute_metrics(
                prediction,
                target,
                sample["points"],
                sample["cells"],
                sample["node_type"],
                0.0016,
            )
            metrics.update(
                boundary_metrics(
                    prediction, pre, sample["initial"], sample["node_type"]
                )
            )
            primary = metrics.get("uv_relative_rmse")
            metrics["finite"] = bool(
                metrics.get("finite") and primary is not None and np.isfinite(primary)
            )
            filename = (
                output / "predictions" / f"trajectory_{index:04d}_seed{label}.npz"
            )
            save_prediction(
                filename,
                prediction=prediction,
                pre_boundary=pre,
                target=target,
                points=sample["points"],
                cells=sample["cells"],
                node_type=sample["node_type"],
                trajectory_index=index,
                seed=label,
                provenance={
                    **provenance,
                    "prng_seed": actual_seed,
                    "member_prng_seeds": member_seeds,
                    "failure": error,
                },
            )
            row = {
                "trajectory_index": index,
                "seed": label,
                "prng_seed": actual_seed,
                "member_prng_seeds": member_seeds,
                **metrics,
                "inference_seconds": seconds,
                "error": error,
                "prediction_file": str(filename.relative_to(output)),
            }
            rows.append(row)
            if resume:
                from .ddp_train import atomic_rows

                write_json(completed_file, rows)
                atomic_rows(output / "case_metrics.jsonl", rows)
            else:
                append_json(output / "case_metrics.jsonl", row)
    if resume:
        write_json(completed_file, rows)
    summary = summarize_trajectories(rows)
    summary.update(
        evaluator=EVALUATOR_VERSION,
        elapsed_seconds=time.perf_counter() - started,
        indices=list(indices),
        sampling_seeds=seeds,
        provenance=provenance,
    )
    write_json(output / "summary.json", summary)
    write_csv(output / "case_metrics.csv", rows)
    write_csv(output / "trajectory_metrics.csv", summary["trajectory_metrics"])
    write_json(
        output / "failures.json",
        {"failures": [row for row in rows if not row["finite"]]},
    )
    return summary


def load_selected(
    checkpoint_file: Path, artifacts: Path, weights: str
) -> tuple[GraphVideoDiT, dict]:
    checkpoint = load_checkpoint(checkpoint_file)
    identity = load_artifacts(artifacts, config=checkpoint["config"])
    if checkpoint.get("format") not in {
        "graph_dit.h1_b1.training.v1",
        "graph_dit.h1_ddp.training.v2",
    }:
        raise ValueError("unsupported standalone training checkpoint")
    if checkpoint["artifact_id"] != identity["artifact_id"]:
        raise ValueError("checkpoint and prepared representation differ")
    model = GraphVideoDiT(**checkpoint["config"]["model"])
    state = checkpoint["model"] if weights == "raw" else checkpoint["ema"][weights]
    model.load_state_dict(state, strict=True)
    return model.eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scope", choices=("monitor", "validation"), default="validation"
    )
    args = parser.parse_args()
    selected = json.loads((args.run / "selection.json").read_text())
    if not selected.get("complete_stage"):
        raise ValueError(
            "finish the allocated stage before evaluating its selected checkpoint"
        )
    model, checkpoint = load_selected(
        args.run / selected["checkpoint"], args.artifacts, selected["weights"]
    )
    device = torch.device(args.device)
    from .train import configure_runtime

    configure_runtime(device, checkpoint["config"]["training"]["precision"])
    model.to(device)
    predictor = Predictor(
        model,
        args.artifacts,
        args.data_dir,
        device,
        checkpoint["config"]["training"]["precision"],
        sampling_steps=6,
    )
    indices = predictor.data.splits["validation"]
    if args.scope == "monitor":
        indices = monitor_indices(indices)
    evaluate_model(
        predictor,
        indices,
        [0],
        args.output_dir,
        {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "training_seed": checkpoint["config"]["seed"],
            "weights": selected["weights"],
            "update": checkpoint["update"],
            "config": checkpoint["config"],
            "artifact_id": checkpoint["artifact_id"],
            "scope": args.scope,
        },
        ensemble_size=8,
    )


if __name__ == "__main__":
    main()
