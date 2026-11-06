"""Validation, Test and end-to-end timing of the joint graph diffusion forecast."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
import time

import numpy as np
import torch

from deterministic_joint.data import EvaluationData
from deterministic_joint.evaluation import (
    evaluation_parser,
    load_selected_weights,
    run_report,
    validation_rows as score_validation_rows,
)
from graph_dit.performance import configure
from graph_dit.runtime import monitor_indices

from . import CHECKPOINT_FORMAT, SELECTION_MEMBERS, SELECTION_STEPS
from .model import GraphLatentDiffusion


def member_seeds(index: int, ensemble_size: int) -> list[int]:
    """Paper sampling label zero: the first K members of the same per-case pool."""
    return [
        ((member + 1) * 1_000_003 + index * 9_176) % (2**31 - 1)
        for member in range(ensemble_size)
    ]


@torch.inference_mode()
def predict(
    model: GraphLatentDiffusion,
    data: EvaluationData,
    sample: dict,
    *,
    sampling_steps: int = SELECTION_STEPS,
    ensemble_size: int = SELECTION_MEMBERS,
    benchmark_output: bool = False,
) -> np.ndarray:
    """Return the physical UVP mean of K trajectories, conditioned only on frame zero."""
    if not 2 <= sampling_steps <= model.diffusion_steps or ensemble_size < 1:
        raise ValueError("sampling requires 2–1000 DDIM steps and at least one member")
    z0, context, graph, decoder = data.encode(sample)
    total = None
    for seed in member_seeds(int(sample["trajectory_index"]), ensemble_size):
        generator = torch.Generator(device=z0.device).manual_seed(seed)
        latent = model.sample(
            z0, context, graph, sampling_steps=sampling_steps, generator=generator
        )
        field = data.decode(latent, sample, decoder)
        if total is None:
            total = field.astype(np.float64)
        else:
            total += field
        del latent, field
    total /= ensemble_size
    mean = total.astype(np.float32) if benchmark_output else total
    mean[0] = sample["initial"]
    return mean


def validation_rows(
    model: GraphLatentDiffusion,
    data: EvaluationData,
    indices: Iterable[int],
    device: torch.device,
) -> list[dict]:
    """Select weights by S6/K8 decoded field mean over the complete Validation24."""
    rows = score_validation_rows(
        model, data, indices, device, predict_case=partial(predict, model, data)
    )
    for row in rows:
        row["member_prng_seeds"] = member_seeds(row["index"], SELECTION_MEMBERS)
    return rows


def main() -> None:
    parser = evaluation_parser(__doc__)
    parser.add_argument("--sampling-steps", type=int, default=SELECTION_STEPS)
    parser.add_argument("--ensemble-size", type=int, default=SELECTION_MEMBERS)
    args = parser.parse_args()
    if args.mode == "benchmark" and args.split != "validation":
        parser.error("the published timing protocol uses fixed Validation24")
    if (
        not 2 <= args.sampling_steps <= GraphLatentDiffusion.diffusion_steps
        or args.ensemble_size < 1
    ):
        parser.error("expected 2–1000 DDIM steps and ensemble-size >= 1")
    device = torch.device("cuda:0")
    configure(device)
    started = time.perf_counter()
    data = EvaluationData(
        args.dataset, args.split, args.data_dir, args.cache_dir, device
    )
    weights, provenance = load_selected_weights(
        args.result_root,
        args.dataset,
        data.identity,
        CHECKPOINT_FORMAT,
        expected_model=GraphLatentDiffusion.architecture(),
    )
    model = GraphLatentDiffusion().to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    provenance.update(
        model="graph_latent_diffusion",
        sampling_steps=args.sampling_steps,
        ensemble_size=args.ensemble_size,
        sampler="DDIM",
        eta=0,
        model_calls=args.sampling_steps * args.ensemble_size,
        ensemble_aggregation="physical_uvp_mean_before_scoring",
        condition_encodings=1,
        accumulator_dtype="float64",
        ensemble_label=0,
        member_seed_rule="((member+1)*1000003 + trajectory_index*9176) mod (2**31-1)",
    )
    if args.mode == "benchmark":
        provenance["member_prng_seeds_by_case"] = {
            str(index): member_seeds(index, args.ensemble_size)
            for index in monitor_indices(tuple(data.indices))
        }
    run_report(
        args,
        model,
        data,
        provenance,
        partial(
            predict,
            model,
            data,
            sampling_steps=args.sampling_steps,
            ensemble_size=args.ensemble_size,
            benchmark_output=args.mode == "benchmark",
        ),
        device,
        time.perf_counter() - started,
        case_metadata=lambda index: {
            "member_prng_seeds": member_seeds(index, args.ensemble_size)
        },
    )


if __name__ == "__main__":
    main()
