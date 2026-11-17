"""Four-GPU conditional diffusion training with a graph U-Net trajectory denoiser."""

from __future__ import annotations

from functools import partial

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from deterministic_joint.train import argument_parser, recipe as joint_recipe, train
from dgn4cfd.graph import Graph
from graph_dit.runtime import ROOT

from . import CHECKPOINT_FORMAT, SELECTION_MEMBERS, SELECTION_STEPS
from .model import GraphLatentDiffusion


def recipe(dataset: str, verification_updates: int | None, width: int = 128) -> dict:
    """Match the joint window budget and record the denoising/selection protocol."""
    config = joint_recipe(dataset, verification_updates)
    config["protocol"] = CHECKPOINT_FORMAT
    config["model"] = GraphLatentDiffusion.architecture(width=width)
    config["validation"] = {
        "sampling_steps": SELECTION_STEPS,
        "ensemble_size": SELECTION_MEMBERS,
        "aggregation": "mean_decoded_physical_fields_before_scoring",
    }
    if width != 128:
        config["campaign"] = "graph_diffusion_width_ladder.v1"
        config["ema_decays"] = [0.999]
        config["recovery_every"] = min(5000, verification_updates or 5000)
        config["validation"] = {
            "sampling_steps": 6,
            "ensemble_size": 1,
            "independent_repeats": 1,
            "sampling_labels": [0],
            "aggregation": "single_K1_score_then_equal_trajectory_mean",
            "weight_names": ["ema_0.999"],
        }
    return config


def denoising_loss(
    ddp: DistributedDataParallel,
    z0: torch.Tensor,
    context: torch.Tensor,
    graph: Graph,
    target: torch.Tensor,
) -> torch.Tensor:
    """Share one noise level across all 64 future frames; leave the observation clean."""
    model = ddp.module
    # Each rank has its own seeded CUDA generator state in the recovery checkpoint.
    step = torch.randint(model.diffusion_steps, (1,), device=target.device)
    noisy, epsilon = model.corrupt(target, step)
    prediction = ddp(z0, context, graph, noisy, step)
    if prediction.shape != epsilon.shape:
        raise ValueError(
            f"predicted {tuple(prediction.shape)} vs epsilon {tuple(epsilon.shape)}"
        )
    return F.mse_loss(prediction, epsilon)


def main() -> None:
    from .evaluation import validation_k1_rows, validation_rows

    parser = argument_parser(__doc__)
    parser.add_argument(
        "--width", type=int, choices=GraphLatentDiffusion.supported_widths, default=128
    )
    args = parser.parse_args()
    train(
        args,
        config=recipe(args.dataset, args.verification_updates, args.width),
        build_model=partial(GraphLatentDiffusion, width=args.width),
        loss_function=denoising_loss,
        score_cases=validation_rows if args.width == 128 else validation_k1_rows,
        extra_sources=[
            *sorted(ROOT.glob("graph_latent_diffusion/*.py")),
            ROOT / "scripts/graph_latent_diffusion_4gpu.sh",
            ROOT / "scripts/graph_diffusion_ladder_4gpu.sh",
            ROOT / "graph_dit/matched_k1.py",
        ],
    )


if __name__ == "__main__":
    main()
