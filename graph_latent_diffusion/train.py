"""Four-GPU conditional diffusion training with a graph U-Net trajectory denoiser."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from deterministic_joint.train import parse_args, recipe as joint_recipe, train
from dgn4cfd.graph import Graph
from graph_dit.runtime import ROOT

from . import CHECKPOINT_FORMAT, SELECTION_MEMBERS, SELECTION_STEPS
from .model import GraphLatentDiffusion


def recipe(dataset: str, verification_updates: int | None) -> dict:
    """Match the joint window budget and record the denoising/selection protocol."""
    config = joint_recipe(dataset, verification_updates)
    config["protocol"] = CHECKPOINT_FORMAT
    config["model"] = GraphLatentDiffusion.architecture()
    config["validation"] = {
        "sampling_steps": SELECTION_STEPS,
        "ensemble_size": SELECTION_MEMBERS,
        "aggregation": "mean_decoded_physical_fields_before_scoring",
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
    from .evaluation import validation_rows

    args = parse_args(__doc__)
    train(
        args,
        config=recipe(args.dataset, args.verification_updates),
        build_model=GraphLatentDiffusion,
        loss_function=denoising_loss,
        score_cases=validation_rows,
        extra_sources=[
            *sorted(ROOT.glob("graph_latent_diffusion/*.py")),
            ROOT / "scripts/graph_latent_diffusion_4gpu.sh",
        ],
    )


if __name__ == "__main__":
    main()
