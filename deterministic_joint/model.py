"""Direct joint regression using the paper's graph-local trajectory Transformer."""

from __future__ import annotations

import torch

from dgn4cfd.nn.diffusion.models.graph_video_dit import (
    GraphTrajectoryBackbone,
    sinusoidal_embedding,
)


def _backbone_config() -> dict:
    """The published GLaDiT H1 architecture on both datasets."""
    return {
        "latent_features": 4,
        "condition_features": 512,
        "width": 512,
        "blocks": 24,
        "heads": 8,
        "mlp_ratio": 4.0,
        "future_frames": 64,
        "attention_mode": "graph_hop_mask",
        "graph_hop_limit": 1,
    }


class JointGraphTransformer(GraphTrajectoryBackbone):
    """Use an observed initial slot and 64 zero-valued, time-positioned queries."""

    def __init__(self) -> None:
        super().__init__(**_backbone_config())
        # Both observed/future slots use the original anchor's sinusoidal basis.
        # Keep the denoiser's MLP and AdaLN gates; the regressor learns its own
        # weights with this constant conditioner and no diffusion schedule.
        self.register_buffer(
            "anchor_basis",
            sinusoidal_embedding(torch.zeros(1, 2), self.width),
            persistent=False,
        )

    @staticmethod
    def architecture() -> dict:
        return {
            **_backbone_config(),
            "future_input": "zero_queries",
            "modulation": "fixed_anchor_embedding",
            "target": "normalized_absolute_future_latent",
        }

    def forward(
        self,
        z0: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        hops: torch.Tensor,
    ) -> torch.Tensor:
        """Predict normalized [64, N, 4] latents in one call for one mesh."""
        if z0.ndim != 2 or z0.shape[-1] != self.latent_features:
            raise ValueError("expected an observed [N,4] normalized initial latent")
        future_queries = z0.new_zeros(
            (1, self.future_frames, z0.size(0), self.latent_features)
        )
        return self._predict_slots(
            z0[None, None],
            future_queries,
            self.anchor_basis,
            node_context[None],
            positions[None],
            graph_hops=hops[None],
        )[0]
