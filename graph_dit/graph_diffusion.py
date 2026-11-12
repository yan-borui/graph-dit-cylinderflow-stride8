"""Conditional epsilon diffusion for a 64-frame block on a frozen VGAE mesh."""

from __future__ import annotations

from copy import copy

import torch
from torch import nn

from dgn4cfd.graph import Graph
from dgn4cfd.nn.blocks import SinusoidalPositionEmbedding
from dgn4cfd.nn.models.multi_scale_gnn import MultiScaleGnn


class GraphLatentDiffusion(nn.Module):
    """Denoise all normalized future latents together, with one noise level per clip.

    The caller supplies the clean initial latent, static context, and the same
    three-level latent mesh used by JointGNN. The 64 ordered time slots are
    channels at each node; the input and output projections mix these slots,
    and the graph U-Net handles spatial communication. Diffusion time is
    embedded separately and shared by all physical frames and mesh levels.
    """

    future_frames = 64
    latent_features = 4
    condition_features = 512
    diffusion_steps = 1000
    beta_start = 1e-4
    beta_end = 2e-2
    width = 128
    depths = (1, 2, 2)

    @classmethod
    def architecture(cls) -> dict:
        """Settings that must match when resuming or scoring a checkpoint."""
        return {
            "backbone": "multiscale_graph_unet",
            "objective": "epsilon_mse",
            "width": cls.width,
            "depths": list(cls.depths),
            "latent_channels": cls.latent_features,
            "condition_channels": cls.condition_features,
            "future_frames": cls.future_frames,
            "noise_readout": "graph_features_and_noise_scaled_noisy_block",
            "diffusion_steps": cls.diffusion_steps,
            "beta_schedule": "linear",
            "beta_start": cls.beta_start,
            "beta_end": cls.beta_end,
        }

    def __init__(self) -> None:
        super().__init__()
        width = self.width
        emb_width = 4 * width
        block_features = self.future_frames * self.latent_features
        self.node_encoder = nn.Linear(
            self.latent_features + self.condition_features + block_features, width
        )
        self.edge_encoder = nn.Linear(2, width)
        self.step_encoder = nn.Sequential(
            SinusoidalPositionEmbedding(width),
            nn.Linear(width, emb_width),
            nn.SiLU(),
            nn.Linear(emb_width, emb_width),
        )
        self.propagator = MultiScaleGnn(
            depths=list(self.depths),
            fnns_width=width,
            fnns_depth=2,
            emb_features=emb_width,
            dim=2,
            scalar_rel_pos=True,
        )
        # Give all 256 noisy components a direct path to epsilon. A readout
        # from 128 hidden features alone has rank at most 128 for every node.
        self.noise_head = nn.Linear(width + block_features, block_features)
        with torch.no_grad():
            nn.init.eye_(self.noise_head.weight[:, width:])
        # This is the submitted GLaDiT forward process (linear betas, FP32).
        betas = torch.linspace(
            self.beta_start, self.beta_end, self.diffusion_steps, dtype=torch.float32
        )
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))

    def _check_step(self, step: torch.Tensor) -> None:
        if step.shape != (1,) or step.dtype not in (torch.int32, torch.int64):
            raise ValueError("pass one integer diffusion step with shape [1]")
        if torch.any((step < 0) | (step >= self.diffusion_steps)):
            raise ValueError("diffusion step is outside the trained schedule")

    def forward(
        self,
        z0: torch.Tensor,
        node_context: torch.Tensor,
        graph: Graph,
        noisy_future: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon [64,N,4] from the observed frame and noisy future."""
        self._check_step(step)
        n = graph.num_nodes
        if z0.shape != (n, self.latent_features) or node_context.shape != (
            n,
            self.condition_features,
        ):
            raise ValueError("expected z0 [N,4] and static node context [N,512]")
        if noisy_future.shape != (self.future_frames, n, self.latent_features):
            raise ValueError("expected noisy future [64,N,4] on the latent mesh")
        block = noisy_future.transpose(0, 1).reshape(n, -1)
        nodes = self.node_encoder(torch.cat((z0, node_context, block), dim=-1))
        edges = self.edge_encoder(graph.edge_attr)
        embedding = self.step_encoder(step)
        # Pooling reassigns edge_index and batch; keep the cached hierarchy reusable.
        nodes, _ = self.propagator(copy(graph), nodes, edges, embedding)
        # For standardized unit Gaussian latents, sqrt(1-alpha_bar)*x_t is
        # the conditional mean of epsilon given x_t. The trainable readout
        # and graph features can learn departures from this initialization.
        noise_input = block * (1.0 - self.alpha_bar[step]).sqrt().to(block.dtype)
        return (
            self.noise_head(torch.cat((nodes, noise_input), dim=-1))
            .reshape(n, self.future_frames, self.latent_features)
            .transpose(0, 1)
            .contiguous()
        )

    def corrupt(
        self,
        clean_future: torch.Tensor,
        step: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw independent elementwise noise at one DDPM level for the full window."""
        self._check_step(step)
        if (
            clean_future.ndim != 3
            or clean_future.shape[0] != self.future_frames
            or clean_future.shape[-1] != self.latent_features
        ):
            raise ValueError("expected normalized future [64,N,4]")
        noise = torch.randn(
            clean_future.shape,
            device=clean_future.device,
            dtype=clean_future.dtype,
            generator=generator,
        )
        alpha = self.alpha_bar[step].to(clean_future.dtype)
        noisy = alpha.sqrt() * clean_future + (1.0 - alpha).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def sample(
        self,
        z0: torch.Tensor,
        node_context: torch.Tensor,
        graph: Graph,
        *,
        sampling_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Generate one normalized future from Gaussian noise via eta-zero DDIM."""
        if not 2 <= sampling_steps <= self.diffusion_steps:
            raise ValueError("sampling_steps must be between 2 and 1000")
        future = torch.randn(
            (self.future_frames, graph.num_nodes, self.latent_features),
            device=z0.device,
            dtype=z0.dtype,
            generator=generator,
        )
        schedule = (
            torch.linspace(
                self.diffusion_steps - 1, 0, sampling_steps, device=z0.device
            )
            .round()
            .long()
        )
        schedule = torch.unique_consecutive(schedule)
        for i, step in enumerate(schedule):
            eps = self(z0, node_context, graph, future, step.reshape(1))
            alpha = self.alpha_bar[step].to(future.dtype)
            clean = (future - (1.0 - alpha).sqrt() * eps) / alpha.sqrt().clamp_min(
                torch.finfo(future.dtype).eps
            )
            if i + 1 == len(schedule):
                future = clean
            else:
                next_alpha = self.alpha_bar[schedule[i + 1]].to(future.dtype)
                future = next_alpha.sqrt() * clean + (1.0 - next_alpha).sqrt() * eps
        return future
