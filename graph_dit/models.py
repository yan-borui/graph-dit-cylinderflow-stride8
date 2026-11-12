"""Ellipse dynamics variants sharing one latent codec and training contract."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch_geometric.utils import coalesce

from dgn4cfd.graph import Graph
from dgn4cfd.graph_distance import normalised_shortest_path_distance
from dgn4cfd.nn.diffusion.models.graph_video_dit import GraphVideoDiT
from dgn4cfd.transforms.mesh import MeshCoarsening
from .graph_diffusion import GraphLatentDiffusion


class DeterministicTransformer(GraphVideoDiT):
    """Same trainable backbone, zero future queries and constant noise embedding."""

    model_type = "deterministic"

    def regress(
        self,
        initial: torch.Tensor,
        context: torch.Tensor,
        positions: torch.Tensor,
        hops: torch.Tensor,
    ) -> torch.Tensor:
        queries = initial.new_zeros(
            initial.size(0), self.future_frames, initial.size(2), self.latent_features
        )
        steps = torch.zeros(initial.size(0), dtype=torch.long, device=initial.device)
        return super().forward(
            initial, queries, steps, context, positions, graph_hops=hops
        )

    def training_loss(
        self,
        clean_z0_raw: torch.Tensor,
        clean_future_raw: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        *,
        graph_hops: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        prediction = self.regress(
            self.normalise_latent(clean_z0_raw), node_context, positions, graph_hops
        )
        return torch.mean((prediction - self.normalise_latent(clean_future_raw)) ** 2)

    @torch.no_grad()
    def sample(
        self,
        clean_z0_raw: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        *,
        graph_hops: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        sampling_steps: int = 6,
    ) -> torch.Tensor:
        return self.denormalise_latent(
            self.regress(
                self.normalise_latent(clean_z0_raw), node_context, positions, graph_hops
            )
        )


class DistanceTransformer(GraphVideoDiT):
    """Use the original normalized Euclidean-weighted graph distance bias."""

    def __init__(self, **config) -> None:
        super().__init__(**config)
        self.distances = {}

    def forward(
        self,
        clean_z0: torch.Tensor,
        noisy_future: torch.Tensor,
        diffusion_step: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        graph_distance: torch.Tensor | None = None,
        graph_hops: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if graph_distance is None:
            if positions.size(0) != 1 or graph_hops is None:
                raise ValueError("distance attention requires one mesh and its hops")
            pos = positions[0].detach().cpu()
            edges = (graph_hops[0].detach().cpu() == 1).nonzero().T.contiguous()
            key = (pos.numpy().tobytes(), edges.numpy().tobytes())
            if key not in self.distances:
                self.distances[key] = torch.from_numpy(
                    normalised_shortest_path_distance(pos, edges)
                )
            graph_distance = self.distances[key].to(positions.device)[None]
        return super().forward(
            clean_z0,
            noisy_future,
            diffusion_step,
            node_context,
            positions,
            graph_distance=graph_distance,
            graph_hops=graph_hops,
        )


class GraphDiffusionAdapter(nn.Module):
    """Expose the established graph U-Net through the shared DiT loss/sample API."""

    model_type = "gnn_diffusion"
    diffusion_steps = 1000

    def __init__(self) -> None:
        super().__init__()
        self.denoiser = GraphLatentDiffusion()
        self.register_buffer("latent_mean", torch.zeros(4))
        self.register_buffer("latent_std", torch.ones(4))
        self.hierarchies = {}

    def architecture(self) -> dict:
        return {
            **self.denoiser.architecture(),
            "latent_features": 4,
            "condition_features": 512,
        }

    def set_latent_statistics(self, mean: list[float], std: list[float]) -> None:
        self.latent_mean.copy_(torch.as_tensor(mean, device=self.latent_mean.device))
        self.latent_std.copy_(torch.as_tensor(std, device=self.latent_std.device))

    def hierarchy(self, positions: torch.Tensor, hops: torch.Tensor) -> Graph:
        if positions.size(0) != 1 or hops is None:
            raise ValueError("graph diffusion requires one mesh per rank")
        pos = positions[0].detach().cpu().float()
        edges = (hops[0].detach().cpu() == 1).nonzero().T.contiguous()
        key = (pos.numpy().tobytes(), edges.numpy().tobytes())
        if key not in self.hierarchies:
            edges = coalesce(edges, num_nodes=len(pos), sort_by_row=False)
            row, col = edges
            graph = Graph(pos=pos, edge_index=edges, edge_attr=pos[col] - pos[row])
            self.hierarchies[key] = MeshCoarsening(
                num_scales=3, rel_pos_scaling=[None] * 3, scalar_rel_pos=True
            )(graph)
        return deepcopy(self.hierarchies[key]).to(positions.device)

    def training_loss(
        self,
        clean_z0_raw: torch.Tensor,
        clean_future_raw: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        *,
        graph_hops: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        graph = self.hierarchy(positions, graph_hops)
        first = (clean_z0_raw[0, 0] - self.latent_mean) / self.latent_std
        future = (clean_future_raw[0] - self.latent_mean) / self.latent_std
        step = torch.randint(1000, (1,), device=future.device, generator=generator)
        noisy, noise = self.denoiser.corrupt(future, step, generator=generator)
        prediction = self.denoiser(first, node_context[0], graph, noisy, step)
        return torch.mean((prediction - noise) ** 2)

    @torch.no_grad()
    def sample(
        self,
        clean_z0_raw: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        *,
        graph_hops: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        sampling_steps: int = 6,
    ) -> torch.Tensor:
        future = self.denoiser.sample(
            (clean_z0_raw[0, 0] - self.latent_mean) / self.latent_std,
            node_context[0],
            self.hierarchy(positions, graph_hops),
            sampling_steps=sampling_steps,
            generator=generator,
        )
        return (future * self.latent_std + self.latent_mean)[None]


def build_model(config: dict) -> nn.Module:
    """Construct the declared dynamics type; old checkpoints remain diffusion."""
    kind = config.get("model_type", "diffusion")
    if kind == "deterministic":
        return DeterministicTransformer(**config["model"])
    if kind == "gnn_diffusion":
        expected = GraphLatentDiffusion.architecture()
        expected["latent_features"] = expected.pop("latent_channels")
        expected["condition_features"] = expected.pop("condition_channels")
        if config["model"] != expected:
            raise ValueError(
                "graph diffusion configuration differs from its fixed backbone"
            )
        return GraphDiffusionAdapter()
    if kind != "diffusion":
        raise ValueError(f"unsupported dynamics model: {kind}")
    if config["model"]["attention_mode"] == "graph_distance_bias":
        return DistanceTransformer(**config["model"])
    return GraphVideoDiT(**config["model"])
