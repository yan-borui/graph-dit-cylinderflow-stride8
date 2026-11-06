"""Joint deterministic prediction on the unchanged VGAE latent mesh."""

from __future__ import annotations

from copy import copy

import torch
from torch import nn
from torch_geometric.utils import coalesce

from dgn4cfd.graph import Graph
from dgn4cfd.nn.models.multi_scale_gnn import MultiScaleGnn
from dgn4cfd.transforms.mesh import MeshCoarsening


def build_hierarchy(positions: torch.Tensor, edge_index: torch.Tensor) -> Graph:
    """Extend a latent mesh by two levels, preserving its finest node order."""
    # Guillard's implementation expects incoming edges contiguous by receiver.
    positions = positions.detach().cpu().float()
    edge_index = coalesce(
        edge_index.detach().cpu().long(), num_nodes=len(positions), sort_by_row=False
    )
    row, col = edge_index
    graph = Graph(
        pos=positions, edge_index=edge_index, edge_attr=positions[col] - positions[row]
    )
    return MeshCoarsening(
        num_scales=3, rel_pos_scaling=[None] * 3, scalar_rel_pos=True
    )(graph)


class JointGNN(nn.Module):
    """One U-Net evaluation predicts the ordered [time, node, channel] latent block."""

    def __init__(self) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(4 + 512, 128)
        self.edge_encoder = nn.Linear(2, 128)
        self.propagator = MultiScaleGnn(
            depths=[1, 2, 2],
            fnns_width=128,
            fnns_depth=2,
            emb_features=0,
            dim=2,
            scalar_rel_pos=True,
        )
        self.trajectory_head = nn.Linear(128, 64 * 4)

    def forward(
        self, z0: torch.Tensor, node_context: torch.Tensor, graph: Graph
    ) -> torch.Tensor:
        if z0.shape != (graph.num_nodes, 4) or node_context.shape != (
            graph.num_nodes,
            512,
        ):
            raise ValueError(
                "expected one [N,4] initial latent and [N,512] static context"
            )
        nodes = self.node_encoder(torch.cat((z0, node_context), dim=-1))
        edges = self.edge_encoder(graph.edge_attr)
        # Pooling changes graph.edge_index and graph.batch. Isolate these attribute
        # assignments from the cached mesh and the original frozen codec context.
        nodes, _ = self.propagator(copy(graph), nodes, edges)
        return (
            self.trajectory_head(nodes).reshape(-1, 64, 4).transpose(0, 1).contiguous()
        )
