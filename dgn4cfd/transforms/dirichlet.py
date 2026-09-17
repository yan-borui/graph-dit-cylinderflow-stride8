import torch
from typing import Union

from ..graph import Graph


class AddDirichletMask:
    def __init__(
        self,
        num_features: int,
        dirichlet_features: Union[int, list[int]],
        dirichlet_boundary_id: Union[int, list[int]],
    ) -> None:
        self.num_features = num_features
        self.dirichlet_features = (
            list(range(dirichlet_features))
            if isinstance(dirichlet_features, int)
            else dirichlet_features
        )
        assert all([0 <= i < num_features for i in self.dirichlet_features]), (
            "Dirichlet features must be in [0, num_features)"
        )
        self.dirichlet_boundary_id = (
            dirichlet_boundary_id
            if isinstance(dirichlet_boundary_id, list)
            else [dirichlet_boundary_id]
        )

    def __call__(self, graph: Graph) -> Graph:
        graph.dirichlet_mask = torch.zeros(
            graph.num_nodes,
            self.num_features,
            dtype=torch.bool,
            device=graph.pos.device,
        )
        if len(self.dirichlet_features) == 0 or len(self.dirichlet_boundary_id) == 0:
            return graph
        for id in self.dirichlet_boundary_id:
            graph.dirichlet_mask[:, self.dirichlet_features] += (
                graph.bound == id
            ).unsqueeze(1)
        return graph
