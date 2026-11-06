"""Multiscale graph U-Net from DGN4CFD (Apache-2.0, see NOTICE.md)."""

import torch
from torch import nn

from ...graph import Graph
from ..blocks import InteractionNetwork, MeshDownMP, MeshUpMP


class MultiScaleGnn(nn.Module):
    r"""Multi-scale graph neural network based "Multi-scale rotation-equivariant graph neural networks for unsteady Eulerian fluid dynamics" (https://doi.org/10.1063/5.0097679)

    Args:
        depths (list[int]): List with the depth of each scale.
        fnns_depth (int): Depth of the FNNs.
        fnns_width (int): Width of the FNNs.
        emb_features (int, optional): Number of embedding features. If `emb_features` > 0, then it is a diffusion model. Defaults to 0.
        aggr: (str, optional): Aggregation method for the message passing. Defaults to 'mean'.
        activation (nn.Module, optional): Activation function. Defaults to nn.SELU.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        level_0 (int, optional): Initial level. In latent models, this is the level of the latent features. Defaults to 0.
        dim (int, optional): Spatial dimension of the graph. Defaults to 2.
        scalar_rel_pos (bool, optional): Provide `True` if the relative position betweeen hr and lr nodes is scalar. Defaults to True.

    Methods:
        reset_parameters: Reset the parameters of the model.
        forward: Forward pass through the multi-scale GNN.
    """

    def __init__(
        self,
        depths: list[int],
        fnns_depth: int,
        fnns_width: int,
        emb_features: int = 0,
        aggr: str = "mean",
        activation: nn.Module = nn.SELU,
        dropout: float = 0.0,
        scale_0: int = 0,
        dim: int = 2,
        scalar_rel_pos: bool = True,
    ) -> None:
        scales = len(depths)
        # Validate the hyperparameters
        assert all([depth > 0 for depth in depths]), (
            "The depth of each scale must be positive"
        )
        assert fnns_depth >= 2, "The depth of the FNNs must be at least 2"
        assert fnns_width > 0, "The width of the FNNs must be positive"
        assert fnns_depth >= 2, "mlps_depth must be at least 2"
        assert dim > 0, "The dimension must be positive"
        assert aggr in ("mean", "sum"), "Aggregation must be either 'mean' or 'sum'"
        assert scale_0 >= 0, "The initial level must be non-negative"
        self.level_0 = scale_0
        super().__init__()
        # Down blocks
        self.down_blocks = nn.ModuleList(
            [
                self.DownBlock(
                    scale=self.level_0 + l,
                    fnns_depth=fnns_depth,
                    fnns_width=fnns_width,
                    emb_features=emb_features,
                    activation=activation,
                    dropout=dropout,
                    aggr=aggr,
                    depth=depths[l],
                    dim=dim,
                    scalar_rel_pos=scalar_rel_pos,
                )
                for l in range(scales - 1)
            ]
        )
        # Bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList(
            [
                InteractionNetwork(
                    in_node_features=fnns_width,
                    in_edge_features=fnns_width,
                    out_node_features=fnns_width,
                    out_edge_features=fnns_width,
                    fnns_depth=fnns_depth,
                    fnns_width=fnns_width,
                    emb_features=emb_features,
                    activation=activation,
                    dropout=dropout,
                    aggr=aggr,
                )
                for _ in range(depths[-1])
            ]
        )
        # Up blocks
        self.up_blocks = nn.ModuleList(
            [
                self.UpBlock(
                    scale=self.level_0 + l,
                    fnns_depth=fnns_depth,
                    fnns_width=fnns_width,
                    emb_features=emb_features,
                    activation=activation,
                    dropout=dropout,
                    aggr=aggr,
                    depth=depths[l],
                    dim=dim,
                    scalar_rel_pos=scalar_rel_pos,
                )
                for l in range(scales - 1)[::-1]
            ]
        )

    def reset_parameters(self):
        modules = [
            module for module in self.children() if hasattr(module, "reset_parameters")
        ]
        for module in modules:
            module.reset_parameters()

    def forward(
        self,
        graph: Graph,
        v: torch.Tensor,
        e: torch.Tensor,
        emb: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the multi-scale GNN.

        Args:
            graph (Graph): Graph.
            v (torch.Tensor): Node features. Dimensions: (num_nodes, mlps_width).
            e (torch.Tensor): Edge features. Dimensions: (num_edges, mlps_width).
            emb (torch.Tensor, optional): Diffusion-step embedding. Dimensions: (batch_size, emb_width). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Node and edge features of the final graph. Dimensions: (num_nodes, mlps_width) and (num_edges, mlps_width), respectively.
        """

        # Apply the down blocks
        v_skips, e_skips, edge_index_skips, batch_skips = [], [], [], []
        for down_block in self.down_blocks:
            edge_index_skips.append(graph.edge_index)
            batch_skips.append(graph.batch)
            graph, v, e, v_skip, e_skip = down_block(
                graph, v, e, emb
            )  # This also updates graph.edge_index and graph.batch
            v_skips.append(v_skip)
            e_skips.append(e_skip)
        # Apply the bottleneck blocks
        for bottleneck_block in self.bottleneck_blocks:
            v, e = bottleneck_block(v, e, graph.edge_index, graph.batch, emb)
        # Apply the upsampling residual blocks
        for up_block in self.up_blocks:
            graph, v, e = up_block(
                graph,
                v,
                v_skips.pop(),
                e_skips.pop(),
                edge_index_skips.pop(),
                batch_skips.pop(),
                emb,
            )  # This also updates graph.edge_index and graph.batch
        return v, e

    class DownBlock(nn.Module):
        def __init__(
            self,
            scale: int,
            fnns_depth: int,
            fnns_width: int,
            emb_features: int,
            activation: nn.Module = nn.SELU,
            dropout: float = 0.0,
            aggr: str = "mean",
            depth: int = 1,
            dim: int = 2,
            scalar_rel_pos: bool = True,
        ) -> None:
            # Validate inputs
            assert fnns_depth >= 2, "The depth of the FNNs must be at least 2"
            assert fnns_width > 0, "The width of the FNNs must be positive"
            assert emb_features >= 0, (
                "The number of embedding features must be non-negative"
            )
            assert scale >= 0, "The level must be non-negative"
            assert depth > 0, "The depth must be positive"
            super().__init__()
            # MP blocks
            self.mp_blocks = nn.ModuleList(
                [
                    InteractionNetwork(
                        in_node_features=fnns_width,
                        in_edge_features=fnns_width,
                        out_node_features=fnns_width,
                        out_edge_features=fnns_width,
                        fnns_depth=fnns_depth,
                        fnns_width=fnns_width,
                        emb_features=emb_features,
                        activation=activation,
                        dropout=dropout,
                        aggr=aggr,
                    )
                    for _ in range(depth)
                ]
            )
            # DownMP layer
            self.pooling = MeshDownMP(
                scale=scale,
                dim=dim,
                in_node_features=fnns_width,
                fnn_depth=fnns_depth,
                fnn_width=fnns_width,
                activation=activation,
                dropout=dropout,
                aggr=aggr,
                scalar_rel_pos=scalar_rel_pos,
            )

        def reset_parameters(self):
            modules = [
                module
                for module in self.children()
                if hasattr(module, "reset_parameters")
            ]
            for module in modules:
                module.reset_parameters()

        def forward(
            self, graph: Graph, v: torch.Tensor, e: torch.Tensor, emb: torch.Tensor
        ) -> tuple[Graph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass through the down block.

            Args:
                graph (Graph): Graph. The `edge_index` and `batch` attributes will be modified.
                v (torch.Tensor): Node features. Dimensions: (num_nodes, mlps_width).
                e (torch.Tensor): Edge features. Dimensions: (num_edges, mlps_width).
                emb (torch.Tensor): Diffusion-step embedding. Dimensions: (batch_size, emb_width).

            Returns:
                Graph: Lower resolution graph. The `edge_index` and `batch` attributes are modified.
                torch.Tensor: Node features of the lower resolution graph. Dimensions: (num_nodes, mlps_width).
                torch.Tensor: Edge features of the lower resolution graph. Dimensions: (num_edges, mlps_width).
                torch.Tensor: Node features of the skip connection. Dimensions: (num_nodes, mlps_width).
                torch.Tensor: Edge features of the skip connection. Dimensions: (num_edges, mlps_width).
            """
            # Apply the MP blocks
            for mp in self.mp_blocks:
                v, e = mp(v, e, graph.edge_index, graph.batch, emb)
            v_skip, e_skip = v, e
            # Apply the graph-pooling layer (Also updates `graph.edge_index` and `graph.batch`)
            graph, v, e = self.pooling(graph, v)
            return graph, v, e, v_skip, e_skip

    class UpBlock(nn.Module):
        def __init__(
            self,
            scale: int,
            fnns_depth: int,
            fnns_width: int,
            emb_features: int,
            activation: nn.Module = nn.SELU,
            dropout: float = 0.0,
            aggr: str = "mean",
            depth: int = 1,
            dim: int = 2,
            scalar_rel_pos: bool = True,
        ) -> None:
            # Validate inputs
            assert scale >= 0, "The level must be non-negative"
            assert fnns_depth >= 2, "The depth of the FNNs must be at least 2"
            assert fnns_width > 0, "The width of the FNNs must be positive"
            assert emb_features >= 0, (
                "The number of embedding features must be non-negative"
            )
            assert depth > 0, "The depth must be positive"
            super().__init__()
            # Graph-unpooling layer
            self.unpooling = MeshUpMP(
                scale=scale,
                dim=dim,
                in_features=fnns_width,
                fnn_depth=fnns_depth,
                fnn_width=fnns_width,
                activation=activation,
                dropout=dropout,
                scalar_rel_pos=scalar_rel_pos,
            )
            # MP blocks
            self.mp_blocks = nn.ModuleList(
                [
                    InteractionNetwork(
                        in_node_features=fnns_width,
                        in_edge_features=fnns_width,
                        out_node_features=fnns_width,
                        out_edge_features=fnns_width,
                        fnns_depth=fnns_depth,
                        fnns_width=fnns_width,
                        emb_features=emb_features,
                        activation=activation,
                        dropout=dropout,
                        aggr=aggr,
                    )
                    for _ in range(depth)
                ]
            )

        def reset_parameters(self):
            modules = [
                module
                for module in self.children()
                if hasattr(module, "reset_parameters")
            ]
            for module in modules:
                module.reset_parameters()

        def forward(
            self,
            graph: Graph,
            v: torch.Tensor,
            v_skip: torch.Tensor,
            e_skip: torch.Tensor,
            edge_index_skip: torch.Tensor,
            batch_skip: torch.Tensor,
            emb: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, Graph]:
            """Forward pass through the up block.

            Args:
                graph (Graph): Graph. The `edge_index` and `batch` attributes will be modified.
                v (torch.Tensor): Node features. Dimensions: (num_nodes, mlps_width).
                v_skip (torch.Tensor): Node features of the skip connection. Dimensions: (num_nodes, mlps_width).
                e_skip (torch.Tensor): Edge features of the skip connection. Dimensions: (num_edges, mlps_width).
                edge_index_skip (torch.Tensor): Edge index of the skip connection.
                batch_skip (torch.Tensor): Batch of the skip connection.
                emb (torch.Tensor): Diffusion-step embedding. Dimensions: (batch_size, emb_width).

            Returns:
                Graph: Higher resolution graph. The `edge_index` and `batch` attributes are modified.
                torch.Tensor: Node features of the higher resolution graph. Dimensions: (num_nodes, mlps_width).
                torch.Tensor: Edge features of the higher resolution graph. Dimensions: (num_edges, mlps_width).
            """

            # Apply the UpMP layer (Also updates `graph.edge_index` and `graph.batch`)
            graph, v = self.unpooling(graph, v, edge_index_skip, v_skip, batch_skip)
            # Restore the edge features
            e = e_skip
            # Apply the residual blocks
            for mp in self.mp_blocks:
                v, e = mp(v, e, graph.edge_index, graph.batch, emb)
            return graph, v, e
