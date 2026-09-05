from typing import Union
import torch
from torch import nn
from torch_geometric.utils import scatter
from torch_geometric.data import Batch
from copy import deepcopy
from tqdm import tqdm

from ..model import Model
from ...graph import Graph
from ..blocks import InteractionNetwork, MeshDownMP, MeshUpMP
from ...loader import Collater


class DownBlock(nn.Module):
    def __init__(
        self,
        scale: int,
        depth: int,
        fnns_depth: int,
        fnns_width: int,
        activation: nn.Module,
        dropout: float,
        aggr: str,
        encode_edges: bool,
        dim: int = 2,
        scalar_rel_pos: bool = True,
    ) -> None:
        # Validate inputs
        assert scale >= 0, "The scale must be non-negative"
        assert depth > 0, "The depth must be positive"
        super().__init__()
        self.scale = scale
        self.depth = depth
        self.fnns_depth = fnns_depth
        self.fnns_width = fnns_width
        self.activation = activation
        self.dropout = dropout
        self.aggr = aggr
        self.encode_edges = encode_edges
        self.dim = dim
        self.scalar_rel_pos = scalar_rel_pos
        # MP blocks
        self.mp_blocks = nn.ModuleList(
            [
                InteractionNetwork(
                    in_node_features=self.fnns_width,
                    in_edge_features=self.fnns_width,
                    out_node_features=self.fnns_width,
                    out_edge_features=self.fnns_width,
                    fnns_depth=self.fnns_depth,
                    fnns_width=self.fnns_width,
                    activation=self.activation,
                    dropout=self.dropout,
                    aggr=self.aggr,
                )
                for _ in range(depth)
            ]
        )
        # Pooling
        self.pooling = MeshDownMP(
            scale=scale,
            dim=self.dim,
            in_node_features=self.fnns_width,
            fnn_depth=self.fnns_depth,
            fnn_width=self.fnns_width,
            activation=self.activation,
            dropout=self.dropout,
            aggr=self.aggr,
            encode_edges=self.encode_edges,
            scalar_rel_pos=self.scalar_rel_pos,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply the MP blocks
        for mp in self.mp_blocks:
            v, e = mp(v, e, graph.edge_index)
        v_skip, e_skip = v.clone(), e.clone()
        # Apply the DownMP layer
        graph, v, e = self.pooling(
            graph, v
        )  # Also updates `graph.edge_index` and `graph.batch`
        return v, e, v_skip, e_skip


class UpBlock(nn.Module):
    def __init__(
        self,
        scale: int,
        depth: int,
        fnns_depth: int,
        fnns_width: int,
        activation: nn.Module,
        dropout: float,
        aggr: str,
        dim: int = 2,
        scalar_rel_pos: bool = True,
    ):
        # Validate inputs
        assert scale >= 0, "The scale must be non-negative"
        assert depth > 0, "The depth must be positive"
        self.scale = scale
        self.depth = depth
        self.fnns_depth = fnns_depth
        self.fnns_width = fnns_width
        self.activation = activation
        self.dropout = dropout
        self.aggr = aggr
        self.dim = dim
        self.scalar_rel_pos = scalar_rel_pos
        super().__init__()
        # Unpooling
        self.unpooling = MeshUpMP(
            scale=self.scale,
            dim=1,
            in_features=self.fnns_width,
            fnn_depth=self.fnns_depth,
            fnn_width=self.fnns_width,
            activation=self.activation,
            dropout=self.dropout,
            skip_connection=False,
            scalar_rel_pos=self.scalar_rel_pos,
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
                    activation=activation,
                    dropout=dropout,
                    aggr=aggr,
                )
                for _ in range(depth)
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
        c_skip: torch.Tensor,
        e_skip: torch.Tensor,
        edge_index_skip: torch.Tensor,
        batch_skip: torch.Tensor,
    ) -> torch.Tensor:
        # Unpooling
        graph, v = self.unpooling(
            graph, v, edge_index_skip, batch_skip=batch_skip
        )  # Also updates `graph.edge_index` and `graph.batch`
        # Restore the edge features
        e = e_skip
        # Add the condition at the current scale
        v += c_skip
        # Apply the residual blocks
        for mp in self.mp_blocks:
            v, e = mp(v, e, graph.edge_index)
        return v


class CondEncoder(nn.Module):
    def __init__(
        self,
        cond_node_features: int,
        cond_edge_features: int,
        depths: Union[int, list[int]],
        fnns_depth: int,
        fnns_width: int,
        activation: nn.Module = nn.SELU,
        dropout: float = 0.0,
        aggr: str = "mean",
        dim: int = 2,
        scalar_rel_pos: bool = True,
    ) -> None:
        super().__init__()
        nun_scales = len(depths)
        # Validate the inputs
        assert cond_node_features >= 0, "Input node features must be non-negative"
        assert cond_edge_features > 0, "Input edge features must be positive"
        assert all([depth > 0 for depth in depths]), (
            "The depth of each scale must be positive"
        )
        # Assign the attributes
        self.cond_node_features = cond_node_features
        self.cond_edge_features = cond_edge_features
        self.depths = depths
        self.fnns_depth = fnns_depth
        self.fnns_width = fnns_width
        self.activation = activation
        self.dropout = dropout
        self.aggr = aggr
        # Input layers
        self.in_node_layer = (
            nn.Linear(self.cond_node_features, self.fnns_width)
            if self.cond_node_features > 0
            else nn.Linear(self.cond_edge_features, self.fnns_width)
        )
        self.in_edge_layer = nn.Linear(self.cond_edge_features, self.fnns_width)
        # Down blocks
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    scale=l,
                    depth=self.depths[l],
                    fnns_depth=self.fnns_depth,
                    fnns_width=self.fnns_width,
                    activation=self.activation,
                    dropout=self.dropout,
                    aggr=self.aggr,
                    encode_edges=True,
                    dim=dim,
                    scalar_rel_pos=scalar_rel_pos,
                )
                for l in range(nun_scales - 1)
            ]
        )
        # Botleneck blocks
        self.bottleneck_blocks = nn.ModuleList(
            [
                InteractionNetwork(
                    in_node_features=self.fnns_width,
                    in_edge_features=self.fnns_width,
                    out_node_features=self.fnns_width,
                    out_edge_features=self.fnns_width,
                    fnns_depth=self.fnns_depth,
                    fnns_width=self.fnns_width,
                    activation=self.activation,
                    dropout=self.dropout,
                    aggr=self.aggr,
                )
                for _ in range(self.depths[-1])
            ]
        )
        # Batch norm
        self.node_bn_list = nn.ModuleList(
            [
                nn.BatchNorm1d(self.fnns_width, affine=False, momentum=0.001)
                for _ in range(nun_scales)
            ]
        )
        self.edge_bn_list = nn.ModuleList(
            [
                nn.BatchNorm1d(self.fnns_width, affine=False, momentum=0.001)
                for _ in range(nun_scales)
            ]
        )

    def reset_parameters(self):
        modules = [
            module for module in self.children() if hasattr(module, "reset_parameters")
        ]
        for module in modules:
            module.reset_parameters()

    def forward(
        self, graph: Graph, c: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.LongTensor],
    ]:
        # Aggregate the input edge features
        e_aggr = scatter(e, edge_index[1], dim=0, reduce=self.aggr)
        # Apply input layers
        c = (
            self.in_node_layer(e_aggr)
            if self.cond_node_features == 0
            else self.in_node_layer(c)
        )
        e = self.in_edge_layer(e)
        # Apply the down blocks
        c_latent_list, e_latent_list, edge_index_list, batch_list = [], [], [], []
        for l, down_block in enumerate(self.down_blocks):
            edge_index_list.append(graph.edge_index)
            batch_list.append(graph.batch)
            c, e, c_skip, e_skip = down_block(graph, c, e)
            c_latent_list.append(self.node_bn_list[l](c_skip))
            e_latent_list.append(self.edge_bn_list[l](e_skip))
        # Apply the bottleneck blocks
        for bottleneck_block in self.bottleneck_blocks:
            c, e = bottleneck_block(c, e, graph.edge_index)
        c_latent_list.append(self.node_bn_list[-1](c))
        e_latent_list.append(self.edge_bn_list[-1](e))
        edge_index_list.append(graph.edge_index)
        batch_list.append(graph.batch)
        return c_latent_list, e_latent_list, edge_index_list, batch_list


class VariationalNodeEncoder(nn.Module):
    def __init__(
        self,
        in_node_features: int,
        out_node_features: int,
        depths: Union[int, list[int]],
        fnns_depth: int,
        fnns_width: int,
        activation: nn.Module = nn.SELU,
        dropout: float = 0.0,
        aggr: str = "mean",
        norm_latents: bool = False,
        dim: int = 2,
        scalar_rel_pos: bool = True,
    ) -> None:
        super().__init__()
        num_scales = len(depths)
        # Validate the inputs
        assert in_node_features >= 0, "Input node features must be positive"
        assert out_node_features > 0, "Output node features must be positive"
        assert all([depth > 0 for depth in depths]), (
            "The depth of each scale must be positive"
        )
        # Assign the attributes
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.depths = depths
        self.fnns_depth = fnns_depth
        self.fnns_width = fnns_width
        self.activation = activation
        self.dropout = dropout
        self.aggr = aggr
        self.norm_latents = norm_latents
        # Input linear layers
        self.in_node_layer = nn.Linear(
            self.in_node_features,
            self.fnns_width,
        )
        self.in_cond_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.fnns_width,
                    self.fnns_width,
                )
                for _ in range(num_scales)
            ]
        )
        self.in_edge_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.fnns_width,
                    self.fnns_width,
                )
                for _ in range(num_scales)
            ]
        )
        # Down blocks
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    scale=l,
                    depth=self.depths[l],
                    fnns_depth=self.fnns_depth,
                    fnns_width=self.fnns_width,
                    activation=self.activation,
                    dropout=self.dropout,
                    aggr=self.aggr,
                    encode_edges=False,
                    dim=dim,
                    scalar_rel_pos=scalar_rel_pos,
                )
                for l in range(num_scales - 1)
            ]
        )
        # Bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList(
            [
                InteractionNetwork(
                    in_node_features=self.fnns_width,
                    in_edge_features=self.fnns_width,
                    out_node_features=self.fnns_width,
                    out_edge_features=self.fnns_width,
                    fnns_depth=self.fnns_depth,
                    fnns_width=self.fnns_width,
                    activation=self.activation,
                    dropout=self.dropout,
                    aggr=self.aggr,
                )
                for _ in range(self.depths[-1])
            ]
        )
        # Output layers
        self.out_layers = nn.Sequential(
            nn.Linear(
                in_features=self.fnns_width,
                out_features=8 * self.out_node_features,
            ),
            self.activation(),
            nn.Linear(
                in_features=8 * self.out_node_features,
                out_features=2 * self.out_node_features,
            ),
        )
        if self.norm_latents:
            # Normalisation
            self.batch_norm = nn.LazyBatchNorm1d(affine=False, momentum=0.001)
            self.bn = lambda x: self.batch_norm(x.reshape(-1, 1)).reshape(x.shape)

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
        c_latent_list: list[torch.Tensor],
        e_latent_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply input layers
        v = self.in_node_layer(v)
        # Apply the down blocks
        for i, down_block in enumerate(self.down_blocks):
            v += self.in_cond_layers[i](c_latent_list.pop(0))
            e = self.in_edge_layers[i](e_latent_list.pop(0))
            v, _, _, _ = down_block(
                graph, v, e
            )  # Also updates `graph.edge_index` and `graph.batch`
        # Apply the bottleneck blocks
        v += self.in_cond_layers[-1](c_latent_list.pop(0))
        e = self.in_edge_layers[-1](e_latent_list.pop(0))
        for bottleneck_block in self.bottleneck_blocks:
            v, e = bottleneck_block(v, e, graph.edge_index)
        # Apply the output layers
        v = self.out_layers(v)
        # Compute the mean and logvar of the latent node features distribution
        mean, logvar = v.chunk(2, dim=1)
        # Compute the latent node features
        eps = torch.randn_like(mean)
        v_latent = mean + eps * torch.exp(0.5 * logvar)
        if self.norm_latents:
            v_latent = self.bn(v_latent)
        return v_latent, mean, logvar


class NodeDecoder(nn.Module):
    def __init__(
        self,
        in_node_features: int,
        out_node_features: int,
        depths: Union[int, list[int]],
        fnns_depth: int,
        fnns_width: int,
        activation: nn.Module = nn.SELU,
        dropout: float = 0.0,
        aggr: str = "mean",
        dim: int = 2,
        scalar_rel_pos: bool = True,
    ) -> None:
        super().__init__()
        self.num_scales = len(depths)
        # Validate the inputs
        assert in_node_features > 0, "Input node features must be positive"
        assert out_node_features > 0, "Output node features must be positive"
        assert all([depth > 0 for depth in depths]), (
            "The depth of each scale must be positive"
        )
        # Assign the attributes
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.depths = depths
        self.fnns_depth = fnns_depth
        self.fnns_width = fnns_width
        self.activation = activation
        self.dropout = dropout
        self.aggr = aggr
        self.dim = dim
        self.scalar_rel_pos = scalar_rel_pos
        # Input layers
        self.in_node_layer = nn.Linear(
            self.in_node_features,
            self.fnns_width,
        )
        self.in_cond_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.fnns_width,
                    self.fnns_width,
                )
                for _ in range(self.num_scales)
            ]
        )
        self.in_edge_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.fnns_width,
                    self.fnns_width,
                )
                for _ in range(self.num_scales)
            ]
        )
        # Bottleneck blocks
        self.bottleneck_blocks = nn.ModuleList(
            [
                InteractionNetwork(
                    in_node_features=self.fnns_width,
                    in_edge_features=self.fnns_width,
                    out_node_features=self.fnns_width,
                    out_edge_features=self.fnns_width,
                    fnns_depth=self.fnns_depth,
                    fnns_width=self.fnns_width,
                    activation=self.activation,
                    dropout=self.dropout,
                    aggr=self.aggr,
                )
                for _ in range(self.depths[0])
            ]
        )
        # Up blocks (except the last one)
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    scale=self.num_scales - l - 1,
                    depth=self.depths[l],
                    fnns_depth=self.fnns_depth,
                    fnns_width=self.fnns_width,
                    activation=self.activation,
                    dropout=self.dropout,
                    aggr=self.aggr,
                    dim=self.dim,
                    scalar_rel_pos=self.scalar_rel_pos,
                )
                for l in range(1, self.num_scales)
            ]
        )
        # Output layer
        self.out_layer = nn.Linear(self.fnns_width, self.out_node_features)

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
        c_latent_list: list[torch.Tensor],
        e_latent_list: list[torch.Tensor],
        edge_index_list: list[torch.Tensor],
        batch_list: list[torch.LongTensor],
    ) -> torch.Tensor:
        graph.edge_index, graph.batch = edge_index_list.pop(), batch_list.pop()
        # Input layers
        v = self.in_node_layer(v)
        # Bottleneck blocks
        v += self.in_cond_layers[0](c_latent_list.pop())
        e = self.in_edge_layers[0](e_latent_list.pop())
        for bottleneck_block in self.bottleneck_blocks:
            v, e = bottleneck_block(v, e, graph.edge_index)
        # Up blocks
        for i, up_block in enumerate(self.up_blocks):
            c = self.in_cond_layers[i + 1](c_latent_list.pop())
            e = self.in_edge_layers[i + 1](e_latent_list.pop())
            v = up_block(graph, v, c, e, edge_index_list.pop(), batch_list.pop())
        # Apply the output layer
        v = self.out_layer(v)
        return v


class VGAE(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # IO features
        self.in_node_features = arch["in_node_features"]
        self.cond_node_features = arch["cond_node_features"]
        self.cond_edge_features = arch.get("cond_edge_features", 0)
        self.out_node_features = self.in_node_features
        if "in_edge_features" in arch:  # To support backward compatibility
            self.cond_edge_features += arch["in_edge_features"]
        # Hyperparameters
        self.latent_node_features = arch["latent_node_features"]
        self.depths = arch["depths"]
        self.fnns_depth = arch.get("fnns_depth", 2)
        self.fnns_width = arch["fnns_width"]
        self.activation = arch.get("activation", nn.SELU)
        self.dropout = arch.get("dropout", 0.0)
        self.aggr = arch.get("aggr", "mean")
        self.norm_latents = arch.get("norm_latents", False)
        self.num_scales = len(self.depths)
        self.dim = arch.get("dim", 2)
        self.scalar_rel_pos = arch.get("scalar_rel_pos", True)
        # Validate the inputs
        assert self.in_node_features > 0, "Input node features must be positive"
        assert self.cond_node_features > 0, "Conditional features must be positive"
        assert self.out_node_features > 0, "Output node features must be positive"
        assert self.latent_node_features > 0, "Latent node features must be positive"
        assert all([depth > 0 for depth in self.depths]), (
            "The depth of each scale must be positive"
        )
        assert self.fnns_depth >= 2, "The depth of the FNNs must be at least 2"
        assert self.fnns_width > 0, "The width of the FNNs must be positive"
        assert self.aggr in ["mean", "sum"], "Unknown aggregation method"
        # Edge encoder
        self.cond_encoder = CondEncoder(
            cond_node_features=self.cond_node_features,
            cond_edge_features=self.cond_edge_features,
            depths=[
                1,
            ]
            * len(self.depths),
            fnns_depth=self.fnns_depth,
            fnns_width=self.fnns_width,
            activation=self.activation,
            dropout=self.dropout,
            aggr=self.aggr,
            dim=self.dim,
            scalar_rel_pos=self.scalar_rel_pos,
        )
        # Node variational encoder
        self.node_encoder = VariationalNodeEncoder(
            in_node_features=self.in_node_features,
            out_node_features=self.latent_node_features,
            depths=self.depths,
            fnns_depth=self.fnns_depth,
            fnns_width=self.fnns_width,
            activation=self.activation,
            dropout=self.dropout,
            aggr=self.aggr,
            norm_latents=self.norm_latents,
            dim=self.dim,
            scalar_rel_pos=self.scalar_rel_pos,
        )
        # Node decoder
        self.node_decoder = NodeDecoder(
            in_node_features=self.latent_node_features,
            out_node_features=self.out_node_features,
            depths=self.depths[::-1],
            fnns_depth=self.fnns_depth,
            fnns_width=self.fnns_width,
            activation=self.activation,
            dropout=self.dropout,
            aggr=self.aggr,
            dim=self.dim,
            scalar_rel_pos=self.scalar_rel_pos,
        )

    @property
    def num_fields(self) -> int:
        return self.out_node_features

    def reset_parameters(self):
        modules = [
            module for module in self.children() if hasattr(module, "reset_parameters")
        ]
        for module in modules:
            module.reset_parameters()

    def encode(
        self,
        graph: Graph,
        v: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.LongTensor],
        list[torch.LongTensor],
    ]:
        # Encode the conditions and the input edge features into latent node and edge features
        c_latent_list, e_latent_list, edge_index_list, batch_list = self.cond_encoder(
            graph,
            torch.cat(
                [
                    f
                    for f in [graph.get("loc"), graph.get("glob"), graph.get("omega")]
                    if f is not None
                ],
                dim=1,
            ),
            torch.cat(
                [
                    f
                    for f in [graph.get("edge_attr"), graph.get("edge_cond")]
                    if f is not None
                ],
                dim=1,
            ),
            graph.edge_index,
        )
        # Encode the input node features into latent node features
        graph.edge_index = edge_index_list[0]
        graph.batch = batch_list[0]
        v_latent, mean, logvar = self.node_encoder(
            graph,
            v,
            [c.clone() for c in c_latent_list],
            [e.clone() for e in e_latent_list],
        )
        return (
            v_latent,
            mean,
            logvar,
            c_latent_list,
            e_latent_list,
            edge_index_list,
            batch_list,
        )

    def decode(
        self,
        graph: Graph,
        v_latent: torch.Tensor,
        c_latent_list: list[torch.Tensor],
        e_latent_list: list[torch.Tensor],
        edge_index_list: list[torch.Tensor],
        batch_list: list[torch.LongTensor],
        dirichlet_mask: torch.Tensor = None,
        v_0: torch.Tensor = None,
    ) -> torch.Tensor:
        # Decode the latent node features (together with the condition node features) into the output node features
        v = self.node_decoder(
            graph, v_latent, c_latent_list, e_latent_list, edge_index_list, batch_list
        )
        # Apply the dirichlet boundary condition (if it exists)
        if dirichlet_mask is not None:
            assert v_0 is not None, (
                "The initial condition (`v_0`) must be provided if the dirichlet boundary condition (`dirichlet_mask`) exists."
            )
            v = torch.where(dirichlet_mask, v_0, v)
        return v

    def forward(
        self,
        graph: Graph,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            v_latent,
            mean,
            logvar,
            c_latent_list,
            e_latent_list,
            edge_index_list,
            batch_list,
        ) = self.encode(graph, graph.field)
        dirichlet_mask = (
            graph.dirichlet_mask if hasattr(graph, "dirichlet_mask") else None
        )
        return (
            self.decode(
                graph,
                v_latent,
                c_latent_list,
                e_latent_list,
                edge_index_list,
                batch_list,
                dirichlet_mask,
                graph.field[:, -self.num_fields :],
            ),
            v_latent,
            mean,
            logvar,
        )

    @torch.no_grad()
    def sample(
        self,
        graph: Graph,
        dirichlet_values: torch.Tensor = None,
        mean: Union[float, torch.Tensor] = 0.0,
        std: Union[float, torch.Tensor] = 1.0,
        seed: int = None,
    ) -> torch.Tensor:
        """Generates samples by sampling from a Gaussian distribution in the latent space.

        Args:
            graph (Graph): The input graph.
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary condition values. Defaults to None.
            mean (Union[float, torch.Tensor], optional): The mean of the Gaussian distribution. Defaults to 0.
            std (Union[float, torch.Tensor], optional): The standard deviation of the Gaussian distribution. Defaults to 1.
            seed (int, optional): The random seed. Defaults to None.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        num_nodes = getattr(graph, f"pos_{len(self.depths)}").size(0)
        if not hasattr(graph, "ptr"):
            graph = Batch.from_data_list([graph])
        graph.to(self.device)
        if isinstance(mean, float):
            mean = mean * torch.ones(
                num_nodes, self.latent_node_features, device=self.device
            )
        if isinstance(std, float):
            std = std * torch.ones(
                num_nodes, self.latent_node_features, device=self.device
            )
        v_latent = mean + std * torch.randn_like(mean)
        if self.norm_latents:
            v_latent = self.node_encoder.bn(v_latent)
        c = torch.cat(
            [
                f
                for f in [graph.get("loc"), graph.get("glob"), graph.get("omega")]
                if f is not None
            ],
            dim=1,
        )
        e = torch.cat(
            [
                f
                for f in [graph.get("edge_attr"), graph.get("edge_cond")]
                if f is not None
            ],
            dim=1,
        )
        c_latent_list, e_latent_list, edge_index_list, batch_list = self.cond_encoder(
            graph, c, e, graph.edge_index
        )
        if dirichlet_values is not None:
            assert hasattr(graph, "dirichlet_mask"), (
                "The graph must have a `dirichlet_mask` attribute"
            )
            dirichlet_mask = graph.dirichlet_mask.to(self.device)
            dirichlet_values = dirichlet_values.to(self.device)
        else:
            dirichlet_mask = None
        return self.decode(
            graph,
            v_latent,
            c_latent_list,
            e_latent_list,
            edge_index_list,
            batch_list,
            dirichlet_mask,
            dirichlet_values,
        )

    @torch.no_grad()
    def transform(self, graph: Graph) -> Graph:
        """Generates the latent features needed for latent diffusion model training."""
        self.eval()
        # Move the graph to the device
        device = graph.pos.device
        self.to(device)
        # Encode the input node features into latent node features
        if hasattr(graph, "field") and graph.field is not None:
            graph.x_latent = self.encode(graph, graph.field)[0]
        # Encode the target node and edge features into latent node and edge features
        if hasattr(graph, "target") and graph.target is not None:
            (
                graph.x_latent_target,
                _,
                _,
                c_latent_list,
                e_latent_list,
                edge_index_list,
                batch_list,
            ) = self.encode(graph, graph.target)
            graph.c_latent = c_latent_list[-1].clone()
            graph.e_latent = e_latent_list[-1].clone()
            graph.edge_index = edge_index_list[-1].clone()
            graph.batch = batch_list[-1].clone()
        return graph

    @torch.no_grad()
    def sample_n(
        self,
        num_samples: int,
        graph: Graph,
        dirichlet_values: torch.Tensor = None,
        batch_size: int = 0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Sample `num_samples` samples from the model.

        Args:
            num_samples (int): The number of samples.
            graph (Graph): The graph.
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary conditions. If `None`, no Dirichlet boundary conditions are applied. Defaults to `None`.
            batch_size (int, optional): Number of samples to generate in parallel. If `batch_size < 2`, the samples are generated one by one. Defaults to `0`.

        Returns:
            torch.Tensor: The samples. Dimension: (num_nodes, num_samples, num_fields
        """
        samples = []
        # Create (num_samples // num_workers) mini-batches with the same graph repeated num_workers times
        if batch_size > 1:
            collater = Collater()
            num_evals = num_samples // batch_size + (num_samples % batch_size > 0)
            for _ in tqdm(
                range(num_evals),
                desc=f"Generating {num_samples} samples",
                leave=False,
                position=0,
            ):
                current_batch_size = min(batch_size, num_samples - len(samples))
                batch = collater.collate(
                    [deepcopy(graph) for _ in range(current_batch_size)]
                )
                # Sample
                sample = self.sample(
                    batch,
                    dirichlet_values.repeat(current_batch_size, 1)
                    if dirichlet_values is not None
                    else None,
                    *args,
                    **kwargs,
                )
                # Split base on the batch index
                sample = torch.stack(sample.chunk(current_batch_size, dim=0), dim=1)
                samples.append(sample)
            return torch.cat(samples, dim=1)
        else:
            for _ in tqdm(
                range(num_samples),
                desc=f"Generating {num_samples} samples",
                leave=False,
                position=0,
            ):
                sample = self.sample(graph, dirichlet_values, *args, **kwargs)
                samples.append(sample)
            return torch.stack(
                samples, dim=1
            )  # Dimension: (num_nodes, num_samples, num_fields)
