import math
import torch
from torch import nn
from typing import Tuple, Callable
from torch_geometric.utils import scatter

from ..graph import Graph


class SinusoidalPositionEmbedding(nn.Module):
    r"""Defines a sinusoidal embedding like in the paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762).

    Args:
        dim (int): The dimension of the embedding.
        theta (float, optional): The theta parameter of the sinusoidal embedding. Defaults to 10000.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        self.theta = theta

    def forward(
        self,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the embedding of position `r`."""
        device = r.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -emb
        )  # Dimensions: [dim/2]
        emb = r.unsqueeze(-1) * emb.unsqueeze(0)  # Dimensions: [batch_size, dim/2]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Dimensions: [batch_size, dim]
        return emb


class SinusoidalTimeEmbedding(nn.Module):
    r"""Defines a sinusoidal embedding for continuos time in [t_min, t_max].

    Args:
        dim (int): The dimension of the embedding.
        t_min (float): The minimum time.
        t_max (float): The maximum time.
    """

    def __init__(
        self,
        dim: int,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        self.t_min = t_min
        self.t_max = t_max

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the embedding of time `t`."""
        assert t.dim() == 1, "Time must be one-dimensional."
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        t = (t - self.t_min) / (self.t_max - self.t_min) * 1000.0
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FNN(nn.Module):
    """Feedforward Neural Network with dropout after the activation function.

    Args:
        in_features (int): Number of input features.
        layers_width (Tuple[int]): Width of the hidden layers.
        activation (Callable): Activation function. Default: nn.ReLU.
        dropout (float): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        in_features: int,
        layers_width: Tuple[int],
        activation: Callable = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = len(layers_width)
        self.layers = nn.Sequential()
        # Input layer
        self.layers.add_module("linear_0", nn.Linear(in_features, layers_width[0]))
        self.layers.add_module("activation_0", activation())
        # Hidden and output layers
        for l in range(1, self.num_layers):
            self.layers.add_module(
                "linear_" + str(l), nn.Linear(layers_width[l - 1], layers_width[l])
            )
            self.layers.add_module("activation_" + str(l), activation())
        # Dropout layer
        if dropout > 0:
            self.layers.add_module("dropout", nn.Dropout(dropout))

    def reset_parameters(self):
        for item in self.layers:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InteractionNetwork(nn.Module):
    """Interaction Network from Battaglia et al. (2016) (https://arxiv.org/abs/1612.00222) with diffusion-step embeddings.

    Args:
        in_node_features (int): Number of input node features.
        in_edge_features (int): Number of input edge features.
        out_node_features (int): Number of output node features.
        out_edge_features (int): Number of output edge features.
        fnns_depth (int): Number of layers in the FNNs.
        fnns_width (int): Width of the FNNs.
        emb_features (int): Number of diffusion-step embedding features. If 0, no diffusion-step embedding is used. Default: 0.
        activation (nn.Module): Activation function. Default: nn.ReLU.
        dropout (float): Dropout probability. Default: 0.0.
        aggr (str): Aggregation operator. Default: 'mean'.
    """

    def __init__(
        self,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int,
        out_edge_features: int,
        fnns_depth: int,
        fnns_width: int,
        emb_features: int = 0,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        aggr: str = "mean",
    ) -> None:
        # Validate inputs
        assert in_node_features > 0, (
            "The number of input node features must be positive"
        )
        assert in_edge_features > 0, (
            "The number of input edge features must be positive"
        )
        assert out_node_features > 0, (
            "The number of output node features must be positive"
        )
        assert out_edge_features > 0, (
            "The number of output edge features must be positive"
        )
        assert fnns_depth >= 2, "The depth of the FNNs must be at least 2"
        assert fnns_width > 0, "The width of the FNNs must be positive"
        assert emb_features >= 0, (
            "The number of embedding features must be non-negative"
        )
        assert aggr in ("mean", "sum"), "Aggregation must be either 'mean' or 'sum'"
        super().__init__()
        # Projection of the diffusion-step embedding
        self.emb_features = emb_features
        if self.emb_features > 0:
            self.node_emb_linear = nn.Linear(emb_features, in_node_features)
        # Edge update function
        input_layer_width = in_edge_features + 2 * in_node_features
        hidden_layers_width = (fnns_depth - 1) * (fnns_width,)
        self.edge_layer_norm = nn.LayerNorm(input_layer_width)
        self.edge_fnn = FNN(
            in_features=input_layer_width,
            layers_width=hidden_layers_width + (out_edge_features,),
            activation=activation,
            dropout=dropout,
        )
        self.edge_linear = nn.Parameter(
            torch.zeros(in_edge_features, out_edge_features)
        )
        # Aggregation function
        self.aggr = aggr
        # Node update function
        input_layer_width = out_edge_features + in_node_features
        self.node_layer_norm = nn.LayerNorm(input_layer_width)
        self.node_fnn = FNN(
            in_features=input_layer_width,
            layers_width=hidden_layers_width + (out_node_features,),
            activation=activation,
            dropout=dropout,
        )
        self.node_linear = nn.Parameter(
            torch.zeros(in_node_features, out_node_features)
        )

    def reset_parameters(self):
        modules = [
            module for module in self.children() if hasattr(module, "reset_parameters")
        ]
        for module in modules:
            module.reset_parameters()

    def forward(
        self,
        v: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        emb: torch.Tensor = None,
    ) -> torch.Tensor:
        # Validate inputs
        if self.emb_features > 0:
            assert emb is not None, "An embedding must be provided"
            assert batch is not None, "A batch vector must be provided"
        row, col = edge_index
        # Project the diffusion-step embedding to the node embedding space
        if self.emb_features > 0:
            v += self.node_emb_linear(emb)[batch]  # Shape (num_nodes, in_node_features)
        # Edge update
        e = e.mm(self.edge_linear) + self.edge_fnn(
            self.edge_layer_norm(torch.cat((e, v[row], v[col]), dim=-1))
        )
        # Edge aggregation
        e_aggr = scatter(e, col, dim=0, dim_size=v.size(0), reduce=self.aggr)
        # Node update
        v = v.mm(self.node_linear) + self.node_fnn(
            self.node_layer_norm(torch.cat((e_aggr, v), dim=-1))
        )
        return v, e


class MeshDownMP(nn.Module):
    def __init__(
        self,
        scale: int,
        dim: int,
        in_node_features: int,
        fnn_depth: int,
        fnn_width: int,
        activation: nn.Module = nn.SELU,
        dropout: float = 0.0,
        aggr: str = "mean",
        encode_edges: bool = True,
        scalar_rel_pos: bool = True,
    ) -> None:
        """MeshDownMP graph-pooling layer.

        Args:
            scale (int):            The scale/level of the graph. MP from scale `scale` to `scale+1`.
            dim (int):              The dimension of the graph.
            in_node_features (int): The number of input node features.
            fnn_depth (int):        The depth of the FNNs.
            fnn_width (int):        The width of the FNNs.
            activation (nn.Module): The activation function. Default: nn.SELU.
            dropout (float):        The dropout probability. Default: 0.0.
            aggr (str):             The aggregation operator. Default: 'mean'.
            encode_edges (bool):    Whether to encode the edges. Default: True.
            scalar_rel_pos (bool):  Whether to use scalar relative positions for the message passing. Default: True.
        """

        # Validate inputs
        assert scale >= 0, "level must be greater than or equal to 0."
        assert dim in (1, 2, 3), "dim must be either 1, 2 or 3."
        assert in_node_features > 0, "in_features must be greater than 0."
        assert fnn_depth >= 2, "fnn_depth must be greater than or equal to 2."
        assert fnn_width > 0, "fnn_width must be greater than 0."
        assert aggr in ("mean", "sum"), "aggr must be either 'mean' or 'sum'."
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.aggr = aggr
        self.encode_edges = encode_edges
        # Resolution/scale/level of the HR and LR graphs
        self.hr_graph_idx = scale + 1
        self.lr_graph_idx = self.hr_graph_idx + 1
        # Linear layer to encode the higher to lower resolution edge features
        self.edge_encoder_hr_to_lr = nn.Linear(
            1 if scalar_rel_pos else dim,
            fnn_width,
        )
        # Layer norm
        self.layer_norm = nn.LayerNorm(fnn_width + in_node_features)
        # FNN
        self.fnn = FNN(
            in_features=fnn_width + in_node_features,
            layers_width=fnn_depth * (fnn_width,),
            activation=activation,
            dropout=dropout,
        )
        # Linear layer for encoding the lower-resolution edge features
        self.edge_encoder_lr = nn.Linear(dim, fnn_width) if self.encode_edges else None

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
    ) -> Tuple[Graph, torch.Tensor, torch.Tensor]:
        r"""Computes the forward pass of the MeshDownMP graph-pooling layer.

        Args:
            graph (Graph):    The graph object.
            v (torch.Tensor): The node features of the high-resolution graph.

        Returns:
            graph (Graph):    The graph object. The `edge_index` and `batch` is modified.
            v (torch.Tensor): The node features of the low-resolution graph.
            e (torch.Tensor): The edge features of the low-resolution graph.
        """
        # Get the needed variables
        idxHr_to_idxLr = getattr(
            graph, f"idx{self.hr_graph_idx}_to_idx{self.lr_graph_idx}"
        )
        e_hr_to_lr = getattr(graph, f"e_{self.hr_graph_idx}{self.lr_graph_idx}")
        batch_lr = getattr(graph, f"batch_{self.lr_graph_idx}")
        edge_index_lr = getattr(graph, f"edge_index_{self.lr_graph_idx}")
        edge_attr_lr = getattr(graph, f"edge_attr_{self.lr_graph_idx}")
        # Encode the edge features
        e_hr_to_lr = self.edge_encoder_hr_to_lr(e_hr_to_lr)
        # Appy edge model
        e_hr_to_lr = self.fnn(self.layer_norm(torch.cat([e_hr_to_lr, v], dim=-1)))
        # Aggregate the edge features
        v = scatter(e_hr_to_lr, idxHr_to_idxLr, dim=0, reduce=self.aggr)
        # Encode the lower-resolution edge features
        e = self.edge_encoder_lr(edge_attr_lr) if self.encode_edges else None
        # Update the batch
        graph.batch = batch_lr
        # Update the edge index
        graph.edge_index = edge_index_lr
        return graph, v, e


class MeshUpMP(nn.Module):
    def __init__(
        self,
        scale: int,
        dim: int,
        in_features: int,
        fnn_depth: int,
        fnn_width: int,
        activation: nn.Module = nn.SELU,
        dropout: float = 0.0,
        skip_connection: bool = True,
        scalar_rel_pos: bool = True,
    ) -> None:
        """MeshUpMP graph-unpooling layer.

        Args:
            scale (int):            The scale/level of the graph. MP from scale `scale` to `scale-1`.
            dim (int):              The dimension of the graph.
            in_features (int):      The number of input node features.
            fnn_depth (int):        The depth of the FNNs.
            fnn_width (int):        The width of the FNNs.
            activation (nn.Module): The activation function. Default: nn.SELU.
            dropout (float):        The dropout probability. Default: 0.0.
            skip_connection (bool): Whether to use skip connections. Default: True.
            scalar_rel_pos (bool):  Whether to use scalar relative positions for the message passing. Default: True
        """
        # Validate inputs
        assert scale >= 0, "level must be greater than or equal to 0."
        assert in_features > 0, "in_features must be greater than 0."
        assert dim in (1, 2, 3), "dim must be either 1, 2 or 3."
        assert fnn_depth >= 2, "fnn_depth must be greater than or equal to 2."
        assert fnn_width > 0, "fnn_width must be greater than 0."
        super().__init__()
        self.skip_connection = skip_connection
        in_msg_features = (
            (fnn_width + 2 * in_features)
            if skip_connection
            else (fnn_width + in_features)
        )
        # Linear layer to encode the lower to higher resolution edge features
        self.edge_encoder_lr_to_hr = nn.Linear(1 if scalar_rel_pos else dim, fnn_width)
        # Layer norm
        self.layer_norm = nn.LayerNorm(in_msg_features)
        # FNN
        self.fnn = FNN(
            in_features=in_msg_features,
            layers_width=fnn_depth * (fnn_width,),
            activation=activation,
            dropout=dropout,
        )
        # Resolution/scale/level of the HR and LR graphs
        self.lr_graph_idx = scale + 2
        self.hr_graph_idx = self.lr_graph_idx - 1
        self.reset_parameters()

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
        edge_index_skip: torch.LongTensor,
        v_skip: torch.Tensor = None,
        batch_skip: torch.LongTensor = None,
    ) -> Tuple[Graph, torch.Tensor]:
        """Computes the forward pass of the UpMeshMP graph-unpooling layer.

        Args:
            graph (Graph):                      The graph object.
            v (torch.Tensor):                   The node features of the low-resolution graph.
            edge_index_skip (torch.LongTensor): The edge indices of the high-resolution graph.
            v_skip (torch.Tensor):              The node features of the high-resolution graph. Default: None.
            batch_skip (torch.LongTensor):      The batch indices of the high-resolution graph. Default: None.

        Returns:
            graph (Graph):    The graph object. The `edge_index` and `batch` is modified.
            v (torch.Tensor): The node features of the high-resolution graph.
        """
        # Get the needed variables
        idxHr_to_idxLr = getattr(
            graph, f"idx{self.hr_graph_idx}_to_idx{self.lr_graph_idx}"
        )
        e_lr_to_hr = -getattr(graph, f"e_{self.hr_graph_idx}{self.lr_graph_idx}")
        # Encode the edge features
        e_lr_to_hr = self.edge_encoder_lr_to_hr(e_lr_to_hr)
        # Edge-model
        msg_features = (
            torch.cat([e_lr_to_hr, v[idxHr_to_idxLr], v_skip], dim=-1)
            if self.skip_connection
            else torch.cat([e_lr_to_hr, v[idxHr_to_idxLr]], dim=-1)
        )
        v = self.fnn(self.layer_norm(msg_features))
        graph.edge_index, graph.batch = edge_index_skip, batch_skip
        return graph, v
