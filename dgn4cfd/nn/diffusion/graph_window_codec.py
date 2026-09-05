"""Frozen UVP autoencoder adapter for Graph-Window DiT rollouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn

from ...graph import Graph
from ..models.vgae import VGAE


# CUDA scatter reductions can change order when identical graphs are packed into a
# larger batch.  The recorded sim-43 differential probe observed <=4.88e-5 in
# latent max error and <=4.33e-7 after decoding, so retain a conservative margin
# while continuing to reject non-finite or materially different encodings.
POSTERIOR_BATCH_ATOL = 1e-4


def posterior_batch_delta_is_acceptable(delta: float) -> bool:
    """Check graph-batched against one-frame posterior-mean agreement."""

    return math.isfinite(delta) and 0.0 <= delta <= POSTERIOR_BATCH_ATOL


@dataclass
class LatentGraphContext:
    """Static AE decoder context and coarsest graph tensors for one CFD mesh."""

    physical_graph: Graph
    c_latent_list: list[torch.Tensor]
    e_latent_list: list[torch.Tensor]
    edge_index_list: list[torch.Tensor]
    batch_list: list[torch.Tensor]
    node_context: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor

    @property
    def num_latent_nodes(self) -> int:
        return int(self.node_context.size(0))

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Return an unpadded batch-of-one Graph-Window condition dictionary."""

        return {
            "node_context": self.node_context.unsqueeze(0),
            "positions": self.positions.unsqueeze(0),
            "edge_index": self.edge_index.unsqueeze(0),
            "edge_mask": torch.ones(
                1,
                self.edge_index.size(1),
                dtype=torch.bool,
                device=self.edge_index.device,
            ),
            "node_mask": torch.ones(
                1,
                self.node_context.size(0),
                dtype=torch.bool,
                device=self.node_context.device,
            ),
        }


@dataclass
class CodecCycle:
    """One latent decode, UV-boundary writeback, and posterior-mean re-encode."""

    uvp: torch.Tensor
    pre_boundary_uvp: torch.Tensor
    reencoded_latent: torch.Tensor


class FrozenUVPLatentCodec(nn.Module):
    """Own and freeze the existing joint-UVP variational graph autoencoder."""

    def __init__(
        self,
        autoencoder_checkpoint: str,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.autoencoder = VGAE(
            checkpoint=autoencoder_checkpoint,
            device=self.device,
        )
        self.autoencoder.eval()
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad = False
        self.latent_features = int(self.autoencoder.arch["latent_node_features"])
        self.condition_features = int(self.autoencoder.arch["fnns_width"])
        self.num_scales = len(self.autoencoder.arch["depths"])

    @staticmethod
    def _ensure_batch(graph: Graph) -> None:
        if graph.batch is None:
            graph.batch = torch.zeros(
                graph.num_nodes, dtype=torch.long, device=graph.pos.device
            )

    @torch.no_grad()
    def encode_field(
        self, graph: Graph, field: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Return the posterior mean and decoder context for one scaled UVP field."""

        self.autoencoder.eval()
        work_graph = deepcopy(graph).to(self.device)
        self._ensure_batch(work_graph)
        _, mean, _, c_list, e_list, edge_list, batch_list = self.autoencoder.encode(
            work_graph, field.to(self.device)
        )
        return mean, c_list, e_list, edge_list, batch_list

    @torch.no_grad()
    def encode_context(
        self, graph: Graph, current_uvp: torch.Tensor
    ) -> tuple[torch.Tensor, LatentGraphContext]:
        """Encode one current frame and capture all static graph information."""

        physical_graph = deepcopy(graph).to(self.device)
        self._ensure_batch(physical_graph)
        mean, c_list, e_list, edge_list, batch_list = self.encode_field(
            physical_graph, current_uvp
        )
        position_name = f"pos_{self.num_scales}"
        if not hasattr(physical_graph, position_name):
            raise ValueError(f"transformed graph has no {position_name}")
        positions = getattr(physical_graph, position_name).clone()
        context = LatentGraphContext(
            physical_graph=physical_graph,
            c_latent_list=[value.clone() for value in c_list],
            e_latent_list=[value.clone() for value in e_list],
            edge_index_list=[value.clone() for value in edge_list],
            batch_list=[value.clone() for value in batch_list],
            node_context=c_list[-1].clone(),
            positions=positions,
            edge_index=edge_list[-1].clone(),
        )
        return mean, context

    @torch.no_grad()
    def decode_writeback_reencode(
        self,
        raw_latent: torch.Tensor,
        context: LatentGraphContext,
        current_uvp: torch.Tensor,
    ) -> CodecCycle:
        """Decode one raw latent, write UV boundaries, and re-encode its mean."""

        self.autoencoder.eval()
        if raw_latent.ndim == 3:
            if raw_latent.size(0) != 1:
                raise ValueError("codec currently handles one graph per cycle")
            raw_latent = raw_latent[0]
        if raw_latent.ndim != 2 or raw_latent.shape != (
            context.num_latent_nodes,
            self.latent_features,
        ):
            raise ValueError("raw_latent has the wrong coarsest-graph shape")
        decode_graph = deepcopy(context.physical_graph)
        pre_boundary = self.autoencoder.decode(
            decode_graph,
            raw_latent.to(self.device),
            [value.clone() for value in context.c_latent_list],
            [value.clone() for value in context.e_latent_list],
            [value.clone() for value in context.edge_index_list],
            [value.clone() for value in context.batch_list],
            None,
            None,
        )
        current_uvp = current_uvp.to(self.device)
        if current_uvp.shape != pre_boundary.shape:
            raise ValueError("current UVP shape differs from decoded UVP")
        if hasattr(decode_graph, "dirichlet_mask"):
            mask = decode_graph.dirichlet_mask.clone().bool()
            if mask.shape != pre_boundary.shape or mask.size(1) != 3:
                raise ValueError("UVP Dirichlet mask must have shape [N, 3]")
            mask[:, 2] = False
            uvp = torch.where(mask, current_uvp, pre_boundary)
        else:
            uvp = pre_boundary
        reencoded, _, _, _, _ = self.encode_field(context.physical_graph, uvp)
        return CodecCycle(
            uvp=uvp,
            pre_boundary_uvp=pre_boundary,
            reencoded_latent=reencoded,
        )
