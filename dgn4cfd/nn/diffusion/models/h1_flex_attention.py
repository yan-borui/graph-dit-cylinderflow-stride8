"""Exact H1 FlexAttention with 64-frame node blocks and cached graph metadata."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch


@dataclass
class FlexLayout:
    order: torch.Tensor
    inverse: torch.Tensor
    block_mask: Any
    nonempty_blocks: int
    full_blocks: int


@lru_cache(maxsize=1)
def compiled_attention():
    """Share compiled kernels across all layers, model instances and graph values."""
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: FlexLayout,
) -> torch.Tensor:
    """Attend in permuted order, returning the original [B, H, tokens, head] axes."""
    if query.device.type != "cuda" or query.dtype != torch.float32:
        raise ValueError("the H1 Flex implementation requires CUDA IEEE FP32")
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError(
            "H1 Flex requires highest matmul precision, with TF32 disabled"
        )
    result = compiled_attention()(
        query,
        key,
        value,
        block_mask=layout.block_mask,
        kernel_options={
            "BACKEND": "TRITON",
            "PRESCALE_QK": False,
            "fwd_BLOCK_M": 32,
            "fwd_BLOCK_N": 32,
            "bwd_BLOCK_M1": 32,
            "bwd_BLOCK_N1": 32,
            "bwd_BLOCK_M2": 32,
            "bwd_BLOCK_N2": 32,
        },
    )
    return result.index_select(2, layout.inverse)


def _block_rows(selected: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    counts = selected.sum(axis=-1).astype(np.int32)
    # BlockMask's transpose interprets the last axis as the full column count.
    capacity = selected.shape[-1]
    indices = np.zeros((*selected.shape[:-1], capacity), dtype=np.int32)
    for batch in range(len(selected)):
        for row in range(selected.shape[1]):
            chosen = np.flatnonzero(selected[batch, row])
            indices[batch, row, : len(chosen)] = chosen
    return torch.from_numpy(counts[:, None]), torch.from_numpy(indices[:, None])


class FlexLayoutCache:
    """Cache graph-only metadata; never retain learned Q/K/V across forwards."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self.layouts: OrderedDict[tuple, FlexLayout] = OrderedDict()
        self.last_tensor: torch.Tensor | None = None
        self.last_version: int | None = None
        self.last_layout: FlexLayout | None = None
        self.builds = 0
        self.hits = 0

    def get(self, graph_hops: torch.Tensor, total_slots: int) -> FlexLayout:
        if total_slots != 65:
            raise ValueError("H1 Flex requires one initial frame and 64 future frames")
        # Inference tensors have no version counter; compare their contents below.
        version = None if graph_hops.is_inference() else graph_hops._version
        if (
            graph_hops is self.last_tensor
            and version is not None
            and version == self.last_version
        ):
            self.hits += 1
            return self.last_layout
        hops = graph_hops.detach().cpu().numpy()
        allowed = (hops >= 0) & (hops <= 1)
        key = (allowed.shape, graph_hops.device, total_slots, allowed.tobytes())
        layout = self.layouts.get(key)
        if layout is None:
            layout = self._build(allowed, graph_hops.device, total_slots)
            self.layouts[key] = layout
            self.builds += 1
            if len(self.layouts) > self.capacity:
                self.layouts.popitem(last=False)
        else:
            self.layouts.move_to_end(key)
            self.hits += 1
        self.last_tensor, self.last_version, self.last_layout = (
            graph_hops,
            version,
            layout,
        )
        return layout

    @staticmethod
    def _build(
        allowed: np.ndarray, device: torch.device, total_slots: int
    ) -> FlexLayout:
        from torch.nn.attention.flex_attention import BlockMask

        _, nodes, _ = allowed.shape
        block_size = 64
        tokens = nodes * total_slots
        original = np.arange(tokens).reshape(total_slots, nodes)
        order = np.concatenate((original[1:].T.reshape(-1), original[0]))
        token_nodes = order % nodes
        blocks = (tokens + block_size - 1) // block_size
        counts = np.zeros((blocks, nodes), dtype=np.int32)
        np.add.at(counts, (np.arange(tokens) // block_size, token_nodes), 1)
        pairs = counts @ allowed.astype(np.int32) @ counts.T
        full = pairs == block_size**2
        partial = (pairs > 0) & ~full
        partial_counts, partial_indices = _block_rows(partial)
        full_counts, full_indices = _block_rows(full)
        allowed_device = torch.from_numpy(allowed.copy()).to(device)
        token_nodes_device = torch.from_numpy(token_nodes).to(device)

        def mask_mod(batch, head, query_index, key_index):
            return allowed_device[
                batch, token_nodes_device[query_index], token_nodes_device[key_index]
            ]

        # Construct transposed backward metadata on the host once per graph.
        mask = BlockMask.from_kv_blocks(
            partial_counts,
            partial_indices,
            full_counts,
            full_indices,
            BLOCK_SIZE=block_size,
            mask_mod=mask_mod,
            seq_lengths=(tokens, tokens),
        ).to(device)
        return FlexLayout(
            torch.from_numpy(order.copy()).to(device),
            torch.from_numpy(np.argsort(order)).to(device),
            mask,
            int(np.count_nonzero(pairs)),
            int(np.count_nonzero(full)),
        )
