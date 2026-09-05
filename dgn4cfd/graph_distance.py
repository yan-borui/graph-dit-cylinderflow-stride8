"""Immutable graph-relation sidecars for graph-aware attention.

Distances are measured on the latent graph with Euclidean edge lengths.  The
all-pairs shortest-path matrix is normalised to ``[0, 1]`` per graph; pairs in
different connected components receive distance one.  The representation is a
soft attention prior and never removes global visibility.  The same sidecar also
stores unweighted hop distance for hard H1/H2 visibility masks; disconnected
pairs use ``-1``.
"""

from __future__ import annotations

import hashlib
import heapq
import os
import uuid
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch


GRAPH_DISTANCE_CACHE_FORMAT_V1 = "dgn4cfd.graph_distance_cache.v1"
GRAPH_DISTANCE_CACHE_FORMAT = "dgn4cfd.graph_distance_cache.v2"


def sha256_file(file_path: str | Path) -> str:
    """Return the SHA-256 identity of one file."""

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_sha256(
    positions: np.ndarray | torch.Tensor,
    edge_index: np.ndarray | torch.Tensor,
) -> str:
    """Hash the exact latent-node positions and stored graph connectivity."""

    position_array = np.ascontiguousarray(
        torch.as_tensor(positions).detach().cpu().numpy(), dtype=np.float32
    )
    edge_array = np.ascontiguousarray(
        torch.as_tensor(edge_index).detach().cpu().numpy(), dtype=np.int64
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(position_array.shape, dtype=np.int64).tobytes())
    digest.update(position_array.tobytes())
    digest.update(np.asarray(edge_array.shape, dtype=np.int64).tobytes())
    digest.update(edge_array.tobytes())
    return digest.hexdigest()


def normalised_shortest_path_distance(
    positions: np.ndarray | torch.Tensor,
    edge_index: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Compute undirected, Euclidean-weighted all-pairs graph distances.

    Args:
        positions: Latent-node coordinates with shape ``[N, 2]``.
        edge_index: Stored graph edges with shape ``[2, E]``.

    Returns:
        A symmetric ``float32`` matrix with shape ``[N, N]``.  Its diagonal is
        zero, its largest connected finite distance is one, and disconnected
        pairs are assigned one.
    """

    position_array = np.asarray(
        torch.as_tensor(positions).detach().cpu().numpy(), dtype=np.float64
    )
    edge_array = np.asarray(
        torch.as_tensor(edge_index).detach().cpu().numpy(), dtype=np.int64
    )
    if position_array.ndim != 2 or position_array.shape[1] != 2:
        raise ValueError("positions must have shape [N, 2]")
    if not np.isfinite(position_array).all():
        raise ValueError("positions must be finite")
    if edge_array.ndim != 2 or edge_array.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    num_nodes = int(position_array.shape[0])
    if num_nodes < 1:
        raise ValueError("the graph must contain at least one node")
    if edge_array.size and (
        int(edge_array.min()) < 0 or int(edge_array.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains a node outside the graph")

    neighbours: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    for source, target in edge_array.T:
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        weight = float(
            np.linalg.norm(position_array[source_index] - position_array[target_index])
        )
        previous = neighbours[source_index].get(target_index)
        if previous is None or weight < previous:
            neighbours[source_index][target_index] = weight
            neighbours[target_index][source_index] = weight

    distances = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    for source in range(num_nodes):
        distances[source, source] = 0.0
        queue = [(0.0, source)]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance > distances[source, node]:
                continue
            for neighbour, weight in neighbours[node].items():
                candidate = current_distance + weight
                if candidate < distances[source, neighbour]:
                    distances[source, neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))

    finite = np.isfinite(distances)
    max_finite = float(distances[finite].max())
    if max_finite > 0.0:
        distances[finite] /= max_finite
    distances[~finite] = 1.0
    np.fill_diagonal(distances, 0.0)
    result = np.clip(distances, 0.0, 1.0).astype(np.float32)
    validate_normalised_distance(result, num_nodes=num_nodes)
    return result


def unweighted_shortest_path_hops(
    edge_index: np.ndarray | torch.Tensor,
    *,
    num_nodes: int,
) -> np.ndarray:
    """Return undirected all-pairs hop distance as ``int32[N,N]``.

    Self distance is zero and disconnected pairs are represented by ``-1``.
    Duplicate and directed stored edges are collapsed into one undirected graph.
    """

    edge_array = np.asarray(
        torch.as_tensor(edge_index).detach().cpu().numpy(), dtype=np.int64
    )
    if num_nodes < 1:
        raise ValueError("the graph must contain at least one node")
    if edge_array.ndim != 2 or edge_array.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_array.size and (
        int(edge_array.min()) < 0 or int(edge_array.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains a node outside the graph")

    neighbours: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in edge_array.T:
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        neighbours[source_index].add(target_index)
        neighbours[target_index].add(source_index)

    hops = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
    for source in range(num_nodes):
        hops[source, source] = 0
        frontier = deque([source])
        while frontier:
            node = frontier.popleft()
            next_hop = int(hops[source, node]) + 1
            for neighbour in neighbours[node]:
                if hops[source, neighbour] == -1:
                    hops[source, neighbour] = next_hop
                    frontier.append(neighbour)
    validate_hop_distance(hops, num_nodes=num_nodes)
    return hops


def validate_normalised_distance(
    distance: np.ndarray | torch.Tensor,
    *,
    num_nodes: int | None = None,
    atol: float = 1e-6,
) -> None:
    """Validate the sidecar's symmetry, diagonal, range, and finiteness."""

    array = np.asarray(
        torch.as_tensor(distance).detach().cpu().numpy(), dtype=np.float64
    )
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("graph distance must have shape [N, N]")
    if num_nodes is not None and array.shape != (num_nodes, num_nodes):
        raise ValueError("graph distance node count does not match the graph")
    if not np.isfinite(array).all():
        raise ValueError("graph distance must be finite")
    if float(array.min()) < -atol or float(array.max()) > 1.0 + atol:
        raise ValueError("graph distance must lie in [0, 1]")
    if not np.allclose(array, array.T, atol=atol, rtol=0.0):
        raise ValueError("graph distance must be symmetric")
    if not np.allclose(np.diag(array), 0.0, atol=atol, rtol=0.0):
        raise ValueError("graph distance diagonal must be zero")


def validate_hop_distance(
    hops: np.ndarray | torch.Tensor,
    *,
    num_nodes: int | None = None,
) -> None:
    """Validate an integer symmetric hop matrix with ``-1`` disconnections."""

    tensor = torch.as_tensor(hops)
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("graph hops must have shape [N, N]")
    if num_nodes is not None and tensor.shape != (num_nodes, num_nodes):
        raise ValueError("graph hops node count does not match the graph")
    if tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("graph hops must have integer dtype")
    if bool(torch.any(tensor < -1).item()):
        raise ValueError("graph hops must use -1 for disconnected pairs")
    if not bool(torch.equal(tensor, tensor.transpose(0, 1))):
        raise ValueError("graph hops must be symmetric")
    if bool(torch.any(torch.diagonal(tensor) != 0).item()):
        raise ValueError("graph hops diagonal must be zero")


def build_graph_distance_cache(
    latent_cache_path: str | Path,
    output_path: str | Path,
    *,
    metadata: dict[str, str | int | float] | None = None,
) -> dict[str, object]:
    """Build an immutable sidecar for every graph in one latent cache."""

    latent_path = Path(latent_cache_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    if not latent_path.is_file():
        raise FileNotFoundError(latent_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    latent_hash = sha256_file(latent_path)
    temporary = target_path.with_name(f".{target_path.name}.partial-{uuid.uuid4().hex}")
    node_counts: list[int] = []
    try:
        with (
            h5py.File(latent_path, "r") as source,
            h5py.File(temporary, "w", libver="latest") as target,
        ):
            sim_indices = np.asarray(source["sim_indices"], dtype=np.int64)
            target.attrs["format"] = GRAPH_DISTANCE_CACHE_FORMAT
            target.attrs["schema_version"] = 2
            target.attrs["latent_cache_sha256"] = latent_hash
            for key, value in (metadata or {}).items():
                target.attrs[key] = value
            target.create_dataset("sim_indices", data=sim_indices)
            for sim_idx in sim_indices:
                group_name = f"sim_{int(sim_idx):05d}"
                source_group = source[group_name]
                positions = np.asarray(source_group["positions"], dtype=np.float32)
                edge_index = np.asarray(source_group["edge_index"], dtype=np.int64)
                distance = normalised_shortest_path_distance(positions, edge_index)
                hops = unweighted_shortest_path_hops(
                    edge_index, num_nodes=int(positions.shape[0])
                )
                graph_hash = graph_sha256(positions, edge_index)
                group = target.create_group(group_name)
                group.attrs["sim_idx"] = int(sim_idx)
                group.attrs["graph_sha256"] = graph_hash
                group.attrs["latent_nodes"] = int(positions.shape[0])
                group.create_dataset(
                    "distance", data=distance, compression="lzf", shuffle=True
                )
                group.create_dataset("hops", data=hops, compression="lzf", shuffle=True)
                node_counts.append(int(positions.shape[0]))
            target.flush()
        os.replace(temporary, target_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "format": GRAPH_DISTANCE_CACHE_FORMAT,
        "output": str(target_path),
        "output_sha256": sha256_file(target_path),
        "latent_cache": str(latent_path),
        "latent_cache_sha256": latent_hash,
        "simulation_count": len(node_counts),
        "latent_nodes_min": min(node_counts),
        "latent_nodes_max": max(node_counts),
    }
