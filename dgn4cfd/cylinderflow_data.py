"""Raw MeshGraphNets CylinderFlow graph data and normalization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch_geometric.utils import coalesce
from torchvision import transforms

from .graph import Graph
from .transforms import AddDirichletMask, Copy, MeshCoarsening, ScaleEdgeAttr


CYLINDERFLOW_FORMAT = "dgn4cfd.mgn_cylinderflow_raw600.v1"
CYLINDERFLOW_TEMPORAL_STRIDE_FORMAT = "dgn4cfd.mgn_cylinderflow_temporal_stride.v1"
SUPPORTED_CYLINDERFLOW_FORMATS = {
    CYLINDERFLOW_FORMAT,
    CYLINDERFLOW_TEMPORAL_STRIDE_FORMAT,
}
CYLINDERFLOW_DATASET_FORMATS = {
    "cylinderflow_raw600": CYLINDERFLOW_FORMAT,
    "cylinderflow_temporal_stride": CYLINDERFLOW_TEMPORAL_STRIDE_FORMAT,
}
CYLINDERFLOW_FRAMES = 600


@dataclass(frozen=True)
class CylinderFlowDataContract:
    """Manifest-derived temporal and split semantics for CylinderFlow data."""

    manifest_format: str
    dataset_format: str
    frames: int
    temporal_stride: int
    raw_frame_dt: float
    frame_dt: float
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    @classmethod
    def from_manifest(cls, payload: dict) -> "CylinderFlowDataContract":
        manifest_format = str(payload.get("format", ""))
        if manifest_format not in SUPPORTED_CYLINDERFLOW_FORMATS:
            raise ValueError("unsupported CylinderFlow manifest format")
        dataset_format = next(
            name
            for name, value in CYLINDERFLOW_DATASET_FORMATS.items()
            if value == manifest_format
        )
        default_frames = (
            CYLINDERFLOW_FRAMES if manifest_format == CYLINDERFLOW_FORMAT else 0
        )
        frames = int(payload.get("frames", default_frames))
        temporal_stride = int(payload.get("temporal_stride", 1))
        raw_frame_dt = float(payload.get("raw_frame_dt", 0.01))
        frame_dt = float(payload.get("frame_dt", raw_frame_dt * temporal_stride))
        if frames < 1 or temporal_stride < 1:
            raise ValueError("CylinderFlow frame count and stride must be positive")
        if raw_frame_dt <= 0 or frame_dt <= 0:
            raise ValueError("CylinderFlow time steps must be positive")
        if not np.isclose(
            frame_dt,
            raw_frame_dt * temporal_stride,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("CylinderFlow frame_dt disagrees with raw dt and stride")
        splits = payload.get("splits", {})
        train_indices = tuple(int(value) for value in splits.get("train", ()))
        validation_indices = tuple(int(value) for value in splits.get("validation", ()))
        if len(set(train_indices)) != len(train_indices):
            raise ValueError("CylinderFlow Train contains duplicate trajectories")
        if len(set(validation_indices)) != len(validation_indices):
            raise ValueError("CylinderFlow Validation contains duplicate trajectories")
        if set(train_indices) & set(validation_indices):
            raise ValueError("CylinderFlow Train and Validation overlap")
        return cls(
            manifest_format=manifest_format,
            dataset_format=dataset_format,
            frames=frames,
            temporal_stride=temporal_stride,
            raw_frame_dt=raw_frame_dt,
            frame_dt=frame_dt,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )

    @property
    def validation_frames(self) -> tuple[int, int, int, int]:
        """Return the locked start/thirds/end frame audit positions."""

        if self.frames < 4:
            raise ValueError("four-frame VGAE validation requires at least four frames")
        return (0, self.frames // 3, 2 * self.frames // 3, self.frames - 1)


def triangle_cells_to_edge_index(cells: torch.Tensor) -> torch.Tensor:
    """Return every directed edge of triangular cells, including edge 2->0."""

    if cells.ndim != 2 or cells.size(1) != 3:
        raise ValueError("cells must have shape [C, 3]")
    cells = cells.to(dtype=torch.long)
    source = cells[:, (0, 1, 2)].reshape(-1)
    target = cells[:, (1, 2, 0)].reshape(-1)
    edge_index = torch.stack((source, target), dim=0)
    valid = (edge_index >= 0).all(dim=0) & (edge_index[0] != edge_index[1])
    edge_index = edge_index[:, valid]
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    return coalesce(edge_index, sort_by_row=False)


@dataclass(frozen=True)
class CylinderFlowNormalization:
    """Train-only affine statistics for UVP and the static inlet speed."""

    field_mean: tuple[float, float, float]
    field_std: tuple[float, float, float]
    inlet_mean: float
    inlet_std: float

    @classmethod
    def from_manifest(cls, path: str | Path) -> "CylinderFlowNormalization":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") not in SUPPORTED_CYLINDERFLOW_FORMATS:
            raise ValueError("unsupported CylinderFlow manifest format")
        stats = payload["train_only_normalization"]
        result = cls(
            field_mean=tuple(float(value) for value in stats["field_mean"]),
            field_std=tuple(float(value) for value in stats["field_std"]),
            inlet_mean=float(stats["inlet_mean"]),
            inlet_std=float(stats["inlet_std"]),
        )
        if len(result.field_mean) != 3 or len(result.field_std) != 3:
            raise ValueError("field normalization must contain u, v, and p")
        if min(result.field_std) <= 0 or result.inlet_std <= 0:
            raise ValueError("normalization standard deviations must be positive")
        return result

    def normalize_graph(self, graph: Graph) -> Graph:
        mean = torch.as_tensor(
            self.field_mean, dtype=graph.target.dtype, device=graph.target.device
        )
        std = torch.as_tensor(
            self.field_std, dtype=graph.target.dtype, device=graph.target.device
        )
        values = graph.target.view(graph.num_nodes, -1, 3)
        graph.target = ((values - mean) / std).reshape(graph.num_nodes, -1)
        graph.glob = (graph.glob - self.inlet_mean) / self.inlet_std
        return graph

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized UVP values whose last axis is three to raw units."""

        if values.size(-1) != 3:
            raise ValueError("values must have UVP on the last axis")
        mean = torch.as_tensor(
            self.field_mean, dtype=values.dtype, device=values.device
        )
        std = torch.as_tensor(self.field_std, dtype=values.dtype, device=values.device)
        return values * std + mean


class NormalizeCylinderFlow:
    """Torchvision-compatible transform using frozen Train-only statistics."""

    def __init__(self, normalization: CylinderFlowNormalization) -> None:
        self.normalization = normalization

    def __call__(self, graph: Graph) -> Graph:
        return self.normalization.normalize_graph(graph)


def build_cylinderflow_transform(
    manifest_path: str | Path,
) -> transforms.Compose:
    """Build the locked VGAE transform for raw CylinderFlow meshes."""

    normalization = CylinderFlowNormalization.from_manifest(manifest_path)
    return transforms.Compose(
        [
            ScaleEdgeAttr(0.15),
            NormalizeCylinderFlow(normalization),
            AddDirichletMask(3, [0, 1], dirichlet_boundary_id=[2, 4]),
            MeshCoarsening(
                num_scales=5,
                rel_pos_scaling=[0.15, 0.3, 0.6, 1.2, 2.4],
                scalar_rel_pos=True,
            ),
            Copy("target", "field"),
        ]
    )


class CylinderFlowH5Dataset(torch.utils.data.Dataset):
    """Random-access CylinderFlow trajectories stored by group."""

    def __init__(
        self,
        path: str | Path,
        *,
        manifest_path: str | Path,
        transform: Callable[[Graph], Graph] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError(
                self.path if not self.path.is_file() else self.manifest_path
            )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.contract = CylinderFlowDataContract.from_manifest(self.manifest)
        self.format = self.contract.manifest_format
        self.frames = self.contract.frames
        self.temporal_stride = self.contract.temporal_stride
        self.raw_frame_dt = self.contract.raw_frame_dt
        self.frame_dt = self.contract.frame_dt
        self.trajectories = tuple(self.manifest["trajectories"])
        self.transform = transform
        with h5py.File(self.path, "r") as handle:
            if str(handle.attrs.get("format", "")) != self.format:
                raise ValueError("unsupported CylinderFlow HDF5 format")
            if int(handle.attrs.get("trajectory_count", -1)) != len(self.trajectories):
                raise ValueError("manifest and HDF5 trajectory counts differ")
            if int(handle.attrs.get("frames", self.frames)) != self.frames:
                raise ValueError("manifest and HDF5 frame counts differ")

    def __len__(self) -> int:
        return len(self.trajectories)

    def split_indices(self, split: str) -> list[int]:
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown split: {split}")
        return [int(value) for value in self.manifest["splits"][split]]

    def get_sequence(
        self,
        idx: int,
        sequence_start: int = 0,
        n_in: int = 1,
        step_size: int = 1,
        cell_list: bool = False,
    ) -> Graph:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if n_in < 1 or step_size < 1 or sequence_start < 0:
            raise ValueError("sequence arguments must be positive")
        last_frame = sequence_start + (n_in - 1) * step_size
        if last_frame >= self.frames:
            raise IndexError(
                f"requested sequence exceeds the {self.frames}-frame trajectory"
            )
        record = self.trajectories[idx]
        with h5py.File(self.path, "r") as handle:
            group = handle[record["group"]]
            frame_slice = slice(
                sequence_start,
                sequence_start + n_in * step_size,
                step_size,
            )
            uvp = torch.from_numpy(np.asarray(group["uvp"][frame_slice])).float()
            pos = torch.from_numpy(np.asarray(group["mesh_pos"])).float()
            cells = torch.from_numpy(np.asarray(group["cells"])).long()
            node_type = torch.from_numpy(np.asarray(group["node_type"])).long()
            inlet_velocity = float(group.attrs["inlet_velocity"])

        if uvp.shape != (n_in, pos.size(0), 3):
            raise RuntimeError("stored CylinderFlow sequence has an invalid shape")
        graph = Graph()
        graph.pos = pos
        graph.target = uvp.permute(1, 0, 2).reshape(pos.size(0), n_in * 3)
        graph.glob = torch.full((pos.size(0), 1), inlet_velocity, dtype=torch.float32)
        graph.original_node_type = node_type.to(dtype=torch.int32)
        graph.bound = torch.full((pos.size(0),), 0, dtype=torch.uint8)
        graph.bound[node_type == 4] = 2
        graph.bound[node_type == 5] = 3
        graph.bound[node_type == 6] = 4
        supported = (
            (node_type == 0) | (node_type == 4) | (node_type == 5) | (node_type == 6)
        )
        if not bool(supported.all()):
            unknown = torch.unique(node_type[~supported]).tolist()
            raise ValueError(f"unsupported CylinderFlow node types: {unknown}")
        graph.omega = torch.zeros(pos.size(0), 3, dtype=torch.float32)
        graph.omega[(node_type == 0) | (node_type == 5), 0] = 1.0
        graph.omega[node_type == 4, 1] = 1.0
        graph.omega[node_type == 6, 2] = 1.0
        graph.edge_index = triangle_cells_to_edge_index(cells)
        graph.edge_attr = pos[graph.edge_index[1]] - pos[graph.edge_index[0]]
        if cell_list:
            graph.cell_list = [cell.clone() for cell in cells]
        return self.transform(graph) if self.transform is not None else graph

    def __getitem__(self, idx: int) -> Graph:
        return self.get_sequence(idx, sequence_start=0, n_in=1)
