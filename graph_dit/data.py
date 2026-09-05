"""Read 75 training frames and a separate 65-frame evaluation prefix."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from . import PROTOCOL_VERSION

DATA_REPOSITORY = "DingDong1921/mgn-cylinderflow-stride8-75frames"
DATA_REVISION = "8eae2c7a697e7d01f3b98f4d642ea476784df84a"
FORMAT = "dgn4cfd.mgn_cylinderflow_temporal_stride.v1"
TRAIN_FRAMES = 75
EVALUATION_FRAMES = 65
DT = 0.08


def edges_from_cells(points: np.ndarray, cells: np.ndarray):
    pairs = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)
    delta = points[edges[:, 0]] - points[edges[:, 1]]
    features = np.concatenate(
        [delta, np.linalg.norm(delta, axis=1, keepdims=True)], axis=1
    )
    return edges.astype(np.int64), features.astype(np.float32)


class Dataset:
    """Read stored frames 0..74; ``initial`` loads only the observed frame."""

    def __init__(self, dataset: str | Path, manifest: str | Path, debug: bool = False):
        self.dataset = Path(dataset).expanduser().resolve(strict=True)
        self.manifest_file = Path(manifest).expanduser().resolve(strict=True)
        self.manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        self.debug = debug
        m = self.manifest
        if m.get("format") != FORMAT or m.get("frames") != 75:
            raise ValueError("expected the released 75-frame stride-8 manifest")
        if m.get("temporal_stride") != 8 or not np.isclose(m.get("frame_dt", 0), DT):
            raise ValueError(
                "the matched physical output interval is dt=0.08, stride=8"
            )
        if not np.isclose(m.get("raw_frame_dt", 0.01), 0.01):
            raise ValueError("raw dt must be 0.01")
        if m.get("test_accessed") not in (False, None):
            raise ValueError("this workflow requires a Test-unaccessed manifest")
        if m.get("phase_offset", 0) != 0 or m.get("phase_augmentation", False):
            raise ValueError(
                "only phase-zero data without phase augmentation is supported"
            )
        if m.get("source_frame_indices", list(range(0, 600, 8))) != list(
            range(0, 600, 8)
        ):
            raise ValueError(
                "stored frames must map exactly to raw indices 0,8,...,592"
            )
        if set(m.get("splits", {})) != {"train", "validation"}:
            raise ValueError("only Train and Validation are supported; Test is sealed")
        self.splits = {
            key: tuple(int(i) for i in value) for key, value in m["splits"].items()
        }
        all_indices = self.splits["train"] + self.splits["validation"]
        if len(set(all_indices)) != len(all_indices) or not all(self.splits.values()):
            raise ValueError("empty, overlapping, or duplicate trajectory split")
        if not debug and self.splits != {
            "train": tuple(range(1000)),
            "validation": tuple(range(1000, 1100)),
        }:
            raise ValueError(
                "formal runs require the released Train 0..999 and Validation 1000..1099 split"
            )
        with h5py.File(self.dataset, "r") as handle:
            if not debug:
                required = {
                    "format": FORMAT,
                    "frames": 75,
                    "temporal_stride": 8,
                    "phase_offset": 0,
                    "phase_augmentation": False,
                    "test_accessed": False,
                    "train_count": 1000,
                    "validation_count": 100,
                    "frame_dt": DT,
                }
                if any(
                    handle.attrs.get(key) != value for key, value in required.items()
                ):
                    raise ValueError(
                        "HDF5 metadata differs from the fixed released data contract"
                    )
            for index in all_indices:
                group = handle[f"trajectory_{index:04d}"]
                if group["uvp"].shape[0] != 75 or group["uvp"].shape[-1] != 3:
                    raise ValueError("uvp must have shape [75,N,3]")
                if group["mesh_pos"].shape != (group["uvp"].shape[1], 2):
                    raise ValueError("mesh and field node counts disagree")
                if not debug and group.attrs.get("source_split") != (
                    "train" if index < 1000 else "validation"
                ):
                    raise ValueError("HDF5 source split and official manifest disagree")
        self._static: dict[int, dict] = {}
        self.graph_cache: Path | None = None

    def identity(self) -> dict:
        return {
            "protocol": PROTOCOL_VERSION,
            "dataset_repository": DATA_REPOSITORY,
            "dataset_revision": DATA_REVISION,
            "split_indices": self.splits,
            "stored_frames": 75,
            "training_frames": [0, TRAIN_FRAMES - 1],
            "evaluation_frames": [0, EVALUATION_FRAMES - 1],
            "training_raw_frame_indices": list(range(0, 600, 8)),
            "evaluation_raw_frame_indices": list(range(0, 513, 8)),
            "dt": DT,
            "debug": self.debug,
        }

    def static(self, index: int) -> dict:
        if index not in self._static:
            if index not in self.splits["train"] + self.splits["validation"]:
                raise ValueError("trajectory is outside Train/Validation")
            with h5py.File(self.dataset, "r") as handle:
                group = handle[f"trajectory_{index:04d}"]
                points = np.asarray(group["mesh_pos"], dtype=np.float32)
                cells = np.asarray(group["cells"], dtype=np.int64)
                node_type = np.asarray(group["node_type"], dtype=np.int64).reshape(-1)
            if not np.isfinite(points).all() or cells.ndim != 2 or cells.shape[1] != 3:
                raise ValueError("invalid triangle mesh")
            if cells.min() < 0 or cells.max() >= len(points):
                raise ValueError("triangle index is outside the mesh")
            if not set(np.unique(node_type)).issubset({0, 4, 5, 6}):
                raise ValueError("unexpected CylinderFlow node labels")
            if self.graph_cache is not None:
                with np.load(
                    self.graph_cache / f"trajectory_{index:04d}.npz", allow_pickle=False
                ) as cache:
                    edges, features = cache["edges"], cache["edge_features"]
            else:
                edges, features = edges_from_cells(points, cells)
            self._static[index] = dict(
                index=index,
                points=points,
                cells=cells,
                node_type=node_type,
                edges=edges,
                edge_features=features,
            )
        return self._static[index]

    def read(self, index: int, start: int = 0, stop: int = TRAIN_FRAMES) -> dict:
        if not 0 <= start < stop <= TRAIN_FRAMES:
            raise ValueError("field access must stay within stored frames 0..74")
        static = self.static(index)
        with h5py.File(self.dataset, "r") as handle:
            field = np.asarray(
                handle[f"trajectory_{index:04d}/uvp"][start:stop], dtype=np.float32
            )
        if not np.isfinite(field).all():
            raise ValueError(f"nonfinite reference field in trajectory {index}")
        return {**static, "field": field}

    def evaluation(self, index: int) -> dict:
        """Read only the observed frame and the 64 scored future frames."""
        return self.read(index, 0, EVALUATION_FRAMES)

    def initial(self, index: int) -> dict:
        sample = self.read(index, 0, 1)
        return {key: value for key, value in sample.items() if key != "field"} | {
            "initial": sample["field"][0].copy()
        }


def _moments(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    return values.sum(axis=0), (values**2).sum(axis=0), len(values)


def compute_statistics(dataset: Dataset, noise_std: float = 0.02) -> dict:
    """Compute unique-frame UVP/q stats and MGN's equal-trajectory moments."""
    sums, squares, count = np.zeros(4), np.zeros(4), 0
    mgn = {key: [] for key in ("velocity", "velocity_diff", "pressure", "edge")}
    normal_fractions = []
    for index in dataset.splits["train"]:
        sample = dataset.read(index)
        field = sample["field"].astype(np.float64)
        q = 0.5 * np.sum(field[..., :2] ** 2, axis=-1, keepdims=True)
        total, squared, n = _moments(np.concatenate([field, q], axis=-1))
        sums += total
        squares += squared
        count += n
        arrays = {
            "velocity": field[:-1, :, :2],
            "velocity_diff": np.diff(field[..., :2], axis=0),
            "pressure": field[1:, :, 2:3],
            "edge": sample["edge_features"],
        }
        for key, values in arrays.items():
            total, squared, n = _moments(values)
            mgn[key].append((total / n, squared / n))
        normal_fractions.append(float(np.mean(sample["node_type"] == 0)))
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-16))
    result = {
        "protocol": PROTOCOL_VERSION,
        "data_identity": dataset.identity(),
        "frame_range": [0, TRAIN_FRAMES - 1],
        "split": "train",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "node_frame_count": count,
        "mgn_noise_std": noise_std,
    }
    for key, records in mgn.items():
        average = np.mean([row[0] for row in records], axis=0)
        variance = np.mean([row[1] for row in records], axis=0) - average**2
        if key == "velocity_diff":
            variance += noise_std**2 * np.mean(normal_fractions)
        result[f"{key}_mean"] = average.tolist()
        result[f"{key}_std"] = np.sqrt(np.maximum(variance, 1e-16)).tolist()
    return result


def load_statistics(file_name: str | Path, dataset: Dataset) -> dict:
    stats = json.loads(Path(file_name).read_text(encoding="utf-8"))
    if stats.get("protocol") != PROTOCOL_VERSION or stats.get("frame_range") != [
        0,
        TRAIN_FRAMES - 1,
    ]:
        raise ValueError(
            "normalizer must be generated by prepare from Train frames 0..74"
        )
    # JSON canonicalization also normalizes tuple/list representations.
    if json.dumps(stats["data_identity"], sort_keys=True) != json.dumps(
        dataset.identity(), sort_keys=True
    ):
        raise ValueError("normalizer belongs to a different data contract")
    return stats
