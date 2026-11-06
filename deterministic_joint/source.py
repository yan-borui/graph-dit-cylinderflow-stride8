"""Read the official trajectory splits, retaining stored frames 0 through 64."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from airfoil_data import ensure as airfoil_ensure
from airfoil_data.portable_lock import DirectoryLock
from airfoil_data.tfrecord_io import decode_example, iter_examples

RAW = list(range(0, 513, 8))
FORMATS = {
    "airfoil": "dgn4cfd.mgn_airfoil_uvp_temporal_stride.v1",
    "cylinderflow": "dgn4cfd.mgn_cylinderflow_temporal_stride.v1",
}
DT = {"airfoil": 0.0016, "cylinderflow": 0.08}
LABELS = {"airfoil": {0, 2, 4}, "cylinderflow": {0, 4, 5, 6}}
REVISION = {
    "airfoil": "airfoil.uvp.stride8.first75.v1",
    "cylinderflow": "8eae2c7a697e7d01f3b98f4d642ea476784df84a",
}


def _expected(test: bool) -> dict[str, list[int]]:
    if test:
        return {"test": list(range(100))}
    return {"train": list(range(1000)), "validation": list(range(1000, 1100))}


def _source_name(dataset: str) -> str:
    return "cylinder_flow" if dataset == "cylinderflow" else "airfoil"


def _convert(dataset: str, raw: Path, out: Path, test: bool, stats: dict) -> None:
    meta = json.loads((raw / "meta.json").read_text(encoding="utf-8"))
    length = 600 if dataset == "cylinderflow" else 601
    if meta.get("trajectory_length") != length or not np.isclose(
        meta.get("dt", np.nan), DT[dataset] / 8, rtol=0, atol=1e-12
    ):
        raise ValueError(f"expected official {dataset} raw metadata")
    features = ("mesh_pos", "cells", "node_type", "velocity", "pressure")
    selected_meta = dict(
        meta, features={name: meta["features"][name] for name in features}
    )
    splits = _expected(test)
    attrs = dict(
        format=FORMATS[dataset],
        frames=65,
        temporal_stride=8,
        phase_offset=0,
        phase_augmentation=False,
        frame_dt=DT[dataset],
        raw_frame_dt=DT[dataset] / 8,
        trajectory_count=sum(map(len, splits.values())),
        test_accessed=test,
    )
    records = []
    with h5py.File(out / "data.h5", "x") as handle:
        handle.attrs.update(attrs)
        for split, indices in splits.items():
            source = raw / ({"validation": "valid"}.get(split, split) + ".tfrecord")
            count = 0
            for ordinal, example in enumerate(iter_examples(source)):
                if ordinal >= len(indices):
                    raise ValueError(f"unexpected record in {split}: {ordinal}")
                # Dynamic TFRecord features are byte views; only selected rows are copied.
                arrays = decode_example(example, selected_meta)
                field = np.concatenate(
                    (arrays["velocity"][RAW], arrays["pressure"][RAW]), axis=-1
                )
                pos = arrays["mesh_pos"][0]
                labels = arrays["node_type"][0].reshape(-1)
                cells = arrays["cells"][0]
                if field.shape != (65, len(pos), 3) or not np.all(np.isfinite(field)):
                    raise ValueError(f"invalid {split} trajectory {ordinal}")
                if (
                    not set(np.unique(labels)).issubset(LABELS[dataset])
                    or not (labels == 4).any()
                ):
                    raise ValueError(f"invalid {dataset} node labels")
                index = indices[ordinal]
                name = f"trajectory_{index:04d}"
                record = dict(
                    group=name, source_split=split, source_local_index=ordinal
                )
                group = handle.create_group(name)
                group.attrs.update(source_split=split, source_local_index=ordinal)
                group.create_dataset(
                    "uvp", data=field, chunks=(1, len(pos), 3), compression="lzf"
                )
                for key, value in (
                    ("mesh_pos", pos),
                    ("node_type", labels),
                    ("cells", cells),
                ):
                    group.create_dataset(key, data=value)
                records.append(record)
                count += 1
            if count != len(indices):
                raise ValueError(
                    f"{source.name} contains {count} records, expected {len(indices)}"
                )
    manifest = dict(
        **attrs,
        dataset="data.h5",
        dataset_bytes=(out / "data.h5").stat().st_size,
        source_frame_indices=RAW,
        train_only_normalization=stats,
        splits=splits,
        trajectories=records,
        source=f"https://storage.googleapis.com/dm-meshgraphnets/{_source_name(dataset)}",
    )
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _locate(
    dataset: str, directory: Path, test: bool, stats: dict
) -> tuple[Path, Path]:
    legacy = "test75" if test else f"{dataset}_stride8_75frames"
    published = directory / ("test65" if test else "trainval65")

    def ready() -> tuple[Path, Path] | None:
        for source, meta in (
            (published / "data.h5", published / "manifest.json"),
            (directory / (legacy + ".h5"), directory / (legacy + "_manifest.json")),
        ):
            if source.exists() or meta.exists():
                if not source.is_file() or not meta.is_file():
                    raise FileNotFoundError(f"incomplete source pair: {source}, {meta}")
                return source, meta
        return None

    existing = ready()
    if existing is not None:
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    with DirectoryLock(directory / f".{dataset}-source", wait=True):
        existing = ready()
        if existing is not None:
            return existing
        raw = Path(
            os.environ.get("RAW_DIR", str(directory / "raw" / _source_name(dataset)))
        )
        raw.mkdir(parents=True, exist_ok=True)
        names = (
            ("meta.json", "test.tfrecord")
            if test
            else ("meta.json", "train.tfrecord", "valid.tfrecord")
        )
        with DirectoryLock(raw / ".download.lock", wait=True):
            for name in names:
                if (raw / name).is_file() and (raw / name).stat().st_size:
                    continue
                if os.environ.get("OFFLINE") == "1":
                    raise FileNotFoundError(
                        f"OFFLINE=1; missing official source {raw / name}"
                    )
                _download(raw, name, dataset)
        staging = Path(tempfile.mkdtemp(prefix=f".{published.name}-", dir=directory))
        try:
            _convert(dataset, raw, staging, test, stats)
            staging.rename(published)
        except BaseException:
            print(f"incomplete prefix source retained at {staging}", flush=True)
            raise
    return published / "data.h5", published / "manifest.json"


def _download(raw: Path, name: str, dataset: str) -> None:
    """Fetch an official record using the established resumable downloader."""
    base_url = (
        f"https://storage.googleapis.com/dm-meshgraphnets/{_source_name(dataset)}"
    )
    airfoil_ensure.download_file(raw, name, source_url=base_url)


class PrefixData:
    """Expose only initial state to inference and the 65-frame prefix to scoring."""

    def __init__(self, dataset: str, directory: Path, split: str, stats: dict):
        if dataset not in FORMATS or split not in ("train", "validation", "test"):
            raise ValueError("unknown dataset or split")
        self.dataset, self.split = dataset, split
        test = split == "test"
        self.file, manifest_file = _locate(dataset, directory, test, stats)
        m = self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        frames = m.get("frames")
        if (
            m.get("format") != FORMATS[dataset]
            or m.get("splits") != _expected(test)
            or m.get("train_only_normalization") != stats
            or frames not in (65, 75)
            or m.get("source_frame_indices") != list(range(0, frames * 8, 8))
            or m.get("temporal_stride") != 8
            or m.get("phase_offset", 0) != 0
            or m.get("phase_augmentation", False)
            or bool(m.get("test_accessed")) != test
            or not np.isclose(
                m.get("frame_dt", np.nan), DT[dataset], rtol=0, atol=1e-10
            )
        ):
            raise ValueError(
                f"{manifest_file}: incorrect dataset, split, time axis or VGAE normalization"
            )
        self.indices = tuple(m["splits"][split])
        if len(m["trajectories"]) != sum(map(len, m["splits"].values())):
            raise ValueError(f"{manifest_file}: incomplete trajectory records")
        with h5py.File(self.file, "r") as handle:
            if (
                handle.attrs.get("format") != FORMATS[dataset]
                or handle.attrs.get("frames") != frames
                or handle.attrs.get("trajectory_count") != len(m["trajectories"])
                or not np.isclose(
                    handle.attrs.get("frame_dt", np.nan),
                    DT[dataset],
                    rtol=0,
                    atol=1e-10,
                )
                or bool(handle.attrs.get("test_accessed")) != test
            ):
                raise ValueError(f"{self.file}: HDF5 and manifest disagree")

    def _group(self, handle: h5py.File, index: int) -> h5py.Group:
        if index not in self.indices:
            raise IndexError(f"{self.split}: trajectory {index}")
        record = self.manifest["trajectories"][index]
        group = handle[record["group"]]
        local = index - self.indices[0]

        def local_id(obj: Any) -> int | None:
            return obj.get("source_local_index", obj.get("source_index"))

        if (
            record.get("source_split") != self.split
            or group.attrs.get("source_split") != self.split
            or local_id(record) != local
            or local_id(group.attrs) != local
            or group["uvp"].shape
            != (self.manifest["frames"], len(group["mesh_pos"]), 3)
        ):
            raise ValueError(
                f"source provenance or UVP shape mismatch: trajectory {index}"
            )
        return group

    def initial(self, index: int) -> dict:
        with h5py.File(self.file, "r") as handle:
            group = self._group(handle, index)
            result = dict(
                index=index,
                initial=group["uvp"][0].astype(np.float32),
                points=group["mesh_pos"][:].astype(np.float32),
                cells=group["cells"][:].astype(np.int64),
                node_type=group["node_type"][:].reshape(-1).astype(np.int64),
            )
        labels = result["node_type"]
        if not set(np.unique(labels)).issubset(LABELS[self.dataset]) or not np.any(
            labels == 4
        ):
            raise ValueError(f"invalid {self.dataset} node labels: {index}")
        if (
            not np.isfinite(result["initial"]).all()
            or not np.isfinite(result["points"]).all()
        ):
            raise ValueError(f"nonfinite initial frame or mesh: {index}")
        cells, points = result["cells"], result["points"]
        if (
            points.shape != (len(labels), 2)
            or cells.ndim != 2
            or cells.shape[1] != 3
            or cells.size == 0
            or cells.min() < 0
            or cells.max() >= len(points)
        ):
            raise ValueError(f"invalid triangular mesh at trajectory {index}")
        return result

    def truth(self, index: int) -> np.ndarray:
        with h5py.File(self.file, "r") as handle:
            fields = self._group(handle, index)["uvp"][:65].astype(np.float32)
        if not np.isfinite(fields).all():
            raise FloatingPointError(f"nonfinite prefix at trajectory {index}")
        return fields
