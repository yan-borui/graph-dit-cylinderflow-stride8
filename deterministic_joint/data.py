"""Frozen paper representation, Train-only latent windows and physical prediction."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import h5py
import numpy as np
import torch

from airfoil_data.portable_lock import DirectoryLock
from dgn4cfd.cylinderflow_data import (
    CylinderFlowNormalization,
    triangle_cells_to_edge_index,
)
from dgn4cfd.graph import Graph
from dgn4cfd.nn.diffusion.graph_window_codec import (
    FrozenUVPLatentCodec,
    LatentGraphContext,
)
from dgn4cfd.transforms import AddDirichletMask, MeshCoarsening, ScaleEdgeAttr

from .model import build_hierarchy
from .source import DT, RAW, REVISION, PrefixData

FORMAT = "deterministic_joint.train65.v1"
CYLINDER_REP = "d0fee50b-8a47-4652-b229-0e82e06ba2d7"
CYLINDER_ART = "f6f3a2f8-d55e-431e-a5e4-65b60279e1d3"


def _phys_stats(values: dict) -> CylinderFlowNormalization:
    if values.get("field_names") != ["u", "v", "p"]:
        raise ValueError("paper export must contain UVP normalization in this order")
    norm = CylinderFlowNormalization(
        tuple(values["field_mean"]),
        tuple(values["field_std"]),
        float(values["inlet_mean"]),
        float(values["inlet_std"]),
    )
    if (
        len(norm.field_mean) != 3
        or len(norm.field_std) != 3
        or min((*norm.field_std, norm.inlet_std)) <= 0
        or not np.isfinite(
            (*norm.field_mean, *norm.field_std, norm.inlet_mean, norm.inlet_std)
        ).all()
    ):
        raise ValueError("invalid paper physical normalization")
    return norm


def _latent_stats(values: dict) -> tuple[np.ndarray, np.ndarray]:
    mean, std = (
        np.asarray(values.get(key), dtype=np.float32)
        for key in ("latent_mean", "latent_std")
    )
    if (
        mean.shape != (4,)
        or std.shape != (4,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("missing or invalid paper artifact latent_mean/latent_std [4]")
    return mean, std


def _verify_id(dataset: str, rep: str, parent_id: str) -> None:
    if dataset == "cylinderflow":
        expected = (CYLINDER_REP, CYLINDER_ART)
    else:
        expected = (
            os.environ.get("PAPER_REPRESENTATION_ID"),
            os.environ.get("PAPER_ARTIFACT_ID"),
        )
        if not all(expected):
            raise ValueError(
                "set PAPER_REPRESENTATION_ID and PAPER_ARTIFACT_ID from the selected Airfoil run"
            )
    if (rep, parent_id) != expected:
        raise ValueError(
            "VGAE/export artifact identity differs from the selected paper representation"
        )


def _paper(dataset: str, directory: Path) -> tuple[dict, Path]:
    record_file = directory / "artifact.json"
    if not record_file.is_file():
        raise FileNotFoundError(
            f"paper representation record with latent normalization required: {record_file}"
        )
    p = json.loads(record_file.read_text(encoding="utf-8"))
    ckpt_file = next(
        (
            directory / name
            for name in ("autoencoder.pt", "dit_autoencoder.pt")
            if (directory / name).is_file()
        ),
        None,
    )
    if ckpt_file is None:
        raise FileNotFoundError(
            f"missing paper autoencoder.pt or dit_autoencoder.pt in {directory}"
        )
    _verify_id(dataset, p.get("representation_id"), p.get("artifact_id"))
    if p.get("state") != "complete" or p.get("debug"):
        raise ValueError("paper artifacts must come from a complete formal run")
    _validate_export(dataset, p, ckpt_file)
    _phys_stats(p["normalization"])
    _latent_stats(p)
    return p, ckpt_file


def _validate_export(dataset: str, record: dict, file: Path) -> None:
    ae = torch.load(file, map_location="meta", weights_only=True)
    fmt = (
        "vgae_cf.airfoil_uvp_dit_autoencoder.v1"
        if dataset == "airfoil"
        else "vgae_cf.uvp_dit_autoencoder.v1"
    )
    if (
        ae.get("format") != fmt
        or ae.get("dataset_revision") != REVISION[dataset]
        or ae.get("representation_id") != record.get("representation_id")
        or ae.get("normalization") != record.get("normalization")
        or ae["arch"].get("latent_node_features") != 4
        or ae["arch"].get("fnns_width") != 512
        or ae["arch"].get("in_node_features") != 3
        or ae["arch"].get("depths") != [4, 4, 2]
    ):
        raise ValueError(
            "paper record and frozen VGAE export disagree on dataset, identity, axes or normalization"
        )


def _static_graph(sample: dict, dataset: str) -> Graph:
    pos = torch.from_numpy(sample["points"])
    cells = torch.from_numpy(sample["cells"])
    labels = torch.from_numpy(sample["node_type"])
    entering = labels == 4
    surface = 2 if dataset == "airfoil" else 6
    graph = Graph()
    graph.pos = pos
    graph.original_node_type = labels.to(torch.int32)
    graph.bound = torch.zeros(len(pos), dtype=torch.uint8)
    graph.bound[labels == 4] = 2
    graph.bound[labels == 5] = 3
    graph.bound[labels == surface] = 4
    graph.omega = torch.stack(
        ((labels == 0) | (labels == 5), entering, labels == surface), dim=1
    ).float()
    graph.edge_index = triangle_cells_to_edge_index(cells)
    graph.edge_attr = pos[graph.edge_index[1]] - pos[graph.edge_index[0]]
    graph = ScaleEdgeAttr(0.15)(graph)
    graph = AddDirichletMask(
        3, [0, 1], dirichlet_boundary_id=[2, 4] if dataset == "cylinderflow" else []
    )(graph)
    return MeshCoarsening(
        num_scales=3, rel_pos_scaling=[0.15, 0.3, 0.6], scalar_rel_pos=True
    )(graph)


def _observed_graph(
    static: Graph, sample: dict, dataset: str, norm: CylinderFlowNormalization
) -> Graph:
    graph = static.clone()
    uvp = torch.from_numpy(sample["initial"])
    inlet_nodes = graph.original_node_type == 4
    inlet = (
        uvp[inlet_nodes, :2].mean(0).norm()
        if dataset == "airfoil"
        else uvp[inlet_nodes, 0].double().mean()
    )
    graph.target = uvp
    graph.glob = torch.full((len(uvp), 1), float(inlet), dtype=torch.float32)
    graph = norm.normalize_graph(graph)
    graph.field = graph.target.clone()
    return graph


def _source_id(source: PrefixData) -> dict:
    m = source.manifest
    return {
        "format": m["format"],
        "frames_in_source": m["frames"],
        "raw_indices": RAW,
        "source_bytes": source.file.stat().st_size,
        "splits": m["splits"],
        "source_records": [[r["group"], r["source_split"]] for r in m["trajectories"]],
        "dt": DT[source.dataset],
        "dataset_revision": REVISION[source.dataset],
    }


def _cache_id(directory: Path) -> dict:
    identity = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if identity.get("format") != FORMAT or identity.get("state") != "complete":
        raise ValueError(f"incomplete or incompatible deterministic cache: {directory}")
    _verify_id(
        identity["dataset"],
        identity["representation_id"],
        identity["parent_artifact_id"],
    )
    _latent_stats(identity)
    _validate_export(identity["dataset"], identity, directory / "autoencoder.pt")
    return identity


@torch.no_grad()
def prepare(
    dataset: str,
    data_dir: Path,
    paper_artifacts: Path,
    output: Path,
    device: str | torch.device,
) -> dict:
    """Prepare once under an exclusive lock; reused artifacts must match their parents."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paper, ae_file = _paper(dataset, paper_artifacts)
    data = PrefixData(dataset, data_dir, "train", paper["normalization"])
    source_id = _source_id(data)
    expected = dict(
        dataset=dataset,
        parent_artifact_id=paper["artifact_id"],
        representation_id=paper["representation_id"],
        source=source_id,
        normalization=paper["normalization"],
        latent_mean=paper["latent_mean"],
        latent_std=paper["latent_std"],
        train_stored_frames=[0, 64],
        future_stored_frames=[1, 64],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with DirectoryLock(output.with_name(output.name + ".prepare.lock"), wait=True):
        if output.exists():
            existing = TrainCache(output)
            if any(existing.identity.get(k) != v for k, v in expected.items()):
                raise ValueError(
                    f"existing cache differs from requested paper/data; select a new CACHE_DIR: {output}"
                )
            return existing.identity
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            artifact_id = str(uuid.uuid4())
            shutil.copyfile(ae_file, staging / "autoencoder.pt")
            codec = FrozenUVPLatentCodec(str(staging / "autoencoder.pt"), device=device)
            normalization = _phys_stats(paper["normalization"])
            mean, std = (
                torch.tensor(v, device=device)
                for v in (normalization.field_mean, normalization.field_std)
            )
            with h5py.File(staging / "train.h5", "x") as cache:
                cache.attrs.update(
                    format=FORMAT,
                    artifact_id=artifact_id,
                    representation_id=paper["representation_id"],
                )
                cache.create_dataset("train_indices", data=data.indices)
                for index in data.indices:
                    sample = data.initial(index)
                    graph = _observed_graph(
                        _static_graph(sample, dataset), sample, dataset, normalization
                    )
                    first, context = codec.encode_context(graph, graph.field)
                    fields = data.truth(index)
                    latents = [first.cpu().numpy()]
                    for uvp in fields[1:]:
                        field = (torch.from_numpy(uvp).to(device) - mean) / std
                        latent, *_ = codec.encode_field(graph, field)
                        latents.append(latent.cpu().numpy())
                    result = np.stack(latents)
                    if not np.isfinite(result).all() or result.shape != (
                        65,
                        context.num_latent_nodes,
                        4,
                    ):
                        raise ValueError(f"invalid posterior means at Train {index}")
                    group = cache.create_group(str(index))
                    for name, array in (
                        ("latents", result),
                        ("node_context", context.node_context.cpu().numpy()),
                        ("positions", context.positions.cpu().numpy()),
                        ("edge_index", context.edge_index.cpu().numpy()),
                    ):
                        group.create_dataset(name, data=array)
                    if (index + 1) % 25 == 0:
                        print(
                            f"cached {index + 1}/{len(data.indices)} Train trajectories",
                            flush=True,
                        )
            identity = dict(
                **expected,
                format=FORMAT,
                artifact_id=artifact_id,
                state="complete",
                inherited_statistics="paper frozen VGAE and bound Train latent normalizer; may include Train frames 65–74",
            )
            (staging / "artifact.json").write_text(
                json.dumps(identity, indent=2) + "\n", encoding="utf-8"
            )
            staging.rename(output)
        except BaseException:
            print(f"incomplete cache retained at {staging}", flush=True)
            raise
    return identity


class TrainCache:
    """One normalized initial latent and 64 supervised absolute latent fields per graph."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.identity = _cache_id(self.directory)
        self.mean, self.std = [
            torch.from_numpy(a) for a in _latent_stats(self.identity)
        ]
        self.graphs: dict[int, Graph] = {}
        with h5py.File(self.directory / "train.h5", "r") as cache:
            self.indices = tuple(map(int, cache["train_indices"][:]))
            if (
                self.indices != tuple(range(1000))
                or cache.attrs["format"] != FORMAT
                or cache.attrs["artifact_id"] != self.identity["artifact_id"]
                or cache.attrs["representation_id"]
                != self.identity["representation_id"]
            ):
                raise ValueError("Train split/cache identity mismatch")

    def window(self, index: int, device: str | torch.device) -> tuple:
        if index not in self.indices:
            raise IndexError(index)
        with h5py.File(self.directory / "train.h5", "r") as handle:
            group = handle[str(index)]
            values = torch.from_numpy(group["latents"][:])
            if values.ndim != 3 or values.shape[0] != 65 or values.shape[2] != 4:
                raise ValueError(f"invalid cached Train latent shape: {index}")
            nodes = torch.from_numpy(group["node_context"][:])
            if index not in self.graphs:
                self.graphs[index] = build_hierarchy(
                    torch.from_numpy(group["positions"][:]),
                    torch.from_numpy(group["edge_index"][:]),
                )
            graph = self.graphs[index].clone().to(device)
        latents = ((values - self.mean) / self.std).to(device)
        return latents[0], nodes.to(device), graph, latents[1:]


class EvaluationData:
    """Predict from a CPU initial field and static mesh; scoring reads truth separately."""

    def __init__(
        self,
        dataset: str,
        split: str,
        data_dir: Path,
        cache_dir: Path,
        device: str | torch.device,
    ):
        if split not in ("validation", "test"):
            raise ValueError("evaluation supports validation and test")
        self.identity = _cache_id(cache_dir)
        if dataset != self.identity["dataset"]:
            raise ValueError("wrong dataset for deterministic cache")
        self.dataset, self.device = dataset, device
        self.data = PrefixData(dataset, data_dir, split, self.identity["normalization"])
        self.source_identity = _source_id(self.data)
        if split == "validation" and self.source_identity != self.identity["source"]:
            raise ValueError(
                "Validation source differs from the Train/Validation cache source"
            )
        self.indices = self.data.indices
        self.norm = _phys_stats(self.identity["normalization"])
        self.mean, self.std = [
            torch.tensor(a, device=device) for a in _latent_stats(self.identity)
        ]
        self.codec = FrozenUVPLatentCodec(
            str(cache_dir / "autoencoder.pt"), device=device
        )
        self.static: dict[int, tuple[Graph, Graph]] = {}

    def initial(self, index: int) -> dict:
        sample = self.data.initial(index)
        if index not in self.static:
            graph = _static_graph(sample, self.dataset)
            coarse = build_hierarchy(graph.pos_3, graph.edge_index_3)
            self.static[index] = (graph, coarse)
        return dict(
            sample,
            trajectory_index=index,
            static_graph=self.static[index][0],
            hierarchy=self.static[index][1],
        )

    @torch.no_grad()
    def encode(self, sample: dict) -> tuple:
        graph = _observed_graph(sample["static_graph"], sample, self.dataset, self.norm)
        first, context = self.codec.encode_context(graph, graph.field)
        return (
            (first - self.mean) / self.std,
            context.node_context,
            sample["hierarchy"].clone().to(self.device),
            context,
        )

    @torch.no_grad()
    def decode(
        self, future: torch.Tensor, sample: dict, context: LatentGraphContext
    ) -> np.ndarray:
        if future.shape != (64, context.num_latent_nodes, 4):
            raise ValueError("deterministic output must be [64, latent nodes, 4]")
        fields = [sample["initial"]]
        for latent in future * self.std + self.mean:
            field = self.codec.autoencoder.decode(
                deepcopy(context.physical_graph),
                latent,
                [v.clone() for v in context.c_latent_list],
                [e.clone() for e in context.e_latent_list],
                [v.clone() for v in context.edge_index_list],
                [v.clone() for v in context.batch_list],
                None,
                None,
            )
            fields.append(self.norm.denormalize(field).cpu().numpy())
        prediction = np.stack(fields)
        if self.dataset == "cylinderflow":
            fixed = np.isin(sample["node_type"], (4, 6))
            prediction[1:, fixed, :2] = sample["initial"][fixed, :2]
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"nonfinite prediction: {sample['index']}")
        return prediction

    def truth(self, index: int) -> np.ndarray:
        return self.data.truth(index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Train65 latent cache."
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DT))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--paper-artifacts", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    a = parser.parse_args()
    identity = prepare(a.dataset, a.data_dir, a.paper_artifacts, a.cache_dir, a.device)
    print(
        json.dumps(
            {k: identity[k] for k in ("dataset", "artifact_id", "representation_id")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
