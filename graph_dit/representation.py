"""Prepare one frozen VGAE/Train cache shared by every independent DiT run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from dgn4cfd.cylinderflow_data import (
    CylinderFlowH5Dataset,
    CylinderFlowNormalization,
    build_cylinderflow_transform,
)
from dgn4cfd.graph_distance import unweighted_shortest_path_hops
from dgn4cfd.nn.diffusion.graph_window_codec import FrozenUVPLatentCodec
from .data import Dataset, DATA_REPOSITORY, DATA_REVISION
from .runtime import seed_everything, write_json

ARTIFACT_FORMAT = "graph_dit.ae75.train_cache.v1"
AE_ASSET = "vgae_stride8_epoch930.pt"
AE_RELEASE = (
    "https://github.com/yan-borui/graph-dit-cylinderflow-stride8/releases/download/representation-v1/"
    + AE_ASSET
)


def paths(data_dir: Path) -> tuple[Path, Path]:
    return (
        data_dir / "cylinderflow_stride8_75frames.h5",
        data_dir / "cylinderflow_stride8_75frames_manifest.json",
    )


def open_data(
    data_dir: Path, *, debug: bool = False
) -> tuple[Dataset, CylinderFlowH5Dataset]:
    dataset_file, manifest_file = paths(data_dir)
    data = Dataset(dataset_file, manifest_file, debug=debug)
    graphs = CylinderFlowH5Dataset(
        dataset_file,
        manifest_path=manifest_file,
        transform=build_cylinderflow_transform(manifest_file),
    )
    return data, graphs


def fetch_autoencoder(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(".partial")
    with (
        urllib.request.urlopen(AE_RELEASE, timeout=60) as response,
        temporary.open("xb") as stream,
    ):
        shutil.copyfileobj(response, stream)
    payload = torch.load(temporary, map_location="cpu", weights_only=True)
    if (
        payload.get("representation_id")
        != "cylinderflow_stride8_vgae_epoch930_release_v1"
    ):
        raise ValueError("unexpected released representation identity")
    os.replace(temporary, output)


def prepare(
    data_dir: Path, autoencoder: Path, output: Path, device: str, *, debug: bool = False
) -> dict:
    data, graphs = open_data(data_dir, debug=debug)
    checkpoint = torch.load(autoencoder, map_location="cpu", weights_only=True)
    if not checkpoint.get("representation_id"):
        raise ValueError(
            "use the released AE or the identified checkpoint produced by graph_dit.ae"
        )
    if checkpoint.get("normalization") != data.manifest["train_only_normalization"]:
        raise ValueError("the AE was trained with a different physical normalizer")
    if checkpoint.get("dataset_revision") != DATA_REVISION and not debug:
        raise ValueError("AE dataset revision differs from the released data")
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(0)
    started = time.perf_counter()
    codec = FrozenUVPLatentCodec(str(autoencoder), device=device)
    shutil.copyfile(autoencoder, output / "autoencoder.pt")
    shutil.copyfile(paths(data_dir)[1], output / "dataset_manifest.json")
    identity = {
        "format": ARTIFACT_FORMAT,
        "artifact_id": str(uuid.uuid4()),
        "representation_id": checkpoint["representation_id"],
        "dataset_repository": DATA_REPOSITORY,
        "dataset_revision": DATA_REVISION,
        "data_identity": data.identity(),
        "normalization": checkpoint["normalization"],
        "train_frames": [0, 74],
        "dit_train_frames": [0, 64],
        "debug": debug,
        "train_count": len(data.splits["train"]),
        "state": "preparing",
    }
    write_json(output / "artifact.json", identity)
    latent_sum = latent_square = None
    count = 0
    with h5py.File(output / "train_latents.partial.h5", "x") as cache:
        cache.attrs["format"] = ARTIFACT_FORMAT
        cache.attrs["artifact_id"] = identity["artifact_id"]
        cache.attrs["representation_id"] = identity["representation_id"]
        cache.create_dataset("sim_indices", data=data.splits["train"])
        for index in data.splits["train"]:
            graph = graphs.get_sequence(index, n_in=1)
            first, context = codec.encode_context(graph, graph.field)
            physical = data.read(index)["field"]
            normalization = CylinderFlowNormalization.from_manifest(paths(data_dir)[1])
            mean = torch.tensor(normalization.field_mean)
            std = torch.tensor(normalization.field_std)
            latent_frames = [first.cpu().numpy()]
            for frame in range(1, 75):
                scaled = (torch.from_numpy(physical[frame]) - mean) / std
                latent, *_ = codec.encode_field(graph, scaled)
                latent_frames.append(latent.cpu().numpy())
            latents = np.stack(latent_frames).astype(np.float32)
            if not np.isfinite(latents).all():
                raise FloatingPointError(
                    f"nonfinite latent cache at Train trajectory {index}"
                )
            group = cache.create_group(f"sim_{index:05d}")
            for name, values in (
                ("latents", latents),
                ("node_context", context.node_context.cpu().numpy()),
                ("positions", context.positions.cpu().numpy()),
                (
                    "graph_hops",
                    unweighted_shortest_path_hops(
                        context.edge_index.cpu(), num_nodes=context.num_latent_nodes
                    ),
                ),
            ):
                group.create_dataset(name, data=values)
            flattened = latents.astype(np.float64).reshape(-1, codec.latent_features)
            total, square = flattened.sum(0), (flattened**2).sum(0)
            latent_sum = total if latent_sum is None else latent_sum + total
            latent_square = square if latent_square is None else latent_square + square
            count += len(flattened)
            print(
                json.dumps(
                    {
                        "event": "cache",
                        "trajectory": index,
                        "latent_nodes": latents.shape[1],
                    }
                ),
                flush=True,
            )
        mean = latent_sum / count
        std = np.sqrt(np.maximum(latent_square / count - mean**2, 1e-12))
        cache.create_dataset("latent_mean", data=mean.astype(np.float32))
        cache.create_dataset("latent_std", data=std.astype(np.float32))
    os.replace(output / "train_latents.partial.h5", output / "train_latents.h5")
    identity.update(
        state="complete",
        elapsed_seconds=time.perf_counter() - started,
        latent_mean=mean.tolist(),
        latent_std=std.tolist(),
        latent_node_frames=count,
        autoencoder_parameters=sum(p.numel() for p in codec.parameters()),
    )
    write_json(output / "artifact.json", identity)
    return identity


def load_artifacts(directory: Path, data: Dataset | None = None) -> dict:
    identity = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if identity.get("format") != ARTIFACT_FORMAT or identity.get("state") != "complete":
        raise ValueError("representation preparation is incomplete or unsupported")
    ae = torch.load(directory / "autoencoder.pt", map_location="cpu", weights_only=True)
    if ae.get("representation_id") != identity["representation_id"]:
        raise ValueError("representation checkpoint ID does not match artifacts")
    if ae.get("normalization") != identity["normalization"]:
        raise ValueError("representation normalization mismatch")
    if data is not None:
        if json.dumps(identity["data_identity"], sort_keys=True) != json.dumps(
            data.identity(), sort_keys=True
        ):
            raise ValueError("dataset contract and representation dependencies differ")
        if identity["normalization"] != data.manifest["train_only_normalization"]:
            raise ValueError(
                "dataset normalization differs from the frozen representation"
            )
    with h5py.File(directory / "train_latents.h5", "r") as cache:
        if cache.attrs.get("artifact_id") != identity["artifact_id"]:
            raise ValueError("latent cache ID does not match artifacts")
        expected = identity["data_identity"]["split_indices"]["train"]
        if list(cache["sim_indices"]) != list(expected):
            raise ValueError("latent cache is not exactly the declared Train split")
        if cache.attrs.get("representation_id") != identity["representation_id"]:
            raise ValueError("cache and autoencoder dependencies differ")
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-ae")
    fetch.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--autoencoder", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "fetch-ae":
        fetch_autoencoder(args.output)
    else:
        prepare(args.data_dir, args.autoencoder, args.output_dir, args.device)


if __name__ == "__main__":
    main()
